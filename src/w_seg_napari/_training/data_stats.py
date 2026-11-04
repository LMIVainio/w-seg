"""
Per-slide image statistics for a folder of training tifs (numpy / scipy only).

Why: when slides come from different scanners or stainings, the model sees
that variation as noise. These numbers show WHICH slides differ and in WHAT
(brightness, contrast, white balance, stain strength, sharpness), so you can
choose the augmentation strength, spot outliers, or decide to re-acquire or
exclude a slide.

Every number is measured on a strided sample of the RGB planes (fast for huge
slides); sharpness and wall contrast use crops that contain walls.
"""

import glob
import os

import numpy as np
from scipy import ndimage as ndi

from w_seg_napari._training.data import detect_rgb_scale, open_stack

# metric -> short explanation (also the CSV / figure order)
METRICS = {
    'median_gray': 'median brightness (0-255)',
    'contrast': 'p99 - p1 of brightness (0-255)',
    'red_green': 'median R / G (white balance)',
    'blue_green': 'median B / G (white balance)',
    'saturation': 'mean HSV saturation (stain strength, 0-1)',
    'wall_contrast': 'cell interior minus wall brightness (0-255)',
    'sharpness': 'wall edge gradient / wall contrast (higher = sharper)',
    'wall_fraction': 'share of pixels labelled wall',
    'median_cell_area': 'median instance area (pixels)',
}


def _gray(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def sampled_colour_stats(rgb):
    """Colour numbers from an [h, w, 3] float array in 0..255."""
    flat = rgb.reshape(-1, 3)
    gray = _gray(flat)
    med = np.median(flat, axis=0)
    mx, mn = flat.max(axis=1), flat.min(axis=1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    p1, p99 = np.percentile(gray, [1, 99])
    return {
        'median_gray': float(np.median(gray)),
        'contrast': float(p99 - p1),
        'red_green': float(med[0] / max(med[1], 1e-6)),
        'blue_green': float(med[2] / max(med[1], 1e-6)),
        'saturation': float(sat.mean()),
    }


def wall_crop_stats(gray, wall):
    """
    (wall_contrast, edge_gradient) of one crop: gray [h, w] 0..255, wall bool.
    wall_contrast = mean brightness of non-wall pixels more than 3 px from a
    wall, minus mean brightness on wall pixels (absolute value).
    edge_gradient = mean Gaussian-gradient magnitude on wall pixels.
    """
    if wall.sum() < 50 or (~wall).sum() < 50:
        return None
    far = ndi.distance_transform_edt(~wall) > 3
    if far.sum() < 50:
        return None
    contrast = abs(float(gray[far].mean() - gray[wall].mean()))
    grad = ndi.gaussian_gradient_magnitude(gray, sigma=1.0)
    return contrast, float(grad[wall].mean())


def slide_stats(path, n_crops=12, crop=512, step=8, seed=0):
    """Statistics of one 5-plane slide tif (RGB, wall, instances)."""
    stack = open_stack(path)
    _, H, W = stack.shape
    scale = 255.0 / detect_rgb_scale(stack[0:3])  # to 0..255
    rgb = np.moveaxis(
        np.asarray(stack[0:3, ::step, ::step], dtype=np.float32), 0, -1
    )
    out = {'file': os.path.basename(path), 'height': H, 'width': W}
    out.update(sampled_colour_stats(rgb * scale))

    sem = np.asarray(stack[3, ::step, ::step]) > 0
    out['wall_fraction'] = float(sem.mean())
    inst = np.asarray(stack[4])
    ids, counts = np.unique(inst[inst > 0], return_counts=True)
    out['median_cell_area'] = float(np.median(counts)) if len(ids) else np.nan

    # crops that contain walls
    rng = np.random.default_rng(seed)
    ch, cw = min(crop, H), min(crop, W)
    contrasts, grads, tries = [], [], 0
    while len(contrasts) < n_crops and tries < n_crops * 8:
        tries += 1
        y = int(rng.integers(0, H - ch + 1))
        x = int(rng.integers(0, W - cw + 1))
        wall = np.asarray(stack[3, y : y + ch, x : x + cw]) > 0
        if wall.mean() < 0.03:
            continue
        g = _gray(
            np.moveaxis(
                np.asarray(stack[0:3, y : y + ch, x : x + cw], np.float32)
                * scale,
                0,
                -1,
            )
        )
        res = wall_crop_stats(g, wall)
        if res:
            contrasts.append(res[0])
            grads.append(res[1])
    out['wall_contrast'] = float(np.mean(contrasts)) if contrasts else np.nan
    out['sharpness'] = (
        float(np.mean(grads) / max(np.mean(contrasts), 1e-6))
        if contrasts
        else np.nan
    )
    return out


def collect_stats(folder, **kw):
    """List of slide_stats dicts for every tif in the folder."""
    paths = sorted(
        glob.glob(os.path.join(folder, '*.tif'))
        + glob.glob(os.path.join(folder, '*.tiff'))
    )
    return [slide_stats(p, **kw) for p in paths]


def robust_z(values, min_rel=0.05):
    """
    (v - median) / scale with scale = max(1.4826 * MAD, min_rel * |median|).
    The floor keeps near-identical slides from being flagged for differences
    of a few percent; 0 where the scale is zero.
    """
    v = np.asarray(values, dtype=np.float64)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med)) * 1.4826
    scale = max(mad, min_rel * abs(med))
    if not np.isfinite(scale) or scale < 1e-12:
        return np.zeros_like(v)
    return (v - med) / scale


def flag_outliers(rows, z_limit=2.5):
    """{file: [(metric, z), ...]} for slides with |robust z| > z_limit."""
    flagged = {}
    for m in METRICS:
        z = robust_z([r[m] for r in rows])
        for r, zi in zip(rows, z):
            if np.isfinite(zi) and abs(zi) > z_limit:
                flagged.setdefault(r['file'], []).append((m, float(zi)))
    return flagged


def spread_summary(rows):
    """{metric: (median, relative spread)} where spread = IQR / median."""
    out = {}
    for m in METRICS:
        v = np.array([r[m] for r in rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        out[m] = (float(med), float((q3 - q1) / max(abs(med), 1e-9)))
    return out


def write_csv(rows, path):
    import csv

    cols = ['file', 'height', 'width', *METRICS]
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(
                [
                    f'{r[c]:.4g}' if isinstance(r[c], float) else r[c]
                    for c in cols
                ]
            )


def save_figure(rows, path, dpi=120):
    """One small panel per metric: slides sorted by value, median line."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = [r['file'] for r in rows]
    n = len(METRICS)
    cols = 3
    fig, axes = plt.subplots(
        -(-n // cols), cols, figsize=(13, 3.2 * -(-n // cols))
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, (m, desc) in zip(axes, METRICS.items()):
        v = np.array([r[m] for r in rows], dtype=np.float64)
        order = np.argsort(v)
        flagged = np.abs(robust_z(v)) > 2.5
        ax.bar(
            range(len(v)),
            v[order],
            color=['#c0392b' if f else '#4a7ab5' for f in flagged[order]],
        )
        ax.axhline(np.nanmedian(v), color='k', lw=0.8, ls='--')
        ax.set_title(desc, fontsize=9)
        if len(v) <= 25:
            ax.set_xticks(range(len(v)))
            ax.set_xticklabels(
                [names[i] for i in order], rotation=75, fontsize=6, ha='right'
            )
        else:
            ax.set_xticks([])
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle('Slide statistics (red = robust z > 2.5 from the median)')
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
