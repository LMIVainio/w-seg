"""
Primary phloem pole distance from the plugin's per-cell tables (no napari).

Input: the tables the plugin writes (batch export: one <image>_morphology.csv
per image, or an xlsx/csv with a `source_file` column), with at least the cell
centroids (`centroid-0` = y, `centroid-1` = x) and a class column
(`predicted_class` from the classifier, or a `class_<layer>` column from a
hand-marked layer).

Per image:
  1. take the cells of the phloem class(es)
  2. k-means with 2 groups on their centroids -> the two phloem poles
  3. distance between the two pole centres
Phloem poles can be looked for in steps ("levels"): first only the most
reliably detected class (e.g. phloem pole parenchyma), and only if those do not
form two clear clusters, that class plus the next one, and so on (see
`pooling_levels` and `clear_poles`).
With xylem cells as well (two xylem poles by the same method) the measuring
line is made perpendicular to the xylem axis instead of simply joining the two
phloem centres. In a diarch root the two axes are at right angles, so this
removes the sideways offset caused by uneven clusters.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

TABLE_SUFFIXES = ('.csv', '.xlsx', '.xls')


# ---------------------------------------------------------------- reading
def _read_one(path, sheet=None):
    path = str(path)
    if path.lower().endswith('.csv'):
        # the plugin's csv export may start with '#' metadata lines
        return pd.read_csv(path, comment='#')
    if sheet is None:
        xl = pd.ExcelFile(path)
        sheet = 'Cell Analytics' if 'Cell Analytics' in xl.sheet_names else 0
    return pd.read_excel(path, sheet_name=sheet)


def image_key(name):
    """Comparable image name: no folder, extension or '_morphology' suffix."""
    s = os.path.basename(str(name)).strip()
    s = re.sub(r'\.(tif|tiff|csv|xlsx|xls|png|jpg)$', '', s, flags=re.I)
    s = re.sub(r'_morphology$', '', s, flags=re.I)
    return s.lower().strip()


def load_tables(path, sheet=None):
    """
    {image name: DataFrame} from a file or a folder.
    A table with a `source_file` column is split by it; otherwise each file is
    one image, named after the file.
    """
    path = Path(path)
    files = (
        sorted(p for p in path.iterdir() if p.suffix.lower() in TABLE_SUFFIXES)
        if path.is_dir()
        else [path]
    )
    if not files:
        raise ValueError(f'No csv / xlsx tables found in {path}')
    out = {}
    for f in files:
        df = _read_one(f, sheet)
        if 'source_file' in df.columns:
            for name, part in df.groupby('source_file', sort=False):
                out[str(name)] = part.reset_index(drop=True)
        else:
            out[f.stem] = df.reset_index(drop=True)
    return out


# ------------------------------------------------------------ clustering
def _class_key(v):
    """'2', 2 and 2.0 all become '2'; text is compared case-insensitively."""
    try:
        f = float(v)
        if f == int(f):
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(v).strip().lower()


def select_cells(df, class_col, classes):
    """Rows of df whose class is one of `classes` (list of ids or names)."""
    if class_col not in df.columns:
        raise KeyError(
            f'Column {class_col!r} not found (have: {list(df.columns)})'
        )
    want = {_class_key(c) for c in classes}
    return df[df[class_col].map(_class_key).isin(want)]


def two_poles(xy, trim=0.0, seed=0):
    """
    Split points [n, 2] (x, y) into two groups with k-means.

    trim > 0 makes it robust against misclassified cells:
      * a group that contains a single cell is an outlier (k-means gives a far
        stray cell a group of its own), so that cell is dropped and the split
        repeated while more than 4 cells remain;
      * cells farther than trim x (median distance to their own centre) from
        their centre are dropped and the split is repeated.
    Returns dict: centres [2, 2] (sorted left to right by x, so pole 1 is the
    left one), sizes [2], spread (mean distance of points to their centre),
    kept (bool mask over the input), labels (-1 for dropped points).
    """
    from sklearn.cluster import KMeans

    xy = np.asarray(xy, dtype=float)
    if len(xy) < 2:
        raise ValueError('need at least 2 cells for 2 poles')

    def fit(mask):
        km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(xy[mask])
        lab = np.full(len(xy), -1)
        lab[mask] = km.labels_
        return km.cluster_centers_, lab

    keep = np.ones(len(xy), dtype=bool)
    centres, lab = fit(keep)
    if trim > 0:
        for _ in range(6):
            if keep.sum() <= 4:
                break
            sizes = np.bincount(lab[keep], minlength=2)
            if sizes.min() == 1:  # a lone cell is its own "pole": drop it
                keep &= ~(lab == int(np.argmin(sizes)))
            else:
                d = np.linalg.norm(xy - centres[np.maximum(lab, 0)], axis=1)
                new_keep = keep & (d <= max(trim * np.median(d[keep]), 1e-9))
                if new_keep.sum() < 4 or new_keep.sum() == keep.sum():
                    break
                keep = new_keep
            centres, lab = fit(keep)
    order = np.argsort(centres[:, 0])  # left pole first
    centres = centres[order]
    remap = {int(old): new for new, old in enumerate(order)}
    lab = np.array([remap.get(int(v), -1) for v in lab])
    sizes = np.array([(lab == i).sum() for i in (0, 1)])
    dist = np.linalg.norm(xy - centres[np.maximum(lab, 0)], axis=1)
    spread = float(dist[lab >= 0].mean()) if (lab >= 0).any() else np.nan
    return {
        'centres': centres,
        'sizes': sizes,
        'spread': spread,
        'kept': lab >= 0,
        'labels': lab,
    }


# ---------------------------------------------------------- measurement
def pooling_levels(order):
    """Cumulative class lists from classes in order of reliability:
    ['a', 'b', 'c'] -> [['a'], ['a', 'b'], ['a', 'b', 'c']]."""
    order = list(order)
    return [order[: i + 1] for i in range(len(order))]


def clear_poles(xy, trim=3.0, min_cells=4, min_pole_cells=2, min_sep=4.0):
    """
    Do the points form two clear clusters? Returns (ok, separation, poles)
    where separation = distance between the two centres / mean distance of the
    cells to their own centre (about 2-2.5 when one round blob is cut in two,
    well above 4 for two real groups) and poles is two_poles()'s result (None
    if there were too few cells). Clear = enough cells, every pole has at
    least `min_pole_cells` of them after trimming, and separation >= min_sep.
    """
    xy = np.asarray(xy, dtype=float)
    if len(xy) < min_cells:
        return False, np.nan, None
    pp = two_poles(xy, trim=trim)
    c = pp['centres']
    dist = float(np.linalg.norm(c[1] - c[0]))
    sep = dist / pp['spread'] if pp['spread'] > 0 else np.inf
    ok = bool(pp['sizes'].min() >= min_pole_cells and sep >= min_sep)
    return ok, float(sep), pp


def _level_name(classes):
    return '+'.join(str(c) for c in classes)


def _describe_levels(tried):
    """'a: 5 cells, sep 6.1; a+b: 11 cells, sep 2.3' for the log."""
    parts = []
    for lv, cells, _, sep in tried:
        text = f'{_level_name(lv)}: {len(cells)} cells'
        if np.isfinite(sep):
            text += f', sep {sep:.1f}'
        parts.append(text)
    return '; '.join(parts)


def measure_image(
    df,
    phloem_classes=None,
    class_col='predicted_class',
    xylem_classes=None,
    y_col='centroid-0',
    x_col='centroid-1',
    trim=3.0,
    min_cells=4,
    phloem_levels=None,
    min_sep=4.0,
    min_pole_cells=2,
):
    """
    One result dict for one image's table (coordinates in pixels, as (x, y)).

    The phloem cells come from `phloem_classes`, or, to look in steps, from
    `phloem_levels` (a list of class lists, most reliable first, e.g. from
    pooling_levels). The first level whose cells form two clear clusters
    (clear_poles) is used. If none does, the best separated level is measured
    but marked: phloem_level = 'unclear' and the flag 'phloem poles unclear'.

    distance_px         between the two phloem pole centres
    distance_xylem_px   along the line perpendicular to the xylem axis (only
                        with xylem_classes; NaN otherwise)
    angle_deg           angle between phloem and xylem axes (about 90 expected)
    flags               short notes when the result deserves a look
    plus the pole / cell counts and centres for plotting.
    """
    res = {
        'phloem_level': '',
        'phloem_levels_tried': '',
        'n_phloem_cells': 0,
        'distance_px': np.nan,
        'distance_xylem_px': np.nan,
        'angle_deg': np.nan,
        'flags': '',
    }
    flags = []
    for col in (y_col, x_col):
        if col not in df.columns:
            raise KeyError(
                f'Column {col!r} not found (have: {list(df.columns)})'
            )

    stepwise = phloem_levels is not None and len(phloem_levels) > 0
    if stepwise:
        levels = [list(lv) for lv in phloem_levels]
    elif phloem_classes is not None:
        levels = [list(phloem_classes)]
    else:
        raise ValueError('give phloem_classes or phloem_levels')
    tried, chosen = [], None
    for lv in levels:
        cells = select_cells(df, class_col, lv)
        ok, sep, _ = clear_poles(
            cells[[x_col, y_col]].to_numpy(float),
            trim=trim,
            min_cells=min_cells,
            min_pole_cells=min_pole_cells,
            min_sep=min_sep,
        )
        tried.append((lv, cells, ok, sep))
        if ok:
            chosen = tried[-1]
            break
    unclear = chosen is None
    if unclear:
        usable = [t for t in tried if len(t[1]) >= min_cells]
        if not usable:
            ph = tried[-1][1]
            res['n_phloem_cells'] = len(ph)
            res['phloem_levels_tried'] = _describe_levels(tried)
            res['flags'] = f'only {len(ph)} phloem cells'
            return res
        chosen = max(usable, key=lambda t: np.nan_to_num(t[3], nan=-1))
    ph = chosen[1]
    res['phloem_level'] = (
        'unclear' if unclear and stepwise else _level_name(chosen[0])
    )
    if stepwise:
        res['phloem_levels_tried'] = _describe_levels(tried)
        if unclear:
            flags.append('phloem poles unclear')
    res['n_phloem_cells'] = len(ph)
    if len(ph) < min_cells:
        res['flags'] = f'only {len(ph)} phloem cells'
        return res
    ph_xy = ph[[x_col, y_col]].to_numpy(float)
    pp = two_poles(ph_xy, trim=trim)
    c = pp['centres']
    res.update(
        pole1_x=c[0, 0], pole1_y=c[0, 1], pole2_x=c[1, 0], pole2_y=c[1, 1],
        pole1_cells=int(pp['sizes'][0]), pole2_cells=int(pp['sizes'][1]),
        n_dropped=int((~pp['kept']).sum()),
    )  # fmt: skip
    axis = c[1] - c[0]
    res['distance_px'] = float(np.linalg.norm(axis))
    if pp['sizes'].min() < 2:
        flags.append('a pole has a single cell')
    if res['distance_px'] < 2.0 * pp['spread']:
        flags.append('poles poorly separated')
    res['_phloem_xy'] = ph_xy
    all_classes = [c for lv in levels for c in lv]
    res['_phloem_all_xy'] = select_cells(df, class_col, all_classes)[
        [x_col, y_col]
    ].to_numpy(float)
    res['_phloem_labels'] = pp['labels']

    if xylem_classes:
        xy_cells = select_cells(df, class_col, xylem_classes)
        if len(xy_cells) < min_cells:
            flags.append(f'only {len(xy_cells)} xylem cells')
        else:
            x_xy = xy_cells[[x_col, y_col]].to_numpy(float)
            xp = two_poles(x_xy, trim=trim)
            xc = xp['centres']
            xaxis = xc[1] - xc[0]
            n = np.linalg.norm(xaxis)
            if n > 0:
                perp = np.array([-xaxis[1], xaxis[0]]) / n
                res['distance_xylem_px'] = float(abs(axis @ perp))
                cosang = abs(axis @ xaxis) / (np.linalg.norm(axis) * n + 1e-12)
                res['angle_deg'] = float(
                    np.degrees(np.arccos(np.clip(cosang, 0, 1)))
                )
                if abs(res['angle_deg'] - 90) > 20:
                    flags.append('axes not perpendicular')
            res.update(
                xylem1_x=xc[0, 0], xylem1_y=xc[0, 1],
                xylem2_x=xc[1, 0], xylem2_y=xc[1, 1],
            )  # fmt: skip
            res['_xylem_xy'] = x_xy
            res['_xylem_labels'] = xp['labels']
    res['flags'] = '; '.join(flags)
    return res


def measure_all(tables, um_per_px=None, scales=None, **kw):
    """
    DataFrame with one row per image. um_per_px: one pixel size for all images;
    scales: {image key: um per pixel} overriding it per image (mixed
    magnifications). Columns ending in _um are added when a size is known.
    Returns (results DataFrame, {image: arrays for plot_image}).
    """
    rows, details = [], {}
    scales = {image_key(k): v for k, v in (scales or {}).items()}
    for name, df in tables.items():
        r = measure_image(df, **kw)
        details[name] = {k: r.pop(k) for k in list(r) if k.startswith('_')}
        size = scales.get(image_key(name), um_per_px)
        r = {'image': name, **r, 'um_per_px': size if size else np.nan}
        if size:
            r['distance_um'] = r['distance_px'] * size
            r['distance_xylem_um'] = r['distance_xylem_px'] * size
        rows.append(r)
    return pd.DataFrame(rows), details


# ------------------------------------------------- compare with manual
def compare_with_manual(
    results, manual, image_col, value_col, use='distance_um'
):
    """
    Merge results with a manual table by image name and summarise the
    agreement. Returns (merged DataFrame, dict of statistics, unmatched names).
    Statistics: n, bias (tool - manual), mean absolute error, relative error
    (%), Pearson r, 95% limits of agreement (Bland-Altman).
    """
    m = manual[[image_col, value_col]].copy()
    m['_key'] = m[image_col].map(image_key)
    m = m.rename(columns={value_col: 'manual'}).drop(columns=[image_col])
    r = results.copy()
    r['_key'] = r['image'].map(image_key)
    merged = r.merge(m, on='_key', how='outer', indicator=True)
    unmatched = merged.loc[merged['_merge'] != 'both', '_key'].tolist()
    merged = merged[merged['_merge'] == 'both'].drop(
        columns=['_merge', '_key']
    )
    ok = merged[use].notna() & merged['manual'].notna()
    a, b = (
        merged.loc[ok, use].to_numpy(float),
        merged.loc[ok, 'manual'].to_numpy(float),
    )
    merged['error'] = merged[use] - merged['manual']
    stats = {'n': int(ok.sum())}
    if stats['n'] >= 2:
        d = a - b
        stats.update(
            bias=float(d.mean()),
            mean_abs_error=float(np.abs(d).mean()),
            relative_error_pct=float(100 * np.abs(d).mean() / b.mean()),
            pearson_r=float(np.corrcoef(a, b)[0, 1]),
            loa_low=float(d.mean() - 1.96 * d.std(ddof=1)),
            loa_high=float(d.mean() + 1.96 * d.std(ddof=1)),
        )
    return merged, stats, unmatched


# ------------------------------------------------------------- figures
def plot_image(name, res_row, details, path):
    """Cells, poles and the measuring line of one image (for checking)."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    if '_phloem_xy' in details:
        xy, lab = details['_phloem_xy'], details['_phloem_labels']
        for i, col in ((0, '#c0392b'), (1, '#e67e22')):
            ax.scatter(
                *xy[lab == i].T, s=14, color=col, label=f'phloem pole {i + 1}'
            )
        ax.scatter(
            *xy[lab < 0].T, s=14, color='grey', marker='x', label='dropped'
        )
        ax.plot(
            [res_row['pole1_x'], res_row['pole2_x']],
            [res_row['pole1_y'], res_row['pole2_y']], 'k-', lw=1,
        )  # fmt: skip
    if '_xylem_xy' in details:
        xy = details['_xylem_xy']
        ax.scatter(*xy.T, s=14, color='#2c7fb8', label='xylem')
        if 'xylem1_x' in res_row and not np.isnan(res_row['xylem1_x']):
            ax.plot(
                [res_row['xylem1_x'], res_row['xylem2_x']],
                [res_row['xylem1_y'], res_row['xylem2_y']], 'b--', lw=1,
            )  # fmt: skip
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(
        f'{name}\n{res_row["distance_px"]:.1f} px  {res_row["flags"]}',
        fontsize=7,
    )
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def find_sessions(path):
    """{image key: session file} from a batch results folder (or the folder
    above it, which holds a wseg_batch folder)."""
    path = Path(path)
    files = list(path.glob('*_session.h5'))
    if not files and (path / 'wseg_batch').is_dir():
        files = list((path / 'wseg_batch').glob('*_session.h5'))
    return {image_key(f.name[: -len('_session.h5')]): f for f in files}


def session_image(path):
    """The RGB image of a batch session (H, W, 3), uint8 for display."""
    import h5py

    with h5py.File(path, 'r') as f:
        raw = f['raw'][()]
    img = np.asarray(raw)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    img = img[..., :3]
    if img.dtype != np.uint8:
        img = img.astype(float)
        lo, hi = np.percentile(img, (1, 99))
        img = np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1)
        img = (img * 255).astype(np.uint8)
    return img


def plot_overlay(name, res_row, details, image, path):
    """The batch's RGB image with the cells used, the pole centroids, the
    line between the phloem poles and the line through the xylem poles, to
    check a measurement by eye. Cells of the phloem classes that were not
    used are shown as grey rings."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    h, w = image.shape[:2]
    fig, ax = plt.subplots(figsize=(7, 7 * h / max(w, 1)))
    ax.imshow(image)
    if '_phloem_all_xy' in details:
        ax.scatter(
            *details['_phloem_all_xy'].T, s=40, facecolors='none',
            edgecolors='#bbbbbb', linewidths=0.8, label='other phloem-type',
        )  # fmt: skip
    if '_phloem_xy' in details:
        xy, lab = details['_phloem_xy'], details['_phloem_labels']
        for i, col in ((0, '#e41a1c'), (1, '#ff7f00')):
            ax.scatter(
                *xy[lab == i].T, s=18, color=col, label=f'phloem pole {i + 1}'
            )
        if (lab < 0).any():
            ax.scatter(*xy[lab < 0].T, s=18, color='k', marker='x')
        p = [
            (res_row['pole1_x'], res_row['pole1_y']),
            (res_row['pole2_x'], res_row['pole2_y']),
        ]
        ax.plot(*zip(*p), color='yellow', lw=1.5)
        ax.scatter(
            *zip(*p), s=140, marker='*', color='yellow', ec='k', zorder=5
        )
    if '_xylem_xy' in details:
        ax.scatter(
            *details['_xylem_xy'].T, s=18, color='#1f78b4', label='xylem'
        )
        if np.isfinite(res_row.get('xylem1_x', np.nan)):
            a = np.array([res_row['xylem1_x'], res_row['xylem1_y']])
            b = np.array([res_row['xylem2_x'], res_row['xylem2_y']])
            d = b - a
            ext = 0.5 * np.linalg.norm(d)  # line goes on past the poles
            u = d / (np.linalg.norm(d) + 1e-12)
            ax.plot(*zip(a - u * ext, b + u * ext), '--', color='cyan', lw=1.2)
            ax.scatter(*zip(a, b), s=140, marker='*', color='cyan', ec='k',
                       zorder=5)  # fmt: skip
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')
    ax.set_title(
        f'{name}  level: {res_row.get("phloem_level", "")}  '
        f'{res_row["distance_px"]:.1f} px  {res_row["flags"]}',
        fontsize=7,
    )
    ax.legend(fontsize=6, loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_comparison(merged, stats, path, use='distance_um'):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4))
    a.scatter(merged['manual'], merged[use], s=18)
    lim = [
        np.nanmin([merged['manual'].min(), merged[use].min()]),
        np.nanmax([merged['manual'].max(), merged[use].max()]),
    ]
    a.plot(lim, lim, 'k--', lw=0.8)
    a.set_xlabel('manual')
    a.set_ylabel(f'tool ({use})')
    a.set_title(
        f'n={stats["n"]}  r={stats.get("pearson_r", float("nan")):.3f}'
    )
    mean = (merged['manual'] + merged[use]) / 2
    b.scatter(mean, merged['error'], s=18)
    for y in (stats.get('bias'), stats.get('loa_low'), stats.get('loa_high')):
        if y is not None:
            b.axhline(
                y,
                color='k',
                lw=0.8,
                ls='--' if y != stats.get('bias') else '-',
            )
    b.set_xlabel('mean of tool and manual')
    b.set_ylabel('tool - manual')
    b.set_title('Bland-Altman (bias, 95% limits)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
