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
        raise KeyError(f'Column {class_col!r} not found (have: {list(df.columns)})')
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
def measure_image(
    df,
    phloem_classes,
    class_col='predicted_class',
    xylem_classes=None,
    y_col='centroid-0',
    x_col='centroid-1',
    trim=3.0,
    min_cells=4,
):
    """
    One result dict for one image's table (coordinates in pixels, as (x, y)).

    distance_px         between the two phloem pole centres
    distance_xylem_px   along the line perpendicular to the xylem axis (only
                        with xylem_classes; NaN otherwise)
    angle_deg           angle between phloem and xylem axes (about 90 expected)
    flags               short notes when the result deserves a look
    plus the pole / cell counts and centres for plotting.
    """
    res = {
        'n_phloem_cells': 0,
        'distance_px': np.nan,
        'distance_xylem_px': np.nan,
        'angle_deg': np.nan,
        'flags': '',
    }
    flags = []
    for col in (y_col, x_col):
        if col not in df.columns:
            raise KeyError(f'Column {col!r} not found (have: {list(df.columns)})')

    ph = select_cells(df, class_col, phloem_classes)
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
                cosang = abs(axis @ xaxis) / (
                    np.linalg.norm(axis) * n + 1e-12
                )
                res['angle_deg'] = float(np.degrees(np.arccos(np.clip(cosang, 0, 1))))
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
def compare_with_manual(results, manual, image_col, value_col, use='distance_um'):
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
    merged = merged[merged['_merge'] == 'both'].drop(columns=['_merge', '_key'])
    ok = merged[use].notna() & merged['manual'].notna()
    a, b = merged.loc[ok, use].to_numpy(float), merged.loc[ok, 'manual'].to_numpy(float)
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
            ax.scatter(*xy[lab == i].T, s=14, color=col, label=f'phloem pole {i + 1}')
        ax.scatter(*xy[lab < 0].T, s=14, color='grey', marker='x', label='dropped')
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
    ax.set_title(f'{name}\n{res_row["distance_px"]:.1f} px  {res_row["flags"]}', fontsize=7)
    ax.legend(fontsize=6)
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
    a.set_title(f'n={stats["n"]}  r={stats.get("pearson_r", float("nan")):.3f}')
    mean = (merged['manual'] + merged[use]) / 2
    b.scatter(mean, merged['error'], s=18)
    for y in (stats.get('bias'), stats.get('loa_low'), stats.get('loa_high')):
        if y is not None:
            b.axhline(y, color='k', lw=0.8, ls='--' if y != stats.get('bias') else '-')
    b.set_xlabel('mean of tool and manual')
    b.set_ylabel('tool - manual')
    b.set_title('Bland-Altman (bias, 95% limits)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
