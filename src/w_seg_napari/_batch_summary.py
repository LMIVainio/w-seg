"""
Summary tables over the results of one or several batch folders (no Qt).

The statistical unit is the IMAGE (one image = one plant at one position and
stage), not the cell: the cells of an image are not independent replicates.
So the tables are:

  cells_all.csv           every cell of every image, with group and image
                          columns (for mixed models with image as a random
                          effect)
  per_image_summary.csv   one row per image: number of cells, mean / median /
                          SD / quartiles of every measurement, the share of
                          every cell type, pixel size, analysed area, density
  per_group_summary.csv   one row per group (folder = line or treatment) and
                          measurement: mean, SD, SEM ... of the per-image
                          values, with the number of images
  summary_info.json       what the tables were made from

A group is a folder of images made with the same settings and scale.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

TABLE_SUFFIX = '_morphology.csv'
SETTINGS_NAME = 'batch_settings.json'
# columns that identify a cell or its place; they are not measurements
ID_COLUMNS = {
    'label',
    'label_id',
    'centroid-0',
    'centroid-1',
    'bbox-0',
    'bbox-1',
    'bbox-2',
    'bbox-3',
    'group',
    'image',
}
TEXT_ID_COLUMNS = {'source_file', 'labels_source'}
STATS = ('mean', 'median', 'sd', 'q25', 'q75')


def image_name(table_path):
    return Path(table_path).name.removesuffix(TABLE_SUFFIX)


def read_group(name, out_dir):
    """The cell tables of one folder as one DataFrame with group and image."""
    frames = []
    for path in sorted(Path(out_dir).glob(f'*{TABLE_SUFFIX}')):
        df = pd.read_csv(path)
        df.insert(0, 'image', image_name(path))
        df.insert(0, 'group', name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def split_columns(cells, max_classes=12):
    """
    (numeric measurement columns, category columns). Text, boolean and
    class-like integer columns (few values, 'class' or 'pred' in the name)
    are categories; ids, positions and coordinates are neither.
    """
    numeric, categories = [], []
    for col in cells.columns:
        if col in ID_COLUMNS or col in TEXT_ID_COLUMNS:
            continue
        s = cells[col]
        if (
            s.dtype == bool
            or not pd.api.types.is_numeric_dtype(s)
            or (
                pd.api.types.is_integer_dtype(s)
                and s.nunique() <= max_classes
                and any(k in col.lower() for k in ('class', 'pred'))
            )
        ):
            categories.append(col)
        else:
            numeric.append(col)
    return numeric, categories


def _group_settings(out_dir):
    """(pixel spacing in um or None, the settings document or None)."""
    path = Path(out_dir) / SETTINGS_NAME
    if not path.is_file():
        return None, None
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None, None
    spacing = (doc.get('morph_params') or {}).get('pixel_spacing')
    return (float(spacing) if spacing else None), doc


def per_image(cells, numeric, categories):
    """One row per image (see the module doc)."""
    rows = []
    for (group, image), df in cells.groupby(['group', 'image'], sort=False):
        row = {'group': group, 'image': image, 'n_cells': len(df)}
        if 'labels_source' in df and df['labels_source'].notna().any():
            row['labels_source'] = df['labels_source'].mode().iat[0]
        if 'flagged' in df:
            row['n_flagged'] = int(
                df['flagged'].fillna(False).astype(bool).sum()
            )
        for col in numeric:
            v = pd.to_numeric(df[col], errors='coerce').dropna()
            row[f'{col}__mean'] = v.mean() if len(v) else np.nan
            row[f'{col}__median'] = v.median() if len(v) else np.nan
            row[f'{col}__sd'] = v.std() if len(v) > 1 else np.nan
            row[f'{col}__q25'] = v.quantile(0.25) if len(v) else np.nan
            row[f'{col}__q75'] = v.quantile(0.75) if len(v) else np.nan
        for col in categories:
            counts = df[col].astype(str).value_counts()
            for value, n in counts.items():
                row[f'share_{col}={value}'] = n / len(df)
                row[f'n_{col}={value}'] = int(n)
        rows.append(row)
    out = pd.DataFrame(rows)
    share = [c for c in out if c.startswith(('share_', 'n_'))]
    out[share] = out[share].fillna(0.0)  # a category absent from an image
    return out


def per_group(images, numeric, categories):
    """Long table: group x measurement, from the per-image values."""
    measures = [(c, f'{c}__mean') for c in numeric]
    measures += [(c, f'{c}__median') for c in numeric]
    measures += [(c[6:], c) for c in images if c.startswith('share_')]
    measures += [('n_cells', 'n_cells'), ('n_flagged', 'n_flagged')]
    rows = []
    for group, df in images.groupby('group', sort=False):
        for label, col in measures:
            if col not in df:
                continue
            v = df[col].dropna()
            name = label
            if col.endswith('__mean'):
                name = f'{label} (mean of image means)'
            elif col.endswith('__median'):
                name = f'{label} (mean of image medians)'
            elif col.startswith('share_'):
                name = f'share of {label}'
            rows.append(
                {
                    'group': group,
                    'measure': name,
                    'n_images': len(v),
                    'mean': v.mean() if len(v) else np.nan,
                    'sd': v.std() if len(v) > 1 else np.nan,
                    'sem': v.std() / np.sqrt(len(v)) if len(v) > 1 else np.nan,
                    'min': v.min() if len(v) else np.nan,
                    'median': v.median() if len(v) else np.nan,
                    'max': v.max() if len(v) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_summary(groups, area_of=None):
    """
    groups: [(group name, results folder)]. area_of(out_dir, image) -> image
    area in pixels or None (the size of what was analysed, for the density).
    Returns {'cells', 'images', 'groups', 'info', 'warnings'}; the tables are
    None when no cell table was found.
    """
    frames, warnings, info = [], [], {'groups': {}}
    spacing = {}
    for name, out_dir in groups:
        df = read_group(name, out_dir)
        if df is None:
            warnings.append(f'{name}: no {TABLE_SUFFIX} tables in {out_dir}.')
            continue
        frames.append(df)
        px, doc = _group_settings(out_dir)
        spacing[name] = px
        info['groups'][name] = {
            'folder': str(out_dir),
            'n_images': int(df['image'].nunique()),
            'pixel_spacing_um': px,
            'model': ((doc or {}).get('model') or {}).get('name'),
            'plugin_version': (doc or {}).get('plugin_version'),
        }
        if doc is None:
            warnings.append(
                f'{name}: no {SETTINGS_NAME}, so the pixel size is unknown.'
            )
    if not frames:
        return {'cells': None, 'images': None, 'groups': None,
                'info': info, 'warnings': warnings}  # fmt: skip
    cells = pd.concat(frames, ignore_index=True)
    known = {k: v for k, v in spacing.items() if v}
    if len(set(known.values())) > 1:
        warnings.append(
            'The groups have different pixel sizes ('
            + ', '.join(f'{k}: {v:g} um' for k, v in known.items())
            + '): lengths and areas are in um but check that this is intended.'
        )
    models = {g['model'] for g in info['groups'].values() if g['model']}
    if len(models) > 1:
        warnings.append(
            'The groups were processed with different models: '
            + ', '.join(sorted(models))
            + '. They are not directly comparable.'
        )
    numeric, categories = split_columns(cells)
    images = per_image(cells, numeric, categories)
    if 'labels_source' in images:
        kinds = images.groupby('group')['labels_source'].nunique()
        if (kinds > 1).any() or images['labels_source'].nunique() > 1:
            warnings.append(
                'The cells come from different label types (corrected, GASP, '
                'superpixels; see labels_source in per_image_summary.csv): '
                'a mix of reviewed and unreviewed images can look like a '
                'treatment effect.'
            )
    dirs = dict(groups)
    px = images['group'].map(spacing)
    images.insert(2, 'pixel_spacing_um', px)
    if area_of is not None:
        area_px = [
            area_of(dirs[g], im)
            for g, im in zip(images['group'], images['image'], strict=True)
        ]
        area_um2 = np.array([a if a else np.nan for a in area_px], float) * (
            px.fillna(1.0).to_numpy() ** 2
        )
        images.insert(3, 'analysed_area_um2', area_um2)
        images.insert(4, 'cells_per_mm2', images['n_cells'] / area_um2 * 1e6)
    return {
        'cells': cells,
        'images': images,
        'groups': per_group(images, numeric, categories),
        'info': info,
        'warnings': warnings,
    }


def write_summary(result, dest):
    """Writes the tables; returns the paths."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, name in (
        ('cells', 'cells_all.csv'),
        ('images', 'per_image_summary.csv'),
        ('groups', 'per_group_summary.csv'),
    ):
        if result[key] is not None:
            paths[key] = dest / name
            result[key].to_csv(paths[key], index=False)
    info = dict(result['info'], warnings=result['warnings'])
    (dest / 'summary_info.json').write_text(json.dumps(info, indent=2))
    return paths


def session_area(out_dir, image):
    """Pixels of the analysed image (the crop if it was cropped); None if unknown."""
    try:
        from w_seg_napari import _session_io as sio

        path = Path(out_dir) / f'{image}{sio.SESSION_SUFFIX}'
        shape = sio._stored_shape(path)  # (H, W) of the session's image
    except Exception:  # no h5py, no session file, ...
        return None
    return int(shape[0] * shape[1]) if shape else None
