"""
Training crops for the paired YOLO cell-type classifier (no napari, no Qt).

From images that are loaded together with their cell labels and a class layer
(class id per painted pixel, 0 = not annotated) this writes the folder layout
that Ultralytics classification training expects:

    <out_dir>/train/<id>_<name>/<image>_<cell>.png
    <out_dir>/val/<id>_<name>/...

Every cell becomes one crop, cut with the SAME function the plugin uses when it
predicts (``crop_yolo_bbox_torch``) and the same crop settings, so training and
prediction see identical pictures. A cell's class is the majority of the
painted class pixels inside it.

Split: when two or more images are given, whole images go to validation, so
the validation accuracy is measured on slides the model has not seen (crops of
neighbouring cells overlap, a random split inside one image is optimistic).
With one image (or when no image split gives every class a place on both
sides) the cells are split at random per class.
"""

import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd

from w_seg_napari._model_config import DEFAULT_YOLO_CROP


def safe_name(text):
    s = re.sub(r'[^\w.-]+', '_', str(text).strip()).strip('._')
    return s or 'x'


def class_folder(class_id, name):
    """'01_pavement': zero-padded id first, so alphabetical order = id order."""
    return f'{int(class_id):02d}_{safe_name(name)}'


# ------------------------------------------------------------- cell table
def cell_table(labels, class_data=None, min_coverage=0.0):
    """
    One row per cell of a label image: label_id, bbox-0..3 (ymin, xmin, ymax,
    xmax), centroid-0/1 (y, x), minor_axis_length (all in label pixels) and,
    with class_data, class_id (majority class of the painted pixels in the
    cell, 0 = none) and class_fraction (painted share of the cell).
    min_coverage: cells whose painted share is below this get class 0.
    """
    from skimage.measure import regionprops_table

    labels = np.asarray(labels)
    props = regionprops_table(
        labels.astype(np.int64),
        properties=('label', 'bbox', 'centroid', 'axis_minor_length', 'area'),
    )
    df = pd.DataFrame(props).rename(
        columns={'label': 'label_id', 'axis_minor_length': 'minor_axis_length'}
    )
    if class_data is not None:
        cls = np.asarray(class_data)
        if cls.shape != labels.shape:
            raise ValueError(
                f'Class layer {cls.shape} and labels {labels.shape} differ in shape'
            )
        lab, c = labels.ravel(), cls.ravel()
        m = (lab > 0) & (c > 0)
        class_id = np.zeros(len(df), dtype=int)
        frac = np.zeros(len(df))
        if m.any():
            width = int(c.max()) + 1
            key = lab[m].astype(np.int64) * width + c[m].astype(np.int64)
            uniq, counts = np.unique(key, return_counts=True)
            cell_of, cls_of = uniq // width, uniq % width
            best = {}  # cell -> (count, class); ties go to the smaller class id
            painted = {}
            for cell, k, n in zip(cell_of, cls_of, counts):
                painted[cell] = painted.get(cell, 0) + n
                if cell not in best or n > best[cell][0]:
                    best[cell] = (n, k)
            for i, cell in enumerate(df['label_id'].to_numpy()):
                if cell in best:
                    class_id[i] = int(best[cell][1])
                    frac[i] = painted[cell] / max(
                        float(df['area'].iloc[i]), 1.0
                    )
        class_id[frac < min_coverage] = 0
        df['class_id'] = class_id
        df['class_fraction'] = frac
    return df.drop(columns=['area'])


# ------------------------------------------------------------------ split
def split_images(counts, val_fraction=0.3, seed=0, max_exhaustive=12):
    """
    Choose validation images. counts: {image: {class_id: n_cells}}.
    Returns a set of image names for validation, or None when no choice of
    whole images leaves every class with at least one cell on both sides.
    Among valid choices the validation cell share closest to val_fraction wins
    (ties broken by `seed`).
    """
    names = sorted(counts)
    if len(names) < 2:
        return None
    classes = sorted({c for d in counts.values() for c in d})
    total = {c: sum(counts[n].get(c, 0) for n in names) for c in classes}
    n_all = sum(total.values())
    rng = np.random.default_rng(seed)
    if len(names) <= max_exhaustive:
        subsets = [
            s
            for r in range(1, len(names))
            for s in itertools.combinations(names, r)
        ]
    else:  # many images: sample candidate subsets
        subsets = []
        for _ in range(4000):
            r = int(rng.integers(1, len(names)))
            subsets.append(tuple(rng.choice(names, r, replace=False)))
    best, best_key = None, None
    for s in subsets:
        val = {c: sum(counts[n].get(c, 0) for n in s) for c in classes}
        if any(val[c] < 1 or total[c] - val[c] < 1 for c in classes):
            continue
        share = sum(val.values()) / max(n_all, 1)
        key = (abs(share - val_fraction), rng.random())
        if best_key is None or key < best_key:
            best, best_key = set(s), key
    return best


def split_cells(df, val_fraction=0.3, seed=0):
    """Boolean array: True = validation, stratified by class_id. A class with a
    single cell stays in train."""
    rng = np.random.default_rng(seed)
    is_val = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby('class_id').indices.items():
        if len(idx) < 2:
            continue
        n_val = int(np.clip(round(len(idx) * val_fraction), 1, len(idx) - 1))
        is_val[rng.choice(idx, n_val, replace=False)] = True
    return is_val


# ------------------------------------------------------------------ crops
def _to_uint8_rgb(crop):
    a = crop.detach().cpu().numpy() if hasattr(crop, 'detach') else np.asarray(crop)
    if a.ndim == 3 and a.shape[0] in (1, 3):  # [C, H, W] -> [H, W, C]
        a = np.moveaxis(a, 0, -1)
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    return np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)


def build_dataset(
    samples,
    class_names,
    out_dir,
    crop=None,
    val_fraction=0.3,
    seed=0,
    min_coverage=0.0,
    crop_fn=None,
    progress=None,
):
    """
    Write the crop dataset. samples: list of dicts with
        name, image (RGB array), labels, classes (class id per pixel, same
        shape as labels), and optionally image_scale / image_translate /
        labels_scale / labels_translate (napari layer transforms; labels and
        image may sit on different grids).
    class_names: {class_id: name} for the classes to train on (cells of other
        classes, and unannotated cells, are skipped).
    crop: crop settings (see DEFAULT_YOLO_CROP).
    Returns a summary dict: split mode, per-class counts, folder names, crop
    settings, warnings, and the ids of the validation images.
    """
    from PIL import Image

    from w_seg_napari._geometry import label_frame_to_image_pixels

    if crop_fn is None:
        from w_seg_napari._wsegutils import crop_yolo_bbox_torch as crop_fn
    crop = {**DEFAULT_YOLO_CROP, **(crop or {})}
    class_names = {int(k): v for k, v in class_names.items()}
    folders = {k: class_folder(k, v) for k, v in class_names.items()}
    if not class_names:
        raise ValueError('No classes given.')
    names = [s['name'] for s in samples]
    if len(set(names)) != len(names):
        raise ValueError('Image names must be unique.')

    # cells per image, in image pixels
    tables, counts, warnings = {}, {}, []
    for s in samples:
        df = cell_table(s['labels'], s['classes'], min_coverage)
        df = df[df['class_id'].isin(class_names)].reset_index(drop=True)
        df = label_frame_to_image_pixels(
            df,
            pixel_spacing=1.0,
            labels_scale=s.get('labels_scale'),
            labels_translate=s.get('labels_translate'),
            image_scale=s.get('image_scale'),
            image_translate=s.get('image_translate'),
        )
        tables[s['name']] = df
        counts[s['name']] = df['class_id'].value_counts().to_dict()
        if df.empty:
            warnings.append(f'{s["name"]}: no annotated cells of the chosen classes')

    n_total = sum(len(d) for d in tables.values())
    if n_total == 0:
        raise ValueError('No annotated cells found for the chosen classes.')

    val_images = split_images(counts, val_fraction, seed)
    if val_images is not None:
        mode = 'by image'
        for name, df in tables.items():
            df['split'] = np.where(name in val_images, 'val', 'train')
    else:
        mode = 'by cell'
        if len(samples) >= 2:
            warnings.append(
                'No split by whole images gives every class cells on both '
                'sides; cells were split at random instead (validation '
                'accuracy will look optimistic).'
            )
        elif len(samples) == 1:
            warnings.append(
                'One image only: cells were split at random, and crops of '
                'neighbouring cells overlap, so validation accuracy will '
                'look optimistic. Use two or more images for an honest value.'
            )
        for name, df in tables.items():
            df['split'] = np.where(split_cells(df, val_fraction, seed), 'val', 'train')

    out_dir = Path(out_dir)
    for split in ('train', 'val'):
        for folder in folders.values():
            (out_dir / split / folder).mkdir(parents=True, exist_ok=True)

    done = 0
    summary_counts = {'train': {}, 'val': {}}
    for s in samples:
        df = tables[s['name']]
        for r in df.to_dict('records'):
            tensor = crop_fn(
                bbox=(int(r['bbox-0']), int(r['bbox-1']), int(r['bbox-2']), int(r['bbox-3'])),
                image_np=s['image'],
                centroids=(round(r['centroid-0']), round(r['centroid-1'])),
                axis_l=round(r['minor_axis_length']),
                target_size=crop['target_size'],
                context_factor=crop['context_factor'],
                min_crop_size=crop['min_crop_size'],
                max_crop_size=crop['max_crop_size'],
            )
            folder = folders[int(r['class_id'])]
            path = out_dir / r['split'] / folder / f'{safe_name(s["name"])}_{int(r["label_id"])}.png'
            Image.fromarray(_to_uint8_rgb(tensor)).save(path)
            summary_counts[r['split']][folders[int(r['class_id'])]] = (
                summary_counts[r['split']].get(folders[int(r['class_id'])], 0) + 1
            )
            done += 1
            if progress is not None and done % 25 == 0:
                progress(done, n_total)
    if progress is not None:
        progress(n_total, n_total)

    for split in ('train', 'val'):
        for folder in folders.values():
            if summary_counts[split].get(folder, 0) == 0:
                warnings.append(f'No {split} crops for class {folder}')
    return {
        'out_dir': str(out_dir),
        'split_mode': mode,
        'val_images': sorted(val_images) if val_images else [],
        'counts': summary_counts,
        'folders': {str(k): v for k, v in folders.items()},
        'crop': crop,
        'n_cells': n_total,
        'warnings': warnings,
    }
