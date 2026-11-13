"""Training data (5-channel tifs) from batch session files, no Qt.

A session holds the image (`raw`), the model's wall prediction (`pred_sem`)
and, after reviewing, the corrected cell labels. The wall channel of the
training tif is the prediction (thresholded: the training reads it as a
yes/no mask) with every wall pixel inside a corrected cell removed, so only
the walls that really lie between the corrected cells are left.
"""

from pathlib import Path

import numpy as np

AUTO = 'auto'


def session_wall(pred_sem, labels, threshold=0.5):
    """
    Binary wall mask (uint8) from a wall prediction and the corrected labels.

    threshold   a number (wall where the prediction is above it) or AUTO for
                Otsu's threshold of this image's prediction
    Wall pixels that lie on a labelled cell are removed: in the corrected
    labels the walls are the pixels between cells, label 0.
    """
    pred = np.squeeze(np.asarray(pred_sem))
    lab = np.squeeze(np.asarray(labels))
    if pred.ndim != 2 or pred.shape != lab.shape:
        raise ValueError(
            f'The prediction {pred.shape} and the labels {lab.shape} must be '
            'two images of the same size.'
        )
    if threshold == AUTO:
        threshold = auto_threshold(pred)
    wall = pred > threshold
    wall &= lab == 0
    return wall.astype(np.uint8)


def auto_threshold(pred):
    """Otsu's threshold; 0.5 when the prediction is one flat value."""
    from skimage.filters import threshold_otsu

    pred = np.asarray(pred, np.float32)
    if pred.min() == pred.max():
        return 0.5
    return float(threshold_otsu(pred))


def session_training_stack(raw, pred_sem, labels, threshold=0.5):
    """(5, H, W) stack: R, G, B, wall (see session_wall), corrected labels."""
    from w_seg_napari._batch_io import to_rgb
    from w_seg_napari._training_data import build_training_stack

    lab = np.squeeze(np.asarray(labels))
    wall = session_wall(pred_sem, lab, threshold)
    return build_training_stack(to_rgb(raw), wall, lab)


def read_session_sample(path, require_corrected=True):
    """
    (sample, None) with name, raw, pred_sem, labels and labels_origin from a
    session file; (None, reason) when it cannot be used.
    """
    import h5py

    from w_seg_napari._session_io import SESSION_SUFFIX, editable_labels

    path = Path(path)
    name = path.name.removesuffix(SESSION_SUFFIX)
    try:
        with h5py.File(path, 'r') as f:
            raw = f['raw'][()] if 'raw' in f else None
            pred = f['pred_sem'][()] if 'pred_sem' in f else None
            store = {
                k: f[k][()]
                for k in ('labels_corrected', 'gasp_labels', 'sp_labels')
                if k in f
            }
    except (OSError, KeyError, ValueError) as e:
        return None, f'cannot read the session ({e})'
    labels, origin = editable_labels(store)
    if raw is None:
        return None, 'it holds no image (raw)'
    if pred is None:
        return None, 'it holds no wall prediction (pred_sem)'
    if labels is None:
        return None, 'it holds no cell labels'
    if require_corrected and origin != 'corrected':
        return None, (
            f'no corrected labels saved (it only has the {origin} labels)'
        )
    return {
        'name': name,
        'raw': raw,
        'pred_sem': pred,
        'labels': labels,
        'labels_origin': origin,
    }, None


def split_names(names, val_fraction=0.0, seed=0):
    """(train names, validation names): whole images, the same split every time."""
    names = sorted(names)
    if not names or val_fraction <= 0:
        return names, []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(names))
    n_val = min(max(1, round(len(names) * val_fraction)), len(names) - 1)
    val = {names[i] for i in order[:n_val]}
    return [n for n in names if n not in val], [n for n in names if n in val]
