"""
Find and visit the cells the error model flagged (no napari or Qt here).

The batch stores a map with every cell filled by its error probability
(error_<image> layer). From it and a labels layer:

    cell_scores       p_error and area of every cell (median over the cell, so a
                      cell you have merged by hand is judged by its larger part)
    select_range      the cells with p_error between a lower and an upper limit,
                      worst first
    extract           a labels image holding only those cells (same label ids)
    changed_ids       which highlighted cells have been edited since
    find_cell         where a cell is now (None when it was merged or deleted)
    next_existing     the next cell of a list that still exists
    pair_error_layer  which error layer goes with a labels layer, by name
"""

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

# prefixes the batch puts before the image name
_LABEL_PREFIXES = ('corrected_', 'gasp_cells_', 'sp_', '[Predicted Classes] ')


def cell_scores(labels, values) -> pd.DataFrame:
    """label_id, p_error (median of `values` inside the cell) and area per cell."""
    labels = np.asarray(labels)
    values = np.asarray(values, np.float32)
    if labels.shape != values.shape:
        raise ValueError(
            f'The labels {labels.shape} and the error layer {values.shape} '
            'do not have the same shape.'
        )
    rows = []
    for i, sl in enumerate(
        ndi.find_objects(labels.astype(np.int64, copy=False))
    ):
        if sl is None:
            continue
        mask = labels[sl] == i + 1
        rows.append(
            (i + 1, float(np.median(values[sl][mask])), int(mask.sum()))
        )
    return pd.DataFrame(rows, columns=['label_id', 'p_error', 'area'])


def select_range(scores, lower, upper, min_area=0) -> pd.DataFrame:
    """Cells with lower <= p_error <= upper, the most likely errors first."""
    keep = (
        (scores['p_error'] >= lower)
        & (scores['p_error'] <= upper)
        & (scores['area'] >= min_area)
        & (scores['label_id'] != 0)
    )
    return (
        scores[keep]
        .sort_values('p_error', ascending=False)
        .reset_index(drop=True)
    )


def extract(labels, ids):
    """The labels with only the cells in `ids` kept (everything else 0)."""
    labels = np.asarray(labels)
    return np.where(np.isin(labels, np.asarray(list(ids))), labels, 0).astype(
        labels.dtype
    )


def changed_ids(labels, areas):
    """
    The labels of `areas` ({label id: pixel count when it was highlighted})
    whose pixel count in `labels` is different now: painted, merged into
    another cell, grown by a merge, or deleted.
    """
    flat = np.asarray(labels).ravel().astype(np.int64, copy=False)
    counts = np.bincount(flat) if flat.size else np.zeros(0, np.int64)
    changed = []
    for cell, area in areas.items():
        now = int(counts[cell]) if cell < len(counts) else 0
        if now != area:
            changed.append(int(cell))
    return changed


def find_cell(labels, label_id):
    """(row, col) of the centre of a cell, None when the label is gone."""
    rows, cols = np.nonzero(np.asarray(labels) == label_id)
    if rows.size == 0:
        return None
    return float(rows.mean()), float(cols.mean())


def next_existing(queue, labels, start, step):
    """
    Index of the first cell in `queue` (a list of label ids) at or after
    `start` (moving by `step`, +1 or -1) that is still in the labels; None when
    there is none in that direction.
    """
    present = set(np.unique(np.asarray(labels)).tolist())
    i = start
    while 0 <= i < len(queue):
        if int(queue[i]) in present:
            return i
        i += step
    return None


def pair_error_layer(target_name, error_names):
    """The error_<image> layer that belongs to a labels layer, None if unclear."""
    base = target_name
    for prefix in _LABEL_PREFIXES:
        if base.startswith(prefix):
            base = base[len(prefix) :]
            break
    wanted = f'error_{base}'
    if wanted in error_names:
        return wanted
    return error_names[0] if len(error_names) == 1 else None
