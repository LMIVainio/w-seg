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
    pair_layer        the same for any prefix (errortype_<image>)
    merge_partners    the touching cells of one cell, longest border first
    merge_cells       merge a cell with chosen neighbours (returns the edit)
"""

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

# prefixes the batch puts before the image name
_LABEL_PREFIXES = ('corrected_', 'gasp_cells_', 'sp_', '[Predicted Classes] ')


# codes of the errortype_<image> layer (same as _segmentation_errors.ERROR_CODES)
KIND_NAMES = {0: 'ok', 1: 'merged', 2: 'fragment', 3: 'spurious'}

# what the Error type box offers: text -> the kinds it stands for (None = all)
KIND_CHOICES = {
    'Any kind': None,
    'Too large: should be split': ('merged',),
    'Too small: should be merged into a neighbour': ('fragment',),
    'Not a cell: should be deleted': ('spurious',),
    'Too small or not a cell: merge or delete': ('fragment', 'spurious'),
}


def cell_scores(labels, values, kinds=None) -> pd.DataFrame:
    """
    label_id, p_error (median of `values` inside the cell), error_type and
    area per cell. `kinds` is the optional errortype_<image> image; without it
    error_type is 'unknown'.
    """
    labels = np.asarray(labels)
    values = np.asarray(values, np.float32)
    if labels.shape != values.shape:
        raise ValueError(
            f'The labels {labels.shape} and the error layer {values.shape} '
            'do not have the same shape.'
        )
    if kinds is not None:
        kinds = np.asarray(kinds)
        if kinds.shape != labels.shape:
            raise ValueError(
                f'The labels {labels.shape} and the error type layer '
                f'{kinds.shape} do not have the same shape.'
            )
    rows = []
    for i, sl in enumerate(
        ndi.find_objects(labels.astype(np.int64, copy=False))
    ):
        if sl is None:
            continue
        mask = labels[sl] == i + 1
        kind = 'unknown'
        if kinds is not None:
            kind = KIND_NAMES.get(int(np.median(kinds[sl][mask])), 'ok')
        rows.append(
            (
                i + 1,
                float(np.median(values[sl][mask])),
                kind,
                int(mask.sum()),
            )
        )
    return pd.DataFrame(
        rows, columns=['label_id', 'p_error', 'error_type', 'area']
    )


def select_range(scores, lower, upper, min_area=0, kinds=None) -> pd.DataFrame:
    """
    Cells with lower <= p_error <= upper, the most likely errors first.
    `kinds`: only cells whose error_type is one of these names (None = all).
    """
    keep = (
        (scores['p_error'] >= lower)
        & (scores['p_error'] <= upper)
        & (scores['area'] >= min_area)
        & (scores['label_id'] != 0)
    )
    if kinds is not None:
        keep &= scores['error_type'].isin(kinds)
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


def pair_layer(target_name, names, prefix):
    """The `<prefix><image>` layer that belongs to a labels layer (by name);
    the only candidate when there is just one; None if unclear."""
    base = target_name
    for p in _LABEL_PREFIXES:
        if base.startswith(p):
            base = base[len(p) :]
            break
    wanted = f'{prefix}{base}'
    if wanted in names:
        return wanted
    return names[0] if len(names) == 1 else None


def pair_error_layer(target_name, error_names):
    """The error_<image> layer that belongs to a labels layer, None if unclear."""
    return pair_layer(target_name, error_names, 'error_')


def _box(mask, pad, shape):
    rows, cols = np.nonzero(mask)
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, shape[1])
    return slice(r0, r1), slice(c0, c1)


def merge_partners(labels, cell, gap=2, min_border=1):
    """
    The cells touching `cell` as [(label id, border length in pixels)],
    longest border first. gap=2 reaches across a one-pixel wall.
    Only the neighbourhood of the cell is looked at.
    """
    from w_seg_napari._curation_ops import border_table

    labels = np.asarray(labels)
    here = labels == cell
    if not here.any():
        return []
    box = _box(here, int(gap) + 1, labels.shape)
    out = {}
    for r in border_table(labels[box], gap):
        if cell not in (r['a'], r['b']) or r['border'] < min_border:
            continue
        other = r['b'] if r['a'] == cell else r['a']
        out[int(other)] = int(r['border'])
    return sorted(out.items(), key=lambda kv: -kv[1])


def merge_cells(labels, cell, others, gap=2):
    """
    Merge `others` into `cell` (the cell keeps its label id). The one-pixel
    wall between them and holes are filled, nothing of other cells is taken.
    Returns (rows, cols, values): the pixels that change and their new
    labels, ready for napari's data_setitem; labels itself is not changed.
    """
    from w_seg_napari._curation_ops import merge_label_groups

    labels = np.asarray(labels)
    ids = [int(cell), *[int(o) for o in others if int(o) != int(cell)]]
    box = _box(np.isin(labels, ids), int(gap) + 1, labels.shape)
    before = labels[box]
    after = before.copy()
    merge_label_groups(
        after,
        [ids],
        rule='First Selected ID',
        fill_holes=True,
        close_gap=max(int(gap) // 2, 0),
    )
    # never take pixels of a third cell (e.g. one enclosed by the merge)
    changed = (after != before) & ((before == 0) | np.isin(before, ids))
    rows, cols = np.nonzero(changed)
    return (
        rows + box[0].start,
        cols + box[1].start,
        after[rows, cols],
    )
