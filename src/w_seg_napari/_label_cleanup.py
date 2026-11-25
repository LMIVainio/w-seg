"""
Clean-up of freshly computed cell labels between GASP and the measurement
(no Qt): cells that touch the image border are removed, and cells the error
model is sure about are fixed: 'fragment' cells are merged into the touching
cell they share the longest border with, 'spurious' (not a cell) cells are
deleted. 'merged' (too large) cells would have to be split, which cannot be
done safely, so they stay.
"""

import numpy as np
import pandas as pd

MODES = ('off', 'merge', 'delete', 'both')
LOG_COLUMNS = [
    'action',
    'label_id',
    'error_type',
    'p_error',
    'merged_with',
    'area_px',
    'inside_fraction',
]


def border_ids(labels):
    """Ids of the cells that touch the edge of the image."""
    edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    ids = np.unique(edge)
    return ids[ids > 0]


def _row(
    action,
    label_id,
    area,
    error_type='',
    p_error=np.nan,
    merged='',
    inside=np.nan,
):
    return {
        'action': action,
        'label_id': int(label_id),
        'error_type': error_type,
        'p_error': p_error,
        'merged_with': merged,
        'area_px': int(area),
        'inside_fraction': inside,
    }


def cleanup_labels(
    labels,
    wall=None,
    bundle=None,
    pixel_size=1.0,
    borders=False,
    mode='off',
    min_p=0.8,
    tissue=None,
    min_inside=0.5,
):
    """
    Returns (cleaned labels, log DataFrame with one row per change).

    borders  remove the cells that touch the image border (first, because a
             cell cut by the edge looks like a fragment to the error model)
    mode     'off', 'merge' (fragments), 'delete' (spurious cells) or 'both';
             needs `bundle` (the error model) and `wall` (the prediction)
    min_p    only cells with an error probability of at least this are changed
    tissue   a _tissue_mask.TissueLookup: cells of which less than
             `min_inside` of the area lies in the tissue are removed (the
             peeling layer, debris). Done first, so that the error model does
             not see them.
    """
    from w_seg_napari._flagged_cells import merge_many
    from w_seg_napari._segmentation_errors import flag_labels

    if mode not in MODES:
        raise ValueError(f'Unknown clean-up mode {mode!r}.')
    labels = np.array(labels, copy=True)
    rows = []

    if tissue is not None:
        ids, frac = tissue.outside_cells(labels, min_inside)
        if ids:
            areas = dict(
                zip(*np.unique(labels, return_counts=True), strict=True)
            )
            labels[np.isin(labels, ids)] = 0
            rows += [
                _row('outside_tissue', i, areas[i], inside=frac[i])
                for i in ids
            ]

    if borders:
        ids = border_ids(labels)
        if len(ids):
            areas = dict(
                zip(*np.unique(labels, return_counts=True), strict=True)
            )
            labels[np.isin(labels, ids)] = 0
            rows += [_row('border', i, areas[i]) for i in ids]

    if mode != 'off':
        if bundle is None or wall is None:
            raise ValueError(
                'The clean-up of flagged cells needs an error model.'
            )
        scores, _ = flag_labels(labels, wall, bundle, pixel_size)
        sure = scores[scores['p_error'] >= min_p]
        areas = dict(zip(*np.unique(labels, return_counts=True), strict=True))
        if mode in ('delete', 'both'):
            bad = sure[sure['error_type'] == 'spurious']
            labels[np.isin(labels, bad['label_id'].to_numpy())] = 0
            rows += [
                _row('delete', r.label_id, areas.get(r.label_id, 0),
                     r.error_type, r.p_error)
                for r in bad.itertuples()
            ]  # fmt: skip
        if mode in ('merge', 'both'):
            frag = sure[sure['error_type'] == 'fragment']
            ids = [
                int(i)
                for i in frag['label_id']
                if areas.get(int(i), 0) and (labels == i).any()
            ]
            r_, c_, v_, merges, alone = merge_many(labels, ids)
            labels[r_, c_] = v_
            info = frag.set_index('label_id')
            for cell, others in merges.items():
                rows.append(
                    _row('merge', cell, areas.get(cell, 0),
                         info.at[cell, 'error_type'], info.at[cell, 'p_error'],
                         ' '.join(str(int(o)) for o in others))
                )  # fmt: skip
            for cell in alone:
                rows.append(
                    _row('not_merged_no_neighbour', cell, areas.get(cell, 0),
                         info.at[cell, 'error_type'], info.at[cell, 'p_error'])
                )  # fmt: skip
    return labels, pd.DataFrame(rows, columns=LOG_COLUMNS)


def summarize(log):
    """'12 border cells removed, 3 merged, 1 deleted' (empty log: nothing)."""
    counts = log['action'].value_counts()
    names = {
        'outside_tissue': 'cell(s) outside the tissue removed',
        'border': 'border cell(s) removed',
        'merge': 'fragment(s) merged',
        'delete': 'not-a-cell(s) deleted',
        'not_merged_no_neighbour': 'fragment(s) without a neighbour left',
    }
    parts = [f'{counts[k]} {v}' for k, v in names.items() if k in counts]
    return ', '.join(parts) if parts else 'nothing to change'
