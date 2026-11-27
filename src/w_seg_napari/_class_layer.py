"""
Cell classes corrected by hand in napari, added to a morphology table
(no Qt or napari).

Work flow: copy the session's YOLO class image (layer `yolo_labels`) to a new
labels layer called `yolo_class`, paint the right class id over the cells that
are wrong (the id is the number in the class name, e.g. 10 for
10_primary_phloem_pole), save the session with that layer ticked under "Also
keep these layers". Then ``scripts/apply_class_layer.py`` adds the columns
`yolo_corrected` and `yolo_edited` to the table.

A cell counts as edited when painted pixels that differ from the original
class image cover at least `min_changed` of it; its new class is the majority
of those pixels. Without an original, every painted pixel counts (the same rule
as for the YOLO training layer).
"""

import re

import numpy as np
import pandas as pd

CLASS_LAYER = 'yolo_class'
LABELS_DATASET = {
    'corrected': 'labels_corrected',
    'gasp': 'gasp_labels',
    'superpixels': 'sp_labels',
}


def class_names_from_table(df, column='yolo_pred_tissue'):
    """{class id: name} from the probability columns (prob_07_periderm) and
    the predicted names themselves; the id is the leading number."""
    names = {}
    candidates = [
        c[len('prob_') :] for c in df.columns if c.startswith('prob_')
    ]
    if column in df.columns:
        candidates += [str(v) for v in df[column].dropna().unique()]
    for name in candidates:
        m = re.match(r'(\d+)_', name)
        if m:
            names.setdefault(int(m.group(1)), name)
    return names


def corrected_classes(labels, layer, original=None, min_changed=0.2):
    """DataFrame label_id, class_id (0 = not edited), class_fraction (share of
    the cell that was changed) for the cells of `labels`."""
    from w_seg_napari._yolo_data import cell_table

    layer = np.asarray(layer)
    if layer.shape != np.asarray(labels).shape:
        raise ValueError(
            f'layer {layer.shape} and labels {np.asarray(labels).shape} differ'
        )
    if original is not None:
        original = np.asarray(original)
        if original.shape != layer.shape:
            raise ValueError('layer and original class image differ in shape')
        painted = np.where((layer > 0) & (layer != original), layer, 0)
    else:
        painted = layer
    t = cell_table(labels, painted, min_coverage=min_changed)
    return t[['label_id', 'class_id', 'class_fraction']]


def add_corrected_column(
    df,
    classes,
    pred_column='yolo_pred_tissue',
    column='yolo_corrected',
    edited_column='yolo_edited',
):
    """Copy of the table with the corrected class (the prediction where the
    cell was not edited) and whether it was edited. Returns (table, names
    missing from the table's class list)."""
    names = class_names_from_table(df, pred_column)
    by_cell = classes.set_index('label_id')['class_id']
    new = df[pred_column].astype(object).copy()
    edited = np.zeros(len(df), dtype=bool)
    unknown = set()
    for i, cell in enumerate(df['label_id'].to_numpy()):
        cid = int(by_cell.get(cell, 0))
        if cid <= 0:
            continue
        if cid not in names:
            unknown.add(cid)
            continue
        new.iloc[i] = names[cid]
        edited[i] = names[cid] != df[pred_column].iloc[i]
    out = df.copy()
    out[column] = new
    out[edited_column] = edited
    return out, sorted(unknown)


def read_table_keep_header(path):
    """(header lines starting with '#', DataFrame)."""
    with open(path, encoding='utf-8') as f:
        head = []
        for line in f:
            if line.startswith('#'):
                head.append(line)
            else:
                break
    return head, pd.read_csv(path, comment='#')


def write_table_keep_header(path, head, df):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.writelines(head)
        df.to_csv(f, index=False, lineterminator='\n')
