"""The YOLO class of every cell as a labels image (no Qt or napari)."""

import numpy as np


def yolo_class_names(class_map):
    """{class id: name} from YOLO's names ({index: name} or a list). The ids
    start at 1 (0 is background), the same as the "YOLO Cell Types" layer."""
    if not class_map:
        return {}
    items = (
        enumerate(class_map)
        if isinstance(class_map, (list, tuple))
        else class_map.items()
    )
    return {int(i) + 1: str(n) for i, n in items}


def yolo_label_map(labels, df, class_map, column='yolo_pred_tissue'):
    """
    Every cell of `labels` filled with its YOLO class id (see
    yolo_class_names), 0 for background and for cells without a prediction.
    Returns (image, {id: name}) or (None, {}) if the table has no prediction.
    """
    if df is None or column not in df.columns or 'label_id' not in df.columns:
        return None, {}
    names = yolo_class_names(class_map)
    if not names:  # no names from the model: number the classes found
        found = sorted(df[column].dropna().unique())
        names = {i + 1: str(n) for i, n in enumerate(found)}
    ids = {n: i for i, n in names.items()}
    valid = df.dropna(subset=[column])
    lut = np.zeros(int(np.max(labels)) + 1, np.uint16)
    cell = valid['label_id'].to_numpy().astype(np.int64)
    cls = valid[column].astype(str).map(ids).fillna(0).to_numpy()
    keep = (cell > 0) & (cell < len(lut))
    lut[cell[keep]] = cls[keep]
    return lut[np.asarray(labels)], names
