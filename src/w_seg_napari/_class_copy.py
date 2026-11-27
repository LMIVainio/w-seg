"""Start a class layer from classes that already exist (no napari, no Qt)."""

import re

import numpy as np


def parse_class_list(text):
    """Set of class ids from text like '1,2,4', '[1, 2, 4]' or '1-3, 7';
    None when the text is empty (= all classes). Raises ValueError."""
    text = re.sub(r'[\[\](){}]', ' ', str(text or '')).strip()
    if not text:
        return None
    ids = set()
    for part in re.split(r'[,;\s]+', text):
        if not part:
            continue
        m = re.fullmatch(r'(\d+)\s*-\s*(\d+)', part)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if hi < lo:
                lo, hi = hi, lo
            ids.update(range(lo, hi + 1))
        elif part.isdigit():
            ids.add(int(part))
        else:
            raise ValueError(f'Cannot read {part!r} as a class number')
    return ids


def copy_classes(template, classes, keep=None):
    """Class layer for painting: the pixels that are cells in `template` (the
    class layer with the cells separated where they touch) get the class of
    `classes`. With `keep` (set of ids) only those classes are copied; the
    other cells become 0 (not annotated) to be painted again."""
    template = np.asarray(template)
    classes = np.asarray(classes)
    if classes.shape != template.shape:
        raise ValueError(
            f'The class layer {classes.shape} and the cells {template.shape} '
            'differ in shape'
        )
    out = np.where(template > 0, classes, 0)
    if keep is not None:
        out = np.where(np.isin(out, list(keep)), out, 0)
    return out.astype(np.uint32)
