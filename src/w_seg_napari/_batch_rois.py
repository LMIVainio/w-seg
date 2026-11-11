"""
The region of each image that the batch should work on (no Qt or napari).

batch_rois.json, in the batch's results folder:

    {"format_version": 1,
     "rois": {"root1.tif": [y0, x0, y1, x1], ...}}

A region is a rectangle in pixels of the original image (y0, x0 included,
y1, x1 excluded). An image without an entry is processed whole.
"""

import json
import os
from pathlib import Path

import numpy as np

FILENAME = 'batch_rois.json'
FORMAT_VERSION = 1


def rois_path(out_dir) -> Path:
    return Path(out_dir) / FILENAME


def load_rois(out_dir) -> dict:
    """{image file name: [y0, x0, y1, x1]}; {} if there is no (valid) file."""
    try:
        doc = json.loads(rois_path(out_dir).read_text(encoding='utf-8'))
        rois = doc['rois']
        return {
            str(k): [int(x) for x in v] for k, v in rois.items() if len(v) == 4
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


def save_rois(out_dir, rois: dict) -> Path:
    """Write the regions (atomically). With no regions the file is removed."""
    path = rois_path(out_dir)
    if not rois:
        path.unlink(missing_ok=True)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        'format_version': FORMAT_VERSION,
        'rois': {k: [int(x) for x in v] for k, v in sorted(rois.items())},
    }
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(doc, indent=1), encoding='utf-8')
    os.replace(tmp, path)
    return path


def normalize_roi(roi, shape):
    """The region clipped to an image of `shape` (rows, columns, ...) as a
    list of ints, or None if nothing is left of it or it covers the whole
    image (then there is nothing to crop)."""
    if roi is None:
        return None
    h, w = int(shape[0]), int(shape[1])
    y0, x0, y1, x1 = (int(round(v)) for v in roi)
    y0, y1 = sorted((max(y0, 0), min(y1, h)))
    x0, x1 = sorted((max(x0, 0), min(x1, w)))
    if y1 - y0 < 2 or x1 - x0 < 2:
        return None
    if (y0, x0, y1, x1) == (0, 0, h, w):
        return None
    return [y0, x0, y1, x1]


def roi_from_vertices(vertices, shape):
    """Bounding box of a drawn shape (vertices as (row, column) in array
    coordinates) as a normalized region."""
    v = np.asarray(vertices, dtype=float)[:, -2:]
    return normalize_roi(
        [
            np.floor(v[:, 0].min()),
            np.floor(v[:, 1].min()),
            np.ceil(v[:, 0].max()),
            np.ceil(v[:, 1].max()),
        ],
        shape,
    )


def crop_image(image, roi):
    """The part of the image inside the region (a copy)."""
    if roi is None:
        return image
    y0, x0, y1, x1 = roi
    return np.ascontiguousarray(image[y0:y1, x0:x1])
