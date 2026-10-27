"""
Small coordinate helpers (pure numpy/pandas, no napari) shared by the ROI and
morphology code.

napari keeps two coordinate systems for every layer:

* data (array) coordinates: the row/column index in layer.data
* world coordinates: data * layer.scale + layer.translate

Two layers show the same place when their world coordinates agree, not when
their array indices agree. Cropping a layer (ROI selection) changes its
translate, so any code that moves between two layers has to go through world
coordinates.
"""

import numpy as np


def yx(values, default):
    """The last two entries of a scale/translate vector as floats (y, x)."""
    if values is None:
        return np.array([default, default], dtype=float)
    v = np.asarray(values, dtype=float).ravel()
    return v[-2:] if v.size >= 2 else np.full(2, float(v[0]))


def data_to_world(index, scale=None, translate=None, axis=0):
    """Array index (or array of them) -> world coordinate along one axis."""
    s, t = yx(scale, 1.0)[axis], yx(translate, 0.0)[axis]
    return np.asarray(index, dtype=float) * s + t


def world_to_data(coord, scale=None, translate=None, axis=0):
    """World coordinate (or array of them) -> array index along one axis."""
    s, t = yx(scale, 1.0)[axis], yx(translate, 0.0)[axis]
    return (np.asarray(coord, dtype=float) - t) / s


def label_frame_to_image_pixels(
    df,
    pixel_spacing=1.0,
    labels_scale=None,
    labels_translate=None,
    image_scale=None,
    image_translate=None,
):
    """
    Cell positions measured on a labels array, expressed as pixel positions in
    another image layer.

    ``df`` is the skimage regionprops table used by the morphology code
    (bbox-0..3, centroid-0/1, minor_axis_length). It was computed with
    ``spacing=pixel_spacing``, so the centroid and the axis length are in
    micrometres while the bounding box is in pixels; both are brought back to
    label pixels first. Returns a copy of df with these columns in image pixels.
    """
    out = df.copy()
    sp = float(pixel_spacing) if pixel_spacing else 1.0

    def to_image(label_pixels, axis):
        world = data_to_world(
            label_pixels, labels_scale, labels_translate, axis
        )
        return world_to_data(world, image_scale, image_translate, axis)

    for col, axis in (
        ('bbox-0', 0),
        ('bbox-1', 1),
        ('bbox-2', 0),
        ('bbox-3', 1),
    ):
        if col in out:
            out[col] = np.rint(to_image(df[col].to_numpy(float), axis)).astype(
                int
            )
    for col, axis in (('centroid-0', 0), ('centroid-1', 1)):
        if col in out:
            out[col] = to_image(df[col].to_numpy(float) / sp, axis)
    if 'minor_axis_length' in out:
        ratio = yx(labels_scale, 1.0).mean() / yx(image_scale, 1.0).mean()
        out['minor_axis_length'] = (
            df['minor_axis_length'].to_numpy(float) / sp * ratio
        )
    return out


def spatial_shape(data):
    """(rows, cols) of a layer's data, or None for data without a plain shape."""
    shape = getattr(data, 'shape', None)
    if shape is None or len(shape) < 2:
        return None
    return tuple(int(n) for n in shape[:2])
