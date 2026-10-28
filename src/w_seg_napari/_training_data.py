"""Build a 5-channel training stack: R, G, B, wall mask, instance labels.

This is the layout the training code reads (``stack[0:3]`` image,
``stack[3]`` wall mask, ``stack[4]`` labels). Pure numpy/tifffile, no Qt.
"""

import numpy as np

MAX_EXACT_FLOAT_LABEL = 2**24  # float32 stores integers exactly up to here


def as_rgb_hwc(image):
    """Return the image as (H, W, 3). Greyscale is repeated into 3 channels."""
    img = np.asarray(image)
    if img.ndim == 2:
        return np.repeat(img[..., None], 3, axis=-1)
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        return img[..., :3]
    if img.ndim == 3 and img.shape[0] in (3, 4):
        return np.moveaxis(img[:3], 0, -1)
    if img.ndim == 3 and img.shape[-1] == 1:
        return np.repeat(img, 3, axis=-1)
    raise ValueError(
        f'Image of shape {img.shape} is not a greyscale or RGB image.'
    )


def wall_mask(semantic, threshold=0.5):
    """Binary wall mask (uint8, 0/1) from a prediction, mask or labels array."""
    sem = np.squeeze(np.asarray(semantic))
    if sem.ndim != 2:
        raise ValueError(f'Semantic layer must be 2D, got shape {sem.shape}.')
    if sem.dtype.kind == 'f':
        return (sem > threshold).astype(np.uint8)
    return (sem > 0).astype(np.uint8)


def build_training_stack(
    image, semantic, labels, threshold=0.5, mask_labels=False
):
    """Stack image (3 ch), wall mask and labels into a (5, H, W) array.

    mask_labels: use the wall mask to cut the walls out of the labels, so wall
    pixels become 0 (cells keep their ids, background keeps its label).
    """
    rgb = as_rgb_hwc(image)
    wall = wall_mask(semantic, threshold)
    lab = np.squeeze(np.asarray(labels))
    if lab.ndim != 2:
        raise ValueError(f'Labels layer must be 2D, got shape {lab.shape}.')
    if lab.min() < 0:
        raise ValueError('Labels contain negative values.')
    shapes = {rgb.shape[:2], wall.shape, lab.shape}
    if len(shapes) != 1:
        raise ValueError(
            'Image, semantic and labels must have the same size, got '
            f'{rgb.shape[:2]}, {wall.shape} and {lab.shape}. Crop them together.'
        )
    if mask_labels:
        lab = np.where(wall > 0, 0, lab)
    top = int(lab.max())
    if rgb.dtype.kind == 'f':
        if top >= MAX_EXACT_FLOAT_LABEL:
            raise ValueError('Too many labels to store next to a float image.')
        dtype = np.float32
    else:
        dtype = np.result_type(rgb.dtype, np.min_scalar_type(top), np.uint8)
    stack = np.empty((5, *wall.shape), dtype=dtype)
    stack[0:3] = np.moveaxis(rgb, -1, 0)
    stack[3] = wall
    stack[4] = lab
    return stack


def save_training_stack(path, stack):
    import tifffile

    tifffile.imwrite(path, stack, photometric='minisblack', compression='zlib')
