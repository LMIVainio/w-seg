"""
Target construction for the W-net. Pure numpy/scipy (no torch), so it can be
tested on its own and reused by the plugin.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_affinities(inst, offsets, crop, pad):
    """
    inst    : [crop + 2*pad, crop + 2*pad] integer labels (0 = wall/background),
              padded so that every offset reaches at most `pad` pixels beyond
              the central crop x crop window.
    offsets : list of (dy, dx); negative components are allowed.

    Returns [crop, crop, n_offsets] float32. Channel i is 1 where the pixel p
    of the central window and the pixel p + offsets[i] carry the same label > 0.
    """
    size = crop + 2 * pad
    if inst.shape != (size, size):
        raise ValueError(f'inst must be {(size, size)}, got {inst.shape}')
    reach = max(max(abs(dy), abs(dx)) for dy, dx in offsets)
    if reach > pad:
        raise ValueError(f'largest offset ({reach}) exceeds pad ({pad})')

    a = inst[pad : pad + crop, pad : pad + crop]
    out = np.empty((crop, crop, len(offsets)), dtype=np.float32)
    for i, (dy, dx) in enumerate(offsets):
        b = inst[pad + dy : pad + dy + crop, pad + dx : pad + dx + crop]
        out[..., i] = (a == b) & (a > 0)
    return out


def edt_weights(wall, w0=10.0, sigma=5.0):
    """
    Weight map for a boolean wall mask: class balance plus exponential decay
    with the distance from the nearest wall pixel (high on and near walls).
    """
    wall = wall.astype(bool)
    n = wall.size
    n_pos = int(wall.sum())
    n_neg = n - n_pos
    if n_pos == 0:  # no walls in this crop: uniform weights
        return np.ones(wall.shape, dtype=np.float32)

    d = distance_transform_edt(~wall)  # distance to nearest wall pixel
    boundary = w0 * np.exp(-(d**2) / (2.0 * sigma**2))
    class_w = np.where(wall, n_neg / n, n_pos / n)
    return (class_w + boundary).astype(np.float32)


def describe_offsets(inst, offsets):
    """
    How informative is each offset on YOUR annotations? Run this on a full
    instance-label plane (or a large crop) before fixing the ranges.

    For every offset, over all valid pixel pairs, returns the fraction that is
        same_cell      : both pixels in the same cell      (affinity = 1)
        different_cell : both in cells, but different ones (affinity = 0)
        involves_wall  : at least one pixel is wall        (affinity = 0)

    Rule of thumb: an offset is useful when BOTH same_cell and different_cell
    are non-negligible (say above about 5 %). same_cell near 0 means the
    offset is larger than most cells; different_cell near 0 means the pairs
    mostly stay in one cell or land in walls.
    """
    h, w = inst.shape
    rows = []
    for dy, dx in offsets:
        y0, y1 = max(0, -dy), h - max(0, dy)
        x0, x1 = max(0, -dx), w - max(0, dx)
        a = inst[y0:y1, x0:x1]
        b = inst[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        n = a.size
        same = np.count_nonzero((a == b) & (a > 0))
        diff = np.count_nonzero((a != b) & (a > 0) & (b > 0))
        rows.append(
            {
                'offset': (dy, dx),
                'same_cell': same / n,
                'different_cell': diff / n,
                'involves_wall': 1.0 - (same + diff) / n,
            }
        )
    return rows
