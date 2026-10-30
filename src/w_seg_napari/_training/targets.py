"""
Target construction for the W-net. Pure numpy/scipy (no torch), so it can be
tested on its own and reused by the plugin.
"""

import numpy as np
from scipy import ndimage
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


def edge_mask(inst):
    """True on wall pixels (label 0) and on pixels whose 4-neighbour has another label."""
    edge = inst == 0
    for axis in (0, 1):
        diff = np.diff(inst, axis=axis) != 0
        lead = [slice(None)] * 2
        trail = [slice(None)] * 2
        lead[axis], trail[axis] = slice(0, -1), slice(1, None)
        edge[tuple(lead)] |= diff
        edge[tuple(trail)] |= diff
    return edge


def compute_geometry(
    inst,
    crop,
    pad,
    skeleton_width=2,
    skeleton_min_rel=0.2,
    min_edge_distance=2.0,
):
    """
    Geometry targets for the geometry head, from the same padded label crop as
    compute_affinities. Returns [crop, crop, 4] float32:
      0  edge distance, divided by the largest edge distance inside the same cell
         (0 on edges and background, 1 on the cell's deepest point)
      1  dir_y   unit vector pointing away from the nearest edge point
      2  dir_x   (zero closer than `min_edge_distance` px to an edge)
      3  skeleton: medial axis of every cell, dilated to `skeleton_width` px;
         skeleton pixels shallower than `skeleton_min_rel` of the cell's
         largest distance are dropped (they are mostly spurs)
    One distance transform covers all cells at once: the nearest edge pixel of
    a pixel inside cell A always lies on A's own outline.
    """
    from skimage.morphology import medial_axis

    size = crop + 2 * pad
    if inst.shape != (size, size):
        raise ValueError(f'inst must be {(size, size)}, got {inst.shape}')

    edge = edge_mask(inst)
    interior = ~edge
    out = np.zeros((size, size, 4), dtype=np.float32)
    if not interior.any():
        return out[pad : pad + crop, pad : pad + crop]

    dist, idx = distance_transform_edt(interior, return_indices=True)
    dist = dist.astype(np.float32)

    # largest distance inside every cell, looked up per pixel
    ids, inv = np.unique(inst, return_inverse=True)
    inv = inv.reshape(inst.shape)
    cell_max = np.zeros(len(ids), dtype=np.float32)
    np.maximum.at(cell_max, inv.ravel(), dist.ravel())
    norm = np.where(interior, dist / np.maximum(cell_max[inv], 1e-6), 0.0)
    out[..., 0] = np.clip(norm, 0.0, 1.0)

    # direction: from the nearest edge point to the pixel, as a unit vector
    yy, xx = np.indices(inst.shape)
    vy = yy - idx[0]
    vx = xx - idx[1]
    valid = interior & (dist > min_edge_distance)
    length = np.maximum(np.hypot(vy, vx), 1e-6)
    out[..., 1] = np.where(valid, vy / length, 0.0)
    out[..., 2] = np.where(valid, vx / length, 0.0)

    # skeleton
    skel = medial_axis(interior)
    skel &= out[..., 0] >= skeleton_min_rel
    skel &= dist >= 1.5
    if skeleton_width > 1:
        skel = ndimage.binary_dilation(skel, iterations=skeleton_width // 2)
        skel &= interior
    out[..., 3] = skel

    return out[pad : pad + crop, pad : pad + crop]
