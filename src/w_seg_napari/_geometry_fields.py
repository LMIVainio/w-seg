"""
Helpers for the geometry head's output (architecture wasp_v4), plain numpy so
they can be tested without napari or torch.

The stitched geometry array has shape [4, H, W] with the channels
    0 edge_distance  0 at cell edges, 1 at each cell's deepest point
    1 dir_y          unit vector pointing away from the nearest edge (y part)
    2 dir_x          (x part)
    3 skeleton       probability of lying on the cell's medial axis
"""

import numpy as np

LAYER_PREFIXES = {
    'edge_distance': 'geo_dist_',
    'direction': 'geo_dir_',
    'vectors': 'geo_vec_',
    'skeleton': 'geo_skel_',
}


def clean_geometry(geo):
    """Clip a stitched geometry array to its valid ranges (float32 copy)."""
    geo = np.array(geo, dtype=np.float32, copy=True)
    if geo.ndim != 3 or geo.shape[0] != 4:
        raise ValueError(
            f'Geometry must have shape [4, H, W], got {geo.shape}'
        )
    geo[0] = np.clip(geo[0], 0.0, 1.0)
    geo[1:3] = np.clip(geo[1:3], -1.0, 1.0)
    geo[3] = np.clip(geo[3], 0.0, 1.0)
    return geo


def direction_to_rgb(vec, valid=None):
    """
    [H, W, 2] (dir_y, dir_x) in -1..1 -> RGB. Grey = no direction.
    Horizontal component x: red (+, pointing right) / cyan (-, left).
    Vertical component y:   blue (+, pointing down) / yellow (-, up).
    `valid` (bool [H, W]) greys out pixels without a meaningful direction.
    """
    vec = np.clip(np.asarray(vec, dtype=np.float64), -1, 1)
    rgb = np.empty(vec.shape[:2] + (3,))
    rgb[..., 0] = 0.5 + 0.5 * vec[..., 1]
    rgb[..., 1] = 0.5
    rgb[..., 2] = 0.5 + 0.5 * vec[..., 0]
    if valid is not None:
        rgb[~np.asarray(valid, dtype=bool)] = 0.5
    return rgb


def rgb_to_direction(rgb):
    """
    Inverse of direction_to_rgb: [H, W, 3] -> [H, W, 2] (dir_y, dir_x).
    Grey (unset) pixels give zero. This lets the geo_dir_ layer carry the
    vector field, so no extra layer is needed.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    return np.stack([2 * rgb[..., 2] - 1, 2 * rgb[..., 0] - 1], axis=-1)


def vector_glyphs(geo, step=12, max_vectors=40000, min_distance=0.05):
    """
    Subsampled direction vectors for a napari Vectors layer.

    Returns [N, 2, 2] float32: for each glyph its (y, x) position and its
    (dy, dx) displacement (length about 0.8 of the grid step). Only pixels
    inside a predicted cell (edge distance > min_distance) with a clear
    direction are used. The grid is coarsened automatically so that at most
    `max_vectors` glyphs are made for very large images.
    """
    geo = np.asarray(geo)
    _, h, w = geo.shape
    step = max(int(step), int(np.ceil(np.sqrt(h * w / max_vectors))), 1)
    ys, xs = np.mgrid[step // 2 : h : step, step // 2 : w : step]
    ys, xs = ys.ravel(), xs.ravel()
    d = geo[0, ys, xs]
    vy, vx = geo[1, ys, xs], geo[2, ys, xs]
    norm = np.hypot(vy, vx)
    keep = (d > min_distance) & (norm > 0.3)
    ys, xs, vy, vx, norm = ys[keep], xs[keep], vy[keep], vx[keep], norm[keep]
    length = 0.8 * step
    out = np.empty((len(ys), 2, 2), dtype=np.float32)
    out[:, 0, 0], out[:, 0, 1] = ys, xs
    out[:, 1, 0], out[:, 1, 1] = vy / norm * length, vx / norm * length
    return out


def geometry_payloads(geo, source, vector_step=12, max_vectors=40000):
    """
    Napari layers for a stitched geometry array [4, H, W].

    Returns a list of dicts with keys kind ('image', 'rgb' or 'vectors'),
    name, data and kwargs (extra arguments for the add_* call). Layer names
    are geo_dist_<source>, geo_dir_<source>, geo_vec_<source> and
    geo_skel_<source>, so later steps (seeded watershed) can find them.
    """
    geo = clean_geometry(geo)
    valid = geo[0] > 0.05
    rgb = direction_to_rgb(np.moveaxis(geo[1:3], 0, -1), valid)
    return [
        {
            'kind': 'image',
            'name': f'{LAYER_PREFIXES["edge_distance"]}{source}',
            'data': geo[0],
            'kwargs': {'colormap': 'magma', 'contrast_limits': [0.0, 1.0]},
        },
        {
            'kind': 'image',
            'name': f'{LAYER_PREFIXES["skeleton"]}{source}',
            'data': geo[3],
            'kwargs': {
                'colormap': 'yellow',
                'blending': 'additive',
                'contrast_limits': [0.0, 1.0],
            },
        },
        {
            'kind': 'rgb',
            'name': f'{LAYER_PREFIXES["direction"]}{source}',
            'data': rgb.astype(np.float32),
            'kwargs': {'visible': False},
        },
        {
            'kind': 'vectors',
            'name': f'{LAYER_PREFIXES["vectors"]}{source}',
            'data': vector_glyphs(geo, vector_step, max_vectors),
            'kwargs': {
                'edge_color': 'cyan',
                'edge_width': 1.0,
                'length': 1.0,
                'visible': False,
            },
        },
    ]
