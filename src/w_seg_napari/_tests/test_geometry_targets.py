import numpy as np
import pytest

from w_seg_napari._model_config import (
    GEOMETRY_CHANNELS,
    has_geometry,
    make_config,
    validate_config,
)
from w_seg_napari._training.targets import compute_geometry, edge_mask


def _two_cells(size=40, pad=4):
    """Two rectangular cells separated by a one-pixel wall, plus background."""
    inst = np.zeros((size, size), np.int32)
    inst[4:36, 4:19] = 1
    inst[4:36, 21:36] = 2
    return inst


def test_config_geometry_switch():
    plain = make_config()
    assert not has_geometry(plain)
    assert plain['architecture'] == {'name': 'wasp_v3', 'geometry_head': False}
    geo = make_config(geometry=True)
    assert has_geometry(geo)
    assert geo['outputs']['geometry'] == 'geometry_head'
    assert geo['architecture']['name'] == 'wasp_v4'
    assert geo['geometry']['channels'] == list(GEOMETRY_CHANNELS)
    validate_config(geo)
    geo['geometry']['channels'] = ['a']
    with pytest.raises(ValueError):
        validate_config(geo)


def test_edge_mask_marks_walls_and_touching_cells():
    inst = np.zeros((4, 6), np.int32)
    inst[:, 0:3] = 1
    inst[:, 3:6] = 2  # touching, no wall pixel between
    edge = edge_mask(inst)
    assert edge[:, 2].all() and edge[:, 3].all()
    assert not edge[1, 0]


def test_geometry_targets_shapes_and_ranges():
    pad, crop = 4, 32
    out = compute_geometry(_two_cells(), crop, pad)
    assert out.shape == (crop, crop, 4) and out.dtype == np.float32
    dist, dy, dx, skel = (out[..., i] for i in range(4))
    assert dist.min() >= 0 and dist.max() <= 1.0
    assert np.isclose(dist.max(), 1.0)  # the deepest point of a cell
    # the wall column between the cells carries nothing
    wall_x = 19 - pad
    assert (dist[:, wall_x] == 0).all() and (skel[:, wall_x] == 0).all()
    # direction vectors are unit length or zero
    length = np.hypot(dy, dx)
    assert np.all((np.isclose(length, 1.0, atol=1e-4)) | (length == 0))
    # skeleton lies inside cells, thin, and where the distance is high
    assert skel.sum() > 0
    assert (dist[skel > 0] > 0).all()
    assert dist[skel > 0].mean() > 0.5


def test_direction_points_away_from_the_nearest_edge():
    pad, crop = 4, 32
    inst = _two_cells()
    out = compute_geometry(inst, crop, pad)
    # a pixel near the left side of cell 1 (padded x = 8): nearest edge is to
    # its left, so the vector points right (dx > 0, dy ~ 0)
    y, x = 20 - pad, 8 - pad
    assert out[y, x, 2] > 0.9 and abs(out[y, x, 1]) < 0.2
    # near the right side of cell 1 (padded x = 15): points left
    x = 15 - pad
    assert out[y, x, 2] < -0.9


def test_cells_without_interior_do_not_crash():
    inst = np.zeros((24, 24), np.int32)
    out = compute_geometry(inst, 16, 4)
    assert out.shape == (16, 16, 4) and out.sum() == 0
