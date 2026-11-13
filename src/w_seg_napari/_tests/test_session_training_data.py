import numpy as np
import pytest

from w_seg_napari import _session_training_data as std


def _case():
    pred = np.zeros((6, 8), np.float32)
    pred[:, 3] = 0.9  # a wall between the cells
    pred[1:3, 5:7] = 0.8  # excess wall inside a cell
    lab = np.zeros((6, 8), np.int32)
    lab[:, :3] = 1
    lab[:, 4:] = 2  # the column 3 stays 0 = the wall
    return pred, lab


def test_wall_inside_cells_is_removed_and_wall_between_stays():
    pred, lab = _case()
    wall = std.session_wall(pred, lab, 0.5)
    assert wall.dtype == np.uint8
    assert wall[:, 3].all()
    assert not wall[1:3, 5:7].any()
    assert wall.sum() == 6


def test_auto_threshold_and_flat_prediction():
    pred, lab = _case()
    assert std.session_wall(pred, lab, std.AUTO).sum() == 6
    flat = np.full((4, 4), 0.2, np.float32)
    assert std.auto_threshold(flat) == 0.5


def test_stack_layout_uses_the_corrected_labels():
    pred, lab = _case()
    raw = np.random.randint(0, 255, (6, 8, 3), dtype=np.uint8)
    st = std.session_training_stack(raw, pred, lab, 0.5)
    assert st.shape == (5, 6, 8)
    assert np.array_equal(st[4], lab)
    assert np.array_equal(st[3], std.session_wall(pred, lab, 0.5))


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError):
        std.session_wall(np.zeros((4, 4)), np.zeros((4, 5)))


def test_split_is_by_image_and_repeatable():
    names = [f'img{i}' for i in range(10)]
    a = std.split_names(names, 0.3)
    assert a == std.split_names(names, 0.3)
    assert len(a[1]) == 3 and set(a[0]) | set(a[1]) == set(names)
    assert std.split_names(names, 0) == (sorted(names), [])
    assert len(std.split_names(['x', 'y'], 0.9)[1]) == 1
