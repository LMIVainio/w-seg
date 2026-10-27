import numpy as np
import pytest

from w_seg_napari._training_data import (
    build_training_stack,
    save_training_stack,
)


def test_rgb_stack_layout_and_dtype():
    img = np.random.randint(0, 255, (8, 6, 3), dtype=np.uint8)
    sem = np.zeros((8, 6), np.float32)
    sem[2:4] = 0.9
    lab = np.zeros((8, 6), np.int32)
    lab[:, :3] = 7
    st = build_training_stack(img, sem, lab)
    assert st.shape == (5, 8, 6)
    assert st.dtype == np.uint8
    assert np.array_equal(st[0], img[..., 0])
    assert st[3].sum() == 12 and st[3].max() == 1
    assert np.array_equal(st[4], lab)


def test_grey_is_repeated_and_large_labels_promote_dtype():
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    lab = np.zeros((3, 4), np.int64)
    lab[0, 0] = 70000
    st = build_training_stack(img, lab > 0, lab)
    assert st.dtype == np.uint32
    assert all(np.array_equal(st[c], img) for c in range(3))
    assert st[4, 0, 0] == 70000


def test_size_mismatch_and_roundtrip(tmp_path):
    img = np.zeros((4, 4, 3), np.uint8)
    with pytest.raises(ValueError):
        build_training_stack(img, np.zeros((5, 4)), np.zeros((4, 4), int))
    st = build_training_stack(img, np.ones((4, 4)), np.ones((4, 4), int))
    save_training_stack(tmp_path / 'a.tif', st)
    import tifffile

    back = tifffile.imread(tmp_path / 'a.tif')
    assert back.shape == (5, 4, 4) and np.array_equal(back, st)
