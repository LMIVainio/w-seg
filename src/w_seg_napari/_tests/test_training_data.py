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


def test_mask_labels_cuts_walls_out():
    img = np.zeros((4, 4, 3), np.uint8)
    wall = np.zeros((4, 4), np.float32)
    wall[1] = 1.0
    lab = np.full((4, 4), 5, np.int32)
    lab[0] = 1  # background label stays
    st = build_training_stack(img, wall, lab, mask_labels=True)
    assert (st[4, 1] == 0).all() and (st[4, 0] == 1).all()
    assert (st[4, 2:] == 5).all()
    plain = build_training_stack(img, wall, lab)
    assert (plain[4, 1] == 5).all()


def test_split_shared_label_gives_each_cell_its_own_id(tmp_path):
    img = np.zeros((6, 9, 3), np.uint8)
    lab = np.zeros((6, 9), np.int32)
    lab[:, 0:3] = 1  # three separate regions share label 1
    lab[:, 4:6] = 1
    lab[:, 7:9] = 1
    lab[0:2, 3] = 9
    st = build_training_stack(img, np.zeros((6, 9)), lab, split_label=1)
    ids = set(np.unique(st[4])) - {0, 9}
    assert len(ids) == 3 and 1 not in ids
    assert (st[4][lab == 9] == 9).all()
    # off by default
    assert set(
        np.unique(build_training_stack(img, np.zeros((6, 9)), lab)[4])
    ) == {0, 1, 9}


def test_saved_stack_is_memory_mappable(tmp_path):
    import tifffile

    st = build_training_stack(
        np.zeros((4, 4, 3), np.uint8), np.ones((4, 4)), np.ones((4, 4), int)
    )
    save_training_stack(tmp_path / 'm.tif', st)
    assert tifffile.memmap(tmp_path / 'm.tif').shape == (5, 4, 4)
