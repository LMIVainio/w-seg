import numpy as np

from w_seg_napari._curation_ops import (
    cut_cell_with_line,
    inherit_parent_labels,
    merge_label_groups,
    touching_groups,
)


def _grid():
    lab = np.zeros((10, 20), np.int32)
    lab[1:9, 1:6] = 1
    lab[1:9, 6:11] = 2  # touches 1
    lab[1:9, 14:19] = 3  # far from 2
    return lab


def test_touching_groups():
    lab = _grid()
    groups = touching_groups(lab, [1, 2, 3], gap=1)
    assert sorted(map(sorted, groups)) == [[1, 2], [3]]
    # a larger tolerance reaches across the 3 px gap
    groups = touching_groups(lab, [1, 2, 3], gap=4)
    assert sorted(map(sorted, groups)) == [[1, 2, 3]]
    assert touching_groups(lab, [], 1) == []
    assert touching_groups(lab, [2], 1) == [[2]]


def test_merge_groups_leaves_singles_alone():
    lab = _grid()
    groups = touching_groups(lab, [3, 2, 1], gap=1)
    targets = merge_label_groups(lab, groups, rule='Lowest ID')
    assert targets == [1]
    assert set(np.unique(lab)) == {0, 1, 3}
    lab = _grid()
    merge_label_groups(lab, [[3, 2]], rule='First Selected ID')
    assert (lab[1:9, 14:19] == 3).all() and (lab[1:9, 6:11] == 3).all()


def test_cut_cell_with_line():
    lab = np.zeros((12, 12), np.int32)
    lab[2:10, 2:10] = 5
    ok, msg, new = cut_cell_with_line(lab, (0, 6), (11, 6))
    assert ok and len(new) == 1
    assert set(np.unique(lab)) == {0, 5, new[0]}
    assert (lab > 0).sum() == 64  # nothing lost, no gap left
    # a line that does not cross the cell
    ok, msg, _ = cut_cell_with_line(lab, (0, 0), (0, 11))
    assert not ok
    # a line that stops inside the cell does not divide it
    lab2 = np.zeros((12, 12), np.int32)
    lab2[2:10, 2:10] = 5
    ok, _, _ = cut_cell_with_line(lab2, (4, 6), (7, 6))
    assert not ok and (lab2 == 5).sum() == 64


def test_inherit_parent_labels():
    child = np.zeros((6, 6), np.int32)
    child[0:3, 0:3] = 1  # inside parent 7
    child[3:6, 0:6] = 2  # overlaps parent 7 (2 px) and 9 (4 px)
    child[0:2, 4:6] = 3  # on parent background
    parent = np.zeros((6, 6), np.int32)
    parent[0:4, 0:4] = 7
    parent[3:6, 2:6] = 9
    out = inherit_parent_labels(child, parent)
    assert (out[0:3, 0:3] == 7).all()
    assert (out[3:6, 0:6] == 9).all()
    assert (out[0:2, 4:6] == 3).all()  # kept
    zero = inherit_parent_labels(child, parent, unmatched='zero')
    assert (zero[0:2, 4:6] == 0).all() and (zero[0:3, 0:3] == 7).all()
