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


def _border_scene():
    lab = np.zeros((40, 80), np.int32)
    lab[0:10, 0:10] = 1  # 1 and 2 share a 10 px border
    lab[0:10, 10:20] = 2
    lab[0:4, 20:24] = 3  # 3 touches 2 along 4 px
    lab[0:10, 30:40] = 5  # 5 and 6 are separated by a one-pixel wall
    lab[0:10, 41:51] = 6
    lab[20:40, 0:20] = 8  # 8 has a 3x3 cell (7) inside
    lab[25:28, 5:8] = 7
    return lab


def test_border_table_counts_pairs_and_fractions():
    from w_seg_napari._curation_ops import border_table

    lab = _border_scene()
    table = {(r['a'], r['b']): r for r in border_table(lab, gap=1)}
    assert table[(1, 2)]['border'] == 10
    assert table[(2, 3)]['border'] == 4
    assert (5, 6) not in table  # the wall keeps them apart at gap=1
    # a cell completely inside another: border covers its whole perimeter
    enclosed = table[(7, 8)]
    assert enclosed['border'] == 12 and enclosed['fraction'] == 1.0
    assert enclosed['area_a'] == 9 and enclosed['perimeter_a'] == 8
    # 1 and 2 are squares: 10 of the 36 boundary pixels of the smaller cell
    assert 0 < table[(1, 2)]['fraction'] < 1
    # the wall is crossed from gap=2 on
    table2 = {(r['a'], r['b']): r for r in border_table(lab, gap=2)}
    assert table2[(5, 6)]['border'] == 10
    # sorted by decreasing border length; empty image has no pairs
    lengths = [r['border'] for r in border_table(lab, 2)]
    assert lengths == sorted(lengths, reverse=True)
    assert border_table(np.zeros((5, 5), np.int32)) == []


def test_groups_by_border_conditions():
    from w_seg_napari._curation_ops import groups_by_border

    lab = _border_scene()
    groups, pairs = groups_by_border(lab, min_border=8)
    assert sorted(map(sorted, groups)) == [[1, 2], [7, 8]]
    assert len(pairs) == 2
    # a lower limit chains 3 on as well: 1-2-3 form one group
    groups, _ = groups_by_border(lab, min_border=4)
    assert [1, 2, 3] in groups
    # the wall pair only qualifies with tolerance
    groups, _ = groups_by_border(lab, min_border=8, gap=2)
    assert [5, 6] in groups
    # relative border: only the enclosed cell is completely surrounded
    groups, _ = groups_by_border(lab, min_border=1, min_fraction=0.9)
    assert groups == [[7, 8]]
    # only absorb small fragments
    groups, _ = groups_by_border(lab, min_border=1, max_small_area=20)
    assert groups == [[2, 3], [7, 8]] or sorted(map(sorted, groups)) == [
        [2, 3],
        [7, 8],
    ]
    # nothing qualifies
    assert groups_by_border(lab, min_border=1000) == ([], [])


def test_merge_groups_by_border_end_to_end():
    from w_seg_napari._curation_ops import groups_by_border, merge_label_groups

    lab = _border_scene()
    groups, _ = groups_by_border(lab, min_border=8, gap=2)
    merge_label_groups(lab, groups, 'Lowest ID', fill_holes=True, close_gap=1)
    assert set(np.unique(lab)) == {0, 1, 3, 5, 7}
    assert (lab[0:10, 0:20] == 1).all()
    # the wall between 5 and 6 was filled, nothing else was touched
    assert (lab[0:10, 30:51] == 5).all()
    assert (lab[10:20, 30:51] == 0).all() and (lab[0:4, 20:24] == 3).all()
    # without close_gap the wall stays
    lab2 = _border_scene()
    merge_label_groups(lab2, [[5, 6]], 'Lowest ID', close_gap=0)
    assert (lab2[0:10, 40] == 0).all()
