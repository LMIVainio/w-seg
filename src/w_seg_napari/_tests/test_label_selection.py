import numpy as np

import w_seg_napari._label_selection as ls


def _scene():
    lab = np.zeros((40, 60), np.int32)
    lab[2:20, 2:12] = 1  # root: three touching cells (one-pixel wall between)
    lab[2:20, 13:22] = 2
    lab[2:20, 23:32] = 3
    lab[30:33, 50:53] = 4  # isolated tiny cell
    lab[30:38, 36:40] = 5  # isolated pair of cells
    lab[30:38, 41:45] = 6
    return lab


def test_size_limits():
    lab = _scene()
    ids, areas = ls.label_areas(lab)
    assert dict(zip(ids, areas, strict=True))[4] == 9
    assert list(ls.size_ids(lab, lower=20)) == [4]
    # areas: 1=180, 2=162, 3=162, 4=9, 5=32, 6=32
    assert set(ls.size_ids(lab, upper=170)) == {1}
    assert set(ls.size_ids(lab, lower=40, upper=170)) == {1, 4, 5, 6}
    assert len(ls.size_ids(lab)) == 0  # both limits off


def test_groups_join_cells_across_a_wall():
    lab = _scene()
    ids, group_of, n_cells, area = ls.cell_groups(lab, gap=2)
    g = dict(zip(ids, group_of, strict=True))
    assert g[1] == g[2] == g[3]
    assert g[5] == g[6] != g[1]
    assert len({g[1], g[4], g[5]}) == 3
    # without a gap the wall separates the cells
    _, apart, _, _ = ls.cell_groups(lab, gap=0)
    assert len(set(apart)) == 6


def test_isolated_single_cells_and_small_groups():
    lab = _scene()
    assert list(ls.isolated_ids(lab, 2, max_cells=1)[0]) == [4]
    assert set(ls.isolated_ids(lab, 2, max_cells=2)[0]) == {4, 5, 6}
    none, _ = ls.isolated_ids(lab, 2, max_cells=0)
    assert len(none) == 0


def test_everything_outside_the_largest_group():
    lab = _scene()
    ids, info = ls.isolated_ids(lab, 2, all_but_largest=True)
    assert set(ids) == {4, 5, 6}
    assert info == {'groups': 3, 'kept': 3}


def test_empty_labels():
    ids, info = ls.isolated_ids(np.zeros((5, 5), np.int32), 2, 1, True)
    assert len(ids) == 0 and info['groups'] == 0
