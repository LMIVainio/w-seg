import numpy as np
import pytest

from w_seg_napari import _flagged_cells as fc


def _scene():
    labels = np.zeros((10, 20), np.int32)
    labels[:, :5], labels[:, 5:10], labels[:, 10:15], labels[:5, 15:] = (
        1,
        2,
        3,
        4,
    )
    err = np.zeros(labels.shape, np.float32)
    for lab, p in {1: 0.1, 2: 0.9, 3: 0.6, 4: 0.97}.items():
        err[labels == lab] = p
    return labels, err


def test_scores_per_cell():
    labels, err = _scene()
    s = fc.cell_scores(labels, err).set_index('label_id')
    assert list(s.index) == [1, 2, 3, 4]
    assert s['p_error'][2] == pytest.approx(0.9)
    assert s['area'][1] == 50


def test_hand_merged_cell_is_judged_by_its_larger_part():
    labels, err = _scene()
    labels[labels == 2] = 1  # merged by hand: 50 px of p=0.1 + 50 px of p=0.9
    labels[:, 5:7] = 1
    labels[:, 7:10] = 2
    s = fc.cell_scores(labels, err).set_index('label_id')
    # cell 1 now: 50 px at 0.1 and 20 px at 0.9 -> mostly fine
    assert s['p_error'][1] == pytest.approx(0.1)


def test_range_limits_and_order():
    labels, err = _scene()
    s = fc.cell_scores(labels, err)
    both = fc.select_range(s, 0.5, 1.0)
    assert list(both['label_id']) == [4, 2, 3]  # worst first
    middle = fc.select_range(s, 0.5, 0.95)  # the very sure ones left out
    assert list(middle['label_id']) == [2, 3]
    assert 4 not in list(fc.select_range(s, 0.0, 1.0, min_area=51)['label_id'])


def test_extract_keeps_ids_and_dtype():
    labels, _ = _scene()
    out = fc.extract(labels, [2, 4])
    assert out.dtype == labels.dtype
    assert set(np.unique(out)) == {0, 2, 4}
    assert (out[labels == 2] == 2).all()


def test_find_cell_and_gone_cells():
    labels, _ = _scene()
    r, c = fc.find_cell(labels, 2)
    assert (r, c) == (4.5, 7.0)
    labels[labels == 2] = 0  # deleted
    assert fc.find_cell(labels, 2) is None


def test_next_existing_skips_cells_that_are_gone():
    labels, _ = _scene()
    labels[labels == 2] = 0
    queue = [4, 2, 3]
    assert fc.next_existing(queue, labels, 1, +1) == 2  # 2 is gone -> 3
    assert fc.next_existing(queue, labels, 1, -1) == 0
    assert fc.next_existing(queue, labels, 3, +1) is None


def test_pairing_by_name():
    names = ['error_a', 'error_b']
    assert fc.pair_error_layer('corrected_a', names) == 'error_a'
    assert fc.pair_error_layer('gasp_cells_b', names) == 'error_b'
    assert fc.pair_error_layer('x', names) is None
    assert fc.pair_error_layer('x', ['error_a']) == 'error_a'  # only one


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError):
        fc.cell_scores(np.zeros((3, 3), int), np.zeros((4, 4)))


def test_changed_ids_finds_edited_cells_only():
    labels, _ = _scene()
    areas = {2: 50, 3: 50, 4: 25}
    assert fc.changed_ids(labels, areas) == []  # untouched
    labels[labels == 2] = 1  # 2 merged into 1: 2 is gone
    assert fc.changed_ids(labels, areas) == [2]
    labels[:, 10:12] = 4  # 4 grows, 3 shrinks
    assert sorted(fc.changed_ids(labels, areas)) == [2, 3, 4]


def test_changed_ids_handles_ids_beyond_the_image():
    labels, _ = _scene()
    assert fc.changed_ids(labels, {999: 10}) == [999]  # gone
    assert fc.changed_ids(labels.astype(np.uint64), {1: 50}) == []


def _grid():
    lab = np.zeros((20, 30), np.int32)
    lab[2:18, 2:10] = 1
    lab[2:18, 11:20] = 2  # one-pixel wall (column 10) between 1 and 2
    lab[2:18, 20:28] = 3  # touches 2
    lab[8:12, 4:7] = 0  # hole inside 1
    return lab


def test_scores_carry_the_error_type_and_it_filters():
    lab = _grid()
    err = np.where(lab > 0, 0.9, 0).astype(np.float32)
    kinds = np.where(lab == 3, 2, np.where(lab == 2, 3, 0)).astype(np.int8)
    s = fc.cell_scores(lab, err, kinds)
    assert dict(zip(s.label_id, s.error_type, strict=True)) == {
        1: 'ok',
        2: 'spurious',
        3: 'fragment',
    }
    both = fc.select_range(s, 0.5, 1.0, kinds=('fragment', 'spurious'))
    assert sorted(both.label_id) == [2, 3]
    assert fc.cell_scores(lab, err)['error_type'].eq('unknown').all()
    assert list(fc.KIND_CHOICES.values())[0] is None


def test_partners_are_listed_by_border_length():
    lab = _grid()
    partners = dict(fc.merge_partners(lab, 2, gap=2))
    assert set(partners) == {1, 3}
    assert fc.merge_partners(lab, 99) == []


def test_merge_cells_keeps_the_id_and_closes_the_wall():
    lab = _grid()
    rows, cols, values = fc.merge_cells(lab, 1, [2], gap=2)
    out = lab.copy()
    out[rows, cols] = values
    assert not (out == 2).any() and (out == 1).any()
    assert (out[5, 10] == 1) and (out[9, 5] == 1)  # wall and hole filled
    assert (out == 3).sum() == (lab == 3).sum()  # third cell untouched
    assert (lab[rows, cols] != values).all()


def test_pair_layer_by_prefix():
    names = ['errortype_a', 'errortype_b']
    assert fc.pair_layer('corrected_b', names, 'errortype_') == 'errortype_b'
    assert fc.pair_layer('x', names, 'errortype_') is None
    assert fc.pair_layer('x', ['errortype_a'], 'errortype_') == 'errortype_a'


def test_merge_many_handles_cells_one_after_another():
    lab = _grid()
    # 1 | 2 | 3 side by side; 2 and 3 selected: 2 goes into its best
    # neighbour, 3 is then part of that cell (or absorbed) and skipped
    rows, cols, values, merges, alone = fc.merge_many(lab, [2, 3], gap=2)
    out = lab.copy()
    out[rows, cols] = values
    assert merges and not alone
    assert len(np.unique(out)) == len(np.unique(lab)) - 1  # one merge made
    assert out.shape == lab.shape


def test_merge_many_reports_cells_without_neighbours():
    lab = np.zeros((10, 10), np.int32)
    lab[1:3, 1:3] = 1
    lab[7:9, 7:9] = 2
    rows, cols, values, merges, alone = fc.merge_many(lab, [1, 2])
    assert alone == [1, 2] and merges == {} and rows.size == 0


def test_merge_many_all_touching():
    lab = _grid()
    rows, cols, values, merges, _ = fc.merge_many(lab, [2], True, gap=2)
    out = lab.copy()
    out[rows, cols] = values
    assert set(merges[2]) == {1, 3}
    assert set(np.unique(out)) == {0, 2}
