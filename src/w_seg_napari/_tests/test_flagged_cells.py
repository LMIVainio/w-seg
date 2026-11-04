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
