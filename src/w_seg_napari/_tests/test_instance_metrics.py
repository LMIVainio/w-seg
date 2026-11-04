import numpy as np
import pytest

from w_seg_napari._instance_metrics import evaluate, pooled, summary_row


def _grid():
    """Four 10x10 reference cells in a row."""
    ref = np.zeros((10, 40), np.int32)
    for i in range(4):
        ref[:, i * 10 : (i + 1) * 10] = i + 1
    return ref


def test_identical_labels_score_one():
    ref = _grid()
    s, cells = evaluate(ref * 7, ref)  # ids differ, shapes agree
    assert s[0.5]['f1'] == 1.0 and s[0.9]['f1'] == 1.0
    assert set(cells['status']) == {'matched'}


def test_merge_is_one_hit_lost_and_classified():
    ref = _grid()
    pred = ref.copy()
    pred[pred == 2] = 1  # cells 1 and 2 merged
    s, cells = evaluate(pred, ref)
    # the merged cell has IoU 0.5 with each: one of them may be matched at 0.5
    assert s[0.5]['tp'] == 3 or s[0.5]['tp'] == 2
    assert s[0.75]['tp'] == 2
    assert (cells['status'] == 'merged').sum() >= 1


def test_split_cell():
    ref = _grid()
    pred = ref.copy()
    pred[:, 13:17] = 9  # cell 2 cut into three pieces of 30 / 40 / 30 px
    pred[:, 17:20] = 10
    s, cells = evaluate(pred, ref)
    status = dict(zip(cells['ref_id'], cells['status']))
    assert status[2] == 'split'
    assert s[0.5]['fn'] == 1 and s[0.5]['fp'] == 3


def test_missed_and_extra_cells():
    ref = _grid()
    pred = ref.copy()
    pred[pred == 4] = 0  # cell 4 not found
    pred[0:3, 0:3] = 8  # spurious little cell inside cell 1 area
    s, cells = evaluate(pred, ref)
    status = dict(zip(cells['ref_id'], cells['status']))
    assert status[4] == 'missed'
    assert s[0.5]['fp'] == 1 and s[0.5]['fn'] == 1


def test_poor_outline():
    ref = np.zeros((20, 20), np.int32)
    ref[:, :10] = 1
    pred = np.zeros_like(ref)
    pred[:, :7] = 1  # IoU 0.7
    s, cells = evaluate(pred, ref)
    assert s[0.5]['tp'] == 1 and s[0.75]['tp'] == 0
    s2, c2 = evaluate(pred, ref, thresholds=(0.9,))
    assert c2['status'][0] == 'outline'


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError):
        evaluate(np.zeros((3, 3)), np.zeros((4, 4)))


def test_pooled_uses_summed_counts():
    ref = _grid()
    rows = []
    for pred in (ref, np.zeros_like(ref)):
        s, c = evaluate(pred, ref)
        rows.append(summary_row('x', s, c))
    p = pooled(rows)
    assert p['tp@0.5'] == 4 and p['fn@0.5'] == 4
    assert p['F1@0.5'] == pytest.approx(2 * 4 / (2 * 4 + 0 + 4), abs=1e-4)
