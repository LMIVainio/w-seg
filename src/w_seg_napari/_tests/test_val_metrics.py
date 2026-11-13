import numpy as np

from w_seg_napari._training import metrics as m


def _walls():
    wall = np.zeros((40, 40), bool)
    wall[:, 19:21] = True  # a 2 px wall between two cells
    return wall


def test_perfect_prediction_scores_one():
    wall = _walls()
    v = m.ValidationMetrics()
    v.add(wall.astype(np.float32), wall)
    r = v.result()
    assert r['dice_best'] == 1 and r['precision'] == 1 and r['recall'] == 1
    assert r['tol_f1'] == 1 and r['cldice'] == 1
    assert r['dice_img'] == 1 and r['dice_img_sd'] == 0


def test_shifted_wall_is_forgiven_by_tolerance_not_by_dice():
    wall = _walls()
    pred = np.roll(wall, 1, axis=1).astype(np.float32)
    v = m.ValidationMetrics(tol=1)
    v.add(pred, wall)
    r = v.result()
    assert r['dice_best'] < 0.6
    assert r['tol_f1'] == 1


def test_gap_in_a_wall_hurts_cells_but_hardly_dice():
    # a closed ring: one cell inside, background outside
    wall = np.zeros((40, 40), bool)
    wall[10:30, 10:30] = True
    wall[12:28, 12:28] = False  # a ring, the cell is the inside
    cut = wall.copy()
    cut[10:12, 18:22] = False  # the ring is broken
    v = m.ValidationMetrics(min_cell=4)
    v.add(wall.astype(np.float32), wall)
    ok = v.result()
    v2 = m.ValidationMetrics(min_cell=4)
    v2.add(cut.astype(np.float32), wall)
    broken = v2.result()
    assert ok['cell_f1'] == 1
    assert broken['cell_f1'] < 1
    assert broken['dice_best'] > 0.95


def test_best_threshold_and_recall_bias():
    wall = _walls()
    pred = np.where(wall, 0.4, 0.0).astype(np.float32)  # under-confident
    v = m.ValidationMetrics()
    v.add(pred, wall)
    r = v.result()
    assert r['recall'] == 0
    assert r['dice_best'] == 1 and r['thr_best'] <= 0.35


def test_selection_score_smooths_last_evaluations():
    h = {'val_cell_f1': [0.0, 0.3, 0.6, 0.9]}
    assert abs(m.selection_score(h, last=3) - 0.6) < 1e-9
    assert m.selection_score({'val_cell_f1': []}) == 0.0


def test_skeleton_on_wall_rate_and_missing_geometry():
    wall = _walls()
    skel = np.zeros(wall.shape, np.float32)
    skel[:, 5] = 1.0  # inside a cell
    skel[:10, 19] = 1.0  # on the wall
    v = m.ValidationMetrics()
    v.add(wall.astype(np.float32), wall, skel)
    assert abs(v.result()['skel_on_wall'] - 10 / 50) < 1e-9
    none = m.ValidationMetrics()
    none.add(wall.astype(np.float32), wall)
    assert np.isnan(none.result()['skel_on_wall'])


def test_centreline_boost_marks_the_middle_of_a_wall_only():
    from w_seg_napari._training.targets import centreline_boost

    wall = np.zeros((20, 20), bool)
    wall[:, 8:11] = True  # 3 px wide
    w = centreline_boost(wall, 2.0)
    assert w.dtype == np.float32
    assert w[10, 9] == 3.0 and w[10, 0] == 1.0
    assert np.all(centreline_boost(wall, 0) == 1)
    assert np.all(centreline_boost(np.zeros((5, 5), bool), 2.0) == 1)
