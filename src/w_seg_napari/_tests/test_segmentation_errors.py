import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

from w_seg_napari import _segmentation_errors as se


def _voronoi(rng, n=60, size=240):
    seeds = rng.uniform(10, size - 10, (n, 2))
    yy, xx = np.mgrid[:size, :size]
    d = (yy[..., None] - seeds[:, 0]) ** 2 + (xx[..., None] - seeds[:, 1]) ** 2
    return d.argmin(-1).astype(np.int32) + 1, seeds


def _wall_from(ref):
    edge = np.zeros(ref.shape, bool)
    edge[:-1] |= ref[:-1] != ref[1:]
    edge[:, :-1] |= ref[:, :-1] != ref[:, 1:]
    return np.clip(
        ndi.gaussian_filter(edge.astype(np.float32), 1.0) * 3.5, 0, 1
    )


def _slide(seed, n_merge=6, n_split=6):
    """reference labels, a faulty prediction of them, and the wall map."""
    rng = np.random.default_rng(seed)
    ref, seeds = _voronoi(rng)
    wall = _wall_from(ref)
    pred = ref.copy()
    # merge neighbouring pairs
    cells = list(rng.permutation(np.unique(ref))[: n_merge * 3])
    used = set()
    done = 0
    for a in cells:
        if a in used or done >= n_merge:
            continue
        dil = ndi.binary_dilation(ref == a, iterations=2)
        nbs = [
            b for b in np.unique(ref[dil]) if b not in (a, 0) and b not in used
        ]
        if nbs:
            pred[pred == nbs[0]] = a
            used |= {a, nbs[0]}
            done += 1
    # split cells with a line through the seed
    nxt = pred.max() + 1
    for a in rng.permutation(np.unique(ref))[:40]:
        if a in used or nxt > pred.max() + n_split:
            continue
        ys, xs = np.nonzero(ref == a)
        if len(ys) < 80:
            continue
        ang = rng.uniform(0, np.pi)
        side = (ys - ys.mean()) * np.cos(ang) + (xs - xs.mean()) * np.sin(
            ang
        ) > 0
        pred[ys[side], xs[side]] = nxt
        nxt += 1
        used.add(a)
    return ref, pred, wall


def test_labelling_finds_the_kinds_of_error():
    ref = np.zeros((20, 60), np.int32)
    ref[:, :20], ref[:, 20:40], ref[:, 40:] = 1, 2, 3
    pred = ref.copy()
    pred[pred == 2] = 1  # cells 1 and 2 merged -> 1 is "merged"
    pred[:10, 40:] = 7  # cell 3 cut in two halves -> both fragments
    pred[:10, 40:] = 7
    pred[:, 40:45] = 8  # a different piece
    s = se.label_predicted_cells(pred, ref).set_index('label_id')['status']
    assert s[1] == 'merged'
    assert s[7] == 'fragment' or s[7] == 'outline'
    pred2 = ref.copy()
    pred2[0:5, 0:5] = 9
    ref2 = ref.copy()
    ref2[0:5, 0:5] = 0
    s2 = se.label_predicted_cells(
        np.where(ref2 == 0, 9 * (pred2 == 9), pred2), ref2
    )
    assert s2.set_index('label_id')['status'][9] == 'spurious'


def test_identical_labels_are_all_ok():
    ref, _, _ = _slide(0)
    s = se.label_predicted_cells(ref, ref)
    assert set(s['status']) == {'ok'}


def test_features_cover_every_cell_and_are_named():
    ref, pred, wall = _slide(1)
    f = se.cell_features(pred, wall, pixel_size=0.5)
    assert list(f.columns) == ['label_id', *se.FEATURES]
    assert len(f) == len(np.unique(pred)) - (0 in pred)
    assert f['area_um2'].min() > 0


def test_merged_cells_have_a_wall_inside():
    ref, pred, wall = _slide(2)
    status = se.label_predicted_cells(pred, ref).set_index('label_id')[
        'status'
    ]
    f = se.cell_features(pred, wall).set_index('label_id')
    merged = f.loc[status[status == 'merged'].index, 'inner_wall_p95']
    ok = f.loc[status[status == 'ok'].index, 'inner_wall_p95']
    assert len(merged) >= 3
    assert merged.mean() > ok.mean() + 0.2


def test_fragments_have_a_weak_border_towards_their_neighbour():
    ref, pred, wall = _slide(3)
    status = se.label_predicted_cells(pred, ref).set_index('label_id')[
        'status'
    ]
    f = se.cell_features(pred, wall).set_index('label_id')
    frag = f.loc[status[status == 'fragment'].index, 'contact_wall_min']
    ok = f.loc[status[status == 'ok'].index, 'contact_wall_min']
    assert len(frag) >= 3
    assert frag.mean() < ok.mean() - 0.2


def _table(seeds):
    parts = []
    for s in seeds:
        ref, pred, wall = _slide(s)
        t = se.cell_features(pred, wall).merge(
            se.label_predicted_cells(pred, ref), on='label_id'
        )
        t['slide'] = f'slide{s}'
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def test_model_ranks_errors_on_unseen_slides_and_round_trips(tmp_path):
    table = _table(range(10, 16))
    bundle, report = se.train_error_model(table, n_splits=3)
    assert report['grouping'] == 'slides'
    assert report['average_precision'] > report['chance_level'] + 0.3
    assert report['recall_at_threshold'] >= 0.8

    path = tmp_path / 'm.joblib'
    se.save_error_model(bundle, path)
    again = se.load_error_model(path)
    ref, pred, wall = _slide(99)
    scores, emap = se.flag_labels(pred, wall, again)
    status = se.label_predicted_cells(pred, ref).set_index('label_id')[
        'status'
    ]
    p = scores.set_index('label_id')['p_error']
    bad = status.isin(se.BAD)
    assert p[bad[bad].index].mean() > p[status[status == 'ok'].index].mean()
    assert emap.shape == pred.shape and emap.max() <= 1.0
    assert set(scores['error_type']) <= {'ok', *se.BAD}


def test_too_little_data_is_a_clear_error():
    table = _table([20])
    table['status'] = 'ok'
    with pytest.raises(ValueError, match='at least 10 wrong'):
        se.train_error_model(table)


def test_foreign_file_is_refused(tmp_path):
    import joblib

    joblib.dump({'hello': 1}, tmp_path / 'x.joblib')
    with pytest.raises(ValueError):
        se.load_error_model(tmp_path / 'x.joblib')
