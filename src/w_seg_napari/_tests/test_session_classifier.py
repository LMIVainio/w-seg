"""Training the cell classifier from batch sessions."""

import numpy as np
import pandas as pd
import pytest

import w_seg_napari._session_classifier as sc


def test_majority_class_and_coverage():
    labels = np.zeros((6, 12), np.int32)
    labels[:, :4], labels[:, 4:8], labels[:, 8:] = 1, 2, 3
    painted = np.zeros_like(labels)
    painted[:, :4] = 5  # all of cell 1
    painted[:2, 4:8] = 6  # a third of cell 2 ...
    painted[2:3, 4:8] = 7  # ... and a sixth of it in another class
    out = sc.majority_classes(labels, painted)
    assert out[1] == (5, 1.0)
    assert out[2][0] == 6 and out[2][1] == pytest.approx(0.5)
    assert 3 not in out  # nothing painted
    assert 2 not in sc.majority_classes(labels, painted, min_coverage=0.6)
    with pytest.raises(ValueError):
        sc.majority_classes(labels, painted[:3])


def _table(slide, n=60, seed=0):
    rng = np.random.default_rng(seed)
    cls = rng.integers(1, 3, n)
    return pd.DataFrame(
        {
            'label_id': np.arange(1, n + 1),
            'centroid-0': 1.0,
            'area': np.where(cls == 1, 30.0, 90.0) + rng.normal(0, 3, n),
            'prob_xylem': np.where(cls == 1, 0.9, 0.2)
            + rng.normal(0, 0.05, n),
            'slide': slide,
            'classifier_label': np.where(np.arange(n) < n - 5, cls, 0),
            'predicted_class': 0,
            'labels_source': 'corrected',
        }
    )


def test_feature_columns_leave_out_ids_and_labels():
    cols = sc.feature_columns(_table('a'))
    assert cols == ['area', 'prob_xylem']


@pytest.mark.parametrize('name', ['Random Forest', 'SVM', 'Gradient Boosting'])
def test_training_gives_a_loadable_payload_and_a_held_out_score(name):
    tables = [_table(f's{i}', seed=i) for i in range(4)]
    payload, r = sc.train_from_tables(
        tables, name, {'n_estimators': 30}, {'pixel_spacing': 0.5}
    )
    assert r['n_slides'] == 4 and r['yolo_columns'] == 1
    assert r['held_out']['accuracy'] > 0.9
    assert payload['feature_names'] == ['area', 'prob_xylem']
    assert payload['feature_extraction_config'] == {'pixel_spacing': 0.5}
    # the tab's keys, and a pipeline that predicts the painted class ids
    assert {'pipeline', 'is_trained', 'schema_version'} <= set(payload)
    X = tables[0][payload['feature_names']]
    assert set(payload['pipeline'].predict(X)) <= {1, 2}


def test_one_class_or_xgboost_is_refused():
    t = _table('a')
    t['classifier_label'] = np.where(t['classifier_label'] > 0, 1, 0)
    with pytest.raises(ValueError, match='Only one class'):
        sc.train_from_tables([t])
    with pytest.raises(ValueError, match='XGBoost'):
        sc.train_from_tables([_table('a')], 'XGBoost')


def test_session_table_adds_classifier_label(tmp_path):
    pytest.importorskip('h5py')
    import w_seg_napari._session_io as sio
    from w_seg_napari._tests.test_session_io import FakeLayer, _batch_arrays

    path = tmp_path / 'a_session.h5'
    arrays = _batch_arrays()
    labels = np.zeros((12, 16), np.uint64)
    labels[:, :8], labels[:, 8:] = 1, 2
    arrays['gasp_labels'] = labels
    sio.save_session(path, arrays, {'source_file': 'a.tif'})
    painted = np.zeros((12, 16), np.int32)
    painted[:, :8] = 4
    sio.save_edits(
        path,
        extras=[
            sio.layer_to_payload(
                FakeLayer('labels', 'classifier_labels', painted)
            )
        ],
    )
    assert 'morphology' in sc.session_table(path)[1]  # no csv yet
    pd.DataFrame(
        {'label_id': [1, 2], 'labels_source': 'gasp', 'area': [1.0, 2.0]}
    ).to_csv(tmp_path / 'a_morphology.csv', index=False)
    table, why = sc.session_table(path)
    assert why is None
    assert list(table['classifier_label']) == [4, 0]
    assert (
        'not the corrected'
        in sc.session_table(path, require_corrected=True)[1]
    )
