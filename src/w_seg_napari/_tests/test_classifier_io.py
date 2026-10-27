"""Saving and loading the cell classifier (CellClassifierManager)."""

import numpy as np
import pandas as pd

from w_seg_napari._cell_classification_widget import CellClassifierManager

FE_CONFIG = {
    'pixel_spacing': 0.5,
    'max_search_radius': 30,
    'radial_sym': False,
}


def _trained_manager(n=60):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] > 0).astype(int) + 1
    m = CellClassifierManager()
    m.feature_names = ['a', 'b', 'c', 'd']
    m.train(X, y, feature_config=FE_CONFIG)
    return m, X


def test_train_save_load_roundtrip(tmp_path):
    m, X = _trained_manager()
    path = tmp_path / 'clf'
    m.save_model(str(path), feature_config=FE_CONFIG)

    loaded = CellClassifierManager()
    cfg = loaded.load_model(str(path) + '.joblib')

    assert loaded.is_trained
    assert loaded.feature_names == ['a', 'b', 'c', 'd']
    assert cfg['feature_extraction_config'] == FE_CONFIG
    assert cfg['classifier_type'] == m.classifier_type
    assert np.array_equal(loaded.pipeline.predict(X), m.pipeline.predict(X))


def test_all_three_stages_use_the_same_config_layout(tmp_path):
    m, _ = _trained_manager()
    after_train = m.model_config
    m.save_model(str(tmp_path / 'clf'), feature_config=FE_CONFIG)
    after_save = m.model_config
    loaded = CellClassifierManager()
    after_load = loaded.load_model(str(tmp_path / 'clf.joblib'))
    for cfg in (after_train, after_save, after_load):
        assert set(cfg) == {
            'schema_version',
            'classifier_type',
            'hyperparameters',
            'feature_names',
            'feature_extraction_config',
        }
        assert cfg['feature_extraction_config'] == FE_CONFIG


def test_save_without_explicit_config_uses_the_training_config(tmp_path):
    m, _ = _trained_manager()
    m.save_model(str(tmp_path / 'clf'))  # used to raise AttributeError
    loaded = CellClassifierManager()
    cfg = loaded.load_model(str(tmp_path / 'clf.joblib'))
    assert cfg['feature_extraction_config'] == FE_CONFIG


def test_old_files_with_classifier_type_in_feature_config(tmp_path):
    import joblib

    m, _ = _trained_manager()
    old = {
        'pipeline': m.pipeline,
        'classifier_type': m.classifier_type,
        'hyperparameters': m.hyperparameters,
        'feature_names': m.feature_names,
        'is_trained': True,
        'feature_config': {**FE_CONFIG, 'classifier_type': 'Random Forest'},
    }
    joblib.dump(old, tmp_path / 'old.joblib')
    cfg = CellClassifierManager().load_model(str(tmp_path / 'old.joblib'))
    assert cfg['feature_extraction_config'] == FE_CONFIG


def test_wrong_file_is_refused(tmp_path):
    import joblib
    import pytest

    joblib.dump({'something': 1}, tmp_path / 'x.joblib')
    with pytest.raises(ValueError):
        CellClassifierManager().load_model(str(tmp_path / 'x.joblib'))


def test_predict_after_load_adds_prediction_column(tmp_path):
    m, X = _trained_manager()
    m.save_model(str(tmp_path / 'clf'), feature_config=FE_CONFIG)
    loaded = CellClassifierManager()
    loaded.load_model(str(tmp_path / 'clf.joblib'))
    df = pd.DataFrame(X, columns=['a', 'b', 'c', 'd'])
    df['label_id'] = np.arange(1, len(df) + 1)
    out, ids, preds = loaded.predict(df)
    assert 'predicted_class' in out and len(ids) == len(preds) == len(df)
