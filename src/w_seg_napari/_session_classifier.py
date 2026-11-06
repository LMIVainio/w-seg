"""Train the Cell Morphology classifier from batch sessions.

No Qt or napari imports, so scripts/train_cell_classifier.py can use it.

A session supplies the cells and the classes you painted for them:

    <image>_session.h5          the cell labels (and the painted layer)
    <image>_morphology.csv      the batch's measurements, YOLO probabilities
                                (prob_*) included
    extra layer `classifier_labels`   class id painted on the cells

Every cell of the csv gets a `classifier_label` (the class that covers most
of the cell's painted pixels, 0 = not painted) and the cells that have one
train the classifier. The file written is the one the Cell Morphology tab
loads ("Load Model").
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from w_seg_napari._classifiers import SUPPORTED_CLASSIFIERS, build_estimator

LAYER_NAME = 'classifier_labels'
SCHEMA_VERSION = 2  # as CellClassifierManager.save_model

# columns of the morphology table that are not measurements to learn from
# (the same list as the Cell Morphology tab) plus the ones added here
NOT_FEATURES = {
    'label_id',
    'centroid-0',
    'centroid-1',
    'label',
    'bbox-0',
    'bbox-1',
    'bbox-2',
    'bbox-3',
    'predicted_class',
    'manual_class',
    'yolo_pred_tissue',
    'classifier_label',
    'class_fraction',
    'slide',
}

LABEL_FILES = {
    'corrected': 'labels_corrected',
    'gasp': 'gasp_labels',
    'superpixels': 'sp_labels',
}


def majority_classes(labels, painted, min_coverage=0.0):
    """
    {label id: (class id, painted share of the cell)} for the cells that have
    painted pixels. The class is the one covering most of the cell; ties go to
    the smaller id. Cells whose painted share is below min_coverage are left
    out.
    """
    labels = np.asarray(labels)
    painted = np.asarray(painted)
    if labels.shape != painted.shape:
        raise ValueError(
            f'the {LAYER_NAME} layer {painted.shape} and the cell labels '
            f'{labels.shape} differ in shape'
        )
    lab, cls = (
        labels.ravel().astype(np.int64),
        painted.ravel().astype(np.int64),
    )
    m = (lab > 0) & (cls > 0)
    if not m.any():
        return {}
    width = int(cls.max()) + 1
    keys, counts = np.unique(lab[m] * width + cls[m], return_counts=True)
    cells, classes = keys // width, keys % width
    area = np.bincount(lab[lab > 0], minlength=int(lab.max()) + 1)
    best, total = {}, {}
    for cell, k, n in zip(cells, classes, counts, strict=True):
        total[cell] = total.get(cell, 0) + n
        if cell not in best or n > best[cell][0]:
            best[cell] = (n, k)
    out = {}
    for cell, (_n, k) in best.items():
        share = total[cell] / max(float(area[cell]), 1.0)
        if share >= min_coverage:
            out[int(cell)] = (int(k), float(share))
    return out


def session_table(
    path, layer_name=LAYER_NAME, require_corrected=False, min_coverage=0.0
):
    """
    The batch's morphology table of one session with `classifier_label`
    (0 = not painted) and `class_fraction` added, plus `slide`.
    Returns (table, None), or (None, reason) if the session cannot be used.
    """
    import h5py

    from w_seg_napari._session_io import SESSION_SUFFIX, read_extras

    path = Path(path)
    slide = path.name.removesuffix(SESSION_SUFFIX)
    csv = path.with_name(f'{slide}_morphology.csv')
    if not csv.exists():
        return None, f'no {csv.name} next to it (run the morphology stage)'
    try:
        layers = {p['name']: p for p in read_extras(path)}
        table = pd.read_csv(csv)
        source = (
            table['labels_source'].iloc[0]
            if 'labels_source' in table.columns and len(table)
            else 'gasp'
        )
        key = LABEL_FILES.get(source, 'gasp_labels')
        with h5py.File(path, 'r') as f:
            labels = np.array(f[key]) if key in f else None
    except (OSError, KeyError, ValueError) as e:
        return None, f'cannot read it ({e})'
    if layer_name not in layers:
        return None, f'no "{layer_name}" layer saved in it'
    if 'label_id' not in table.columns:
        return None, f'{csv.name} has no label_id column'
    if labels is None:
        return None, f'it has no {key} that {csv.name} was measured on'
    if require_corrected and source != 'corrected':
        return None, (
            f'its table was measured on the {source} labels, not the '
            'corrected ones'
        )
    painted = np.squeeze(np.asarray(layers[layer_name]['data']))
    try:
        found = majority_classes(labels, painted, min_coverage)
    except ValueError as e:
        return None, str(e)
    table = table.copy()
    table['slide'] = slide
    table['classifier_label'] = [
        found.get(int(i), (0, 0.0))[0] for i in table['label_id']
    ]
    table['class_fraction'] = [
        found.get(int(i), (0, 0.0))[1] for i in table['label_id']
    ]
    if not (table['classifier_label'] > 0).any():
        return None, 'none of its measured cells has a painted class'
    table.attrs['labels_source'] = source
    return table, None


def feature_columns(table):
    """Numeric measurement columns, in table order."""
    return [
        c
        for c in table.select_dtypes('number').columns
        if c not in NOT_FEATURES and not c.startswith('class_')
    ]


def _clean(values):
    return (
        values.replace([np.inf, -np.inf], np.nan).clip(-1e9, 1e9).astype(float)
    )


def _pipeline(classifier, hyperparameters):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', build_estimator(classifier, hyperparameters)),
        ]
    )


def train_from_tables(
    tables,
    classifier='Random Forest',
    hyperparameters=None,
    feature_config=None,
    n_splits=5,
):
    """
    Fit the classifier on all painted cells of the tables. Returns
    (payload, report). The payload is what the Cell Morphology tab's
    "Load Model" reads. The report holds the scores on slides the model has
    not seen (whole slides held out) when there are two or more.
    """
    import sklearn
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_fscore_support,
    )
    from sklearn.model_selection import GroupKFold

    if classifier == 'XGBoost':
        raise ValueError(
            'XGBoost needs the classes numbered 0, 1, 2...; use another '
            'classifier here.'
        )
    hp = dict(hyperparameters or {'n_estimators': 100, 'max_depth': 15})
    # columns every slide has, so a slide measured without YOLO cannot
    # silently turn a feature into mostly-missing values
    big = pd.concat(tables, ignore_index=True)
    features = [c for c in feature_columns(tables[0]) if c in big.columns]
    labelled = big[big['classifier_label'] > 0].reset_index(drop=True)
    if labelled.empty:
        raise ValueError('No painted cells found.')
    X = _clean(labelled[features])
    y = labelled['classifier_label'].to_numpy()
    groups = labelled['slide'].to_numpy()
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(
            f'Only one class ({classes[0]}) was painted; at least two are '
            'needed.'
        )

    report = {
        'n_cells': int(len(labelled)),
        'n_slides': int(len(np.unique(groups))),
        'classes': {int(c): int((y == c).sum()) for c in classes},
        'n_features': len(features),
        'yolo_columns': sum(c.startswith('prob_') for c in features),
        'classifier': classifier,
    }
    if report['n_slides'] >= 2:
        oof = np.full(len(y), -1)
        folds = GroupKFold(n_splits=min(n_splits, report['n_slides']))
        for tr, te in folds.split(X, y, groups):
            if len(np.unique(y[tr])) < 2:
                continue
            m = _pipeline(classifier, hp).fit(X.iloc[tr], y[tr])
            oof[te] = m.predict(X.iloc[te])
        seen = oof >= 0
        if seen.any():
            p, r, f, n = precision_recall_fscore_support(
                y[seen], oof[seen], labels=classes, zero_division=0
            )
            report['held_out'] = {
                'cells': int(seen.sum()),
                'accuracy': float(accuracy_score(y[seen], oof[seen])),
                'macro_f1': float(
                    f1_score(y[seen], oof[seen], average='macro')
                ),
                'per_class': {
                    int(c): {
                        'precision': float(p[i]),
                        'recall': float(r[i]),
                        'cells': int(n[i]),
                    }
                    for i, c in enumerate(classes)
                },
            }
    pipeline = _pipeline(classifier, hp).fit(X, y)
    payload = {
        'schema_version': SCHEMA_VERSION,
        'sklearn_version': sklearn.__version__,
        'pipeline': pipeline,
        'classifier_type': classifier,
        'hyperparameters': hp,
        'feature_names': features,
        'is_trained': True,
        'feature_extraction_config': {
            k: v
            for k, v in (feature_config or {}).items()
            if k != 'classifier_type'
        },
    }
    return payload, report


def morphology_config(session_paths):
    """The measurement settings of the batch (morph_params of the
    batch_settings.json in the first session's folder that has one)."""
    from w_seg_napari._batch_settings import FILENAME

    for p in session_paths:
        f = Path(p).with_name(FILENAME)
        if f.exists():
            try:
                return json.loads(f.read_text(encoding='utf-8')).get(
                    'morph_params', {}
                )
            except (OSError, ValueError):
                return {}
    return {}


__all__ = [
    'LAYER_NAME',
    'SUPPORTED_CLASSIFIERS',
    'feature_columns',
    'majority_classes',
    'morphology_config',
    'session_table',
    'train_from_tables',
]
