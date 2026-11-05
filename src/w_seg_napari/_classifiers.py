"""Classifier list shared by the Cell Morphology tab and the error model.

No Qt or napari imports, so scripts can use it.
"""

import numpy as np

try:
    from xgboost import XGBClassifier
except ImportError:  # optional install
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # optional install
    LGBMClassifier = None

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.svm import SVC

DEFAULT = 'Random Forest'


def _rf(hp):
    return RandomForestClassifier(
        n_estimators=hp.get('n_estimators', 100),
        max_depth=hp.get('max_depth', 15),
        class_weight='balanced_subsample' if hp.get('balanced') else None,
        random_state=42,
        n_jobs=-1,
    )


def _svm(hp):
    return SVC(
        C=hp.get('C', 1.0),
        kernel=hp.get('kernel', 'rbf'),
        probability=True,
        class_weight='balanced' if hp.get('balanced') else None,
        random_state=42,
    )


def _hgb(hp):
    return HistGradientBoostingClassifier(
        max_depth=hp.get('max_depth', 4),
        learning_rate=hp.get('learning_rate', 0.06),
        max_iter=hp.get('n_estimators', 200),
        class_weight='balanced' if hp.get('balanced') else None,
        random_state=hp.get('seed', 42),
    )


SUPPORTED_CLASSIFIERS = {
    'Random Forest': _rf,
    'SVM': _svm,
    'Gradient Boosting': _hgb,
}

if XGBClassifier is not None:
    SUPPORTED_CLASSIFIERS['XGBoost'] = lambda hp: XGBClassifier(
        n_estimators=hp.get('n_estimators', 100),
        max_depth=hp.get('max_depth', 6),
        learning_rate=hp.get('learning_rate', 0.1),
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
    )

if LGBMClassifier is not None:
    SUPPORTED_CLASSIFIERS['LightGBM'] = lambda hp: LGBMClassifier(
        n_estimators=hp.get('n_estimators', 100),
        max_depth=hp.get('max_depth', -1),
        class_weight='balanced' if hp.get('balanced') else None,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def build_estimator(name, hyperparameters=None):
    if name not in SUPPORTED_CLASSIFIERS:
        raise ValueError(
            f"Unsupported classifier type: '{name}'. "
            f'Available: {list(SUPPORTED_CLASSIFIERS)}'
        )
    return SUPPORTED_CLASSIFIERS[name](hyperparameters or {})


class TextLabelClassifier:
    """
    Any listed classifier, fitted on text labels ('ok', 'merged', ...) with
    missing values filled by the training median and, when `balanced`, class
    weights on every algorithm (XGBoost has no class_weight, so it gets
    sample weights). Behaves like a sklearn classifier: fit, predict_proba,
    classes_.
    """

    def __init__(self, name='Gradient Boosting', hyperparameters=None):
        self.name = name
        self.hyperparameters = dict(hyperparameters or {})

    def fit(self, X, y):
        from sklearn.impute import SimpleImputer

        y = np.asarray(y)
        self.classes_, codes = np.unique(y, return_inverse=True)
        self.imputer_ = SimpleImputer(strategy='median')
        Xi = self.imputer_.fit_transform(X)
        self.estimator_ = build_estimator(self.name, self.hyperparameters)
        weight = None
        if self.hyperparameters.get('balanced') and self.name == 'XGBoost':
            counts = np.bincount(codes)
            weight = (len(codes) / (len(counts) * counts))[codes]
        if weight is None:
            self.estimator_.fit(Xi, codes)
        else:
            self.estimator_.fit(Xi, codes, sample_weight=weight)
        self.estimator_classes_ = np.asarray(self.estimator_.classes_)
        return self

    def predict_proba(self, X):
        raw = self.estimator_.predict_proba(self.imputer_.transform(X))
        out = np.zeros((len(raw), len(self.classes_)))
        out[:, self.estimator_classes_.astype(int)] = raw
        return out
