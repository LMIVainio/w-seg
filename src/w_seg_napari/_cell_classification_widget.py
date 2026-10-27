import json
import os
from typing import Any

import h5py
import joblib
import napari
import numpy as np
import pandas as pd
import sklearn
import tifffile
from qtpy.QtCore import QStandardPaths
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import binary_dilation, generate_binary_structure
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import w_seg_napari._wsegutils as w

# Optional imports for advanced classifiers
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

# Version of the classifier file written by CellClassifierManager.save_model
CLASSIFIER_SCHEMA_VERSION = 2


def _clean_extraction_config(config) -> dict:
    """Copy of a feature extraction config without non-measurement entries."""
    return {k: v for k, v in (config or {}).items() if k != 'classifier_type'}


# Factory Registry for extensible estimator instantiation
SUPPORTED_CLASSIFIERS = {
    'Random Forest': lambda hp: RandomForestClassifier(
        n_estimators=hp.get('n_estimators', 100),
        max_depth=hp.get('max_depth', 15),
        random_state=42,
        n_jobs=-1,
    ),
    'SVM': lambda hp: SVC(
        C=hp.get('C', 1.0),
        kernel=hp.get('kernel', 'rbf'),
        probability=True,
        random_state=42,
    ),
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
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


class CellClassifierManager:
    def __init__(
        self,
        classifier_type: str = 'Random Forest',
        hyperparameters: dict | None = None,
    ):
        # Setup classifier manager
        self.classifier_type = classifier_type
        self.hyperparameters = hyperparameters or {
            'n_estimators': 100,
            'max_depth': 15,
        }
        self.feature_names: list[str] = []
        self.is_trained: bool = False
        self.model_config: dict[str, Any] = {}
        self.loaded_model_path: str | None = None
        self.pipeline: Pipeline | None = None

        self._build_pipeline()

    def _build_pipeline(self):
        """Instantiates the scikit-learn Pipeline based on classifier_type."""
        if self.classifier_type not in SUPPORTED_CLASSIFIERS:
            raise ValueError(
                f"Unsupported classifier type: '{self.classifier_type}'. "
                f'Available: {list(SUPPORTED_CLASSIFIERS.keys())}'
            )

        estimator = SUPPORTED_CLASSIFIERS[self.classifier_type](
            self.hyperparameters
        )
        self.pipeline = Pipeline(
            [
                ('imputer', SimpleImputer(strategy='median')),
                ('clf', estimator),
            ]
        )

    def set_classifier(
        self, classifier_type: str, hyperparameters: dict | None = None
    ):
        """Reconfigures the active classifier type and rebuilds pipeline."""
        self.classifier_type = classifier_type
        self.hyperparameters = hyperparameters or {}
        self.is_trained = False
        self._build_pipeline()

    def create_annotation_layer_data(
        self, labels_layer: np.ndarray, erosion_steps: int = 1
    ) -> np.ndarray:
        """Creates an annotation layer from labels layer.

        Converts all labels to 1 (binary mask) and protects touching labels from
        merging by separating shared boundaries.
        """
        if labels_layer is None or not np.any(labels_layer):
            return np.zeros_like(labels_layer, dtype=np.uint32)

        ndim = labels_layer.ndim
        # 8-connectivity for 2D, 26-connectivity for 3D
        struct_full = generate_binary_structure(ndim, ndim)

        # 1. Detect boundary pixels where adjacent label IDs differ (excluding background)
        touching_boundaries = np.zeros_like(labels_layer, dtype=bool)

        # Fast vector check along each spatial axis
        for axis in range(ndim):
            slice1 = [slice(None)] * ndim
            slice2 = [slice(None)] * ndim

            slice1[axis] = slice(0, -1)
            slice2[axis] = slice(1, None)

            val1 = labels_layer[tuple(slice1)]
            val2 = labels_layer[tuple(slice2)]

            # Find pixels where labels differ and neither is background (0)
            diff_mask = (val1 != val2) & (val1 != 0) & (val2 != 0)

            # Mark both sides of the interface as boundary
            touching_boundaries[tuple(slice1)] |= diff_mask
            touching_boundaries[tuple(slice2)] |= diff_mask

        # 2. Expand boundary thickness according to erosion_steps
        if erosion_steps > 1:
            touching_boundaries = binary_dilation(
                touching_boundaries,
                structure=struct_full,
                iterations=erosion_steps - 1,
            )

        # 3. Create binary foreground mask and carve out the boundaries
        fg_mask = (labels_layer > 0) & (~touching_boundaries)

        return fg_mask.astype(np.uint32)

    def extract_training_data(
        self,
        labels_layer: np.ndarray,
        annotation_layer: np.ndarray,
        df_spatial_features: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, pd.Series]:

        if 'label_id' not in df_spatial_features.columns:
            raise KeyError(
                "df_spatial_features must contain a 'label_id' column."
            )

        # 1. Sanitize the full DataFrame directly
        df_clean = w.sanitize_feature_matrix(df_spatial_features)

        # 2. Identify target training features
        non_feature_cols = [
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
        ]
        feature_cols = [
            c
            for c in df_clean.columns
            if c not in non_feature_cols and not c.startswith('class_')
        ]
        self.feature_names = feature_cols

        cell_ids = df_clean['label_id'].values
        y_labels = []
        valid_indices = []

        for idx, cell_id in enumerate(cell_ids):
            mask = labels_layer == cell_id
            if not np.any(mask):
                continue

            annotated_vals = annotation_layer[mask]
            valid_vals = annotated_vals[annotated_vals > 0]

            if len(valid_vals) > 0:
                classes, counts = np.unique(valid_vals, return_counts=True)
                majority_class = classes[np.argmax(counts)]
                y_labels.append(majority_class)
                valid_indices.append(idx)

        # 3. Extract clean X and y arrays
        X = df_clean.iloc[valid_indices][feature_cols].values
        y = np.array(y_labels)
        valid_cell_ids = df_clean.iloc[valid_indices]['label_id']

        return X, y, valid_cell_ids

    def train(
        self, X: np.ndarray, y: np.ndarray, feature_config: dict | None = None
    ) -> dict[str, Any]:
        """Trains pipeline and updates active model configuration."""
        if len(X) == 0:
            raise ValueError('No annotated training samples found.')

        self.pipeline.fit(X, y)
        self.is_trained = True

        # Store standardized model config
        self.model_config = self._make_model_config(feature_config)

        # Calculate feature importances if model supports it
        clf_step = self.pipeline.named_steps['clf']
        top_features = []
        if hasattr(clf_step, 'feature_importances_'):
            importances = clf_step.feature_importances_
            top_features = sorted(
                zip(self.feature_names, importances, strict=False),
                key=lambda x: x[1],
                reverse=True,
            )[:5]

        return {
            'n_samples': len(X),
            'classes': np.unique(y).tolist(),
            'top_features': top_features or {},
            'classifier_type': self.classifier_type,
        }

    def predict(
        self, df_spatial_features: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        if not self.is_trained:
            raise RuntimeError('Classifier is not trained yet!')

        # Sanitize full DataFrame prior to feature slicing
        df_clean = w.sanitize_feature_matrix(df_spatial_features)
        # Feature Alignment Guard: Ensure exact feature columns exist in exact order
        for col in self.feature_names:
            if col not in df_clean.columns:
                print(
                    f'[w-seg] Warning {col} not included in data frame and was replaced with NaNs'
                )
                df_clean[col] = np.nan  # Imputer will handle missing features
        X = df_clean[self.feature_names].values
        cell_ids = df_clean['label_id'].values
        preds = self.pipeline.predict(X)

        # Attach prediction column into the dataframe so it persists during export
        df_spatial_features['predicted_class'] = preds
        return df_spatial_features, cell_ids, preds

    def generate_prediction_layer(
        self,
        labels_layer: np.ndarray,
        cell_ids: np.ndarray,
        predictions: np.ndarray,
    ) -> np.ndarray:
        """Fast vectorized mapping of predicted class IDs back onto full 2D image coordinates using direct array lookup."""
        max_id = int(np.max(labels_layer))
        max_req_id = int(np.max(cell_ids)) if len(cell_ids) > 0 else 0
        lut_size = max(max_id, max_req_id) + 1

        # Create array-based lookup table
        lut = np.zeros(lut_size, dtype=np.uint32)
        lut[cell_ids.astype(int)] = predictions.astype(np.uint32)

        # Single-pass 2D array re-indexing
        return lut[labels_layer]

    def _make_model_config(self, feature_config: dict | None) -> dict:
        """The one config layout used after training, saving and loading.

        feature_extraction_config holds the morphology measurement settings
        (pixel spacing, search radius, ...) the classifier was trained with.
        """
        return {
            'schema_version': CLASSIFIER_SCHEMA_VERSION,
            'classifier_type': self.classifier_type,
            'hyperparameters': dict(self.hyperparameters),
            'feature_names': list(self.feature_names),
            'feature_extraction_config': _clean_extraction_config(
                feature_config
            ),
        }

    def save_model(self, filepath: str, feature_config: dict | None = None):
        """Saves trained model pipeline and unified payload schema."""
        if not filepath.endswith('.joblib'):
            filepath += '.joblib'

        if not feature_config:  # fall back to the config of the training run
            feature_config = self.model_config.get(
                'feature_extraction_config', {}
            )
        self.model_config = self._make_model_config(feature_config)

        payload = {
            'schema_version': CLASSIFIER_SCHEMA_VERSION,
            'sklearn_version': sklearn.__version__,
            'pipeline': self.pipeline,
            'classifier_type': self.classifier_type,
            'hyperparameters': self.hyperparameters,
            'feature_names': list(self.feature_names),
            'is_trained': self.is_trained,
            'feature_extraction_config': self.model_config[
                'feature_extraction_config'
            ],
        }

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(payload, filepath)
        self.loaded_model_path = filepath
        print(f'[w-seg] Model payload saved successfully to: {filepath}')

    def load_model(self, filepath: str) -> dict[str, Any]:
        """Loads trained model payload and updates active state.

        Only load classifier files from sources you trust: joblib files can
        run code when opened.
        """
        payload = joblib.load(filepath)
        if not isinstance(payload, dict) or 'pipeline' not in payload:
            raise ValueError(
                'This file is not a w-seg classifier (no pipeline found).'
            )

        saved_with = payload.get('sklearn_version')
        if saved_with and saved_with != sklearn.__version__:
            print(
                f'[w-seg] Warning: classifier was saved with scikit-learn {saved_with}, '
                f'this is {sklearn.__version__}. Predictions may differ or fail.'
            )

        self.pipeline = payload['pipeline']
        self.feature_names = list(payload.get('feature_names', []))
        self.is_trained = payload.get('is_trained', True)
        self.hyperparameters = payload.get('hyperparameters', {})
        self.loaded_model_path = filepath

        # Resolve classifier_type explicitly
        if 'classifier_type' in payload:
            self.classifier_type = payload['classifier_type']
        elif (
            hasattr(self.pipeline, 'named_steps')
            and 'clf' in self.pipeline.named_steps
        ):
            self.classifier_type = type(
                self.pipeline.named_steps['clf']
            ).__name__
        else:
            self.classifier_type = 'Loaded Classifier'

        # Files from before schema 2 used 'feature_config' (and sometimes
        # stored classifier_type inside it); both are handled here.
        fe_config = (
            payload.get('feature_extraction_config')
            or payload.get('feature_config')
            or {}
        )
        self.model_config = self._make_model_config(fe_config)
        return self.model_config


class CellClassificationWidget(QWidget):
    def __init__(
        self,
        napari_viewer: napari.Viewer | None = None,
        viewer: napari.Viewer | None = None,
        df: pd.DataFrame | None = None,
        labels_data: np.ndarray | None = None,
        feature_extraction_config: dict | None = None,
        layer_name: str = 'Unspecified Layer',
        parent=None,
    ):
        super().__init__(parent=parent)
        self.viewer = napari_viewer or viewer
        self.clf_manager = CellClassifierManager()

        # --- Frozen Data Pair State ---
        self.cached_df: pd.DataFrame | None = None
        self.cached_labels_data: np.ndarray | None = None
        self.feature_extraction_config: dict | None = (
            feature_extraction_config or {}
        )
        self.model_config: dict | None = None
        self.current_layer_name: str = layer_name
        self.annotation_layer: napari.layers.Labels | None = None

        self._init_ui()

        # Bind initial data if provided
        if df is not None and labels_data is not None:
            self.bind_measured_data(
                df, labels_data, layer_name, feature_extraction_config
            )

    def _init_ui(self):
        layout = QVBoxLayout()

        # --- 1. Model Selection box ---
        model_group = QGroupBox('Classifier Selection')
        model_layout = QVBoxLayout(model_group)

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel('Algorithm:'))
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(list(SUPPORTED_CLASSIFIERS.keys()))
        self.combo_algo.currentTextChanged.connect(self._on_algorithm_changed)
        h_layout.addWidget(self.combo_algo)
        model_layout.addLayout(h_layout)

        # Hyperparameter inputs (Example: Trees/Estimators)
        hp_layout = QHBoxLayout()
        hp_layout.addWidget(QLabel('N Estimators:'))
        self.spin_estimators = QSpinBox()
        self.spin_estimators.setRange(10, 1000)
        self.spin_estimators.setValue(100)
        hp_layout.addWidget(self.spin_estimators)
        model_layout.addLayout(hp_layout)

        layout.addWidget(model_group)

        # --- 2. Frozen Data Pair Display ---
        info_group = QGroupBox('Active Measured Data Source')
        info_layout = QVBoxLayout()

        self.lbl_active_source = QLabel(
            'Source Layer: <i>No measurements bound yet</i>'
        )
        self.lbl_active_source.setWordWrap(True)
        info_layout.addWidget(self.lbl_active_source)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # --- 3. Class Annotation Control ---
        annot_group = QGroupBox('1. Interactive Annotation')
        annot_layout = QVBoxLayout()

        self.btn_create_annot = QPushButton('Create/Reset Manual Class Layer')
        self.btn_create_annot.setEnabled(False)
        self.btn_create_annot.clicked.connect(self._on_create_annotation_layer)
        annot_layout.addWidget(self.btn_create_annot)

        self.lbl_annot_status = QLabel('Class Layer: None')
        annot_layout.addWidget(self.lbl_annot_status)

        annot_group.setLayout(annot_layout)
        layout.addWidget(annot_group)

        # --- 4. Model Training ---
        train_group = QGroupBox('2. Classifier Training')
        train_layout = QVBoxLayout()

        self.btn_train = QPushButton('Train Classifier')
        self.btn_train.setEnabled(False)
        self.btn_train.clicked.connect(self._on_train_model)
        train_layout.addWidget(self.btn_train)

        self.lbl_status = QLabel('Status: Model Untrained/Not Loaded')
        train_layout.addWidget(self.lbl_status)

        train_group.setLayout(train_layout)
        layout.addWidget(train_group)

        # --- 4. IO Operations ---
        io_layout = QHBoxLayout()
        self.btn_save = QPushButton('Save Model')
        self.btn_save.clicked.connect(self._on_save_model)
        self.btn_load = QPushButton('Load Model')
        self.btn_load.clicked.connect(self._on_load_model)

        io_layout.addWidget(self.btn_save)
        io_layout.addWidget(self.btn_load)
        layout.addLayout(io_layout)

        # --- 5. Apply parameters ---
        self.btn_copy_config = QPushButton(
            '📋 Apply Model Parameters to Inputs'
        )
        self.btn_copy_config.setEnabled(
            False
        )  # Enable only when a model is loaded/trained
        self.btn_copy_config.clicked.connect(self._on_copy_config_clicked)

        layout.addWidget(self.btn_copy_config)

        # --- 6. Prediction ---
        self.btn_predict = QPushButton('Predict Cell Classes')
        self.btn_predict.setEnabled(False)
        self.btn_predict.setStyleSheet('font-weight: bold; padding: 6px;')
        self.btn_predict.clicked.connect(self._on_predict_classes)
        layout.addWidget(self.btn_predict)

        # --- 7. Save Analytics ---
        export_group = QGroupBox('3. Export & Analytics')
        export_layout = QVBoxLayout()

        layer_select_layout = QHBoxLayout()
        lbl_combo_title = QLabel('Annotation/Class Layer:')
        self.combo_export_layer = QComboBox()
        self.combo_export_layer.setToolTip(
            'Select the Napari Labels layer containing your custom or predicted class IDs.'
        )

        self.btn_refresh_layers = QPushButton('🔄')
        self.btn_refresh_layers.setFixedWidth(30)
        self.btn_refresh_layers.setToolTip('Refresh Napari Layers')
        self.btn_refresh_layers.clicked.connect(
            self._refresh_export_layer_choices
        )

        layer_select_layout.addWidget(lbl_combo_title)
        layer_select_layout.addWidget(self.combo_export_layer, stretch=1)
        layer_select_layout.addWidget(self.btn_refresh_layers)
        export_layout.addLayout(layer_select_layout)

        self.btn_export_data = QPushButton('📊 Export Data and Metadata')
        self.btn_export_data.setEnabled(
            False
        )  # Enable when cached_df is bound/predicted
        self.btn_export_data.clicked.connect(self._on_export_analytics)
        export_layout.addWidget(self.btn_export_data)

        self.btn_export_layers = QPushButton('📦 Export Napari Layers')
        self.btn_export_layers.clicked.connect(self._on_export_napari_layers)
        export_layout.addWidget(self.btn_export_layers)

        self.btn_import_layers = QPushButton('📂 Import Napari Layers')
        self.btn_import_layers.clicked.connect(self._on_import_napari_layers)
        export_layout.addWidget(self.btn_import_layers)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        layout.addStretch()
        self.setLayout(layout)

    # Add function to refresh layer choices
    def _refresh_export_layer_choices(self):
        """Populates the export dropdown with all active Napari Labels layers."""
        if self.viewer is None:
            return

        current_selection = self.combo_export_layer.currentText()
        self.combo_export_layer.clear()
        self.combo_export_layer.addItem('-- None (Features Only) --', None)

        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                # Pass actual layer object as userData in the item
                self.combo_export_layer.addItem(layer.name, layer)

        # Restore previous selection if still present
        index = self.combo_export_layer.findText(current_selection)
        if index >= 0:
            self.combo_export_layer.setCurrentIndex(index)

    def _on_algorithm_changed(self, algo_name: str):
        """Updates classifier type when user changes dropdown."""
        hp = {'n_estimators': self.spin_estimators.value()}
        self.clf_manager.set_classifier(algo_name, hp)
        self._refresh_status_label()

    def _refresh_status_label(self):
        """Refreshes status text reliably."""
        algo = self.clf_manager.classifier_type
        trained = 'Trained' if self.clf_manager.is_trained else 'Not Trained'
        path = self.clf_manager.loaded_model_path or 'In-Memory'
        self.lbl_status.setText(f'Active: {algo} ({trained}) | Source: {path}')

    def _refresh_active_classifier(self):
        """Refreshes the active classifier"""
        if hasattr(self.clf_manager, 'classifier_type'):
            return self.clf_manager.classifier_type
        else:
            return 'Not Trained/Loaded'

    def bind_measured_data(
        self,
        df: pd.DataFrame,
        labels_data: np.ndarray,
        layer_name: str = 'Active Labels',
        feature_extraction_config: dict | None = None,
    ):
        self.cached_df = w.sanitize_feature_matrix(df)
        self.cached_labels_data = labels_data.copy()
        self.current_layer_name = layer_name

        # Cache active measurement extraction config
        if feature_extraction_config is not None:
            self.feature_extraction_config = feature_extraction_config

        n_cells = len(self.cached_df)
        self.lbl_active_source.setText(
            f'<b>Source Layer:</b> {self.current_layer_name}<br/>'
            f'<b>Bound Cells:</b> {n_cells}'
        )
        self.btn_create_annot.setEnabled(True)
        self.btn_export_data.setEnabled(True)

        # Refresh choices
        self._refresh_export_layer_choices()
        if self.clf_manager.is_trained:
            self.btn_predict.setEnabled(True)

    def _on_create_annotation_layer(self):
        if self.cached_labels_data is None:
            QMessageBox.warning(
                self, 'Warning', 'No measured labels layer bound.'
            )
            return

        # Build eroded 0/1 background class template
        annot_data = self.clf_manager.create_annotation_layer_data(
            self.cached_labels_data, erosion_steps=1
        )
        annot_layer_name = f'[Class Annotations] {self.current_layer_name}'

        # If annotation layer already exists in Napari, replace or select it
        if self.viewer is not None:
            if annot_layer_name in self.viewer.layers:
                self.annotation_layer = self.viewer.layers[annot_layer_name]
                self.annotation_layer.data = annot_data
            else:
                self.annotation_layer = self.viewer.add_labels(
                    annot_data, name=annot_layer_name, opacity=0.6
                )

            # Configure active painting mode
            self.annotation_layer.mode = 'fill'
            self.annotation_layer.selected_label = 2  # Start painting Class 2

            self.lbl_annot_status.setText(
                f'Class Layer: <i>{annot_layer_name}</i>'
            )
            self.btn_train.setEnabled(True)

    def _on_train_model(self):
        if (
            self.annotation_layer is None
            or self.cached_labels_data is None
            or self.cached_df is None
        ):
            QMessageBox.warning(
                self,
                'Warning',
                'Annotation layer or spatial feature matrix is missing.',
            )
            return

        try:
            """Handles classifier training from GUI"""
            # Update hyperparameters
            self.clf_manager.hyperparameters['n_estimators'] = (
                self.spin_estimators.value()
            )
            self.clf_manager._build_pipeline()

            X, y, _ = self.clf_manager.extract_training_data(
                labels_layer=self.cached_labels_data,
                annotation_layer=self.annotation_layer.data,
                df_spatial_features=self.cached_df,
            )

            metrics = self.clf_manager.train(
                X, y, feature_config=self.feature_extraction_config
            )

            # Measurement settings the model was trained with (flat dict)
            self.model_config = dict(
                self.clf_manager.model_config['feature_extraction_config']
            )

            top_feat_str = '\n'.join(
                [
                    f'- {name}: {imp:.3f}'
                    for name, imp in metrics['top_features']
                ]
            )
            self.lbl_status.setText(
                f'Status: Trained ({metrics["n_samples"]} samples)'
                '\n'
                f'Classifer: {self.clf_manager.classifier_type}'
            )
            self.btn_predict.setEnabled(True)
            self.btn_copy_config.setEnabled(True)
            self._refresh_status_label()
            QMessageBox.information(
                self,
                'Training Complete',
                f'Model successfully trained on {self.current_layer_name}!\n\n'
                f'Samples: {metrics["n_samples"]}\n'
                f'Classes Found: {metrics["classes"]}\n\n'
                f'Top Features:\n{top_feat_str}',
            )

        except (OSError, RuntimeError, ValueError, TypeError) as e:
            QMessageBox.critical(self, 'Training Error', str(e))

    def _on_predict_classes(self):
        if not self.clf_manager.is_trained:
            QMessageBox.warning(
                self, 'Warning', 'Please train or load a model first.'
            )
            return

        if self.cached_labels_data is None or self.cached_df is None:
            QMessageBox.warning(
                self, 'Warning', 'No measured labels layer bound.'
            )
            return

        # --- Parameter Mismatch Guard ---
        if not self.check_config_compatibility():
            return  # User clicked 'No' on warning dialog

        try:
            self.cached_df, cell_ids, preds = self.clf_manager.predict(
                self.cached_df
            )
            pred_2d_layer = self.clf_manager.generate_prediction_layer(
                labels_layer=self.cached_labels_data,
                cell_ids=cell_ids,
                predictions=preds,
            )

            pred_layer_name = f'[Predicted Classes] {self.current_layer_name}'

            if self.viewer is not None:
                if pred_layer_name in self.viewer.layers:
                    self.viewer.layers[pred_layer_name].data = pred_2d_layer
                else:
                    self.viewer.add_labels(
                        pred_2d_layer, name=pred_layer_name, opacity=0.7
                    )
            # Enable export button
            if hasattr(self, 'btn_export_data'):
                self.btn_export_data.setEnabled(True)

        except (RuntimeError, ValueError, TypeError, OSError) as e:
            QMessageBox.critical(self, 'Prediction Error', str(e))

    def _on_save_model(self):
        if not self.clf_manager.is_trained:
            QMessageBox.warning(
                self, 'Warning', 'Train a Classifier before saving.'
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Classifier', '', 'Joblib Model (*.joblib)'
        )
        if path:
            active_config = self.model_config or self.feature_extraction_config
            self.clf_manager.save_model(path, feature_config=active_config)
            QMessageBox.information(self, 'Success', f'Model saved to {path}')
            self._refresh_status_label()

    def _on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load Classifier', '', 'Joblib Model (*.joblib)'
        )
        if not path:
            return

        if path:
            try:
                self.clf_manager.load_model(path)
                # Measurement settings the model was trained with (flat dict)
                self.model_config = dict(
                    self.clf_manager.model_config['feature_extraction_config']
                )
                n_est = self.clf_manager.hyperparameters.get('n_estimators')
                if n_est:
                    self.spin_estimators.setValue(int(n_est))
                self.lbl_status.setText(
                    f'Status: {self.clf_manager.classifier_type} Loaded'
                )
                self.check_config_compatibility()
                if self.cached_df is not None:
                    self.btn_predict.setEnabled(True)
                QMessageBox.information(
                    self,
                    'Success',
                    f'{self.clf_manager.classifier_type} loaded successfully.',
                )
                # Enable copy settings button
                self.btn_copy_config.setEnabled(True)
                # Sync combo box without triggering signal
                self.btn_copy_config.setEnabled(bool(self.model_config))
                self.combo_algo.blockSignals(True)
                idx = self.combo_algo.findText(
                    self.clf_manager.classifier_type
                )
                if idx >= 0:
                    self.combo_algo.setCurrentIndex(idx)
                self.combo_algo.blockSignals(False)

                self._refresh_status_label()
            except (RuntimeError, ValueError, TypeError, OSError) as e:
                QMessageBox.critical(
                    self, 'Error', f'Failed to load model: {str(e)}'
                )

    def _on_copy_config_clicked(self):
        if not self.model_config:
            QMessageBox.warning(
                self,
                'Warning',
                'No loaded or trained model configuration found.',
            )
            return

        # Callback set by the parent bridge (Tab3Analysis)
        if (
            hasattr(self, 'on_apply_config_callback')
            and self.on_apply_config_callback
        ):
            self.on_apply_config_callback(dict(self.model_config))
            info_txt = (
                '<b>Successfully copied model parameters to input fields!</b>'
                '<b>Remember</b> to Remeasure Cell Wall Morphology to apply changes!'
            )
            QMessageBox.information(self, 'Parameters Restored', info_txt)

            self.combo_algo.blockSignals(True)
            idx = self.combo_algo.findText(self.clf_manager.classifier_type)
            if idx >= 0:
                self.combo_algo.setCurrentIndex(idx)
            self.combo_algo.blockSignals(False)

    def _update_ui_state(self):
        """Enable the copy button whenever a model config is available."""
        has_config = bool(self.model_config)
        self.btn_copy_config.setEnabled(has_config)

    def check_config_compatibility(self) -> bool:
        """Compares active measurement config with model config.

        Warns the user if parameters mismatch.
        """
        # Debug prints to trace empty state issues
        # print(f"[w-seg Debug] Model Config: {self.model_config}")
        # print(f"[w-seg Debug] Active Config: {self.feature_extraction_config}")

        if not self.model_config or not self.feature_extraction_config:
            from napari.utils.notifications import show_info

            show_info('Warning: classifier configuration is missing')
            return True

        mismatches = []

        # Normalize keys for comparison
        for key, model_val in self.model_config.items():
            active_val = {}
            if key in self.feature_extraction_config:
                active_val = self.feature_extraction_config[key]

            # Float tolerance check to prevent False Positives (e.g., 0.5 vs 0.50)
            if isinstance(model_val, (int, float)) and isinstance(
                active_val, (int, float)
            ):
                is_different = not np.isclose(model_val, active_val)
            else:
                is_different = str(model_val) != str(active_val)

            if is_different:
                mismatches.append(
                    f'• <b>{key}</b>: Model used <code>{model_val}</code> vs Active'
                    f' <code>{active_val}</code>'
                )

        if mismatches:
            mismatch_str = '<br/>'.join(mismatches)
            warning_txt = (
                f'<b>Warning:</b> Active measurement parameters do not match model extraction parameters.<br/><br/>'
                f'<b>Mismatched Parameters:</b><br/>{mismatch_str}<br/><br/>'
                f'<i>Remember to Remeasure Cell Wall Morphology to apply changes!</i><br/><br/>'
                f'Proceed with prediction anyway?'
            )
            reply = QMessageBox.warning(
                self,
                'Parameter Mismatch Warning',
                warning_txt,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            return reply == QMessageBox.Yes

        return True

    def export_analytics_data(
        self, filepath: str, class_layer=None
    ) -> pd.DataFrame:
        """
        Headless exporter: Merges cached feature data with class labels and writes to disk (.xlsx or .csv).

        Parameters:
            filepath (str): Destination file path ending in .xlsx or .csv.
            class_layer (napari.layers.Layer, optional): Napari layer containing custom class markings.
                                                        If None, tries reading from combo_export_layer.

        Returns:
            pd.DataFrame: The generated export DataFrame containing all features and class columns.
        """
        if self.cached_df is None or self.cached_df.empty:
            raise ValueError('No measured data available in cache to export.')

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        # Build complete export DataFrame (merging features + prediction labels)
        export_df = self.cached_df.copy()

        # Determine class layer source (argument override or UI dropdown)
        selected_layer = class_layer
        if selected_layer is None and hasattr(self, 'combo_export_layer'):
            selected_layer = self.combo_export_layer.currentData()

        # Map custom labels from the selected layer (if available)
        if selected_layer is not None and self.cached_labels_data is not None:
            custom_classes = self._extract_classes_from_layer(
                segmentation_labels=self.cached_labels_data,
                class_layer_data=selected_layer.data,
                cell_ids=export_df['label_id'].values,
            )
            # Attach custom class column named after the Napari layer
            column_name = f'class_{selected_layer.name.replace(" ", "_")}'
            export_df[column_name] = custom_classes

        # Active extraction config metadata dictionary
        meta_config = dict(
            getattr(self, 'model_config', None)
            or getattr(self, 'feature_extraction_config', None)
            or {}
        )
        if self.clf_manager.is_trained:
            meta_config['classifier_type'] = self.clf_manager.classifier_type

        # Dispatch to specific writer depending on extension
        if filepath.lower().endswith('.xlsx'):
            self._save_excel_with_meta(filepath, export_df, meta_config)
        else:
            self._save_csv_with_meta_header(filepath, export_df, meta_config)

        return export_df

    def _on_export_analytics(self):
        """Interactive GUI button callback."""
        if self.cached_df is None or self.cached_df.empty:
            QMessageBox.warning(
                self, 'Export Error', 'No measured data to export.'
            )
            return

        docs_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation
        )
        default_filename = os.path.join(
            docs_dir, 'cell_morphology_analytics.xlsx'
        )

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            'Export Analytics Data',
            default_filename,
            'Excel Workbook (*.xlsx);;CSV File with Header Meta (*.csv)',
        )

        if not filepath:
            return

        try:
            self.export_analytics_data(filepath)
            QMessageBox.information(
                self,
                'Export Successful',
                f'Analytics exported successfully to:<br/><code>{filepath}</code>',
            )
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            QMessageBox.critical(
                self, 'Export Failed', f'Failed to save file:\n{str(e)}'
            )

    def export_to_h5(self, save_path: str) -> int:
        """Handler: Extracts current viewer layers into a payload and forwards it to the headless HDF5 exporter."""
        if self.viewer is None or len(self.viewer.layers) == 0:
            return 0

        layers_payload = []

        for layer in self.viewer.layers:
            layer_type = layer._type_string
            base_meta = {
                'opacity': float(layer.opacity),
                'scale': list(layer.scale),
                'visible': bool(layer.visible),
                'blending_mode': str(layer.blending),
            }

            if isinstance(layer, (napari.layers.Image, napari.layers.Labels)):
                meta = base_meta.copy()
                meta['colormap'] = getattr(
                    layer.colormap, 'name', str(layer.colormap)
                )

                if layer_type == 'image':
                    meta['rgb'] = bool(getattr(layer, 'rgb', False))

                elif layer_type == 'labels':
                    color_dict = None
                    if (
                        hasattr(layer.colormap, 'color_dict')
                        and layer.colormap.color_dict
                    ):
                        color_dict = layer.colormap.color_dict
                    elif (
                        hasattr(layer, 'color')
                        and isinstance(layer.color, dict)
                        and layer.color
                    ):
                        color_dict = layer.color

                    if color_dict is not None:
                        meta['custom_label_colors'] = color_dict

                layers_payload.append(
                    {
                        'name': layer.name,
                        'type': layer_type,
                        'data': layer.data,
                        'metadata': meta,
                    }
                )

            elif isinstance(layer, napari.layers.Shapes):
                meta = base_meta.copy()
                meta['shape_types'] = layer.shape_types

                layers_payload.append(
                    {
                        'name': layer.name,
                        'type': layer_type,
                        'shapes_data': layer.data,
                        'metadata': meta,
                    }
                )

        # Delegate execution to headless utility
        return w.export_layers_to_h5(layers_payload, save_path)

    def export_to_tiff_dir(self, dirpath: str) -> int:
        """Headless exporter: Saves all Image/Labels layers as individual TIFF files.

        Returns the count of exported layers.
        """
        if self.viewer is None or len(self.viewer.layers) == 0:
            return 0

        os.makedirs(dirpath, exist_ok=True)
        exported_count = 0

        for index, layer in enumerate(self.viewer.layers):
            if isinstance(layer, (napari.layers.Image, napari.layers.Labels)):
                data = np.ascontiguousarray(layer.data)

                if data.ndim < 2:
                    data = np.atleast_2d(data)

                # Downcast 64-bit integers safely for ImageJ compatibility
                if data.dtype in (np.int64, np.uint64):
                    min_val = data.min() if data.size > 0 else 0
                    max_val = data.max() if data.size > 0 else 0

                    if min_val >= 0 and max_val <= 65535:
                        data = data.astype(np.uint16)
                    elif min_val >= -2147483648 and max_val <= 2147483647:
                        data = data.astype(np.int32)
                    else:
                        print(
                            f'[w-seg] Warning: Layer "{layer.name}" values exceed int32 range. '
                            f'Saving as {data.dtype}.'
                        )
                elif data.dtype == np.float64:
                    data = data.astype(np.float32)

                dtype_str = str(data.dtype)
                layer_name_safe = layer.name.replace(' ', '_').replace(
                    '/', '_'
                )
                save_path = os.path.join(
                    dirpath, f'{index:02d}_{layer_name_safe}_{dtype_str}.tif'
                )

                is_rgb = getattr(layer, 'rgb', False) or (
                    data.ndim >= 3
                    and data.shape[-1] in (3, 4)
                    and data.dtype == np.uint8
                )

                if is_rgb:
                    if data.shape[-1] == 4:
                        data = data[..., :3]
                    tifffile.imwrite(
                        save_path, data, photometric='rgb', imagej=True
                    )
                elif data.dtype in (np.int32, np.uint32):
                    tifffile.imwrite(
                        save_path,
                        data,
                        photometric='minisblack',
                        metadata={'axes': 'YX' if data.ndim == 2 else 'YX'},
                        imagej=False,
                    )
                else:
                    tifffile.imwrite(save_path, data, imagej=True)

                exported_count += 1

        return exported_count

    def _on_export_napari_layers(self):
        """Interactive GUI button callback."""
        if self.viewer is None or len(self.viewer.layers) == 0:
            QMessageBox.warning(
                self, 'Export Error', 'No Napari viewer or layers available.'
            )
            return

        shortest_name = min(
            (layer.name for layer in self.viewer.layers),
            key=len,
            default='napari_layers_export',
        )

        items = ['HDF5 Single File (.h5)', 'Individual TIFF Files (.tif)']
        item, ok = QInputDialog.getItem(
            self, 'Export Format', 'Select export format:', items, 0, False
        )
        if not ok or not item:
            return

        docs_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation
        )
        default_dir = os.path.join(docs_dir, 'napari_layers_export')

        try:
            # OPTION A: HDF5 Export
            if 'HDF5' in item:
                save_path, _ = QFileDialog.getSaveFileName(
                    self,
                    'Save Napari Project Session',
                    os.path.join(
                        default_dir, f'{shortest_name}_segmentation_session.h5'
                    ),
                    'HDF5 Files (*.h5 *.hdf5)',
                )
                if not save_path:
                    return

                count = self.export_to_h5(save_path)
                QMessageBox.information(
                    self,
                    'Export Successful',
                    f'Successfully saved {count} layer(s) to HDF5 file:<br/><code>{save_path}</code>',
                )

            # OPTION B: TIFF Directory Export
            else:
                dirpath = QFileDialog.getExistingDirectory(
                    self,
                    'Select Directory to Export Napari Layers',
                    default_dir,
                    QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
                )
                if not dirpath:
                    return

                exported_count = self.export_to_tiff_dir(dirpath)
                QMessageBox.information(
                    self,
                    'Export Successful',
                    f'{exported_count} layer(s) exported successfully to:<br/><code>{dirpath}</code>',
                )

        except (RuntimeError, ValueError, TypeError, OSError) as e:
            QMessageBox.critical(
                self, 'Export Failed', f'Failed to export layers:\n{str(e)}'
            )

    def import_h5_file(self, file_path: str) -> int:
        """
        Headless loader method: Reads an HDF5 session file and rebuilds napari layers.
        Returns the number of imported layers.
        """
        if self.viewer is None or not os.path.isfile(file_path):
            return 0

        layers_to_import = []

        with h5py.File(file_path, 'r') as h5f:
            for key in h5f:
                item = h5f[key]

                # Retrieve layer metadata
                order = int(item.attrs.get('order', 0))
                layer_type = item.attrs.get('layer_type', 'image')
                opacity = float(item.attrs.get('opacity', 1.0))
                scale = item.attrs.get('scale', None)
                visible = bool(item.attrs.get('visible', True))
                blending = item.attrs.get('blending_mode', 'translucent')

                if scale is not None:
                    scale = tuple(scale)

                # Load Shape layer Group
                if layer_type == 'shapes' and isinstance(item, h5py.Group):
                    shape_types = json.loads(
                        item.attrs.get('shape_types', '[]')
                    )
                    shapes_data = []
                    if 'shapes_data' in item:
                        s_group = item['shapes_data']
                        sorted_keys = sorted(
                            s_group.keys(),
                            key=lambda x: (
                                int(x.split('_')[-1]) if '_' in x else 0
                            ),
                        )
                        for s_key in sorted_keys:
                            shapes_data.append(s_group[s_key][()])

                    layers_to_import.append(
                        {
                            'order': order,
                            'name': key,
                            'data': shapes_data,
                            'layer_type': 'shapes',
                            'shape_type': shape_types,
                            'opacity': opacity,
                            'scale': scale,
                            'visible': visible,
                            'blending_mode': blending,
                        }
                    )

                # Load Image or Labels Data
                elif isinstance(item, h5py.Dataset):
                    data = item[()]
                    is_rgb = bool(item.attrs.get('rgb', False))
                    colormap = item.attrs.get('colormap', 'gray')

                    custom_label_colors = None
                    if 'custom_label_colors' in item.attrs:
                        try:
                            raw_colors = json.loads(
                                item.attrs['custom_label_colors']
                            )
                            custom_label_colors = {
                                int(k): v for k, v in raw_colors.items()
                            }
                        except (json.JSONDecodeError, ValueError):
                            custom_label_colors = None

                    if (
                        not is_rgb
                        and data.ndim >= 3
                        and data.shape[-1] in (3, 4)
                        and data.dtype == np.uint8
                    ):
                        is_rgb = True

                    layers_to_import.append(
                        {
                            'order': order,
                            'name': key,
                            'data': data,
                            'layer_type': layer_type,
                            'opacity': opacity,
                            'blending_mode': blending,
                            'colormap': colormap,
                            'custom_label_colors': custom_label_colors,
                            'scale': scale,
                            'visible': visible,
                            'is_rgb': is_rgb,
                        }
                    )

        # Sort layers by saved order in ascending index
        layers_to_import.sort(key=lambda x: x['order'])

        # Add layers sequentially to the viewer
        for item in layers_to_import:
            ltype = item['layer_type']

            if ltype == 'labels' or (
                ltype != 'shapes'
                and np.issubdtype(item['data'].dtype, np.integer)
                and not item['is_rgb']
            ):
                labels_kwargs = {
                    'name': item['name'],
                    'opacity': item['opacity'],
                    'scale': item['scale'],
                    'visible': item['visible'],
                    'blending': item['blending_mode'],
                }

                if item.get('custom_label_colors'):
                    from napari.utils.colormaps import DirectLabelColormap

                    labels_kwargs['colormap'] = DirectLabelColormap(
                        color_dict=item['custom_label_colors'],
                        default_color=(0.0, 0.0, 0.0, 0.0),
                    )
                elif item.get('colormap') and item['colormap'] not in (
                    'label_colormap',
                    'custom',
                    'gray',
                ):
                    try:
                        labels_kwargs['colormap'] = item['colormap']
                    except Exception:
                        pass

                self.viewer.add_labels(item['data'], **labels_kwargs)

            elif ltype == 'shapes':
                self.viewer.add_shapes(
                    item['data'],
                    shape_type=item['shape_type'],
                    name=item['name'],
                    opacity=item['opacity'],
                    scale=item['scale'],
                    visible=item['visible'],
                    blending=item['blending_mode'],
                )

            else:
                self.viewer.add_image(
                    item['data'],
                    name=item['name'],
                    opacity=item['opacity'],
                    scale=item['scale'],
                    visible=item['visible'],
                    blending=item['blending_mode'],
                    colormap=item['colormap'],
                    rgb=item['is_rgb'],
                )

        return len(layers_to_import)

    def _on_import_napari_layers(self):
        """Interactive GUI button callback."""
        if self.viewer is None:
            QMessageBox.warning(
                self, 'Import Error', 'No Napari viewer instance available.'
            )
            return

        docs_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation
        )
        default_dir = os.path.join(docs_dir, 'napari_layers_export')

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Import Napari Session (HDF5)',
            default_dir,
            'HDF5 Files (*.h5 *.hdf5)',
        )

        if not file_path:
            return

        try:
            count = self.import_h5_file(file_path)
            QMessageBox.information(
                self,
                'Import Successful',
                f'Successfully restored {count} layer(s) in original order from:<br/><code>{file_path}</code>',
            )
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            QMessageBox.critical(
                self,
                'Import Failed',
                f'Failed to import session file:\n{str(e)}',
            )

    def _extract_classes_from_layer(
        self,
        segmentation_labels: np.ndarray,
        class_layer_data: np.ndarray,
        cell_ids: np.ndarray,
    ) -> np.ndarray:
        """Samples the selected class layer for each cell ID (majority voting)."""
        extracted_classes = []

        for cell_id in cell_ids:
            mask = segmentation_labels == cell_id
            if not np.any(mask):
                extracted_classes.append(0)
                continue

            annotated_vals = class_layer_data[mask]
            valid_vals = annotated_vals[annotated_vals > 0]

            if len(valid_vals) > 0:
                # Majority vote for overlapping annotations
                classes, counts = np.unique(valid_vals, return_counts=True)
                extracted_classes.append(int(classes[np.argmax(counts)]))
            else:
                extracted_classes.append(0)  # Unlabeled cell

        return np.array(extracted_classes)

    def _save_excel_with_meta(
        self, filepath: str, df: pd.DataFrame, config: dict
    ):
        """Saves Data and Config into two separate tabs of an Excel workbook."""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Tab 1: Full cell feature data + labels
            df.to_excel(writer, sheet_name='Cell Analytics', index=True)

            # Tab 2: Feature extraction configuration
            meta_items = [
                {'Parameter': k, 'Value': v} for k, v in config.items()
            ]
            # Add source metadata details
            meta_items.insert(
                0,
                {
                    'Parameter': 'Source Layer',
                    'Value': self.current_layer_name,
                },
            )
            if hasattr(self, 'model_config') and self.model_config:
                meta_items.append(
                    {
                        'Parameter': 'Model Status',
                        'Value': 'Classifier Applied',
                    }
                )

            meta_df = pd.DataFrame(meta_items)
            meta_df.to_excel(
                writer, sheet_name='Extraction Config', index=False
            )

    def _save_csv_with_meta_header(
        self, filepath: str, df: pd.DataFrame, config: dict
    ):
        """Saves CSV with commented (#) YAML/JSON-style header metadata at the top."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('# === METADATA & CONFIGURATION ===\n')
            f.write(f'# Source Layer: {self.current_layer_name}\n')
            for k, v in config.items():
                f.write(f'# Config_{k}: {v}\n')
            f.write('# === DATA TABLE ===\n')

            # Append standard dataframe below header lines
            df.to_csv(f, index=True)

    def get_model_config(self):
        return self.clf_manager.model_config

    def get_active_config(self) -> dict:
        """Returns unified configuration snapshot for Tab 4 / Batch Processing."""
        return {
            'classifier_type': self.clf_manager.classifier_type,
            'loaded_model_path': self.clf_manager.loaded_model_path,
            'is_trained': self.clf_manager.is_trained,
            'feature_names': self.clf_manager.feature_names,
            'model_config': self.clf_manager.model_config,
        }
