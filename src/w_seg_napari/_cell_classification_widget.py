import os
from typing import Any

import joblib
import napari
import numpy as np
import pandas as pd
from qtpy.QtCore import QStandardPaths
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import binary_erosion
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import w_seg_napari._wsegutils as w


class CellClassifierManager:
    def __init__(self, n_estimators: int = 100, max_depth: int | None = 15):
        self.pipeline = Pipeline(
            [
                ('imputer', SimpleImputer(strategy='median')),
                (
                    'rf',
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        self.feature_names = []
        self.is_trained = False
        self.model_config = {}

    def create_annotation_layer_data(
        self, labels_layer: np.ndarray, erosion_steps: int = 1
    ) -> np.ndarray:
        """
        Erodes each unique cell label individually by `erosion_steps` before setting
        the interior to 1. Preserves cell separation regardless of label proximity.
        """
        if labels_layer is None:
            return None
        print('[w-seg] generating annotation layer...')
        annotated_labels = np.zeros_like(labels_layer, dtype=np.uint32)
        unique_ids = np.unique(labels_layer)
        unique_ids = unique_ids[unique_ids != 0]

        for cell_id in unique_ids:
            cell_mask = labels_layer == cell_id
            eroded_mask = binary_erosion(cell_mask, iterations=erosion_steps)

            # Fallback for very small/thin cells so they don't vanish completely
            if not np.any(eroded_mask):
                eroded_mask = cell_mask

            annotated_labels[eroded_mask] = 1

        return annotated_labels

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

    def train(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Trains the Random Forest model and returns metrics."""
        if len(X) == 0:
            raise ValueError('No annotated training samples found.')

        self.pipeline.fit(X, y)
        self.is_trained = True

        # Calculate out-of-bag feature importances
        rf_model = self.pipeline.named_steps['rf']
        importances = rf_model.feature_importances_
        top_features = sorted(
            zip(self.feature_names, importances, strict=False),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        return {
            'n_samples': len(X),
            'classes': np.unique(y).tolist(),
            'top_features': top_features,
        }

    def predict(
        self, df_spatial_features: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:

        if not self.is_trained:
            raise RuntimeError('Classifier is not trained yet!')

        # Sanitize full DataFrame prior to feature slicing
        df_clean = w.sanitize_feature_matrix(
            df_spatial_features, feature_cols=self.feature_names
        )

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

    def save_model(self, filepath: str, config: dict | None = None):
        """Saves trained model pipeline, feature names, and extraction parameters."""
        # Append .joblib extension if not present
        if not filepath.endswith('.joblib'):
            filepath += '.joblib'

        payload = {
            'pipeline': self.pipeline,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'feature_config': config or {},
        }

        # Ensure target directory exists before saving
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        # Dump classifier to file
        joblib.dump(payload, filepath)
        print(f'[w-seg] Model payload saved successfully to: {filepath}')

    def load_model(self, filepath: str):
        """Loads trained model pipeline and feature names."""
        payload = joblib.load(filepath)
        self.pipeline = payload['pipeline']
        self.feature_names = payload['feature_names']
        self.is_trained = payload['is_trained']
        self.model_config = payload.get('feature_config', {})
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
        self.feature_extraction_config: dict | None = feature_extraction_config
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

        # --- 1. Frozen Data Pair Display ---
        info_group = QGroupBox('Active Measured Data Source')
        info_layout = QVBoxLayout()

        self.lbl_active_source = QLabel(
            'Source Layer: <i>No measurements bound yet</i>'
        )
        self.lbl_active_source.setWordWrap(True)
        info_layout.addWidget(self.lbl_active_source)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # --- 2. Class Annotation Control ---
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

        # --- 3. Model Training ---
        train_group = QGroupBox('2. Classifier Training')
        train_layout = QVBoxLayout()

        self.btn_train = QPushButton('Train Classifier')
        self.btn_train.setEnabled(False)
        self.btn_train.clicked.connect(self._on_train_model)
        train_layout.addWidget(self.btn_train)

        self.lbl_status = QLabel('Status: Model Untrained')
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

        self.btn_export_data = QPushButton('💾 Export Results & Metadata')
        self.btn_export_data.setEnabled(
            False
        )  # Enable when cached_df is bound/predicted
        self.btn_export_data.clicked.connect(self._on_export_analytics)
        export_layout.addWidget(self.btn_export_data)

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
            X, y, _ = self.clf_manager.extract_training_data(
                labels_layer=self.cached_labels_data,
                annotation_layer=self.annotation_layer.data,
                df_spatial_features=self.cached_df,
            )

            metrics = self.clf_manager.train(X, y)

            # Store current active feature config as trained model config
            if self.feature_extraction_config:
                self.model_config = self.feature_extraction_config.copy()

            top_feat_str = '\n'.join(
                [
                    f'- {name}: {imp:.3f}'
                    for name, imp in metrics['top_features']
                ]
            )
            self.lbl_status.setText(
                f'Status: Trained ({metrics["n_samples"]} samples)'
            )
            self.btn_predict.setEnabled(True)

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
                self, 'Warning', 'Train a model before saving.'
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Model', '', 'Joblib Model (*.joblib)'
        )
        if path:
            active_config = self.model_config or self.feature_extraction_config
            self.clf_manager.save_model(path, config=active_config)
            QMessageBox.information(self, 'Success', f'Model saved to {path}')

    def _on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load Model', '', 'Joblib Model (*.joblib)'
        )
        if not path:
            return

        if path:
            try:
                self.clf_manager.load_model(path)
                self.model_config = getattr(
                    self.clf_manager, 'model_config', {}
                )
                self.lbl_status.setText('Status: Model Loaded')
                self.check_config_compatibility()
                if self.cached_df is not None:
                    self.btn_predict.setEnabled(True)
                QMessageBox.information(
                    self, 'Success', 'Model loaded successfully.'
                )
                self.btn_copy_config.setEnabled(bool(self.model_config))
            except (RuntimeError, ValueError, TypeError, OSError) as e:
                QMessageBox.critical(
                    self, 'Error', f'Failed to load model: {str(e)}'
                )

    # Inside CellClassificationWidget
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
            self.on_apply_config_callback(self.model_config)
            info_txt = (
                '<b>Successfully copied model parameters to input fields!</b>'
                '<b>Remember</b> to Remeasure Cell Wall Morphology to apply changes!'
            )
            QMessageBox.information(self, 'Parameters Restored', info_txt)

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

            show_info(
                'Proceeding with prediction: Parameter check skipped (missing config'
                ' metadata).'
            )
            return True

        mismatches = []

        # Normalize keys for comparison
        for key, model_val in self.model_config.items():
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

    def _on_export_analytics(self):
        if self.cached_df is None or self.cached_df.empty:
            QMessageBox.warning(
                self, 'Export Error', 'No measured data to export.'
            )
            return

        # 1. Default directory to system 'Documents' folder
        docs_dir = QStandardPaths.writableLocation(
            QStandardPaths.DocumentsLocation
        )
        default_filename = os.path.join(
            docs_dir, 'cell_morphology_analytics.xlsx'
        )

        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            'Export Analytics Data',
            default_filename,
            'Excel Workbook (*.xlsx);;CSV File with Header Meta (*.csv)',
        )

        if not filepath:
            return

        try:
            # Build complete export DataFrame (merging features + prediction labels)
            export_df = self.cached_df.copy()

            # Map custom labels from the selected dropdown layer (if selected)
            selected_layer = self.combo_export_layer.currentData()
            if (
                selected_layer is not None
                and self.cached_labels_data is not None
            ):
                custom_classes = self._extract_classes_from_layer(
                    segmentation_labels=self.cached_labels_data,
                    class_layer_data=selected_layer.data,
                    cell_ids=export_df['label_id'].values,
                )
                # Attach custom class column named after the Napari layer
                column_name = f'class_{selected_layer.name.replace(" ", "_")}'
                export_df[column_name] = custom_classes

            # Active extraction config metadata dictionary
            meta_config = (
                self.model_config or self.feature_extraction_config or {}
            )

            if filepath.endswith('.xlsx'):
                self._save_excel_with_meta(filepath, export_df, meta_config)
            else:
                self._save_csv_with_meta_header(
                    filepath, export_df, meta_config
                )

            QMessageBox.information(
                self,
                'Export Successful',
                f'Analytics exported successfully to:<br/><code>{filepath}</code>',
            )

        except (RuntimeError, ValueError, TypeError, OSError) as e:
            QMessageBox.critical(
                self, 'Export Failed', f'Failed to save file:\n{str(e)}'
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
        meta_df.to_excel(writer, sheet_name='Extraction Config', index=False)

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
