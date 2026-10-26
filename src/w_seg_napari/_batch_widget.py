import os

from qtpy.QtCore import QObject, QThread, Signal, Slot
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# ==============================================================================
# Worker Thread to Run Pipeline without Freezing the UI
# ==============================================================================
class BatchWorker(QObject):
    progress_changed = Signal(int, str)
    finished = Signal(int, list)  # (success_count, list_of_failed_files)

    def __init__(self, dirpath, settings, pipeline_runner):
        super().__init__()
        self.dirpath = dirpath
        self.settings = settings
        self.pipeline_runner = pipeline_runner
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        valid_exts = ('.tif', '.tiff', '.h5', '.hdf5', '.png')
        image_files = [
            f
            for f in os.listdir(self.dirpath)
            if f.lower().endswith(valid_exts) and not f.endswith('_session.h5')
        ]

        total = len(image_files)
        if total == 0:
            self.finished.emit(0, [])
            return

        successful_count = 0
        failed_files = []

        for idx, fname in enumerate(image_files):
            if self._is_cancelled:
                break

            self.progress_changed.emit(
                int((idx / total) * 100),
                f'Processing ({idx + 1}/{total}): {fname}',
            )

            full_path = os.path.join(self.dirpath, fname)
            base_name, _ = os.path.splitext(fname)
            out_h5_path = os.path.join(self.dirpath, f'{base_name}_session.h5')

            try:
                # Delegate execution back to parent/main widget runner
                self.pipeline_runner(full_path, out_h5_path, self.settings)
                successful_count += 1
            except Exception as e:
                failed_files.append((fname, str(e)))

        self.finished.emit(successful_count, failed_files)


# ==============================================================================
# Dedicated Batch Tab Widget UI
# ==============================================================================
class BatchTabWidget(QWidget):
    def __init__(self, parent_widget, parent=None):
        super().__init__(parent)
        self.main_widget = (
            parent_widget  # Reference to CellClassificationWidget
        )
        self.selected_dir = ''
        self.worker_thread = None
        self.worker = None

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)

        # ----------------------------------------------------------------------
        # Section 1: Input / Output Directory Selection
        # ----------------------------------------------------------------------
        group_io = QGroupBox('Target Directory')
        io_layout = QVBoxLayout()

        dir_btn_layout = QHBoxLayout()
        self.btn_select_dir = QPushButton('Select Folder...')
        self.btn_select_dir.clicked.connect(self._select_folder)
        self.lbl_selected_dir = QLabel('No folder selected.')
        self.lbl_selected_dir.setStyleSheet('color: gray; font-style: italic;')
        self.lbl_selected_dir.setWordWrap(True)

        dir_btn_layout.addWidget(self.btn_select_dir)
        dir_btn_layout.addWidget(self.lbl_selected_dir, stretch=1)

        self.lbl_file_count = QLabel('Found: 0 compatible image files')

        io_layout.addLayout(dir_btn_layout)
        io_layout.addWidget(self.lbl_file_count)
        group_io.setLayout(io_layout)

        # ----------------------------------------------------------------------
        # Section 2: Active Model & Settings Snapshot Summary
        # ----------------------------------------------------------------------
        group_summary = QGroupBox('Active Pipeline Models & Configurations')
        summary_layout = QVBoxLayout()

        self.predict_label = QLabel('<b>Model:</b> No model loaded')
        self.superpixel_label = QLabel(
            '<b>Superpixels:</b> Default parameters'
        )
        self.gasp_linkage_label = QLabel(
            '<b>GASP linkage:</b> Default linkage'
        )
        self.classifier_label = QLabel('<b>Classifier:</b> No model loaded')

        summary_layout.addWidget(self.predict_label)
        summary_layout.addWidget(self.superpixel_label)
        summary_layout.addWidget(self.gasp_linkage_label)
        summary_layout.addWidget(self.classifier_label)
        group_summary.setLayout(summary_layout)

        # ----------------------------------------------------------------------
        # Section 3: Batch Pipeline Stages Selection & Layer Routing
        # ----------------------------------------------------------------------
        group_stages = QGroupBox('Execution Stages & Layer Naming')
        stage_layout = QVBoxLayout()

        # Step 0: Interactive mode
        self.chk_render_viewer = QCheckBox(
            'Render results in Napari viewer during batch processing'
        )
        self.chk_render_viewer.setChecked(
            False
        )  # Default to Fast Headless mode
        self.chk_render_viewer.setToolTip(
            'If checked, each processed file will load and update Napari layers in real time.\n'
            'Disabling this speeds up batch execution significantly.'
        )
        self.layout().addWidget(self.chk_render_viewer)

        # Step 1: Prediction
        self.chk_predict = QCheckBox(
            '1. Run Model Prediction / Feature Extraction'
        )
        self.chk_predict.setChecked(True)

        # Step 2: Superpixels
        self.chk_superpixels = QCheckBox('2. Run Superpixel Segmentation')
        self.chk_superpixels.setChecked(True)

        # Step 3: GASP
        self.chk_gasp = QCheckBox('3. Run GASP Graph Partitioning')
        self.chk_gasp.setChecked(True)

        # Step 4: Morphology Post-processing
        self.chk_morphology = QCheckBox(
            '4. Run Morphological Filtering & Extraction'
        )
        self.chk_morphology.setChecked(True)
        self.chk_morphology.toggled.connect(self._toggle_morphology_options)

        # Morphology Layer Routing Options
        self.layout_morph_opts = QVBoxLayout()
        self.layout_morph_opts.setContentsMargins(20, 0, 0, 0)

        # Labels Layer selection
        lbl_layout = QHBoxLayout()
        lbl_layout.addWidget(QLabel('Labels Input:'))
        self.morphology_labels_layer = QComboBox()
        self.morphology_labels_layer.addItem('GASP labels', 'gasp_')
        self.morphology_labels_layer.addItem('Superpixel labels', 'sp_')
        self.morphology_labels_layer.addItem('Custom prefix...', 'custom')
        self.morphology_labels_layer.currentIndexChanged.connect(
            self._on_labels_combo_changed
        )

        self.txt_custom_labels_prefix = QLineEdit()
        self.txt_custom_labels_prefix.setPlaceholderText('e.g. seg_')
        self.txt_custom_labels_prefix.setEnabled(False)

        lbl_layout.addWidget(self.morphology_labels_layer)
        lbl_layout.addWidget(self.txt_custom_labels_prefix)
        self.layout_morph_opts.addLayout(lbl_layout)

        # Signal Layer selection
        sig_layout = QHBoxLayout()
        sig_layout.addWidget(QLabel('Signal Input:'))
        self.morphology_signal_layer = QComboBox()
        self.morphology_signal_layer.addItem('Prediction features', 'pred_')
        self.morphology_signal_layer.addItem('Custom prefix...', 'custom')
        self.morphology_signal_layer.currentIndexChanged.connect(
            self._on_signal_combo_changed
        )

        self.txt_custom_signal_prefix = QLineEdit()
        self.txt_custom_signal_prefix.setPlaceholderText('e.g. raw_')
        self.txt_custom_signal_prefix.setEnabled(False)

        sig_layout.addWidget(self.morphology_signal_layer)
        sig_layout.addWidget(self.txt_custom_signal_prefix)
        self.layout_morph_opts.addLayout(sig_layout)

        # Step 5: Cell Classification
        self.chk_classification = QCheckBox(
            '5. Run Random Forest Cell Classification'
        )
        self.chk_classification.setChecked(False)

        # Step 6: Export Table
        self.chk_export_xlsx = QCheckBox(
            '6. Export Morphology Table (.csv / .xlsx)'
        )
        self.chk_export_xlsx.setChecked(True)

        # Assemble Stage Group
        stage_layout.addWidget(self.chk_predict)
        stage_layout.addWidget(self.chk_superpixels)
        stage_layout.addWidget(self.chk_gasp)
        stage_layout.addWidget(self.chk_morphology)
        stage_layout.addLayout(self.layout_morph_opts)
        stage_layout.addWidget(self.chk_classification)
        stage_layout.addWidget(self.chk_export_xlsx)
        group_stages.setLayout(stage_layout)

        # ----------------------------------------------------------------------
        # Section 4: Execution & Progress Controls
        # ----------------------------------------------------------------------
        group_exec = QGroupBox('Run Control')
        exec_layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        self.btn_start_batch = QPushButton('Start Batch Processing')
        self.btn_start_batch.setStyleSheet(
            'font-weight: bold; padding: 8px; background-color: #2b5b84;'
        )
        self.btn_start_batch.setEnabled(False)
        self.btn_start_batch.clicked.connect(self._on_start_batch)

        self.btn_cancel = QPushButton('Cancel')
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel_batch)

        btn_layout.addWidget(self.btn_start_batch, stretch=2)
        btn_layout.addWidget(self.btn_cancel, stretch=1)

        self.status_label = QLabel('Select a folder to begin.')
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        exec_layout.addLayout(btn_layout)
        exec_layout.addWidget(self.status_label)
        exec_layout.addWidget(self.progress_bar)
        group_exec.setLayout(exec_layout)

        # Add all groups to main layout
        main_layout.addWidget(group_io)
        main_layout.addWidget(group_summary)
        main_layout.addWidget(group_stages)
        main_layout.addWidget(group_exec)
        main_layout.addStretch()

    # --------------------------------------------------------------------------
    # UI Event Handlers & Dynamic Controls
    # --------------------------------------------------------------------------
    def _select_folder(self):
        dirpath = QFileDialog.getExistingDirectory(
            self,
            'Select Directory for Batch Processing',
            '',
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if dirpath:
            self.selected_dir = dirpath
            self.lbl_selected_dir.setText(dirpath)
            self.lbl_selected_dir.setStyleSheet(
                'color: gray; font-style: normal;'
            )

            # Count valid files
            valid_exts = (
                '.tif',
                '.tiff',
                '.h5',
                '.hdf5',
                '.png',
                '.jpg',
                '.jpeg',
            )
            files = [
                f
                for f in os.listdir(dirpath)
                if f.lower().endswith(valid_exts)
                and not f.endswith('_session.h5')
            ]
            count = len(files)
            self.lbl_file_count.setText(
                f'Found: {count} compatible image file(s)'
            )
            self.btn_start_batch.setEnabled(count > 0)
            self.status_label.setText('Ready to start batch process.')

    def _toggle_morphology_options(self, enabled):
        self.morphology_labels_layer.setEnabled(enabled)
        self.morphology_signal_layer.setEnabled(enabled)
        if not enabled:
            self.txt_custom_labels_prefix.setEnabled(False)
            self.txt_custom_signal_prefix.setEnabled(False)
        else:
            self._on_labels_combo_changed()
            self._on_signal_combo_changed()

    def _on_labels_combo_changed(self):
        is_custom = self.morphology_labels_layer.currentData() == 'custom'
        self.txt_custom_labels_prefix.setEnabled(
            is_custom and self.chk_morphology.isChecked()
        )

    def _on_signal_combo_changed(self):
        is_custom = self.morphology_signal_layer.currentData() == 'custom'
        self.txt_custom_signal_prefix.setEnabled(
            is_custom and self.chk_morphology.isChecked()
        )

    def update_model_summary(self):
        """Call this whenever tabs are switched or new models/settings are loaded."""
        if hasattr(self.main_widget, 'get_active_models_info'):
            info = self.main_widget.get_active_models_info()
            self.predict_label.setText(
                f'<b>Model:</b> {info.get("model_name", "None")}'
            )
            self.superpixel_label.setText(
                f'<b>Superpixels:</b> {info.get("sp_algo", "Default")}'
            )
            self.gasp_linkage_label.setText(
                f'<b>GASP linkage:</b> {info.get("gasp_linkage", "Default")}'
            )
            self.classifier_label.setText(
                f'<b>Classifier:</b> {info.get("classifier_type", "None")}'
            )

    # --------------------------------------------------------------------------
    # Settings Assembly & Execution
    # --------------------------------------------------------------------------
    def _get_batch_pipeline_settings(self):
        """Collect current checkbox toggles and custom text inputs."""
        # Labels prefix logic
        lbl_data = self.morphology_labels_layer.currentData()
        labels_prefix = (
            self.txt_custom_labels_prefix.text().strip()
            if lbl_data == 'custom'
            else lbl_data
        )

        # Signal prefix logic
        sig_data = self.morphology_signal_layer.currentData()
        signal_prefix = (
            self.txt_custom_signal_prefix.text().strip()
            if sig_data == 'custom'
            else sig_data
        )

        return {
            'render_viewer': self.chk_render_viewer.isChecked(),
            'predict': self.chk_predict.isChecked(),
            'superpixels': self.chk_superpixels.isChecked(),
            'gasp': self.chk_gasp.isChecked(),
            'morphology': self.chk_morphology.isChecked(),
            'morphology_labels_prefix': labels_prefix,
            'morphology_signal_prefix': signal_prefix,
            'classification': self.chk_classification.isChecked(),
            'export_xlsx': self.chk_export_xlsx.isChecked(),
        }

    def _get_settings(self):
        """Bundle both batch UI toggles and global model/segmentation parameters."""
        batch_settings = self._get_batch_pipeline_settings()

        # Pull algorithm parameters (model weights, thresholds, scale factors) from main widget
        global_settings = {}
        if hasattr(self.main_widget, '_get_all_active_parameters'):
            global_settings = self.main_widget._get_all_active_parameters()

        return {**batch_settings, 'global': global_settings}

    def _on_start_batch(self):
        if not self.selected_dir:
            return

        settings = self._get_settings()

        # Lock controls during execution
        self.btn_select_dir.setEnabled(False)
        self.btn_start_batch.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        # Setup worker thread
        self.worker_thread = QThread()
        self.worker = BatchWorker(
            self.selected_dir,
            settings,
            self.main_widget._run_batch_pipeline,
        )
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress_changed.connect(self._on_progress_update)
        self.worker.finished.connect(self._on_batch_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    @Slot(int, str)
    def _on_progress_update(self, percent, msg):
        self.progress_bar.setValue(percent)
        self.status_label.setText(msg)

    def _on_cancel_batch(self):
        if self.worker:
            self.worker.cancel()
            self.status_label.setText(
                'Cancelling after current file finishes...'
            )

    @Slot(int, list)
    def _on_batch_finished(self, success_count, failed_files):
        self.btn_select_dir.setEnabled(True)
        self.btn_start_batch.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setValue(100)

        msg = f'Batch processing finished.\nSuccessfully processed: {success_count} file(s).'
        if failed_files:
            msg += f'\n\nFailed on {len(failed_files)} file(s):\n' + '\n'.join(
                f'- {f}: {err}' for f, err in failed_files[:5]
            )
        QMessageBox.information(self, 'Batch Execution Complete', msg)
        self.status_label.setText('Ready for next batch.')
