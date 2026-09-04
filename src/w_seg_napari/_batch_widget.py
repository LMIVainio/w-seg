import json
import os
import h5py
import numpy as np
from qtpy.QtCore import QObject, QThread, Qt, Signal, Slot
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QLabel,
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
            if f.lower().endswith(valid_exts)
            and not f.endswith('_session.h5')
        ]

        total = len(image_files)
        successful_count = 0
        failed_files = []

        for idx, fname in enumerate(image_files):
            if self._is_cancelled:
                break

            self.progress_changed.emit(
                int((idx / total) * 100), f'Processing ({idx + 1}/{total}): {fname}'
            )

            full_path = os.path.join(self.dirpath, fname)
            base_name, _ = os.path.splitext(fname)
            out_h5_path = os.path.join(self.dirpath, f'{base_name}_session.h5')

            try:
                # Delegate to the widget pipeline executor
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
        self.main_widget = parent_widget  # Reference to CellClassificationWidget
        self.worker_thread = None
        self.worker = None

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # ----------------------------------------------------------------------
        # Pipeline Stage Controls
        # ----------------------------------------------------------------------
        group_stages = QGroupBox('Batch Pipeline Stages')
        stage_layout = QVBoxLayout()

        self.chk_predict = QCheckBox('Include Model Prediction / Features')
        self.chk_predict.setChecked(True)
        self.predict_label = QLabel('Model not selected')

        self.chk_superpixels = QCheckBox('Run Superpixel Segmentation')
        self.chk_superpixels.setChecked(True)
        self.superpixel_label = QLabel('Algorithm not selected')

        self.chk_gasp = QCheckBox('Run GASP Graph Partitioning')
        self.chk_gasp.setChecked(True)

        self.chk_morphology = QCheckBox('Run Morphological Filtering / Post-proc')
        self.chk_morphology.setChecked(True)
        self.morphology_labels_layer = QComboBox()
        self.morphology_labels_layer.addItem('GASP lables', 'gasp_')
        self.morphology_labels_layer.addItem('Superpixels labels', 'sp_')

        self.chk_classification = QCheckBox(
            'Run Random Forest Cell Classification'
        )
        self.chk_classification.setChecked(False)

        self.chk_export_xlsx = QCheckBox('Export Morphology Table')
        self.chk_export_xlsx.setChecked(True)

        stage_layout.addWidget(self.chk_predict)
        stage_layout.addWidget(self.chk_superpixels)
        stage_layout.addWidget(self.chk_gasp)
        stage_layout.addWidget(self.chk_morphology)
        stage_layout.addWidget(self.chk_classification)
        stage_layout.addWidget(self.chk_export_xlsx)
        group_stages.setLayout(stage_layout)

        # ----------------------------------------------------------------------
        # Execution & Progress Controls
        # ----------------------------------------------------------------------
        self.btn_select_
        self.btn_select_dir = QPushButton('Select Folder')
        self.btn_select_dir.clicked.connect(self._select_folder)

        self.folder_label = QLabel("Folder not selected")

        self.btn_start_batch = QPushButton('Start Batch')
        self.btn_start_batch.setStyleSheet(
            'font-weight: bold; padding: 8px;'
        )
        self.btn_select_dir.clicked.connect(self._on_start_batch)

        self.btn_cancel = QPushButton('Cancel Batch')
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel_batch)

        self.status_label = QLabel('Ready for batch processing.')
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        layout.addWidget(group_stages)
        layout.addWidget(self.btn_select_dir)
        layout.addWidget(self.btn_cancel)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addStretch()

    def _get_settings(self):
        """Collect current checkbox toggles."""
        return {
            'predict': self.chk_predict.isChecked(),
            'superpixels': self.chk_superpixels.isChecked(),
            'gasp': self.chk_gasp.isChecked(),
            'morphology': self.chk_morphology.isChecked(),
            'classification': self.chk_classification.isChecked(),
        }

    def _on_start_batch(self):
        dirpath = QFileDialog.getExistingDirectory(
            self,
            'Select Directory for Batch Processing',
            '',
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if not dirpath:
            return

        settings = self._get_settings()

        # Disable controls during run
        self.btn_select_dir.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        # Setup worker QThread
        self.worker_thread = QThread()
        self.worker = BatchWorker(
            dirpath, settings, self.main_widget._run_batch_single_file
        )
        self.worker.moveToThread(self.worker_thread)

        # Wire Signals
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
            self.status_label.setText('Cancelling after current file...')

    @Slot(int, list)
    def _on_batch_finished(self, success_count, failed_files):
        self.btn_select_dir.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setValue(100)

        msg = f'Batch processing completed.\nSuccessfully processed: {success_count} file(s).'
        if failed_files:
            msg += f'\n\nFailed on {len(failed_files)} file(s):\n' + '\n'.join(
                f'- {f}: {err}' for f, err in failed_files[:5]
            )
        QMessageBox.information(self, 'Batch Complete', msg)
        self.status_label.setText('Ready for batch processing.')