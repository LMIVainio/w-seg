import threading
from pathlib import Path

from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from w_seg_napari._batch_io import list_input_files, output_dir
from w_seg_napari._batch_pipeline import run_batch


class BatchTabWidget(QWidget):
    """Batch tab: runs the pipeline of tabs 1 and 4 on every image of a folder.

    The batch runs in a background thread, so napari stays responsive. While it
    runs, the other tabs are locked (their settings were copied when the batch
    started) and this tab shows a progress bar and a cancel button.
    """

    def __init__(self, parent_widget, parent=None):
        super().__init__(parent)
        self.main_widget = parent_widget  # the WSegMasterWidget
        self.selected_dir = ''
        self.files = []
        self.worker = None
        self.cancel_event = threading.Event()
        self.failed = []

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)

        # --- Section 1: folder --------------------------------------------
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
        self.lbl_output = QLabel('')
        self.lbl_output.setWordWrap(True)

        io_layout.addLayout(dir_btn_layout)
        io_layout.addWidget(self.lbl_file_count)
        io_layout.addWidget(self.lbl_output)
        group_io.setLayout(io_layout)

        # --- Section 2: active models ---------------------------------------
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
        for lbl in (
            self.predict_label,
            self.superpixel_label,
            self.gasp_linkage_label,
            self.classifier_label,
        ):
            summary_layout.addWidget(lbl)
        group_summary.setLayout(summary_layout)

        # --- Section 3: stages ------------------------------------------------
        group_stages = QGroupBox('Execution Stages')
        stage_layout = QVBoxLayout()

        self.chk_render_viewer = QCheckBox(
            'Show results in the viewer while processing'
        )
        self.chk_render_viewer.setChecked(False)
        self.chk_render_viewer.setToolTip(
            'Adds the layers of each processed image to the viewer.\n'
            'Off is faster and uses less memory; results are saved to disk either way.'
        )

        self.chk_predict = QCheckBox(
            '1. Run Model Prediction (walls + affinities)'
        )
        self.chk_predict.setChecked(True)
        self.chk_superpixels = QCheckBox('2. Run Superpixel Segmentation')
        self.chk_superpixels.setChecked(True)
        self.chk_gasp = QCheckBox('3. Run GASP Graph Partitioning')
        self.chk_gasp.setChecked(True)
        self.chk_morphology = QCheckBox(
            '4. Run Morphology & Cell Wall Measurements'
        )
        self.chk_morphology.setChecked(True)
        self.chk_morphology.toggled.connect(self._toggle_morphology_options)

        morph_opts = QVBoxLayout()
        morph_opts.setContentsMargins(20, 0, 0, 0)
        lbl_layout = QHBoxLayout()
        lbl_layout.addWidget(QLabel('Labels Input:'))
        self.morphology_labels = QComboBox()
        self.morphology_labels.addItem('GASP labels', 'gasp')
        self.morphology_labels.addItem('Superpixel labels', 'sp')
        lbl_layout.addWidget(self.morphology_labels)
        morph_opts.addLayout(lbl_layout)
        sig_layout = QHBoxLayout()
        sig_layout.addWidget(QLabel('Signal Input:'))
        self.morphology_signal = QComboBox()
        self.morphology_signal.addItem('Wall prediction', 'prediction')
        self.morphology_signal.addItem('Raw image (gray)', 'image')
        sig_layout.addWidget(self.morphology_signal)
        morph_opts.addLayout(sig_layout)

        self.chk_classification = QCheckBox(
            '5. Run Cell Classification (classifier from tab 4)'
        )
        self.chk_classification.setChecked(False)
        self.chk_export_table = QCheckBox('6. Export Morphology Table (.csv)')
        self.chk_export_table.setChecked(True)

        stage_layout.addWidget(self.chk_render_viewer)
        stage_layout.addWidget(self.chk_predict)
        stage_layout.addWidget(self.chk_superpixels)
        stage_layout.addWidget(self.chk_gasp)
        stage_layout.addWidget(self.chk_morphology)
        stage_layout.addLayout(morph_opts)
        stage_layout.addWidget(self.chk_classification)
        stage_layout.addWidget(self.chk_export_table)
        group_stages.setLayout(stage_layout)
        self.group_stages = group_stages

        # --- Section 4: run control -------------------------------------------
        group_exec = QGroupBox('Run Control')
        exec_layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        self.btn_start_batch = QPushButton('Start Batch Processing')
        self.btn_start_batch.setStyleSheet(
            'font-weight: bold; background-color: #2b5b84; color: white; padding: 8px;'
        )
        self.btn_start_batch.setEnabled(False)
        self.btn_start_batch.clicked.connect(self._on_start_batch)

        self.btn_cancel = QPushButton('Cancel')
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel_batch)

        btn_layout.addWidget(self.btn_start_batch, stretch=2)
        btn_layout.addWidget(self.btn_cancel, stretch=1)

        self.status_label = QLabel('Select a folder to begin.')
        self.status_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        exec_layout.addLayout(btn_layout)
        exec_layout.addWidget(self.status_label)
        exec_layout.addWidget(self.progress_bar)
        group_exec.setLayout(exec_layout)

        main_layout.addWidget(group_io)
        main_layout.addWidget(group_summary)
        main_layout.addWidget(group_stages)
        main_layout.addWidget(group_exec)
        main_layout.addStretch()

    # --------------------------------------------------------------------------
    # UI event handlers
    # --------------------------------------------------------------------------
    def _select_folder(self):
        dirpath = QFileDialog.getExistingDirectory(
            self,
            'Select Directory for Batch Processing',
            '',
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if not dirpath:
            return
        self.selected_dir = dirpath
        self.lbl_selected_dir.setText(dirpath)
        self.lbl_selected_dir.setStyleSheet('color: gray; font-style: normal;')

        self.files = list_input_files(dirpath)
        self.lbl_file_count.setText(
            f'Found: {len(self.files)} compatible image file(s)'
        )
        self.lbl_output.setText(f'Results go to: {output_dir(dirpath)}')
        self.btn_start_batch.setEnabled(len(self.files) > 0)
        self.status_label.setText('Ready to start batch process.')

    def _toggle_morphology_options(self, enabled):
        self.morphology_labels.setEnabled(enabled)
        self.morphology_signal.setEnabled(enabled)

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

    def refresh_layer_choices(self):
        """Nothing to refresh: the batch works on files, not on viewer layers."""

    # --------------------------------------------------------------------------
    # Running
    # --------------------------------------------------------------------------
    def _batch_ui_settings(self) -> dict:
        return {
            'stages': {
                'predict': self.chk_predict.isChecked(),
                'superpixels': self.chk_superpixels.isChecked(),
                'gasp': self.chk_gasp.isChecked(),
                'morphology': self.chk_morphology.isChecked(),
                'classification': self.chk_classification.isChecked()
                and self.chk_morphology.isChecked(),
                'export': self.chk_export_table.isChecked()
                and self.chk_morphology.isChecked(),
            },
            'render': self.chk_render_viewer.isChecked(),
            'morph_labels': self.morphology_labels.currentData(),
            'morph_signal': self.morphology_signal.currentData(),
        }

    def _set_running(self, running: bool):
        """Locks everything except the cancel button while a batch runs."""
        self.btn_select_dir.setEnabled(not running)
        self.group_stages.setEnabled(not running)
        self.btn_start_batch.setEnabled(not running and bool(self.files))
        self.btn_cancel.setEnabled(running)
        if hasattr(self.main_widget, 'set_batch_running'):
            self.main_widget.set_batch_running(running)

    def _on_start_batch(self):
        if not self.selected_dir or not self.files or self.worker is not None:
            return

        try:
            # Copies the settings of tabs 1 and 4 on the main thread
            settings = self.main_widget.collect_batch_settings(
                self._batch_ui_settings()
            )
        except ValueError as e:
            QMessageBox.warning(self, 'Cannot start batch', str(e))
            return

        self.cancel_event.clear()
        self.failed = []
        self.progress_bar.setValue(0)
        self._set_running(True)

        files, dirpath = list(self.files), self.selected_dir
        is_cancelled = self.cancel_event.is_set

        @thread_worker
        def _work():
            return (
                yield from run_batch(files, dirpath, settings, is_cancelled)
            )

        self.worker = _work()
        self.worker.yielded.connect(self._on_event)
        self.worker.returned.connect(self._on_returned)
        self.worker.errored.connect(self._on_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_event(self, event):
        kind = event[0]
        if kind == 'progress':
            _, percent, message = event
            self.progress_bar.setValue(int(percent))
            self.status_label.setText(message)
        elif kind == 'render':
            _, name, results = event
            self.main_widget.render_batch_results(name, results)
        elif kind == 'failed':
            _, name, message = event
            print(f'[w-seg] Batch: {name} failed: {message}')

    def _on_cancel_batch(self):
        self.cancel_event.set()
        self.btn_cancel.setEnabled(False)
        self.status_label.setText(
            'Cancelling... (the current step finishes first)'
        )

    def _on_returned(self, result):
        n_ok, failed = result if result else (0, [])
        self.failed = failed
        cancelled = self.cancel_event.is_set()
        msg = f'{"Batch cancelled" if cancelled else "Batch finished"}.\nProcessed: {n_ok} of {len(self.files)} file(s).'
        if failed:
            msg += f'\n\nFailed on {len(failed)} file(s):\n' + '\n'.join(
                f'- {f}: {err}' for f, err in failed[:5]
            )
            if len(failed) > 5:
                msg += f'\n... and {len(failed) - 5} more (see the console)'
        QMessageBox.information(self, 'Batch Execution', msg)
        self.status_label.setText(
            f'Results are in {Path(self.selected_dir) / "wseg_batch"}'
        )

    def _on_error(self, exc):
        print(f'[w-seg] Batch stopped by an error: {exc}')
        QMessageBox.critical(
            self, 'Batch stopped', f'{type(exc).__name__}: {exc}'
        )
        self.status_label.setText('Batch stopped by an error.')

    def _on_worker_finished(self):
        self.worker = None
        self._set_running(False)
