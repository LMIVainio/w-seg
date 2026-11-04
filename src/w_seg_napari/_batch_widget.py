import threading
from pathlib import Path

from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from w_seg_napari._batch_io import list_input_files, output_dir
from w_seg_napari import _session_io as sio
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
        self.notes = []
        self.session = None  # {'path': Path | None, 'base': str} once opened

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
        self.seed_label = QLabel('')
        self.seed_label.setWordWrap(True)
        self.seed_label.setVisible(False)
        self.gasp_linkage_label = QLabel(
            '<b>GASP linkage:</b> Default linkage'
        )
        self.classifier_label = QLabel('<b>Classifier:</b> No model loaded')
        for lbl in (
            self.predict_label,
            self.superpixel_label,
            self.seed_label,
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
        self.morphology_labels.addItem(
            'Corrected labels (saved sessions)', 'corrected'
        )
        self.morphology_labels.setToolTip(
            'Corrected: the labels you saved with "Save session"; an image '
            'without them is measured from its GASP labels (and reported).'
        )
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

        # --- Section: sessions --------------------------------------------------
        group_sess = QGroupBox('Sessions (open, correct, save)')
        sess_layout = QVBoxLayout()
        sess_btns = QHBoxLayout()
        self.btn_open_session = QPushButton('Open Session...')
        self.btn_open_session.setToolTip(
            'Opens a *_session.h5 file in the viewer: image, predictions, '
            'labels and an editable copy of the labels.\n'
            'Old "Export Napari Layers" files open too (layers only).'
        )
        self.btn_open_session.clicked.connect(self._on_open_session)
        self.btn_save_session = QPushButton('Save Session')
        self.btn_save_session.clicked.connect(self._on_save_session)
        sess_btns.addWidget(self.btn_open_session)
        sess_btns.addWidget(self.btn_save_session)
        self.lbl_session = QLabel('No session open.')
        self.lbl_session.setWordWrap(True)
        self.lbl_session.setStyleSheet('color: gray; font-style: italic;')
        lab_row = QHBoxLayout()
        lab_row.addWidget(QLabel('Corrected labels:'))
        self.combo_save_labels = QComboBox()
        self.combo_save_labels.setToolTip(
            'The Labels layer saved as the corrected cell labels. The '
            'original batch labels stay in the file.'
        )
        lab_row.addWidget(self.combo_save_labels, stretch=1)
        self.list_extra = QListWidget()
        self.list_extra.setMaximumHeight(110)
        self.list_extra.setToolTip(
            'Other layers to keep in the session (class layers, shapes, ...).'
        )
        sess_layout.addLayout(sess_btns)
        sess_layout.addWidget(self.lbl_session)
        sess_layout.addLayout(lab_row)
        sess_layout.addWidget(QLabel('Also keep these layers:'))
        sess_layout.addWidget(self.list_extra)
        group_sess.setLayout(sess_layout)
        self.group_sess = group_sess

        main_layout.addWidget(group_io)
        main_layout.addWidget(group_sess)
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
            seed = info.get('seed') or {}
            if seed.get('text'):
                color = '#c0392b' if seed.get('problem') else '#2e7d32'
                note = (
                    f'<br><span style="color:{color}">{seed["problem"]}</span>'
                    if seed.get('problem')
                    else ''
                )
                self.seed_label.setText(
                    f'<b>Seeds</b> (from the Superpixel tab): '
                    f'{seed["text"]}{note}'
                )
                self.seed_label.setVisible(True)
            else:
                self.seed_label.setVisible(False)
            self.gasp_linkage_label.setText(
                f'<b>GASP linkage:</b> {info.get("gasp_linkage", "Default")}'
            )
            self.classifier_label.setText(
                f'<b>Classifier:</b> {info.get("classifier_type", "None")}'
            )

    # --------------------------------------------------------------------------
    # Sessions
    # --------------------------------------------------------------------------
    @property
    def viewer(self):
        return self.main_widget.viewer

    def refresh_layer_choices(self):
        """Refills the layer choices of the Sessions box (the batch itself uses files)."""
        v = self.viewer
        labels = [la.name for la in v.layers if la._type_string == 'labels']
        keep = self.combo_save_labels.currentText()
        self.combo_save_labels.blockSignals(True)
        self.combo_save_labels.clear()
        self.combo_save_labels.addItem('(do not save labels)')
        self.combo_save_labels.addItems(labels)
        pick = keep if keep in labels else self._default_label_layer(labels)
        if pick:
            self.combo_save_labels.setCurrentText(pick)
        self.combo_save_labels.blockSignals(False)

        ticked = {
            self.list_extra.item(i).text()
            for i in range(self.list_extra.count())
            if self.list_extra.item(i).checkState() == Qt.Checked
        }
        known = {
            self.list_extra.item(i).text()
            for i in range(self.list_extra.count())
        }
        self.list_extra.clear()
        for la in v.layers:
            if la._type_string not in ('image', 'labels', 'shapes'):
                continue
            item = QListWidgetItem(la.name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            on = (
                la.name in ticked
                if la.name in known
                else not self._is_batch_layer(la.name)
                and la.name != self.combo_save_labels.currentText()
            )
            item.setCheckState(Qt.Checked if on else Qt.Unchecked)
            self.list_extra.addItem(item)

    def _default_label_layer(self, labels):
        if self.session:
            name = f'corrected_{self.session["base"]}'
            if name in labels:
                return name
        return None

    @staticmethod
    def _is_batch_layer(name):
        return name.startswith(
            ('pred_', 'sp_', 'gasp_cells_', 'geo_', '[Predicted Classes]')
        ) or name.startswith(('aff_', 'affinity'))

    def _on_open_session(self):
        start = (
            str(output_dir(self.selected_dir)) if self.selected_dir else ''
        )
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open session', start, 'HDF5 sessions (*.h5 *.hdf5)'
        )
        if not path:
            return
        try:
            self._open_session(Path(path))
        except Exception as e:  # unreadable file, missing h5py, ...
            QMessageBox.warning(self, 'Cannot open session', f'{type(e).__name__}: {e}')

    def _open_session(self, path: Path):
        v = self.viewer
        kind = sio.session_kind(path)
        if kind == 'layers':
            names = sio.add_layers_to_viewer(v, sio.read_layer_file(path))
            self.session = None
            self.lbl_session.setText(
                f'Opened {path.name}: {len(names)} layer(s). This is an old '
                'layers-only file; "Save Session" asks for a new file.'
            )
            self.refresh_layer_choices()
            return
        if kind != 'session':
            raise ValueError('This file is not a w-seg session.')

        results, _store, info = sio.session_results(path)
        base = info['base']
        if 'raw' in results and 'pred_sem' in results:
            self.main_widget.render_batch_results(
                info['source_file'], results
            )
        elif 'raw' in results:
            v.add_image(results['raw'], name=base)
        if info['editable'] is not None:
            name = f'corrected_{base}'
            if name in v.layers:
                v.layers[name].data = info['editable']
            else:
                v.add_labels(info['editable'], name=name, opacity=0.8)
        names = {la.name for la in v.layers}
        extras = [p for p in info['extras'] if p['name'] not in names]
        sio.add_layers_to_viewer(v, extras)
        self.session = {'path': path, 'base': base}
        origin = {
            'corrected': 'your saved corrections',
            'gasp': 'a copy of the GASP labels',
            'superpixels': 'a copy of the superpixel labels',
        }.get(info['origin'], 'no labels')
        self.lbl_session.setText(
            f'Session: {path.name}. Edit "corrected_{base}" ({origin}), '
            'then Save Session.'
        )
        self.lbl_session.setStyleSheet('font-style: normal;')
        self.refresh_layer_choices()

    def _on_save_session(self):
        v = self.viewer
        path = self.session['path'] if self.session else None
        if path is None:
            start = (
                str(output_dir(self.selected_dir)) if self.selected_dir else ''
            )
            chosen, _ = QFileDialog.getSaveFileName(
                self, 'Save session', start, 'HDF5 sessions (*.h5)'
            )
            if not chosen:
                return
            path = Path(chosen)
        corrected, layer_name = None, None
        label_name = self.combo_save_labels.currentText()
        if label_name in v.layers:
            corrected = v.layers[label_name].data
            layer_name = label_name
        payloads = []
        for i in range(self.list_extra.count()):
            item = self.list_extra.item(i)
            name = item.text()
            if (
                item.checkState() == Qt.Checked
                and name in v.layers
                and name != label_name
            ):
                payload = sio.layer_to_payload(v.layers[name], order=i)
                if payload is not None:
                    payloads.append(payload)
        try:
            sio.save_edits(
                path,
                corrected=corrected,
                corrected_layer=layer_name,
                extras=payloads,
            )
        except Exception as e:
            QMessageBox.warning(self, 'Cannot save session', f'{type(e).__name__}: {e}')
            return
        if self.session is None:
            self.session = {'path': path, 'base': Path(path).stem.removesuffix('_session')}
        self.lbl_session.setText(
            f'Saved to {path.name}: '
            + (f'labels "{layer_name}"' if layer_name else 'no labels')
            + f' and {len(payloads)} other layer(s).'
        )

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
        self.group_sess.setEnabled(not running)
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
        self.notes = []
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
        elif kind == 'note':
            self.notes.append(event[1])
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
        if self.notes:
            msg += '\n\nNotes:\n' + '\n'.join(
                f'- {n}' for n in self.notes[:8]
            )
            if len(self.notes) > 8:
                msg += f'\n... and {len(self.notes) - 8} more'
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
