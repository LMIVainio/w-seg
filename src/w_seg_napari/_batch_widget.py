import threading
from pathlib import Path

import numpy as np
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

from w_seg_napari._batch_io import (
    find_session_dir,
    list_input_files,
    list_session_names,
    output_dir,
)
from w_seg_napari import _session_io as sio
from w_seg_napari._batch_settings import (
    FILENAME as SETTINGS_FILENAME,
    apply_loaded,
    describe,
    load_settings,
    model_differences,
    settings_document,
)
from w_seg_napari._batch_pipeline import is_done, run_batch
from w_seg_napari._session_io import CORRECTED, has_dataset


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
        self.n_skipped = 0
        self._rendered = []  # layers the batch added for the last image
        self.settings_file = None  # a settings file chosen with "Load..."
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

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel('Input:'))
        self.combo_input = QComboBox()
        self.combo_input.addItem('Image files', 'images')
        self.combo_input.addItem('Sessions (*_session.h5)', 'sessions')
        self.combo_input.setToolTip(
            'Sessions: process the sessions of an earlier run, or sessions '
            'you saved by hand. The image is taken from the session, so the '
            'original image files are not needed.'
        )
        self.combo_input.currentIndexChanged.connect(self._refresh_files)
        mode_row.addWidget(self.combo_input, stretch=1)

        self.lbl_file_count = QLabel('Found: 0 compatible image files')
        self.lbl_output = QLabel('')
        self.lbl_output.setWordWrap(True)

        io_layout.addLayout(dir_btn_layout)
        io_layout.addLayout(mode_row)
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

        self.chk_only_latest = QCheckBox(
            'Keep only the latest image in the viewer'
        )
        self.chk_only_latest.setChecked(True)
        self.chk_only_latest.setToolTip(
            'When an image is finished, its layers replace those of the '
            'previous image. The previous image stays visible while the '
            'next one is processed.\nLayers you made yourself are never '
            'removed.'
        )
        self.chk_render_viewer.toggled.connect(self.chk_only_latest.setEnabled)
        self.chk_only_latest.setEnabled(False)
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
        self.chk_errors = QCheckBox('Flag likely segmentation errors')
        self.chk_errors.setToolTip(
            'Adds p_error and error_type columns and an error_<image> layer '
            '(needs an error model trained with scripts/error_model.py '
            'from your corrected sessions).'
        )
        self.chk_errors.setChecked(False)
        self.chk_errors.toggled.connect(self._toggle_error_options)
        err_row = QHBoxLayout()
        err_row.setContentsMargins(20, 0, 0, 0)
        self.btn_error_model = QPushButton('Error model...')
        self.btn_error_model.clicked.connect(self._select_error_model)
        self.lbl_error_model = QLabel('No model selected.')
        self.lbl_error_model.setWordWrap(True)
        self.lbl_error_model.setStyleSheet('color: gray; font-style: italic;')
        err_row.addWidget(self.btn_error_model)
        err_row.addWidget(self.lbl_error_model, stretch=1)
        self.error_model_path = ''
        self._toggle_error_options(False)
        self.chk_export_table = QCheckBox('6. Export Morphology Table (.csv)')
        self.chk_export_table.setChecked(True)
        for chk in (
            self.chk_predict,
            self.chk_superpixels,
            self.chk_gasp,
            self.chk_morphology,
            self.chk_classification,
            self.chk_export_table,
        ):
            chk.toggled.connect(self._update_done_count)

        self.chk_skip_done = QCheckBox('Skip images that are already processed')
        self.chk_skip_done.setToolTip(
            'Lets you continue a large batch later: an image is skipped when '
            'its session (and table) from an earlier run already hold the '
            'results of the selected stages.\n'
            'Switch this off after changing the model or the settings, '
            'otherwise old results are kept.'
        )
        self.chk_skip_done.toggled.connect(self._update_done_count)
        self.lbl_done = QLabel('')
        self.lbl_done.setContentsMargins(20, 0, 0, 0)

        stage_layout.addWidget(self.chk_render_viewer)
        stage_layout.addWidget(self.chk_only_latest)
        stage_layout.addWidget(self.chk_skip_done)
        stage_layout.addWidget(self.lbl_done)
        stage_layout.addWidget(self.chk_predict)
        stage_layout.addWidget(self.chk_superpixels)
        stage_layout.addWidget(self.chk_gasp)
        stage_layout.addWidget(self.chk_morphology)
        stage_layout.addLayout(morph_opts)
        stage_layout.addWidget(self.chk_classification)
        stage_layout.addWidget(self.chk_errors)
        stage_layout.addLayout(err_row)
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

        self.chk_save_settings = QCheckBox(
            f'Save the settings ({SETTINGS_FILENAME}) with the results, '
            'before processing'
        )
        self.chk_save_settings.setChecked(True)
        self.chk_save_settings.setToolTip(
            'Written first, before the first image. An older file that '
            'differs is kept as batch_settings_<date>.json.'
        )
        self.chk_use_saved = QCheckBox(
            'Use the saved settings of the results folder, if it has them'
        )
        self.chk_use_saved.setToolTip(
            'Continue a batch with exactly the settings of the earlier run '
            '(Superpixel, GASP and Morphology tabs, patch size, stages).\n'
            'The loaded model and the cell classifier stay as they are.'
        )
        self.chk_use_saved.toggled.connect(self._update_settings_label)
        set_row = QHBoxLayout()
        self.btn_load_settings = QPushButton('Load settings file...')
        self.btn_load_settings.clicked.connect(self._on_load_settings_file)
        self.btn_clear_settings = QPushButton('Clear')
        self.btn_clear_settings.setEnabled(False)
        self.btn_clear_settings.clicked.connect(self._on_clear_settings_file)
        set_row.addWidget(self.btn_load_settings)
        set_row.addWidget(self.btn_clear_settings)
        self.lbl_settings = QLabel('Settings: the current tabs.')
        self.lbl_settings.setWordWrap(True)
        self.lbl_settings.setStyleSheet('color: gray;')

        exec_layout.addWidget(self.chk_save_settings)
        exec_layout.addWidget(self.chk_use_saved)
        exec_layout.addLayout(set_row)
        exec_layout.addWidget(self.lbl_settings)
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
        img_row = QHBoxLayout()
        img_row.addWidget(QLabel('Image (new session):'))
        self.combo_save_image = QComboBox()
        self.combo_save_image.setToolTip(
            'Only for a session made by hand, without a batch run behind it: '
            'the image layer saved as the session image, so the batch can '
            'open the session later (Input: Sessions).'
        )
        img_row.addWidget(self.combo_save_image, stretch=1)
        self.list_extra = QListWidget()
        self.list_extra.setMaximumHeight(110)
        self.list_extra.setToolTip(
            'Other layers to keep in the session (class layers, shapes, ...).'
        )
        sess_layout.addLayout(sess_btns)
        sess_layout.addWidget(self.lbl_session)
        sess_layout.addLayout(lab_row)
        sess_layout.addLayout(img_row)
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

        self._refresh_files()

    def _sessions_mode(self):
        return self.combo_input.currentData() == 'sessions'

    def _out_dir(self):
        """Where the sessions and tables of this folder are (or will be)."""
        if self._sessions_mode():
            return find_session_dir(self.selected_dir)
        return output_dir(self.selected_dir)

    def _refresh_files(self, *_):
        if not self.selected_dir:
            return
        if self._sessions_mode():
            self.files = list_session_names(self.selected_dir)
            self.lbl_file_count.setText(
                f'Found: {len(self.files)} session file(s)'
            )
        else:
            self.files = list_input_files(self.selected_dir)
            self.lbl_file_count.setText(
                f'Found: {len(self.files)} compatible image file(s)'
            )
        self.lbl_output.setText(f'Results go to: {self._out_dir()}')
        self.btn_start_batch.setEnabled(len(self.files) > 0)
        self.status_label.setText('Ready to start batch process.')
        self._update_done_count()
        self._update_settings_label()

    # --------------------------------------------------------------------------
    # Settings file
    # --------------------------------------------------------------------------
    def _folder_settings_path(self):
        if not self.selected_dir:
            return None
        path = self._out_dir() / SETTINGS_FILENAME
        return path if path.is_file() else None

    def _settings_source(self):
        """(path or None, is it the results folder's own file)."""
        if self.settings_file:
            return Path(self.settings_file), False
        folder = self._folder_settings_path()
        if self.chk_use_saved.isChecked() and folder:
            return folder, True
        return None, False

    def _update_settings_label(self, *_):
        src, _own = self._settings_source()
        if src is None:
            note = ''
            if self.chk_use_saved.isChecked():
                note = ' The results folder has no saved settings yet.'
            self.lbl_settings.setText(f'Settings: the current tabs.{note}')
            return
        try:
            text = describe(load_settings(src))
        except ValueError as e:
            text = str(e)
        self.lbl_settings.setText(f'Settings from {src.name}: {text}')

    def _on_load_settings_file(self):
        start = str(self._out_dir()) if self.selected_dir else ''
        path, _ = QFileDialog.getOpenFileName(
            self, 'Batch settings', start, 'Settings (*.json)'
        )
        if not path:
            return
        try:
            load_settings(path)
        except ValueError as e:
            QMessageBox.warning(self, 'Cannot use this file', str(e))
            return
        self.settings_file = path
        self.btn_clear_settings.setEnabled(True)
        self._update_settings_label()

    def _on_clear_settings_file(self):
        self.settings_file = None
        self.btn_clear_settings.setEnabled(False)
        self._update_settings_label()

    def _apply_doc_to_ui(self, doc):
        """Shows the stages / choices of a settings file in this tab."""
        st = doc['stages']
        for key, chk in (
            ('predict', self.chk_predict),
            ('superpixels', self.chk_superpixels),
            ('gasp', self.chk_gasp),
            ('morphology', self.chk_morphology),
            ('classification', self.chk_classification),
            ('export', self.chk_export_table),
        ):
            chk.setChecked(bool(st.get(key)))
        for combo, key in (
            (self.morphology_labels, 'morph_labels'),
            (self.morphology_signal, 'morph_signal'),
        ):
            i = combo.findData(doc[key])
            if i >= 0:
                combo.setCurrentIndex(i)
        path = doc.get('error_model')
        self.chk_errors.setChecked(bool(path))
        self.error_model_path = path or ''
        self.lbl_error_model.setText(Path(path).name if path else 'No model selected.')

    def _update_done_count(self, *_):
        """Shows how many files a run with the skip option would leave out."""
        if not self.chk_skip_done.isChecked() or not self.files:
            self.lbl_done.setText('')
            return
        stages = self._batch_ui_settings()['stages']
        out = self._out_dir()
        n = sum(is_done(f, out, stages) for f in self.files)
        self.lbl_done.setText(
            f'{n} of {len(self.files)} file(s) already processed '
            f'(for the selected stages); {len(self.files) - n} to do.'
        )

    def _toggle_morphology_options(self, enabled):
        self.morphology_labels.setEnabled(enabled)
        self.morphology_signal.setEnabled(enabled)

    def _toggle_error_options(self, enabled):
        self.btn_error_model.setEnabled(enabled)

    def _select_error_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Error model', '', 'Error model (*.joblib)'
        )
        if path:
            self.error_model_path = path
            self.lbl_error_model.setText(Path(path).name)
            self.lbl_error_model.setStyleSheet('font-style: normal;')

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

        images = [
            la.name
            for la in v.layers
            if la._type_string == 'image' and not self._is_batch_layer(la.name)
        ]
        keep_img = self.combo_save_image.currentText()
        self.combo_save_image.blockSignals(True)
        self.combo_save_image.clear()
        self.combo_save_image.addItem('(do not save an image)')
        self.combo_save_image.addItems(images)
        if keep_img in images:
            self.combo_save_image.setCurrentText(keep_img)
        elif images and not self.session:
            self.combo_save_image.setCurrentText(images[0])
        self.combo_save_image.blockSignals(False)

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
            ('pred_', 'sp_', 'gasp_cells_', 'geo_', 'error_', '[Predicted Classes]')
        ) or name.startswith(('aff_', 'affinity'))

    def _on_open_session(self):
        start = (
            str(self._out_dir()) if self.selected_dir else ''
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
        raw, source_name = None, None
        if path is None:  # a session made by hand: the image goes in too
            image_name = self.combo_save_image.currentText()
            if image_name in v.layers:
                layer = v.layers[image_name]
                raw = layer.data[0] if layer.multiscale else layer.data
                source_name = image_name
            start = str(self._out_dir()) if self.selected_dir else ''
            if source_name:
                start = str(Path(start) / f'{source_name}_session.h5')
            chosen, _ = QFileDialog.getSaveFileName(
                self, 'Save session', start, 'HDF5 sessions (*.h5)'
            )
            if not chosen:
                return
            path = Path(chosen)
            if not path.name.endswith(sio.SESSION_SUFFIX):
                path = path.with_name(path.stem + sio.SESSION_SUFFIX)
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
                and name != source_name  # the image is stored as raw
            ):
                payload = sio.layer_to_payload(v.layers[name], order=i)
                if payload is not None:
                    payloads.append(payload)
        if corrected is None and not payloads and raw is None:
            QMessageBox.information(
                self,
                'Nothing to save',
                'Pick a Labels layer for the corrected labels or tick at '
                'least one other layer. The session file was not changed.',
            )
            return
        if corrected is not None and not np.any(corrected):
            answer = QMessageBox.question(
                self,
                'Empty labels',
                f'The layer "{label_name}" has no labels at all. Saving it '
                'replaces the corrected labels in the session with an empty '
                'image. Save anyway?',
            )
            if answer != QMessageBox.Yes:
                return
        try:
            sio.save_edits(
                path,
                corrected=corrected,
                corrected_layer=layer_name,
                # no ticked layers: keep the ones already in the file
                extras=payloads or None,
                raw=raw,
                source_name=source_name,
            )
        except Exception as e:
            QMessageBox.warning(self, 'Cannot save session', f'{type(e).__name__}: {e}')
            return
        if self.session is None:
            self.session = {
                'path': path,
                'base': sio.base_name(source_name or path.stem.removesuffix('_session')),
            }
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
            'skip_done': self.chk_skip_done.isChecked(),
            'input_mode': self.combo_input.currentData(),
            'morph_labels': self.morphology_labels.currentData(),
            'morph_signal': self.morphology_signal.currentData(),
        }

    def _set_running(self, running: bool):
        """Locks everything except the cancel button while a batch runs."""
        self.btn_select_dir.setEnabled(not running)
        self.combo_input.setEnabled(not running)
        self.group_stages.setEnabled(not running)
        for w in (
            self.chk_save_settings,
            self.chk_use_saved,
            self.btn_load_settings,
        ):
            w.setEnabled(not running)
        self.group_sess.setEnabled(not running)
        self.btn_start_batch.setEnabled(not running and bool(self.files))
        self.btn_cancel.setEnabled(running)
        if hasattr(self.main_widget, 'set_batch_running'):
            self.main_widget.set_batch_running(running)

    def _confirm_discard(self) -> bool:
        """Warns when this run would throw away corrected labels."""
        ui = self._batch_ui_settings()
        stages = ui['stages']
        if not (stages.get('superpixels') or stages.get('gasp')):
            return True  # the labels are not recomputed
        out = self._out_dir()
        from w_seg_napari._batch_io import session_path

        n = 0
        for f in self.files:
            if ui['skip_done'] and is_done(f, out, stages):
                continue
            sp = session_path(out, f)
            try:
                if sp.is_file() and has_dataset(sp, CORRECTED):
                    n += 1
            except OSError:
                pass
        if not n:
            return True
        answer = QMessageBox.question(
            self,
            'Corrected labels will be discarded',
            f'{n} session(s) hold corrected labels. This run computes new '
            'superpixel / GASP labels, which discards those corrections.\n\n'
            'To keep them, switch off stages 2 and 3 (and choose "Corrected '
            'labels" for the morphology). Continue anyway?',
        )
        return answer == QMessageBox.Yes

    def _on_start_batch(self):
        if not self.selected_dir or not self.files or self.worker is not None:
            return

        try:
            doc = None
            src, own_file = self._settings_source()
            if src is not None:
                doc = load_settings(src)
                self._apply_doc_to_ui(doc)
            # Copies the settings of tabs 1 and 4 on the main thread
            settings = self.main_widget.collect_batch_settings(
                self._batch_ui_settings()
            )
            info = self.main_widget.get_active_models_info()
            if doc is not None:
                diffs = model_differences(
                    doc, info.get('model_name'), settings.get('model_cfg')
                )
                if diffs and QMessageBox.question(
                    self,
                    'Different model',
                    f'The settings in {src.name} were made with another '
                    'model:\n- ' + '\n- '.join(diffs) + '\n\nContinue anyway?',
                ) != QMessageBox.Yes:
                    return
                settings = apply_loaded(settings, doc)
            if self.chk_errors.isChecked() and self.chk_morphology.isChecked():
                if not self.error_model_path:
                    raise ValueError('Choose an error model, or switch the flagging off.')
                from w_seg_napari._segmentation_errors import load_error_model

                settings['error_bundle'] = load_error_model(
                    self.error_model_path
                )
            # the file that goes with the results (built now, so that a value
            # that cannot be saved is found before the run starts)
            settings['settings_doc'] = settings_document(
                settings,
                model_name=info.get('model_name', ''),
                classifier=info.get('classifier_type', ''),
                error_model_path=(
                    self.error_model_path if settings.get('error_bundle') else None
                ),
            )
            # the folder's own file is already what will be used
            settings['save_settings'] = self.chk_save_settings.isChecked() and not (
                doc is not None and own_file
            )
        except (ValueError, TypeError) as e:
            QMessageBox.warning(self, 'Cannot start batch', str(e))
            return

        if not self._confirm_discard():
            return

        self.cancel_event.clear()
        self.failed = []
        self.notes = []
        self.n_skipped = 0
        self._rendered = []
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
            self._render(name, results)
        elif kind == 'skipped':
            self.n_skipped += 1
        elif kind == 'note':
            self.notes.append(event[1])
        elif kind == 'failed':
            _, name, message = event
            print(f'[w-seg] Batch: {name} failed: {message}')

    def _render(self, name, results):
        """Shows one finished image; the previous image's layers go afterwards."""
        layers = self.viewer.layers
        before = {id(la) for la in layers}
        self.main_widget.render_batch_results(name, results)
        added = [la for la in layers if id(la) not in before]
        if self.chk_only_latest.isChecked():
            for la in self._rendered:
                if la in layers:
                    layers.remove(la)
            self._rendered = added
        else:
            self._rendered = []

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
        if self.n_skipped:
            msg += f'\nSkipped (already processed): {self.n_skipped}.'
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
            f'Results are in {self._out_dir()}'
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
