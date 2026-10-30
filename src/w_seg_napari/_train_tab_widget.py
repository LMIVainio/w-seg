"""Tab 6: train a new W-seg model from 5-channel training tifs.

The training itself runs in a separate Python process
(``python -m w_seg_napari._training``), so napari stays responsive, a crash
cannot take napari down, and "Stop" is clean. This widget builds the command,
reads the process's progress lines and shows the progress picture.
"""

import json
import sys
import tempfile
from pathlib import Path

from qtpy.QtCore import QProcess, QProcessEnvironment
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from w_seg_napari._train_args import (
    build_args,
    count_tifs,
    parse_ranges,
    safe_name,
)
from w_seg_napari._training_data_widget import TrainingDataWidget
from w_seg_napari._wsegutils import BUNDLE_DIR, list_local_bundles

MARKER = '@@'
MAX_LOG_LINES = 400
PROGRESS_LAYER = 'Training progress'


class FolderRow(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.btn = QPushButton('Browse...')
        self.btn.clicked.connect(self._browse)
        row.addWidget(self.edit, 1)
        row.addWidget(self.btn)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(
            self, self.title, self.edit.text() or str(Path.home())
        )
        if folder:
            self.edit.setText(folder)

    def text(self):
        return self.edit.text().strip()


class TrainWidget(QGroupBox):
    def __init__(self, viewer=None, main_widget=None, parent=None):
        super().__init__('Train a new model', parent)
        self.viewer = viewer
        self.main_widget = main_widget
        self.process = None
        self.stop_file = None
        self._buffer = ''
        self._bundle_dir = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        layout = QVBoxLayout(self)

        info = QLabel(
            'Training data: folders of 5-channel tifs (R, G, B, wall mask, '
            'labels), made with "Create training data" in tab 1. Mono images '
            'are stored as grey RGB. Training runs in a separate process.'
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()
        self.train_row = FolderRow('Training folder')
        self.val_row = FolderRow('Validation folder')
        self.train_row.edit.textChanged.connect(self._update_counts)
        self.val_row.edit.textChanged.connect(self._update_counts)
        self.count_label = QLabel('')
        self.base_combo = QComboBox()
        self.base_combo.setToolTip(
            'Continue training a model you already have. Its affinity ranges '
            'and preprocessing are kept. "(last checkpoint)" is the model at the end '
            'of that run, "name" alone is its best one. Tip: use a lower learning rate and '
            'put old and new data together in the training folder.'
        )
        self.base_combo.currentIndexChanged.connect(self._on_base_changed)
        self.name_edit = QLineEdit('my_model')
        self.out_row = FolderRow('Output folder')
        self.out_row.edit.setText(str(BUNDLE_DIR))
        form.addRow('Training data:', self.train_row)
        form.addRow('Validation data:', self.val_row)
        form.addRow('', self.count_label)
        form.addRow('Start from:', self.base_combo)
        self.geometry_cb = QCheckBox('Add geometry head (new architecture)')
        self.geometry_cb.setToolTip(
            'Adds a head that predicts distance to the nearest edge, the '
            'direction away from it and a skeleton line for every cell. The '
            'training tifs need no changes; the targets are computed from '
            'the labels. Only for new models (a model you continue keeps its '
            'own architecture).'
        )
        form.addRow('', self.geometry_cb)
        form.addRow('Model name:', self.name_edit)
        form.addRow('Output folder:', self.out_row)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 1_000_000)
        self.steps_spin.setSingleStep(500)
        self.steps_spin.setValue(5000)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(8)
        self.eval_spin = QSpinBox()
        self.eval_spin.setRange(1, 100_000)
        self.eval_spin.setValue(200)
        form.addRow('Training steps:', self.steps_spin)
        form.addRow('Batch size:', self.batch_spin)
        form.addRow('Evaluate every (steps):', self.eval_spin)
        layout.addLayout(form)

        # Advanced -------------------------------------------------------
        self.adv_box = QGroupBox('Advanced settings')
        self.adv_box.setCheckable(True)
        self.adv_box.setChecked(False)
        adv = QFormLayout()
        self.adv_body = QWidget()
        adv_form = QFormLayout(self.adv_body)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setValue(2e-4)
        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(0, 100_000)
        self.warmup_spin.setValue(200)
        self.val_samples_spin = QSpinBox()
        self.val_samples_spin.setRange(8, 10_000)
        self.val_samples_spin.setValue(64)
        self.ranges_edit = QLineEdit('6 32')
        self.ranges_edit.setToolTip(
            'Affinity ranges in pixels. Should match the cell sizes in your data.'
        )
        self.zoom_min = QDoubleSpinBox()
        self.zoom_max = QDoubleSpinBox()
        for s, v in ((self.zoom_min, 0.5), (self.zoom_max, 1.5)):
            s.setRange(0.1, 5.0)
            s.setSingleStep(0.1)
            s.setValue(v)
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(self.zoom_min)
        zoom_row.addWidget(self.zoom_max)
        self.bias_spin = QDoubleSpinBox()
        self.bias_spin.setRange(0.0, 1.0)
        self.bias_spin.setSingleStep(0.1)
        self.bias_spin.setValue(0.7)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setValue(4)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 1_000_000)
        adv_form.addRow('Learning rate:', self.lr_spin)
        adv_form.addRow('Warm-up steps:', self.warmup_spin)
        adv_form.addRow('Validation crops:', self.val_samples_spin)
        adv_form.addRow('Affinity ranges (px):', self.ranges_edit)
        adv_form.addRow('Zoom range (min, max):', zoom_row)
        adv_form.addRow('Tissue bias (0-1):', self.bias_spin)
        adv_form.addRow('Data workers:', self.workers_spin)
        adv_form.addRow('Random seed:', self.seed_spin)
        adv.addRow(self.adv_body)
        self.adv_box.setLayout(adv)
        self.adv_body.setVisible(False)
        self.adv_box.toggled.connect(self.adv_body.setVisible)
        layout.addWidget(self.adv_box)

        # Run controls -----------------------------------------------------
        buttons = QHBoxLayout()
        self.btn_start = QPushButton('Start training')
        self.btn_stop = QPushButton('Stop (keeps best model)')
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        buttons.addWidget(self.btn_start)
        buttons.addWidget(self.btn_stop)
        layout.addLayout(buttons)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.status = QLabel('Idle.')
        self.status.setWordWrap(True)
        layout.addWidget(self.bar)
        layout.addWidget(self.status)

        self.show_log = QCheckBox('Show training log')
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(MAX_LOG_LINES)
        self.log.setVisible(False)
        self.show_log.toggled.connect(self.log.setVisible)
        layout.addWidget(self.show_log)
        layout.addWidget(self.log)

        self.refresh_base_models()
        self._update_counts()

    # ------------------------------------------------------------------
    def refresh_base_models(self):
        """List the bundles in ~/.wseg_models as starting points."""
        current = self.base_combo.currentText()
        self.base_combo.blockSignals(True)
        self.base_combo.clear()
        self.base_combo.addItem('New model (from scratch)', None)
        for name, folder in list_local_bundles():
            self.base_combo.addItem(name, folder)
            last = Path(folder) / 'run' / 'last'
            if (last / 'model.keras').is_file() and (
                last / 'config.json'
            ).is_file():
                self.base_combo.addItem(f'{name} (last checkpoint)', str(last))
        idx = self.base_combo.findText(current)
        self.base_combo.setCurrentIndex(max(idx, 0))
        self.base_combo.blockSignals(False)

    def _on_base_changed(self, _index=None):
        folder = self.base_combo.currentData()
        self.geometry_cb.setEnabled(folder is None)
        if folder is None:
            self.lr_spin.setValue(2e-4)
            self.warmup_spin.setValue(200)
            self.ranges_edit.setEnabled(True)
            return
        # Continuing: gentler learning rate, short warm-up, ranges from the model
        self.lr_spin.setValue(5e-5)
        self.warmup_spin.setValue(50)
        self.ranges_edit.setEnabled(False)
        if self.name_edit.text() in ('', 'my_model'):
            self.name_edit.setText(
                f'{self.base_combo.currentText()}_continued'
            )

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_base_models()

    def _update_counts(self):
        n_t = count_tifs(self.train_row.text())
        n_v = count_tifs(self.val_row.text())
        self.count_label.setText(
            f'{n_t} training tif(s), {n_v} validation tif(s)'
        )

    def settings(self):
        """Validated settings dict; raises ValueError with a readable message."""
        train_dir, val_dir = self.train_row.text(), self.val_row.text()
        if count_tifs(train_dir) == 0:
            raise ValueError('The training folder has no .tif files.')
        if count_tifs(val_dir) == 0:
            raise ValueError('The validation folder has no .tif files.')
        if Path(train_dir).resolve() == Path(val_dir).resolve():
            raise ValueError(
                'Training and validation folders must be different, or the '
                'validation score will not mean anything.'
            )
        out_root = self.out_row.text()
        if not out_root:
            raise ValueError('Choose an output folder.')
        warm = self.warmup_spin.value()
        if warm >= self.steps_spin.value():
            raise ValueError(
                'Warm-up steps must be fewer than training steps.'
            )
        return {
            'train_dir': train_dir,
            'val_dir': val_dir,
            'out_root': out_root,
            'name': safe_name(self.name_edit.text()),
            'stop_file': self.stop_file or '',
            'init_from': self.base_combo.currentData(),
            'geometry': self.geometry_cb.isChecked(),
            'steps': self.steps_spin.value(),
            'batch_size': self.batch_spin.value(),
            'eval_every': self.eval_spin.value(),
            'lr': self.lr_spin.value(),
            'warmup': warm,
            'val_samples': self.val_samples_spin.value(),
            'ranges': parse_ranges(self.ranges_edit.text()),
            'zoom': (self.zoom_min.value(), self.zoom_max.value()),
            'tissue_bias': self.bias_spin.value(),
            'workers': self.workers_spin.value(),
            'seed': self.seed_spin.value(),
        }

    def _set_running(self, running):
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        for w in (
            self.train_row, self.val_row, self.out_row, self.name_edit, self.base_combo,
            self.steps_spin, self.batch_spin, self.eval_spin, self.adv_box,
        ):  # fmt: skip
            w.setEnabled(not running)
        self.geometry_cb.setEnabled(
            not running and self.base_combo.currentData() is None
        )

    # ------------------------------------------------------------------
    def _on_start(self):
        if self.process is not None:
            return
        self.stop_file = str(
            Path(tempfile.gettempdir()) / f'wseg_stop_{id(self)}.flag'
        )
        Path(self.stop_file).unlink(missing_ok=True)
        try:
            p = self.settings()
        except ValueError as e:
            QMessageBox.warning(self, 'Training', str(e))
            return
        self.log.clear()
        self.bar.setValue(0)
        self.status.setText('Starting...')
        self._buffer = ''
        self._bundle_dir = None

        proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert('KERAS_BACKEND', 'torch')
        env.insert('PYTHONUNBUFFERED', '1')
        proc.setProcessEnvironment(env)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_output)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        self.process = proc
        self._set_running(True)
        proc.start(
            sys.executable, ['-m', 'w_seg_napari._training', *build_args(p)]
        )

    def _on_stop(self):
        if self.process is None:
            return
        Path(self.stop_file).write_text('stop')
        self.btn_stop.setEnabled(False)
        self.status.setText('Stopping after the current step and saving...')

    def _on_error(self, error):
        if (
            self.process is not None
            and self.process.state() == QProcess.NotRunning
        ):
            self.status.setText(f'Could not run training ({error}).')

    def _on_output(self):
        data = bytes(self.process.readAllStandardOutput()).decode(
            'utf-8', errors='replace'
        )
        self._buffer += data
        *lines, self._buffer = self._buffer.split('\n')
        for line in lines:
            self._handle_line(line.rstrip('\r'))

    def _handle_line(self, line):
        if not line.startswith(MARKER):
            if line.strip():
                self.log.appendPlainText(line)
            return
        try:
            ev = json.loads(line[len(MARKER) :])
        except ValueError:
            return
        kind = ev.get('event')
        if kind == 'start':
            self._bundle_dir = ev['bundle_dir']
            device = ev.get('device', '?')
            self.status.setText(
                f'Training on {device.upper()}'
                + (' (slow without a GPU)' if device == 'cpu' else '')
            )
        elif kind == 'status':
            self.status.setText(ev['message'])
        elif kind in ('step', 'eval'):
            steps = max(1, ev.get('steps', 1))
            self.bar.setValue(int(100 * ev['step'] / steps))
            if kind == 'eval':
                self.status.setText(
                    f'Step {ev["step"]}/{steps} | val Dice {ev["val_dice"]:.3f} '
                    f'(best {ev["best_dice"]:.3f}) | {ev["minutes"]:.1f} min'
                )
                self._show_progress(ev.get('progress_png'))
            else:
                self.bar.setFormat(f'{ev["step"]}/{steps}')
        elif kind == 'error':
            self.status.setText(ev.get('message', 'Training failed.'))

    def _show_progress(self, path):
        """Show the progress picture as an image layer in the napari viewer."""
        if self.viewer is None or not path or not Path(path).is_file():
            return
        from skimage.io import imread

        img = imread(path)
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        if PROGRESS_LAYER in self.viewer.layers:
            self.viewer.layers[PROGRESS_LAYER].data = img
            return
        # First picture: place it to the right of the existing layers
        offset = 0.0
        if len(self.viewer.layers):
            offset = float(self.viewer.layers.extent.world[1][-1]) + 50.0
        self.viewer.add_image(
            img, name=PROGRESS_LAYER, rgb=True, translate=(0.0, offset)
        )
        self.viewer.reset_view()

    def _on_finished(self, exit_code, _status):
        self.process = None
        self._set_running(False)
        Path(self.stop_file).unlink(missing_ok=True)
        if exit_code == 0 and self._bundle_dir:
            self.bar.setValue(100)
            self.status.setText(
                f'Finished. Best model saved to {self._bundle_dir}. '
                'Select it in tab 1 (model list).'
            )
            tab1 = getattr(self.main_widget, 'tab1', None)
            if tab1 is not None and hasattr(tab1, '_populate_weights'):
                tab1._populate_weights()
        else:
            self.status.setText(
                f'Training ended with an error (exit code {exit_code}). '
                'Tick "Show training log" for details.'
            )
            self.show_log.setChecked(True)

    def closeEvent(self, event):
        if self.process is not None:
            self.process.kill()
        super().closeEvent(event)


class Tab6Train(QScrollArea):
    def __init__(self, viewer=None, main_widget=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        self.train_widget = TrainWidget(viewer=viewer, main_widget=main_widget)
        self.training_data_widget = TrainingDataWidget(viewer=viewer)
        layout.addWidget(self.training_data_widget)
        layout.addWidget(self.train_widget)
        layout.addStretch()
        self.setWidget(container)
