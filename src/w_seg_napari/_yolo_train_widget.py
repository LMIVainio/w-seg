"""Train-tab section: train a YOLO cell-type classifier from the layers that
are open in napari and attach it to an existing W-seg model bundle."""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import napari
import numpy as np
from napari.qt.threading import thread_worker
from qtpy.QtCore import QProcess, QProcessEnvironment, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from w_seg_napari._model_config import DEFAULT_YOLO_CROP
from w_seg_napari._train_args import (
    build_yolo_args,
    safe_name,
    unique_bundle_dir,
)
from w_seg_napari._wsegutils import BUNDLE_DIR, list_local_bundles

MARKER = '@@'
MAX_LOG_LINES = 400
BASE_MODELS = ('yolo11n-cls.pt', 'yolo11s-cls.pt', 'yolov8n-cls.pt')


def layer_array(layer):
    """The full-resolution array of a napari layer (first level if multiscale)."""
    data = layer.data
    if getattr(layer, 'multiscale', False):
        data = data[0]
    return np.asarray(data)


class YoloTrainWidget(QGroupBox):
    build_progress = Signal(int, int)

    def __init__(self, viewer=None, main_widget=None, parent=None):
        super().__init__('Train a YOLO cell-type classifier', parent)
        self.viewer = viewer
        self.main_widget = main_widget
        self.peer = None  # the W-net TrainWidget: disabled while this runs
        self.process = None
        self.worker = None
        self.stop_file = None
        self.samples = []  # dicts, see _on_add_sample
        self._names = {}  # class id -> name, kept while samples change
        self._buffer = ''
        self._new_bundle = None
        self._last_done = None
        self.setCheckable(True)
        self.setChecked(False)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.body = QWidget(self)
        lay = QVBoxLayout(self.body)
        info = QLabel(
            'Learns the tissue type of each cell from crops around it and '
            'stores the classifier inside a copy of one of your models. '
            'Open the image, the cell labels and a class layer (class id '
            'painted on the cells, 0 = not annotated) in napari, add them as '
            'a sample, and repeat for every image. With two or more images, '
            'whole images are held out to measure the accuracy.'
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        form = QFormLayout()
        self.image_combo = QComboBox()
        self.labels_combo = QComboBox()
        self.class_combo = QComboBox()
        self.btn_add = QPushButton('Add this image as a sample')
        self.btn_add.clicked.connect(self._on_add_sample)
        self.btn_remove = QPushButton('Remove selected')
        self.btn_remove.clicked.connect(self._on_remove_sample)
        self.sample_list = QListWidget()
        self.sample_list.setMaximumHeight(90)
        form.addRow('RGB image:', self.image_combo)
        form.addRow('Cell labels:', self.labels_combo)
        form.addRow('Class layer:', self.class_combo)
        row = QHBoxLayout()
        row.addWidget(self.btn_add)
        row.addWidget(self.btn_remove)
        form.addRow(row)
        form.addRow('Samples:', self.sample_list)
        lay.addLayout(form)

        self.class_table = QTableWidget(0, 3)
        self.class_table.setHorizontalHeaderLabels(['Use', 'Class id', 'Name'])
        self.class_table.verticalHeader().setVisible(False)
        self.class_table.setMaximumHeight(150)
        self.class_table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(QLabel('Classes (names are stored in the model):'))
        lay.addWidget(self.class_table)

        tr = QFormLayout()
        self.base_combo = QComboBox()
        self.base_combo.setEditable(True)
        for b in BASE_MODELS:
            self.base_combo.addItem(b)
        self.base_combo.setToolTip(
            'Pre-trained Ultralytics classification weights, or the path of '
            'an earlier classifier (.pt) to continue training. A name is '
            'downloaded the first time.'
        )
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 2000)
        self.epochs_spin.setValue(100)
        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(0.05, 0.6)
        self.val_spin.setSingleStep(0.05)
        self.val_spin.setValue(0.3)
        self.val_spin.setToolTip('Share of the cells held out for validation.')
        self.attach_combo = QComboBox()
        self.attach_combo.setToolTip(
            'The classifier is written into a COPY of this model (the '
            'original is not changed).'
        )
        self.name_edit = QLineEdit('')
        self.name_edit.setPlaceholderText('name of the new model')
        tr.addRow('Start from:', self.base_combo)
        tr.addRow('Epochs:', self.epochs_spin)
        tr.addRow('Validation share:', self.val_spin)
        tr.addRow('Attach to model:', self.attach_combo)
        tr.addRow('New model name:', self.name_edit)
        lay.addLayout(tr)

        self.adv_cb = QCheckBox('Advanced: crop and training settings')
        self.adv = QWidget()
        af = QFormLayout(self.adv)
        self.context_spin = QDoubleSpinBox()
        self.context_spin.setRange(1.0, 6.0)
        self.context_spin.setSingleStep(0.25)
        self.context_spin.setValue(DEFAULT_YOLO_CROP['context_factor'])
        self.context_spin.setToolTip(
            'Crop side = this x the cell\'s minor axis: how much neighbouring '
            'tissue the classifier sees. Stored in the model and used at prediction.'
        )
        self.min_crop_spin = QSpinBox()
        self.min_crop_spin.setRange(16, 2048)
        self.min_crop_spin.setValue(DEFAULT_YOLO_CROP['min_crop_size'])
        self.max_crop_spin = QSpinBox()
        self.max_crop_spin.setRange(32, 4096)
        self.max_crop_spin.setValue(DEFAULT_YOLO_CROP['max_crop_size'])
        self.size_spin = QSpinBox()
        self.size_spin.setRange(64, 1024)
        self.size_spin.setSingleStep(32)
        self.size_spin.setValue(DEFAULT_YOLO_CROP['target_size'])
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(32)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 1000)
        self.patience_spin.setValue(15)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setValue(4)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 1_000_000)
        af.addRow('Context factor:', self.context_spin)
        af.addRow('Min crop size (px):', self.min_crop_spin)
        af.addRow('Max crop size (px):', self.max_crop_spin)
        af.addRow('Crop resolution (px):', self.size_spin)
        af.addRow('Batch size:', self.batch_spin)
        af.addRow('Early-stop patience:', self.patience_spin)
        af.addRow('Data workers:', self.workers_spin)
        af.addRow('Random seed:', self.seed_spin)
        self.adv.setVisible(False)
        self.adv_cb.toggled.connect(self.adv.setVisible)
        lay.addWidget(self.adv_cb)
        lay.addWidget(self.adv)

        buttons = QHBoxLayout()
        self.btn_start = QPushButton('Build crops and train')
        self.btn_stop = QPushButton('Stop (keeps best classifier)')
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        buttons.addWidget(self.btn_start)
        buttons.addWidget(self.btn_stop)
        lay.addLayout(buttons)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.status = QLabel('Idle.')
        self.status.setWordWrap(True)
        self.show_log = QCheckBox('Show training log')
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(MAX_LOG_LINES)
        self.log.setVisible(False)
        self.show_log.toggled.connect(self.log.setVisible)
        lay.addWidget(self.bar)
        lay.addWidget(self.status)
        lay.addWidget(self.show_log)
        lay.addWidget(self.log)

        outer = QVBoxLayout(self)
        outer.addWidget(self.body)
        self.body.setVisible(False)
        self.toggled.connect(self._on_toggled)
        self.build_progress.connect(self._on_build_progress)

        if self.viewer is not None:
            for ev in (
                self.viewer.layers.events.inserted,
                self.viewer.layers.events.removed,
            ):
                ev.connect(self.refresh_layer_choices)
        self.refresh_layer_choices()
        self.refresh_bundles()

    # ---------------------------------------------------------- choices
    def _on_toggled(self, checked):
        self.body.setVisible(checked)
        if checked:
            self.refresh_layer_choices()
            self.refresh_bundles()

    def refresh_layer_choices(self, event=None):
        if self.viewer is None:
            return
        images = [
            l for l in self.viewer.layers if isinstance(l, napari.layers.Image)
        ]
        labels = [
            l for l in self.viewer.layers if isinstance(l, napari.layers.Labels)
        ]
        for combo, layers in (
            (self.image_combo, images),
            (self.labels_combo, labels),
            (self.class_combo, labels),
        ):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            for lyr in layers:
                combo.addItem(lyr.name, lyr)
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def refresh_bundles(self):
        current = self.attach_combo.currentText()
        self.attach_combo.clear()
        for name, folder in list_local_bundles():
            self.attach_combo.addItem(name, folder)
        idx = self.attach_combo.findText(current)
        if idx >= 0:
            self.attach_combo.setCurrentIndex(idx)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_bundles()

    # ---------------------------------------------------------- samples
    def _on_add_sample(self):
        img, lab, cls = (
            self.image_combo.currentData(),
            self.labels_combo.currentData(),
            self.class_combo.currentData(),
        )
        if img is None or lab is None or cls is None:
            QMessageBox.warning(
                self, 'YOLO', 'Pick an RGB image, a labels layer and a class layer.'
            )
            return
        if lab is cls:
            QMessageBox.warning(
                self, 'YOLO',
                'The class layer must be a different layer from the cell labels.',
            )
            return
        labels, classes = layer_array(lab), layer_array(cls)
        if labels.shape != classes.shape:
            QMessageBox.warning(
                self, 'YOLO',
                f'The class layer {classes.shape} and the cell labels '
                f'{labels.shape} must have the same shape.',
            )
            return
        ids = [int(i) for i in np.unique(classes) if i > 0]
        if not ids:
            QMessageBox.warning(self, 'YOLO', 'The class layer has no annotated pixels.')
            return
        name = img.name
        taken = {s['name'] for s in self.samples}
        n = 2
        while name in taken:
            name = f'{img.name}_{n}'
            n += 1
        self.samples.append(
            {
                'name': name,
                'image': layer_array(img),
                'labels': labels,
                'classes': classes,
                'image_scale': tuple(img.scale),
                'image_translate': tuple(img.translate),
                'labels_scale': tuple(lab.scale),
                'labels_translate': tuple(lab.translate),
                'class_ids': ids,
            }
        )
        self._refresh_sample_views()
        if not self.name_edit.text() and self.attach_combo.currentText():
            self.name_edit.setText(f'{self.attach_combo.currentText()}_yolo')

    def _on_remove_sample(self):
        row = self.sample_list.currentRow()
        if row >= 0:
            del self.samples[row]
            self._refresh_sample_views()

    def _refresh_sample_views(self):
        self.sample_list.clear()
        for s in self.samples:
            self.sample_list.addItem(
                f'{s["name"]}   classes {s["class_ids"]}'
            )
        self._rebuild_class_table()

    def _rebuild_class_table(self):
        self._remember_names()
        ids = sorted({i for s in self.samples for i in s['class_ids']})
        self.class_table.setRowCount(len(ids))
        for r, cid in enumerate(ids):
            use = QTableWidgetItem()
            use.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            use.setCheckState(Qt.Checked)
            idi = QTableWidgetItem(str(cid))
            idi.setFlags(Qt.ItemIsEnabled)
            self.class_table.setItem(r, 0, use)
            self.class_table.setItem(r, 1, idi)
            self.class_table.setItem(
                r, 2, QTableWidgetItem(self._names.get(cid, f'class{cid}'))
            )

    def _remember_names(self):
        for r in range(self.class_table.rowCount()):
            cid = int(self.class_table.item(r, 1).text())
            self._names[cid] = self.class_table.item(r, 2).text().strip()

    def class_names(self):
        """{class id: name} of the ticked classes."""
        out = {}
        for r in range(self.class_table.rowCount()):
            if self.class_table.item(r, 0).checkState() == Qt.Checked:
                cid = int(self.class_table.item(r, 1).text())
                out[cid] = self.class_table.item(r, 2).text().strip() or f'class{cid}'
        return out

    # ------------------------------------------------------------- run
    def _set_running(self, running):
        for w in (
            self.btn_add, self.btn_remove, self.sample_list, self.class_table,
            self.base_combo, self.epochs_spin, self.val_spin, self.attach_combo,
            self.name_edit, self.adv_cb, self.adv, self.image_combo,
            self.labels_combo, self.class_combo,
        ):  # fmt: skip
            w.setEnabled(not running)
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running and self.process is not None)
        if self.peer is not None:
            self.peer.setEnabled(not running)
        lock = getattr(self.main_widget, 'set_training_running', None)
        if lock is not None:
            lock(running)

    def _validate(self):
        if not self.samples:
            raise ValueError('Add at least one image as a sample.')
        names = self.class_names()
        if len(names) < 2:
            raise ValueError('Tick at least two classes.')
        folder = self.attach_combo.currentData()
        if folder is None:
            raise ValueError('There is no model to attach the classifier to.')
        if not self.name_edit.text().strip():
            raise ValueError('Give the new model a name.')
        if len(set(n.lower() for n in names.values())) != len(names):
            raise ValueError('Class names must be different.')
        if self.min_crop_spin.value() > self.max_crop_spin.value():
            raise ValueError('Min crop size must not exceed max crop size.')
        return names, folder

    def _crop_settings(self):
        return {
            'target_size': self.size_spin.value(),
            'context_factor': self.context_spin.value(),
            'min_crop_size': self.min_crop_spin.value(),
            'max_crop_size': self.max_crop_spin.value(),
        }

    def _on_start(self):
        if self.process is not None or self.worker is not None:
            return
        self._remember_names()
        try:
            names, bundle = self._validate()
        except ValueError as e:
            QMessageBox.warning(self, 'YOLO', str(e))
            return
        name = safe_name(self.name_edit.text())
        run_root = BUNDLE_DIR / 'yolo_runs'
        run_root.mkdir(parents=True, exist_ok=True)
        dataset = run_root / f'{name}_dataset'
        if dataset.exists():
            shutil.rmtree(dataset)
        self._new_bundle = unique_bundle_dir(BUNDLE_DIR, name)
        self._run = {
            'name': name, 'run_root': str(run_root), 'dataset': dataset,
            'bundle': bundle, 'names': names,
        }  # fmt: skip
        self.log.clear()
        self.bar.setValue(0)
        self.status.setText('Cutting crops...')
        self._set_running(True)
        self.btn_stop.setEnabled(False)

        samples = [dict(s) for s in self.samples]
        crop = self._crop_settings()
        val = self.val_spin.value()
        seed = self.seed_spin.value()
        emit = self.build_progress.emit

        @thread_worker
        def build():
            from w_seg_napari._yolo_data import build_dataset

            return build_dataset(
                samples, names, dataset, crop=crop, val_fraction=val,
                seed=seed, progress=lambda d, t: emit(d, t),
            )  # fmt: skip

        self.worker = build()
        self.worker.returned.connect(self._on_built)
        self.worker.errored.connect(self._on_build_error)
        self.worker.start()

    def _on_build_progress(self, done, total):
        self.bar.setValue(int(100 * done / max(total, 1)) // 2)  # first half
        self.status.setText(f'Cutting crops... {done}/{total}')

    def _on_build_error(self, err):
        self.worker = None
        self._set_running(False)
        self.status.setText(f'Could not build the crops: {err}')
        QMessageBox.warning(self, 'YOLO', f'Could not build the crops:\n{err}')

    def _on_built(self, summary):
        self.worker = None
        run = self._run
        info_path = Path(run['run_root']) / f'{run["name"]}_dataset_info.json'
        info_path.write_text(json.dumps(summary, indent=2))
        counts = summary['counts']
        msg = (
            f'{summary["n_cells"]} crops ({summary["split_mode"]}): '
            f'{sum(counts["train"].values())} train, {sum(counts["val"].values())} val.'
        )
        if summary['warnings']:
            msg += ' ' + ' '.join(summary['warnings'])
        self.log.appendPlainText(msg)
        self.status.setText(msg)

        prepare = getattr(self.main_widget, 'prepare_for_training', None)
        if prepare is not None:
            prepare()
        self.stop_file = str(
            Path(tempfile.gettempdir()) / f'wseg_yolo_stop_{id(self)}.flag'
        )
        Path(self.stop_file).unlink(missing_ok=True)
        settings = {
            'dataset': str(run['dataset']),
            'out_root': run['run_root'],
            'name': run['name'],
            'stop_file': self.stop_file,
            'base': self.base_combo.currentText().strip() or BASE_MODELS[0],
            'epochs': self.epochs_spin.value(),
            'imgsz': self.size_spin.value(),
            'batch': self.batch_spin.value(),
            'patience': self.patience_spin.value(),
            'seed': self.seed_spin.value(),
            'workers': self.workers_spin.value(),
            'info_json': str(info_path),
            'attach_to': run['bundle'],
            'out_bundle': str(self._new_bundle),
        }
        proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert('PYTHONUNBUFFERED', '1')
        proc.setProcessEnvironment(env)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_output)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        self.process = proc
        self._buffer = ''
        self._last_done = None
        self.btn_stop.setEnabled(True)
        proc.start(
            sys.executable,
            ['-m', 'w_seg_napari._training.yolo', *build_yolo_args(settings)],
        )

    def _on_stop(self):
        if self.process is None:
            return
        Path(self.stop_file).write_text('stop')
        self.btn_stop.setEnabled(False)
        self.status.setText('Stopping after the current epoch and saving...')

    def _on_error(self, error):
        if (
            self.process is not None
            and self.process.state() == QProcess.NotRunning
        ):
            self.status.setText(f'Could not run YOLO training ({error}).')
            self.process = None
            self._set_running(False)

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
            dev = str(ev.get('device', '?'))
            self.status.setText(
                f'Training on {"GPU " + dev if dev != "cpu" else "CPU (slow)"}'
            )
        elif kind == 'progress':
            self.bar.setValue(50 + int(50 * ev['epoch'] / max(ev['epochs'], 1)))
            self.status.setText(
                f'Epoch {ev["epoch"]}/{ev["epochs"]} | val accuracy '
                f'{ev["top1"]:.3f} (best {ev["best_top1"]:.3f})'
            )
        elif kind == 'done':
            self._last_done = ev

    def _on_finished(self, exit_code, _status):
        self.process = None
        self._set_running(False)
        if self.stop_file:
            Path(self.stop_file).unlink(missing_ok=True)
        done = self._last_done
        if exit_code == 0 and done and done.get('bundle'):
            self.bar.setValue(100)
            self.status.setText(
                f'Finished. Best validation accuracy {done["best_top1"]:.3f}. '
                f'Saved the model with its classifier as '
                f'"{Path(done["bundle"]).name}"; select it in tab 1.'
            )
            tab1 = getattr(self.main_widget, 'tab1', None)
            if tab1 is not None and hasattr(tab1, '_populate_weights'):
                tab1._populate_weights()
        else:
            self.status.setText(
                f'YOLO training ended with an error (exit code {exit_code}). '
                'Tick "Show training log" for details.'
            )
            self.show_log.setChecked(True)

    def closeEvent(self, event):
        if self.process is not None:
            self.process.kill()
        super().closeEvent(event)
