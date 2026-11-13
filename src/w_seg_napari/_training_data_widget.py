from pathlib import Path

import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QWidget,
)

from w_seg_napari._training_data import (
    build_training_stack,
    save_training_stack,
)


class TrainingDataWidget(QGroupBox):
    """Collapsible tool that saves image + wall mask + labels as a 5-channel tif."""

    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__('Create training data', parent)
        self.viewer = viewer
        self.setCheckable(True)
        self.setChecked(False)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.body = QWidget(self)
        form = QFormLayout(self.body)
        self.signal_combo = QComboBox()
        self.semantic_combo = QComboBox()
        self.labels_combo = QComboBox()
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.setToolTip(
            'Prediction values above this become wall (1). Integer masks '
            'and label layers use "greater than 0".'
        )
        self.mask_cb = QCheckBox(
            'Cut walls out of the labels (wall pixels = 0)'
        )
        self.mask_cb.setToolTip(
            'Uses the semantic layer as a mask on the labels: wall pixels '
            'become 0, cells and background keep their label.'
        )
        self.split_spin = QSpinBox()
        self.split_spin.setRange(0, 1_000_000)
        self.split_spin.setToolTip(
            'If many cells share one label (for example 1 = not curated yet), '
            'give each separate region of that label its own id. 0 = off.'
        )
        self.btn_save = QPushButton('Save 5-channel training tif...')
        self.btn_save.clicked.connect(self._on_save)
        self.info = QLabel(
            'Channels: R, G, B, wall mask, labels. A single-channel image is '
            'repeated into RGB.'
        )
        self.info.setWordWrap(True)
        form.addRow('Signal image:', self.signal_combo)
        form.addRow('Semantic (wall):', self.semantic_combo)
        form.addRow('Labels:', self.labels_combo)
        form.addRow('Wall threshold:', self.threshold_spin)
        form.addRow(self.mask_cb)
        form.addRow('Split shared label (0 = off):', self.split_spin)
        form.addRow(self.info)
        form.addRow(self.btn_save)

        form.addRow(QLabel('<b>From batch session files</b>'))
        self.batch_info = QLabel(
            'One tif per session: the image (raw), the corrected cell '
            "labels, and the model's wall prediction (pred_sem) as the wall "
            'mask, with the wall pixels that lie inside corrected cells '
            'removed. The training reads the wall as a yes/no mask, so the '
            'prediction is thresholded.'
        )
        self.batch_info.setWordWrap(True)
        self.batch_mode = QComboBox()
        self.batch_mode.addItems(['Fixed threshold', 'Automatic (Otsu)'])
        self.batch_mode.setToolTip(
            'Fixed: the value below, the same for every image. Automatic: '
            "Otsu's threshold of each image's own prediction."
        )
        self.batch_threshold = QDoubleSpinBox()
        self.batch_threshold.setRange(0.0, 1.0)
        self.batch_threshold.setSingleStep(0.05)
        self.batch_threshold.setValue(0.5)
        self.batch_mode.currentIndexChanged.connect(
            lambda i: self.batch_threshold.setEnabled(i == 0)
        )
        self.batch_corrected = QCheckBox('Only sessions with corrected labels')
        self.batch_corrected.setChecked(True)
        self.batch_corrected.setToolTip(
            'Without this, a session with no saved corrections is used with '
            'its GASP labels, which are not checked by you.'
        )
        self.batch_val = QDoubleSpinBox()
        self.batch_val.setRange(0.0, 0.9)
        self.batch_val.setSingleStep(0.05)
        self.batch_val.setValue(0.0)
        self.batch_val.setToolTip(
            'Share of the images (whole images, always the same ones) put in '
            'a "validation" folder next to "training". 0 = everything in one '
            'folder.'
        )
        self.btn_batch = QPushButton(
            'Create training data from batch files...'
        )
        self.btn_batch.clicked.connect(self._on_batch)
        form.addRow(self.batch_info)
        form.addRow('Wall threshold:', self.batch_mode)
        form.addRow('Fixed threshold value:', self.batch_threshold)
        form.addRow(self.batch_corrected)
        form.addRow('Validation share:', self.batch_val)
        form.addRow(self.btn_batch)

        outer = QFormLayout(self)
        outer.addRow(self.body)
        self.body.setVisible(False)
        self.toggled.connect(self._on_toggled)

        if self.viewer is not None:
            for ev in (
                self.viewer.layers.events.inserted,
                self.viewer.layers.events.removed,
            ):
                ev.connect(self.refresh_layer_choices)
        self.refresh_layer_choices()

    def _on_toggled(self, checked):
        self.body.setVisible(checked)
        if checked:
            self.refresh_layer_choices()

    def refresh_layer_choices(self, event=None):
        if self.viewer is None:
            return
        images = [
            lyr
            for lyr in self.viewer.layers
            if isinstance(lyr, napari.layers.Image)
        ]
        # Masks may be Image or Labels layers
        masks = [
            lyr
            for lyr in self.viewer.layers
            if isinstance(lyr, napari.layers.Image | napari.layers.Labels)
        ]
        labels = [
            lyr
            for lyr in self.viewer.layers
            if isinstance(lyr, napari.layers.Labels)
        ]
        for combo, layers in (
            (self.signal_combo, images),
            (self.semantic_combo, masks),
            (self.labels_combo, labels),
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

    def _on_save(self):
        sig = self.signal_combo.currentData()
        sem = self.semantic_combo.currentData()
        lab = self.labels_combo.currentData()
        if sig is None or sem is None or lab is None:
            QMessageBox.warning(
                self,
                'Training data',
                'Pick a signal, semantic and labels layer.',
            )
            return
        try:
            stack = build_training_stack(
                sig.data,
                sem.data,
                lab.data,
                self.threshold_spin.value(),
                mask_labels=self.mask_cb.isChecked(),
                split_label=self.split_spin.value(),
            )
        except (ValueError, TypeError) as e:
            QMessageBox.warning(self, 'Training data', str(e))
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            'Save training data',
            f'{lab.name}_training.tif',
            'TIFF (*.tif *.tiff)',
        )
        if not path:
            return
        if not path.lower().endswith(('.tif', '.tiff')):
            path += '.tif'
        try:
            save_training_stack(path, stack)
        except OSError as e:
            QMessageBox.critical(self, 'Training data', str(e))
            return
        QMessageBox.information(
            self,
            'Training data',
            f'Saved {stack.shape} {stack.dtype} stack to\n{path}',
        )

    def _on_batch(self):
        from w_seg_napari import _session_training_data as std

        files, _ = QFileDialog.getOpenFileNames(
            self,
            'Session files',
            '',
            'Session files (*_session.h5);;HDF5 (*.h5)',
        )
        if not files:
            return
        folder = QFileDialog.getExistingDirectory(
            self, 'Save the training tifs in', str(Path(files[0]).parent)
        )
        if not folder:
            return
        threshold = (
            self.batch_threshold.value()
            if self.batch_mode.currentIndex() == 0
            else std.AUTO
        )
        # read every session first, so a split by image is possible
        samples, skipped = {}, []
        for f in files:
            sample, why = std.read_session_sample(
                f, self.batch_corrected.isChecked()
            )
            if sample is None:
                skipped.append(f'{Path(f).name}: {why}')
            elif sample['name'] in samples:
                skipped.append(f'{Path(f).name}: same name as another file')
            else:
                samples[sample['name']] = sample
        train, val = std.split_names(samples, self.batch_val.value())
        base = Path(folder)
        jobs = [(n, base / 'training' if val else base) for n in train]
        jobs += [(n, base / 'validation') for n in val]
        progress = QProgressDialog(
            'Writing training tifs...', 'Cancel', 0, len(jobs), self
        )
        progress.setWindowModality(Qt.WindowModal)
        done, uncorrected = 0, []
        for i, (name, out) in enumerate(jobs):
            progress.setValue(i)
            if progress.wasCanceled():
                break
            sample = samples[name]
            try:
                stack = std.session_training_stack(
                    sample['raw'],
                    sample['pred_sem'],
                    sample['labels'],
                    threshold,
                )
                out.mkdir(parents=True, exist_ok=True)
                save_training_stack(out / f'{name}_training.tif', stack)
            except (ValueError, TypeError, OSError) as e:
                skipped.append(f'{name}: {e}')
                continue
            done += 1
            if sample['labels_origin'] != 'corrected':
                uncorrected.append(f'{name} ({sample["labels_origin"]})')
        progress.setValue(len(jobs))
        text = f'Saved {done} of {len(files)} training tifs in\n{folder}'
        if val:
            text += f'\n({len(train)} training, {len(val)} validation)'
        if skipped:
            text += '\n\nSkipped:\n' + '\n'.join(skipped)
        if uncorrected:
            text += '\n\nUsed UNCORRECTED labels:\n' + '\n'.join(uncorrected)
        QMessageBox.information(self, 'Training data', text)
