import napari
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
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
