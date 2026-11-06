"""Curation tab box: highlight labels by size or isolation and remove them."""

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from w_seg_napari import _label_selection as ls

SELECTION_LAYER = 'label_selection'
SELECTION_COLOR = (1.0, 0.1, 0.8, 1.0)
NO_LIMIT = 100_000_000


class LabelSelectorWidget(QGroupBox):
    """
    Pick labels you do not want, then remove them in one step:
      * by size: smaller than a lower or larger than an upper pixel area;
      * by isolation: labels that touch (or lie within a few pixels of) each
        other form a group. Highlight single isolated labels, small groups,
        or everything outside the largest group (the one root you keep).
    "Calculate and highlight" does the work (it can take a while on large
    images) and draws the picked labels in the label_selection layer.
    "Remove highlighted labels" deletes exactly those, as one undo step.
    """

    def __init__(self, viewer, parent=None):
        super().__init__('Select labels by size or isolation', parent)
        self.viewer = viewer
        self.ids = np.array([], dtype=np.int64)
        self._build()
        self.refresh_layer_choices()

    def _build(self):
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel('Labels:'))
        self.combo_labels = QComboBox()
        row.addWidget(self.combo_labels, stretch=1)
        layout.addLayout(row)

        self.cb_size = QCheckBox('By size (pixels)')
        self.cb_size.setChecked(True)
        layout.addWidget(self.cb_size)
        row = QHBoxLayout()
        row.addWidget(QLabel('smaller than'))
        self.spin_lower = QSpinBox()
        self.spin_lower.setRange(0, NO_LIMIT)
        self.spin_lower.setToolTip('Area in pixels; 0 = no lower limit.')
        row.addWidget(self.spin_lower)
        row.addWidget(QLabel('larger than'))
        self.spin_upper = QSpinBox()
        self.spin_upper.setRange(0, NO_LIMIT)
        self.spin_upper.setToolTip('Area in pixels; 0 = no upper limit.')
        row.addWidget(self.spin_upper)
        layout.addLayout(row)

        self.cb_iso = QCheckBox('Isolated labels / groups')
        self.cb_iso.setToolTip(
            'Labels that touch, or lie within the gap of each other, form '
            'one group.'
        )
        layout.addWidget(self.cb_iso)
        row = QHBoxLayout()
        row.addWidget(QLabel('gap (px)'))
        self.spin_gap = QSpinBox()
        self.spin_gap.setRange(0, 50)
        self.spin_gap.setValue(2)
        self.spin_gap.setToolTip(
            'Labels this close count as touching (2 = across a 1 px wall).'
        )
        row.addWidget(self.spin_gap)
        row.addWidget(QLabel('groups of up to'))
        self.spin_cells = QSpinBox()
        self.spin_cells.setRange(0, 100000)
        self.spin_cells.setValue(1)
        self.spin_cells.setToolTip(
            'Highlight groups with at most this many labels. 1 = single '
            'isolated labels; 0 = off.'
        )
        row.addWidget(self.spin_cells)
        row.addWidget(QLabel('labels'))
        layout.addLayout(row)
        self.cb_largest = QCheckBox('Everything outside the largest group')
        self.cb_largest.setToolTip(
            'Keeps the biggest connected group (one root) and highlights all '
            'the rest.'
        )
        layout.addWidget(self.cb_largest)

        self.btn_calc = QPushButton('Calculate and highlight')
        self.btn_calc.clicked.connect(self._on_calculate)
        layout.addWidget(self.btn_calc)
        self.lbl = QLabel('')
        self.lbl.setWordWrap(True)
        layout.addWidget(self.lbl)
        row = QHBoxLayout()
        self.btn_remove = QPushButton('Remove highlighted labels')
        self.btn_remove.setEnabled(False)
        self.btn_remove.clicked.connect(self._on_remove)
        self.btn_clear = QPushButton('Clear highlight')
        self.btn_clear.clicked.connect(self._clear)
        row.addWidget(self.btn_remove)
        row.addWidget(self.btn_clear)
        layout.addLayout(row)

    # ------------------------------------------------------------- layers
    def refresh_layer_choices(self):
        names = [
            la.name
            for la in self.viewer.layers
            if la._type_string == 'labels'
            and not la.name.startswith(
                ('flagged_', 'review_cell', SELECTION_LAYER)
            )
        ]
        keep = self.combo_labels.currentText()
        self.combo_labels.blockSignals(True)
        self.combo_labels.clear()
        self.combo_labels.addItems(names)
        if keep in names:
            self.combo_labels.setCurrentText(keep)
        self.combo_labels.blockSignals(False)

    def _labels(self):
        name = self.combo_labels.currentText()
        if name not in self.viewer.layers:
            self.lbl.setText('Choose a labels layer first.')
            return None
        layer = self.viewer.layers[name]
        if np.squeeze(layer.data).ndim != 2:
            self.lbl.setText('Only 2D labels are supported here.')
            return None
        return layer

    # ------------------------------------------------------------ highlight
    def _on_calculate(self):
        layer = self._labels()
        if layer is None:
            return
        if not (self.cb_size.isChecked() or self.cb_iso.isChecked()):
            self.lbl.setText('Tick "By size" and/or "Isolated labels".')
            return
        data = np.squeeze(np.asarray(layer.data))
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.lbl.setText('Calculating...')
        QApplication.processEvents()
        try:
            picked, parts = [], []
            if self.cb_size.isChecked():
                s = ls.size_ids(
                    data, self.spin_lower.value(), self.spin_upper.value()
                )
                picked.append(s)
                parts.append(f'{len(s)} by size')
            info = None
            if self.cb_iso.isChecked():
                s, info = ls.isolated_ids(
                    data,
                    self.spin_gap.value(),
                    self.spin_cells.value(),
                    self.cb_largest.isChecked(),
                )
                picked.append(s)
                parts.append(f'{len(s)} isolated')
        finally:
            QApplication.restoreOverrideCursor()
        self.ids = (
            np.unique(np.concatenate(picked)) if picked else self.ids[:0]
        )
        self._draw(layer, data)
        text = f'{len(self.ids)} labels highlighted ({", ".join(parts)}).'
        if info:
            text += (
                f' {info["groups"]} groups; the largest has '
                f'{info["kept"]} labels.'
            )
        self.lbl.setText(text)
        self.btn_remove.setEnabled(len(self.ids) > 0)

    def _draw(self, layer, data):
        v = self.viewer
        mask = np.isin(data, self.ids).astype(np.uint8)
        if SELECTION_LAYER in v.layers:
            sel = v.layers[SELECTION_LAYER]
            sel.data = mask
        else:
            sel = v.add_labels(mask, name=SELECTION_LAYER, opacity=0.9)
            try:
                from napari.utils.colormaps import DirectLabelColormap

                sel.colormap = DirectLabelColormap(
                    color_dict={None: (0, 0, 0, 0), 1: SELECTION_COLOR}
                )
            except (ImportError, AttributeError, ValueError, TypeError):
                pass
            sel.editable = False
        sel.scale, sel.translate = layer.scale, layer.translate
        v.layers.selection.active = layer

    def _clear(self):
        self.ids = self.ids[:0]
        if SELECTION_LAYER in self.viewer.layers:
            self.viewer.layers.remove(SELECTION_LAYER)
        self.btn_remove.setEnabled(False)
        self.lbl.setText('Highlight cleared.')

    # --------------------------------------------------------------- remove
    def _on_remove(self):
        layer = self._labels()
        if layer is None or len(self.ids) == 0:
            return
        data = np.asarray(layer.data)
        present = np.isin(self.ids, np.unique(data))
        ids = self.ids[present]
        if len(ids) == 0:
            self.lbl.setText('None of the highlighted labels exists any more.')
            return
        answer = QMessageBox.question(
            self,
            'Remove labels',
            f'Remove {len(ids)} highlighted labels from "{layer.name}"?\n'
            'You can undo this with Ctrl+Z.',
        )
        if answer != QMessageBox.Yes:
            return
        idx = np.nonzero(np.isin(data, ids))
        try:
            layer.data_setitem(idx, 0)
        except (AttributeError, TypeError):
            layer.data[idx] = 0
            layer.refresh()
        self._clear()
        self.lbl.setText(f'Removed {len(ids)} labels.')
