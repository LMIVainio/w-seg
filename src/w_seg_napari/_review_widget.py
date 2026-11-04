"""Batch tab box: pull the cells the error model flagged out of a labels layer and visit them."""

import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from w_seg_napari import _flagged_cells as fc

FLAGGED_PREFIX = 'flagged_'
CURRENT_LAYER = 'review_cell'


class ReviewFlaggedWidget(QGroupBox):
    """
    1. Extract: the cells of a labels layer whose error probability lies
       between a lower and an upper limit go into a new flagged_<name> layer.
    2. Visit: Next / Previous step through them, worst first. The viewer
       moves to the cell, its outline is drawn in the review_cell layer, and
       the cell's label is selected in the labels layer, so you can fix it
       with the normal napari tools right there (pick / fill / erase).
    """

    def __init__(self, viewer, parent=None):
        super().__init__('Review flagged cells', parent)
        self.viewer = viewer
        self.scores = None  # DataFrame from cell_scores for the chosen pair
        self.queue = []  # label ids of the extracted cells, worst first
        self.pos = -1
        self._p = {}  # label id -> error probability of the extracted cells
        self._init_ui()

    # ------------------------------------------------------------------ UI
    def _init_ui(self):
        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel('Labels:'))
        self.combo_labels = QComboBox()
        self.combo_labels.setToolTip(
            'The labels layer to take the cells from.'
        )
        row.addWidget(self.combo_labels, stretch=1)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('Error layer:'))
        self.combo_error = QComboBox()
        self.combo_error.setToolTip(
            'The error_<image> layer made by a batch run with error flagging '
            '(each cell filled with its error probability).'
        )
        row.addWidget(self.combo_error, stretch=1)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('Error between'))
        self.spin_lower = self._spin(0.5)
        self.spin_upper = self._spin(1.0)
        row.addWidget(self.spin_lower)
        row.addWidget(QLabel('and'))
        row.addWidget(self.spin_upper)
        layout.addLayout(row)
        self.spin_lower.setToolTip(
            'Cells below this error probability are left out.'
        )
        self.spin_upper.setToolTip(
            'Cells above this are left out, e.g. the very sure ones that '
            'you handle separately.'
        )

        self.lbl_count = QLabel('')
        self.lbl_count.setWordWrap(True)
        layout.addWidget(self.lbl_count)

        self.btn_extract = QPushButton('Extract cells to a new layer')
        self.btn_extract.clicked.connect(self._on_extract)
        layout.addWidget(self.btn_extract)

        nav = QHBoxLayout()
        self.btn_prev = QPushButton('< Previous')
        self.btn_next = QPushButton('Next >')
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next.clicked.connect(lambda: self._step(+1))
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        layout.addLayout(nav)
        self.lbl_cell = QLabel('')
        self.lbl_cell.setWordWrap(True)
        layout.addWidget(self.lbl_cell)
        self._enable_nav(False)

        self.combo_labels.currentIndexChanged.connect(self._on_labels_changed)
        self.combo_error.currentIndexChanged.connect(self._rescore)
        self.spin_lower.valueChanged.connect(self._update_count)
        self.spin_upper.valueChanged.connect(self._update_count)

    @staticmethod
    def _spin(value):
        s = QDoubleSpinBox()
        s.setRange(0.0, 1.0)
        s.setSingleStep(0.05)
        s.setDecimals(2)
        s.setValue(value)
        return s

    def _enable_nav(self, on):
        self.btn_prev.setEnabled(on)
        self.btn_next.setEnabled(on)

    # ------------------------------------------------------------- choices
    def refresh_layer_choices(self):
        """Refills the two layer boxes (called when layers come and go)."""
        labels = [
            la.name
            for la in self.viewer.layers
            if la._type_string == 'labels'
            and not la.name.startswith((FLAGGED_PREFIX, CURRENT_LAYER))
        ]
        errors = [
            la.name
            for la in self.viewer.layers
            if la._type_string == 'image' and la.name.startswith('error_')
        ]
        before = (
            self.combo_labels.currentText(),
            self.combo_error.currentText(),
        )
        for combo, items in (
            (self.combo_labels, labels),
            (self.combo_error, errors),
        ):
            keep = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            if keep in items:
                combo.setCurrentText(keep)
            combo.blockSignals(False)
        if self.combo_error.currentText() != before[1] or not before[1]:
            self._pair()
        if (
            self.combo_labels.currentText(),
            self.combo_error.currentText(),
        ) != before:
            self.scores = None  # scored again when needed

    def _pair(self):
        errors = [
            self.combo_error.itemText(i)
            for i in range(self.combo_error.count())
        ]
        match = fc.pair_error_layer(self.combo_labels.currentText(), errors)
        if match:
            self.combo_error.blockSignals(True)
            self.combo_error.setCurrentText(match)
            self.combo_error.blockSignals(False)

    def _on_labels_changed(self, *_):
        self._pair()
        self._rescore()

    def _layers(self):
        """(labels layer, error layer) or (None, None)."""
        v = self.viewer
        a, b = self.combo_labels.currentText(), self.combo_error.currentText()
        if a in v.layers and b in v.layers:
            return v.layers[a], v.layers[b]
        return None, None

    # ------------------------------------------------------------- scoring
    def _rescore(self, *_):
        labels, err = self._layers()
        self.scores = None
        if labels is None:
            self.lbl_count.setText(
                'Needs a labels layer and an error_<image> layer (run the '
                'batch with "Flag likely segmentation errors").'
            )
            return
        try:
            self.scores = fc.cell_scores(labels.data, err.data)
        except ValueError as e:
            self.lbl_count.setText(str(e))
            return
        self._update_count()

    def _update_count(self, *_):
        if self.scores is None:
            self._rescore()  # (calls this again once it has scores)
            return
        n = len(
            fc.select_range(
                self.scores, self.spin_lower.value(), self.spin_upper.value()
            )
        )
        self.lbl_count.setText(
            f'{n} of {len(self.scores)} cells in this range.'
        )

    # ------------------------------------------------------------- extract
    def _on_extract(self):
        labels, err = self._layers()
        if labels is None:
            self._rescore()
            return
        self._rescore()  # the labels may have been edited since
        if self.scores is None:
            return
        chosen = fc.select_range(
            self.scores, self.spin_lower.value(), self.spin_upper.value()
        )
        self.queue = [int(i) for i in chosen['label_id']]
        self.pos = -1
        self._p = dict(zip(chosen['label_id'].astype(int), chosen['p_error']))
        data = fc.extract(labels.data, self.queue)
        name = f'{FLAGGED_PREFIX}{labels.name}'
        if name in self.viewer.layers:
            self.viewer.layers[name].data = data
        else:
            self.viewer.add_labels(
                data,
                name=name,
                opacity=0.8,
                scale=labels.scale,
                translate=labels.translate,
            )
        self.lbl_count.setText(
            f'{len(self.queue)} cells extracted to "{name}". '
            'Use Next / Previous to visit them, worst first.'
        )
        self.lbl_cell.setText('')
        self._enable_nav(bool(self.queue))
        if self.queue:
            self._step(+1)

    # ------------------------------------------------------------ visiting
    def _step(self, direction):
        labels, _err = self._layers()
        if labels is None or not self.queue:
            return
        start = self.pos + direction if self.pos >= 0 else 0
        idx = fc.next_existing(self.queue, labels.data, start, direction)
        if idx is None:
            self.lbl_cell.setText(
                'No more cells in that direction (merged or deleted cells '
                'are skipped).'
            )
            return
        self.pos = idx
        self._show_cell(labels, self.queue[idx], idx)

    def _show_cell(self, labels, cell_id, idx):
        v = self.viewer
        centre = fc.find_cell(labels.data, cell_id)
        if centre is None:
            return
        # outline of the current cell
        outline = (np.asarray(labels.data) == cell_id) * cell_id
        if CURRENT_LAYER in v.layers:
            v.layers[CURRENT_LAYER].data = outline
        else:
            layer = v.add_labels(
                outline,
                name=CURRENT_LAYER,
                opacity=1.0,
                scale=labels.scale,
                translate=labels.translate,
            )
            try:
                layer.contour = 3
            except (AttributeError, ValueError):
                pass
            layer.editable = False
        # move there and make the labels layer ready for editing this cell
        world = tuple(
            c * s + t
            for c, s, t in zip(
                centre, labels.scale[-2:], labels.translate[-2:]
            )
        )
        try:
            v.camera.center = (0.0, *world)
        except (AttributeError, ValueError, TypeError):
            pass
        labels.selected_label = int(cell_id)
        v.layers.selection.active = labels
        p = self._p.get(int(cell_id))
        self.lbl_cell.setText(
            f'Cell {idx + 1} of {len(self.queue)}: label {cell_id}'
            + (f', error {p:.2f}' if p is not None else '')
        )
