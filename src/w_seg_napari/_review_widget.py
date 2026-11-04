"""Batch tab box: pull the cells the error model flagged out of a labels layer and visit them."""

import numpy as np
from qtpy.QtCore import QTimer
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


DIM_OPACITY = 0.6  # the labels layer while cells are highlighted
SETTLE_MS = 150  # wait after an edit, so a brush stroke counts as one edit


class ReviewFlaggedWidget(QGroupBox):
    """
    1. Highlight: the cells of a labels layer whose error probability lies
       between a lower and an upper limit are copied to a flagged_<name>
       layer on top (opacity 1, same ids so the same colours), while the
       labels layer itself is dimmed. A cell that you edit (painted, merged,
       deleted, or the neighbour it was merged into) drops out of the
       highlight at once and shows as normal again.
    2. Visit: Next / Previous step through the cells that are still
       highlighted, worst first. The viewer moves to the cell, its outline is
       drawn in the review_cell layer, and the cell's label is selected in
       the labels layer, so you can fix it with the normal napari tools.
    """

    def __init__(self, viewer, parent=None):
        super().__init__('Review flagged cells', parent)
        self.viewer = viewer
        self.scores = None  # DataFrame from cell_scores for the chosen pair
        self.queue = []  # label ids of the extracted cells, worst first
        self.pos = -1
        self._p = {}  # label id -> error probability of the extracted cells
        self._areas = {}  # label id -> pixel count when it was highlighted
        self._watched = None  # (labels layer, [(event, callback)])
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(SETTLE_MS)
        self._timer.timeout.connect(self._sync_highlight)
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

        btns = QHBoxLayout()
        self.btn_extract = QPushButton('Highlight cells')
        self.btn_extract.setToolTip(
            'Highlights the cells in this range (again, from the current '
            'labels): this also resets the highlight.'
        )
        self.btn_extract.clicked.connect(self._on_extract)
        self.btn_clear = QPushButton('Clear highlight')
        self.btn_clear.clicked.connect(self._on_clear)
        btns.addWidget(self.btn_extract)
        btns.addWidget(self.btn_clear)
        layout.addLayout(btns)

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
        self._p = dict(
            zip(chosen['label_id'].astype(int), chosen['p_error'], strict=True)
        )
        self._areas = dict(
            zip(
                chosen['label_id'].astype(int),
                chosen['area'].astype(int),
                strict=True,
            )
        )
        data = fc.extract(labels.data, self.queue)
        name = f'{FLAGGED_PREFIX}{labels.name}'
        if name in self.viewer.layers:
            overlay = self.viewer.layers[name]
            overlay.data = data
        else:
            overlay = self.viewer.add_labels(
                data,
                name=name,
                scale=labels.scale,
                translate=labels.translate,
            )
        overlay.opacity = 1.0
        overlay.editable = False
        try:
            overlay.colormap = (
                labels.colormap
            )  # the same colours as the labels
        except (AttributeError, ValueError, TypeError):
            pass
        if labels.opacity > DIM_OPACITY:
            labels.opacity = DIM_OPACITY
        self._watch(labels)
        self._report()
        self.lbl_cell.setText('')
        self._enable_nav(bool(self.queue))
        if self.queue:
            self._step(+1)

    def _on_clear(self):
        self._unwatch()
        for layer in list(self.viewer.layers):
            if layer.name.startswith((FLAGGED_PREFIX, CURRENT_LAYER)):
                self.viewer.layers.remove(layer)
        self.queue, self.pos, self._areas = [], -1, {}
        self.lbl_cell.setText('')
        self._enable_nav(False)
        self.lbl_count.setText('Highlight cleared.')

    # --------------------------------------------- edits drop out of the highlight
    def _watch(self, labels):
        """Call _on_target_edited whenever the labels layer is edited."""
        self._unwatch()
        hooks = []
        for name in ('labels_update', 'set_data'):
            event = getattr(labels.events, name, None)
            if event is not None:
                event.connect(self._on_target_edited)
                hooks.append((event, self._on_target_edited))
        self._watched = (labels, hooks)

    def _unwatch(self):
        if self._watched is not None:
            for event, callback in self._watched[1]:
                try:
                    event.disconnect(callback)
                except (ValueError, TypeError, RuntimeError):
                    pass
        self._watched = None

    def _on_target_edited(self, *_):
        self._timer.start()  # restarts: waits until the editing pauses

    def _overlay(self, labels=None):
        labels = labels if labels is not None else self._layers()[0]
        if labels is None:
            return None
        name = f'{FLAGGED_PREFIX}{labels.name}'
        if name in self.viewer.layers:
            return self.viewer.layers[name]
        return None

    def _sync_highlight(self):
        """Takes the cells that were edited since out of the highlight."""
        if self._watched is None:
            return
        labels = self._watched[0]  # the layer that was highlighted
        overlay = self._overlay(labels)
        if overlay is None or not self._areas:
            return
        edited = fc.changed_ids(labels.data, self._areas)
        if not edited:
            return
        for cell in edited:
            self._areas.pop(cell, None)
        overlay.data[np.isin(overlay.data, edited)] = 0
        overlay.refresh()
        self._report()

    def _report(self):
        self.lbl_count.setText(
            f'{len(self._areas)} of {len(self.queue)} cells still highlighted.'
        )

    # ------------------------------------------------------------ visiting
    def _step(self, direction):
        labels, _err = self._layers()
        if labels is None or not self.queue:
            return
        start = self.pos + direction if self.pos >= 0 else 0
        overlay = self._overlay()
        still = overlay.data if overlay is not None else labels.data
        idx = fc.next_existing(self.queue, still, start, direction)
        if idx is None:
            self.lbl_cell.setText(
                'No more highlighted cells in that direction (cells you '
                'have edited are skipped).'
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
