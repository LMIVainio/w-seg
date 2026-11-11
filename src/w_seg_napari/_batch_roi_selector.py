"""Batch tab panel: draw the region of each image that the batch works on."""

from pathlib import Path

import numpy as np
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from w_seg_napari import _batch_rois as br
from w_seg_napari._batch_io import load_image

IMAGE_LAYER = 'region_image'
SHAPE_LAYER = 'region'


class RegionSelectorWidget(QGroupBox):
    """
    Steps through the images of the folder. Each image is shown in the viewer;
    draw one rectangle around the part to analyse (for example one root) and
    press Next. No rectangle means the whole image. The regions are written
    to batch_rois.json in the results folder as you go.
    """

    closed = Signal()

    def __init__(self, viewer, parent=None):
        super().__init__('Select regions', parent)
        self.viewer = viewer
        self.files, self.folder, self.out_dir = [], None, None
        self.rois = {}
        self.index = 0
        self._shape = None  # shape of the image on display
        self._pruning = False
        self._build()
        self.setVisible(False)

    def _build(self):
        layout = QVBoxLayout(self)
        self.lbl = QLabel('')
        self.lbl.setWordWrap(True)
        layout.addWidget(self.lbl)
        row = QHBoxLayout()
        self.btn_prev = QPushButton('Previous')
        self.btn_next = QPushButton('Next')
        self.btn_clear = QPushButton('Whole image')
        self.btn_clear.setToolTip('Remove the rectangle: process all of it.')
        self.btn_done = QPushButton('Finish')
        self.btn_prev.clicked.connect(lambda: self._go(-1))
        self.btn_next.clicked.connect(lambda: self._go(+1))
        self.btn_clear.clicked.connect(self._clear_shape)
        self.btn_done.clicked.connect(self.finish)
        for b in (self.btn_prev, self.btn_next, self.btn_clear, self.btn_done):
            row.addWidget(b)
        layout.addLayout(row)

    # ------------------------------------------------------------ session
    def start(self, files, folder, out_dir, first=0):
        self.files, self.folder, self.out_dir = list(files), folder, out_dir
        self.rois = br.load_rois(out_dir)
        self._active = True
        self.index = min(max(first, 0), len(self.files) - 1)
        self.setVisible(True)
        self._show()

    def finish(self):
        if getattr(self, '_active', False):
            self._store()
            self._active = False
        self._remove_layers()
        self.setVisible(False)
        self.lbl.setText('')
        self.closed.emit()

    def n_regions(self):
        return len(self.rois)

    # ------------------------------------------------------------- layers
    def _remove_layers(self):
        for name in (SHAPE_LAYER, IMAGE_LAYER):
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)

    def _show(self):
        name = self.files[self.index]
        try:
            image = load_image(Path(self.folder) / name)
        except (OSError, ValueError) as e:
            self.lbl.setText(f'{name}: cannot be read ({e})')
            return
        self._shape = image.shape
        v = self.viewer
        self._remove_layers()
        v.add_image(image, name=IMAGE_LAYER, rgb=True)
        shapes = v.add_shapes(
            name=SHAPE_LAYER,
            edge_color='yellow',
            face_color='transparent',
            edge_width=3,
        )
        saved = self.rois.get(name)
        if saved:
            y0, x0, y1, x1 = saved
            shapes.add_rectangles(
                np.array([[[y0, x0], [y1, x1]]], float),
            )
        shapes.events.data.connect(self._keep_last)
        v.layers.selection.active = shapes
        shapes.mode = 'add_rectangle'
        v.reset_view()
        self._update_label()

    def _keep_last(self, event=None):
        """One region per image: drawing a new rectangle replaces the old."""
        if self._pruning or SHAPE_LAYER not in self.viewer.layers:
            return
        shapes = self.viewer.layers[SHAPE_LAYER]
        if len(shapes.data) > 1:
            self._pruning = True
            try:
                shapes.selected_data = set(range(len(shapes.data) - 1))
                shapes.remove_selected()
            finally:
                self._pruning = False
        self._update_label()

    def _clear_shape(self):
        if SHAPE_LAYER in self.viewer.layers:
            shapes = self.viewer.layers[SHAPE_LAYER]
            shapes.selected_data = set(range(len(shapes.data)))
            shapes.remove_selected()
        self._update_label()

    # ---------------------------------------------------------- navigation
    def _drawn(self):
        """The rectangle on display as a region, None if there is none."""
        if SHAPE_LAYER not in self.viewer.layers or self._shape is None:
            return None
        data = self.viewer.layers[SHAPE_LAYER].data
        if not len(data):
            return None
        return br.roi_from_vertices(data[-1], self._shape)

    def _store(self):
        name = self.files[self.index]
        roi = self._drawn()
        if roi is None:
            self.rois.pop(name, None)
        else:
            self.rois[name] = roi
        br.save_rois(self.out_dir, self.rois)

    def _go(self, step):
        self._store()
        new = self.index + step
        if not 0 <= new < len(self.files):
            if step > 0:
                self.finish()
            return
        self.index = new
        self._show()

    def _update_label(self):
        name = self.files[self.index]
        roi = self._drawn()
        what = (
            f'region {roi}'
            if roi is not None
            else 'whole image (no rectangle)'
        )
        self.lbl.setText(
            f'Image {self.index + 1} of {len(self.files)}: {name}\n{what}. '
            f'{len(self.rois)} image(s) with a region saved. Draw a rectangle '
            'around the part to analyse, then Next.'
        )
        self.btn_prev.setEnabled(self.index > 0)
        self.btn_next.setText(
            'Next' if self.index < len(self.files) - 1 else 'Save and close'
        )
