# w_seg_napari/_ROI_selection_widget.py

import napari
import numpy as np
from napari.layers import Image, Labels, Shapes
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from skimage.draw import line


class ROISelectionWidget(QWidget):
    """Self-contained ROI selection panel.

    Handles tool selection, dynamic layer matching, UI event wiring, and matrix
    slicing/label filtering.
    """

    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent=parent)
        self.viewer = viewer
        self.layer_checkboxes = {}
        self.roi_layer: Shapes = None

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self._init_ui()
        self._connect_events()
        self.refresh_layer_choices()

    # -------------------------------------------------------------------------
    # UI Setup
    # -------------------------------------------------------------------------

    def _init_ui(self):
        # 1. Tool Selection Header & Buttons
        self.main_layout.addWidget(QLabel('<b>1. Selection Mode</b>'))
        tools_layout = QHBoxLayout()

        self.btn_box = QPushButton(' ▭ Box')
        self.btn_ellipse = QPushButton(' O Ellipse')
        self.btn_lasso = QPushButton(' ◌ Lasso')
        self.btn_poly = QPushButton(' ⬡ Polygon')

        self.btn_box.setToolTip('Box Selection (Rectangle) [Key R]')
        self.btn_ellipse.setToolTip('Ellipse Selection [Key E]')
        self.btn_lasso.setToolTip('Freehand Lasso Loop [Key L]')
        self.btn_poly.setToolTip(
            'Click-Point Polygon [Key P]\nRight-click or double-click to connect 1st and Nth points.'
        )

        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(False)  # Allows toggling off

        for i, btn in enumerate(
            [self.btn_box, self.btn_ellipse, self.btn_lasso, self.btn_poly]
        ):
            btn.setCheckable(True)
            self.tool_group.addButton(btn, i)
            tools_layout.addWidget(btn)

        self.main_layout.addLayout(tools_layout)

        # 2. Dynamic Target Layers Panel with Select/Deselect All Toggle
        target_header_layout = QHBoxLayout()
        target_header_layout.addWidget(QLabel('<b>2. Target Layers</b>'))

        self.btn_toggle_all = QPushButton('Deselect All')
        self.btn_toggle_all.setFlat(True)
        self.btn_toggle_all.setStyleSheet('color: #4a90e2; font-size: 11px;')
        target_header_layout.addWidget(
            self.btn_toggle_all, alignment=Qt.AlignRight
        )

        self.main_layout.addLayout(target_header_layout)

        self.layers_widget = QWidget()
        self.layers_layout = QVBoxLayout(self.layers_widget)
        self.layers_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.addWidget(self.layers_widget)

        # 3. Processing Options
        self.main_layout.addWidget(QLabel('<b>3. Options</b>'))

        self.chk_merge_shapes = QCheckBox('Merge shapes')
        self.chk_merge_shapes.setChecked(False)
        self.chk_merge_shapes.setToolTip(
            'Unchecked: Drawing a new shape deletes previous shapes.\n'
            'Checked: Combines all drawn shapes on layer into a single ROI mask.'
        )

        self.chk_invert = QCheckBox('Invert selection (Keep outside)')
        self.chk_invert.setToolTip(
            'Check to crop/filter the region outside selection.'
        )

        self.chk_keep_original = QCheckBox('Keep original layers')
        self.chk_keep_original.setChecked(True)
        self.chk_keep_original.setToolTip(
            'Creates new layer(s) with cropped result instead of modifying in-place.'
        )

        self.chk_keep_dimensions = QCheckBox('Keep original image dimensions')
        self.chk_keep_dimensions.setChecked(False)
        self.chk_keep_dimensions.setToolTip(
            'Pads areas outside ROI as zeroes but keeps the same shape as original canvas'
        )

        # Label Handling Mode Selection
        self.main_layout.addWidget(QLabel('<b>Labels Extraction Mode:</b>'))
        self.combo_label_mode = QComboBox()
        self.combo_label_mode.addItems(
            [
                'Keep 100% complete objects only (Drop partial)',
                'Keep full intact objects (Allow partial overlap)',
                'Clip label pixels strictly to selection boundary',
            ]
        )
        self.combo_label_mode.setToolTip(
            '• Keep 100% complete objects only: Any label with pixels outside the ROI selection is completely removed.\n'
            '• Keep full intact objects: Preserves the complete shape of any label that touches the ROI.\n'
            '• Clip label pixels: Cuts label shapes exactly along the crop box/selection boundary.'
        )

        self.main_layout.addWidget(self.chk_merge_shapes)
        self.main_layout.addWidget(self.chk_invert)
        self.main_layout.addWidget(self.chk_keep_original)
        self.main_layout.addWidget(self.chk_keep_dimensions)
        self.main_layout.addWidget(self.combo_label_mode)

        # 4. Action Buttons
        self.btn_crop = QPushButton('Crop / Filter Selected Area')
        self.btn_crop.setStyleSheet(
            'font-weight: bold; background-color: #2b5b84; color: white; padding: 6px;'
        )
        self.btn_clear = QPushButton('Clear Selection ROI')

        self.main_layout.addWidget(self.btn_crop)
        self.main_layout.addWidget(self.btn_clear)

    def _connect_events(self):
        # Connect tool buttons
        self.btn_box.clicked.connect(
            lambda checked: self._handle_tool_toggle(
                self.btn_box, 'rectangle', checked
            )
        )
        self.btn_ellipse.clicked.connect(
            lambda checked: self._handle_tool_toggle(
                self.btn_ellipse, 'ellipse', checked
            )
        )
        self.btn_lasso.clicked.connect(
            lambda checked: self._handle_tool_toggle(
                self.btn_lasso, 'lasso', checked
            )
        )
        self.btn_poly.clicked.connect(
            lambda checked: self._handle_tool_toggle(
                self.btn_poly, 'polygon', checked
            )
        )

        # Action buttons
        self.btn_crop.clicked.connect(self._execute_crop)
        self.btn_clear.clicked.connect(self._clear_roi)
        self.btn_toggle_all.clicked.connect(self._toggle_all_layers)

        # Viewer layer list signals
        self.viewer.layers.events.inserted.connect(self.refresh_layer_choices)
        self.viewer.layers.events.removed.connect(self.refresh_layer_choices)

    def refresh_layer_choices(self, event=None):
        """Refreshes layer checkboxes, showing ONLY visible layers."""
        previous_states = {
            name: cb.isChecked() for name, cb in self.layer_checkboxes.items()
        }

        for cb in self.layer_checkboxes.values():
            cb.deleteLater()
        self.layer_checkboxes.clear()

        for layer in self.viewer.layers:
            # 1. Connect layer visibility change events if not already connected
            try:
                layer.events.visible.disconnect(self.refresh_layer_choices)
            except (ValueError, TypeError, RuntimeError):
                pass
            layer.events.visible.connect(self.refresh_layer_choices)

            # 2. Filter: Must be Image or Labels, not ROI layer, AND currently visible
            if (
                isinstance(layer, (Image, Labels))
                and not layer.name.startswith('[ROI')
                and layer.visible
            ):
                cb = QCheckBox(f'{layer.name} ({type(layer).__name__})')
                is_checked = previous_states.get(layer.name, True)
                cb.setChecked(is_checked)

                self.layer_checkboxes[layer.name] = cb
                self.layers_layout.addWidget(cb)

        self._update_toggle_all_button_label()

    def _toggle_all_layers(self):
        """Switches between selecting all or deselecting all target layer checkboxes."""
        all_checked = all(
            cb.isChecked() for cb in self.layer_checkboxes.values()
        )
        target_state = not all_checked

        for cb in self.layer_checkboxes.values():
            cb.setChecked(target_state)

        self._update_toggle_all_button_label()

    def _update_toggle_all_button_label(self):
        """Updates text of the Select/Deselect All button based on current checklist states."""
        if not self.layer_checkboxes:
            self.btn_toggle_all.setText('Select All')
            self.btn_toggle_all.setEnabled(False)
            return
        self.btn_toggle_all.setEnabled(True)
        all_checked = all(
            cb.isChecked() for cb in self.layer_checkboxes.values()
        )
        self.btn_toggle_all.setText(
            'Deselect All' if all_checked else 'Select All'
        )

    # -------------------------------------------------------------------------
    # ROI Shape Layer & Slicing Methods
    # -------------------------------------------------------------------------

    def _ensure_roi_layer(self) -> Shapes:
        if self.roi_layer not in self.viewer.layers:
            self.roi_layer = self.viewer.add_shapes(
                name='[ROI Selection]',
                edge_color='cyan',
                face_color=[0, 1, 1, 0.2],
                edge_width=2,
            )
            # Sync button highlighting when mode changes
            self.roi_layer.events.mode.connect(self._on_roi_mode_changed)

            # Listen for shape data changes to auto-reset on completion
            self.roi_layer.events.data.connect(self._on_shape_created)

        return self.roi_layer

    def _on_roi_mode_changed(self, event=None):
        """Called whenever Napari's shape mode changes."""
        if not self.roi_layer:
            return

        current_mode = str(self.roi_layer.mode)

        # Synchronize Qt button checked states
        self._update_button_states(current_mode)

        # If mode switched to pan_zoom and Merge is off, safely prune older shapes
        if current_mode == 'pan_zoom':
            if (
                not self.chk_merge_shapes.isChecked()
                and len(self.roi_layer.data) > 1
            ):
                self._prune_to_last_shape()

    def _prune_to_last_shape(self):
        """Safely retain only the most recent shape and its type using public API."""
        if not self.roi_layer:
            return

        num_shapes = len(self.roi_layer.data)
        if num_shapes <= 1:
            return

        # Block event emission while removing older shapes
        self.roi_layer.events.data.block()
        try:
            # Select all shape indices EXCEPT the last one (0 to N-2)
            self.roi_layer.selected_data = set(range(num_shapes - 1))

            # Delete selected shapes — Napari updates shape types automatically
            self.roi_layer.remove_selected()

            # Deselect remaining shape so no bounding handles stay active
            self.roi_layer.selected_data = set()
        finally:
            self.roi_layer.events.data.unblock()

    def _on_shape_created(self, event=None):
        """Auto-switch to pan_zoom strictly when mouse button is NOT pressed (drawing finished)."""
        if not self.roi_layer:
            return

        current_mode = str(self.roi_layer.mode)

        # 1. Ignore if already in pan_zoom or select mode
        if current_mode in ('pan_zoom', 'select', 'direct'):
            return

        # 2. Ignore if layer contains no shapes yet
        if len(self.roi_layer.data) == 0:
            return

        # 3. CRITICAL: Check if user is still holding down left mouse button while dragging
        # If left mouse button is currently down, user is still actively drawing!
        mouse_buttons = QApplication.mouseButtons()
        if mouse_buttons & Qt.LeftButton:
            return

        # 4. Mouse is NOT down -> Shape creation interaction is finalized
        # Block signals temporarily to prevent event loops during mode switch
        self.roi_layer.events.data.block()
        self.roi_layer.events.mode.block()
        try:
            self.roi_layer.mode = 'pan_zoom'
        finally:
            self.roi_layer.events.data.unblock()
            self.roi_layer.events.mode.unblock()

        # Update UI state and prune older shapes
        self._on_roi_mode_changed()

    def _handle_tool_toggle(
        self, clicked_btn: QPushButton, target_mode: str, checked: bool
    ):
        """Handles tool button clicks."""
        shapes = self._ensure_roi_layer()
        self.viewer.layers.selection.active = shapes

        if not checked:
            shapes.mode = 'pan_zoom'
        else:
            # 1. Uncheck other tool buttons visually
            for btn in [
                self.btn_box,
                self.btn_ellipse,
                self.btn_lasso,
                self.btn_poly,
            ]:
                if btn != clicked_btn:
                    btn.setChecked(False)

            # 2. Deselect existing shape selections
            shapes.selected_data = set()

            # 3. Explicitly set shape mode
            if target_mode == 'ellipse':
                shapes.mode = 'add_ellipse'
            elif target_mode == 'lasso':
                try:
                    shapes.mode = 'add_polygon_lasso'
                except (ValueError, AttributeError):
                    shapes.mode = 'add_polygon'
            elif target_mode == 'rectangle':
                shapes.mode = 'add_rectangle'
            elif target_mode == 'polygon':
                shapes.mode = 'add_polygon'

    def _update_button_states(self, current_mode: str):
        """Synchronizes Qt button checked states with active mode."""
        for btn in [
            self.btn_box,
            self.btn_ellipse,
            self.btn_lasso,
            self.btn_poly,
        ]:
            btn.blockSignals(True)

        self.btn_box.setChecked(current_mode == 'add_rectangle')
        self.btn_ellipse.setChecked(current_mode == 'add_ellipse')
        self.btn_lasso.setChecked('lasso' in current_mode)
        self.btn_poly.setChecked(
            current_mode == 'add_polygon' and 'lasso' not in current_mode
        )

        for btn in [
            self.btn_box,
            self.btn_ellipse,
            self.btn_lasso,
            self.btn_poly,
        ]:
            btn.blockSignals(False)

    def _clear_roi(self):
        if self.roi_layer:
            self.roi_layer.data = []

    def _execute_crop(self):
        if not self.roi_layer or len(self.roi_layer.data) == 0:
            print('w-seg-napari: No ROI shape drawn.')
            return

        # Explicitly get the exact list of checked layer names
        target_layers = [
            name
            for name, cb in self.layer_checkboxes.items()
            if cb.isChecked()
        ]

        if not target_layers:
            print('w-seg-napari: No target layers selected.')
            return

        # Disconnect layer inserted/removed events temporarily so adding cropped layers doesn't reset UI mid-crop
        self.viewer.layers.events.inserted.disconnect(
            self.refresh_layer_choices
        )
        self.viewer.layers.events.removed.disconnect(
            self.refresh_layer_choices
        )

        try:
            keep_dims = self.chk_keep_dimensions.isChecked()
            # 1. Determine active shape coordinates
            if self.chk_merge_shapes.isChecked():
                shapes_to_use = self.roi_layer.data  # All shapes
            else:
                shapes_to_use = [self.roi_layer.data[-1]]  # Latest shape only

            # 2. Compute bounding box across selected target shape(s)
            all_verts = np.vstack(shapes_to_use)
            ymin, xmin = np.floor(all_verts.min(axis=0)).astype(int)
            ymax, xmax = np.ceil(all_verts.max(axis=0)).astype(int)

            for name in target_layers:
                # STRICT GUARD: Ensure layer exists AND is explicitly in the checked target list
                if name not in self.viewer.layers:
                    continue

                layer = self.viewer.layers[name]
                h, w = layer.data.shape[:2]

                y0, x0 = max(0, ymin), max(0, xmin)
                y1, x1 = min(h, ymax), min(w, xmax)

                if y1 <= y0 or x1 <= x0:
                    continue

                # 3. Rasterize mask strictly matching target shape choice
                masks = self.roi_layer.to_masks(mask_shape=(h, w))

                if self.chk_merge_shapes.isChecked():
                    mask_full = np.logical_or.reduce(masks)
                else:
                    # Explicitly use mask of the last shape
                    mask_full = masks[-1]

                if self.chk_invert.isChecked():
                    mask_full = ~mask_full

                if keep_dims:
                    # Retain full original canvas size
                    bbox = (0, h, 0, w)
                    local_mask = mask_full
                else:
                    local_mask = mask_full[y0:y1, x0:x1]
                    bbox = (y0, y1, x0, x1)

                if isinstance(layer, Image):
                    self._crop_image_layer(layer, bbox, local_mask)
                elif isinstance(layer, Labels):
                    self._filter_labels_layer(layer, bbox, local_mask)

        finally:
            # Re-connect layer insertion listeners and update UI checklist once finished
            self.viewer.layers.events.inserted.connect(
                self.refresh_layer_choices
            )
            self.viewer.layers.events.removed.connect(
                self.refresh_layer_choices
            )
            self.refresh_layer_choices()

    def _crop_image_layer(
        self, layer: Image, bbox: tuple, local_mask: np.ndarray
    ):
        y0, y1, x0, x1 = bbox
        keep_orig = self.chk_keep_original.isChecked()

        cropped_data = layer.data[y0:y1, x0:x1].copy()
        cropped_data[~local_mask] = 0

        new_name = f'{layer.name}_cropped' if keep_orig else layer.name
        new_translate = (layer.translate[0] + y0, layer.translate[1] + x0)

        if keep_orig:
            # 1. Store original opacity
            orig_opacity = layer.opacity

            # 2. Add new cropped layer with full original opacity
            new_layer = self.viewer.add_image(
                cropped_data, name=new_name, translate=new_translate
            )
            new_layer.opacity = orig_opacity

            # 3. Dim the original background layer to 50% of its current opacity
            # layer.opacity = orig_opacity * 0.5
            # Dim the original layer opacity to .5
            layer.opacity = 0.5
        else:
            layer.data = cropped_data
            layer.translate = new_translate

    def _filter_labels_layer(self, layer, bbox, local_mask):
        """Filters a Napari Labels layer according to the selected mode.

        Modes:
          0: Keep 100% complete objects only (Drop partials)
          1: Keep full intact objects (Include partials)
          2: Clip label pixels strictly to selection boundary
        """
        y0, y1, x0, x1 = bbox
        mode = self.combo_label_mode.currentIndex()

        # Extract sub-array corresponding to bounding box
        sliced_data = layer.data[y0:y1, x0:x1]

        if mode in (0, 1):
            # 1. Candidate IDs touching/overlapping the selection mask inside bbox
            candidate_ids = np.unique(sliced_data[local_mask])
            candidate_ids = candidate_ids[candidate_ids != 0]

            if len(candidate_ids) == 0:
                selected_ids = []
            elif mode == 1:
                # Mode 1: Keep full intact objects touching selection
                selected_ids = candidate_ids
            else:
                # Mode 0: Keep ONLY labels that are 100% inside the ROI (Drop partials)
                # Compute total pixel count for each candidate across the WHOLE layer
                unique_all, counts_all = np.unique(
                    layer.data, return_counts=True
                )
                total_counts = dict(zip(unique_all, counts_all))

                # Compute pixel count for each candidate STRICTLY inside the active ROI mask
                roi_pixels = sliced_data[local_mask]
                unique_roi, counts_roi = np.unique(
                    roi_pixels, return_counts=True
                )
                roi_counts = dict(zip(unique_roi, counts_roi))

                # Keep candidate ID only if its ROI pixel count equals its TOTAL pixel count
                selected_ids = [
                    lbl_id
                    for lbl_id in candidate_ids
                    if roi_counts.get(lbl_id, 0) == total_counts.get(lbl_id, 0)
                ]

            # Build full image mask with selected IDs
            if len(selected_ids) > 0:
                valid_mask = np.isin(layer.data, selected_ids)
            else:
                valid_mask = np.zeros_like(layer.data, dtype=bool)

            if self.chk_keep_original.isChecked():
                new_data = np.where(valid_mask, layer.data, 0)
                self.viewer.add_labels(
                    new_data,
                    name=f'[ROI] {layer.name}',
                    scale=layer.scale,
                    translate=layer.translate,
                )
            else:
                layer.data[~valid_mask] = 0
                layer.refresh()

        else:
            # Mode 2: Strictly clip label pixels to the selection boundary
            cropped_data = np.zeros_like(sliced_data)
            cropped_data[local_mask] = sliced_data[local_mask]

            if self.chk_keep_original.isChecked():
                self.viewer.add_labels(
                    cropped_data,
                    name=f'[ROI] {layer.name}',
                    scale=layer.scale,
                    translate=(
                        layer.translate[0] + y0 * layer.scale[0],
                        layer.translate[1] + x0 * layer.scale[1],
                    ),
                )
            else:
                layer.data[y0:y1, x0:x1] = cropped_data
                layer.data[:y0, :] = 0
                layer.data[y1:, :] = 0
                layer.data[:, :x0] = 0
                layer.data[:, x1:] = 0
                layer.refresh()


class LineLabelSelectionWidget(QWidget):
    """Widget for selecting full intact Napari Labels along drawn lines, polylines, or paths."""

    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent=parent)
        self.viewer = viewer
        self.roi_layer = None
        self.layer_checkboxes = {}

        self.main_layout = QVBoxLayout(self)
        self._init_ui()
        self._connect_events()

    def _init_ui(self):
        # 1. Tool Selection Header & Buttons
        self.main_layout.addWidget(QLabel('<b>1. Line Selection Tool</b>'))
        tools_layout = QHBoxLayout()

        self.btn_line = QPushButton(' ╱ Single Line')
        self.btn_path = QPushButton(' 〰 Freehand / Path')
        self.btn_poly = QPushButton(' ⬡ Polyline')

        self.btn_line.setToolTip('Straight 2-point Line [Key L]')
        self.btn_path.setToolTip('Freehand drawn path [Key P]')
        self.btn_poly.setToolTip('Multi-segment connected lines')

        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(False)

        for i, btn in enumerate([self.btn_line, self.btn_path, self.btn_poly]):
            btn.setCheckable(True)
            self.tool_group.addButton(btn, i)
            tools_layout.addWidget(btn)

        self.main_layout.addLayout(tools_layout)

        # 2. Dynamic Target Labels Panel with Select/Deselect All Toggle
        target_header_layout = QHBoxLayout()
        target_header_layout.addWidget(
            QLabel('<b>2. Target Labels Layers</b>')
        )

        self.btn_toggle_all = QPushButton('Deselect All')
        self.btn_toggle_all.setFlat(True)
        self.btn_toggle_all.setStyleSheet('color: #4a90e2; font-size: 11px;')
        target_header_layout.addWidget(
            self.btn_toggle_all, alignment=Qt.AlignRight
        )

        self.main_layout.addLayout(target_header_layout)

        self.layers_widget = QWidget()
        self.layers_layout = QVBoxLayout(self.layers_widget)
        self.layers_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.addWidget(self.layers_widget)

        # 3. Processing Options
        self.main_layout.addWidget(QLabel('<b>3. Extraction Options</b>'))

        self.chk_merge_shapes = QCheckBox('Merge multiple line strokes')
        self.chk_merge_shapes.setChecked(False)
        self.chk_merge_shapes.setToolTip(
            'Unchecked: Uses only the most recently drawn line.\n'
            'Checked: Combines all drawn lines on the layer to select labels.'
        )

        self.chk_keep_original = QCheckBox('Keep original layers')
        self.chk_keep_original.setChecked(True)
        self.chk_keep_original.setToolTip(
            'Creates a new layer with extracted labels instead of modifying in-place.'
        )

        self.main_layout.addWidget(self.chk_merge_shapes)
        self.main_layout.addWidget(self.chk_keep_original)

        # 4. Action Buttons
        self.btn_extract = QPushButton('Extract Intersecting Labels')
        self.btn_extract.setStyleSheet(
            'font-weight: bold; background-color: #2b5b84; color: white; padding: 6px;'
        )
        self.btn_clear = QPushButton('Clear Line ROI')

        self.main_layout.addWidget(self.btn_extract)
        self.main_layout.addWidget(self.btn_clear)

    def _connect_events(self):
        self.btn_line.clicked.connect(
            lambda checked: self._handle_tool_toggle(
                self.btn_line, 'line', checked
            )
        )
        self.btn_path.clicked.connect(
            lambda checked: self._handle_tool_toggle(
                self.btn_path, 'path', checked
            )
        )
        self.btn_poly.clicked.connect(
            lambda checked: self._handle_tool_toggle(
                self.btn_poly, 'polyline', checked
            )
        )

        self.btn_extract.clicked.connect(self._execute_extraction)
        self.btn_clear.clicked.connect(self._clear_roi)
        self.btn_toggle_all.clicked.connect(self._toggle_all_layers)

        self.viewer.layers.events.inserted.connect(self.refresh_layer_choices)
        self.viewer.layers.events.removed.connect(self.refresh_layer_choices)
        self.refresh_layer_choices()

    def _get_or_create_roi_layer(self):
        """Ensures a dedicated Shapes layer exists for drawing selection lines."""
        layer_name = '[ROI] Line Selection'
        if layer_name in self.viewer.layers:
            return self.viewer.layers[layer_name]

        roi_layer = self.viewer.add_shapes(
            name=layer_name,
            edge_color='white',
            edge_width=2,
            face_color='transparent',
        )
        self.roi_layer = roi_layer
        return roi_layer

    def _handle_tool_toggle(self, btn, shape_type, checked):
        roi_layer = self._get_or_create_roi_layer()
        self.viewer.layers.selection.active = roi_layer

        if checked:
            for b in self.tool_group.buttons():
                if b != btn:
                    b.setChecked(False)
            roi_layer.mode = f'add_{shape_type}'
        else:
            roi_layer.mode = 'pan_zoom'

    def refresh_layer_choices(self, event=None):
        """Refreshes layer checkboxes, showing ONLY visible Labels layers."""
        previous_states = {
            name: cb.isChecked() for name, cb in self.layer_checkboxes.items()
        }

        for cb in self.layer_checkboxes.values():
            cb.deleteLater()
        self.layer_checkboxes.clear()

        for layer in self.viewer.layers:
            try:
                layer.events.visible.disconnect(self.refresh_layer_choices)
            except (ValueError, TypeError, RuntimeError):
                pass
            layer.events.visible.connect(self.refresh_layer_choices)

            # Labels layers only, excluding ROI layers and hidden layers
            if (
                isinstance(layer, Labels)
                and not layer.name.startswith('[ROI')
                and layer.visible
            ):
                cb = QCheckBox(f'{layer.name} ({type(layer).__name__})')
                is_checked = previous_states.get(layer.name, True)
                cb.setChecked(is_checked)

                self.layer_checkboxes[layer.name] = cb
                self.layers_layout.addWidget(cb)

        self._update_toggle_all_button_label()

    def _toggle_all_layers(self):
        all_checked = all(
            cb.isChecked() for cb in self.layer_checkboxes.values()
        )
        target_state = not all_checked

        for cb in self.layer_checkboxes.values():
            cb.setChecked(target_state)

        self._update_toggle_all_button_label()

    def _update_toggle_all_button_label(self):
        if not self.layer_checkboxes:
            self.btn_toggle_all.setText('Select All')
            self.btn_toggle_all.setEnabled(False)
            return

        self.btn_toggle_all.setEnabled(True)
        all_checked = all(
            cb.isChecked() for cb in self.layer_checkboxes.values()
        )
        self.btn_toggle_all.setText(
            'Deselect All' if all_checked else 'Select All'
        )

    def _clear_roi(self):
        if self.roi_layer and self.roi_layer in self.viewer.layers:
            self.roi_layer.data = []

    def _rasterize_line_shapes(self, shapes, mask_shape):
        """Rasterizes line/polyline/path shape coordinates onto a 2D boolean mask."""
        line_mask = np.zeros(mask_shape, dtype=bool)
        h, w = mask_shape

        for shape in shapes:
            coords = np.round(shape).astype(int)

            for i in range(len(coords) - 1):
                y0, x0 = coords[i]
                y1, x1 = coords[i + 1]

                rr, cc = line(y0, x0, y1, x1)
                valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                line_mask[rr[valid], cc[valid]] = True

        return line_mask

    def _execute_extraction(self):
        if not self.roi_layer or len(self.roi_layer.data) == 0:
            print('w-seg-napari: No line selection drawn.')
            return

        target_layers = [
            name
            for name, cb in self.layer_checkboxes.items()
            if cb.isChecked()
        ]

        if not target_layers:
            print('w-seg-napari: No target Labels layers selected.')
            return

        # Temporarily disconnect event listeners during layer updates
        self.viewer.layers.events.inserted.disconnect(
            self.refresh_layer_choices
        )
        self.viewer.layers.events.removed.disconnect(
            self.refresh_layer_choices
        )

        try:
            if self.chk_merge_shapes.isChecked():
                shapes_to_use = self.roi_layer.data
            else:
                shapes_to_use = [self.roi_layer.data[-1]]

            for name in target_layers:
                if name not in self.viewer.layers:
                    continue

                layer = self.viewer.layers[name]
                h, w = layer.data.shape[:2]

                # Rasterize 1-pixel wide line path
                line_mask = self._rasterize_line_shapes(shapes_to_use, (h, w))

                # Identify all unique label IDs touched by the line path
                intersecting_ids = np.unique(layer.data[line_mask])
                intersecting_ids = intersecting_ids[intersecting_ids != 0]

                if len(intersecting_ids) == 0:
                    continue

                # Build mask for full intact objects matching the intersecting IDs
                valid_mask = np.isin(layer.data, intersecting_ids)

                if self.chk_keep_original.isChecked():
                    new_data = np.where(valid_mask, layer.data, 0)
                    self.viewer.add_labels(
                        new_data,
                        name=f'[Line ROI] {layer.name}',
                        scale=layer.scale,
                        translate=layer.translate,
                    )
                else:
                    layer.data[~valid_mask] = 0
                    layer.refresh()

        finally:
            self.viewer.layers.events.inserted.connect(
                self.refresh_layer_choices
            )
            self.viewer.layers.events.removed.connect(
                self.refresh_layer_choices
            )
            self.refresh_layer_choices()
