import sys
from pathlib import Path

import napari
from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import w_seg_napari._wsegutils as w
from w_seg_napari._batch_widget import BatchTabWidget
from w_seg_napari._cell_classification_widget import CellClassificationWidget
from w_seg_napari._combine_layers_widget import create_layer_combination_widget
from w_seg_napari._gasp_widget import GaspWidget
from w_seg_napari._morphology_widget import MorphologyWidget
from w_seg_napari._ROI_selection_widget import (
    LineLabelSelectionWidget,
    ROISelectionWidget,
)
from w_seg_napari._superpixel_widget import SuperpixelWidget
from w_seg_napari._w_net_widget import WNetWidget
from w_seg_napari._wsegutils import YOLO_WEIGHT_URL_MAP, CurationWidget

# Import your widget instantiators / utilities
# import w_seg_napari._wsegutils as w


class Tab1Pipeline(QScrollArea):
    model_changed = Signal(str)

    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Ensure scrollbar only appears when content actually overflows
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)

        self.device = w.get_device()
        self.model_state = {'generator': None, 'device': self.device}
        self.loaded_weights = 'No weights loaded'

        # Build UI
        container = QWidget()
        container.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.MinimumExpanding
        )
        layout = QVBoxLayout(container)

        # Instantiate Sub-widgets
        selected_generator = self.model_state['generator']
        self.w_net_widget = WNetWidget(
            viewer=self.viewer,
            generator=selected_generator,
            device=self.device,
        )
        self.superpixel_widget = SuperpixelWidget(viewer=self.viewer)
        self.gasp_widget = GaspWidget(viewer=self.viewer)

        layout.addWidget(self._build_model_group())
        layout.addWidget(self.w_net_widget)
        layout.addWidget(self.superpixel_widget)
        layout.addWidget(self.gasp_widget)
        layout.addStretch()

        container.setLayout(layout)
        self.setWidget(container)

    # Ugly size hint override on windows - it should fallback on default on other systems
    def sizeHint(self) -> QSize:
        if sys.platform == 'win32':
            return QSize(400, 1200)
        return super().sizeHint()

    def _build_model_group(self) -> QGroupBox:
        model_group = QGroupBox('Model & Weights Selection')
        layout = QVBoxLayout()

        self.combo_generator = QComboBox()
        self.combo_generator.addItems(['W-seg'])

        self.combo_weights = QComboBox()
        self._populate_weights()

        self.btn_browse = QPushButton('📂 Add model bundle folder...')
        self.btn_browse.setToolTip(
            'A model bundle is a folder with model.keras and config.json, '
            'written by the training code. Bundles stored in ~/.wseg_models '
            'are listed automatically.'
        )
        self.btn_browse.clicked.connect(self._on_browse_bundle)

        self.btn_reload = QPushButton('🔄 Load / Reload Model')
        self.status_label = QLabel(
            f'Device: {self.device.type.upper()} | Model not loaded'
        )

        self.btn_reload.clicked.connect(self._on_reload_clicked)
        self.combo_weights.currentTextChanged.connect(self._emit_model_change)

        layout.addWidget(QLabel('Generator Architecture:'))
        layout.addWidget(self.combo_generator)
        layout.addWidget(QLabel('Staining Weights:'))
        layout.addWidget(self.combo_weights)
        layout.addWidget(self.btn_browse)
        layout.addWidget(self.btn_reload)
        layout.addWidget(self.status_label)
        model_group.setLayout(layout)

        # Trigger default model load on init
        self._on_reload_clicked()
        return model_group

    def _populate_weights(self):
        """Old-style weights (downloaded by name) plus bundles found in ~/.wseg_models."""
        self.combo_weights.blockSignals(True)
        self.combo_weights.clear()
        for w_name in w.WEIGHT_URL_MAP:
            self.combo_weights.addItem(w_name, userData=w_name)
        for name, folder in w.list_local_bundles():
            self.combo_weights.addItem(f'[bundle] {name}', userData=folder)
        self.combo_weights.blockSignals(False)

    def _on_browse_bundle(self):
        start = w.BUNDLE_DIR if w.BUNDLE_DIR.is_dir() else Path.home()
        folder = QFileDialog.getExistingDirectory(
            self,
            'Select a model bundle folder (contains model.keras and config.json)',
            str(start),
        )
        if not folder:
            return
        if w.find_bundle(folder) is None:
            QMessageBox.warning(
                self,
                'Not a model bundle',
                'This folder does not contain both model.keras and config.json.',
            )
            return
        label = f'[bundle] {Path(folder).name}'
        index = self.combo_weights.findText(label)
        if index == -1:
            self.combo_weights.addItem(label, userData=folder)
            index = self.combo_weights.findText(label)
        self.combo_weights.setCurrentIndex(index)
        self._on_reload_clicked()

    def _emit_model_change(self, text: str):
        # currentData() can be None or an object/Path; fall back to text string
        selected_weight = self.combo_weights.currentData()
        if not selected_weight or not str(selected_weight).strip():
            selected_weight = text or self.combo_weights.currentText()

        if selected_weight:
            self.model_changed.emit(str(selected_weight))

    def _on_reload_clicked(self):
        selected_weight_identifier = (
            self.combo_weights.currentData()
            or self.combo_weights.currentText()
        )
        self.status_label.setText('Loading model weights...')
        self.viewer.status = (
            f'[w-seg] Loading {self.combo_weights.currentText()}...'
        )

        try:
            new_generator = w.load_my_generator(
                device=self.device, w_name=selected_weight_identifier
            )
            self.model_state['generator'] = new_generator

            if hasattr(self, 'w_net_widget'):
                self.w_net_widget.generator = new_generator

            self.status_label.setText(
                f'Active: {self.combo_weights.currentText()} ({self.device.type.upper()})'
            )
            self.viewer.status = f'[w-seg] Successfully loaded {self.combo_weights.currentText()}'
            self.loaded_weights = self.combo_weights.currentText()
            # Send signal to morphology widget
            self.model_changed.emit(selected_weight_identifier)
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            self.status_label.setText('❌ Error loading model!')
            self.viewer.status = f'[w-seg] Failed to load model: {str(e)}'

    def refresh_layer_choices(self):
        """Refreshes layer dropdown choices for widgets contained in this tab."""
        for widget in (
            self.w_net_widget,
            self.superpixel_widget,
            self.gasp_widget,
        ):
            if hasattr(widget, 'refresh_layer_choices'):
                widget.refresh_layer_choices()

    def get_settings(self) -> dict:
        """Returns a flat dictionary combining Tab1 state and all sub-widget parameters."""
        # Top-level tab parameters
        settings = {
            'generator': self.model_state.get('generator'),
            'device': self.device,
            'weights': self.loaded_weights,
        }

        # Safely extract flat sub-widget dictionaries
        w_net_settings = (
            self.w_net_widget.get_settings()
            if hasattr(self, 'w_net_widget')
            and hasattr(self.w_net_widget, 'get_settings')
            else {}
        )
        superpixel_settings = (
            self.superpixel_widget.get_settings()
            if hasattr(self, 'superpixel_widget')
            and hasattr(self.superpixel_widget, 'get_settings')
            else {}
        )
        gasp_settings = (
            self.gasp_widget.get_settings()
            if hasattr(self, 'gasp_widget')
            and hasattr(self.gasp_widget, 'get_settings')
            else {}
        )

        # Merge all sub-dictionaries into the flat top-level dictionary
        return {
            **settings,
            **w_net_settings,
            **superpixel_settings,
            **gasp_settings,
        }


class Tab2ROI_selection(QScrollArea):
    """Tab 2 Container in w-seg-napari."""

    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent=parent)
        self.viewer = viewer

        # QScrollArea Layout setup
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)

        # Internal scroll container
        self.container = QWidget()
        self.container.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.MinimumExpanding
        )
        self.main_layout = QVBoxLayout(self.container)

        # 1. GroupBox for ROI Selection
        self.roi_group = QGroupBox('ROI selection')
        roi_group_layout = QVBoxLayout(self.roi_group)
        roi_group_layout.setContentsMargins(8, 12, 8, 8)

        self.ROI_selection = ROISelectionWidget(viewer=self.viewer)
        roi_group_layout.addWidget(self.ROI_selection)
        self.main_layout.addWidget(self.roi_group)

        # 2. GroupBox for Select Labels Lineage
        self.lineage_group = QGroupBox('Select labels lineage')
        lineage_group_layout = QVBoxLayout(self.lineage_group)
        lineage_group_layout.setContentsMargins(8, 12, 8, 8)

        self.labels_line_selection = LineLabelSelectionWidget(
            viewer=self.viewer
        )
        lineage_group_layout.addWidget(self.labels_line_selection)
        self.main_layout.addWidget(self.lineage_group)

        self.refresh_layer_choices = self.ROI_selection.refresh_layer_choices
        self.refresh_labels_choices = (
            self.labels_line_selection.refresh_layer_choices
        )
        self.main_layout.addStretch()
        self.setWidget(self.container)


class Tab3Curation(QScrollArea):
    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Ensure scrollbar only appears when content actually overflows
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout()

        self.curation_panel = CurationWidget(self.viewer)

        # Instantiate widget instance using the factory
        self.layer_blender_widget = create_layer_combination_widget(
            self.viewer
        )

        layout.addWidget(self.curation_panel)
        layout.addWidget(self.layer_blender_widget.native)
        # layout.addStretch()

        container.setLayout(layout)
        self.setWidget(container)

    def refresh_layer_choices(self):
        """Refreshes dynamic layer choice dropdowns."""
        if hasattr(self.layer_blender_widget, 'layer_a'):
            self.layer_blender_widget.layer_a.reset_choices()
        if hasattr(self.layer_blender_widget, 'layer_b'):
            self.layer_blender_widget.layer_b.reset_choices()
        if hasattr(self.layer_blender_widget, 'output_layer'):
            self.layer_blender_widget.output_layer.reset_choices()
        if hasattr(self.layer_blender_widget, 'clean_target_layer'):
            self.layer_blender_widget.clean_target_layer.reset_choices()


class Tab4Analysis(QScrollArea):
    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Ensure scrollbar only appears when content actually overflows
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout()

        # Instantiate widgets and pass the cached data
        self.morphology_widget = MorphologyWidget(viewer=self.viewer)
        self.cell_classifier = CellClassificationWidget(
            viewer=self.viewer,
            df=self.morphology_widget.cached_df,
            labels_data=self.morphology_widget.cached_labels_data,
            feature_extraction_config=self.morphology_widget.cached_feature_extraction_config,
        )

        original_on_measure = self.morphology_widget._on_measure_clicked
        # Connect YOLO settings
        self.morphology_widget.use_YOLO_cb.toggled.connect(
            self._on_use_YOLO_changed
        )
        self._on_use_YOLO_changed(
            self.morphology_widget.use_YOLO_cb.isChecked()
        )

        def _on_measure_and_forward():
            # 1. Execute standard morphology calculation
            original_on_measure()

            # 2. Connect signal to forward results to classifier widget
            self.morphology_widget.measurement_finished_signal.connect(
                self.cell_classifier.bind_measured_data
            )

        # Bridge the "Apply Config" action from ClassificationWidget back to MorphologyWidget
        def _apply_model_config_to_morphology(config: dict):
            self.morphology_widget.set_feature_extraction_config(config)

        self.cell_classifier.on_apply_config_callback = (
            _apply_model_config_to_morphology
        )

        self.morphology_widget._on_measure_clicked = _on_measure_and_forward
        # Re-bind click event to wrapper
        # Safely disconnect any previously bound _on_measure_and_forward slot (if connected)
        try:
            self.morphology_widget.run_btn.clicked.disconnect(
                _on_measure_and_forward
            )
        except (TypeError, RuntimeError):
            # Safe catch in case the slot wasn't previously connected
            pass

        # Connect the new click handler
        self.morphology_widget.run_btn.clicked.connect(_on_measure_and_forward)

        # Frame each widget cleanly in a titled group box
        morphology_box = QGroupBox('1. Cell Morphology & CW Thickness')
        morphology_layout = QVBoxLayout()
        morphology_layout.addWidget(self.morphology_widget)
        morphology_box.setLayout(morphology_layout)

        classification_box = QGroupBox(
            '2. Cell Classification & Contact Topology'
        )
        classification_layout = QVBoxLayout()
        classification_layout.addWidget(self.cell_classifier)
        classification_box.setLayout(classification_layout)
        # Add group boxes to layout
        layout.addWidget(morphology_box)
        layout.addWidget(classification_box)
        # layout.addStretch()

        container.setLayout(layout)
        self.setWidget(container)

        if self.viewer is not None:
            self.viewer.layers.events.inserted.connect(
                self.refresh_layer_choices
            )
            self.viewer.layers.events.removed.connect(
                self.refresh_layer_choices
            )
            self.viewer.layers.events.reordered.connect(
                self.refresh_layer_choices
            )
        self.refresh_layer_choices()

    def refresh_layer_choices(self, event=None):
        """Refreshes layer dropdown choices manually from the viewer."""
        if self.viewer is None:
            return

        # Gather current layers directly from Napari viewer
        label_layers = [
            lay
            for lay in self.viewer.layers
            if isinstance(lay, napari.layers.Labels)
        ]
        image_layers = [
            lay
            for lay in self.viewer.layers
            if isinstance(lay, napari.layers.Image)
        ]

        # Update Morphology Widget
        if hasattr(self, 'morphology_widget'):
            if hasattr(self.morphology_widget, 'labels_layer'):
                self.morphology_widget.labels_layer.choices = label_layers
            if hasattr(self.morphology_widget, 'signal_layer'):
                self.morphology_widget.signal_layer.choices = image_layers

        # Update Classification Widget
        if hasattr(self, 'cell_classifier'):
            if hasattr(self.cell_classifier, 'labels_layer'):
                self.cell_classifier.labels_layer.choices = label_layers
            if hasattr(self.cell_classifier, 'signal_layer'):
                self.cell_classifier.signal_layer.choices = image_layers

    def update_model_selection(self, model_weight_name: str):
        """Called when Tab1 changes the active model weights"""

        if not model_weight_name or not str(model_weight_name).strip():
            print('[w-seg] Empty model_weight_name received. Aborting sync.')
            return

        name_lower = str(model_weight_name).lower()
        is_arabidopsis = 'arabidopsis' in name_lower
        is_populus = 'populus' in name_lower

        # 1. Update Checkbox safely
        if hasattr(self, 'morphology_widget') and hasattr(
            self.morphology_widget, 'radial_symmetry_cb'
        ):
            self.morphology_widget.radial_symmetry_cb.setChecked(
                is_arabidopsis
            )

        # 2. Update YOLO ComboBox
        if hasattr(self, 'morphology_widget') and hasattr(
            self.morphology_widget, 'YOLO_weights_combo'
        ):
            target_species = (
                'arabidopsis'
                if is_arabidopsis
                else ('populus' if is_populus else None)
            )

            if target_species:
                # Extract keys from  choices tuple/list format
                available_choices = [
                    c[0] if isinstance(c, tuple) else c
                    for c in self.morphology_widget.YOLO_weights_combo.choices
                ]
                matching_key = next(
                    (
                        key
                        for key in available_choices
                        if target_species in str(key).lower()
                    ),
                    None,
                )
                if matching_key is not None:
                    self.morphology_widget.YOLO_weights_combo.setCurrentText(
                        matching_key
                    )

    def _on_use_YOLO_changed(self, enabled: bool):
        self.morphology_widget.YOLO_weights_combo.enabled = enabled
        self.morphology_widget.YOLO_layer_combo.enabled = enabled
        if enabled:
            self.morphology_widget.YOLO_weights_combo.choices = list(
                YOLO_WEIGHT_URL_MAP.keys()
            )
        else:
            self.morphology_widget.YOLO_weights_combo.choices = []

    def get_settings(self):
        return self.cell_classifier.get_settings()


class Tab5Batch(QScrollArea):
    """Dedicated tab container for automated directory batch processing and HDF5 export."""

    def __init__(self, viewer: 'napari.Viewer', main_widget=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.main_widget = main_widget  # Reference to master plugin widget (e.g., CellClassificationWidget)

        # ----------------------------------------------------------------------
        # Scroll Area Setup
        # ----------------------------------------------------------------------
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ----------------------------------------------------------------------
        # Main Scroll Container & Layout
        # ----------------------------------------------------------------------
        self.container = QWidget()
        self.container.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.MinimumExpanding
        )

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ----------------------------------------------------------------------
        # Inner Batch Widget Instantiation
        # ----------------------------------------------------------------------
        # We pass self.main_widget so BatchTabWidget can call parent pipeline runners
        self.batch_widget = BatchTabWidget(
            parent_widget=self.main_widget, parent=self.container
        )

        layout.addWidget(self.batch_widget)
        layout.addStretch()

        self.setWidget(self.container)

    # --------------------------------------------------------------------------
    # Sync Hooks & Layer Refreshing
    # --------------------------------------------------------------------------
    def refresh_layer_choices(self):
        """Refreshes dynamic layer dropdowns in batch UI when napari layers change."""
        if hasattr(self.batch_widget, 'refresh_layer_choices'):
            self.batch_widget.refresh_layer_choices()

    def update_active_models_summary(self):
        """Refreshes active model/segmentation snapshots from sibling tabs."""
        if hasattr(self.batch_widget, 'update_model_summary'):
            self.batch_widget.update_model_summary()

    def showEvent(self, event):
        """Automatically refresh layer dropdowns and model summaries whenever tab becomes visible."""
        super().showEvent(event)
        self.refresh_layer_choices()
        self.update_active_models_summary()
