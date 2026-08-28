import napari
import torch
from qtpy.QtCore import Signal

# from PyQt6.QtWidgets import (
#    QWidget, QVBoxLayout, QGroupBox, QComboBox,
#    QPushButton, QLabel, QScrollArea
# )
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

import w_seg_napari._wsegutils as w
from w_seg_napari._cell_classification_widget import CellClassificationWidget
from w_seg_napari._combine_layers_widget import create_layer_combination_widget
from w_seg_napari._gasp_widget import make_gasp_widget
from w_seg_napari._morphology_widget import MorphologyWidget
from w_seg_napari._superpixel_widget import make_superpixel_widget
from w_seg_napari._w_net_impl import WNetWidget
from w_seg_napari._wsegutils import CurationWidget

# Import your widget instantiators / utilities
# import w_seg_napari._wsegutils as w


class Tab1Pipeline(QScrollArea):
    model_changed = Signal(str)

    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWidgetResizable(True)

        self.device = (
            'cuda'
            if torch.cuda.is_available()
            else 'mps'
            if torch.backends.mps.is_available()
            else 'cpu'
        )
        self.model_state = {'generator': None, 'device': self.device}

        # Build UI
        container = QWidget()
        layout = QVBoxLayout()

        # Model Selection Group
        layout.addWidget(self._build_model_group())

        # Instantiate Sub-widgets
        selected_generator = self.model_state['generator']
        self.w_net_widget = WNetWidget(
            viewer=self.viewer,
            generator=selected_generator,
            device=self.device,
        )
        self.superpixel_widget = make_superpixel_widget()
        self.gasp_widget = make_gasp_widget()

        layout.addWidget(self.w_net_widget.native)
        layout.addWidget(self.superpixel_widget.native)
        layout.addWidget(self.gasp_widget.native)
        layout.addStretch()

        container.setLayout(layout)
        self.setWidget(container)

    def _build_model_group(self) -> QGroupBox:
        model_group = QGroupBox('Model & Weights Selection')
        layout = QVBoxLayout()

        self.combo_generator = QComboBox()
        self.combo_generator.addItems(['W-seg'])

        self.combo_weights = QComboBox()
        for w_name in w.WEIGHT_URL_MAP:
            self.combo_weights.addItem(w_name, userData=w_name)

        self.btn_reload = QPushButton('🔄 Load / Reload Model')
        self.status_label = QLabel(
            f'Device: {self.device.upper()} | Model not loaded'
        )

        self.btn_reload.clicked.connect(self._on_reload_clicked)
        self.combo_weights.currentTextChanged.connect(self._emit_model_change)

        layout.addWidget(QLabel('Generator Architecture:'))
        layout.addWidget(self.combo_generator)
        layout.addWidget(QLabel('Staining Weights:'))
        layout.addWidget(self.combo_weights)
        layout.addWidget(self.btn_reload)
        layout.addWidget(self.status_label)
        model_group.setLayout(layout)

        # Trigger default model load on init
        self._on_reload_clicked()
        return model_group

    def _emit_model_change(self, text: str):
        selected_weight = self.combo_weights.currentData() or text
        if selected_weight:
            self.model_changed.emit(selected_weight)

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
                f'Active: {self.combo_weights.currentText()} ({self.device.upper()})'
            )
            self.viewer.status = f'[w-seg] Successfully loaded {self.combo_weights.currentText()}'
            # Send signal to morphology widget
            self.model_changed.emit(selected_weight_identifier)
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            self.status_label.setText('❌ Error loading model!')
            self.viewer.status = f'[w-seg] Failed to load model: {str(e)}'

    def refresh_layer_choices(self):
        """Refreshes layer dropdown choices for widgets contained in this tab."""
        if hasattr(self.w_net_widget, 'image_layer'):
            self.w_net_widget.image_layer.reset_choices()
        if hasattr(self.superpixel_widget, 'boundary_layer'):
            self.superpixel_widget.boundary_layer.reset_choices()
        if hasattr(self.gasp_widget, 'horizontal_affinity'):
            self.gasp_widget.horizontal_affinity.reset_choices()
        if hasattr(self.gasp_widget, 'vertical_affinity'):
            self.gasp_widget.vertical_affinity.reset_choices()
        if hasattr(self.gasp_widget, 'superpixel_layer'):
            self.gasp_widget.superpixel_layer.reset_choices()


class Tab2Curation(QScrollArea):
    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
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
        layout.addStretch()

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


class Tab3Analysis(QScrollArea):
    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout()

        # Instantiate widgets and pass the cached data
        self.morphology_widget = MorphologyWidget(viewer=self.viewer)
        self.cell_wall_classification = CellClassificationWidget(
            viewer=self.viewer,
            df=self.morphology_widget.cached_df,
            labels_data=self.morphology_widget.cached_labels_data,
            feature_extraction_config=self.morphology_widget.cached_feature_extraction_config,
        )

        original_on_measure = self.morphology_widget._on_measure_clicked

        def _on_measure_and_forward():
            # 1. Execute standard morphology calculation
            original_on_measure()

            # 2. Extract bound data pair & active layer name
            cached_df = getattr(self.morphology_widget, 'cached_df', None)
            cached_labels = getattr(
                self.morphology_widget, 'cached_labels_data', None
            )
            cached_feature_config = getattr(
                self.morphology_widget,
                'cached_feature_extraction_config',
                None,
            )

            # Fetch active label layer name from MorphologyWidget selection dropdown/combo
            active_layer_name = 'Segmentation Labels'
            if hasattr(self.morphology_widget, 'labels_select'):
                selected_layer = self.morphology_widget.labels_select.value
                if selected_layer is not None:
                    active_layer_name = selected_layer.name

            # 3. Synchronize immutable data pair to classifier widget
            if cached_df is not None and cached_labels is not None:
                self.cell_wall_classification.bind_measured_data(
                    df=cached_df,
                    labels_data=cached_labels,
                    layer_name=active_layer_name,
                    feature_extraction_config=cached_feature_config,
                )

        # Bridge the "Apply Config" action from ClassificationWidget back to MorphologyWidget
        def _apply_model_config_to_morphology(config: dict):
            self.morphology_widget.set_feature_extraction_config(config)

        self.cell_wall_classification.on_apply_config_callback = (
            _apply_model_config_to_morphology
        )

        self.morphology_widget._on_measure_clicked = _on_measure_and_forward
        # Re-bind click event to wrapper
        self.morphology_widget.run_btn.changed.disconnect()
        self.morphology_widget.run_btn.changed.connect(_on_measure_and_forward)

        # layout.addWidget(self.morphology_widget.native)
        # layout.addWidget(self.cell_wall_classification.native)

        # Frame each widget cleanly in a titled group box
        morphology_box = QGroupBox('1. Cell Morphology & CW Thickness')
        morphology_layout = QVBoxLayout()
        morphology_layout.addWidget(self.morphology_widget.native)
        morphology_box.setLayout(morphology_layout)

        classification_box = QGroupBox(
            '2. Cell Classification & Contact Topology'
        )
        classification_layout = QVBoxLayout()
        classification_layout.addWidget(self.cell_wall_classification)
        classification_box.setLayout(classification_layout)
        # Add group boxes to layout
        layout.addWidget(morphology_box)
        layout.addWidget(classification_box)
        layout.addStretch()

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
        if hasattr(self, 'cell_wall_classification'):
            if hasattr(self.cell_wall_classification, 'labels_layer'):
                self.cell_wall_classification.labels_layer.choices = (
                    label_layers
                )
            if hasattr(self.cell_wall_classification, 'signal_layer'):
                self.cell_wall_classification.signal_layer.choices = (
                    image_layers
                )

    def update_model_selection(self, model_weight_name: str):
        """Called when Tab1 changes the active model weights"""
        is_arabidopsis = 'arabidopsis' in model_weight_name.lower()
        # Update checkbox
        if hasattr(self, 'morphology_widget') and hasattr(
            self.morphology_widget, 'radial_symmetry'
        ):
            self.morphology_widget.radial_symmetry.value = is_arabidopsis
