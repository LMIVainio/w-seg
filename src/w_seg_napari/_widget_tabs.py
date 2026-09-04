import sys

import napari
from qtpy.QtCore import QSize, Qt, Signal, Slot, QThread
from qtpy.QtWidgets import (
    QComboBox,
    QCheckBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QProgressBar,
    QFileDialog,
    QMessageBox,
)

import w_seg_napari._wsegutils as w
from w_seg_napari._cell_classification_widget import CellClassificationWidget
from w_seg_napari._combine_layers_widget import create_layer_combination_widget
from w_seg_napari._gasp_widget import GaspWidget
from w_seg_napari._morphology_widget import MorphologyWidget
from w_seg_napari._superpixel_widget import SuperpixelWidget
from w_seg_napari._w_net_widget import WNetWidget
from w_seg_napari._wsegutils import CurationWidget
from w_seg_napari._batch_widget import BatchWorker

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

        # Build UI
        container = QWidget()
        container.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.MinimumExpanding
        )
        layout = QVBoxLayout()

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
        for w_name in w.WEIGHT_URL_MAP:
            self.combo_weights.addItem(w_name, userData=w_name)

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
                f'Active: {self.combo_weights.currentText()} ({self.device.type.upper()})'
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


class Tab3Analysis(QScrollArea):
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


class Tab4Batch(QScrollArea):
    """Dedicated tab for automated directory batch processing and HDF5 export."""

    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)

        self.worker_thread = None
        self.worker = None

        container = QWidget()
        container.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.MinimumExpanding
        )
        layout = QVBoxLayout()

        # Build pipeline options group
        layout.addWidget(self._build_settings_group())

        # Execution controls
        self.btn_start = QPushButton('📁 Select Folder & Run Batch')
        self.btn_start.setStyleSheet('font-weight: bold; padding: 8px;')
        self.btn_start.clicked.connect(self._on_start_batch)

        self.btn_cancel = QPushButton('Cancel Batch')
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel_batch)

        self.status_label = QLabel('Ready for batch processing.')
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_cancel)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addStretch()

        container.setLayout(layout)
        self.setWidget(container)

    def _build_settings_group(self) -> QGroupBox:
        group = QGroupBox('Batch Pipeline Stages')
        layout = QVBoxLayout()

        self.chk_superpixels = QCheckBox('Run Superpixels (Watershed)')
        self.chk_superpixels.setChecked(True)

        self.chk_gasp = QCheckBox('Run GASP Graph Partitioning')
        self.chk_gasp.setChecked(True)

        self.chk_morphology = QCheckBox('Run Cell Morphology Measurements')
        self.chk_morphology.setChecked(True)

        self.chk_classification = QCheckBox(
            'Run Random Forest Cell Classification'
        )
        self.chk_classification.setChecked(False)

        self.chk_export_h5 = QCheckBox('Export Session to HDF5 (.h5)')
        self.chk_export_h5.setChecked(True)
        self.chk_export_h5.setEnabled(False)  # Mandated output format

        layout.addWidget(self.chk_superpixels)
        layout.addWidget(self.chk_gasp)
        layout.addWidget(self.chk_morphology)
        layout.addWidget(self.chk_classification)
        layout.addWidget(self.chk_export_h5)
        group.setLayout(layout)
        return group

    def refresh_layer_choices(self):
        """No dynamic layer widgets required for batch configuration."""
        pass

    def _get_settings(self) -> dict:
        return {
            'superpixels': self.chk_superpixels.isChecked(),
            'gasp': self.chk_gasp.isChecked(),
            'morphology': self.chk_morphology.isChecked(),
            'classification': self.chk_classification.isChecked(),
        }

    def _on_start_batch(self):
        dirpath = QFileDialog.getExistingDirectory(
            self,
            'Select Directory for Batch Processing',
            '',
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if not dirpath:
            return

        settings = self._get_settings()

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        self.worker_thread = QThread()
        self.worker = BatchWorker(
            dirpath, settings, self._run_single_file_pipeline
        )
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress_changed.connect(self._on_progress_update)
        self.worker.finished.connect(self._on_batch_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    @Slot(int, str)
    def _on_progress_update(self, percent: int, msg: str):
        self.progress_bar.setValue(percent)
        self.status_label.setText(msg)

    def _on_cancel_batch(self):
        if self.worker:
            self.worker.cancel()
            self.status_label.setText('Cancelling batch run...')

    @Slot(int, list)
    def _on_batch_finished(self, success_count: int, failed_files: list):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setValue(100)

        msg = f'Batch processing finished.\nSuccessfully processed: {success_count} file(s).'
        if failed_files:
            msg += f'\n\nFailed on {len(failed_files)} file(s):\n' + '\n'.join(
                f'- {f}: {err}' for f, err in failed_files[:5]
            )
        QMessageBox.information(self, 'Batch Complete', msg)
        self.status_label.setText('Ready for batch processing.')