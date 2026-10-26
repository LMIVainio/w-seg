import napari
import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
)


class GaspWidget(QGroupBox):
    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__('GASP Agglomerative Clustering', parent)
        self.viewer = viewer

        # Flexible size policy to expand nicely inside QScrollArea / QDockWidget
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._init_ui()

        # Wire up layer event listeners for automatic dropdown population
        if self.viewer is not None:
            self.viewer.layers.events.inserted.connect(
                self.refresh_layer_choices
            )
            self.viewer.layers.events.removed.connect(
                self.refresh_layer_choices
            )
            self.viewer.layers.events.changed.connect(
                self.refresh_layer_choices
            )

        self.refresh_layer_choices()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # 1. Layer Selection Controls
        self.combo_aff_h = QComboBox()
        self.combo_aff_h.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        form_layout.addRow('Horizontal Affinity:', self.combo_aff_h)

        self.combo_aff_v = QComboBox()
        self.combo_aff_v.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        form_layout.addRow('Vertical Affinity:', self.combo_aff_v)

        self.combo_superpixel = QComboBox()
        self.combo_superpixel.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        form_layout.addRow('Superpixel Layer:', self.combo_superpixel)

        # 2. Algorithm Parameters
        self.combo_linkage = QComboBox()
        self.combo_linkage.addItems(
            [
                'average',
                'single_linkage',
                'complete_linkage',
                'sum',
                'abs_max',
            ]
        )
        form_layout.addRow('Linkage Type:', self.combo_linkage)

        self.spin_beta = QDoubleSpinBox()
        self.spin_beta.setRange(0.0, 1.0)
        self.spin_beta.setSingleStep(0.01)
        self.spin_beta.setValue(0.90)
        form_layout.addRow('Beta:', self.spin_beta)

        self.spin_k_offset = QSpinBox()
        self.spin_k_offset.setRange(1, 100)
        self.spin_k_offset.setValue(8)
        form_layout.addRow('K Offset:', self.spin_k_offset)

        # 3. Action Button
        self.btn_run = QPushButton('Run GASP Clustering')
        self.btn_run.setStyleSheet(
            'font-weight: bold; background-color: #2b5b84; color: white; padding: 6px;'
        )
        self.btn_run.clicked.connect(self._on_run_clicked)

        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.btn_run)

    def refresh_layer_choices(self, event=None):
        """Populates affinity image layers and superpixel label layers dynamically."""
        if self.viewer is None:
            return

        def _update_combo(combo: QComboBox, layer_type, prefix=None):
            current_selection = combo.currentText()
            combo.blockSignals(True)
            combo.clear()

            matching_layers = [
                lay
                for lay in self.viewer.layers
                if isinstance(lay, layer_type)
                and (prefix is None or lay.name.startswith(prefix))
            ]

            for layer in matching_layers:
                combo.addItem(layer.name, userData=layer)

            index = combo.findText(current_selection)
            if index != -1:
                combo.setCurrentIndex(index)

            combo.blockSignals(False)

        # Update each dropdown with its matching layer criteria
        _update_combo(self.combo_aff_h, napari.layers.Image, prefix='affH_')
        _update_combo(self.combo_aff_v, napari.layers.Image, prefix='affV_')
        _update_combo(self.combo_superpixel, napari.layers.Labels)

    def _on_run_clicked(self):
        """Executes the GaspFromAffinities pipeline and posts the output layer."""
        aff_h_layer = self.combo_aff_h.currentData()
        aff_v_layer = self.combo_aff_v.currentData()
        sp_layer = self.combo_superpixel.currentData()

        if aff_h_layer is None or aff_v_layer is None or sp_layer is None:
            msg = '[w-seg] Error: Missing required layers for GASP.'
            print(msg)
            if self.viewer:
                self.viewer.status = msg
            return

        linkage = self.combo_linkage.currentText()
        beta_bias = self.spin_beta.value()
        k_offset = self.spin_k_offset.value()

        if self.viewer:
            self.viewer.status = '[w-seg] Running GaspFromAffinities...'

        final_instances = self._compute_gasp(
            horizontal_affinity=aff_h_layer.data,
            vertical_affinity=aff_v_layer.data,
            superpixels=sp_layer.data,
            linkage=linkage,
            beta_bias=beta_bias,
            k_offset=k_offset,
        )

        if final_instances is None:
            if self.viewer:
                self.viewer.status = (
                    '[w-seg] GASP failed due to input shape mismatch.'
                )
            return

        # Hide superpixel layer upon successful clustering completion
        sp_layer.visible = False

        # Retrieve spatial coordinate metadata from input superpixel layer
        layer_scale = sp_layer.scale
        layer_translate = sp_layer.translate

        # Derive clean layer name
        clean_sp_name = sp_layer.name.replace('sp_', '')
        layer_name = f'gasp_cells_{clean_sp_name}'

        # Update existing layer or add new Labels layer matching spatial offset
        if layer_name in self.viewer.layers:
            existing_layer = self.viewer.layers[layer_name]
            existing_layer.data = final_instances
            existing_layer.scale = layer_scale
            existing_layer.translate = layer_translate
            existing_layer.refresh()
            if self.viewer:
                self.viewer.status = (
                    f'[w-seg] Updated GASP layer: {layer_name}'
                )
        else:
            self.viewer.add_labels(
                final_instances,
                name=layer_name,
                opacity=0.8,
                scale=layer_scale,
                translate=layer_translate,
            )
            if self.viewer:
                self.viewer.status = f'[w-seg] Added GASP layer: {layer_name}'

    @staticmethod
    def _compute_gasp(
        horizontal_affinity: np.ndarray,
        vertical_affinity: np.ndarray,
        superpixels: np.ndarray,
        linkage: str,
        beta_bias: float,
        k_offset: int,
    ) -> np.ndarray:
        """Internal computation executing GaspFromAffinities on 2D/3D shapes."""
        from elf.segmentation.gasp import GaspFromAffinities

        # 1. Extract Arrays & Ensure Types
        fragments = np.squeeze(superpixels).astype(np.uint64)
        aff_h = np.squeeze(horizontal_affinity).astype(np.float32)
        aff_v = np.squeeze(vertical_affinity).astype(np.float32)

        # 2. Guard against invalid shapes
        if aff_h.shape != aff_v.shape:
            print(
                f'[w-seg] Error: Affinity shapes do not match: {aff_h.shape} vs {aff_v.shape}'
            )
            return None

        # 3. Format into 4D (C, Z, Y, X) and 3D (Z, Y, X) for elf/gasp
        if aff_h.ndim == 2:
            aff_channels = np.stack([aff_h, aff_v], axis=0)[
                :, np.newaxis, :, :
            ]
            fragments_3d = fragments[np.newaxis, :, :]
        elif aff_h.ndim == 3:
            aff_channels = np.stack([aff_h, aff_v], axis=0)
            fragments_3d = fragments
        else:
            print(
                f'[w-seg] Error: Unexpected affinity dimensions: {aff_h.ndim}D (shape: {aff_h.shape})'
            )
            return None

        if aff_channels.ndim != 4:
            print(
                f'[w-seg] Error: Expected 4D affinities array, but created shape {aff_channels.shape}'
            )
            return None

        # 3D offsets: [Z, Y, X]
        offsets = [[0, 0, k_offset], [0, k_offset, 0]]

        # 4. Setup engine instance
        gasp_engine = GaspFromAffinities(
            offsets=offsets,
            beta_bias=beta_bias,
            superpixel_generator=lambda affs, **kwargs: fragments_3d,
            run_GASP_kwargs={
                'linkage_criteria': linkage,
                'use_efficient_implementations': True,
            },
            set_only_direct_neigh_as_mergeable=True,
        )

        # 5. Compute Clustering
        final_instances_3d, _ = gasp_engine(aff_channels)

        # Squeeze back to match input layout (2D vs 3D)
        return np.squeeze(final_instances_3d)

    def get_settings(self):
        return {
            'linkage': self.combo_linkage.currentText(),
            'beta_bias': self.spin_beta.value(),
            'k_offset': self.spin_k_offset.value(),
        }
