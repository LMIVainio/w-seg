import napari
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import felzenszwalb, watershed


class SuperpixelWidget(QGroupBox):

    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__("Superpixel & Seed Generation", parent)
        self.viewer = viewer

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._init_ui()

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

        # 1. Algorithm Selection Dropdown
        self.combo_algorithm = QComboBox()
        self.combo_algorithm.addItems(
            [
                "Distance Watershed",
                "Local Affinity Watershed",
                "Felzenszwalb Superpixels",
                "Mutex Watershed (elf)",
            ]
        )
        self.combo_algorithm.currentIndexChanged.connect(
            self._on_algorithm_changed
        )
        form_layout.addRow("Algorithm:", self.combo_algorithm)

        # 2. Layer Dropdowns
        self.combo_sem_layer = QComboBox()
        self.combo_sem_layer.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        form_layout.addRow("Semantic / Pred Layer:", self.combo_sem_layer)

        self.combo_aff_h = QComboBox()
        self.combo_aff_h.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        form_layout.addRow("Horizontal Affinity (affH):", self.combo_aff_h)

        self.combo_aff_v = QComboBox()
        self.combo_aff_v.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        form_layout.addRow("Vertical Affinity (affV):", self.combo_aff_v)

        # 3. Global Pre-Smoothing Parameter
        self.spin_smooth_sigma = QDoubleSpinBox()
        self.spin_smooth_sigma.setRange(0.0, 10.0)
        self.spin_smooth_sigma.setSingleStep(0.5)
        self.spin_smooth_sigma.setValue(1.0)
        self.spin_smooth_sigma.setToolTip(
            "Gaussian blur sigma applied to input maps to reduce noise before segmentation."
        )
        form_layout.addRow("Pre-Smoothing (Sigma):", self.spin_smooth_sigma)

        main_layout.addLayout(form_layout)

        # 4. Stacked Widget for Algorithm-Specific Hyperparameters
        self.param_stack = QStackedWidget()

        # --- Page 0: Distance Watershed Params ---
        page_dist = QWidget()
        layout_dist = QFormLayout(page_dist)
        layout_dist.setContentsMargins(0, 0, 0, 0)

        self.spin_dist_min_dist = QSpinBox()
        self.spin_dist_min_dist.setRange(1, 100)
        self.spin_dist_min_dist.setValue(5)
        layout_dist.addRow("Min Peak Distance:", self.spin_dist_min_dist)

        self.spin_dist_thresh = QDoubleSpinBox()
        self.spin_dist_thresh.setRange(0.0, 1.0)
        self.spin_dist_thresh.setSingleStep(0.05)
        self.spin_dist_thresh.setValue(0.3)
        layout_dist.addRow("Mask Threshold:", self.spin_dist_thresh)

        self.param_stack.addWidget(page_dist)

        # --- Page 1: Local Affinity Watershed Params ---
        page_aff_ws = QWidget()
        layout_aff_ws = QFormLayout(page_aff_ws)
        layout_aff_ws.setContentsMargins(0, 0, 0, 0)

        self.spin_aff_compactness = QDoubleSpinBox()
        self.spin_aff_compactness.setRange(0.0, 10.0)
        self.spin_aff_compactness.setSingleStep(0.01)
        self.spin_aff_compactness.setValue(0.0)
        layout_aff_ws.addRow("Compactness:", self.spin_aff_compactness)

        self.spin_aff_min_dist = QSpinBox()
        self.spin_aff_min_dist.setRange(1, 100)
        self.spin_aff_min_dist.setValue(7)
        layout_aff_ws.addRow("Seed Min Distance:", self.spin_aff_min_dist)

        self.param_stack.addWidget(page_aff_ws)

        # --- Page 2: Felzenszwalb Params ---
        page_felz = QWidget()
        layout_felz = QFormLayout(page_felz)
        layout_felz.setContentsMargins(0, 0, 0, 0)

        self.spin_felz_scale = QDoubleSpinBox()
        self.spin_felz_scale.setRange(1.0, 2000.0)
        self.spin_felz_scale.setSingleStep(10.0)
        self.spin_felz_scale.setValue(100.0)
        layout_felz.addRow("Scale:", self.spin_felz_scale)

        self.spin_felz_min_size = QSpinBox()
        self.spin_felz_min_size.setRange(1, 5000)
        self.spin_felz_min_size.setValue(15)
        layout_felz.addRow("Min Size:", self.spin_felz_min_size)

        self.param_stack.addWidget(page_felz)

        # --- Page 3: Mutex Watershed Params ---
        page_mutex = QWidget()
        layout_mutex = QFormLayout(page_mutex)
        layout_mutex.setContentsMargins(0, 0, 0, 0)

        self.spin_mutex_strides = QSpinBox()
        self.spin_mutex_strides.setRange(1, 16)
        self.spin_mutex_strides.setValue(1)
        layout_mutex.addRow("Attr. Strides:", self.spin_mutex_strides)

        self.param_stack.addWidget(page_mutex)

        main_layout.addWidget(self.param_stack)

        # 5. Run Button
        self.btn_run = QPushButton("Generate Superpixels")
        self.btn_run.clicked.connect(self._on_run_clicked)
        main_layout.addWidget(self.btn_run)

        self._on_algorithm_changed(0)

    def _on_algorithm_changed(self, index: int):
        self.param_stack.setCurrentIndex(index)

    def refresh_layer_choices(self, event=None):
        if self.viewer is None:
            return

        def _update_combo(combo: QComboBox, prefix=None):
            current_selection = combo.currentText()
            combo.blockSignals(True)
            combo.clear()

            matching_layers = [
                lay
                for lay in self.viewer.layers
                if isinstance(lay, napari.layers.Image)
                and (prefix is None or lay.name.startswith(prefix))
            ]

            for layer in matching_layers:
                combo.addItem(layer.name, userData=layer)

            index = combo.findText(current_selection)
            if index != -1:
                combo.setCurrentIndex(index)

            combo.blockSignals(False)

        _update_combo(self.combo_sem_layer, prefix="pred_")
        _update_combo(self.combo_aff_h, prefix="affH_")
        _update_combo(self.combo_aff_v, prefix="affV_")

    def _on_run_clicked(self):
        algo_idx = self.combo_algorithm.currentIndex()
        algo_name = self.combo_algorithm.currentText()

        sem_layer = self.combo_sem_layer.currentData()
        aff_h_layer = self.combo_aff_h.currentData()
        aff_v_layer = self.combo_aff_v.currentData()
        sigma = self.spin_smooth_sigma.value()

        if self.viewer:
            self.viewer.status = f"[w-seg] Computing {algo_name}..."

        superpixels = None

        if algo_idx == 0:  # Distance Watershed
            if sem_layer is None:
                print("[w-seg] Error: Semantic prediction layer required.")
                return
            superpixels = self._compute_distance_watershed(
                sem_layer.data,
                min_distance=self.spin_dist_min_dist.value(),
                threshold=self.spin_dist_thresh.value(),
                sigma=sigma,
            )

        elif algo_idx == 1:  # Local Affinity Watershed
            if aff_h_layer is None or aff_v_layer is None:
                print(
                    "[w-seg] Error: Both Horizontal and Vertical affinities are required."
                )
                return
            superpixels = self._compute_affinity_watershed(
                aff_h_layer.data,
                aff_v_layer.data,
                compactness=self.spin_aff_compactness.value(),
                min_distance=self.spin_aff_min_dist.value(),
                sigma=sigma,
            )

        elif algo_idx == 2:  # Felzenszwalb Superpixels
            if aff_h_layer is None or aff_v_layer is None:
                print(
                    "[w-seg] Error: Both Horizontal and Vertical affinities are required."
                )
                return
            superpixels = self._compute_felzenszwalb(
                aff_h_layer.data,
                aff_v_layer.data,
                scale=self.spin_felz_scale.value(),
                sigma=sigma,
                min_size=self.spin_felz_min_size.value(),
            )

        elif algo_idx == 3:  # Mutex Watershed
            if aff_h_layer is None or aff_v_layer is None:
                print(
                    "[w-seg] Error: Both Horizontal and Vertical affinities are required."
                )
                return

            sem_data = sem_layer.data if sem_layer is not None else None
            superpixels = self._compute_mutex_watershed(
                aff_h_layer.data,
                aff_v_layer.data,
                semantic_map=sem_data,
                strides=self.spin_mutex_strides.value(),
                sigma=sigma,
            )

        if superpixels is None:
            if self.viewer:
                self.viewer.status = (
                    f"[w-seg] Failed to generate {algo_name}."
                )
            return

        source_name = (
            sem_layer.name.replace("pred_", "")
            if sem_layer
            else aff_h_layer.name.replace("affH_", "")
        )
        layer_name = f"sp_{source_name}"

        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = superpixels
        else:
            self.viewer.add_labels(
                superpixels, name=layer_name, opacity=0.7
            )

        if self.viewer:
            self.viewer.status = (
                f"[w-seg] Created superpixel layer: {layer_name}"
            )

    # ------------------ Core Algorithms ------------------

    @staticmethod
    def _compute_distance_watershed(
        semantic_map: np.ndarray, min_distance: int, threshold: float, sigma: float
    ) -> np.ndarray:
        """Distance watershed on inverted semantic map with pre-smoothing."""
        sem_clean = np.squeeze(semantic_map).astype(np.float32)
        if sigma > 0:
            sem_clean = ndi.gaussian_filter(sem_clean, sigma=sigma)

        cell_interiors = sem_clean < threshold
        distance = ndi.distance_transform_edt(cell_interiors)

        coords = peak_local_max(
            distance, min_distance=min_distance, labels=cell_interiors
        )

        markers = np.zeros(distance.shape, dtype=np.int32)
        for idx, (r, c) in enumerate(coords, start=1):
            markers[r, c] = idx

        if markers.max() == 0:
            return np.zeros_like(sem_clean, dtype=np.uint64)

        ws_labels = watershed(-distance, markers, mask=cell_interiors)
        return ws_labels.astype(np.uint64)

    @staticmethod
    def _compute_affinity_watershed(
        aff_h: np.ndarray,
        aff_v: np.ndarray,
        compactness: float,
        min_distance: int,
        sigma: float,
    ) -> np.ndarray:
        """Local affinity watershed using directional affinity vectors."""
        aff_h_clean = np.squeeze(aff_h).astype(np.float32)
        aff_v_clean = np.squeeze(aff_v).astype(np.float32)

        boundary_map = np.sqrt((1.0 - aff_h_clean) ** 2 + (1.0 - aff_v_clean) ** 2)
        boundary_norm = boundary_map / (boundary_map.max() + 1e-8)

        if sigma > 0:
            boundary_norm = ndi.gaussian_filter(boundary_norm, sigma=sigma)

        interior_mask = boundary_norm < 0.5
        distance = ndi.distance_transform_edt(interior_mask)

        coords = peak_local_max(
            distance, min_distance=min_distance, labels=interior_mask
        )

        markers = np.zeros(boundary_map.shape, dtype=np.int32)
        for idx, (r, c) in enumerate(coords, start=1):
            markers[r, c] = idx

        if markers.max() == 0:
            return np.zeros_like(boundary_map, dtype=np.uint64)

        ws_labels = watershed(
            boundary_norm, markers=markers, compactness=compactness
        )
        return ws_labels.astype(np.uint64)

    @staticmethod
    def _compute_felzenszwalb(
        aff_h: np.ndarray,
        aff_v: np.ndarray,
        scale: float,
        sigma: float,
        min_size: int,
    ) -> np.ndarray:
        """Felzenszwalb segmentation over stacked 2D affinity channels."""
        aff_h_clean = np.squeeze(aff_h).astype(np.float32)
        aff_v_clean = np.squeeze(aff_v).astype(np.float32)

        aff_stack = np.stack(
            [aff_h_clean, aff_v_clean, (aff_h_clean + aff_v_clean) / 2.0],
            axis=-1,
        )

        labels = felzenszwalb(
            aff_stack,
            scale=scale,
            sigma=sigma,
            min_size=min_size,
            channel_axis=-1,
        )
        return labels.astype(np.uint64)

    @staticmethod
    def _compute_mutex_watershed(
        aff_h: np.ndarray,
        aff_v: np.ndarray,
        semantic_map: np.ndarray = None,
        strides: int = 1,
        sigma: float = 1.0,
        seed_min_distance: int = 7,
        mask_threshold: float = 0.5,
    ) -> np.ndarray:
        """Seeded Mutex Watershed with corrected attractive channel inversion."""
        try:
            from elf.segmentation.mutex_watershed import mutex_watershed_with_seeds
        except ImportError:
            print("[w-seg] Error: elf module is required for Mutex Watershed.")
            return None

        # Squeeze and float32 cast
        aff_h_clean = np.squeeze(aff_h).astype(np.float32)
        aff_v_clean = np.squeeze(aff_v).astype(np.float32)

        if sigma > 0:
            aff_h_clean = ndi.gaussian_filter(aff_h_clean, sigma=sigma)
            aff_v_clean = ndi.gaussian_filter(aff_v_clean, sigma=sigma)

        # 1. Masking & Seeds
        if semantic_map is not None:
            sem_clean = np.squeeze(semantic_map).astype(np.float32)
            if sigma > 0:
                sem_clean = ndi.gaussian_filter(sem_clean, sigma=sigma)
            fg_mask = sem_clean < mask_threshold
        else:
            sem_clean = 1.0 - ((aff_h_clean + aff_v_clean) / 2.0)
            fg_mask = sem_clean < mask_threshold

        # Seeds from distance transform on cell interior mask
        dist_map = ndi.distance_transform_edt(fg_mask)
        coords = peak_local_max(
            dist_map, min_distance=seed_min_distance, labels=fg_mask
        )

        seeds = np.zeros(fg_mask.shape, dtype=np.uint64)
        for idx, (r, c) in enumerate(coords, start=1):
            seeds[r, c] = idx

        if seeds.max() == 0:
            return np.zeros_like(fg_mask, dtype=np.uint64)

        # 2. Offsets (2 Attractive + 2 Repulsive)
        offsets = [
            [0, 1],  # Channel 0: Horizontal attractive (+X)
            [1, 0],  # Channel 1: Vertical attractive (+Y)
            [0, 8],  # Channel 2: Horizontal repulsive (+X, k=8)
            [8, 0],  # Channel 3: Vertical repulsive (+Y, k=8)
        ]

        # 3. Channel Pre-processing
        # Because elf internal code executes: affs[:2] = 1 - affs[:2]
        # We pass (1.0 - aff) so elf converts it back to 1.0 = interior
        att_h = 1.0 - aff_h_clean
        att_v = 1.0 - aff_v_clean

        # Repulsive channels expect boundary probability (1.0 = wall)
        rep_h = sem_clean / (sem_clean.max() + 1e-8)
        rep_v = sem_clean / (sem_clean.max() + 1e-8)

        affs_stacked = np.stack([att_h, att_v, rep_h, rep_v], axis=0)
        affs_input = np.ascontiguousarray(affs_stacked, dtype=np.float32)

        # 4. Execute Mutex Watershed
        labels = mutex_watershed_with_seeds(
            affs=affs_input,
            offsets=offsets,
            seeds=seeds,
            strides=[strides, strides],
            mask=fg_mask,
        )

        # Zero out background / cell wall region
        labels[~fg_mask] = 0

        return labels.astype(np.uint64)