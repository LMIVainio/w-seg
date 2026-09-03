import napari
import numpy as np
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from skimage.segmentation import felzenszwalb, relabel_sequential, watershed


class SuperpixelWidget(QGroupBox):
    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__('Superpixel & Seed Generation', parent)
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
                'Distance Watershed',
                'Local Affinity Watershed',
                'Felzenszwalb Superpixels',
                'Seeded Mutex Watershed (elf)',
                'Mutex Watershed (elf)',
            ]
        )
        self.combo_algorithm.currentIndexChanged.connect(
            self._on_algorithm_changed
        )
        form_layout.addRow('Algorithm:', self.combo_algorithm)

        # 2. Layer Dropdowns
        self.combo_sem_layer = QComboBox()
        self.combo_sem_layer.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        form_layout.addRow('Semantic / Pred Layer:', self.combo_sem_layer)

        self.combo_aff_h = QComboBox()
        self.combo_aff_h.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        form_layout.addRow('Horizontal Affinity (affH):', self.combo_aff_h)

        self.combo_aff_v = QComboBox()
        self.combo_aff_v.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        form_layout.addRow('Vertical Affinity (affV):', self.combo_aff_v)

        # 3. Global Pre-Smoothing Parameter
        self.spin_smooth_sigma = QDoubleSpinBox()
        self.spin_smooth_sigma.setRange(0.0, 10.0)
        self.spin_smooth_sigma.setSingleStep(0.1)
        self.spin_smooth_sigma.setValue(1.0)
        self.spin_smooth_sigma.setToolTip(
            'Gaussian blur sigma applied to semantic maps to reduce noise before segmentation.'
        )
        form_layout.addRow('Pre-Smoothing (Sigma):', self.spin_smooth_sigma)

        self.spin_aff_sigma = QDoubleSpinBox()
        self.spin_aff_sigma.setRange(0.0, 10.0)
        self.spin_aff_sigma.setSingleStep(0.1)
        self.spin_aff_sigma.setValue(1.0)
        self.spin_aff_sigma.setToolTip(
            'Gaussian blur sigma applied to affinity maps to reduce noise before segmentation.'
        )
        form_layout.addRow('Affinity Smoothing (Sigma):', self.spin_aff_sigma)

        self.spin_mask_thresh = QDoubleSpinBox()
        self.spin_mask_thresh.setRange(0.0, 1.0)
        self.spin_mask_thresh.setSingleStep(0.05)
        self.spin_mask_thresh.setValue(0.35)
        form_layout.addRow('Mask Threshold:', self.spin_mask_thresh)

        self.spin_min_cell_size = QSpinBox()
        self.spin_min_cell_size.setRange(0, 1000)
        self.spin_min_cell_size.setValue(15)
        self.spin_min_cell_size.setToolTip(
            'Minimum size of superpixel regions to retain. Smaller regions will be removed.'
        )
        form_layout.addRow('Min Superpixel Size:', self.spin_min_cell_size)

        main_layout.addLayout(form_layout)

        # 4. Stacked Widget for Algorithm-Specific Hyperparameters
        self.param_stack = QStackedWidget()

        # --- Page 0: Distance Watershed Params ---
        page_dist = QWidget()
        layout_dist = QFormLayout(page_dist)
        layout_dist.setContentsMargins(0, 0, 0, 0)

        self.spin_compactness = QDoubleSpinBox()
        self.spin_compactness.setRange(0.0, 10.0)
        self.spin_compactness.setSingleStep(0.01)
        self.spin_compactness.setValue(0.0)
        layout_dist.addRow('Compactness:', self.spin_compactness)

        self.spin_min_dist = QSpinBox()
        self.spin_min_dist.setRange(1, 100)
        self.spin_min_dist.setValue(10)
        layout_dist.addRow('Min Peak Distance:', self.spin_min_dist)

        self.spin_min_peak_height = QDoubleSpinBox()
        self.spin_min_peak_height.setRange(0.0, 10.0)
        self.spin_min_peak_height.setSingleStep(0.1)
        self.spin_min_peak_height.setValue(2.0)
        layout_dist.addRow('Min Peak Height:', self.spin_min_peak_height)

        self.param_stack.addWidget(page_dist)

        # --- Page 1: Local Affinity Watershed Params ---
        page_aff_ws = QWidget()
        layout_aff_ws = QFormLayout(page_aff_ws)
        layout_aff_ws.setContentsMargins(0, 0, 0, 0)

        self.spin_aff_compactness = QDoubleSpinBox()
        self.spin_aff_compactness.setRange(0.0, 10.0)
        self.spin_aff_compactness.setSingleStep(0.01)
        self.spin_aff_compactness.setValue(0.0)
        layout_aff_ws.addRow('Compactness:', self.spin_aff_compactness)

        self.spin_aff_min_dist = QSpinBox()
        self.spin_aff_min_dist.setRange(1, 100)
        self.spin_aff_min_dist.setValue(7)
        layout_aff_ws.addRow('Seed Min Distance:', self.spin_aff_min_dist)

        self.spin_aff_min_peak_height = QDoubleSpinBox()
        self.spin_aff_min_peak_height.setRange(0.0, 10.0)
        self.spin_aff_min_peak_height.setSingleStep(0.1)
        self.spin_aff_min_peak_height.setValue(2.0)
        layout_aff_ws.addRow('Min Peak Height:', self.spin_aff_min_peak_height)

        self.spin_aff_threshold = QDoubleSpinBox()
        self.spin_aff_threshold.setRange(0.0, 1.0)
        self.spin_aff_threshold.setSingleStep(0.05)
        self.spin_aff_threshold.setValue(0.5)
        layout_aff_ws.addRow('Affinity Threshold:', self.spin_aff_threshold)

        self.param_stack.addWidget(page_aff_ws)

        # --- Page 2: Felzenszwalb Params ---
        page_felz = QWidget()
        layout_felz = QFormLayout(page_felz)
        layout_felz.setContentsMargins(0, 0, 0, 0)

        self.spin_felz_scale = QDoubleSpinBox()
        self.spin_felz_scale.setRange(1.0, 2000.0)
        self.spin_felz_scale.setSingleStep(10.0)
        self.spin_felz_scale.setValue(100.0)
        layout_felz.addRow('Scale:', self.spin_felz_scale)

        self.spin_felz_aff_weight = QDoubleSpinBox()
        self.spin_felz_aff_weight.setRange(0.1, 10.0)
        self.spin_felz_aff_weight.setSingleStep(0.1)
        self.spin_felz_aff_weight.setValue(1.0)
        layout_felz.addRow('Affinity Weight:', self.spin_felz_aff_weight)

        self.felz_signal = QComboBox()
        self.felz_signal.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        layout_felz.addRow('Signal layer:', self.felz_signal)

        self.param_stack.addWidget(page_felz)

        # --- Page 3: Mutex seeded Watershed Params ---
        page_smutex = QWidget()
        layout_smutex = QFormLayout(page_smutex)
        layout_smutex.setContentsMargins(0, 0, 0, 0)

        self.spin_smutex_seed_min_distance = QSpinBox()
        self.spin_smutex_seed_min_distance.setRange(1, 100)
        self.spin_smutex_seed_min_distance.setValue(7)
        layout_smutex.addRow(
            'Seed Min Distance:', self.spin_smutex_seed_min_distance
        )

        self.spin_smutex_strides = QSpinBox()
        self.spin_smutex_strides.setRange(1, 16)
        self.spin_smutex_strides.setValue(1)
        layout_smutex.addRow('Attr. Strides:', self.spin_smutex_strides)

        self.check_smutex_randomize_strides = QCheckBox()
        self.check_smutex_randomize_strides.setChecked(False)
        layout_smutex.addRow(
            'Randomize Strides:', self.check_smutex_randomize_strides
        )

        self.spin_smutex_noise_level = QDoubleSpinBox()
        self.spin_smutex_noise_level.setRange(0.0, 1.0)
        self.spin_smutex_noise_level.setSingleStep(0.01)
        self.spin_smutex_noise_level.setValue(0.0)
        layout_smutex.addRow('Noise Level:', self.spin_smutex_noise_level)

        self.smutex_signal = QComboBox()
        self.smutex_signal.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        layout_smutex.addRow('Signal layer:', self.smutex_signal)

        self.param_stack.addWidget(page_smutex)

        # --- Page 4: Mutex  Watershed Params ---
        page_mutex = QWidget()
        layout_mutex = QFormLayout(page_mutex)
        layout_mutex.setContentsMargins(0, 0, 0, 0)

        self.spin_mutex_strides = QSpinBox()
        self.spin_mutex_strides.setRange(1, 16)
        self.spin_mutex_strides.setValue(1)
        layout_mutex.addRow('Attr. Strides:', self.spin_mutex_strides)

        self.check_mutex_randomize_strides = QCheckBox()
        self.check_mutex_randomize_strides.setChecked(False)
        layout_mutex.addRow(
            'Randomize Strides:', self.check_mutex_randomize_strides
        )

        self.spin_mutex_noise_level = QDoubleSpinBox()
        self.spin_mutex_noise_level.setRange(0.0, 1.0)
        self.spin_mutex_noise_level.setSingleStep(0.01)
        self.spin_mutex_noise_level.setValue(0.0)
        layout_mutex.addRow('Noise Level:', self.spin_mutex_noise_level)

        self.mutex_signal = QComboBox()
        self.mutex_signal.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        layout_mutex.addRow('Signal layer:', self.mutex_signal)

        self.param_stack.addWidget(page_mutex)

        main_layout.addWidget(self.param_stack)

        # 5. Run Button
        self.btn_run = QPushButton('Generate Superpixels')
        self.btn_run.clicked.connect(self._on_run_clicked)
        main_layout.addWidget(self.btn_run)

        self._on_algorithm_changed(0)

    def _on_algorithm_changed(self, index: int):
        self.param_stack.setCurrentIndex(index)
        # Get selected algorithm name
        selected_algo = self.combo_algorithm.currentText()
        # Get algorithms that require affinity maps
        affinity_algorithms = [
            'Local Affinity Watershed',
            'Felzenszwalb Superpixels',
            'Seeded Mutex Watershed (elf)',
            'Mutex Watershed (elf)',
        ]
        # Check if the selected algorithm requires affinity maps
        uses_affinity = selected_algo in affinity_algorithms
        self.combo_aff_h.setEnabled(uses_affinity)
        self.combo_aff_v.setEnabled(uses_affinity)
        self.spin_aff_sigma.setEnabled(uses_affinity)

        # Get algorithms that require semantic/prediction maps
        semantic_algorithms = [
            'Distance Watershed',
            'Local Affinity Watershed',
            'Seeded Mutex Watershed (elf)',
        ]
        uses_semantic = selected_algo in semantic_algorithms
        self.combo_sem_layer.setEnabled(uses_semantic)

    def refresh_layer_choices(self, event=None):
        if self.viewer is None:
            return

        def _update_combo(combo: QComboBox, prefix=None, allow_none=False):
            current_selection = combo.currentText()
            combo.blockSignals(True)
            combo.clear()

            if allow_none:
                combo.addItem('None', userData=None)

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

        # Standar layers with selected prfixes
        _update_combo(self.combo_sem_layer, prefix='pred_')
        _update_combo(self.combo_aff_h, prefix='affH_')
        _update_combo(self.combo_aff_v, prefix='affV_')

        # Layers that accept None
        _update_combo(self.felz_signal, prefix=None, allow_none=True)
        _update_combo(self.smutex_signal, prefix='pred_', allow_none=True)
        _update_combo(self.mutex_signal, prefix='pred_', allow_none=True)

    def _on_run_clicked(self):
        algo_idx = self.combo_algorithm.currentIndex()
        algo_name = self.combo_algorithm.currentText()

        sem_layer = self.combo_sem_layer.currentData()
        aff_h_layer = self.combo_aff_h.currentData()
        aff_v_layer = self.combo_aff_v.currentData()
        sigma = self.spin_smooth_sigma.value()
        aff_sigma = self.spin_aff_sigma.value()

        if self.viewer:
            self.viewer.status = f'[w-seg] Computing {algo_name}...'

        superpixels = None

        if algo_idx == 0:  # Distance Watershed
            if sem_layer is None:
                print('[w-seg] Error: Semantic prediction layer required.')
                return
            superpixels = self._compute_distance_watershed(
                sem_layer.data,
                min_distance=self.spin_min_dist.value(),
                threshold=self.spin_mask_thresh.value(),
                sigma=sigma,
                min_peak_height=self.spin_min_peak_height.value(),
                compactness=self.spin_compactness.value(),
                min_cell_size=self.spin_min_cell_size.value(),
                fill_holes=False,
            )

        elif algo_idx == 1:  # Local Affinity Watershed
            if aff_h_layer is None or aff_v_layer is None:
                print(
                    '[w-seg] Error: Both Horizontal and Vertical affinities are required.'
                )
                return
            superpixels = self._compute_affinity_watershed(
                aff_h_layer.data,
                aff_v_layer.data,
                sem_mask=sem_layer.data if sem_layer is not None else None,
                compactness=self.spin_aff_compactness.value(),
                min_distance=self.spin_aff_min_dist.value(),
                threshold=self.spin_mask_thresh.value(),
                aff_threshold=self.spin_aff_threshold.value(),
                min_cell_size=self.spin_min_cell_size.value(),
                sigma=sigma,
                aff_sigma=aff_sigma,
                min_peak_height=self.spin_aff_min_peak_height.value(),
            )

        elif algo_idx == 2:  # Felzenszwalb Superpixels
            if aff_h_layer is None or aff_v_layer is None:
                print(
                    '[w-seg] Error: Both Horizontal and Vertical affinities are required.'
                )
                return
            signal_layer = self.felz_signal.currentData()
            signal_data = (
                signal_layer.data if signal_layer is not None else None
            )
            superpixels = self._compute_felzenszwalb(
                aff_h_layer.data,
                aff_v_layer.data,
                scale=self.spin_felz_scale.value(),
                aff_sigma=aff_sigma,
                sigma=sigma,
                min_size=self.spin_min_cell_size.value(),
                signal_or_sem=signal_data,
                affinity_weight=self.spin_felz_aff_weight.value(),
                bg_threshold=self.spin_mask_thresh.value(),
            )

        elif algo_idx == 3:  # Mutex seeded Watershed
            if aff_h_layer is None or aff_v_layer is None:
                print(
                    '[w-seg] Error: Both Horizontal and Vertical affinities are required.'
                )
                return
            smutex_sem_layer = self.smutex_signal.currentData()
            smutex_sem_data = (
                smutex_sem_layer.data if smutex_sem_layer is not None else None
            )
            superpixels = self._compute_mutex_watershed_seeded(
                aff_h_layer.data,
                aff_v_layer.data,
                semantic_map=smutex_sem_data,
                strides=self.spin_smutex_strides.value(),
                randomize_strides=self.check_smutex_randomize_strides.isChecked(),
                noise_level=self.spin_smutex_noise_level.value(),
                min_cell_size=self.spin_min_cell_size.value(),
                sigma=sigma,
                aff_sigma=aff_sigma,
                seed_min_distance=self.spin_smutex_seed_min_distance.value(),
                mask_threshold=self.spin_mask_thresh.value(),
            )

        elif algo_idx == 4:  # Mutex Watershed
            if aff_h_layer is None or aff_v_layer is None:
                print(
                    '[w-seg] Error: Both Horizontal and Vertical affinities are required.'
                )
                return
            mutex_sem_layer = self.mutex_signal.currentData()
            mutex_sem_data = (
                mutex_sem_layer.data if mutex_sem_layer is not None else None
            )
            superpixels = self._compute_mutex_watershed(
                aff_h_layer.data,
                aff_v_layer.data,
                semantic_map=mutex_sem_data,
                strides=self.spin_mutex_strides.value(),
                randomize_strides=self.check_mutex_randomize_strides.isChecked(),
                noise_level=self.spin_mutex_noise_level.value(),
                min_cell_size=self.spin_min_cell_size.value(),
                sigma=sigma,
                aff_sigma=aff_sigma,
                mask_threshold=self.spin_mask_thresh.value(),
            )

        if superpixels is None:
            if self.viewer:
                self.viewer.status = f'[w-seg] Failed to generate {algo_name}.'
            return

        source_name = (
            sem_layer.name.replace('pred_', '')
            if sem_layer
            else aff_h_layer.name.replace('affH_', '')
        )
        layer_name = f'sp_{source_name}'

        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = superpixels
        else:
            self.viewer.add_labels(superpixels, name=layer_name, opacity=0.7)

        if self.viewer:
            self.viewer.status = (
                f'[w-seg] Created superpixel layer: {layer_name}'
            )

    # ------------------ Core Algorithms ------------------

    @staticmethod
    def _compute_distance_watershed(
        semantic_map: np.ndarray,
        min_distance: int = 10,
        threshold: float = 0.35,
        sigma: float = 1.0,
        min_peak_height: float = 2.0,  # Ignore peaks near borders/noise
        compactness: float = 0.0,  # Smoother cell shapes (>0)
        min_cell_size: int = 15,  # Remove tiny spurious labels
        fill_holes: bool = True,  # Prevent internal false peaks
    ) -> np.ndarray:
        """Enhanced distance watershed with morph-cleaning and vectorized seeding."""
        sem_clean = np.squeeze(semantic_map).astype(np.float32)
        # Smooth the semantic map to reduce noise and small artifacts
        if sigma > 0:
            sem_clean = ndi.gaussian_filter(sem_clean, sigma=sigma)
        # Threshold to create a binary mask of cell interiors
        cell_interiors = sem_clean < threshold

        if not np.any(cell_interiors):
            return np.zeros_like(sem_clean, dtype=np.uint64)

        # Distance Transform
        print('[w-seg] Computing distance transform for seed generation...')
        distance = ndi.distance_transform_edt(cell_interiors)

        # Local Maxima with height threshold
        coords = peak_local_max(
            distance,
            min_distance=min_distance,
            threshold_abs=min_peak_height,
            labels=cell_interiors,
            exclude_border=False,
        )

        if len(coords) == 0:
            return np.zeros_like(sem_clean, dtype=np.uint64)

        # Fast Vectorized Marker Placement
        seeds = np.zeros(distance.shape, dtype=bool)
        seeds[tuple(coords.T)] = True
        markers, _ = ndi.label(seeds)
        print('[w-seg] Running distance watershed segmentation...')
        # Compact Watershed
        ws_labels = watershed(
            -distance,
            markers=markers,
            mask=cell_interiors,
            compactness=compactness,
        )

        # Post-processing
        # Max size is the threshold that removes objects smaller than
        # **or equal to** its value, while the previous parameter called min_size
        #  only removed smaller ones. Min cell size is used for clarity
        if min_cell_size > 0:
            ws_labels = remove_small_objects(ws_labels, max_size=min_cell_size)
        ws_labels = relabel_sequential(ws_labels)[0]
        return ws_labels.astype(np.uint64)

    @staticmethod
    def _compute_affinity_watershed(
        aff_h: np.ndarray,
        aff_v: np.ndarray,
        sem_mask: np.ndarray,
        compactness: float,
        min_distance: int,
        threshold: float = 0.5,
        aff_threshold: float = 0.5,
        min_cell_size: int = 15,
        sigma: float = 1.0,
        aff_sigma: float = 1.0,
        min_peak_height: float = 2.0,
    ) -> np.ndarray:
        """Local affinity watershed using directional affinity vectors."""
        aff_h_clean = np.squeeze(aff_h).astype(np.float32)
        aff_v_clean = np.squeeze(aff_v).astype(np.float32)
        # Compute boundary map as the magnitude of the affinity vectors
        boundary_map = np.sqrt(
            (1.0 - aff_h_clean) ** 2 + (1.0 - aff_v_clean) ** 2
        )
        # Normalize boundary map to [0, 1] range
        boundary_norm = boundary_map / (boundary_map.max() + 1e-8)
        # Apply Gaussian smoothing to the boundary map to reduce noise and small artifacts
        if aff_sigma > 0:
            boundary_norm = ndi.gaussian_filter(boundary_norm, sigma=aff_sigma)
        # Normalize to strictly [0, 1] range safely
        b_max = boundary_norm.max()
        if b_max > 0:
            boundary_norm /= b_max
        # Apply Gaussian smoothing to the semantic mask to reduce noise and small artifacts
        if sigma > 0:
            sem_mask = ndi.gaussian_filter(sem_mask, sigma=sigma)

        # Threshold to create a binary mask of cell interiors (low boundary probability)
        interior_mask = boundary_norm < aff_threshold
        sem_mask = sem_mask < threshold
        # Return empty if no interior regions are found
        if not np.any(interior_mask):
            return np.zeros_like(boundary_map, dtype=np.uint64)

        # Compute distance transform of the interior mask to find seed locations
        print('[w-seg] Computing distance transform for seed generation...')
        distance = ndi.distance_transform_edt(interior_mask)
        # Find local maxima in the distance map to use as markers for watershed
        coords = peak_local_max(
            distance,
            min_distance=min_distance,
            threshold_abs=min_peak_height,
            labels=interior_mask,
            exclude_border=False,
        )
        if len(coords) == 0:
            return np.zeros_like(boundary_map, dtype=np.uint64)

        # Fast Vectorized Marker Placement
        seeds = np.zeros(distance.shape, dtype=bool)
        seeds[tuple(coords.T)] = True
        markers, _ = ndi.label(seeds)
        print('[w-seg] Running watershed segmentation...')
        ws_labels = watershed(
            boundary_norm,
            markers=markers,
            mask=sem_mask,
            compactness=compactness,
        )
        # Remove small objects from the watershed labels to clean up spurious regions
        if min_cell_size > 0:
            ws_labels = remove_small_objects(ws_labels, max_size=min_cell_size)
        ws_labels = relabel_sequential(ws_labels)[0]
        print('[w-seg] Watershed segmentation completed.')
        return ws_labels.astype(np.uint64)

    @staticmethod
    def _compute_felzenszwalb(
        aff_h: np.ndarray,
        aff_v: np.ndarray,
        scale: float = 100.0,
        aff_sigma: float = 0.8,
        sigma: float = 0.8,
        min_size: int = 20,
        signal_or_sem=None,  # 1-channel or 3-channel (RGB) array/layer
        affinity_weight: float = 1.5,  # Strength of boundary walls added to image
        bg_threshold: float = None,
    ) -> np.ndarray:
        """Felzenszwalb segmentation with blended boundary magnitude support."""
        # 1. Clean & Extract Arrays
        aff_h_clean = np.squeeze(
            aff_h.data if hasattr(aff_h, 'data') else aff_h
        ).astype(np.float32)
        aff_v_clean = np.squeeze(
            aff_v.data if hasattr(aff_v, 'data') else aff_v
        ).astype(np.float32)

        if hasattr(signal_or_sem, 'data'):
            signal_data = signal_or_sem.data
        else:
            signal_data = signal_or_sem

        # 2. Convert Affinities to Boundary Costs (0 = interior, 1 = boundary)
        b_h = 1.0 - np.clip(aff_h_clean, 0.0, 1.0)
        b_v = 1.0 - np.clip(aff_v_clean, 0.0, 1.0)

        # 3. Compute Isotropic Boundary Magnitude
        boundary_mag = np.sqrt(0.5 * (b_h**2 + b_v**2))

        # 4. Process Signal / Semantic Input
        if signal_data is not None:
            sig_clean = np.squeeze(signal_data).astype(np.float32)
            if sigma > 0:
                sig_clean = ndi.gaussian_filter(sig_clean, sigma=sigma)
            # Normalize signal/semantic map to [0, 1]
            sig_min, sig_max = sig_clean.min(), sig_clean.max()
            if sig_max > sig_min:
                sig_norm = (sig_clean - sig_min) / (sig_max - sig_min)
            else:
                sig_norm = np.zeros_like(sig_clean)

            # Blend boundary magnitude directly into signal channel(s)
            # For RGB (M, N, 3), broadcast boundary_mag along the channel axis
            if sig_norm.ndim == 3 and sig_norm.shape[-1] == 3:
                boundary_weight = affinity_weight * boundary_mag
                feature_stack = sig_norm + boundary_weight[..., np.newaxis]
                channel_axis = -1
            else:
                # 1-channel grayscale or semantic map
                # Ivert signal channel if it's a semantic map (0 = interior, 1 = boundary)
                sig_norm = 1.0 - sig_norm
                feature_stack = sig_norm + (affinity_weight * boundary_mag)
                channel_axis = None
        else:
            # Fallback: Just use the boundary magnitude map
            feature_stack = boundary_mag
            channel_axis = None
        print('[w-seg] Running Felzenszwalb segmentation...')
        # 5. Run Felzenszwalb
        labels = felzenszwalb(
            feature_stack,
            scale=scale,
            sigma=aff_sigma,
            min_size=min_size,
            channel_axis=channel_axis,
        )

        # 6. Optional Background Masking
        if bg_threshold is not None and signal_data is not None:
            if sig_norm.ndim == 3:
                bg_mask = sig_norm.mean(axis=-1) < bg_threshold
            else:
                bg_mask = sig_norm < bg_threshold
            labels[bg_mask] = 0
        labels = relabel_sequential(labels)[0]
        return labels.astype(np.uint64)

    @staticmethod
    def _compute_mutex_watershed_seeded(
        aff_h: np.ndarray,
        aff_v: np.ndarray,
        semantic_map: np.ndarray = None,
        strides: int = 1,
        randomize_strides: bool = False,
        noise_level: float = 0.0,
        min_cell_size: int = 15,
        return_graph: bool = False,
        sigma: float = 1.0,
        aff_sigma: float = 1.0,
        seed_min_distance: int = 7,
        mask_threshold: float = 0.5,
    ) -> np.ndarray:
        """Seeded Mutex Watershed with corrected attractive channel inversion."""
        try:
            from elf.segmentation.mutex_watershed import (
                mutex_watershed_with_seeds,
            )
        except ImportError:
            print('[w-seg] Error: elf module is required for Mutex Watershed.')
            return None

        # Squeeze and float32 cast
        aff_h_clean = np.squeeze(aff_h).astype(np.float32)
        aff_v_clean = np.squeeze(aff_v).astype(np.float32)

        if aff_sigma > 0:
            aff_h_clean = ndi.gaussian_filter(aff_h_clean, sigma=aff_sigma)
            aff_v_clean = ndi.gaussian_filter(aff_v_clean, sigma=aff_sigma)
        print('[w-seg] Generating seeds and mask for Mutex Watershed...')
        # 1. Masking & Seeds
        if semantic_map is not None:
            # Semantic map has cell interior as low values (0) and cell wall as high values (1)
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
        print('[w-seg] Executing Mutex Watershed with seeds...')
        # Execute Mutex Watershed
        labels = mutex_watershed_with_seeds(
            affs=affs_input,
            offsets=offsets,
            seeds=seeds,
            strides=[strides, strides],
            randomize_strides=randomize_strides,
            mask=fg_mask,
            noise_level=noise_level,
            return_graph=return_graph,
        )

        # Zero out background / cell wall region
        labels[~fg_mask] = 0

        if min_cell_size > 0:
            labels = remove_small_objects(labels, max_size=min_cell_size)
        labels = relabel_sequential(labels)[0]
        '[w-seg] Mutex Watershed completed'
        return labels.astype(np.uint64)

    @staticmethod
    def _compute_mutex_watershed(
        aff_h: np.ndarray,
        aff_v: np.ndarray,
        semantic_map: np.ndarray = None,
        strides: int = 1,
        randomize_strides: bool = False,
        noise_level: float = 0.0,
        min_cell_size: int = 15,
        return_graph: bool = False,
        sigma: float = 1.0,
        aff_sigma: float = 1.0,
        mask_threshold: float = 0.5,
    ) -> np.ndarray:
        """Seeded Mutex Watershed with corrected attractive channel inversion."""
        try:
            from elf.segmentation.mutex_watershed import mutex_watershed
        except ImportError:
            print('[w-seg] Error: elf module is required for Mutex Watershed.')
            return None

        # Squeeze and float32 cast
        aff_h_clean = np.squeeze(aff_h).astype(np.float32)
        aff_v_clean = np.squeeze(aff_v).astype(np.float32)

        if aff_sigma > 0:
            aff_h_clean = ndi.gaussian_filter(aff_h_clean, sigma=aff_sigma)
            aff_v_clean = ndi.gaussian_filter(aff_v_clean, sigma=aff_sigma)
        print('[w-seg] Generating seeds and mask for Mutex Watershed...')
        # 1. Masking & Seeds
        if semantic_map is not None:
            # Semantic map has cell interior as low values (0) and cell wall as high values (1)
            sem_clean = np.squeeze(semantic_map).astype(np.float32)
            if sigma > 0:
                sem_clean = ndi.gaussian_filter(sem_clean, sigma=sigma)
            fg_mask = sem_clean < mask_threshold
        else:
            sem_clean = 1.0 - ((aff_h_clean + aff_v_clean) / 2.0)
            fg_mask = sem_clean < mask_threshold

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
        print('[w-seg] Executing Mutex Watershed with seeds...')
        # Execute Mutex Watershed
        labels = mutex_watershed(
            affs=affs_input,
            offsets=offsets,
            strides=[strides, strides],
            randomize_strides=randomize_strides,
            mask=fg_mask,
            noise_level=noise_level,
        )

        # Zero out background / cell wall region
        labels[~fg_mask] = 0

        if min_cell_size > 0:
            labels = remove_small_objects(labels, max_size=min_cell_size)
        labels = relabel_sequential(labels)[0]
        '[w-seg] Mutex Watershed completed'
        return labels.astype(np.uint64)
