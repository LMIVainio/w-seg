import os
from typing import TYPE_CHECKING

import h5py
import napari
import numpy as np
from qtpy.QtWidgets import (
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from tifffile import imread

from w_seg_napari._widget_tabs import (
    Tab1Pipeline,
    Tab2ROI_selection,
    Tab3Curation,
    Tab4Analysis,
    Tab5Batch,
)

if TYPE_CHECKING:
    import napari


class WSegMasterWidget(QWidget):
    """
    Master container widget orchestrating segmentation, curation,
    morphology analysis, and batch pipeline sub-tabs.
    """

    def __init__(self, viewer: 'napari.Viewer', parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._setup_keybindings()

        # 1. Instantiate plugin tabs
        self.tab1 = Tab1Pipeline(self.viewer)
        self.tab2 = Tab2ROI_selection(self.viewer)
        self.tab3 = Tab3Curation(self.viewer)
        self.tab4 = Tab4Analysis(self.viewer)
        # Pass self as main_widget to Tab5Batch
        self.tab5 = Tab5Batch(self.viewer, main_widget=self)

        # 2. Tab Container Assembly
        self.tabs = QTabWidget(self)
        self.tabs.addTab(self.tab1, '1. W-seg and GASP')
        self.tabs.addTab(self.tab2, '2. ROI Selection')
        self.tabs.addTab(self.tab3, '3. Curation')
        self.tabs.addTab(self.tab4, '4. Cell Morphology')
        # Tab 5 or batch processing is under development
        # self.tabs.addTab(self.tab5, '5. Batch Mode')

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.tabs)

        # 3. Connect Tab Inter-Communications
        self._wire_tab_signals()

        # 4. Connect Napari Event Listeners
        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)

        # 5. Attach renaming listeners to all layers currently in the viewer
        for layer in self.viewer.layers:
            layer.events.name.connect(self._on_layer_renamed)

        # Initial refresh across all tab layer dropdowns
        self.refresh_all_layer_choices()

        self.destroyed.connect(self._on_close)

    def _connect_layer_name_listener(self, layer: napari.layers.Layer):
        """Helper to attach the rename listener to a single layer safely."""
        try:
            # Disconnect first to ensure single connection instance
            layer.events.name.disconnect(self._on_layer_renamed)
        except (RuntimeError, ValueError):
            pass
        layer.events.name.connect(self._on_layer_renamed)

    def _disconnect_layer_name_listener(self, layer: napari.layers.Layer):
        """Helper to detach the rename listener from a single layer safely."""
        try:
            layer.events.name.disconnect(self._on_layer_renamed)
        except (RuntimeError, ValueError):
            pass

    def _setup_keybindings(self):
        """Binds custom hotkeys for quick label curation."""

        def set_label_to_zero(v):
            active_layer = v.layers.selection.active
            if isinstance(active_layer, napari.layers.Labels):
                active_layer.selected_label = 0
                v.status = 'Selected Label set to 0 (Background)'

        def set_to_smallest_missing_label(v):
            active_layer = v.layers.selection.active
            if (
                isinstance(active_layer, napari.layers.Labels)
                and active_layer.data is not None
            ):
                unique_labels = set(np.unique(active_layer.data))
                next_label = 1
                while next_label in unique_labels:
                    next_label += 1
                active_layer.selected_label = next_label
                v.status = f'Selected Label set to smallest unused label: {next_label}'

        def increase_brush_size(v):
            active_layer = v.layers.selection.active
            if isinstance(active_layer, napari.layers.Labels):
                active_layer.brush_size += 1
                v.status = f'Brush size: {active_layer.brush_size}'

        def decrease_brush_size(v):
            active_layer = v.layers.selection.active
            if isinstance(active_layer, napari.layers.Labels):
                active_layer.brush_size = max(1, active_layer.brush_size - 1)
                v.status = f'Brush size: {active_layer.brush_size}'

        def increment_selected_label(v):
            active_layer = v.layers.selection.active
            if isinstance(active_layer, napari.layers.Labels):
                active_layer.selected_label += 1
                v.status = f'Selected Label: {active_layer.selected_label}'

        def decrement_selected_label(v):
            active_layer = v.layers.selection.active
            if isinstance(active_layer, napari.layers.Labels):
                active_layer.selected_label = max(
                    0, active_layer.selected_label - 1
                )
                v.status = f'Selected Label: {active_layer.selected_label}'

        self.viewer.bind_key('numpad0', set_label_to_zero, overwrite=True)
        self.viewer.bind_key(
            'n', set_to_smallest_missing_label, overwrite=True
        )
        self.viewer.bind_key(
            'numpad_minus', decrement_selected_label, overwrite=True
        )
        self.viewer.bind_key(
            'numpad_plus', increment_selected_label, overwrite=True
        )
        self.viewer.bind_key(
            'numpad_divide', decrease_brush_size, overwrite=True
        )
        self.viewer.bind_key(
            'numpad_multiply', increase_brush_size, overwrite=True
        )

    def _wire_tab_signals(self):
        """Wires inter-tab signals and tab switching events."""
        if hasattr(self.tab1, 'model_changed') and hasattr(
            self.tab4, 'update_model_selection'
        ):
            self.tab1.model_changed.connect(self.tab4.update_model_selection)

            # Get text directly if currentData() is empty
            initial_weight = (
                self.tab1.combo_weights.currentData()
                or self.tab1.combo_weights.currentText()
            )

            if initial_weight and str(initial_weight).strip():
                self.tab4.update_model_selection(str(initial_weight))
            else:
                # Debug log if initial selection was unexpectedly empty
                print(
                    '[DEBUG Sync Wire] Tab1 combo_weights was empty at wire time.'
                )
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index: int):
        """Triggers updates when tabs become active."""
        # Refresh current active tab choices
        active_tab = self.tabs.widget(index)

        if active_tab == self.tab1:
            self.tab1.refresh_layer_choices()
        elif active_tab == self.tab2:
            self.tab2.refresh_layer_choices()
            self.tab2.refresh_labels_choices()
        elif active_tab == self.tab3:
            self.tab3.refresh_layer_choices()
        elif active_tab == self.tab4:
            self.tab4.refresh_layer_choices()
            self.tab4.update_model_selection(
                self.tab1.combo_weights.currentData()
            )
        elif active_tab == self.tab5:
            self.tab5.update_active_models_summary()
            self.tab5.refresh_layer_choices()

    def refresh_all_layer_choices(self, *args, **kwargs):
        """
        Refreshes layer dropdown choices across all tabs.
        Accepts *args and **kwargs so it can safely be passed directly as a callback
        to napari event signals without signature mismatches.
        """
        if hasattr(self.tab1, 'refresh_layer_choices'):
            self.tab1.refresh_layer_choices()
        if hasattr(self.tab2, 'refresh_layer_choices'):
            self.tab2.refresh_layer_choices()
        if hasattr(self.tab2, 'refresh_labels_choices'):
            self.tab2.refresh_labels_choices()
        if hasattr(self.tab3, 'refresh_layer_choices'):
            self.tab3.refresh_layer_choices()
        if hasattr(self.tab4, 'refresh_layer_choices'):
            self.tab4.refresh_layer_choices()
        if hasattr(self.tab5, 'refresh_layer_choices'):
            self.tab5.refresh_layer_choices()

    def _on_layer_inserted(self, event):
        """Called when a new layer is added to the viewer."""
        layer = event.value
        self._connect_layer_name_listener(layer)
        self.refresh_all_layer_choices()

    def _on_layer_removed(self, event):
        """Called when a layer is removed from the viewer."""
        layer = event.value
        self._disconnect_layer_name_listener(layer)
        self.refresh_all_layer_choices()

    def _on_layer_renamed(self, event=None):
        """Called when any layer's name changes."""
        print(
            f'[DEBUG] Layer renamed event triggered! Event source: {getattr(event, "source", None)}'
        )
        self.refresh_all_layer_choices()

    def _on_close(self):
        """Disconnects event listeners when the master widget is closed."""
        try:
            self.viewer.layers.events.inserted.disconnect(
                self._on_layer_inserted
            )
            self.viewer.layers.events.removed.disconnect(
                self._on_layer_removed
            )
            for layer in self.viewer.layers:
                try:
                    layer.events.name.disconnect(self._on_layer_renamed)
                except (RuntimeError, ValueError):
                    pass
        except (RuntimeError, ValueError):
            pass

    # =========================================================================
    # PARAMETER COLLECTION & BATCH ORCHESTRATION HOOKS
    # =========================================================================

    def get_active_models_info(self) -> dict:
        """Returns a summary string dict for Tab 4 active model status panel."""
        model_str = getattr(self.tab1, 'loaded_weights', 'No weights loaded')
        sp_algo = 'Default'
        if hasattr(self.tab1, 'superpixel_widget') and hasattr(
            self.tab1.superpixel_widget, 'combo_algorithm'
        ):
            sp_algo = self.tab1.superpixel_widget.combo_algorithm.currentText()
        gasp_link = 'Default'
        if hasattr(self.tab1, 'gasp_widget') and hasattr(
            self.tab1.gasp_widget, 'combo_linkage'
        ):
            gasp_link = self.tab1.gasp_widget.combo_linkage.currentText()
        classifier_str = 'Default'
        if hasattr(self.tab3, 'cell_classifier'):
            classifier_str = (
                self.tab3.cell_classifier._refresh_active_classifier()
            )
        return {
            'model_name': model_str,
            'sp_algo': sp_algo,
            'gasp_linkage': gasp_link,
            'classifier_type': classifier_str,
        }

    def _get_all_active_parameters(self) -> dict:
        """Aggregates active model configs and parameter dicts from sub-widgets."""
        return {
            'feature_extraction_config': self.morphology_widget.get_feature_extraction_config(),
            'model_config': self.cell_classifier.get_model_config(),
            'model_path': getattr(self, 'active_prediction_model_path', None),
            # Add Gasp, Superpixel, or Affinities parameters here as needed
        }

    def get_active_models_info2(self) -> dict:
        """Pulls user-readable summaries for the Batch Tab UI summary box."""
        clf_config = self.cell_classifier.get_model_config()
        return {
            'model_name': os.path.basename(
                getattr(self, 'active_prediction_model_path', 'None')
            ),
            'sp_algo': 'SLIC / Watershed',
            'gasp_linkage': 'Average',
            'classifier_type': clf_config.get('classifier_type', 'Not Loaded'),
        }

    def _run_batch_pipeline(
        self, image_path: str, out_h5_path: str, settings: dict
    ):
        """Headless execution of pipeline stages per image file.
        Executed inside BatchWorker thread (DO NOT modify Qt UI elements directly here).
        """
        global_settings = settings.get('global', {})
        render_in_viewer = settings.get('render_viewer', False)

        # ----------------------------------------------------------------------
        # Stage 0: Load Raw Image
        # ----------------------------------------------------------------------
        if image_path.endswith(('.h5', '.hdf5')):
            with h5py.File(image_path, 'r') as f:
                # Adjust dataset key matching your HDF5 layout
                raw_image = (
                    f['raw'][:] if 'raw' in f else f[list(f.keys())[0]][:]
                )
        else:
            raw_image = imread(image_path)

        session_data = {'raw': raw_image}

        # ----------------------------------------------------------------------
        # Stage 1: Prediction / Feature Map Generation
        # ----------------------------------------------------------------------
        if settings.get('predict', True):
            # Fetch active neural net model or feature extractor instance
            predict_model = getattr(self, 'prediction_model', None)

            if predict_model is not None:
                # Perform headless inference (e.g., cell boundaries/affinity maps)
                predictions = predict_model.predict(raw_image)
            else:
                # Fallback / Dummy placeholder if no prediction model is attached
                predictions = raw_image.astype(np.float32) / 255.0

            session_data['pred_features'] = predictions

        # ----------------------------------------------------------------------
        # Stage 2: Superpixel Segmentation
        # ----------------------------------------------------------------------
        if (
            settings.get('superpixels', True)
            and 'pred_features' in session_data
        ):
            # Example using headless superpixel calculation
            # superpixel_labels = run_superpixels_headless(session_data['pred_features'], global_settings)
            # session_data['sp_labels'] = superpixel_labels
            pass

        # ----------------------------------------------------------------------
        # Stage 3: GASP Graph Partitioning
        # ----------------------------------------------------------------------
        if settings.get('gasp', True) and 'sp_labels' in session_data:
            # gasp_labels = run_gasp_headless(session_data['pred_features'], session_data['sp_labels'])
            # session_data['gasp_labels'] = gasp_labels
            pass

        # ----------------------------------------------------------------------
        # Stage 4 & 5: Save Results to HDF5 Session File
        # ----------------------------------------------------------------------
        with h5py.File(out_h5_path, 'w') as h5f:
            for key, val in session_data.items():
                if isinstance(val, np.ndarray):
                    h5f.create_dataset(key, data=val, compression='gzip')

            # Save execution metadata & configs
            h5f.attrs['source_file'] = os.path.basename(image_path)
            h5f.attrs['fe_config_json'] = str(
                global_settings.get('feature_extraction_config', {})
            )
            h5f.attrs['model_config_json'] = str(
                global_settings.get('model_config', {})
            )

        # ----------------------------------------------------------------------
        # Optional: Render back into Napari Viewer
        # ----------------------------------------------------------------------
        if render_in_viewer and self.viewer is not None:
            # Schedule rendering on the main thread
            from qtpy.QtCore import Q_ARG, QMetaObject, Qt

            QMetaObject.invokeMethod(
                self,
                '_render_batch_results_in_viewer',
                Qt.QueuedConnection,
                Q_ARG(str, os.path.basename(image_path)),
                Q_ARG(dict, session_data),
            )

    def _render_batch_results_in_viewer(self, fname: str, session_data: dict):
        """Thread-safe update of the Napari canvas if 'Render results in Napari viewer' is checked."""
        if 'raw' in session_data:
            self.viewer.add_image(session_data['raw'], name=f'{fname}_raw')
        if 'pred_features' in session_data:
            self.viewer.add_image(
                session_data['pred_features'], name=f'{fname}_pred'
            )
        if 'gasp_labels' in session_data:
            self.viewer.add_labels(
                session_data['gasp_labels'], name=f'{fname}_gasp'
            )


# =====================================================================
# NPE2 PLUGIN ENTRY POINT
# =====================================================================


def w_seg_master_widget(viewer: 'napari.Viewer' = None) -> QWidget:
    """Plugin entry point for napari / npe2."""
    if viewer is None:
        viewer = napari.current_viewer()

    return WSegMasterWidget(viewer)
