import os
from typing import TYPE_CHECKING

import napari
import numpy as np
from qtpy.QtWidgets import (
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from w_seg_napari._widget_tabs import (
    Tab1Pipeline,
    Tab2ROI_selection,
    Tab3Curation,
    Tab4Analysis,
    Tab5Batch,
)
from w_seg_napari._wsegutils import get_model_config

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
        self.tabs.addTab(self.tab5, '5. Batch Mode')

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
        """Wires tab switching events (tabs read the active model themselves)."""
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
            self.tab4.refresh_paired_model()
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
        if hasattr(self.tab4, 'cell_classifier'):
            classifier_str = (
                self.tab4.cell_classifier._refresh_active_classifier()
            )
        return {
            'model_name': model_str,
            'sp_algo': sp_algo,
            'gasp_linkage': gasp_link,
            'classifier_type': classifier_str,
        }

    # =========================================================================
    # BATCH HOOKS (used by the batch tab)
    # =========================================================================

    def collect_batch_settings(self, batch_ui: dict) -> dict:
        """Copies the settings of tabs 1 and 4 for a batch run (call on the main thread).

        batch_ui holds what the batch tab itself decides: stages, render,
        morph_labels, morph_signal. Raises ValueError when something the
        chosen stages need is missing.
        """
        stages = batch_ui['stages']
        model = self.tab1.model_state.get('generator')
        if stages.get('predict') and model is None:
            raise ValueError('Load a model in tab 1 first.')
        if stages.get('classification') and not (
            self.tab4.cell_classifier.clf_manager.is_trained
        ):
            raise ValueError(
                'Train or load a cell classifier in tab 4, or switch off '
                'the classification stage.'
            )

        w_net = self.tab1.w_net_widget.get_settings()
        return {
            **batch_ui,
            'model': model,
            'model_cfg': get_model_config(model),
            'device': self.tab1.device,
            'patch_size': w_net['patch_size'],
            'stride': w_net['stride'],
            'sp_params': self.tab1.superpixel_widget.snapshot_params(),
            'gasp_params': self.tab1.gasp_widget.snapshot_params(),
            'morph_params': self.tab4.morphology_widget.snapshot_params(),
            'classifier': self.tab4.cell_classifier.clf_manager,
        }

    def set_batch_running(self, running: bool):
        """Locks every tab except the batch tab while a batch runs."""
        for i in range(self.tabs.count()):
            if self.tabs.widget(i) is not self.tab5:
                self.tabs.setTabEnabled(i, not running)

    def render_batch_results(self, image_name: str, results: dict):
        """Adds the results of one batch image to the viewer (main thread only)."""
        base = os.path.splitext(image_name)[0]
        v = self.viewer

        def put(layer_name, kind, data, **kwargs):
            if layer_name in v.layers:
                v.layers[layer_name].data = data
            else:
                getattr(v, f'add_{kind}')(data, name=layer_name, **kwargs)

        put(base, 'image', results['raw'])
        for payload in results.get('affinity_layers', []):
            put(
                payload['name'],
                'image',
                payload['data'],
                metadata=payload['metadata'],
                colormap='gray',
                blending='additive',
            )
        put(
            f'pred_{base}',
            'image',
            results['pred_sem'],
            colormap='red',
            contrast_limits=[0.0, 1.0],
            blending='additive',
        )
        if 'sp_labels' in results:
            put(f'sp_{base}', 'labels', results['sp_labels'], opacity=0.7)
        if 'gasp_labels' in results:
            put(
                f'gasp_cells_{base}',
                'labels',
                results['gasp_labels'],
                opacity=0.8,
            )
        if 'class_labels' in results:
            put(
                f'[Predicted Classes] {base}',
                'labels',
                results['class_labels'],
                opacity=0.7,
            )


# =====================================================================
# NPE2 PLUGIN ENTRY POINT
# =====================================================================


def w_seg_master_widget(viewer: 'napari.Viewer' = None) -> QWidget:
    """Plugin entry point for napari / npe2."""
    if viewer is None:
        viewer = napari.current_viewer()

    return WSegMasterWidget(viewer)
