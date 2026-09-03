from typing import TYPE_CHECKING

# =====================================================================
# Import napari last to avoid dll conflict issues on Windows
# =====================================================================
import napari
import numpy as np

# Napari GUI libraries
from qtpy.QtWidgets import (
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# =====================================================================
# Widget Tabs import
# =====================================================================
from w_seg_napari._widget_tabs import Tab1Pipeline, Tab2Curation, Tab3Analysis

if TYPE_CHECKING:
    import napari

# ===================================================================
# ================ MASTER TAB WIDGET LAYOUT =========================
# ===================================================================

# from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget


def main_tabbed_widget(viewer: 'napari.Viewer') -> QWidget:
    """
    Master container widget orchestrating segmentation, curation,
    and analysis sub-tabs.
    """

    # 1. Keybindings Setup
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
            # Extract unique positive labels currently present in the data array
            unique_labels = set(np.unique(active_layer.data))

            # Find the smallest missing integer starting at 1
            next_label = 1
            while next_label in unique_labels:
                next_label += 1

            active_layer.selected_label = next_label
            v.status = (
                f'Selected Label set to smallest unused label: {next_label}'
            )

    # Bind keys
    viewer.bind_key('0', set_label_to_zero, overwrite=True)
    viewer.bind_key('numpad0', set_label_to_zero, overwrite=True)
    viewer.bind_key('n', set_to_smallest_missing_label, overwrite=True)

    # 2. Instantiate Modular Tabs
    tab1 = Tab1Pipeline(viewer)
    tab2 = Tab2Curation(viewer)
    tab3 = Tab3Analysis(viewer)

    # 3. Layer Events Orchestrator
    def refresh_all_layer_choices(event=None):
        tab1.refresh_layer_choices()
        tab2.refresh_layer_choices()
        tab3.refresh_layer_choices()

    viewer.layers.events.inserted.connect(refresh_all_layer_choices)
    viewer.layers.events.removed.connect(refresh_all_layer_choices)
    refresh_all_layer_choices()

    def _on_close():
        try:
            viewer.layers.events.inserted.disconnect(refresh_all_layer_choices)
            viewer.layers.events.removed.disconnect(refresh_all_layer_choices)
        except (RuntimeError, ValueError):
            pass

    # 4. Tab Container Assembly
    tabs = QTabWidget()
    tabs.addTab(tab1, '1. W-seg and GASP')
    tabs.addTab(tab2, '2. Curation')
    tabs.addTab(tab3, '3. Cell Morphology')

    # 5. Connect Tab 1 signal directly to Tab 3
    if hasattr(tab1, 'model_changed') and hasattr(
        tab3, 'update_model_selection'
    ):
        tab1.model_changed.connect(tab3.update_model_selection)

        # Retrieve current initial weight and push to Tab 3
        initial_weight = (
            tab1.combo_weights.currentData()
            or tab1.combo_weights.currentText()
        )
        if initial_weight:
            tab3.update_model_selection(initial_weight)

    main_widget = QWidget()
    main_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    main_layout = QVBoxLayout()
    main_layout.setContentsMargins(0, 0, 0, 0)
    main_layout.addWidget(tabs)
    main_widget.setLayout(main_layout)

    main_widget.destroyed.connect(_on_close)

    return main_widget


# =====================================================================
# NPE2 PLUGIN ENTRY POINTS
# =====================================================================


def w_seg_master_widget(viewer: 'napari.Viewer' = None) -> QWidget:
    """
    Plugin entry point. Defaults viewer to None so npe2/in_n_out can call
    it without throwing a positional argument error.
    """
    if viewer is None:
        viewer = napari.current_viewer()

    return main_tabbed_widget(viewer)
