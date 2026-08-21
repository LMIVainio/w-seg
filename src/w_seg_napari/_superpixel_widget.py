import napari
import numpy as np
import torch
from magicgui import magic_factory


def get_pred_layers(widget=None):
    viewer = napari.current_viewer()
    if viewer is None:
        return []
    # Return list of matching layer objects
    return [
        layer
        for layer in viewer.layers
        if isinstance(layer, napari.layers.Image)
        and layer.name.startswith('pred_')
    ]


@magic_factory(
    call_button='Generate Superpixels',
    boundary_layer={
        'widget_type': 'ComboBox',
        'choices': get_pred_layers,
        'label': 'Boundary Layer:',
    },
    threshold={
        'widget_type': 'FloatSlider',
        'min': 0.0,
        'max': 1.0,
        'step': 0.01,
        'label': 'Threshold:',
    },
    min_distance={
        'widget_type': 'Slider',
        'min': 1,
        'max': 50,
        'step': 1,
        'label': 'Min Distance:',
    },
)
def make_superpixel_widget(
    viewer: 'napari.Viewer',
    boundary_layer: 'napari.layers.Image',
    threshold: float = 0.35,
    min_distance: int = 10,
) -> 'napari.types.LayerDataTuple':
    """
    Napari widget that takes a boundary image layer and generates
    watershed-based superpixel fragments as a Labels layer.
    """
    # Keep mathematical package imports internal to prevent startup lag
    from scipy.ndimage import distance_transform_edt, label
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    if boundary_layer is None:
        print('[w-seg] Error: Please select a valid boundary image layer.')
        return

    boundary_2d = boundary_layer.data

    # 0. AUTOMATIC CONVERSION CHECK
    if isinstance(boundary_2d, torch.Tensor):
        boundary_2d = boundary_2d.detach().cpu().numpy()

    # Squeeze out extra channel dimensions if it's (1, H, W) or (H, W, 1)
    boundary_2d = np.squeeze(boundary_2d)

    # 1. Threshold the boundary map
    binary_cells = boundary_2d < threshold

    # 2. Compute the distance transform
    distance = distance_transform_edt(binary_cells)

    # 3. Find local peaks in the distance map
    local_max_coords = peak_local_max(
        distance, min_distance=min_distance, labels=binary_cells
    )

    # Create a boolean mask of the peaks, then label them uniquely to seed the watershed
    local_max_mask = np.zeros_like(distance, dtype=bool)
    if len(local_max_coords) > 0:
        local_max_mask[tuple(local_max_coords.T)] = True

    markers, _ = label(local_max_mask)

    # 4. Run the watershed algorithm
    superpixels = watershed(-distance, markers, mask=binary_cells)

    # 5. Return a LayerDataTuple to automatically add/update a Labels layer in Napari
    layer_name = f'sp_{boundary_layer.name}'

    layer_attributes = {'name': layer_name, 'opacity': 0.5}

    return (superpixels, layer_attributes, 'labels')


# Helper function to instantiate and wire up layer listeners automatically:


def create_superpixel_widget(viewer: 'napari.Viewer'):
    widget = make_superpixel_widget()

    def _refresh_choices(event=None):
        """Refreshes the boundary layer dropdown choices."""
        if hasattr(widget, 'boundary_layer') and hasattr(
            widget.boundary_layer, 'reset_choices'
        ):
            widget.boundary_layer.reset_choices()

    if viewer is not None:
        # 1. Catch added / removed layers
        viewer.layers.events.inserted.connect(_refresh_choices)
        viewer.layers.events.removed.connect(_refresh_choices)

        # 2. Catch layer renames and metadata updates
        viewer.layers.events.changed.connect(_refresh_choices)

        # Initial populate
        _refresh_choices()

    return widget
