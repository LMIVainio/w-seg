import napari
import numpy as np
import pandas as pd
from magicgui import magic_factory
from skimage.measure import regionprops_table


def get_label_layers(widget=None):
    viewer = napari.current_viewer()
    if viewer is None:
        return []
    return [
        layer
        for layer in viewer.layers
        if isinstance(layer, napari.layers.Labels)
    ]


@magic_factory(
    call_button='Extract Moments & Colorize',
    labels_layer={
        'widget_type': 'ComboBox',
        'choices': get_label_layers,
        'label': 'Target Labels:',
    },
    feature_to_color={
        'widget_type': 'ComboBox',
        'choices': ['area', 'perimeter', 'eccentricity', 'centroid'],
        'label': 'Color By Feature:',
    },
    colormap_name={
        'widget_type': 'ComboBox',
        'choices': [
            'viridis',
            'plasma',
            'inferno',
            'magma',
            'jet',
            'coolwarm',
        ],
        'label': 'Colormap:',
    },
)
def make_moments_widget(
    viewer: napari.Viewer,
    labels_layer: napari.layers.Labels,
    feature_to_color: str = 'area',
    colormap_name: str = 'viridis',
):
    if labels_layer is None:
        return

    labels_data = labels_layer.data

    # 1. Calculate region properties with scikit-image
    props = regionprops_table(
        labels_data,
        properties=('label', 'area', 'perimeter', 'eccentricity', 'centroid'),
    )
    df = pd.DataFrame(props)

    if feature_to_color not in df.columns:
        print(f'Feature {feature_to_color} not found in properties.')
        return

    # 2. Map label IDs to their feature values in a lookup array
    max_label = labels_data.max()
    lookup = np.zeros(max_label + 1, dtype=np.float32)

    # Populate lookup array: lookup[label_id] = feature_value
    labels = df['label'].values.astype(int)
    values = df[feature_to_color].values.astype(np.float32)
    lookup[labels] = values

    # 3. Create the float feature map by indexing the labels image
    feature_map = lookup[labels_data]

    # 4. Add or update as an Image layer
    new_layer_name = f'{labels_layer.name}_{feature_to_color}'

    if new_layer_name in viewer.layers:
        # Update existing image layer data & colormap
        layer = viewer.layers[new_layer_name]
        layer.data = feature_map
        layer.colormap = colormap_name
        layer.contrast_limits = (values.min(), values.max())
    else:
        # Create new Image layer using native napari colormaps!
        viewer.add_image(
            feature_map,
            name=new_layer_name,
            colormap=colormap_name,
            contrast_limits=(values.min(), values.max()),
            rendering='attenuated_mip',  # optional: helps transparency
        )
