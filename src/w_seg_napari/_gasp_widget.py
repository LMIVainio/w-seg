import napari
import numpy as np
from magicgui import magic_factory


def get_affH_layers(widget=None):
    viewer = napari.current_viewer()
    if viewer is None:
        return []
    # Return list of matching layer objects
    return [
        layer
        for layer in viewer.layers
        if isinstance(layer, napari.layers.Image)
        and layer.name.startswith('affH_')
    ]


def get_affV_layers(widget=None):
    viewer = napari.current_viewer()
    if viewer is None:
        return []
    return [
        layer
        for layer in viewer.layers
        if isinstance(layer, napari.layers.Image)
        and layer.name.startswith('affV_')
    ]


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
    call_button='Run GASP Clustering',
    horizontal_affinity={
        'widget_type': 'ComboBox',
        'choices': get_affH_layers,
        'label': 'Horizontal Affinity:',
    },
    vertical_affinity={
        'widget_type': 'ComboBox',
        'choices': get_affV_layers,
        'label': 'Vertical Affinity:',
    },
    superpixel_layer={
        'widget_type': 'ComboBox',
        'choices': get_label_layers,
        'label': 'Superpixel Layer:',
    },
    linkage={
        'widget_type': 'ComboBox',
        'choices': [
            'average',
            'single_linkage',
            'complete_linkage',
            'sum',
            'abs_max',
        ],
        'label': 'Linkage Type:',
    },
    beta_bias={
        'widget_type': 'FloatSpinBox',
        'min': 0.0,
        'max': 1.0,
        'step': 0.01,
        'value': 0.9,
        'label': 'Beta:',
    },
)
def make_gasp_widget(
    viewer: 'napari.Viewer',
    horizontal_affinity: 'napari.layers.Image',
    vertical_affinity: 'napari.layers.Image',
    superpixel_layer: 'napari.layers.Labels',
    linkage: str = 'average',
    beta_bias: float = 0.95,
    k_offset: int = 8,
) -> 'napari.types.LayerDataTuple':
    """
    Refactored with consistent Z-padding to support 2D images
    in the bioimage_cpp 3D backend pipeline.
    """
    from elf.segmentation.gasp import GaspFromAffinities

    if (
        superpixel_layer is None
        or horizontal_affinity is None
        or vertical_affinity is None
    ):
        print('[w-seg] Error: Missing required layers.')
        return None

    # 1. Extract Arrays
    fragments = np.squeeze(superpixel_layer.data).astype(np.uint64)
    aff_h = np.squeeze(horizontal_affinity.data).astype(np.float32)
    aff_v = np.squeeze(vertical_affinity.data).astype(np.float32)

    # 2. Guard against invalid shapes/mismatches during startup
    if aff_h.shape != aff_v.shape:
        print(
            f'[w-seg] Error: Affinity shapes do not match: {aff_h.shape} vs {aff_v.shape}'
        )
        return None

    # 3. Format into 4D (C, Z, Y, X) and 3D (Z, Y, X) for elf/gasp
    if aff_h.ndim == 2:
        # 2D input -> stack into 3D (C, Y, X) then add fake Z -> (C=2, Z=1, Y, H)
        aff_channels = np.stack([aff_h, aff_v], axis=0)[:, np.newaxis, :, :]
        fragments_3d = fragments[np.newaxis, :, :]
    elif aff_h.ndim == 3:
        # 3D input -> stack directly into (C=2, Z, Y, X) without adding fake Z
        aff_channels = np.stack([aff_h, aff_v], axis=0)
        fragments_3d = fragments
    else:
        print(
            f'[w-seg] Error: Unexpected affinity dimensions: {aff_h.ndim}D (shape: {aff_h.shape})'
        )
        return None

    # Final shape verification for GASP
    if aff_channels.ndim != 4:
        print(
            f'[w-seg] Error: Expected 4D affinities array, but created shape {aff_channels.shape}'
        )
        return None

    print(
        f'[w-seg] Running GaspFromAffinities (Criterion: {linkage}, Beta Bias: {beta_bias})...'
    )

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
    print('[w-seg] Running agglomerative pipeline...')
    final_instances_3d, runtime = gasp_engine(aff_channels)

    # Squeeze back to match input layout (2D vs 3D)
    final_instances = np.squeeze(final_instances_3d)

    # 6. Package Output Properties
    layer_name = f'gasp_cells_{superpixel_layer.name.replace("sp_", "")}'
    layer_attributes = {'name': layer_name, 'opacity': 0.8}

    print(f'[w-seg] Pipeline complete! (Runtime: {runtime:.2f}s)')
    return (final_instances, layer_attributes, 'labels')
