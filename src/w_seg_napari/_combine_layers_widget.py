import napari
import numpy as np
from magicgui import magicgui
from napari.utils.notifications import show_warning
from scipy import ndimage
from skimage.measure import regionprops

from w_seg_napari._curation_ops import inherit_parent_labels

ONLY_A = 'Layer A only (threshold one layer)'


def combine_label_layers(
    small_labels: np.ndarray, large_labels: np.ndarray
) -> np.ndarray:
    """Combines two label arrays based on spatial overlap rules:
    - Labels inside larger labels take the large label's ID.
    - Labels on background (0) receive new, non-conflicting unique IDs.
    """
    if small_labels is None and large_labels is None:
        return None
    if small_labels is None:
        return large_labels.copy()
    if large_labels is None:
        return small_labels.copy()

    # Identify all non-zero unique labels from the small layer
    unique_small = np.unique(small_labels)
    unique_small = unique_small[unique_small != 0]

    if len(unique_small) == 0:
        return large_labels.copy()

    max_large_id = int(large_labels.max())
    next_new_id = max_large_id + 1

    output_labels = large_labels.copy()
    remapping = {}

    for s_id in unique_small:
        mask = small_labels == s_id
        overlapping_targets = large_labels[mask]

        # Count frequencies of target IDs inside the small label's footprint
        targets, counts = np.unique(overlapping_targets, return_counts=True)
        dominant_target = targets[np.argmax(counts)]

        if dominant_target != 0:
            # Rule 1: Contained within an existing larger label -> adopt ID
            remapping[s_id] = dominant_target
        else:
            # Rule 2: Check for any secondary non-zero overlap before fallback
            non_zero_mask = targets != 0
            if np.any(non_zero_mask):
                best_non_zero = targets[non_zero_mask][
                    np.argmax(counts[non_zero_mask])
                ]
                remapping[s_id] = best_non_zero
            else:
                # Fully on background -> assign a brand-new label ID
                remapping[s_id] = next_new_id
                next_new_id += 1

    # Remap background regions of the output array
    for s_id, target_id in remapping.items():
        s_mask = small_labels == s_id
        output_mask = (output_labels == 0) & s_mask
        output_labels[output_mask] = target_id

    return output_labels


def apply_smooth_blend(
    img1: np.ndarray, img2: np.ndarray, alpha: float, blend_mode: str
) -> np.ndarray:
    """Combines two normalized float images [0.0, 1.0] using smooth blending."""
    img1 = np.clip(img1, 0.0, 1.0)
    img2 = np.clip(img2, 0.0, 1.0)

    if blend_mode == 'Linear Alpha':
        return (1.0 - alpha) * img1 + alpha * img2

    elif blend_mode == 'Screen':
        return 1.0 - (1.0 - img1) * (1.0 - img2 * alpha)

    elif blend_mode == 'Multiply':
        return img1 * ((1.0 - alpha) + alpha * img2)

    elif blend_mode == 'Max Intensity':
        return np.maximum(img1 * (1.0 - alpha), img2 * alpha)

    return img1


def apply_boolean_logic(
    img1: np.ndarray,
    img2: np.ndarray,
    thresh1: float,
    thresh2: float,
    logic_mode: str,
    smooth_edge: bool = False,
    sigma: float = 1.0,
) -> np.ndarray:
    """Applies dynamic thresholding and Boolean logic combinations across two float layers."""
    mask1 = img1 > thresh1
    mask2 = img2 > thresh2

    if logic_mode == 'AND':
        combined_mask = np.logical_and(mask1, mask2)
    elif logic_mode == 'OR':
        combined_mask = np.logical_or(mask1, mask2)
    elif logic_mode == 'XOR':
        combined_mask = np.logical_xor(mask1, mask2)
    elif logic_mode == 'NAND':
        combined_mask = np.logical_not(np.logical_and(mask1, mask2))
    elif logic_mode == 'Layer 1 NOT Layer 2':
        combined_mask = np.logical_and(mask1, np.logical_not(mask2))
    elif logic_mode == ONLY_A:
        combined_mask = mask1
    else:
        combined_mask = mask1

    out = combined_mask.astype(np.float32)

    if smooth_edge and sigma > 0:
        from scipy.ndimage import gaussian_filter

        out = gaussian_filter(out, sigma=sigma)

    return out


def _normalise(data, invert=False):
    data = np.asarray(data, dtype=np.float32)
    lo, hi = float(data.min()), float(data.max())
    out = (data - lo) / (hi - lo) if hi - lo > 1e-7 else np.zeros_like(data)
    return 1.0 - out if invert else out


def create_blend_widget(viewer: napari.Viewer):
    """Blend two images, or threshold one or two images with Boolean logic."""

    @magicgui(
        call_button='Combine & Render Layers',
        layer_a={'label': 'Layer A (Base)'},
        invert_a={'label': 'Invert Layer A'},
        layer_b={'label': 'Layer B (Target)'},
        invert_b={'label': 'Invert Layer B'},
        operation={
            'label': 'Mode',
            'widget_type': 'RadioButtons',
            'choices': ['Smooth Blending', 'Threshold Logic (Boolean)'],
            'value': 'Smooth Blending',
        },
        blend_mode={
            'label': 'Blend Style',
            'choices': ['Linear Alpha', 'Screen', 'Multiply', 'Max Intensity'],
        },
        alpha={
            'label': 'Blend Ratio (Alpha)',
            'widget_type': 'FloatSlider',
            'min': 0.0,
            'max': 1.0,
            'step': 0.02,
            'value': 0.5,
        },
        logic_mode={
            'label': 'Boolean Operator',
            'choices': [
                ONLY_A,
                'AND',
                'OR',
                'XOR',
                'NAND',
                'Layer 1 NOT Layer 2',
            ],
        },
        thresh_a={
            'label': 'Threshold A',
            'widget_type': 'FloatSlider',
            'min': 0.0,
            'max': 1.0,
            'step': 0.01,
            'value': 0.3,
        },
        thresh_b={
            'label': 'Threshold B',
            'widget_type': 'FloatSlider',
            'min': 0.0,
            'max': 1.0,
            'step': 0.01,
            'value': 0.3,
        },
        smooth_mask={'label': 'Smooth Mask Edges'},
        edge_sigma={
            'label': 'Gaussian Sigma',
            'widget_type': 'FloatSlider',
            'min': 0.5,
            'max': 5.0,
            'step': 0.5,
            'value': 1.0,
        },
        output_layer_name={
            'label': 'Output Name',
            'value': 'combined_curated_layer',
        },
        auto_update={
            'label': 'Live Preview on Parameter Change',
            'value': True,
        },
    )
    def widget(
        layer_a: napari.layers.Image = None,
        invert_a: bool = False,
        layer_b: napari.layers.Image = None,
        invert_b: bool = False,
        operation: str = 'Smooth Blending',
        blend_mode: str = 'Linear Alpha',
        alpha: float = 0.5,
        logic_mode: str = ONLY_A,
        thresh_a: float = 0.3,
        thresh_b: float = 0.3,
        smooth_mask: bool = False,
        edge_sigma: float = 1.0,
        output_layer_name: str = 'combined_curated_layer',
        auto_update: bool = True,
    ):
        blending = operation == 'Smooth Blending'
        only_a = (not blending) and logic_mode == ONLY_A
        if layer_a is None or (layer_b is None and not only_a):
            return

        norm_a = _normalise(layer_a.data, invert_a)
        norm_b = (
            norm_a if layer_b is None else _normalise(layer_b.data, invert_b)
        )
        if blending:
            result = apply_smooth_blend(
                norm_a, norm_b, alpha=alpha, blend_mode=blend_mode
            )
        else:
            result = apply_boolean_logic(
                norm_a,
                norm_b,
                thresh1=thresh_a,
                thresh2=thresh_b,
                logic_mode=logic_mode,
                smooth_edge=smooth_mask,
                sigma=edge_sigma,
            )
        result = np.clip(result, 0.0, 1.0).astype(np.float32)

        if output_layer_name in viewer.layers:
            out_layer = viewer.layers[output_layer_name]
            out_layer.data = result
            out_layer.contrast_limits = (0.0, 1.0)
            out_layer.refresh()
        else:
            viewer.add_image(
                result,
                name=output_layer_name,
                colormap='magma' if blending else 'gray',
                contrast_limits=(0.0, 1.0),
                scale=layer_a.scale,
                translate=layer_a.translate,
            )

    def _update_visibility(value=None):
        blending = widget.operation.value == 'Smooth Blending'
        only_a = (not blending) and widget.logic_mode.value == ONLY_A
        widget.blend_mode.visible = blending
        widget.alpha.visible = blending
        widget.logic_mode.visible = not blending
        widget.thresh_a.visible = not blending
        widget.thresh_b.visible = (not blending) and not only_a
        widget.smooth_mask.visible = not blending
        widget.edge_sigma.visible = not blending
        widget.layer_b.visible = not only_a
        widget.invert_b.visible = not only_a

    def _on_parameter_change(value=None):
        _update_visibility()
        if widget.auto_update.value:
            widget()

    for name in (
        'invert_a',
        'invert_b',
        'alpha',
        'thresh_a',
        'thresh_b',
        'logic_mode',
        'blend_mode',
        'operation',
    ):
        getattr(widget, name).changed.connect(_on_parameter_change)
    _update_visibility()
    return widget


def create_cleanup_widget(viewer: napari.Viewer):
    """Remove small unconnected blobs from an image."""

    @magicgui(
        call_button='Remove Unconnected Noise',
        clean_target_layer={'label': '1) Target Layer'},
        cleanup_thresh={
            'label': '2) Binarization Threshold',
            'widget_type': 'FloatSlider',
            'min': 0.0,
            'max': 1.0,
            'step': 0.01,
            'value': 0.5,
        },
        min_area={
            'label': '3a) Min Area (Pixels)',
            'widget_type': 'SpinBox',
            'min': 1,
            'max': 100000,
            'value': 50,
        },
        min_distance={
            'label': '3b) Isolation Distance (px)',
            'widget_type': 'FloatSlider',
            'min': 0.0,
            'max': 100.0,
            'step': 0.5,
            'value': 5.0,
        },
        dist_mode={
            'label': '4) Distance Metric',
            'choices': [
                'Euclidean Distance Map (EDT)',
                'Closest Neighbor Contour Distance',
            ],
            'value': 'Euclidean Distance Map (EDT)',
        },
    )
    def widget(
        clean_target_layer: napari.layers.Image = None,
        cleanup_thresh: float = 0.5,
        min_area: int = 50,
        min_distance: float = 5.0,
        dist_mode: str = 'Euclidean Distance Map (EDT)',
    ):
        target_layer = clean_target_layer
        if target_layer is None:
            return
        raw_data = np.copy(target_layer.data).astype(np.float32)
        d_min, d_max = np.min(raw_data), np.max(raw_data)
        d_range = d_max - d_min
        norm_data = (
            (raw_data - d_min) / d_range if d_range > 1e-7 else raw_data
        )
        binary_mask = norm_data >= cleanup_thresh
        labeled_mask, num_features = ndimage.label(binary_mask)
        if num_features == 0:
            return

        use_edt = dist_mode == 'Euclidean Distance Map (EDT)'
        if use_edt:
            dist_map = ndimage.distance_transform_edt(binary_mask)
        else:
            dist_map = ndimage.distance_transform_edt(~binary_mask)

        regions = regionprops(labeled_mask)
        cleaned_mask = np.copy(binary_mask)
        centroids = np.array([r.centroid for r in regions])
        for idx, region in enumerate(regions):
            if region.area >= min_area:
                continue
            if use_edt:
                region_dist = np.max(dist_map[labeled_mask == region.label])
            elif len(centroids) > 1:
                dists = np.linalg.norm(centroids - centroids[idx], axis=1)
                dists[idx] = np.inf
                region_dist = np.min(dists)
            else:
                region_dist = 0.0
            if region_dist < min_distance or not use_edt:
                cleaned_mask[labeled_mask == region.label] = False

        cleaned_result = np.where(cleaned_mask, norm_data, 0.0).astype(
            np.float32
        )
        output_name = f'{target_layer.name}_cleaned'
        if output_name in viewer.layers:
            viewer.layers[output_name].data = cleaned_result
            viewer.layers[output_name].refresh()
        else:
            viewer.add_image(
                cleaned_result,
                name=output_name,
                colormap=target_layer.colormap,
                contrast_limits=(0.0, 1.0),
                scale=target_layer.scale,
                translate=target_layer.translate,
            )

    return widget


INHERIT = 'Inherit parent labels (majority vote)'
SUBREGIONS = 'Small regions into large labels'


def create_label_combine_widget(viewer: napari.Viewer):
    """Combine two labels layers: sub-regions into large labels, or inherit a parent's labels."""

    @magicgui(
        call_button='Combine Labels',
        mode={
            'label': 'Mode',
            'widget_type': 'RadioButtons',
            'choices': [INHERIT, SUBREGIONS],
            'value': INHERIT,
        },
        child_layer={'label': 'Child labels (gets the new labels)'},
        parent_layer={'label': 'Parent labels (e.g. YOLO classes)'},
        unmatched={
            'label': 'Child without parent',
            'choices': ['Keep own label', 'Set to 0'],
        },
        output_name={'label': 'Output Name', 'value': 'inherited_labels'},
    )
    def widget(
        mode: str = INHERIT,
        child_layer: napari.layers.Labels = None,
        parent_layer: napari.layers.Labels = None,
        unmatched: str = 'Keep own label',
        output_name: str = 'inherited_labels',
    ):
        if child_layer is None or parent_layer is None:
            return
        try:
            if mode == INHERIT:
                result = inherit_parent_labels(
                    child_layer.data,
                    parent_layer.data,
                    unmatched='zero' if unmatched == 'Set to 0' else 'keep',
                )
            else:
                result = combine_label_layers(
                    child_layer.data, parent_layer.data
                )
        except ValueError as e:
            show_warning(str(e))
            return
        if output_name in viewer.layers:
            viewer.layers[output_name].data = result
            viewer.layers[output_name].refresh()
        else:
            viewer.add_labels(
                result,
                name=output_name,
                scale=child_layer.scale,
                translate=child_layer.translate,
            )

    def _update(value=None):
        inherit = widget.mode.value == INHERIT
        widget.unmatched.visible = inherit
        if inherit:
            widget.child_layer.label = 'Child labels (gets the new labels)'
            widget.parent_layer.label = 'Parent labels (e.g. YOLO classes)'
            if widget.output_name.value == 'merged_labels_layer':
                widget.output_name.value = 'inherited_labels'
        else:
            widget.child_layer.label = 'Small regions (Layer A)'
            widget.parent_layer.label = 'Large labels (Layer B)'
            if widget.output_name.value == 'inherited_labels':
                widget.output_name.value = 'merged_labels_layer'

    widget.mode.changed.connect(_update)
    _update()
    return widget
