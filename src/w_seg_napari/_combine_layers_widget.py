import napari
import numpy as np
from magicgui import magicgui
from scipy import ndimage
from skimage.measure import regionprops


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
    else:
        combined_mask = mask1

    out = combined_mask.astype(np.float32)

    if smooth_edge and sigma > 0:
        from scipy.ndimage import gaussian_filter

        out = gaussian_filter(out, sigma=sigma)

    return out


def create_layer_combination_widget(viewer: napari.Viewer):
    """Factory function creating a combined Napari layer blender, noise cleaner, and label merger widget."""

    @magicgui(
        # --- LAYER BLENDING / LOGIC CONTROLS ---
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
            'choices': ['AND', 'OR', 'XOR', 'NAND', 'Layer 1 NOT Layer 2'],
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
        # --- UNCONNECTED NOISE REMOVAL PARAMETERS ---
        clean_target_layer={'label': '1) Target Layer for Cleanup'},
        min_area={
            'label': '2a) Min Area (Pixels)',
            'widget_type': 'SpinBox',
            'min': 1,
            'max': 100000,
            'value': 50,
        },
        min_distance={
            'label': '2b) Isolation Distance (px)',
            'widget_type': 'FloatSlider',
            'min': 0.0,
            'max': 100.0,
            'step': 0.5,
            'value': 5.0,
        },
        dist_mode={
            'label': '3) Distance Calculation Metric',
            'choices': [
                'Euclidean Distance Map (EDT)',
                'Closest Neighbor Contour Distance',
            ],
            'value': 'Euclidean Distance Map (EDT)',
        },
        cleanup_thresh={
            'label': '4) Binarization Threshold',
            'widget_type': 'FloatSlider',
            'min': 0.0,
            'max': 1.0,
            'step': 0.01,
            'value': 0.5,
        },
        clean_button={
            'widget_type': 'PushButton',
            'text': 'Remove Unconnected Noise',
        },
        # --- LABEL COMBINATION PARAMETERS ---
        label_a={'label': 'Label Layer A (Sub-regions / Small)'},
        label_b={'label': 'Label Layer B (Main / Large)'},
        label_output_name={
            'label': 'Merged Labels Name',
            'value': 'merged_labels_layer',
        },
        combine_labels_button={
            'widget_type': 'PushButton',
            'text': 'Combine Labels Layers',
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
        logic_mode: str = 'AND',
        thresh_a: float = 0.3,
        thresh_b: float = 0.3,
        smooth_mask: bool = False,
        edge_sigma: float = 1.0,
        output_layer_name: str = 'combined_curated_layer',
        auto_update: bool = True,
        # Cleanup parameters:
        clean_target_layer: napari.layers.Image = None,
        min_area: int = 50,
        min_distance: float = 5.0,
        dist_mode: str = 'Euclidean Distance Map (EDT)',
        cleanup_thresh: float = 0.5,
        clean_button: bool = False,
        # Labels parameters:
        label_a: napari.layers.Labels = None,
        label_b: napari.layers.Labels = None,
        label_output_name: str = 'merged_labels_layer',
        combine_labels_button: bool = False,
    ):
        if layer_a is None or layer_b is None:
            return

        # 1. Copy raw data to prevent in-place mutation
        data_a = np.copy(layer_a.data).astype(np.float32)
        data_b = np.copy(layer_b.data).astype(np.float32)

        # 2. Safe normalization [0.0, 1.0]
        max_a, min_a = np.max(data_a), np.min(data_a)
        max_b, min_b = np.max(data_b), np.min(data_b)
        range_a, range_b = max_a - min_a, max_b - min_b

        norm_a = (
            (data_a - min_a) / range_a
            if range_a > 1e-7
            else np.zeros_like(data_a)
        )
        norm_b = (
            (data_b - min_b) / range_b
            if range_b > 1e-7
            else np.zeros_like(data_b)
        )

        if invert_a:
            norm_a = 1.0 - norm_a
        if invert_b:
            norm_b = 1.0 - norm_b

        # 3. Operations execution
        if operation == 'Smooth Blending':
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

        # 4. Canvas Push
        if output_layer_name in viewer.layers:
            out_layer = viewer.layers[output_layer_name]
            out_layer.data = result
            out_layer.contrast_limits = (0.0, 1.0)
            out_layer.refresh()
        else:
            viewer.add_image(
                result,
                name=output_layer_name,
                colormap='magma' if operation == 'Smooth Blending' else 'gray',
                contrast_limits=(0.0, 1.0),
            )

    # --- UNCONNECTED NOISE REMOVAL CALLBACK ---
    def _on_remove_unconnected_clicked():
        target_layer = widget.clean_target_layer.value
        if target_layer is None:
            return

        raw_data = np.copy(target_layer.data).astype(np.float32)

        d_min, d_max = np.min(raw_data), np.max(raw_data)
        d_range = d_max - d_min
        norm_data = (
            (raw_data - d_min) / d_range if d_range > 1e-7 else raw_data
        )

        thresh_val = widget.cleanup_thresh.value
        binary_mask = norm_data >= thresh_val

        labeled_mask, num_features = ndimage.label(binary_mask)
        if num_features == 0:
            return

        min_area_val = widget.min_area.value
        min_dist_val = widget.min_distance.value
        use_edt = widget.dist_mode.value == 'Euclidean Distance Map (EDT)'

        clean_count = 0
        if use_edt:
            dist_map = ndimage.distance_transform_edt(binary_mask)
        else:
            dist_map = ndimage.distance_transform_edt(~binary_mask)

        regions = regionprops(labeled_mask)
        cleaned_mask = np.copy(binary_mask)

        centroids = np.array([r.centroid for r in regions])
        for idx, region in enumerate(regions):
            label_id = region.label
            area = region.area

            if area < min_area_val:
                clean_count += 1
                if use_edt:
                    region_dist = np.max(dist_map[labeled_mask == label_id])
                else:
                    if len(centroids) > 1:
                        dists = np.linalg.norm(
                            centroids - centroids[idx], axis=1
                        )
                        dists[idx] = np.inf
                        region_dist = np.min(dists)
                    else:
                        region_dist = 0.0

                if region_dist < min_dist_val or not use_edt:
                    cleaned_mask[labeled_mask == label_id] = False

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
            )

    # --- LABEL COMBINATION CALLBACK ---
    def _on_combine_labels_clicked():
        lbl_layer_a = widget.label_a.value
        lbl_layer_b = widget.label_b.value

        if lbl_layer_a is None and lbl_layer_b is None:
            return

        data_a = lbl_layer_a.data if lbl_layer_a is not None else None
        data_b = lbl_layer_b.data if lbl_layer_b is not None else None

        merged_data = combine_label_layers(data_a, data_b)
        if merged_data is None:
            return

        out_name = widget.label_output_name.value
        if out_name in viewer.layers:
            viewer.layers[out_name].data = merged_data
            viewer.layers[out_name].refresh()
        else:
            viewer.add_labels(merged_data, name=out_name)

    # Wire button signals
    widget.clean_button.changed.connect(_on_remove_unconnected_clicked)
    widget.combine_labels_button.changed.connect(_on_combine_labels_clicked)

    def _on_parameter_change(value=None):
        if widget.auto_update.value:
            widget()

    widget.invert_a.changed.connect(_on_parameter_change)
    widget.invert_b.changed.connect(_on_parameter_change)
    widget.alpha.changed.connect(_on_parameter_change)
    widget.thresh_a.changed.connect(_on_parameter_change)
    widget.thresh_b.changed.connect(_on_parameter_change)
    widget.logic_mode.changed.connect(_on_parameter_change)
    widget.blend_mode.changed.connect(_on_parameter_change)
    widget.operation.changed.connect(_on_parameter_change)

    return widget
