import concurrent.futures
import sys
from concurrent.futures import as_completed
from pathlib import Path

import napari
import numpy as np
import pandas as pd
import torch
from napari.qt.threading import thread_worker
from napari.utils.colormaps import AVAILABLE_COLORMAPS, DirectLabelColormap
from qtpy.QtCore import QSignalBlocker, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import (
    binary_erosion,
    find_objects,
    gaussian_filter1d,
    label,
    map_coordinates,
)
from scipy.spatial import Delaunay
from skimage.color import rgb2gray
from skimage.measure import find_contours, regionprops_table
from skimage.segmentation import (
    clear_border,
    expand_labels,
    relabel_sequential,
)

from w_seg_napari._geometry import (
    label_frame_to_image_pixels,
    spatial_shape,
    world_to_data,
)
from w_seg_napari._wsegutils import (
    apply_feature_colormap,
    build_cell_contact_graph,
    create_yolo_classification_labels_layer,
    extract_spatial_feature_matrix,
    get_active_model_info,
    load_yolo_model,
    predict_yolo_tissue_on_image,
)

# =========================================================================
# HELPER FUNCTIONS
# =========================================================================


def compute_contour_normals(
    contour: np.ndarray, smooth_sigma: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes unit normal vectors along a contour in (y, x) array coordinates.

    Parameters
    ----------
    contour : np.ndarray
        Array of shape (N, 2) containing (y, x) points from find_contours.
    smooth_sigma : float
        Gaussian smoothing scale for tangent estimation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (smoothed_contour, unit_normals) both of shape (N, 2) in (y, x) order.
    """
    # 1. Smooth coordinates individually (use 'nearest' or 'reflect' for open contours)
    y_smooth = gaussian_filter1d(
        contour[:, 0], sigma=smooth_sigma, mode='nearest'
    )
    x_smooth = gaussian_filter1d(
        contour[:, 1], sigma=smooth_sigma, mode='nearest'
    )

    # 2. Compute tangent components along the curve: T = (dy, dx)
    dy = np.gradient(y_smooth)
    dx = np.gradient(x_smooth)

    # 3. Rotate tangent 90 degrees to get normal in (ny, nx) order:
    #     Tangent = [dy, dx]
    #     Normal  = [-dx, dy]  --> ny = -dx, nx = dy
    # (If normals point inside the cell instead of outside, change to [dx, -dy])
    ny = -dx
    nx = dy

    normals = np.column_stack([ny, nx])

    # 4. Normalize to unit length
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit_normals = normals / norms

    smoothed_contour = np.column_stack([y_smooth, x_smooth])

    return smoothed_contour, unit_normals


def clean_border_cells(labels_layer):
    """Removes all cells touching the image border."""
    return clear_border(labels_layer)


def fix_labels(labels_layer):
    # Relabel sequentially
    relabeled, _, _ = relabel_sequential(labels_layer)
    return relabeled


def preprocess_signal_layer(
    signal_image: np.ndarray,
    invert_signal: bool = False,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """
    Standardizes and normalizes the intensity signal image for robust sampling.
    """
    if signal_image is None:
        return None

    # Convert to float32 for processing
    cell_wall_signal = signal_image.astype(np.float32)

    # Handle NaN values safely
    if np.isnan(cell_wall_signal).any():
        cell_wall_signal = np.nan_to_num(cell_wall_signal, nan=0.0)

    # Dimensionality reduction (squeeze extra 1-dims)
    if cell_wall_signal.ndim > 2:
        cell_wall_signal = np.squeeze(cell_wall_signal)

    # Handle multi-channel images (convert RGB/RGBA to 2D grayscale, or pick channel 0)
    if cell_wall_signal.ndim == 3:
        n_channels = cell_wall_signal.shape[-1]
        if n_channels in (3, 4):
            # Trim Alpha channel if RGBA before passing to rgb2gray
            rgb_only = cell_wall_signal[..., :3]
            cell_wall_signal = rgb2gray(rgb_only)
        elif cell_wall_signal.shape[0] in (3, 4):
            # Channels-first format (C, H, W)
            rgb_only = np.moveaxis(cell_wall_signal[:3], 0, -1)
            cell_wall_signal = rgb2gray(rgb_only)
        else:
            # Multi-z or hyperstack slice: pick channel 0
            cell_wall_signal = cell_wall_signal[0]

    # Clip intensities to specified percentiles and normalize to [0, 1]
    if upper_percentile > lower_percentile:
        p_low, p_high = np.percentile(
            cell_wall_signal, [lower_percentile, upper_percentile]
        )
        if p_high > p_low:
            cell_wall_signal = np.clip(
                (cell_wall_signal - p_low) / (p_high - p_low), 0.0, 1.0
            )
        else:
            max_val = cell_wall_signal.max()
            if max_val > 0:
                cell_wall_signal /= max_val
    else:
        max_val = cell_wall_signal.max()
        if max_val > 0:
            cell_wall_signal /= max_val

    # Invert signal (e.g. brightfield vs fluorescence contrast)
    if invert_signal:
        cell_wall_signal = 1.0 - cell_wall_signal

    return cell_wall_signal


def get_safe_padded_bbox(
    bbox: tuple[int, int, int, int], pad: int, img_shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    """
    Pads a bounding box and clamps it strictly within image bounds.

    Parameters
    ----------
    bbox : tuple[int, int, int, int]
        Input bounding box from skimage regionprops: (min_r, min_c, max_r, max_c).
    pad : int
        Padding distance in pixels.
    img_shape : tuple[int, int]
        Shape of the image (H, W).

    Returns
    -------
    tuple[int, int, int, int]
        Clamped padded bounding box in standardized format: (r0, r1, c0, c1).
    """
    min_r, min_c, max_r, max_c = bbox
    H, W = img_shape[:2]

    r0 = max(0, min_r - pad)
    r1 = min(H, max_r + pad)
    c0 = max(0, min_c - pad)
    c1 = min(W, max_c + pad)

    return (r0, r1, c0, c1)


def add_radial_features(
    df: pd.DataFrame,
    center_coords: tuple[float, float] | np.ndarray | None = None,
    enabled: bool = True,
    y_col: str = 'centroid-0',
    x_col: str = 'centroid-1',
    pixel_scale: float = 1.0,
    angle_unit: str = 'degrees',
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Computes and appends distance and angle relative to a center point to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing cell centroids (e.g., regionprops output).
    center_coords : tuple or np.ndarray, optional
        (y, x) coordinates of the center point in pixels.
    enabled : bool
        If False or center_coords is None, fills radial features with `fill_value`.
    y_col : str
        Column name for Y centroid coordinate. Default 'centroid-0'.
    x_col : str
        Column name for X centroid coordinate. Default 'centroid-1'.
    pixel_scale : float
        Physical units per pixel (e.g., µm/px) to scale distances.
    angle_unit : str
        'degrees' (0 to 360) or 'radians' (-pi to +pi).
    fill_value : float
        Value to fill columns with when disabled (default 0.0).

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'distance_from_center' and 'angle_from_center' columns.
    """
    df = df.copy()

    # Define standardized feature names
    dist_col = 'distance_from_center'
    angle_col = 'angle_from_center'

    # Handle disabled case or missing center point while maintaining column schema
    if not enabled or center_coords is None or len(center_coords) < 2:
        df[dist_col] = fill_value
        df[angle_col] = fill_value
        return df
    print('[w-seg] Extracting polar coordinates')
    # Extract coordinates
    y_center, x_center = center_coords[0], center_coords[1]
    y_cells = df[y_col].values
    x_cells = df[x_col].values

    # 1. Delta coordinates relative to center
    dy = y_cells - y_center
    dx = x_cells - x_center

    # 2. Euclidean Distance (scaled by pixel resolution)
    distances = np.hypot(dy, dx) * pixel_scale

    # 3. Angle in polar space (arctan2 handles quadrants automatically)
    # Note: -dy reverses image Y-axis (where top-left is 0,0) to standard Cartesian Y-axis
    angles_rad = np.arctan2(-dy, dx)

    if angle_unit == 'degrees':
        angles = np.degrees(angles_rad)
        # Convert [-180, 180] range to intuitive [0, 360] range
        angles = np.mod(angles, 360.0)
    else:
        angles = angles_rad

    # Assign columns back to DataFrame
    df[dist_col] = distances
    df[angle_col] = angles

    return df


def resolve_center_coords(
    df: pd.DataFrame,
    center_coords: tuple[float, float] | np.ndarray | None = None,
    y_col: str = 'centroid-0',
    x_col: str = 'centroid-1',
) -> np.ndarray:
    """Returns provided center coordinates, or calculates the average centroid of all cells if unspecified.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing cell centroids.
    center_coords : tuple, np.ndarray, or None
        User-specified center point coordinates (y, x).
    y_col : str
        Column name for Y centroid coordinate. Default 'centroid-0'.
    x_col : str
        Column name for X centroid coordinate. Default 'centroid-1'.

    Returns
    -------
    np.ndarray
        Array of shape (2,) containing (y_center, x_center).
    """
    if center_coords is not None and len(center_coords) >= 2:
        return np.array([center_coords[0], center_coords[1]])

    if df.empty or y_col not in df or x_col not in df:
        raise ValueError(
            f"DataFrame must contain '{y_col}' and '{x_col}' columns to estimate center."
        )

    # Geometric mean of all cell centroids
    mean_y = float(df[y_col].mean())
    mean_x = float(df[x_col].mean())

    return np.array([mean_y, mean_x])


def sample_cell_wall(
    cell_id: int,
    labels_layer: np.ndarray,
    expanded_labels: np.ndarray,
    cell_wall_signal: np.ndarray,
    raw_signal_image: np.ndarray | None = None,
    bbox: tuple[int, int, int, int] | None = None,
    max_search_radius: int = 30,
    min_search_radius: int = 4,
    radius_scale: float = 0.33,
    sample_step: int = 3,
    max_rays: int = np.inf,
    offset: float = 0.5,
    erode_labs: int = 4,
    pixel_spacing: float = 1.0,
    verbose: bool = False,
) -> dict:
    """
    Measures wall thickness and geometry vectors using
    preprocessed signal, and samples real color values from raw_signal_image.
    """
    H, W = labels_layer.shape[:2]

    # Use provided padded bbox or fall back to local calculation
    if bbox is None:
        y_indices, x_indices = np.where(labels_layer == cell_id)
        if len(y_indices) == 0:
            return None

        pad = int(max_search_radius + (1.1 * max_search_radius) + erode_labs)
        raw_bbox = (
            y_indices.min(),
            x_indices.min(),
            y_indices.max() + 1,
            x_indices.max() + 1,
        )

        # Standardized (r0, r1, c0, c1) format
        r0_pad, r1_pad, c0_pad, c1_pad = get_safe_padded_bbox(
            raw_bbox, pad=pad, img_shape=(H, W)
        )
    else:
        # Passed bbox is already in (r0, r1, c0, c1) format
        r0_pad, r1_pad, c0_pad, c1_pad = bbox

    # Slice local sub-arrays (ROI)
    sub_labels = labels_layer[r0_pad:r1_pad, c0_pad:c1_pad]
    sub_expanded = expanded_labels[r0_pad:r1_pad, c0_pad:c1_pad]
    sub_wall_signal = cell_wall_signal[r0_pad:r1_pad, c0_pad:c1_pad]

    cell_mask = sub_labels == cell_id
    if not np.any(cell_mask):
        return None

    # Calculate adaptive search radius
    lumen_area = np.sum(cell_mask)
    eq_radius = np.sqrt(lumen_area / np.pi)
    adaptive_search_radius = float(
        np.clip(eq_radius * radius_scale, min_search_radius, max_search_radius)
    )

    # Fallback if raw image is not supplied
    if raw_signal_image is None:
        sub_raw = sub_wall_signal
    else:
        if (
            raw_signal_image.ndim == 3
            and raw_signal_image.shape[0] in (3, 4)
            and raw_signal_image.shape[-1] not in (3, 4)
        ):
            raw_signal_image = np.moveaxis(raw_signal_image, 0, -1)
        sub_raw = raw_signal_image[r0_pad:r1_pad, c0_pad:c1_pad].astype(
            np.float32
        )

    is_multichannel = sub_raw.ndim == 3 and sub_raw.shape[-1] >= 3

    # --- Estimate Baseline Lumen Threshold ---
    eroded_mask = binary_erosion(cell_mask, iterations=erode_labs)
    if not np.any(eroded_mask):
        eroded_mask = cell_mask

    # Calculate lumen intensity statistics
    lumen_mean = float(np.mean(sub_wall_signal[eroded_mask]))
    lumen_std = float(np.std(sub_wall_signal[eroded_mask]))
    wall_threshold = lumen_mean + offset * lumen_std

    if verbose:
        print(
            f'[DEBUG] Cell {cell_id} Lumen mean={lumen_mean:.4f}, std={lumen_std:.4f}, threshold={wall_threshold:.4f}'
        )

    # Mask signal to zero outside the target cell domain
    valid_region_mask = (
        (sub_labels == cell_id) | (sub_expanded == cell_id) | (sub_labels == 0)
    )
    isolated_wall_signal = np.where(valid_region_mask, sub_wall_signal, 0.0)

    # --- Contour Extraction ---
    contours = find_contours(sub_expanded == cell_id, level=0.5)
    if not contours:
        return {
            'label_id': cell_id,
            'mean_wall_thickness': np.nan,
            'median_wall_thickness': np.nan,
            'std_wall_thickness': np.nan,
            'min_wall_thickness': np.nan,
            '10_percentile_thickness': np.nan,
            '90_percentile_thickness': np.nan,
            'max_wall_thickness': np.nan,
            'mean_wall_color_r': np.nan,
            'mean_wall_color_g': np.nan,
            'mean_wall_color_b': np.nan,
            'lumen_baseline_intensity': lumen_mean,
            'wall_threshold': wall_threshold,
            'full_ray_lines': [],
            'thresholded_wall_lines': [],
            'anchor_points': [],
        }

    contour = max(contours, key=len)
    contour, normals = compute_contour_normals(contour)

    n_points = len(contour)
    target_rays = (
        max(1, n_points // sample_step) if sample_step > 0 else n_points
    )
    if verbose:
        print(
            f'[DEBUG] Cell {cell_id}: Contour points={n_points}, Target rays={target_rays}, Max rays={max_rays}'
        )
    if np.isfinite(max_rays) and target_rays > max_rays:
        n_rays = max(1, int(max_rays))
        # Distribute indices evenly between 0 and n_points - 1
        sample_indices = np.round(np.linspace(0, n_points - 1, n_rays)).astype(
            int
        )
    else:
        # Ensure step is at least 1 to avoid infinite loops/errors
        step = max(1, int(sample_step))
        sample_indices = np.arange(0, n_points, step, dtype=int)

    # Dynamic t-vector generation based on adaptive search radius
    # Maintain ~4 sub-pixel samples per pixel lenght along the ray
    num_samples = int(np.ceil(adaptive_search_radius * 8))
    t = np.linspace(
        -adaptive_search_radius, adaptive_search_radius, num_samples
    )
    center_idx = len(t) // 2

    thicknesses = []
    sampled_wall_colors = []
    full_ray_lines = []
    thresholded_wall_lines = []
    anchor_points = []

    # --- Ray Sampling ---
    for idx in sample_indices:
        pt_local = contour[idx]
        norm = normals[idx]

        ray_y_local = pt_local[0] + t * norm[0]
        ray_x_local = pt_local[1] + t * norm[1]

        # Convert local line endpoints back to global coordinates for napari visualization
        global_pt = [pt_local[0] + r0_pad, pt_local[1] + c0_pad]
        anchor_points.append(global_pt)
        full_ray_lines.append(
            [
                [ray_y_local[0] + r0_pad, ray_x_local[0] + c0_pad],
                [ray_y_local[-1] + r0_pad, ray_x_local[-1] + c0_pad],
            ]
        )

        coords_local = np.vstack([ray_y_local, ray_x_local])

        # Fast map_coordinates on small 2D ROI sub-array
        profile = map_coordinates(
            isolated_wall_signal, coords_local, order=1, mode='nearest'
        )
        above_threshold = profile > wall_threshold

        if np.any(above_threshold):
            labeled_profile, _ = label(above_threshold)
            central_label = labeled_profile[center_idx]

            if central_label == 0:
                active_indices = np.where(above_threshold)[0]
                if len(active_indices) > 0:
                    closest_idx = active_indices[
                        np.argmin(np.abs(active_indices - center_idx))
                    ]
                    central_label = labeled_profile[closest_idx]

            if central_label > 0:
                wall_indices = np.where(labeled_profile == central_label)[0]

                # Sub-pixel wall thickness interpolation
                idx_start = wall_indices[0]
                idx_end = wall_indices[-1]
                sub_idx_start, sub_idx_end = float(idx_start), float(idx_end)

                if 0 < idx_start < len(profile) - 1:
                    y0, y1 = profile[idx_start - 1], profile[idx_start]
                    if y1 != y0:
                        sub_idx_start = (
                            idx_start
                            - 1
                            + np.clip(
                                (wall_threshold - y0) / (y1 - y0), 0.0, 1.0
                            )
                        )

                if 0 < idx_end < len(profile) - 1:
                    y0, y1 = profile[idx_end], profile[idx_end + 1]
                    if y1 != y0:
                        sub_idx_end = idx_end + np.clip(
                            (wall_threshold - y0) / (y1 - y0), 0.0, 1.0
                        )

                sub_t_start = t[0] + sub_idx_start * (t[1] - t[0])
                sub_t_end = t[0] + sub_idx_end * (t[1] - t[0])

                thickness = (sub_t_end - sub_t_start) * pixel_spacing
                thicknesses.append(thickness)

                # Line endpoints converted back to global space
                start_y = pt_local[0] + sub_t_start * norm[0] + r0_pad
                start_x = pt_local[1] + sub_t_start * norm[1] + c0_pad
                end_y = pt_local[0] + sub_t_end * norm[0] + r0_pad
                end_x = pt_local[1] + sub_t_end * norm[1] + c0_pad
                thresholded_wall_lines.append(
                    [[start_y, start_x], [end_y, end_x]]
                )

                # Extract Color from local cropped raw image
                if is_multichannel:
                    rgb_vals = [
                        float(
                            np.mean(
                                map_coordinates(
                                    sub_raw[..., c],
                                    coords_local,
                                    order=1,
                                    mode='nearest',
                                )[wall_indices]
                            )
                        )
                        for c in range(3)
                    ]
                    sampled_wall_colors.append(rgb_vals)
                else:
                    raw_profile = map_coordinates(
                        sub_raw, coords_local, order=1, mode='nearest'
                    )
                    val = float(np.mean(raw_profile[wall_indices]))
                    sampled_wall_colors.append([val, val, val])

    has_thick = len(thicknesses) > 0

    if has_thick and len(sampled_wall_colors) > 0:
        avg_colors = np.mean(sampled_wall_colors, axis=0)
        mean_r, mean_g, mean_b = (
            float(avg_colors[0]),
            float(avg_colors[1]),
            float(avg_colors[2]),
        )
    else:
        mean_r = mean_g = mean_b = np.nan

    return {
        'label_id': cell_id,
        'mean_wall_thickness': float(np.mean(thicknesses))
        if has_thick
        else np.nan,
        'median_wall_thickness': float(np.median(thicknesses))
        if has_thick
        else np.nan,
        'std_wall_thickness': float(np.std(thicknesses))
        if has_thick
        else np.nan,
        'min_wall_thickness': float(np.min(thicknesses))
        if has_thick
        else np.nan,
        # "10_percentile_thickness": float(np.percentile(thicknesses, 10)) if has_thick else np.nan,
        # "90_percentile_thickness": float(np.percentile(thicknesses, 90)) if has_thick else np.nan,
        'max_wall_thickness': float(np.max(thicknesses))
        if has_thick
        else np.nan,
        'mean_wall_color_r': mean_r,
        'mean_wall_color_g': mean_g,
        'mean_wall_color_b': mean_b,
        'lumen_baseline_intensity': lumen_mean,
        'wall_threshold': wall_threshold,
        'full_ray_lines': full_ray_lines,
        'thresholded_wall_lines': thresholded_wall_lines,
        'anchor_points': anchor_points,
    }


def _transform_kwargs(scale, translate):
    """add_* keyword arguments carrying a layer's transform (empty if none)."""
    tf = {}
    if scale is not None:
        tf['scale'] = tuple(scale)
    if translate is not None:
        tf['translate'] = tuple(translate)
    return tf


def visualize_cell_rays(
    viewer: 'napari.Viewer',
    labels_layer: np.ndarray,
    cell_wall_signal: np.ndarray | None = None,
    target_label_id: int = 1,
    max_search_radius: int = 30,
    min_search_radius: int = 4,
    radius_scale: float = 0.33,
    expand_labs_dist: int = 4,
    erode_labs: int = 4,
    sample_step: int = 3,
    max_rays: int = np.inf,
    offset: float = 0.5,
    invert_signal: bool = False,
    show_processed_signal: bool = True,
    scale=None,
    translate=None,
) -> None:
    """Renders rays, detected wall segments, and anchors for a single targeted cell label."""
    if labels_layer is None or target_label_id is None:
        return

    # Same transform as the labels layer, so overlays line up on cropped layers
    tf = _transform_kwargs(scale, translate)

    # Convert binary mask slice to (min_r, min_c, max_r, max_c)
    locs = find_objects((labels_layer == target_label_id).astype(int))
    # locs = find_objects(labels_layer == target_label_id)
    if not locs or locs[0] is None:
        print(
            f'[w-seg] Target label ID {target_label_id} not found in labels layer.'
        )
        return

    y_slice, x_slice = locs[0]
    raw_bbox = (y_slice.start, x_slice.start, y_slice.stop, x_slice.stop)

    # Expand bounding box using standardized (r0, r1, c0, c1)
    bbox_padding = int(
        max_search_radius + (1.3 * max_search_radius) + erode_labs
    )
    bbox = get_safe_padded_bbox(
        raw_bbox, pad=bbox_padding, img_shape=labels_layer.shape
    )

    # Preprocess Signal
    if cell_wall_signal is not None:
        cell_wall_signal = preprocess_signal_layer(
            cell_wall_signal, invert_signal=invert_signal
        )
    else:
        cell_wall_signal = (labels_layer > 0).astype(np.float32)

    # Signal Debug Layer
    if show_processed_signal:
        sig_layer_name = 'debug_processed_signal'
        if sig_layer_name in viewer.layers:
            viewer.layers[sig_layer_name].data = cell_wall_signal
            viewer.layers[sig_layer_name].scale = tf.get('scale', (1.0, 1.0))
            viewer.layers[sig_layer_name].translate = tf.get(
                'translate', (0.0, 0.0)
            )
            viewer.layers[sig_layer_name].refresh()
        else:
            viewer.add_image(
                cell_wall_signal,
                name=sig_layer_name,
                colormap='gray',
                contrast_limits=(0.0, 1.0),
                opacity=0.7,
                **tf,
            )

    # Expand labels
    expanded_labels = expand_labels(labels_layer, distance=expand_labs_dist)

    # Run core single-cell engine
    result = sample_cell_wall(
        cell_id=target_label_id,
        labels_layer=labels_layer,
        expanded_labels=expanded_labels,
        cell_wall_signal=cell_wall_signal,
        bbox=bbox,
        max_search_radius=max_search_radius,
        min_search_radius=min_search_radius,
        radius_scale=radius_scale,
        sample_step=sample_step,
        max_rays=max_rays,
        offset=offset,
        erode_labs=erode_labs,
        verbose=True,
    )

    if result is None or not result['full_ray_lines']:
        print(
            f'[w-seg] Warning: No contour/rays computed for Label ID {target_label_id}'
        )
        return

    # Clear prior debug layers for this specific label
    layer_prefixes = (
        f'rays_full_{target_label_id}',
        f'rays_wall_{target_label_id}',
        f'anchors_{target_label_id}',
        f'bbox_{target_label_id}',
    )
    for lyr in list(viewer.layers):
        if lyr.name.startswith(layer_prefixes):
            viewer.layers.remove(lyr)

    # Construct Bounding Box Rectangle: [min_r, min_c] -> [max_r, max_c]
    r0, r1, c0, c1 = bbox
    bbox_rect = np.array([[r0, c0], [r0, c1], [r1, c1], [r1, c0]])

    # Render Bounding Box
    viewer.add_shapes(
        [bbox_rect],
        shape_type='rectangle',
        edge_color='magenta',
        face_color='transparent',
        edge_width=1.0,
        name=f'bbox_{target_label_id}',
        opacity=0.8,
        **tf,
    )

    # Add shapes/points to Napari
    if result['full_ray_lines']:
        viewer.add_shapes(
            np.array(result['full_ray_lines']),
            shape_type='path',
            edge_color='cyan',
            edge_width=0.4,
            name=f'rays_full_{target_label_id}',
            opacity=0.5,
            **tf,
        )

    if result['thresholded_wall_lines']:
        viewer.add_shapes(
            np.array(result['thresholded_wall_lines']),
            shape_type='path',
            edge_color='lime',
            edge_width=1.5,
            name=f'rays_wall_{target_label_id}',
            opacity=0.9,
            **tf,
        )

    if result['anchor_points']:
        viewer.add_points(
            np.array(result['anchor_points']),
            size=2,
            face_color='yellow',
            border_color='transparent',
            name=f'anchors_{target_label_id}',
            opacity=0.8,
            **tf,
        )
    print(
        f'[w-seg] Cell {target_label_id}: Plotted {len(result["full_ray_lines"])} rays | '
        f'{len(result["thresholded_wall_lines"])} valid wall segments | '
        f'Threshold = {result["wall_threshold"]:.4f}'
    )
    print('mean_wall_thickness:', result['mean_wall_thickness'])
    print('std_wall_thickness:', result['std_wall_thickness'])
    print('min_wall_thickness:', result['min_wall_thickness'])
    print('max_wall_thickness:', result['max_wall_thickness'])
    print('lumen_baseline_intensity:', result['lumen_baseline_intensity'])


def visualize_cell_neighborhood(
    viewer: napari.Viewer,
    cell_id: int,
    labels_layer: np.ndarray,
    df_single_cell_metrics: pd.DataFrame | None = None,
    search_radius: int = 8,
    use_delaunay_fallback: bool = True,
    max_delaunay_dist_px: float = 120.0,
    scale=None,
    translate=None,
):
    """Visualizes a target cell, its physical/Delaunay neighbors, and connectivity

    edges directly in a napari viewer, computing centroids on the fly from the
    labels array.
    """
    if cell_id not in labels_layer:
        raise ValueError(f'Cell ID {cell_id} was not found in labels_layer.')
    tf = _transform_kwargs(scale, translate)

    # 1. Compute Centroids Directly from Labels Layer
    props = regionprops_table(labels_layer, properties=('label', 'centroid'))
    df_centroids = pd.DataFrame(props).rename(columns={'label': 'label_id'})

    # Optionally merge with single-cell metrics if provided
    if df_single_cell_metrics is not None:
        df_metrics = df_single_cell_metrics.copy()
        if (
            'label_id' not in df_metrics.columns
            and 'label' in df_metrics.columns
        ):
            df_metrics.rename(columns={'label': 'label_id'}, inplace=True)
        df_full = pd.merge(df_centroids, df_metrics, on='label_id', how='left')
    else:
        df_full = df_centroids

    df_full = df_full.drop_duplicates(subset=['label_id']).copy()

    label_ids = df_full['label_id'].values
    centroids = df_full[['centroid-0', 'centroid-1']].values
    id_to_centroid = dict(zip(label_ids, centroids, strict=True))

    # 2. Build Physical Contact Graph
    G = build_cell_contact_graph(labels_layer, search_radius)

    # 3. Apply Delaunay Fallback using Computed Centroids
    if use_delaunay_fallback and len(centroids) >= 4:
        tri = Delaunay(centroids)
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    u_idx, v_idx = simplex[i], simplex[j]
                    u_id, v_id = label_ids[u_idx], label_ids[v_idx]

                    if G.degree(u_id) == 0 or G.degree(v_id) == 0:
                        dist = np.linalg.norm(
                            centroids[u_idx] - centroids[v_idx]
                        )
                        if dist <= max_delaunay_dist_px:
                            G.add_edge(u_id, v_id, contact_pixels=0)

    # 4. Separate Neighbors into Physical vs. Delaunay
    neighbors = list(G.neighbors(cell_id)) if cell_id in G else []
    physical_nbrs = []
    delaunay_nbrs = []

    for nbr in neighbors:
        edge_data = G.get_edge_data(cell_id, nbr, {})
        if edge_data.get('contact_pixels', 0) > 0:
            physical_nbrs.append(nbr)
        else:
            delaunay_nbrs.append(nbr)

    # 5. Create Color-Coded Mask
    highlight_mask = np.zeros_like(labels_layer, dtype=np.uint32)

    # 1 = Target Cell (Red), 2 = Physical Neighbor (Cyan), 3 = Delaunay Fallback (Yellow)
    highlight_mask[labels_layer == cell_id] = 1
    for nbr in physical_nbrs:
        highlight_mask[labels_layer == nbr] = 2
    for nbr in delaunay_nbrs:
        highlight_mask[labels_layer == nbr] = 3

    color_dict = {
        0: (0.0, 0.0, 0.0, 0.0),  # Label 0 -> Transparent
        1: (0.8, 0.1, 0.0, 0.85),  # Label 1 -> Red
        2: (0.0, 0.8, 0.2, 0.85),  # Label 2 -> Green
        3: (1.0, 0.8, 0.0, 0.7),  # Delaunay Neighbor
    }
    # Layer cmap
    cmap = DirectLabelColormap(
        color_dict=color_dict,
        default_color=(0.0, 0.0, 0.0, 0.0),
    )
    viewer.add_labels(
        highlight_mask,
        name=f'Cell {cell_id} Neighborhood',
        colormap=cmap,
        opacity=0.8,
        **tf,
    )

    # 6. Draw Neighbor Connection Lines
    lines = []
    line_colors = []
    focal_centroid = id_to_centroid[cell_id]

    for nbr in physical_nbrs:
        lines.append([focal_centroid, id_to_centroid[nbr]])
        line_colors.append([0.0, 0.8, 1.0, 1.0])  # Cyan

    for nbr in delaunay_nbrs:
        lines.append([focal_centroid, id_to_centroid[nbr]])
        line_colors.append([1.0, 0.8, 0.0, 1.0])  # Yellow

    if lines:
        viewer.add_shapes(
            lines,
            shape_type='line',
            edge_color=line_colors,
            edge_width=2.5,
            name=f'Cell {cell_id} Connections',
            **tf,
        )

    print(
        f'[INFO] Cell {cell_id}: Found {len(physical_nbrs)} physical contact neighbors and {len(delaunay_nbrs)} Delaunay fallback neighbors.'
    )


def measure_cell_morphology(
    labels_layer: np.ndarray,
    cell_wall_signal: np.ndarray,
    pixel_spacing: float = 1.0,
    max_search_radius: int = 12,
    min_search_radius: int = 4,
    radius_scale: float = 0.33,
    sample_step: int = 3,
    max_rays: int = np.inf,
    offset: float = 0.5,
    erode_labs: int = 4,
    expand_labs_dist: int = 4,
    num_workers: int = 8,
    clear_borders: bool = True,
    invert_signal: bool = False,
    center_coords: tuple = None,
    radial_sym: bool = False,
    use_YOLO: bool = False,
    YOLO_weights: str = '',
    yolo_image: np.ndarray = None,
    yolo_image_scale=None,
    yolo_image_translate=None,
    labels_scale=None,
    labels_translate=None,
    worker=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Multithreaded batch computation of cell wall metrics for all cell labels.
    """
    # Take a copy of labels
    labels_layer = labels_layer.copy()

    # Clean border cells
    if clear_borders:
        labels_layer = clean_border_cells(labels_layer)
    # Relabel from 1 to n
    # labels_layer = fix_labels(labels_layer)
    unique_ids = np.unique(labels_layer)
    unique_ids = unique_ids[unique_ids != 0]

    if len(unique_ids) == 0:
        print('[w-seg] Warning: No labels found.')
        return pd.DataFrame(), labels_layer, {}

    # Preprocess Signal
    if cell_wall_signal is not None:
        raw_signal = cell_wall_signal.copy()
        cell_wall_signal = preprocess_signal_layer(
            cell_wall_signal, invert_signal=invert_signal
        )
    else:
        cell_wall_signal = (labels_layer > 0).astype(np.float32)
        raw_signal = cell_wall_signal.copy()
    # Compute base morphology using skimage regionprops
    props_to_extract = (
        'label',
        'bbox',
        'area',
        'area_convex',
        'perimeter',
        'perimeter_crofton',
        'equivalent_diameter_area',
        'eccentricity',
        'solidity',
        'extent',
        'major_axis_length',
        'minor_axis_length',
        'orientation',
        'centroid',
    )
    # Physical spacing scale
    spacing = (pixel_spacing, pixel_spacing) if pixel_spacing else None
    # Compute regionprops table for all labels
    print('[w-seg] Computing base morphology...')
    props_dict = regionprops_table(
        labels_layer, properties=props_to_extract, spacing=spacing
    )
    # Prepare morphology base dataframe
    df_morphology = pd.DataFrame(props_dict)
    if 'label' in df_morphology.columns:
        df_morphology.rename(columns={'label': 'label_id'}, inplace=True)

    # Build padded bounding box lookup: cell_id -> safe (min_r, min_c, max_r, max_c)
    bbox_padding = int(
        max_search_radius + (1.1 * max_search_radius) + erode_labs
    )
    img_shape = labels_layer.shape
    bbox_lookup = {
        int(row['label_id']): get_safe_padded_bbox(
            (
                int(row['bbox-0']),
                int(row['bbox-1']),
                int(row['bbox-2']),
                int(row['bbox-3']),
            ),
            pad=bbox_padding,
            img_shape=img_shape,
        )
        for _, row in df_morphology.iterrows()
    }
    # Expand labels
    expanded_labels = expand_labels(labels_layer, distance=expand_labs_dist)
    records = []
    tot_cells = len(unique_ids)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers
    ) as executor:
        # Submit all single-cell tasks
        future_to_cell = {
            executor.submit(
                sample_cell_wall,
                cell_id=cell_id,
                labels_layer=labels_layer,
                expanded_labels=expanded_labels,
                cell_wall_signal=cell_wall_signal,
                raw_signal_image=raw_signal,
                bbox=bbox_lookup.get(int(cell_id)),
                max_search_radius=max_search_radius,
                min_search_radius=min_search_radius,
                radius_scale=radius_scale,
                sample_step=sample_step,
                max_rays=max_rays,
                offset=offset,
                erode_labs=erode_labs,
                pixel_spacing=pixel_spacing,
            ): cell_id
            for cell_id in unique_ids
        }
        completed = 0
        for future in as_completed(future_to_cell):
            # 1st cancellation check
            if worker is not None and worker.abort_requested:
                print(
                    '\n[w-seg] Cancellation requested. Shutting down worker threads...'
                )
                executor.shutdown(wait=False, cancel_futures=True)
                return pd.DataFrame(), labels_layer, {}

            cell_id = future_to_cell[future]
            try:
                res = future.result()
                if res is not None:
                    # Strip out heavy visual vector data to free RAM
                    res.pop('full_ray_lines', None)
                    res.pop('thresholded_wall_lines', None)
                    res.pop('anchor_points', None)

                    # force key name and integer type alignment
                    res['label_id'] = int(cell_id)
                    records.append(res)
            except (RuntimeError, ValueError, TypeError, OSError) as e:
                print(f'\n[w-seg] Cell {cell_id} failed measurement: {e}')

            completed += 1
            # Yield progress
            if worker is not None:
                yield (
                    completed,
                    tot_cells,
                    f'Ray sampling: {completed}/{tot_cells} cells',
                )
            if verbose:
                sys.stdout.write(
                    f'\r[w-seg] Extracting cell wall features: {completed}/{tot_cells}'
                )
                sys.stdout.flush()

    # --- Merge Sequence ---
    df_wall = pd.DataFrame(records)

    # Force explicit int type before merge
    df_morphology['label_id'] = df_morphology['label_id'].astype(int)

    if not df_wall.empty and 'label_id' in df_wall.columns:
        df_wall['label_id'] = df_wall['label_id'].astype(int)

        # Deduplicate if sample_cell_wall returned multiple points per cell
        if df_wall['label_id'].duplicated().any():
            numeric_cols = [
                c
                for c in df_wall.select_dtypes(include=[np.number]).columns
                if c != 'label_id'
            ]
            df_wall = df_wall.groupby('label_id', as_index=False)[
                numeric_cols
            ].mean()

        # Left join morphology with wall measurements
        df_combined = pd.merge(
            df_morphology, df_wall, on='label_id', how='left'
        )
    else:
        df_combined = df_morphology.copy()

    # Fill NaNs for any missing wall measurements
    df_combined.fillna(0, inplace=True)

    if verbose:
        sys.stdout.write(
            f'\r[w-seg] Extracted all cell wall features from: {completed}/{tot_cells} cells\n'
        )

    # Derived descriptor: Circularity
    if 'area' in df_combined.columns and 'perimeter' in df_combined.columns:
        df_combined['circularity'] = np.where(
            df_combined['perimeter'] > 0,
            4 * np.pi * df_combined['area'] / (df_combined['perimeter'] ** 2),
            0.0,  # Fallback value when perimeter is 0 or invalid
        )

    # Centroids in df_combined are in physical units (um). A user-picked
    # center arrives in array pixels, so convert it to the same units.
    if center_coords is None and radial_sym:
        print('[w-seg] calculating center point')
        center_coords = resolve_center_coords(df_combined)
    elif center_coords is not None:
        center_coords = tuple(
            float(c) * pixel_spacing for c in center_coords[-2:]
        )

    # Calculate distance from center point
    df_combined = add_radial_features(
        df_combined,
        center_coords=center_coords,
        enabled=radial_sym,
        pixel_scale=1.0,  # centroids and center are already in um
        angle_unit='degrees',
    )

    # Classify cell crops using YOLO model
    class_map = {}
    if use_YOLO and YOLO_weights:
        # 2nd Cancellation check
        if worker is not None and worker.abort_requested:
            return pd.DataFrame(), labels_layer, {}
        if worker is not None:
            yield (tot_cells, tot_cells, 'Running YOLO classification....')
        print('[w-seg] Running YOLO cell-type classification...')
        yolo_model = load_yolo_model(YOLO_weights)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # The crops are cut from the YOLO image (the RGB image), not from the
        # cell wall signal. Cell positions were measured on the labels array,
        # so they are mapped through world coordinates into the pixel grid of
        # that image (this also handles pixel spacing and ROI-cropped layers).
        if yolo_image is None:
            yolo_image = raw_signal
            yolo_image_scale, yolo_image_translate = (
                labels_scale,
                labels_translate,
            )
        df_for_crops = label_frame_to_image_pixels(
            df_combined,
            pixel_spacing=pixel_spacing,
            labels_scale=labels_scale,
            labels_translate=labels_translate,
            image_scale=yolo_image_scale,
            image_translate=yolo_image_translate,
        )
        df_pred, class_map = predict_yolo_tissue_on_image(
            df_new=df_for_crops,
            image_np=yolo_image,
            img_translation=None,
            img_scale=None,
            yolo_model=yolo_model,
            target_size=256,
            context_factor=2.5,
            device=device,
        )
        # Keep the measured values; add only the YOLO columns
        for col in df_pred.columns.difference(df_combined.columns):
            df_combined[col] = df_pred[col]
    if worker is not None and worker._abort_requested:
        return pd.DataFrame(), labels_layer, {}
    if worker is not None:
        yield (tot_cells, tot_cells, 'Exctracting spatial feature matirx...')
    # Finally build and return spatial features for each cell
    df_final = extract_spatial_feature_matrix(
        labels_layer, df_combined, search_radius=max_search_radius
    )

    return df_final, labels_layer, class_map


# =========================================================================
# MorphologyWidget: Napari GUI Container for Morphology Measurement and Visualization
# =========================================================================


class MorphologyWidget(QWidget):
    """
    Standalone Container class managing morphology calculation, feature colormap visualization,
    and ray tracing debugging in napari.
    """

    # Signal for passing measurement results back to tab4
    measurement_finished_signal = Signal(pd.DataFrame, np.ndarray, str, dict)

    def __init__(self, viewer: 'napari.Viewer', parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self._current_worker = None

        # Internal state caches
        self.cached_df = None
        self.cached_labels_data = None
        self.cached_feature_extraction_config = {}

        # Center point tracking
        self.center_layer_name = 'Center point'
        self.center_coords = None  # (y, x) coordinates of the center point

        # Paired YOLO classifier of the model loaded in tab 1
        self._paired_yolo = None
        self._last_model_name = None

        # Build UI & Connect Signals
        self._init_ui()
        self._connect_signals()
        self.refresh_paired_model()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        # -----------------------------------------------------------------
        # LAYER SELECTION GROUP (Form Layout)
        # -----------------------------------------------------------------
        layer_group = QGroupBox('Layer Selection')
        layer_layout = QFormLayout()

        self.labels_combo = QComboBox()
        self.signal_combo = QComboBox()

        layer_layout.addRow('Labels Layer:', self.labels_combo)
        layer_layout.addRow('Signal Layer:', self.signal_combo)
        layer_group.setLayout(layer_layout)
        layout.addWidget(layer_group)

        # -----------------------------------------------------------------
        # MORPHOLOGY PARAMETERS GROUP (Form Layout)
        # -----------------------------------------------------------------
        params_group = QGroupBox('Sampling Parameters')
        params_layout = QFormLayout()

        self.pixel_spacing_spin = QDoubleSpinBox()
        self.pixel_spacing_spin.setRange(0.001, 100.0)
        self.pixel_spacing_spin.setValue(1.0)

        self.max_search_spin = QSpinBox()
        self.max_search_spin.setRange(1, 200)
        self.max_search_spin.setValue(30)

        self.min_search_spin = QSpinBox()
        self.min_search_spin.setRange(2, 10)
        self.min_search_spin.setValue(4)

        self.radius_scale_spin = QDoubleSpinBox()
        self.radius_scale_spin.setRange(0.1, 1.0)
        self.radius_scale_spin.setValue(0.33)

        self.expand_labs_spin = QSpinBox()
        self.expand_labs_spin.setRange(0, 20)
        self.expand_labs_spin.setValue(4)

        self.erode_labs_spin = QSpinBox()
        self.erode_labs_spin.setRange(1, 20)
        self.erode_labs_spin.setValue(4)

        self.sample_step_spin = QSpinBox()
        self.sample_step_spin.setRange(1, 50)
        self.sample_step_spin.setValue(3)

        self.max_rays_spin = QSpinBox()
        self.max_rays_spin.setRange(5, 1000)
        self.max_rays_spin.setValue(80)

        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-10.0, 15.0)
        self.offset_spin.setValue(0.5)

        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(1, 32)
        self.num_workers_spin.setValue(8)

        # Add pairs directly to the form
        params_layout.addRow('Pixel Spacing (um/px):', self.pixel_spacing_spin)
        params_layout.addRow('Max Search Radius (px):', self.max_search_spin)
        params_layout.addRow('Min Search Radius (px):', self.min_search_spin)
        params_layout.addRow('Ray search scale:', self.radius_scale_spin)
        params_layout.addRow('Label Expansion (px):', self.expand_labs_spin)
        params_layout.addRow('Lumen Erosion (px):', self.erode_labs_spin)
        params_layout.addRow('Ray Sampling step:', self.sample_step_spin)
        params_layout.addRow('Max Ray samples:', self.max_rays_spin)
        params_layout.addRow('Threshold Offset (σ):', self.offset_spin)
        params_layout.addRow('Parallel Threads:', self.num_workers_spin)

        # Checkboxes spans both columns
        self.clear_border_cb = QCheckBox('Clear Border Cells')
        self.clear_border_cb.setChecked(False)
        self.invert_signal_cb = QCheckBox('Invert Signal Intensity')
        self.invert_signal_cb.setChecked(True)
        self.show_processed_signal_cb = QCheckBox(
            'Show processed Signal Image'
        )
        self.radial_symmetry_cb = QCheckBox(
            'Use Radial Symmetry and center point'
        )

        params_layout.addRow(self.clear_border_cb)
        params_layout.addRow(self.invert_signal_cb)
        params_layout.addRow(self.show_processed_signal_cb)
        params_layout.addRow(self.radial_symmetry_cb)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # -----------------------------------------------------------------
        # YOLO SETTINGS GROUP (Form Layout)
        # -----------------------------------------------------------------
        YOLO_group = QGroupBox('YOLO Parameters')
        YOLO_layout = QFormLayout()

        self.use_YOLO_cb = QCheckBox('Use YOLO Classifier')
        self.use_YOLO_cb.setChecked(True)
        self.YOLO_layer_combo = QComboBox()
        # The classifier comes with the model loaded in tab 1 (see
        # refresh_paired_model); there is nothing to choose here.
        self.YOLO_paired_label = QLabel('No model loaded')
        self.YOLO_paired_label.setWordWrap(True)

        YOLO_layout.addRow(self.use_YOLO_cb)
        YOLO_layout.addRow('RGB image for YOLO:', self.YOLO_layer_combo)
        YOLO_layout.addRow('Paired YOLO weights:', self.YOLO_paired_label)

        YOLO_group.setLayout(YOLO_layout)
        layout.addWidget(YOLO_group)

        # -----------------------------------------------------------------
        # ACTION CONTROL GROUP (Vertical + Horizontal Button Pair)
        # -----------------------------------------------------------------
        action_group = QGroupBox('Measure Morphology')
        action_layout = QVBoxLayout()

        self.set_center_point_btn = QPushButton('Manually Set Center Point')

        self.run_btn = QPushButton('Measure Cell Walls & Morphology')
        self.run_btn.setStyleSheet(
            'font-weight: bold; background-color: #2b5b84; color: white; padding: 6px; border-radius: 4px;'
        )

        self.cancel_btn = QPushButton('Cancel Measurement')
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet(
            'font-weight: bold; background-color: #8b2626; color: white; padding: 6px; border-radius: 4px;'
        )

        # Side-by-side run / cancel buttons
        run_cancel_layout = QHBoxLayout()
        run_cancel_layout.addWidget(self.run_btn)
        run_cancel_layout.addWidget(self.cancel_btn)

        action_layout.addWidget(self.set_center_point_btn)
        action_layout.addLayout(run_cancel_layout)

        action_group.setLayout(action_layout)
        layout.addWidget(action_group)

        # -----------------------------------------------------------------
        # VISUALIZATION GROUP (Form Layout)
        # -----------------------------------------------------------------
        vis_group = QGroupBox('Feature Visualization')
        vis_layout = QFormLayout()

        self.feature_combo = QComboBox()
        self.colormap_combo = QComboBox()
        self.vis_btn = QPushButton('Visualize Feature Map')

        colormap_names = sorted(AVAILABLE_COLORMAPS.keys())
        self.colormap_combo.addItems(colormap_names)
        default_cmap = 'viridis'
        if default_cmap in colormap_names:
            self.colormap_combo.setCurrentText(default_cmap)

        vis_layout.addRow('Feature to Visualize:', self.feature_combo)
        vis_layout.addRow('Colormap:', self.colormap_combo)
        vis_layout.addRow(self.vis_btn)

        vis_group.setLayout(vis_layout)
        layout.addWidget(vis_group)

        # -----------------------------------------------------------------
        # INSPECTION TOOLS (Form Layout + Horizontal Buttons)
        # -----------------------------------------------------------------
        inspect_group = QGroupBox('Label Inspection Tools')
        inspect_layout = QFormLayout()

        self.inspect_cell_id_spin = QSpinBox()
        self.inspect_cell_id_spin.setRange(1, 999999)
        self.inspect_cell_id_spin.setValue(1)

        self.draw_rays_btn = QPushButton('Draw Sampling Rays')
        self.draw_neighbours_btn = QPushButton('Draw Neighbours')

        inspect_btn_layout = QHBoxLayout()
        inspect_btn_layout.addWidget(self.draw_rays_btn)
        inspect_btn_layout.addWidget(self.draw_neighbours_btn)

        inspect_layout.addRow(
            'Label ID to inspect:', self.inspect_cell_id_spin
        )
        inspect_layout.addRow(inspect_btn_layout)

        inspect_group.setLayout(inspect_layout)
        layout.addWidget(inspect_group)

        layout.addStretch()

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    def _connect_signals(self):
        """Connects all PyQt/PySide UI signals, button callbacks, and Napari viewer listeners."""
        # --- Action Button Connections ---
        self.set_center_point_btn.clicked.connect(
            self._enable_center_point_picking
        )
        self.run_btn.clicked.connect(self._on_measure_clicked)
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        self.vis_btn.clicked.connect(self._on_visualize_clicked)
        self.draw_rays_btn.clicked.connect(self._on_draw_rays_clicked)
        self.draw_neighbours_btn.clicked.connect(
            self._on_draw_neighbours_clicked
        )

        # --- Checkbox Dependent UI Toggles ---
        # YOLO Controls enable/disable
        self.use_YOLO_cb.toggled.connect(self.YOLO_layer_combo.setEnabled)
        self.YOLO_layer_combo.setEnabled(self.use_YOLO_cb.isChecked())

        # Radial Symmetry / Center Point Button toggle
        self.radial_symmetry_cb.toggled.connect(
            self._on_radial_symmetry_toggled
        )
        self.set_center_point_btn.setEnabled(
            self.radial_symmetry_cb.isChecked()
        )

        # --- Dropdown Selection Sync ---
        # Update spinbox bounds when selected label layer changes
        self.labels_combo.currentIndexChanged.connect(
            self._update_inspection_bounds
        )
        # Only images with the same size as the labels can be paired with them
        self.labels_combo.currentIndexChanged.connect(
            self._refresh_image_choices
        )

        # --- Napari Layer List Event Listeners ---
        if self.viewer is not None:
            # Refresh layer combo boxes on layer addition, removal, or reordering
            self.viewer.layers.events.inserted.connect(
                self.refresh_layer_choices
            )
            self.viewer.layers.events.removed.connect(
                self.refresh_layer_choices
            )
            self.viewer.layers.events.reordered.connect(
                self.refresh_layer_choices
            )

            # Initial populating of layer choices
            self.refresh_layer_choices()

    def _on_measure_clicked(self):
        """Snapshots parameters on main thread and launches background measurement worker."""
        if self._current_worker is not None:
            print('[w-seg] Job already running. Please wait or cancel.')
            return

        lbl_layer = self.labels_combo.currentData()
        sig_layer = self.signal_combo.currentData()

        if lbl_layer is None or lbl_layer.data is None:
            print('[w-seg] Error: Please select a valid Labels layer.')
            return

        # 1. Snapshot UI parameters ON MAIN THREAD prior to worker thread start
        lbl_data = lbl_layer.data
        sig_data = sig_layer.data if sig_layer is not None else None

        fe_config = self.get_feature_extraction_config()
        num_workers = self.num_workers_spin.value()
        invert_signal = self.invert_signal_cb.isChecked()
        self.refresh_paired_model()
        yolo_weights = self._paired_yolo
        use_yolo = self.use_YOLO_cb.isChecked() and yolo_weights is not None
        yolo_layer = self.YOLO_layer_combo.currentData() if use_yolo else None
        yolo_data = yolo_layer.data if yolo_layer is not None else None
        yolo_scale = yolo_layer.scale if yolo_layer is not None else None
        yolo_translate = (
            yolo_layer.translate if yolo_layer is not None else None
        )
        labels_scale = lbl_layer.scale
        labels_translate = lbl_layer.translate
        clear_borders = self.clear_border_cb.isChecked()
        center_coords = self.center_coords
        if center_coords is not None:
            # The picked point is in world coordinates; use array pixels
            center_coords = tuple(
                float(world_to_data(c, labels_scale, labels_translate, axis=a))
                for a, c in enumerate(center_coords[-2:])
            )

        # Snapshot worker reference on main thread
        worker_ref = None

        @thread_worker
        def run_measurement_worker():
            return (
                yield from measure_cell_morphology(
                    labels_layer=lbl_data,
                    cell_wall_signal=sig_data,
                    num_workers=num_workers,
                    invert_signal=invert_signal,
                    center_coords=center_coords,
                    use_YOLO=use_yolo,
                    YOLO_weights=yolo_weights,
                    yolo_image=yolo_data,
                    yolo_image_scale=yolo_scale,
                    yolo_image_translate=yolo_translate,
                    labels_scale=labels_scale,
                    labels_translate=labels_translate,
                    clear_borders=clear_borders,
                    worker=worker_ref,
                    **fe_config,
                )
            )

        self._current_worker = run_measurement_worker()
        worker_ref = self._current_worker

        # 4. Connect worker Qt signals
        self._current_worker.yielded.connect(self._on_yield_progress)
        self._current_worker.returned.connect(self._on_measurement_finished)
        self._current_worker.finished.connect(self._on_worker_finished)
        self._current_worker.errored.connect(self._on_worker_errored)

        # 5. Update Qt UI Controls
        self.run_btn.setEnabled(False)
        self.run_btn.setText('Measuring...')
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setText('Cancel Measurement')

        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)

        # Start thread execution
        self._current_worker.start()

    def _on_cancel_clicked(self):
        """Triggers cancellation request on active thread worker."""
        if self._current_worker is not None:
            self.cancel_btn.setText('Cancelling...')
            self.cancel_btn.setEnabled(False)
            self._current_worker.quit()  # Signals worker.abort_requested = True

    def _on_yield_progress(self, progress_info):
        """Runs on UI thread whenever worker yields (completed, total, status_msg)."""
        if not progress_info or len(progress_info) < 3:
            return

        completed, total, status_msg = progress_info

        # Update Napari status bar
        if self.viewer is not None:
            self.viewer.status = f'[w-seg] {status_msg}'

        # Update QProgressBar if available
        if hasattr(self, 'progress_bar') and total > 0:
            progress_pct = int((completed / total) * 100)
            self.progress_bar.setValue(progress_pct)

    def _on_measurement_finished(self, return_value):
        """Runs on UI thread when measure_cell_morphology completes and returns data."""
        if not return_value:
            print('[w-seg] Measurement returned no results or was cancelled.')
            return

        df_results, measured_labels_data, class_map = return_value

        # Cache data directly on instance
        self.cached_df = df_results
        self.cached_labels_data = measured_labels_data
        self.cached_feature_extraction_config = (
            self.get_feature_extraction_config()
        )

        if df_results.empty:
            print('[w-seg] Measurement returned empty dataframe.')
            return

        # Pass signal to tab4 for downstream processing
        self.measurement_finished_signal.emit(
            df_results,
            measured_labels_data,
            self.labels_combo.currentText(),
            self.cached_feature_extraction_config,
        )

        # Populate numeric features dropdown for visualization
        self._update_feature_dropdown(df_results)

        # Create YOLO Cell Types overlay layer if classification map returned
        if self.use_YOLO_cb.isChecked() and class_map:
            ref_layer = self.labels_combo.currentData()
            if ref_layer is not None:
                create_yolo_classification_labels_layer(
                    viewer=self.viewer,
                    df_results=df_results,
                    reference_labels_layer=ref_layer,
                    class_names_map=class_map,
                    pred_column='yolo_pred_tissue',
                    layer_name='YOLO Cell Types',
                    scale=getattr(ref_layer, 'scale', None),
                    translate=getattr(ref_layer, 'translate', None),
                )

        print(f'[w-seg] Successfully processed {len(df_results)} cells!')

    def _on_worker_finished(self):
        """Resets buttons and progress UI when worker finishes normally or aborts."""
        self._current_worker = None
        self.run_btn.setEnabled(True)
        self.run_btn.setText('Measure Cell Walls & Morphology')
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText('Cancel Measurement')

        if hasattr(self, 'progress_bar'):
            self.progress_bar.setVisible(False)

    def _on_worker_errored(self, exc):
        """Handles background exceptions raised inside worker thread."""
        print(f'[w-seg] Worker error occurred: {exc}')
        self._on_worker_finished()

    def _on_visualize_clicked(self):
        """Maps quantitative cell feature values onto a copy of the Labels layer using a custom colormap."""
        lbl_layer = self.labels_combo.currentData()
        feat = self.feature_combo.currentText()
        cmap = self.colormap_combo.currentText()

        if lbl_layer is None or lbl_layer.data is None:
            print('[w-seg] Error: Please select a valid Labels layer.')
            return

        df_metrics = getattr(self, 'cached_df', None)
        if df_metrics is None or df_metrics.empty:
            print(
                '[w-seg] Error: No cached measurement data found. Run measurements first.'
            )
            return

        if feat not in df_metrics.columns:
            print(
                f"[w-seg] Error: Feature '{feat}' not found in measurement results."
            )
            return

        # Extract spatial transform metadata to align colormapped overlay with original image
        layer_scale = getattr(lbl_layer, 'scale', (1.0, 1.0))
        layer_translate = getattr(lbl_layer, 'translate', (0.0, 0.0))

        # Colorize features directly on the Napari layer
        apply_feature_colormap(
            viewer=self.viewer,
            base_layer_name=lbl_layer.name,
            labels_layer=lbl_layer.data,
            df_features=df_metrics,
            feature_name=feat,
            cmap_name=cmap,
            scale=layer_scale,
            translate=layer_translate,
        )

    def _update_feature_dropdown(self, df: pd.DataFrame):
        """Populates the feature selection combobox with numerical columns from measurement results."""
        if df is None or df.empty:
            return

        # Block signals during combobox update to prevent premature triggers
        _blocker = QSignalBlocker(self.feature_combo)

        cur_feature = self.feature_combo.currentText()
        self.feature_combo.clear()

        # Filter out non-numeric and identifier metadata columns
        exclude_cols = {
            'label',
            'label_id',
            'centroid-0',
            'centroid-1',
            'bbox-0',
            'bbox-1',
            'bbox-2',
            'bbox-3',
        }
        numeric_df = df.select_dtypes(include=['number'])
        numeric_cols = [
            col for col in numeric_df.columns if col not in exclude_cols
        ]

        self.feature_combo.addItems(numeric_cols)

        # Restore previous feature selection if still available
        idx = self.feature_combo.findText(cur_feature)
        if idx != -1:
            self.feature_combo.setCurrentIndex(idx)

    def _on_draw_rays_clicked(self):
        """Visualizes sampling rays and profiles for the targeted cell ID in Napari."""
        lbl_layer = self.labels_combo.currentData()
        sig_layer = self.signal_combo.currentData()

        if lbl_layer is None or lbl_layer.data is None:
            print('[w-seg] Error: Select a valid Labels layer first.')
            return

        target_id = self.inspect_cell_id_spin.value()

        # Verify the target label ID actually exists in the layer
        if target_id not in lbl_layer.data:
            print(
                f'[w-seg] Warning: Label ID {target_id} not found in active Labels layer.'
            )
            return

        visualize_cell_rays(
            viewer=self.viewer,
            labels_layer=lbl_layer.data,
            cell_wall_signal=sig_layer.data if sig_layer is not None else None,
            target_label_id=target_id,
            max_search_radius=self.max_search_spin.value(),
            min_search_radius=self.min_search_spin.value(),
            radius_scale=self.radius_scale_spin.value(),
            expand_labs_dist=self.expand_labs_spin.value(),
            erode_labs=self.erode_labs_spin.value(),
            sample_step=self.sample_step_spin.value(),
            max_rays=self.max_rays_spin.value(),
            offset=self.offset_spin.value(),
            invert_signal=self.invert_signal_cb.isChecked(),
            show_processed_signal=self.show_processed_signal_cb.isChecked(),
            scale=lbl_layer.scale,
            translate=lbl_layer.translate,
        )

    def _on_draw_neighbours_clicked(self):
        """Visualizes spatial neighbor connections and Delaunay fallback vectors for the targeted cell ID."""
        lbl_layer = self.labels_combo.currentData()

        if lbl_layer is None or lbl_layer.data is None:
            print('[w-seg] Error: Select a valid Labels layer first.')
            return

        target_id = self.inspect_cell_id_spin.value()

        if target_id not in lbl_layer.data:
            print(
                f'[w-seg] Warning: Label ID {target_id} not found in active Labels layer.'
            )
            return

        # Check if we have cached dataframe measurements to extract true cell-wall contact neighbors
        df_metrics = getattr(self, 'df_single_cell_metrics', None)

        visualize_cell_neighborhood(
            viewer=self.viewer,
            cell_id=target_id,
            labels_layer=lbl_layer.data,
            df_single_cell_metrics=df_metrics,
            search_radius=float(self.max_search_spin.value()),
            use_delaunay_fallback=True,
            max_delaunay_dist_px=120.0,
            scale=lbl_layer.scale,
            translate=lbl_layer.translate,
        )

    def _init_center_point_signals(self):
        """Connect center point UI signals. Call this in _connect_signals()."""
        # Connect checkbox toggle
        self.radial_symmetry_cb.toggled.connect(
            self._on_radial_symmetry_toggled
        )

        # Connect button click
        self.set_center_point_btn.clicked.connect(
            self._enable_center_point_picking
        )

        # Initial UI statesync
        self._on_radial_symmetry_toggled(self.radial_symmetry_cb.isChecked())

    def _on_radial_symmetry_toggled(self, checked: bool):
        """Enable/disable the center point button and layer visibility based on checkbox."""
        self.set_center_point_btn.setEnabled(checked)

        if self.center_layer_name in self.viewer.layers:
            self.viewer.layers[self.center_layer_name].visible = checked

    def _ensure_center_point_layer(self):
        """Create or retrieve the dedicated Points layer for center point selection."""
        if self.viewer is None:
            return None

        if self.center_layer_name in self.viewer.layers:
            layer = self.viewer.layers[self.center_layer_name]
            layer.visible = True
            return layer

        # Determine ND dimensions from existing layers
        ndim = self.viewer.dims.ndim if hasattr(self.viewer, 'dims') else 2

        layer = self.viewer.add_points(
            data=np.empty((0, ndim)),
            name=self.center_layer_name,
            size=10,
            face_color='green',
            symbol='cross',
            opacity=0.8,
        )

        # Connect layer data event listener for external edits/deletions
        layer.events.data.connect(self._on_point_data_changed)
        return layer

    def _enable_center_point_picking(self):
        """Set active layer to Center Point layer and activate click listener."""
        layer = self._ensure_center_point_layer()
        if layer is None:
            return

        self.viewer.layers.selection.active = layer

        # Change button visual style to reflect active picking state
        self.set_center_point_btn.setStyleSheet(
            'background-color: #2e6f40; color: white; font-weight: bold; padding: 6px; border-radius: 4px;'
        )

        # Ensure we don't attach duplicate callbacks
        if self._on_center_point_clicked not in layer.mouse_drag_callbacks:
            layer.mouse_drag_callbacks.append(self._on_center_point_clicked)

        # Switch mode to add points
        layer.mode = 'add'

    def _on_center_point_clicked(self, layer, event):
        """Single-point picker that captures position on click release."""
        # Only handle left clicks
        if event.button != 1:
            return

        yield  # Wait until mouse release completes

        # Extract cursor coordinates in layer data spatial coordinates
        pos = event.position  # Yields tuple matching layer spatial dimensions

        # Format coordinates array matching layer ndim
        coords = np.array([pos])

        # Overwrite layer data with single new point using event blocker to avoid loop recursion
        with layer.events.data.blocker():
            layer.data = coords

        # Store 2D (y, x) or 3D (z, y, x) coordinates internally
        self.center_coords = tuple(pos)
        print(f'[w-seg] Center point set at coordinates: {self.center_coords}')

        # Self-cleanup: remove callback and reset interactive states
        if self._on_center_point_clicked in layer.mouse_drag_callbacks:
            layer.mouse_drag_callbacks.remove(self._on_center_point_clicked)

        layer.mode = 'pan_zoom'

        # Reset button visual state
        self.set_center_point_btn.setStyleSheet(
            'font-weight: bold; background-color: #2b5b84; color: white; padding: 6px; border-radius: 4px;'
        )

    def _on_point_data_changed(self, event):
        """Syncs internal center_coords state if user edits or clears points manually."""
        layer = event.source
        if len(layer.data) == 0:
            self.center_coords = None
        else:
            # Enforce maximum 1 point limit on manual point addition
            if len(layer.data) > 1:
                with layer.events.data.blocker():
                    layer.data = layer.data[-1:]
            self.center_coords = tuple(layer.data[0])

    def refresh_layer_choices(self, event=None):
        """Populates combo boxes with current napari layers based on type

        and restores previous selections cleanly without triggering signals.
        """
        if self.viewer is None:
            return

        # Block signals during combobox updates to avoid cascading callbacks
        _b1 = QSignalBlocker(self.labels_combo)

        # Cache previous layer names rather than raw layer instances
        cur_label_name = self.labels_combo.currentText()
        self.labels_combo.clear()
        for lyr in self.viewer.layers:
            if isinstance(lyr, napari.layers.Labels):
                self.labels_combo.addItem(lyr.name, lyr)

        # Restore previous selection by layer name if still present
        idx_label = self.labels_combo.findText(cur_label_name)
        if idx_label != -1:
            self.labels_combo.setCurrentIndex(idx_label)

        self._refresh_image_choices()

        # Update inspection bounds based on active label layer without locking UI
        self._update_inspection_bounds()

    def _refresh_image_choices(self, *_):
        """
        Lists the image layers that can be paired with the selected labels layer:
        only images with the same size (rows x columns). After an ROI selection
        the cropped layers have the new size, so the uncropped originals drop
        out of the lists.
        """
        if self.viewer is None:
            return
        lbl_layer = self.labels_combo.currentData()
        lbl_shape = (
            spatial_shape(lbl_layer.data) if lbl_layer is not None else None
        )

        _b2 = QSignalBlocker(self.signal_combo)
        _b3 = QSignalBlocker(self.YOLO_layer_combo)
        cur_signal_name = self.signal_combo.currentText()
        cur_yolo_name = self.YOLO_layer_combo.currentText()
        self.signal_combo.clear()
        self.YOLO_layer_combo.clear()

        n_hidden = 0
        for lyr in self.viewer.layers:
            if not isinstance(lyr, napari.layers.Image):
                continue
            if lbl_shape is not None and spatial_shape(lyr.data) != lbl_shape:
                n_hidden += 1
                continue
            self.signal_combo.addItem(lyr.name, lyr)
            self.YOLO_layer_combo.addItem(lyr.name, lyr)

        idx_signal = self.signal_combo.findText(cur_signal_name)
        if idx_signal != -1:
            self.signal_combo.setCurrentIndex(idx_signal)
        idx_yolo = self.YOLO_layer_combo.findText(cur_yolo_name)
        if idx_yolo != -1:
            self.YOLO_layer_combo.setCurrentIndex(idx_yolo)

        tip = (
            f'{n_hidden} image layer(s) with a different size than the labels are not listed.'
            if n_hidden
            else ''
        )
        self.signal_combo.setToolTip(tip)
        self.YOLO_layer_combo.setToolTip(tip)

    def refresh_paired_model(self):
        """
        Reads the model loaded in tab 1 and shows the YOLO classifier that is
        paired with it. When the model changes, the radial-symmetry option is
        preset from the model; later manual changes are kept.
        """
        info = get_active_model_info()
        self._paired_yolo = info['yolo']
        if self._paired_yolo:
            self.YOLO_paired_label.setText(Path(self._paired_yolo).name)
        else:
            self.YOLO_paired_label.setText(
                'none (this model has no paired YOLO classifier)'
            )
        self.use_YOLO_cb.setEnabled(self._paired_yolo is not None)

        if info['name'] != self._last_model_name:
            self._last_model_name = info['name']
            self.use_YOLO_cb.setChecked(self._paired_yolo is not None)
            if info['radial_symmetry'] is not None:
                self.radial_symmetry_cb.setChecked(
                    bool(info['radial_symmetry'])
                )
        elif self._paired_yolo is None:
            self.use_YOLO_cb.setChecked(False)

    def _update_inspection_bounds(self):
        """Updates min/max bounds for inspect_cell_id_spin based on selected Labels layer."""
        selected_label_layer = self.labels_combo.currentData()

        if (
            selected_label_layer is not None
            and selected_label_layer.data is not None
        ):
            lbls = selected_label_layer.data
            # Fast scalar check using max label ID rather than scanning all unique elements
            max_id = int(lbls.max()) if lbls.size > 0 else 1
            if max_id > 0:
                self.inspect_cell_id_spin.setRange(1, max_id)
            else:
                self.inspect_cell_id_spin.setRange(1, 1)
        else:
            self.inspect_cell_id_spin.setRange(1, 999999)

    def snapshot_params(self) -> dict:
        """Everything measure_cell_morphology needs from the UI (except the
        layer data), for the batch tab. Take it on the main thread."""
        self.refresh_paired_model()
        yolo = self._paired_yolo if self.use_YOLO_cb.isChecked() else None
        return {
            **self.get_feature_extraction_config(),
            'num_workers': self.num_workers_spin.value(),
            'invert_signal': self.invert_signal_cb.isChecked(),
            'clear_borders': self.clear_border_cb.isChecked(),
            'center_coords': self.center_coords,
            'use_YOLO': yolo is not None,
            'YOLO_weights': yolo or '',
        }

    def get_feature_extraction_config(self) -> dict:
        """Extracts locked feature extraction parameters from UI widgets."""
        fe_config = {
            'pixel_spacing': float(self.pixel_spacing_spin.value()),
            'max_search_radius': int(self.max_search_spin.value()),
            'min_search_radius': int(self.min_search_spin.value()),
            'radius_scale': float(self.radius_scale_spin.value()),
            'expand_labs_dist': int(self.expand_labs_spin.value()),
            'erode_labs': int(self.erode_labs_spin.value()),
            'sample_step': int(self.sample_step_spin.value()),
            'max_rays': int(self.max_rays_spin.value()),
            'offset': float(self.offset_spin.value()),
            'radial_sym': bool(self.radial_symmetry_cb.isChecked()),
        }
        # print(f"[DEBUG] Feature extraction config: {fe_config}")
        return fe_config

    def set_feature_extraction_config(self, config: dict):
        """Populates UI input controls with values from a config dictionary."""
        if not config or not isinstance(config, dict):
            print(
                '[w-seg] Cannot set config: Provided config is empty or invalid.'
            )
            return

        # Block signals during batch updates to prevent unneeded recalculations
        _blockers = [
            QSignalBlocker(self.pixel_spacing_spin),
            QSignalBlocker(self.max_search_spin),
            QSignalBlocker(self.min_search_spin),
            QSignalBlocker(self.radius_scale_spin),
            QSignalBlocker(self.expand_labs_spin),
            QSignalBlocker(self.erode_labs_spin),
            QSignalBlocker(self.sample_step_spin),
            QSignalBlocker(self.max_rays_spin),
            QSignalBlocker(self.offset_spin),
            QSignalBlocker(self.radial_symmetry_cb),
        ]

        if 'pixel_spacing' in config:
            self.pixel_spacing_spin.setValue(float(config['pixel_spacing']))
        if 'max_search_radius' in config:
            self.max_search_spin.setValue(int(config['max_search_radius']))
        if 'min_search_radius' in config:
            self.min_search_spin.setValue(int(config['min_search_radius']))
        if 'radius_scale' in config:
            self.radius_scale_spin.setValue(float(config['radius_scale']))
        if 'expand_labs_dist' in config:
            self.expand_labs_spin.setValue(int(config['expand_labs_dist']))
        if 'erode_labs' in config:
            self.erode_labs_spin.setValue(int(config['erode_labs']))
        if 'sample_step' in config:
            self.sample_step_spin.setValue(int(config['sample_step']))
        if 'max_rays' in config:
            self.max_rays_spin.setValue(int(config['max_rays']))
        if 'offset' in config:
            self.offset_spin.setValue(float(config['offset']))
        if 'radial_sym' in config:
            enabled = bool(config['radial_sym'])
            self.radial_symmetry_cb.setChecked(enabled)
            # Sync dependent center point button state explicitly
            self._on_radial_symmetry_toggled(enabled)

        print(f'[w-seg] Successfully updated UI widgets with config: {config}')

    def load_session_metadata(self, hdf5_path: str):
        """Loads saved feature extraction settings from an HDF5 session file back into UI."""
        import json

        import h5py

        with h5py.File(hdf5_path, 'r') as hf:
            if 'config' in hf.attrs:
                config = json.loads(hf.attrs['config'])
                self.set_feature_extraction_config(config)

    # Backward-compatible property wrappers for other tabs/widgets
    @property
    def selected_labels_layer(self):
        """Returns currently selected Labels layer instance."""
        return self.labels_combo.currentData()

    @property
    def selected_signal_layer(self):
        """Returns currently selected Image/Signal layer instance."""
        return self.signal_combo.currentData()

    @property
    def is_radial_symmetry_enabled(self) -> bool:
        """Returns state of radial symmetry checkbox."""
        return self.radial_symmetry_cb.isChecked()
