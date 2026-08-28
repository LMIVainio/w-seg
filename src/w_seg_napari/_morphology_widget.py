import concurrent.futures
import sys
from concurrent.futures import as_completed

# from typing import Dict, Tuple
import napari
import numpy as np
import pandas as pd
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FloatSpinBox,
    PushButton,
    SpinBox,
)
from scipy.ndimage import (
    binary_erosion,
    find_objects,
    gaussian_filter1d,
    label,
    map_coordinates,
)
from skimage.color import rgb2gray
from skimage.measure import find_contours, regionprops_table
from skimage.segmentation import (
    clear_border,
    expand_labels,
    relabel_sequential,
)

from w_seg_napari._wsegutils import (
    apply_feature_colormap,
    extract_spatial_feature_matrix,
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
    search_radius: int = 12,
    sample_step: int = 3,
    max_rays: int = np.inf,
    offset: float = 0.5,
    erode_labs: int = 4,
    pixel_spacing: float = 1.0,
    verbose: bool = False,
) -> dict:
    """
    Single Source of Truth: Measures wall thickness and geometry vectors using
    preprocessed signal, and samples real color values from raw_signal_image.
    """
    H, W = labels_layer.shape[:2]

    # Use provided padded bbox or fall back to local calculation
    if bbox is None:
        y_indices, x_indices = np.where(labels_layer == cell_id)
        if len(y_indices) == 0:
            return None

        pad = search_radius + (2 * search_radius) + erode_labs
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

    # --- Contour Extraction (FIX: Run on LOCAL sub-array) ---
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

    t = np.linspace(-search_radius, search_radius, search_radius * 4)
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


def visualize_cell_rays(
    viewer: 'napari.Viewer',
    labels_layer: np.ndarray,
    cell_wall_signal: np.ndarray | None = None,
    target_label_id: int = 1,
    search_radius: int = 12,
    expand_labs_dist: int = 4,
    erode_labs: int = 4,
    sample_step: int = 3,
    max_rays: int = np.inf,
    offset: float = 0.5,
    invert_signal: bool = False,
    show_processed_signal: bool = True,
) -> None:
    """Renders rays, detected wall segments, and anchors for a single targeted cell label."""
    if labels_layer is None or target_label_id is None:
        return

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
    bbox_padding = search_radius + (4 * expand_labs_dist) + erode_labs
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
            viewer.layers[sig_layer_name].refresh()
        else:
            viewer.add_image(
                cell_wall_signal,
                name=sig_layer_name,
                colormap='gray',
                contrast_limits=(0.0, 1.0),
                opacity=0.7,
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
        search_radius=search_radius,
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
        )

    if result['thresholded_wall_lines']:
        viewer.add_shapes(
            np.array(result['thresholded_wall_lines']),
            shape_type='path',
            edge_color='lime',
            edge_width=1.5,
            name=f'rays_wall_{target_label_id}',
            opacity=0.9,
        )

    if result['anchor_points']:
        viewer.add_points(
            np.array(result['anchor_points']),
            size=2,
            face_color='yellow',
            border_color='transparent',
            name=f'anchors_{target_label_id}',
            opacity=0.8,
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


def measure_cell_morphology(
    labels_layer: np.ndarray,
    cell_wall_signal: np.ndarray,
    pixel_spacing: float = 1.0,
    search_radius: int = 12,
    sample_step: int = 3,
    max_rays: int = np.inf,
    offset: float = 0.5,
    erode_labs: int = 4,
    expand_labs_dist: int = 4,
    num_workers: int = 8,
    clear_borders: bool = True,
    invert_signal: bool = False,
    center_coords: tuple = None,
    radial_symmetry: bool = False,
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
    labels_layer = fix_labels(labels_layer)
    unique_ids = np.unique(labels_layer)
    unique_ids = unique_ids[unique_ids != 0]

    if len(unique_ids) == 0:
        print('[w-seg] Warning: No labels found after border clearing.')
        return pd.DataFrame(), labels_layer

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
    bbox_padding = (2 * expand_labs_dist) + search_radius
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
                search_radius=search_radius,
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

    # Check if center point is defined
    if center_coords is None and radial_symmetry:
        print('[w-seg] calculating center point')
        center_coords = resolve_center_coords(df_combined)

    # Calculate distance from center point
    df_combined = add_radial_features(
        df_combined,
        center_coords=center_coords,
        enabled=radial_symmetry,
        pixel_scale=pixel_spacing,
        angle_unit='degrees',
    )

    # Finally build and return spatial features for each cell
    print('[w-seg] Extracting spatial features for all cells...')
    df_final = extract_spatial_feature_matrix(
        labels_layer, df_combined, search_radius=search_radius
    )

    return df_final, labels_layer


# =========================================================================
# Get morphology statistics class
# =========================================================================
# w_seg_napari/widgets/morphology_widget.py


class MorphologyWidget(Container):
    """
    Standalone Container class managing morphology calculation, feature colormap visualization,
    and ray tracing debugging in napari.
    """

    def __init__(self, viewer: 'napari.Viewer'):
        super().__init__(labels=True)
        self.viewer = viewer

        # Internal state caches
        self.cached_df = None
        self.cached_labels_data = None
        self.cached_feature_extraction_config = None

        # Center point tracking
        self.center_layer_name = 'Center point'
        self.center_coords = None  # (y, x) coordinates of the center point

        # ---------------------------------------------------------------------
        # 1. Instantiate UI Controls
        # ---------------------------------------------------------------------
        self.labels_layer = ComboBox(
            label='Labels Layer:', annotation='napari.layers.Labels'
        )
        self.signal_layer = ComboBox(
            label='Signal Image / Pred:',
            annotation='napari.layers.Image',
            nullable=True,
        )

        self.pixel_spacing = FloatSpinBox(
            label='Pixel Scale (µm/px):',
            value=1.0,
            min=0.001,
            max=100.0,
            step=0.01,
        )
        self.search_radius = SpinBox(
            label='Ray Search Radius (px):', value=12, min=2, max=100
        )
        self.expand_labs_dist = SpinBox(
            label='Label Expansion (px):', value=4, min=0, max=20
        )
        self.erode_labs = SpinBox(
            label='Lumen Erosion (px):', value=4, min=1, max=20
        )
        self.sample_step = SpinBox(
            label='Ray Sampling Step:', value=3, min=1, max=10
        )
        self.max_rays = SpinBox(
            label='Max ray samples:', value=80, min=5, max=1000
        )
        self.offset = FloatSpinBox(
            label='Threshold Offset (σ):',
            value=0.5,
            min=-10.0,
            max=15.0,
            step=0.05,
        )
        self.num_workers = SpinBox(
            label='Parallel Threads:', value=8, min=1, max=32
        )
        # Checkboxes for boolean options
        self.clear_borders = CheckBox(label='Clear Border Cells:', value=False)
        self.invert_signal = CheckBox(
            label='Invert Signal Polarity (1 - S):', value=True
        )
        self.show_processed_signal = CheckBox(
            label='Show Processed Signal Image:', value=False
        )
        self.radial_symmetry = CheckBox(
            label='Use Radial Symmetry and center point:', value=True
        )
        self.set_center_point_btn = PushButton(text='Set Center Point')
        self.run_btn = PushButton(text='Measure Cell Walls & Morphology')

        # Visualization controls
        self.feature_to_plot = ComboBox(
            label='Feature to Visualize:', choices=['mean_wall_thickness']
        )
        self.colormap = ComboBox(
            label='Colormap:',
            choices=[
                'viridis',
                'magma',
                'plasma',
                'inferno',
                'coolwarm',
                'turbo',
            ],
        )
        self.vis_btn = PushButton(text='Visualize Feature Map')

        # Ray inspection controls
        self.inspect_cell_id = SpinBox(
            label='Label ID to Inspect:', value=1, min=1, max=99999
        )
        self.draw_rays_btn = PushButton(text='Draw Sampling Rays')

        # Connect center point button to callback and layer visibility toggle
        self.radial_symmetry.changed.connect(self._on_radial_symmetry_toggled)
        # Connect push button to activate interactive point selection in Napari
        self.set_center_point_btn.changed.connect(
            self._enable_center_point_picking
        )
        # Initialize state sync
        self._on_radial_symmetry_toggled(self.radial_symmetry.value)

        # ---------------------------------------------------------------------
        # 2. Add Controls to Widget Container
        # ---------------------------------------------------------------------
        self.extend(
            [
                self.labels_layer,
                self.signal_layer,
                self.pixel_spacing,
                self.search_radius,
                self.expand_labs_dist,
                self.erode_labs,
                self.sample_step,
                self.max_rays,
                self.offset,
                self.num_workers,
                self.clear_borders,
                self.invert_signal,
                self.show_processed_signal,
                self.radial_symmetry,
                self.set_center_point_btn,
                self.run_btn,
                self.feature_to_plot,
                self.colormap,
                self.vis_btn,
                self.inspect_cell_id,
                self.draw_rays_btn,
            ]
        )

        # ---------------------------------------------------------------------
        # 3. Connect Event Handlers
        # ---------------------------------------------------------------------
        self.run_btn.changed.connect(self._on_measure_clicked)
        self.vis_btn.changed.connect(self._on_visualize_clicked)
        self.draw_rays_btn.changed.connect(self._on_draw_rays_clicked)

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
            self.labels_layer.changed.connect(self.refresh_layer_choices)
            self.refresh_layer_choices()

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    def _on_measure_clicked(self):
        lbl_layer = self.labels_layer.value
        sig_layer = self.signal_layer.value

        if lbl_layer is None:
            print('[w-seg] Error: Please select a valid Labels layer.')
            return

        lbl_data = lbl_layer.data
        sig_data = sig_layer.data if sig_layer is not None else None

        # Execute standalone backend engine function
        df_results, measured_labels_layer = measure_cell_morphology(
            labels_layer=lbl_data,
            cell_wall_signal=sig_data,
            pixel_spacing=self.pixel_spacing.value,
            search_radius=self.search_radius.value,
            sample_step=self.sample_step.value,
            max_rays=self.max_rays.value,
            offset=self.offset.value,
            erode_labs=self.erode_labs.value,
            expand_labs_dist=self.expand_labs_dist.value,
            num_workers=self.num_workers.value,
            invert_signal=self.invert_signal.value,
            center_coords=self.center_coords,
            radial_symmetry=self.radial_symmetry,
            clear_borders=self.clear_borders.value,
        )

        # Cache data directly on instance
        self.cached_df = df_results
        self.cached_labels_data = measured_labels_layer
        self.cached_feature_extraction_config = (
            self.get_feature_extraction_config()
        )

        # Pass data to classifier widget
        if (
            hasattr(self, 'classification_widget')
            and self.classification_widget is not None
        ):
            self.classification_widget.bind_measured_data(
                df=self.cached_df,
                labels_data=self.cached_labels_data,
                layer_name=lbl_layer.name,
                feature_extraction_config=self.cached_feature_extraction_config,  # <-- CRITICAL PASS
            )

        # Update available numeric features dropdown
        numeric_cols = [
            c
            for c in df_results.select_dtypes(include=[np.number]).columns
            if c not in ['label_id', 'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']
        ]
        self.feature_to_plot.choices = numeric_cols
        if 'mean_wall_thickness' in numeric_cols:
            self.feature_to_plot.value = 'mean_wall_thickness'

        print('[w-seg] Done!')

    def _on_visualize_clicked(self):
        lbl_layer = self.labels_layer.value
        feat = self.feature_to_plot.value
        cmap = self.colormap.value

        if (
            lbl_layer is None
            or self.cached_df is None
            or self.cached_labels_data is None
        ):
            print('[w-seg] Please select a layer and run measurements first.')
            return

        if feat not in self.cached_df.columns:
            print(f"[w-seg] Feature '{feat}' not found in calculated data.")
            return

        apply_feature_colormap(
            viewer=self.viewer,
            base_layer_name=lbl_layer.name,
            labels_layer=self.cached_labels_data,
            df_features=self.cached_df,
            feature_name=feat,
            cmap_name=cmap,
        )

    def _on_draw_rays_clicked(self):
        lbl_layer = self.labels_layer.value
        sig_layer = self.signal_layer.value

        if lbl_layer is None:
            print('[w-seg] Select a Labels layer first.')
            return

        visualize_cell_rays(
            viewer=self.viewer,
            labels_layer=lbl_layer.data,
            cell_wall_signal=sig_layer.data if sig_layer else None,
            target_label_id=int(self.inspect_cell_id.value),
            search_radius=self.search_radius.value,
            expand_labs_dist=self.expand_labs_dist.value,
            erode_labs=self.erode_labs.value,
            sample_step=self.sample_step.value,
            max_rays=self.max_rays.value,
            offset=self.offset.value,
            invert_signal=self.invert_signal.value,
            show_processed_signal=self.show_processed_signal.value,
        )

    def _on_radial_symmetry_toggled(self, value=None):
        """Enable/disable the center point button based on checkbox"""
        enabled = (
            bool(value) if value is not None else self.radial_symmetry.value
        )
        self.set_center_point_btn.enabled = enabled
        if self.center_layer_name in self.viewer.layers:
            self.viewer.layers[self.center_layer_name].visible = enabled

    def _ensure_center_point_layer(self):
        """Create aor get the dedicate Points layer for center point selection."""
        if self.viewer is None:
            return

        if self.center_layer_name in self.viewer.layers:
            layer = self.viewer.layers[self.center_layer_name]
            layer.visible = True
            return layer

        return self.viewer.add_points(
            data=np.empty((0, 2)),
            name=self.center_layer_name,
            size=7,
            face_color='green',
            symbol='cross',
            opacity=0.8,
        )

    def _enable_center_point_picking(self, *args):
        """Set active layer to Center Point layer and attach click listener."""
        layer = self._ensure_center_point_layer()
        self.viewer.layers.selection.active = layer

        # Ensure we don't attach duplicate callbacks
        if self._on_center_point_clicked not in layer.mouse_drag_callbacks:
            layer.mouse_drag_callbacks.append(self._on_center_point_clicked)

        # Set layer mode to select so standard point adding doesn't fight our callback
        layer.mode = 'select'

    def _on_point_added(self, event):
        """Enforce single point behavior without triggering infinite recursion."""
        if self.center_layer_name not in self.viewer.layers:
            return

        layer = self.viewer.layers[self.center_layer_name]

        if len(layer.data) > 1:
            # Block events while updating layer.data to prevent infinite recursion
            with layer.events.data.blocker():
                layer.data = layer.data[-1:]

        if len(layer.data) > 0:
            self.center_coords = layer.data[0]

    def _on_center_point_clicked(self, layer, event):
        """Single-point picker that bypasses event-loop recursion."""
        # Only handle left clicks
        if event.button != 1:
            return

        yield  # Wait until mouse release / click completes

        # Get spatial cursor position (y, x) from world coordinates
        cursor_pos = self.viewer.cursor.position
        if len(cursor_pos) == 3:
            coords = np.array([[cursor_pos[1], cursor_pos[2]]])
        else:
            coords = np.array([[cursor_pos[0], cursor_pos[1]]])

        # Directly overwrite layer data with the single new point
        # Modifying layer.data here will NOT trigger mouse_drag_callbacks!
        layer.data = coords
        self.center_coords = coords[0]
        print(f'Center point set at (y, x): {self.center_coords}')

    def refresh_layer_choices(self, event=None):
        if self.viewer is None:
            return

        # Synchronize Napari layers with ComboBoxes
        label_layers = [
            lay
            for lay in self.viewer.layers
            if isinstance(lay, napari.layers.Labels)
        ]
        image_layers = [
            lay
            for lay in self.viewer.layers
            if isinstance(lay, napari.layers.Image)
        ]

        self.labels_layer.choices = label_layers
        self.signal_layer.choices = image_layers

        # Update available label IDs for inspect_cell_id
        if self.labels_layer.value is not None:
            lbls = self.labels_layer.value.data
            unique_ids = sorted([int(x) for x in np.unique(lbls) if x != 0])
            if unique_ids:
                self.inspect_cell_id.choices = unique_ids

    def get_feature_extraction_config(self) -> dict:
        """Extracts locked feature extraction parameters from UI widgets."""
        fe_config = {
            'pixel_spacing': float(self.pixel_spacing.value),
            'search_radius': int(self.search_radius.value),
            'expand_labs_dist': int(self.expand_labs_dist.value),
            'erode_labs': int(self.erode_labs.value),
            'sample_step': int(self.sample_step.value),
            'max_rays': int(self.max_rays.value),
            'offset': float(self.offset.value),
            'radial_sym': bool(self.radial_symmetry),
        }
        # print(f"DEBUG] Feature extraction config: {fe_config}")
        return fe_config

    def set_feature_extraction_config(self, config: dict):
        """Populates UI input controls with values from a config dictionary."""
        if not config or not isinstance(config, dict):
            print(
                '[DEBUG] Cannot set config: Provided config is empty or invalid.'
            )
            return

        # Directly assign to widget .value attributes
        if 'pixel_spacing' in config:
            self.pixel_spacing.value = float(config['pixel_spacing'])
        if 'search_radius' in config:
            self.search_radius.value = int(config['search_radius'])
        if 'expand_labs_dist' in config:
            self.expand_labs_dist.value = int(config['expand_labs_dist'])
        if 'erode_labs' in config:
            self.erode_labs.value = int(config['erode_labs'])
        if 'sample_step' in config:
            self.sample_step.value = int(config['sample_step'])
        if 'max_rays' in config:
            self.max_rays.value = int(config['max_rays'])
        if 'offset' in config:
            self.offset.value = float(config['offset'])
        if 'radial_sym' in config:
            self.radial_symmetry = bool(config['radial_sym'])

        print(f'[DEBUG] Updated UI widgets with config: {config}')
