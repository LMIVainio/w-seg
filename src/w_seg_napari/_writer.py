"""
Writer plugin for napari to export layers as 16-bit TIFF files using PIL.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = tuple[DataType, dict, str]


def _save_as_16bit_tiff(data: np.ndarray, path: str) -> str:
    """Helper function to convert numpy array to 16-bit TIFF and save via PIL."""
    # Convert input data to numpy array
    arr = np.asarray(data)

    # Handle float images or out-of-range integer values safely
    if np.issubdtype(arr.dtype, np.floating):
        # Scale float values (assuming range [0, 1] or arbitrary range) to uint16
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = ((arr - arr_min) / (arr_max - arr_min) * 65535).astype(
                np.uint16
            )
        else:
            arr = np.zeros_like(arr, dtype=np.uint16)
    elif arr.dtype != np.uint16:
        # Cast integer labels/data directly to uint16
        arr = np.clip(arr, 0, 65535).astype(np.uint16)

    # PIL expects 2D images. Handle 2D or 3D single-channel shapes
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode='I;16')
        img.save(path)
    elif arr.ndim == 3:
        # If it's a 3D volume/stack, save as a multi-page TIFF
        frames = [
            Image.fromarray(slice_data, mode='I;16') for slice_data in arr
        ]
        frames[0].save(path, save_all=True, append_images=frames[1:])
    else:
        raise ValueError(
            f'Unsupported array shape {arr.shape} for 16-bit TIFF export.'
        )

    return path


def write_single_image(path: str, data: Any, meta: dict) -> list[str]:
    """Writes a single image or labels layer as a 16-bit TIFF file."""
    p = Path(path)
    # Ensure file has a .tif or .tiff extension
    if p.suffix.lower() not in {'.tif', '.tiff'}:
        p = p.with_suffix('.tif')

    saved_path = _save_as_16bit_tiff(data, str(p))
    return [str(saved_path)]


def write_multiple(path: str, data: list[FullLayerData]) -> list[str]:
    """Writes multiple layers as separate 16-bit TIFF files in a target directory or prefix."""
    written_paths = []

    p = Path(path)
    # If path is a directory, save each layer inside it
    is_dir = p.is_dir() or not p.suffix
    if is_dir:
        p.mkdir(parents=True, exist_ok=True)

    for i, (layer_data, meta, _layer_type) in enumerate(data):
        # Generate filename based on layer name or index
        layer_name = meta.get('name', f'layer_{i}')

        if is_dir:
            file_path = p / f'{layer_name}_16bit.tif'
        else:
            # If original path had an extension, insert layer name before it
            if p.suffix:
                file_path = p.parent / f'{p.stem}_{layer_name}{p.suffix}'
            else:
                file_path = p.parent / f'{p.name}_{layer_name}.tif'

        saved_path = _save_as_16bit_tiff(layer_data, str(file_path))
        written_paths.append(str(saved_path))

    return written_paths
