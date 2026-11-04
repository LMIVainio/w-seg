"""
File handling for the batch tab (no napari/torch needed, so it is easy to test).
"""

from pathlib import Path

import numpy as np

VALID_EXTS = ('.tif', '.tiff', '.h5', '.hdf5', '.png', '.jpg', '.jpeg')
OUTPUT_DIRNAME = 'wseg_batch'


# Seed choices that do not need a layer: they stand for "the same map of
# every image", so the batch can be set up before any image is predicted.
# kind -> (label in the combo box, name prefix of the layer in the viewer)
VIRTUAL_SEEDS = {
    'skeleton': ('Predicted skeleton (geo_skel) - every image', 'geo_skel_'),
    'edge_distance': (
        'Predicted edge distance (geo_dist) - every image',
        'geo_dist_',
    ),
    'prediction': ('Wall prediction (pred) - every image', 'pred_'),
}


def virtual_seed_layer_name(kind, source):
    """Name of the viewer layer that a virtual seed choice means for one image."""
    return f'{VIRTUAL_SEEDS[kind][1]}{source}'


def list_input_files(dirpath) -> list[str]:
    """Names of the image files in a folder (sorted, results of earlier runs excluded)."""
    folder = Path(dirpath)
    return sorted(
        f.name
        for f in folder.iterdir()
        if f.is_file()
        and f.suffix.lower() in VALID_EXTS
        and not f.name.endswith('_session.h5')
    )


def find_session_dir(dirpath) -> Path:
    """Where the sessions are: the wseg_batch folder of an image folder, or the folder itself."""
    folder = Path(dirpath)
    sub = folder / OUTPUT_DIRNAME
    if sub.is_dir() and any(sub.glob('*_session.h5')):
        return sub
    return folder


def list_session_names(dirpath) -> list[str]:
    """
    Sessions of a folder as virtual image names ('a.h5' stands for
    a_session.h5), so the same name rules (session_path, table_path) apply.
    """
    suffix = '_session.h5'
    return sorted(
        f'{f.name[: -len(suffix)]}.h5'
        for f in find_session_dir(dirpath).glob(f'*{suffix}')
    )


def output_dir(dirpath) -> Path:
    """Results go into a sub folder, so the input folder stays clean."""
    return Path(dirpath) / OUTPUT_DIRNAME


def session_path(out_dir, image_name: str) -> Path:
    return Path(out_dir) / f'{Path(image_name).stem}_session.h5'


def table_path(out_dir, image_name: str) -> Path:
    return Path(out_dir) / f'{Path(image_name).stem}_morphology.csv'


def to_rgb(arr: np.ndarray) -> np.ndarray:
    """Any 2D or 3D image array -> (H, W, 3), keeping the dtype.

    Accepts grayscale (H, W), channels last (H, W, C) and channels first
    (C, H, W) with C in 1, 3, 4; an alpha channel is dropped.
    """
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f'Unsupported image shape {arr.shape}')
    small = (1, 3, 4)
    if arr.shape[-1] not in small and arr.shape[0] in small:
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] not in small:
        raise ValueError(
            f'Cannot tell the colour channels of an image with shape {arr.shape}'
        )
    if arr.shape[-1] == 1:
        return np.repeat(arr, 3, axis=-1)
    return arr[..., :3]


def load_image(path) -> np.ndarray:
    """Read an image file as (H, W, 3)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in ('.h5', '.hdf5'):
        import h5py

        with h5py.File(path, 'r') as f:
            key = 'raw' if 'raw' in f else next(iter(f.keys()))
            arr = f[key][:]
    elif ext in ('.tif', '.tiff'):
        import tifffile

        arr = tifffile.imread(path)
    else:
        from skimage.io import imread

        arr = imread(path)
    return to_rgb(arr)


def to_model_input_range(rgb: np.ndarray) -> np.ndarray:
    """float32 image; 8-bit data is scaled to 0..1 (the model input handles the rest)."""
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float32) / 255.0
    return rgb.astype(np.float32)
