"""
Data pipeline for the W-net.

Design:
  1. Each slide is preprocessed ONCE as a whole (global percentile stretch,
     denoise, LAB + HSV-derived channels) and cached on disk as a [H, W, 6]
     float16 .npy. Training and inference share exactly this code.
  2. The Dataset then only crops from the cached array (memory-mapped),
     resizes, flips/rotates, applies mild photometric augmentation to the
     IMAGE ONLY, and builds targets from the annotation planes.
  3. Affinity targets follow an arbitrary list of (dy, dx) offsets (see
     wseg_config.py), so ranges and directions are configuration, not code.

Photometric augmentation is applied AFTER the global stretch on purpose:
a brightness/contrast change applied before a global stretch would largely be
undone by it. Here it simulates the residual variation that remains.

Returned sample (all channels-last), with n = len(offsets):
    x : [T, T, 6]      float32, about [-1, 1]
    y : [T, T, 1 + n (+ 4)]  channel 0 = wall mask, channels 1..n = affinities,
        and with a geometry head 4 more: edge distance, dir_y, dir_x, skeleton
    w : [T, T, 1]      EDT wall weight map

Source TIFF layout: planar stack of any numeric dtype; planes 0-2 = RGB with
values 0..255 or 0..1 (detected), 3 = semantic (wall) mask, 4 = instance labels. For inference,
any planar stack whose first three planes are RGB (0..255) works with
ensure_slide_cache().
"""

import glob
import math
import os

import kornia.color as K_color
import numpy as np
import tifffile
import torch
from skimage.transform import resize
from torch.utils.data import Dataset

from w_seg_napari._model_config import DEFAULT_RANGES, make_offsets
from w_seg_napari._training.targets import (
    compute_affinities,
    compute_geometry,
    edt_weights,
)

COARSE = 8  # block size of the coarse tissue map
N_CHANNELS = 6


# =====================================================================
# Whole-slide preprocessing
# =====================================================================
def open_stack(path):
    """The (planes, H, W) stack of a tif: memory-mapped when the file allows it.

    Compressed tifs cannot be memory-mapped; those are read into memory instead
    (fine for training slides of a few hundred MB).
    """
    try:
        return tifffile.memmap(path)
    except ValueError:
        return tifffile.imread(path)


def detect_rgb_scale(rgb_planes, step=50):
    """
    255.0 if the RGB planes hold values 0..255 (uint8 or float32 alike),
    1.0 if they hold 0..1. Anything above 255 is an error.
    """
    top = float(
        np.max(np.asarray(rgb_planes[:, ::step, ::step], dtype=np.float32))
    )
    if top > 255.0 + 1e-3:
        raise ValueError(
            f'RGB planes must hold values in 0..255 or 0..1 (found a maximum of {top:.1f})'
        )
    return 255.0 if top > 1.0 + 1e-3 else 1.0


def compute_stretch_range(rgb_planes, step=50, rgb_scale=255.0):
    """
    1st / 99th percentile per colour channel for a WHOLE slide, from a strided
    subsample. rgb_planes: [3, H, W], any dtype (a memmap is fine), values
    0..rgb_scale. Returns two float32 arrays of shape [3], in 0..1.
    """
    sample = (
        np.asarray(rgb_planes[:, ::step, ::step], dtype=np.float32) / rgb_scale
    )
    flat = sample.reshape(3, -1)
    q_low = np.percentile(flat, 1, axis=1).astype(np.float32)
    q_high = np.percentile(flat, 99, axis=1).astype(np.float32)
    return q_low, q_high


def preprocess_tile(
    rgb_hwc,
    q_low,
    q_high,
    denoise_fn=None,
    enable_lab=True,
    hue_encoding='chroma',
    rgb_scale=255.0,
):
    """
    [h, w, 3] with values 0..rgb_scale (any dtype) -> [h, w, 6] float32, about [-1, 1].

    Channels 0-2: LAB (or RGB if enable_lab=False), centred to [-1, 1].
    Channels 3-5: hue_encoding='chroma' -> (s*cos(h), s*sin(h), v)
                  hue_encoding='raw'    -> (h, s, v) centred to [-1, 1]

    'chroma' avoids two problems of raw hue: the 0/2pi wrap-around (which
    breaks bilinear resizing) and noisy hue on unsaturated pixels.

    denoise_fn: takes and returns a [1, 3, h, w] float tensor in 0..1
    (e.g. your safe_bilateral_blur), or None to skip denoising.
    """
    t = torch.from_numpy(np.ascontiguousarray(rgb_hwc, dtype=np.float32))
    t = t.permute(2, 0, 1).unsqueeze(0) / rgb_scale

    lo = torch.from_numpy(q_low).view(1, 3, 1, 1)
    hi = torch.from_numpy(q_high).view(1, 3, 1, 1)
    t = ((t - lo) / torch.clamp(hi - lo, min=1e-6)).clamp(0.0, 1.0)

    if denoise_fn is not None:
        t = denoise_fn(t)

    hsv = K_color.rgb_to_hsv(t)  # kornia: hue in radians, [0, 2pi]
    if enable_lab:
        lab = K_color.rgb_to_lab(t)
        first = torch.cat(
            [
                lab[:, 0:1] / 100.0,
                (lab[:, 1:2] + 128.0) / 255.0,
                (lab[:, 2:3] + 128.0) / 255.0,
            ],
            dim=1,
        )
    else:
        first = t
    first = (first - 0.5) / 0.5

    h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    if hue_encoding == 'chroma':
        second = torch.cat(
            [s * torch.cos(h), s * torch.sin(h), (v - 0.5) / 0.5], dim=1
        )
    elif hue_encoding == 'raw':
        second = torch.cat(
            [(h - math.pi) / math.pi, (s - 0.5) / 0.5, (v - 0.5) / 0.5], dim=1
        )
    else:
        raise ValueError(
            f"hue_encoding must be 'chroma' or 'raw', got {hue_encoding!r}"
        )

    out = torch.cat([first, second], dim=1)
    return out.squeeze(0).permute(1, 2, 0).numpy()


def ensure_slide_cache(
    tif_path,
    cache_dir,
    tile=1024,
    halo=32,
    denoise_fn=None,
    enable_lab=True,
    hue_encoding='chroma',
):
    """
    Build (once) the preprocessed [H, W, 6] float16 cache for one slide and
    return its path. Rebuilt automatically if the TIFF is newer than the cache.
    If you change the denoise settings, delete the cache folder.

    Processing is tile by tile; `halo` extra pixels are read around each tile
    so the denoising has no seams. Set halo >= your blur kernel radius.
    """
    tag = (
        f'lab{int(enable_lab)}_{hue_encoding}_dn{int(denoise_fn is not None)}'
    )
    stem = os.path.splitext(os.path.basename(tif_path))[0]
    out_path = os.path.join(cache_dir, f'{stem}.{tag}.npy')
    if os.path.exists(out_path) and os.path.getmtime(
        out_path
    ) >= os.path.getmtime(tif_path):
        return out_path

    os.makedirs(cache_dir, exist_ok=True)
    stack = open_stack(tif_path)
    _, H, W = stack.shape
    print(f'Building preprocessed cache for {stem} ({H}x{W}) ...')

    rgb_scale = detect_rgb_scale(stack[0:3])
    q_low, q_high = compute_stretch_range(stack[0:3], rgb_scale=rgb_scale)

    tmp_path = out_path + '.tmp'
    out = np.lib.format.open_memmap(
        tmp_path, mode='w+', dtype=np.float16, shape=(H, W, N_CHANNELS)
    )
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            ya, yb = max(y0 - halo, 0), min(y1 + halo, H)
            xa, xb = max(x0 - halo, 0), min(x1 + halo, W)
            rgb = np.moveaxis(stack[0:3, ya:yb, xa:xb], 0, -1)
            res = preprocess_tile(
                rgb,
                q_low,
                q_high,
                denoise_fn,
                enable_lab,
                hue_encoding,
                rgb_scale,
            )
            out[y0:y1, x0:x1] = res[y0 - ya : y1 - ya, x0 - xa : x1 - xa]
    out.flush()
    del out
    os.replace(tmp_path, out_path)  # only a finished cache gets the real name
    return out_path


def coarse_wall_map(sem, B=None, rows=1024):
    """
    Block map of the wall plane: one boolean per BxB block, True when the block
    contains at least one wall pixel. 'Tissue' means 'contains cell walls', so it
    also works when the background is stored as one big label.
    """
    B = B or COARSE
    H, W = sem.shape
    out = np.zeros((-(-H // B), -(-W // B)), dtype=bool)
    for y0 in range(0, H, rows):  # rows is a multiple of B
        chunk = np.asarray(sem[y0 : y0 + rows]) > 0
        h, w = chunk.shape
        ph, pw = (-h) % B, (-w) % B
        if ph or pw:
            chunk = np.pad(chunk, ((0, ph), (0, pw)))
        out[y0 // B : y0 // B + chunk.shape[0] // B] = chunk.reshape(
            chunk.shape[0] // B, B, chunk.shape[1] // B, B
        ).any(axis=(1, 3))
    return out


def ensure_wall_map(tif_path, cache_dir):
    """Build (once) the coarse wall map of one slide and return its path."""
    stem = os.path.splitext(os.path.basename(tif_path))[0]
    out_path = os.path.join(cache_dir, f'{stem}.walls{COARSE}.npy')
    if os.path.exists(out_path) and os.path.getmtime(
        out_path
    ) >= os.path.getmtime(tif_path):
        return out_path
    os.makedirs(cache_dir, exist_ok=True)
    np.save(out_path, coarse_wall_map(open_stack(tif_path)[3]))
    return out_path


# =====================================================================
# Dataset
# =====================================================================
class LargeTiffCellDataset(Dataset):
    """
    offsets    : list of (dy, dx) affinity offsets, e.g.
                 make_config()['affinity']['offsets']. Default: the default
                 ranges from wseg_config (4 directions x 2 ranges = 8 channels).
    scale_range: random zoom (min, max) while augmenting. Cells in your data
                 span roughly 9 to 130 px in diameter, so consider widening
                 this, e.g. (0.5, 1.5).
    tissue_bias: probability (0..1) that a crop is redrawn until at least
                 `min_tissue` of its window's 8x8 blocks contain cell walls
                 ('tissue'); background, even if labelled, is not tissue. 0 =
                 uniform over the slide, 0.7 = mostly tissue but some pure
                 background is still seen.
    seed=None  : fresh random crops every call (training).
    seed=<int> : crop `idx` is always the same (validation). Combine with
                 augment=False, which also fixes the zoom at 1.0.
    """

    def __init__(
        self,
        data_dir,
        cache_dir=None,
        offsets=None,
        target_crop_size=256,
        samples_per_epoch=1000,
        augment=True,
        photometric=True,
        scale_range=(0.8, 1.2),
        seed=None,
        enable_lab=True,
        hue_encoding='chroma',
        denoise_fn=None,
        w0=10.0,
        sigma=5.0,
        tissue_bias=0.0,
        min_tissue=0.05,
        geometry=None,
    ):
        if photometric and not enable_lab:
            raise ValueError(
                'photometric augmentation assumes enable_lab=True '
                '(channel 0 = L); adapt _photometric otherwise'
            )
        self.offsets = [
            tuple(o)
            for o in (
                offsets
                if offsets is not None
                else make_offsets(DEFAULT_RANGES)
            )
        ]
        # The label crop is padded by the largest offset on every side so that
        # affinities near the window border are computed from real labels.
        self.pad = max(max(abs(dy), abs(dx)) for dy, dx in self.offsets)

        self.target_crop_size = target_crop_size
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        self.photometric = photometric
        self.scale_range = scale_range
        self.seed = seed
        self.w0, self.sigma = w0, sigma
        self.tissue_bias = tissue_bias
        # None = no geometry targets; or a dict of compute_geometry options
        # (the 'geometry' block of the model config).
        self.geometry = geometry
        self.min_tissue = min_tissue

        paths = sorted(
            glob.glob(os.path.join(data_dir, '*.tif'))
            + glob.glob(os.path.join(data_dir, '*.tiff'))
        )
        if not paths:
            raise FileNotFoundError(
                f'No TIFF images found in folder: {data_dir}'
            )

        cache_dir = cache_dir or os.path.join(data_dir, 'cache')
        self.slides = []
        for p in paths:
            cache = ensure_slide_cache(
                p,
                cache_dir,
                denoise_fn=denoise_fn,
                enable_lab=enable_lab,
                hue_encoding=hue_encoding,
            )
            self.slides.append(
                {
                    'tif': p,
                    'cache': cache,
                    'walls': ensure_wall_map(p, cache_dir),
                }
            )

        # Memmaps are opened lazily, per process, so nothing large is pickled
        # when DataLoader workers start (matters on Windows / spawn).
        self._open = {}
        self._coarse = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_open'] = {}
        state['_coarse'] = {}
        return state

    def _tissue_integral(self, i, sem):
        """Integral image of the coarse wall map, so the tissue fraction of any
        window is four lookups."""
        if i not in self._coarse:
            if 'walls' in self.slides[i]:
                tissue = np.load(self.slides[i]['walls'])
            else:
                tissue = coarse_wall_map(sem)
            ii = np.zeros(
                (tissue.shape[0] + 1, tissue.shape[1] + 1), dtype=np.int64
            )
            ii[1:, 1:] = tissue.astype(np.int64).cumsum(0).cumsum(1)
            self._coarse[i] = ii
        return self._coarse[i]

    def _tissue_fraction(self, ii, y, x, size):
        B = COARSE
        y0, y1 = y // B, min((y + size) // B, ii.shape[0] - 1)
        x0, x1 = x // B, min((x + size) // B, ii.shape[1] - 1)
        area = max((y1 - y0) * (x1 - x0), 1)
        return (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]) / area

    def __len__(self):
        return self.samples_per_epoch

    def _slide(self, i):
        if i not in self._open:
            stack = open_stack(self.slides[i]['tif'])
            x6 = np.load(self.slides[i]['cache'], mmap_mode='r')
            self._open[i] = (x6, stack[3], stack[4])
        return self._open[i]

    def _photometric(self, x, rng):
        """Mild, image-only jitter on a [T, T, 6] float32 array (in place)."""
        gain = rng.uniform(0.85, 1.15)
        bias = rng.uniform(-0.10, 0.10)
        for c in (0, 5):  # luminance-like channels: L and V
            x[..., c] = x[..., c] * gain + bias
        x[..., 1:5] += rng.normal(0.0, 0.03, size=4).astype(
            np.float32
        )  # colour shift
        noise_sigma = rng.uniform(0.0, 0.03)
        x += rng.standard_normal(x.shape, dtype=np.float32) * noise_sigma
        return np.clip(x, -1.0, 1.0)

    def __getitem__(self, idx):
        rng = (
            np.random.default_rng()
            if self.seed is None
            else np.random.default_rng([self.seed, idx])
        )
        T, pad = self.target_crop_size, self.pad
        S = T + 2 * pad  # padded window that is cropped, resized and augmented

        slide_i = int(rng.integers(len(self.slides)))
        x6_full, sem_full, inst_full = self._slide(slide_i)
        H, W = sem_full.shape

        # Zoom factor (random when augmenting, fixed at 1.0 otherwise)
        min_scale = (S + 2) / min(H, W)
        lower = max(self.scale_range[0], min_scale)
        upper = max(self.scale_range[1], lower)
        scale = (
            rng.uniform(lower, upper) if self.augment else max(1.0, min_scale)
        )

        # Crop in original pixel space BEFORE resizing
        cs = int(np.ceil(S / scale))
        # With probability tissue_bias, redraw the location (up to 50 times)
        # until the window has walls in at least min_tissue of its blocks, so slides
        # that are mostly empty background do not waste most of the samples.
        biased = rng.random() < self.tissue_bias
        ii = self._tissue_integral(slide_i, sem_full) if biased else None
        for _ in range(50 if biased else 1):
            y = int(rng.integers(0, H - cs + 1))
            x = int(rng.integers(0, W - cs + 1))
            if (
                not biased
                or self._tissue_fraction(ii, y, x, cs) >= self.min_tissue
            ):
                break
        x6 = np.asarray(x6_full[y : y + cs, x : x + cs], dtype=np.float32)
        sem = np.asarray(sem_full[y : y + cs, x : x + cs])
        inst = np.asarray(inst_full[y : y + cs, x : x + cs])

        x6 = resize(
            x6, (S, S), order=1, preserve_range=True, anti_aliasing=True
        )
        sem = resize(sem, (S, S), order=0, preserve_range=True)
        inst = resize(inst, (S, S), order=0, preserve_range=True).astype(
            np.int32
        )

        # Geometric augmentation (image and labels together)
        if self.augment:
            if rng.random() > 0.5:  # left-right
                x6, sem, inst = x6[:, ::-1], sem[:, ::-1], inst[:, ::-1]
            if rng.random() > 0.5:  # up-down
                x6, sem, inst = x6[::-1], sem[::-1], inst[::-1]
            r = int(rng.integers(0, 4))
            if r:
                x6, sem, inst = (
                    np.rot90(a, r, axes=(0, 1)) for a in (x6, sem, inst)
                )

        # Targets: affinities for every offset, computed AFTER augmentation
        aff = compute_affinities(inst, self.offsets, T, pad)  # [T, T, n]
        c = slice(pad, pad + T)  # central T x T window
        wall = sem[c, c] > 0

        x6 = np.array(x6[c, c], dtype=np.float32, order='C')  # contiguous copy
        if self.augment and self.photometric:
            x6 = self._photometric(x6, rng)

        parts = [wall[..., None].astype(np.float32), aff]
        if self.geometry is not None:
            g = self.geometry
            parts.append(
                compute_geometry(
                    inst,
                    T,
                    pad,
                    skeleton_width=g.get('skeleton_width', 2),
                    skeleton_min_rel=g.get('skeleton_min_rel', 0.2),
                    min_edge_distance=g.get('min_edge_distance', 2.0),
                )
            )
        target = np.concatenate(parts, axis=-1)
        weights = edt_weights(wall, self.w0, self.sigma)[
            ..., None
        ]  # [T, T, 1]
        return (
            torch.from_numpy(x6),
            torch.from_numpy(target),
            torch.from_numpy(weights),
        )
