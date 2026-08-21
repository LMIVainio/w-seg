import os
import sys

os.environ['KERAS_BACKEND'] = 'torch'
import math

# =====================================================================
# Get device safely across platforms (Windows/Linux/macOS) with CUDA, MPS, and CPU fallback
# =====================================================================
import platform

# =====================================================================
# Downloading the model
# =====================================================================
import urllib.request

# =====================================================================
# Loading and saving images
# =====================================================================
from pathlib import Path
from typing import Union
from urllib.error import HTTPError, URLError

import keras
import kornia.color as K_color

# =====================================================================
# Image preprocessing and filtering
# =====================================================================
import kornia.filters as K_filters

# Import napari last for avoiding missing dll issues on Windows
import napari

# =====================================================================
# NetworkX for cell contact graph and spatial feature extraction
# =====================================================================
import networkx as nx

# =====================================================================
# CNN and basic utils
# =====================================================================
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from keras import layers
from napari.utils.notifications import show_info, show_warning

# =====================================================================
# QtPy for GUI elements for curation Widget
# =====================================================================
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
)
from scipy.ndimage import label as label_connected_components
from scipy.spatial import Delaunay, Voronoi
from shapely.geometry import Polygon, box
from skimage.segmentation import expand_labels, watershed


def get_device() -> torch.device:
    current_os = platform.system()

    # ---------------------------------------------------------
    # Windows Strategy: CUDA -> NPU -> DirectML (DirectX) -> CPU
    # ---------------------------------------------------------
    if current_os == 'Windows':
        # 1. Check CUDA
        try:
            if torch.cuda.is_available():
                return torch.device('cuda')
        except Exception:  # noqa: BLE001
            pass  # Handles missing CUDA DLLs safely

        # 2. Check NPU (Intel NPU via torch_npu or IPEX if available)
        try:
            import torch_npu  # noqa: F401

            if torch.npu.is_available():
                return torch.device('npu')
        except ImportError:
            pass
        except Exception:  # noqa: BLE001
            pass

        # 3. Check DirectX / DirectML (requires torch-directml package)
        try:
            import torch_directml

            if torch_directml.is_available():
                return torch_directml.device()
        except ImportError:
            pass
        except Exception:  # noqa: BLE001
            pass

        return torch.device('cpu')

    # ---------------------------------------------------------
    # Linux Strategy: CUDA -> CPU
    # ---------------------------------------------------------
    elif current_os == 'Linux':
        try:
            if torch.cuda.is_available():
                return torch.device('cuda')
        except Exception:  # noqa: BLE001
            pass

        return torch.device('cpu')

    # ---------------------------------------------------------
    # macOS Strategy: Metal Performance Shaders (MPS) -> CPU
    # ---------------------------------------------------------
    elif current_os == 'Darwin':
        try:
            if (
                hasattr(torch.backends, 'mps')
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            ):
                return torch.device('mps')
        except Exception:  # noqa: BLE001
            pass

        return torch.device('cpu')

    # Fallback for other operating systems
    return torch.device('cpu')


# Get device at module load time
device = get_device()


# =====================================================================
# Custom normalization class for Keras 3 (channels_last format)
# =====================================================================
@keras.saving.register_keras_serializable(package='w_seg')
class InstanceNormalization(layers.Layer):
    """
    Framework-agnostic Instance Normalization layer for Keras 3 (channels_last format).
    Calculates mean and variance across spatial dimensions [Height, Width] dynamically.
    """

    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        # input_shape is (Batch, Height, Width, Channels)
        num_channels = input_shape[-1]
        self.gamma = self.add_weight(
            name='gamma',
            shape=(num_channels,),
            initializer='ones',
            trainable=True,
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(num_channels,),
            initializer='zeros',
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # Calculate mean and variance across Height (axis 1) and Width (axis 2)
        mean = keras.ops.mean(inputs, axis=[1, 2], keepdims=True)
        variance = keras.ops.var(inputs, axis=[1, 2], keepdims=True)
        normalized = (inputs - mean) / keras.ops.sqrt(variance + self.epsilon)
        # Apply learnable scale and shift parameters
        return self.gamma * normalized + self.beta


# =====================================================================
# Functions for setting up the generator
# =====================================================================
def conv_block(filters, size, stride=1, apply_norm=True):
    """Standard Convolution block with optional InstanceNorm."""
    initializer = keras.initializers.RandomNormal(0.0, 0.02)
    result = keras.Sequential()
    result.add(
        layers.Conv2D(
            filters,
            size,
            strides=stride,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False,
        )
    )
    if apply_norm:
        result.add(InstanceNormalization())
    result.add(layers.LeakyReLU(0.2))
    return result


def conv_block_sep(filters, size, stride=1, apply_norm=True):
    """Separable Convolution block with optional InstanceNorm."""
    initializer = keras.initializers.RandomNormal(0.0, 0.02)
    result = keras.Sequential()
    result.add(
        layers.SeparableConv2D(
            filters,
            size,
            strides=stride,
            padding='same',
            depthwise_initializer=initializer,
            pointwise_initializer=initializer,
            use_bias=False,
        )
    )
    if apply_norm:
        result.add(InstanceNormalization())
    result.add(layers.LeakyReLU(0.2))
    return result


def upsample_block(filters, size, apply_dropout=False):
    """Transposed convolution for smoother boundary upsampling without block artifacts."""
    initializer = keras.initializers.RandomNormal(0.0, 0.02)
    result = keras.Sequential()
    # Using Conv2DTranspose instead of Nearest UpSampling + Conv2D reduces grid artifacts on cell walls
    result.add(
        layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False,
        )
    )
    result.add(InstanceNormalization())
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result


def attention_gate(gated_signal, skip_signal, filters):
    theta_x = layers.Conv2D(
        filters, 1, strides=1, padding='same', use_bias=False
    )(skip_signal)
    phi_g = layers.Conv2D(
        filters, 1, strides=1, padding='same', use_bias=False
    )(gated_signal)

    add = layers.add([theta_x, phi_g])
    relu = layers.Activation('relu')(add)
    psi = layers.Conv2D(
        1, 1, padding='same', activation='sigmoid', use_bias=True
    )(relu)

    return layers.multiply([skip_signal, psi])


def Generator():
    inputs = layers.Input(shape=[256, 256, 6])

    # --- 1. Initial High-Res Feature Preserving Layer (Stride 1) ---
    x_enc_init = conv_block(32, 3, stride=1, apply_norm=False)(inputs)
    x_sep_init = conv_block_sep(32, 3, stride=1, apply_norm=False)(inputs)

    # --- 2. Downsampling Stacks (Down to 16x16) ---
    down_stack = [
        conv_block(64, 4, stride=2),  # 256 -> 128
        conv_block(128, 4, stride=2),  # 128 -> 64
        conv_block(256, 4, stride=2),  # 64 -> 32
        conv_block(512, 4, stride=2),  # 32 -> 16
    ]

    down_stack_s = [
        conv_block_sep(64, 4, stride=2),
        conv_block_sep(128, 4, stride=2),
        conv_block_sep(256, 4, stride=2),
        conv_block_sep(512, 4, stride=2),
    ]

    # Encoder forward passes
    x_enc = x_enc_init
    skips = [x_enc_init]  # Preserve raw high-res features at 256x256
    for down in down_stack:
        x_enc = down(x_enc)
        skips.append(x_enc)

    x_sep = x_sep_init
    skips_s = [x_sep_init]
    for down in down_stack_s:
        x_sep = down(x_sep)
        skips_s.append(x_sep)

    # --- 3. Bottleneck ---
    bottleneck_concat = layers.Concatenate(axis=-1)([skips[-1], skips_s[-1]])

    current_decoder_tensor = layers.Conv2D(
        512, 3, padding='same', dilation_rate=2, use_bias=False
    )(bottleneck_concat)
    current_decoder_tensor = InstanceNormalization()(current_decoder_tensor)
    current_decoder_tensor = layers.LeakyReLU(0.2)(current_decoder_tensor)

    # Reverse skip connections excluding the bottleneck
    skips = list(reversed(skips[:-1]))  # [32x32, 64x64, 128x128, 256x256]
    skips_s = list(reversed(skips_s[:-1]))

    # --- 4. Decoder Stacks (4 stages to return to 256x256) ---
    up_stack = [
        upsample_block(256, 4, apply_dropout=True),  # 16 -> 32
        upsample_block(128, 4, apply_dropout=True),  # 32 -> 64
        upsample_block(64, 4),  # 64 -> 128
        upsample_block(32, 4),  # 128 -> 256
    ]

    for i in range(len(skips)):
        up_scaled = up_stack[i](current_decoder_tensor)
        filters_skip = skips[i].shape[-1]

        gated_skip1 = attention_gate(up_scaled, skips[i], filters_skip)
        gated_skip2 = attention_gate(up_scaled, skips_s[i], filters_skip)

        merged = layers.Concatenate(axis=-1)(
            [up_scaled, gated_skip1, gated_skip2]
        )

        out_filters = skips[i].shape[-1]
        current_decoder_tensor = layers.Conv2D(
            out_filters, 3, padding='same', use_bias=False
        )(merged)
        current_decoder_tensor = InstanceNormalization()(
            current_decoder_tensor
        )
        current_decoder_tensor = layers.LeakyReLU(0.2)(current_decoder_tensor)

    # --- 5. Output Heads ---
    initializer = keras.initializers.RandomNormal(0.0, 0.02)

    # Head A: Semantic Segmentation Branch
    x_sem = layers.Conv2D(32, 3, padding='same', activation='relu')(
        current_decoder_tensor
    )
    semantic_output = layers.Conv2D(
        filters=1,
        kernel_size=1,
        padding='same',
        kernel_initializer=initializer,
        activation='sigmoid',
        name='semantic_head',
    )(x_sem)

    # Head B: Affinity Prediction Branch
    x_aff = layers.Conv2D(32, 3, padding='same', activation='relu')(
        current_decoder_tensor
    )
    affinity_output = layers.Conv2D(
        filters=2,
        kernel_size=1,
        padding='same',
        kernel_initializer=initializer,
        activation='sigmoid',
        name='affinity_head',
    )(x_aff)

    return keras.Model(
        inputs=inputs,
        outputs=[semantic_output, affinity_output],
        name='WASP-Generator-V2-HighRes',
    )


# =====================================================================
# Download weights and load model
# =====================================================================
# Base URLs for downloading w-seg model weights
WEIGHT_URL_MAP = {
    # Main w-seg release
    'w-seg_TolBlue_Populus.weights.h5': (
        'https://github.com/LMIVainio/w-seg/releases/download/w-seg/w-seg_TolBlue_Populus.weights.h5'
    ),
    'w-seg_RuthRed-TolBlu_Arabidopsis.weights.h5': (
        'https://github.com/LMIVainio/w-seg/releases/download/w-seg/w-seg_RuthRed-TolBlu_Arabidopsis.weights.h5'
    ),
    'w-seg_TolBlu_lignin.weights.h5': (
        'https://github.com/LMIVainio/w-seg/releases/download/w-seg/w-seg_TolBlu_lignin.weights.h5'
    ),
}
# Fallback release URL
DEFAULT_RELEASE_BASE_URL = (
    'https://github.com/LMIVainio/w-seg/releases/download/w-seg/'
)


def get_generator_weights(w_name: str) -> str:
    """
    Resolves weight file location. If w_name is a local path that exists, uses it directly.
    Otherwise, checks WEIGHT_URL_MAP for a specific download URL (or falls back to default base),
    caches the file in ~/.wseg_models, and downloads if missing.
    """
    # 1. If user passed a full local path that exists, use it immediately
    local_path = Path(w_name)
    if local_path.is_file():
        return str(local_path)

    # 2. Set up local cache path in home directory
    cache_dir = Path.home() / '.wseg_models'
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = local_path.name
    weights_path = cache_dir / filename

    # 3. Determine target download URL
    if w_name in WEIGHT_URL_MAP:
        weights_url = WEIGHT_URL_MAP[w_name]
    elif filename in WEIGHT_URL_MAP:
        weights_url = WEIGHT_URL_MAP[filename]
    else:
        # Fallback constructor if an unmapped filename is passed
        weights_url = f'{DEFAULT_RELEASE_BASE_URL}{filename}'

    # 4. Download if not locally cached
    if not weights_path.exists():
        print(
            f'Downloading model weights ({filename}) from {weights_url} to {weights_path}...'
        )
        try:
            urllib.request.urlretrieve(weights_url, str(weights_path))
            print(f'Download complete: {filename}')
        except (URLError, HTTPError, OSError) as e:
            raise RuntimeError(
                f"Failed to download weight file '{filename}' from {weights_url}: {e}"
            ) from e

    return str(weights_path)


def load_my_generator(
    device: torch.device, w_name: str = 'w-seg_TolBlue_Populus.weights.h5'
):
    """
    Instantiates the Generator, fetches/downloads weights specified by w_name,
    loads parameters, and prepares the model for inference.
    """
    weights_file = get_generator_weights(w_name)

    generator = Generator()
    generator.to(device)
    generator.load_weights(weights_file)
    generator.eval()

    print(f'Model loaded successfully on device: {device}')
    print(f'Model loaded from path: {weights_file}')

    return generator


# =====================================================================
# Image processing functions
# =====================================================================


def create_blend_mask(patch_size=256):
    """Creates a 2D weight mask that tapers to 0 at the edges to blend overlapping regions."""
    # Linear ramp from 0 to 1, then back to 0
    ramp = np.linspace(0, 1, patch_size // 2)
    ramp = np.concatenate([ramp, ramp[::-1]])

    # 2D mask via outer product
    mask_2d = np.outer(ramp, ramp)

    # Keep a small flat zone in the dead center if desired, or just use the ramp
    # To avoid dividing by absolute zero later, clip the minimum weight slightly
    mask_2d = np.clip(mask_2d, 1e-4, 1.0)
    return torch.from_numpy(mask_2d).float()


def mem_safe_bilateral_blur(
    tensor: torch.Tensor, tile_size: int = 2000, pad: int = 2
) -> torch.Tensor:
    """
    Applies Kornia bilateral blur in spatial tiles to keep peak RAM < 2GB.
    """
    print('')
    b, c, h, w = tensor.shape
    output = torch.empty_like(tensor)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            # Define tile bounds with overlap padding
            y_min, y_max = max(0, y - pad), min(h, y + tile_size + pad)
            x_min, x_max = max(0, x - pad), min(w, x + tile_size + pad)

            tile = tensor[:, :, y_min:y_max, x_min:x_max]

            # Blur the tile
            blurred_tile = K_filters.bilateral_blur(
                tile,
                kernel_size=(5, 5),
                sigma_color=0.1,
                sigma_space=(1.5, 1.5),
            )

            # Calculate slice indices to crop out the overlap halo
            out_y_start = y
            out_y_end = min(h, y + tile_size)
            out_x_start = x
            out_x_end = min(w, x + tile_size)

            tile_y_start = y - y_min
            tile_y_end = tile_y_start + (out_y_end - out_y_start)
            tile_x_start = x - x_min
            tile_x_end = tile_x_start + (out_x_end - out_x_start)

            # Paste back into final tensor
            output[:, :, out_y_start:out_y_end, out_x_start:out_x_end] = (
                blurred_tile[
                    :, :, tile_y_start:tile_y_end, tile_x_start:tile_x_end
                ]
            )

    return output


def preprocess_image_channels(
    img_tensor: torch.Tensor,
    enable_stretch: bool = True,
    enable_denoise: bool = True,
    enable_lab: bool = True,
    enable_centering: bool = True,
) -> torch.Tensor:
    """
    Modular preprocessing pipeline converting [B, 3, H, W] tensor inputs
    into normalized/centered 6-channel tensors. Handles ultra-large images safely.
    """
    show_info('Started preprocessing pipeline...')
    tensor = img_tensor.float()
    print('Initial preprocessing: Input tensor shape:', tensor.shape)
    # 1. Percentile Stretch with Subsampling Guard for Massive Images
    if enable_stretch:
        b, c, h, w = tensor.shape
        flat = tensor.view(b, c, -1)
        num_pixels = flat.shape[-1]

        # PyTorch quantile safety limit: subsample if close to 16,000,000 pixels (torch hard limit)
        if num_pixels > 15_000_000:
            stride = num_pixels // 500_000
            sample_flat = flat[:, :, ::stride]
            show_warning(
                'large image detected. Preprocessing might take a while'
            )
        else:
            sample_flat = flat

        q_low = torch.quantile(sample_flat, 0.01, dim=-1, keepdim=True).view(
            b, c, 1, 1
        )
        q_high = torch.quantile(sample_flat, 0.99, dim=-1, keepdim=True).view(
            b, c, 1, 1
        )
        denom = torch.clamp(q_high - q_low, min=1e-6)
        tensor = torch.clamp((tensor - q_low) / denom, 0.0, 1.0)
        # show_info("Percentile stretch applied")
    # 2. Bilateral Denoise
    if enable_denoise:
        if num_pixels > 4_000_000:
            print('Applying bilateral blur in tiles...')
            tensor = mem_safe_bilateral_blur(tensor, tile_size=2000, pad=2)
        else:
            tensor = K_filters.bilateral_blur(
                tensor,
                kernel_size=(5, 5),
                sigma_color=0.1,
                sigma_space=(1.5, 1.5),
            )
        # show_info("Bilateral denoising applied")

    # 3. HSV Conversion
    tensor_hsv = K_color.rgb_to_hsv(tensor)
    # show_info("HSV conversion applied")

    # 4. Color Channels: LAB vs RGB
    if enable_lab:
        raw_lab = K_color.rgb_to_lab(tensor)
        l_chan = raw_lab[:, 0:1, :, :] / 100.0
        a_chan = (raw_lab[:, 1:2, :, :] + 128.0) / 255.0
        b_chan = (raw_lab[:, 2:3, :, :] + 128.0) / 255.0
        first_3 = torch.cat([l_chan, a_chan, b_chan], dim=1)
        # show_info("LAB conversion applied")
    else:
        first_3 = tensor

    # 5. Zero Centering to [-1, 1]
    if enable_centering:
        first_3 = (first_3 - 0.5) / 0.5
        h_chan = (tensor_hsv[:, 0:1, :, :] - math.pi) / math.pi
        s_chan = (tensor_hsv[:, 1:2, :, :] - 0.5) / 0.5
        v_chan = (tensor_hsv[:, 2:3, :, :] - 0.5) / 0.5
        tensor_hsv = torch.cat([h_chan, s_chan, v_chan], dim=1)
        # show_info("Zero-centering applied to [-1, 1] range")

    # 6. Concatenate to 6 Channels -> [B, 6, H, W]
    return torch.cat([first_3, tensor_hsv], dim=1)


def process_large_image(
    image,
    model,
    patch_size=256,
    stride=128,
    device=device,
    live_layer=None,
    live_view=True,
):
    """
    Memory-safe inference pipeline for Napari.
    Runs global 6-channel preprocessing on CPU (RAM) to avoid GPU VRAM spikes,
    then transfers individual 256x256 patches to CUDA for model inference.
    """
    model.eval()

    # 1. ENSURE INPUT IS ON CPU & NORMALIZED [0, 1]
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image)

    image_cpu = image.float().cpu()  # Force CPU
    if image_cpu.max() > 1.0:
        image_cpu = image_cpu / 255.0

    # Ensure [B, 3, H, W] layout for preprocess_image_channels
    if image_cpu.ndim == 3:
        image_batch_cpu = image_cpu.unsqueeze(0)  # (1, 3, H, W)
    else:
        image_batch_cpu = image_cpu

    # =========================================================================
    # 2. GLOBAL PREPROCESSING ON CPU (System RAM)
    # =========================================================================
    # Returns (6, H, W) 6-channel preprocessed image in system RAM
    large_6ch_cpu = preprocess_image_channels(
        image_batch_cpu,
        enable_stretch=True,
        enable_denoise=True,
        enable_lab=True,
        enable_centering=True,
    ).squeeze(0)  # Shape: (6, H, W) on CPU

    _, H, W = large_6ch_cpu.shape

    # 3. INITIALIZE CANVASES IN SYSTEM RAM (CPU) TO SAVE VRAM
    sem_canvas_cpu = torch.zeros((1, H, W), dtype=torch.float32)
    sem_weight_cpu = torch.zeros((1, H, W), dtype=torch.float32)

    aff_canvas_cpu = torch.zeros((2, H, W), dtype=torch.float32)
    aff_weight_cpu = torch.zeros((1, H, W), dtype=torch.float32)

    # Blend mask can live on CPU for canvas accumulation
    blend_mask_cpu = create_blend_mask(patch_size).unsqueeze(0)

    # Generate coordinates for sliding window
    y_coords = list(range(0, H - patch_size + 1, stride))
    if y_coords[-1] != H - patch_size:
        y_coords.append(H - patch_size)

    x_coords = list(range(0, W - patch_size + 1, stride))
    if x_coords[-1] != W - patch_size:
        x_coords.append(W - patch_size)

    # Calculate n patches for progress output
    total_patches = len(y_coords) * len(x_coords)
    progress_count = 0
    # =========================================================================
    # 4. SLIDING WINDOW: PUSH ONLY 256x256 PATCHES TO GPU
    # =========================================================================
    with torch.no_grad():
        for y in y_coords:
            for x in x_coords:
                # Print progress
                progress_count += 1
                progress_out = progress_count / total_patches * 100
                sys.stdout.flush()
                sys.stdout.write(
                    f'\r[w-seg] Prediction {progress_out:.2f}% complete'
                )
                sys.stdout.flush()
                # Slice 6-channel patch on CPU -> (6, 256, 256)
                patch_6ch_cpu = large_6ch_cpu[
                    :, y : y + patch_size, x : x + patch_size
                ]

                # 1. Push PATCH to GPU -> Channels-Last (1, H, W, 6)
                patch_gpu_cl = (
                    patch_6ch_cpu.unsqueeze(0).permute(0, 2, 3, 1).to(device)
                )

                # Record actual extracted spatial dimensions (e.g., 150x150 at borders)
                _, h, w, _ = patch_gpu_cl.shape

                # 2. Resize input patch to (256, 256) if needed
                if (h, w) != (256, 256):
                    # F.interpolate expects Channels-First: (1, 6, H, W)
                    patch_gpu_cf = patch_gpu_cl.permute(0, 3, 1, 2)
                    patch_resized = F.interpolate(
                        patch_gpu_cf,
                        size=(256, 256),
                        mode='bilinear',
                        align_corners=False,
                    )
                    # Return to Channels-Last for model pass: (1, 256, 256, 6)
                    patch_gpu_cl = patch_resized.permute(0, 2, 3, 1)

                # 3. GPU Model Pass
                pred_sem_gpu, pred_aff_gpu = model(
                    patch_gpu_cl, training=False
                )

                # 4. Convert model outputs (1, 256, 256, C) -> Channels-First (1, C, 256, 256)
                pred_sem_gpu = pred_sem_gpu.permute(
                    0, 3, 1, 2
                )  # (1, 1, 256, 256)
                pred_aff_gpu = pred_aff_gpu.permute(
                    0, 3, 1, 2
                )  # (1, 2, 256, 256)

                # 5. Resize predictions back to extracted patch dimensions (h, w)
                if (h, w) != (256, 256):
                    pred_sem_gpu = F.interpolate(
                        pred_sem_gpu,
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False,
                    )
                    pred_aff_gpu = F.interpolate(
                        pred_aff_gpu,
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False,
                    )

                # 6. Move predictions to CPU (already in C, H, W format after squeeze)
                pred_sem = pred_sem_gpu.squeeze(0).cpu()  # (1, h, w)
                pred_aff = pred_aff_gpu.squeeze(0).cpu()  # (2, h, w)

                # 7. Slice blend mask to match current patch dimensions (h, w)
                mask_patch = blend_mask_cpu[
                    :h, :w
                ]  # (h, w) or (1, h, w) depending on setup

                # 8. Accumulate on CPU canvases using actual patch bounds (y:y+h, x:x+w)
                sem_canvas_cpu[:, y : y + h, x : x + w] += (
                    pred_sem * mask_patch
                )
                sem_weight_cpu[:, y : y + h, x : x + w] += mask_patch

                aff_canvas_cpu[:, y : y + h, x : x + w] += (
                    pred_aff * mask_patch
                )
                aff_weight_cpu[:, y : y + h, x : x + w] += mask_patch

                # Live Napari Update (CPU numpy slice)
                if live_view and live_layer is not None:
                    local_sem = sem_canvas_cpu[
                        :, y : y + patch_size, x : x + patch_size
                    ]
                    local_weights = sem_weight_cpu[
                        :, y : y + patch_size, x : x + patch_size
                    ]

                    local_norm = (local_sem / local_weights).squeeze(0).numpy()
                    local_norm = np.clip(local_norm, 0.0, 1.0)

                    live_layer.data[y : y + patch_size, x : x + patch_size] = (
                        local_norm
                    )
                    live_layer.refresh()
                    napari.qt.get_qapp().processEvents()

    # Final normalization on CPU
    final_sem = torch.clamp(sem_canvas_cpu / sem_weight_cpu, 0.0, 1.0).numpy()
    final_aff = torch.clamp(aff_canvas_cpu / aff_weight_cpu, 0.0, 1.0).numpy()
    # Print newline after sysstdout.flush
    print('\n')
    return final_sem, final_aff


# =====================================================================
# Cell neighbor graph and spatial feature extraction
# =====================================================================


def build_cell_contact_graph(
    labels_layer: np.ndarray, wall_expansion_px: int = 8
) -> nx.Graph:
    """Constructs an undirected contact graph by temporarily expanding cell labels

    across cell-wall/background gaps (value 0).
    """
    print('[w-seg] Building cell contact graph...')
    G = nx.Graph()
    unique_ids = np.unique(labels_layer)
    unique_ids = unique_ids[unique_ids != 0]
    G.add_nodes_from(unique_ids)

    # Dilate labels to bridge the zero-value cell walls
    # Size depends on cell wall thickness (e.g. 5-9 px footprint)
    # Expand labels outward without overlapping each other
    # print(f"[w-seg] Expanding labels by {wall_expansion_px}")
    expanded = expand_labels(labels_layer, distance=wall_expansion_px)

    # 4-connectivity neighbor shifts on dilated labels
    slices = [
        (slice(0, -1), slice(0, -1), slice(1, None), slice(0, -1)),  # Right
        (slice(0, -1), slice(0, -1), slice(0, -1), slice(1, None)),  # Down
        (
            slice(0, -1),
            slice(0, -1),
            slice(1, None),
            slice(1, None),
        ),  # Down-Right
        (
            slice(1, None),
            slice(0, -1),
            slice(0, -1),
            slice(1, None),
        ),  # Up-Right
    ]

    for y1, x1, y2, x2 in slices:
        lab1 = expanded[y1, x1]
        lab2 = expanded[y2, x2]

        # Find adjacent distinct non-zero cells
        mask = (lab1 != lab2) & (lab1 > 0) & (lab2 > 0)
        edges = np.column_stack((lab1[mask], lab2[mask]))

        for u, v in edges:
            if G.has_edge(u, v):
                G[u][v]['contact_pixels'] += 1
            else:
                G.add_edge(u, v, contact_pixels=1)

    return G


def build_delaunay_neighbor_graph(
    df: pd.DataFrame, max_dist_px: float = 150.0
) -> nx.Graph:
    """Alternative/Fallback: Builds a topology-aware neighborhood graph via Delaunay

    triangulation of centroids, pruned by maximum distance.
    """
    print('[w-seg] Building Delaunay neighbor graph...')
    G = nx.Graph()
    label_ids = df['label_id'].values
    centroids = df[['centroid-0', 'centroid-1']].values
    G.add_nodes_from(label_ids)

    if len(centroids) < 4:
        return G

    tri = Delaunay(centroids)
    # Delaunay simplex gives natural spatial neighbors
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                u_idx, v_idx = simplex[i], simplex[j]
                u_id, v_id = label_ids[u_idx], label_ids[v_idx]

                # Calculate Euclidean distance between centroids
                dist = np.linalg.norm(centroids[u_idx] - centroids[v_idx])
                if dist <= max_dist_px:
                    G.add_edge(u_id, v_id)

    return G


def compute_bounded_voronoi_areas(
    df: pd.DataFrame, img_shape: tuple
) -> dict[int, float]:
    """Computes exact Voronoi region areas clipped to the image boundary box."""
    centroids = df[['centroid-0', 'centroid-1']].values
    label_ids = df['label_id'].values
    vor_areas = {}

    if len(centroids) < 4:
        return dict.fromkeys(label_ids, np.nan)

    img_box = box(0, 0, img_shape[1], img_shape[0])  # (minx, miny, maxx, maxy)
    vor = Voronoi(centroids)

    for idx, reg_idx in enumerate(vor.point_region):
        region = vor.regions[reg_idx]
        lid = label_ids[idx]

        if len(region) > 0 and -1 not in region:
            poly = Polygon([vor.vertices[i] for i in region])
            clipped_poly = poly.intersection(img_box)
            vor_areas[lid] = (
                clipped_poly.area if not clipped_poly.is_empty else np.nan
            )
        else:
            # Unbounded regions: reconstruct semi-infinite ridge lines and clip to image box
            vor_areas[lid] = np.nan

    # Fill remaining edge cell NaNs with median of bounded cells
    valid_vals = [v for v in vor_areas.values() if not np.isnan(v)]
    fallback_area = np.median(valid_vals) if valid_vals else 1.0
    return {
        lid: (val if not np.isnan(val) else fallback_area)
        for lid, val in vor_areas.items()
    }


def extract_spatial_feature_matrix(
    labels_layer: np.ndarray,
    df_single_cell_metrics: pd.DataFrame,
    search_radius: int = 8,
    use_delaunay_fallback: bool = True,
    max_delaunay_dist_px: float = 120.0,
) -> pd.DataFrame:
    """Combines single-cell morphometry with connectivity graph metrics and

    aggregated neighbor statistics into a single classification table.
    """
    df = df_single_cell_metrics.copy()
    if 'label_id' not in df.columns and 'label' in df.columns:
        df.rename(columns={'label': 'label_id'}, inplace=True)

    df = df.drop_duplicates(subset=['label_id']).copy()
    label_ids = df['label_id'].values
    centroids = df[['centroid-0', 'centroid-1']].values

    # 1. Physical Contact Graph with controlled expansion radius
    G = build_cell_contact_graph(labels_layer, search_radius)

    # 2. Topology-Aware Delaunay Fallback for Isolated Nodes
    if use_delaunay_fallback and len(centroids) >= 4:
        tri = Delaunay(centroids)
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    u_idx, v_idx = simplex[i], simplex[j]
                    u_id, v_id = label_ids[u_idx], label_ids[v_idx]

                    # If either cell has 0 neighbors in physical graph, check spatial proximity
                    if G.degree(u_id) == 0 or G.degree(v_id) == 0:
                        dist = np.linalg.norm(
                            centroids[u_idx] - centroids[v_idx]
                        )
                        if dist <= max_delaunay_dist_px:
                            G.add_edge(u_id, v_id, contact_pixels=0)

    # 3. Exact Image-Bounded Voronoi Crowding Areas
    vor_areas = compute_bounded_voronoi_areas(df, labels_layer.shape)

    # 4. Aggregations & Metrics
    records = []
    exclude_cols = {
        'label_id',
        'label',
        'centroid-0',
        'centroid-1',
        'bbox-0',
        'bbox-1',
        'bbox-2',
        'bbox-3',
    }
    numeric_df = df.select_dtypes(include=[np.number])
    target_agg_cols = [c for c in numeric_df.columns if c not in exclude_cols]

    df_indexed = df.set_index('label_id')

    for cell_id in label_ids:
        row_dict = {'label_id': cell_id}

        neighbors = list(G.neighbors(cell_id)) if cell_id in G else []
        degree = len(neighbors)

        contact_lens = [
            G[cell_id][nbr].get('contact_pixels', 0) for nbr in neighbors
        ]
        total_contact_len = sum(contact_lens) if neighbors else 0
        mean_contact_len = (
            float(np.mean(contact_lens))
            if neighbors and sum(contact_lens) > 0
            else 0.0
        )

        row_dict['neighbor_count'] = degree
        row_dict['total_shared_boundary_px'] = total_contact_len
        row_dict['mean_shared_boundary_px'] = mean_contact_len
        row_dict['voronoi_area'] = vor_areas.get(cell_id, 1.0)

        if cell_id in df_indexed.index and 'area' in df_indexed.columns:
            cell_area = df_indexed.loc[cell_id, 'area']
            row_dict['crowding_ratio'] = (
                row_dict['voronoi_area'] / cell_area if cell_area > 0 else 1.0
            )

        # Aggregated Neighbor Metrics
        valid_neighbors = [n for n in neighbors if n in df_indexed.index]
        if valid_neighbors:
            nbr_data = df_indexed.loc[valid_neighbors]

            for col in target_agg_cols:
                vals = nbr_data[col].dropna().values
                self_val = float(df_indexed.loc[cell_id, col])

                if len(vals) > 0:
                    mean_val = float(np.mean(vals))
                    std_val = float(np.std(vals))

                    row_dict[f'nbr_mean_{col}'] = mean_val
                    row_dict[f'nbr_std_{col}'] = std_val
                    row_dict[f'delta_{col}'] = self_val - mean_val
                    row_dict[f'ratio_{col}'] = self_val / (mean_val + 1e-6)
                else:
                    row_dict[f'nbr_mean_{col}'] = self_val
                    row_dict[f'nbr_std_{col}'] = 0.0
                    row_dict[f'delta_{col}'] = 0.0
                    row_dict[f'ratio_{col}'] = 1.0
        else:
            for col in target_agg_cols:
                self_val = float(df_indexed.loc[cell_id, col])
                row_dict[f'nbr_mean_{col}'] = self_val
                row_dict[f'nbr_std_{col}'] = 0.0
                row_dict[f'delta_{col}'] = 0.0
                row_dict[f'ratio_{col}'] = 1.0

        records.append(row_dict)

    df_spatial = pd.DataFrame(records)
    df_final = pd.merge(df, df_spatial, on='label_id', how='left')
    df_final.fillna(0, inplace=True)

    return df_final


class CurationWidget(QWidget):
    def __init__(self, viewer: 'napari.Viewer'):
        super().__init__()
        self.viewer = viewer
        self.labels_layer = None

        # Internal state tracking
        self.mode = 'navigate'  # "navigate", "merge", or "erode_split"
        self.staging_layer = None
        self.selected_label_ids = []  # Kept as a list to preserve selection order

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout()

        # 0. Global Status Header Dashboard
        self.status_lbl = QLabel('Status: Idle (Ready to select layer)')
        self.status_lbl.setStyleSheet('font-weight: bold; padding: 5px;')
        main_layout.addWidget(self.status_lbl)

        # -------------------------------------------------------------
        # 1. MERGE ENGINE INTERFACE GROUP
        # -------------------------------------------------------------
        merge_group = QGroupBox('Cell Agglomeration / Merging Tools')
        merge_layout = QVBoxLayout()

        self.merge_btn = QPushButton('Merge Cells Mode 🤝')
        self.merge_btn.setCheckable(True)
        self.merge_btn.clicked.connect(self._toggle_merge_mode)
        merge_layout.addWidget(self.merge_btn)

        # Dropdown: Choose target label strategy
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel('Merge Into Target:'))
        self.target_label_combo = QComboBox()
        self.target_label_combo.addItems(
            ['Lowest ID', 'Highest ID', 'First Selected ID']
        )
        target_layout.addWidget(self.target_label_combo)
        merge_layout.addLayout(target_layout)

        # Checkbox: Fill holes inside the merged cell
        self.fill_holes_chk = QCheckBox('Fill internal gaps / holes')
        self.fill_holes_chk.setChecked(True)
        merge_layout.addWidget(self.fill_holes_chk)

        self.apply_merge_btn = QPushButton('Apply Merge Execution ✅')
        self.apply_merge_btn.setEnabled(False)
        self.apply_merge_btn.clicked.connect(self._apply_merge)
        merge_layout.addWidget(self.apply_merge_btn)

        merge_group.setLayout(merge_layout)
        main_layout.addWidget(merge_group)

        main_layout.addSpacing(5)

        # -------------------------------------------------------------
        # 2. EROSION SPLIT INTERFACE GROUP
        # -------------------------------------------------------------
        erode_group = QGroupBox('Cell Fragmentation (Auto-Erosion)')
        erode_layout = QVBoxLayout()

        self.erode_split_btn = QPushButton('Split Cell Mode (Click Target) 🧼')
        self.erode_split_btn.setCheckable(True)
        self.erode_split_btn.clicked.connect(self._toggle_erode_split_mode)
        erode_layout.addWidget(self.erode_split_btn)

        # Minimum Cell Area/Size Control
        min_size_layout = QHBoxLayout()
        min_size_layout.addWidget(QLabel('Min Cell Size (px):'))
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(10, 100000)
        self.min_size_spin.setValue(100)
        self.min_size_spin.setToolTip(
            'Cells smaller than this total pixel count will not be split.'
        )
        min_size_layout.addWidget(self.min_size_spin)
        erode_layout.addLayout(min_size_layout)

        # Maximum Erosion Depth/Iteration Control
        max_iter_layout = QHBoxLayout()
        max_iter_layout.addWidget(QLabel('Max Erosion Steps:'))
        self.max_erode_spin = QSpinBox()
        self.max_erode_spin.setRange(5, 200)
        self.max_erode_spin.setValue(50)
        max_iter_layout.addWidget(self.max_erode_spin)
        erode_layout.addLayout(max_iter_layout)

        erode_group.setLayout(erode_layout)
        main_layout.addWidget(erode_group)

        main_layout.addStretch()
        self.setLayout(main_layout)

        self.viewer.layers.selection.events.changed.connect(
            self._on_layer_selection_changed
        )
        self._on_layer_selection_changed(None)

    def _on_layer_selection_changed(self, event):
        active_layer = self.viewer.layers.selection.active
        if (
            self.labels_layer
            and self.labels_layer in self.viewer.layers
            and self._mouse_callback in self.labels_layer.mouse_drag_callbacks
        ):
            self.labels_layer.mouse_drag_callbacks.remove(self._mouse_callback)

        if (
            isinstance(active_layer, napari.layers.Labels)
            and active_layer.name != '[Staging] Merge Selection'
        ):
            self.labels_layer = active_layer
            self.labels_layer.mouse_drag_callbacks.append(self._mouse_callback)
            self.labels_layer.interactive = True
            self.status_lbl.setText(f'Active: {self.labels_layer.name}')
            self.merge_btn.setEnabled(True)
            self.erode_split_btn.setEnabled(True)
        elif active_layer is None:
            self.labels_layer = None
            self.status_lbl.setText('Status: Select a Labels layer...')
            self.merge_btn.setEnabled(False)
            self.erode_split_btn.setEnabled(False)
            self._reset_modes()

    def _reset_modes(self):
        self.mode = 'navigate'
        self.selected_label_ids.clear()
        self.merge_btn.setChecked(False)
        self.erode_split_btn.setChecked(False)
        self.apply_merge_btn.setEnabled(False)

        self.viewer.cursor.style = 'standard'

        if self.staging_layer and self.staging_layer in self.viewer.layers:
            self.staging_layer.data[:] = 0
            self.staging_layer.refresh()

        if self.labels_layer and self.labels_layer in self.viewer.layers:
            self.labels_layer.mode = 'pan_zoom'

    def _toggle_merge_mode(self, checked):
        if checked:
            self._reset_modes()
            self.mode = 'merge'
            self.merge_btn.setChecked(True)
            self.apply_merge_btn.setEnabled(True)
            self.viewer.cursor.style = 'pointing'
            self.status_lbl.setText(
                'Merge Mode: Left-click cells to select. Right-click to deselect.'
            )

            self.labels_layer.mode = 'pan_zoom'
            staging_name = '[Staging] Merge Selection'
            if staging_name in self.viewer.layers:
                self.staging_layer = self.viewer.layers[staging_name]
                self.staging_layer.data[:] = 0
                self.staging_layer.refresh()
            else:
                staging_data = np.zeros_like(
                    self.labels_layer.data, dtype=np.uint16
                )
                self.staging_layer = self.viewer.add_labels(
                    staging_data, name=staging_name, opacity=0.6
                )
            self.viewer.layers.selection.active = self.labels_layer
        else:
            self._reset_modes()
            self.status_lbl.setText('Status: Idle')

    def _toggle_erode_split_mode(self, checked):
        if checked:
            self._reset_modes()
            self.mode = 'erode_split'
            self.erode_split_btn.setChecked(True)
            self.labels_layer.mode = 'pan_zoom'
            self.viewer.cursor.style = 'cross'
            self.status_lbl.setText(
                'Erosion Split Mode: Left-click a cell to auto-shrink/split it.'
            )
        else:
            self._reset_modes()
            self.status_lbl.setText('Status: Idle')

    def _mouse_callback(self, layer, event):
        if self.mode == 'navigate' or event.type != 'mouse_press':
            return

        data_coordinates = layer.world_to_data(event.position)
        cords = tuple(int(np.round(c)) for c in data_coordinates)
        y, x = cords[-2], cords[-1]

        try:
            val = layer.data[y, x]
        except IndexError:
            return

        if val == 0:
            return  # Ignore background

        # Route 1: Merge Engine Click Handler
        if self.mode == 'merge' and self.staging_layer:
            if event.button == 1 and val not in self.selected_label_ids:
                self.selected_label_ids.append(val)
                self.staging_layer.data[layer.data == val] = val
                self.staging_layer.refresh()
                self.labels_layer.refresh()
            elif event.button == 2 and val in self.selected_label_ids:
                self.selected_label_ids.remove(val)
                self.staging_layer.data[layer.data == val] = 0
                self.staging_layer.refresh()
                self.labels_layer.refresh()

        # Route 2: Erosion Segmentation Engine Click Handler
        elif self.mode == 'erode_split' and event.button == 1:
            self._execute_erosion_split(val)

    def _apply_merge(self):
        if not self.selected_label_ids or not self.labels_layer:
            self.status_lbl.setText('Merge aborted: No cells were selected.')
            self._reset_modes()
            return

        # 1. Determine target master ID based on dropdown choice
        choice = self.target_label_combo.currentText()
        if choice == 'Lowest ID':
            target_master_id = min(self.selected_label_ids)
        elif choice == 'Highest ID':
            target_master_id = max(self.selected_label_ids)
        else:  # "First Selected ID"
            target_master_id = self.selected_label_ids[0]

        # 2. Build joint binary mask of all selected cells
        combined_mask = np.isin(
            self.labels_layer.data, self.selected_label_ids
        )

        # 3. Fill internal holes if checkbox is checked
        if self.fill_holes_chk.isChecked():
            combined_mask = binary_fill_holes(combined_mask)

        # 4. Clear out all merged labels and assign the chosen master ID
        for label_id in self.selected_label_ids:
            self.labels_layer.data[self.labels_layer.data == label_id] = 0

        self.labels_layer.data[combined_mask] = target_master_id
        self.labels_layer.refresh()

        # 5. KEY FEATURE: Update active layer's selected label in layer controls!
        self.labels_layer.selected_label = int(target_master_id)

        self.viewer.status = f'Merged {len(self.selected_label_ids)} items into label ID: {target_master_id}'
        self.status_lbl.setText(
            f'Merged successfully! Active paint label set to {target_master_id}.'
        )

        from qtpy.QtCore import QTimer

        QTimer.singleShot(50, self._reset_modes)

    def _execute_erosion_split(self, target_id):
        """Extracts cell binary mask, evaluates minimum size, erodes down to seeds, and reconstructs boundaries."""
        self.status_lbl.setText(f'Processing cell {target_id}...')

        cell_mask = self.labels_layer.data == target_id
        cell_size = np.count_nonzero(cell_mask)

        min_size_threshold = self.min_size_spin.value()
        if cell_size < min_size_threshold:
            self.status_lbl.setText(
                f'Erosion aborted: Cell size ({cell_size} px) is below minimum threshold ({min_size_threshold} px).'
            )
            from qtpy.QtCore import QTimer

            QTimer.singleShot(1500, self._reset_modes)
            return

        shrunk_mask = cell_mask.copy()
        structuring_element = np.array(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool
        )
        max_iters = self.max_erode_spin.value()

        seeds_found = False
        num_features = 1

        for _i in range(max_iters):
            eroded = binary_erosion(shrunk_mask, structure=structuring_element)
            labeled_seeds, num_features = label_connected_components(eroded)

            if num_features >= 2:
                shrunk_mask = eroded
                seeds_found = True
                break
            elif num_features == 0:
                break

            shrunk_mask = eroded

        if not seeds_found or num_features < 2:
            self.status_lbl.setText(
                'Erosion aborted: Cell cannot be split further at this size.'
            )
            from qtpy.QtCore import QTimer

            QTimer.singleShot(1500, self._reset_modes)
            return

        markers, num_features = label_connected_components(shrunk_mask)
        distances = distance_transform_edt(cell_mask)

        # 1. Compute initial watershed segmentation inside the mask
        watershed_output = watershed(-distances, markers, mask=cell_mask)

        # -----------------------------------------------------------------
        # 2. POST-PROCESSING: Clean up tiny fragments below min size
        # -----------------------------------------------------------------
        # Measure pixel count of each fragment ID (1 .. num_features)
        fragment_ids = list(range(1, num_features + 1))
        fragment_sizes = {
            idx: np.count_nonzero(watershed_output == idx)
            for idx in fragment_ids
        }

        # Identify tiny fragments that fail the min size threshold
        too_small = [
            idx
            for idx, size in fragment_sizes.items()
            if size < min_size_threshold
        ]

        # If any fragment is too small, merge it into the largest valid neighbor
        if too_small:
            valid_fragments = [
                idx for idx in fragment_ids if idx not in too_small
            ]

            # If all fragments are too small, abort split so we don't destroy the cell
            if not valid_fragments:
                self.status_lbl.setText(
                    f'Erosion aborted: All resulting fragments (<{min_size_threshold} px) are too small.'
                )
                from qtpy.QtCore import QTimer

                QTimer.singleShot(1500, self._reset_modes)
                return

            # Merge each small fragment into the largest neighboring fragment
            for small_idx in too_small:
                small_mask = watershed_output == small_idx

                # Dilate the small fragment to find touching neighbor IDs
                dilated_small = binary_dilation(
                    small_mask, structure=structuring_element
                )
                touching_neighbors = set(
                    np.unique(watershed_output[dilated_small & cell_mask])
                ) - {0, small_idx}

                # Filter to neighbors that are valid (or pick the largest available)
                valid_touching = [
                    n for n in touching_neighbors if n in valid_fragments
                ]
                if valid_touching:
                    # Pick the largest neighbor
                    target_neighbor = max(
                        valid_touching, key=lambda n: fragment_sizes[n]
                    )
                elif touching_neighbors:
                    target_neighbor = max(
                        touching_neighbors, key=lambda n: fragment_sizes[n]
                    )
                else:
                    # Fallback to absolute largest fragment if no neighbor touches directly
                    target_neighbor = max(
                        valid_fragments, key=lambda n: fragment_sizes[n]
                    )

                # Re-assign the pixels
                watershed_output[small_mask] = target_neighbor
                fragment_sizes[target_neighbor] += fragment_sizes[small_idx]

            # Re-label remaining fragments sequentially
            remaining_ids = np.unique(watershed_output[watershed_output > 0])
            num_features = len(remaining_ids)

            # If cleanup eliminated the split, abort
            if num_features < 2:
                self.status_lbl.setText(
                    'Erosion aborted: Split created fragments smaller than min size threshold.'
                )
                from qtpy.QtCore import QTimer

                QTimer.singleShot(1500, self._reset_modes)
                return

            # Map remaining fragment IDs back to continuous 1..N indices
            remap = {
                old_id: new_id
                for new_id, old_id in enumerate(remaining_ids, start=1)
            }
            cleaned_watershed = np.zeros_like(watershed_output)
            for old_id, new_id in remap.items():
                cleaned_watershed[watershed_output == old_id] = new_id
            watershed_output = cleaned_watershed

        # -----------------------------------------------------------------
        # 3. Write new IDs back to Napari Labels layer
        # -----------------------------------------------------------------
        new_id_base = int(self.labels_layer.data.max() + 1)
        self.labels_layer.data[cell_mask] = 0

        for fragment_idx in range(1, num_features + 1):
            assigned_id = (
                target_id
                if fragment_idx == 1
                else (new_id_base + fragment_idx - 2)
            )
            self.labels_layer.data[
                cell_mask & (watershed_output == fragment_idx)
            ] = assigned_id

        self.labels_layer.refresh()
        self.status_lbl.setText(
            f'Erosion successful! Split cell {target_id} into {num_features} valid fragments.'
        )

        from qtpy.QtCore import QTimer

        QTimer.singleShot(50, self._reset_modes)


def apply_feature_colormap(
    viewer: 'napari.Viewer',
    base_layer_name: str,
    labels_layer: Union['napari.layers.Labels', np.ndarray],
    df_features: pd.DataFrame | None = None,
    feature_name: str = 'mean_wall_thickness',
    cmap_name: str = 'viridis',
):
    """
    Maps feature measurements onto the label mask geometry to create a true
    float32 Image layer displaying continuous feature values upon hover/inspection.
    """
    if labels_layer is None:
        print('[w-seg] Error: No Labels layer or array provided.')
        return

    # 1. Resolve label image data
    if isinstance(labels_layer, napari.layers.Labels):
        labels_data = labels_layer.data
    elif isinstance(labels_layer, np.ndarray):
        labels_data = labels_layer
    else:
        print(
            f'[w-seg] Error: Unsupported label layer type: {type(labels_layer)}'
        )
        return

    # 2. Extract features DataFrame
    features_df = df_features

    if (features_df is None or features_df.empty) and isinstance(
        labels_layer, napari.layers.Labels
    ):
        features_df = getattr(labels_layer, 'features', None)

        if (features_df is None or features_df.empty) and getattr(
            labels_layer, 'properties', None
        ):
            features_df = pd.DataFrame(labels_layer.properties)

    if features_df is None or features_df.empty:
        print(
            '[w-seg] Error: Selected layer lacks measured features. Run measurements first!'
        )
        return

    if feature_name not in features_df.columns:
        print(
            f"[w-seg] Error: Column '{feature_name}' not found in feature dataframe."
        )
        return

    # 3. Parse features and label IDs
    raw_vals = features_df[feature_name].to_numpy(dtype=float)

    if 'label_id' in features_df.columns:
        label_ids = features_df['label_id'].to_numpy(dtype=int)
    elif 'label' in features_df.columns:
        label_ids = features_df['label'].to_numpy(dtype=int)
    else:
        label_ids = np.arange(1, len(raw_vals) + 1, dtype=int)

    # 4. Build lookup table for array mapping (Label ID -> Feature Value)
    max_label = int(np.max(labels_data)) if labels_data.size > 0 else 0
    lookup_table = np.full(max_label + 1, np.nan, dtype=np.float32)

    # Filter out-of-bounds or invalid negative label IDs
    valid_mask = (label_ids >= 0) & (label_ids <= max_label)
    lookup_table[label_ids[valid_mask]] = raw_vals[valid_mask]

    # Map the integer label image array to float32 feature map
    feature_map = lookup_table[labels_data]

    # 5. Compute contrast limits based on valid measured numbers
    valid_vals = raw_vals[~np.isnan(raw_vals)]
    if len(valid_vals) == 0:
        val_min, val_max = 0.0, 1.0
    else:
        val_min = float(np.min(valid_vals))
        val_max = float(np.max(valid_vals))
        if val_min == val_max:  # Edge case: all values are equal
            val_max = val_min + 1e-5

    # Target naming scheme
    target_layer_name = (
        f'{base_layer_name} [{feature_name}]'
        if base_layer_name
        else f'features [{feature_name}]'
    )

    # 6. Create or update the Napari Image layer
    if target_layer_name in viewer.layers and isinstance(
        viewer.layers[target_layer_name], napari.layers.Image
    ):
        img_layer = viewer.layers[target_layer_name]
        img_layer.data = feature_map
        img_layer.colormap = cmap_name
        img_layer.contrast_limits = (val_min, val_max)
        img_layer.refresh()
        print(
            f"[w-seg] Updated Image layer '{target_layer_name}' -> ({val_min:.2f} to {val_max:.2f})"
        )
    else:
        if target_layer_name in viewer.layers:
            viewer.layers.remove(target_layer_name)

        viewer.add_image(
            feature_map,
            name=target_layer_name,
            colormap=cmap_name,
            contrast_limits=(val_min, val_max),
            opacity=0.85,
            rendering='attenuated_mip',
        )
        print(
            f"[w-seg] Created Image layer '{target_layer_name}' -> ({val_min:.2f} to {val_max:.2f})"
        )


def sanitize_feature_matrix(
    data: Union[pd.DataFrame, np.ndarray],
    feature_cols: list[str] | None = None,
    exclude_cols: list[str] | None = None,
    clip_value: float = 1e9,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Sanitizes feature matrices by replacing infinite values (Inf, -Inf) with NaN,
    filling NaN/null values with median statistics, and clipping extreme float overflows.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        The raw feature matrix containing single-cell and spatial metrics.
    feature_cols : list of str, optional
        Specific numeric columns to sanitize. If None, targets all numeric columns.
    exclude_cols : list of str, optional
        Metadata/ID columns to skip (e.g., ['label_id', 'centroid-0', 'centroid-1']).
    clip_value : float
        Upper and lower bound magnitude to prevent float64 overflow errors.

    Returns
    -------
    pd.DataFrame or np.ndarray
        A cleaned copy of the data ready for scikit-learn training or inference.
    """
    if data is None:
        return None

    if exclude_cols is None:
        exclude_cols = [
            'label_id',
            'label',
            'centroid-0',
            'centroid-1',
            'bbox-0',
            'bbox-1',
            'bbox-2',
            'bbox-3',
        ]

    # --- Case 1: Pandas DataFrame ---
    if isinstance(data, pd.DataFrame):
        df_clean = data.copy()

        # Select numeric columns to sanitize
        if feature_cols is None:
            feature_cols = [
                c
                for c in df_clean.select_dtypes(include=[np.number]).columns
                if c not in exclude_cols and not c.startswith('class_')
            ]

        # 1. Convert infinite values (Inf, -Inf) to NaN
        df_clean[feature_cols] = df_clean[feature_cols].replace(
            [np.inf, -np.inf], np.nan
        )

        # 2. Fill NaNs with column median (fallback to 0.0 if entire column is NaN)
        medians = df_clean[feature_cols].median()
        medians = medians.fillna(0.0)
        df_clean[feature_cols] = df_clean[feature_cols].fillna(medians)

        # 3. Clip extreme values to prevent float64 boundary overflow
        df_clean[feature_cols] = df_clean[feature_cols].clip(
            lower=-clip_value, upper=clip_value
        )

        return df_clean

    # --- Case 2: NumPy Array ---
    elif isinstance(data, np.ndarray):
        arr = data.astype(np.float64, copy=True)

        # Replace Inf and -Inf with NaN
        arr[np.isinf(arr)] = np.nan

        # Replace NaNs with column medians
        col_medians = np.nanmedian(arr, axis=0)
        col_medians[np.isnan(col_medians)] = (
            0.0  # Fallback if entire column was NaN
        )

        inds = np.where(np.isnan(arr))
        arr[inds] = np.take(col_medians, inds[1])

        # Clip bounds
        np.clip(arr, -clip_value, clip_value, out=arr)

        return arr

    else:
        raise TypeError(
            f'Unsupported input type for sanitize_feature_matrix: {type(data)}'
        )
