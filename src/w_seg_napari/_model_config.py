"""
(Copy of wseg_config.py from the training code; keep the two in sync.)

Self-describing model format: weights + a small JSON config next to them.

The plugin and any decoder read the config instead of hard-coding the
affinity layout. Channel order is RANGE-MAJOR:

    channels [0:4]  = range 0 (shortest), directions E, S, SE, SW
    channels [4:8]  = range 1,            directions E, S, SE, SW
    ...

so "one napari layer per range" = 4 consecutive channels, and adding a third
range later adds a third layer without touching any plugin code.

Offsets are (dy, dx) in pixels. An affinity is 1 when pixel p and pixel
p + offset lie in the same cell (instance label > 0), otherwise 0 (different
cell, or one of the two pixels is wall). Diagonal offsets move k pixels along
BOTH axes, i.e. a distance of k * sqrt(2).
"""

import json
import os

FORMAT_VERSION = 1

DIRECTION_NAMES = ('E', 'S', 'SE', 'SW')
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))  # unit steps (dy, dx)

DEFAULT_RANGES = (6, 32)
# Things that travel with a model and are used by other plugin tabs:
#   yolo            file name of a YOLO cell-type classifier inside the bundle (or None)
#   radial_symmetry True/False = preset the 'radial symmetry' option, None = leave as is
DEFAULT_COMPANIONS = {'yolo': None, 'radial_symmetry': None}

DEFAULT_DENOISE = {
    'kind': 'bilateral',
    'kernel_size': [5, 5],
    'sigma_color': 0.1,
    'sigma_space': [1.5, 1.5],
}


def make_offsets(ranges, directions=DIRECTIONS):
    """All (dy, dx) offsets, range-major: every direction of ranges[0], then ranges[1], ..."""
    return [(dy * k, dx * k) for k in ranges for (dy, dx) in directions]


def make_config(
    ranges=DEFAULT_RANGES,
    directions=DIRECTIONS,
    direction_names=DIRECTION_NAMES,
    enable_lab=True,
    hue_encoding='chroma',
    patch_size=256,
    input_channels=6,
    denoise=None,
    companions=None,
):
    """The config dict that is saved next to the model weights."""
    if len(directions) != len(direction_names):
        raise ValueError(
            'directions and direction_names must have equal length'
        )
    offsets = make_offsets(ranges, directions)
    return {
        'format_version': FORMAT_VERSION,
        'input': {'channels': input_channels, 'patch_size': patch_size},
        'preprocessing': {
            'enable_lab': enable_lab,
            'hue_encoding': hue_encoding,
            'stretch_percentiles': [1, 99],
            'denoise': denoise if denoise is not None else DEFAULT_DENOISE,
        },
        'outputs': {'wall': 'semantic_head', 'affinity': 'affinity_head'},
        'companions': {**DEFAULT_COMPANIONS, **(companions or {})},
        'affinity': {
            'ranges': list(ranges),
            'direction_names': list(direction_names),
            'directions': [list(d) for d in directions],
            'offsets': [list(o) for o in offsets],
            'convention': '1 = same cell, 0 = different cell or wall',
        },
    }


def legacy_config():
    """Config describing the OLD model: 2 channels at k = 8 (E and S), raw hue."""
    cfg = make_config(
        ranges=(8,),
        directions=((0, 1), (1, 0)),
        direction_names=('E', 'S'),
        hue_encoding='raw',
    )
    cfg['legacy'] = True
    return cfg


def validate_config(cfg):
    """Raise ValueError if the config is internally inconsistent."""
    if cfg.get('format_version') != FORMAT_VERSION:
        raise ValueError(
            f'Unsupported config format_version: {cfg.get("format_version")}'
        )
    aff = cfg['affinity']
    expected = len(aff['ranges']) * len(aff['directions'])
    if len(aff['offsets']) != expected:
        raise ValueError(
            f'offsets has {len(aff["offsets"])} entries, expected {expected}'
        )
    if len(aff['direction_names']) != len(aff['directions']):
        raise ValueError('direction_names and directions differ in length')
    return cfg


def get_companions(cfg):
    """The companions of a model (older configs without the key get the defaults)."""
    return {**DEFAULT_COMPANIONS, **(cfg.get('companions') or {})}


def save_config(path, cfg):
    validate_config(cfg)
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)


def load_config(path):
    with open(path) as f:
        return validate_config(json.load(f))


def save_bundle(model, directory, cfg):
    """Write model.keras (architecture + weights) and config.json side by side."""
    os.makedirs(directory, exist_ok=True)
    model.save(os.path.join(directory, 'model.keras'))
    save_config(os.path.join(directory, 'config.json'), cfg)


def load_bundle(directory):
    """Return (model, config). The model is loaded without its training loss."""
    import keras

    cfg = load_config(os.path.join(directory, 'config.json'))
    model = keras.saving.load_model(
        os.path.join(directory, 'model.keras'), compile=False
    )
    return model, cfg


def check_bundle(model, cfg):
    """
    Run one blank patch through the model and check that the number of
    affinity channels matches the config. Call once right after loading.
    """
    import numpy as np

    p, c = cfg['input']['patch_size'], cfg['input']['channels']
    out = model(np.zeros((1, p, p, c), dtype='float32'), training=False)
    if not isinstance(out, dict):  # legacy model returns (semantic, affinity)
        out = {'semantic_head': out[0], 'affinity_head': out[1]}
    n_model = int(out[cfg['outputs']['affinity']].shape[-1])
    n_cfg = len(cfg['affinity']['offsets'])
    if n_model != n_cfg:
        raise ValueError(
            f'Model has {n_model} affinity channels, config lists {n_cfg}'
        )
