"""
The settings of a batch run, as a json file next to its results.

    <results folder>/batch_settings.json

It is written before the first image is processed, so it is there even if the
run is interrupted, and it lets a large batch be continued later with exactly
the same settings (or repeated on another folder). It records:

    stages, morph_labels, morph_signal      what the Batch tab was set to
    sp_params, gasp_params, morph_params    the settings of the Superpixel, GASP
                                            and Morphology tabs
    patch_size, stride, device              prediction
    model                                   weights name and model config
    classifier, error_model                 which ones were used (names only)

Loading a file replaces the settings of the Superpixel, GASP and Morphology
tabs and the prediction patch size / stride for the run; the model, the cell
classifier and the hardware are always the ones currently loaded. When a file
is saved over an older one that differs, the older one is kept as
batch_settings_<date>.json.

No napari or Qt in this module.
"""

import copy
import json
from datetime import datetime
from pathlib import Path

FILENAME = 'batch_settings.json'
FORMAT_VERSION = 1
# what a loaded file replaces in the run's settings
OVERRIDE_KEYS = (
    'stages',
    'morph_labels',
    'morph_signal',
    'sp_params',
    'gasp_params',
    'morph_params',
    'patch_size',
    'stride',
)
REQUIRED = (*OVERRIDE_KEYS, 'model')


def _plugin_version():
    try:
        from w_seg_napari._version import version

        return str(version)
    except ImportError:
        return 'unknown'


def settings_document(
    settings: dict,
    model_name='',
    classifier='',
    error_model_path=None,
):
    """The json document for a batch run's settings (plain values only)."""
    doc = {
        'format_version': FORMAT_VERSION,
        'saved': datetime.now().isoformat(timespec='seconds'),
        'plugin_version': _plugin_version(),
        'input_mode': settings.get('input_mode', 'images'),
        'skip_done': bool(settings.get('skip_done', False)),
        'model': {
            'name': str(model_name),
            'config': settings.get('model_cfg'),
        },
        'classifier': str(classifier),
        'error_model': str(error_model_path) if error_model_path else None,
        'device': str(settings.get('device', '')),
    }
    for key in OVERRIDE_KEYS:
        doc[key] = copy.deepcopy(settings[key])
    # round trip through json now, so that a value that cannot be saved is
    # found before the run starts, not after the first image
    return json.loads(json.dumps(doc, default=_default))


def _default(obj):
    if hasattr(obj, 'tolist'):  # numpy values
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f'Cannot save a setting of type {type(obj).__name__}')


def _same(a, b):
    drop = ('saved',)
    strip = lambda d: {k: v for k, v in d.items() if k not in drop}  # noqa: E731
    return strip(a) == strip(b)


def save_settings(out_dir, doc) -> Path:
    """
    Write batch_settings.json into the results folder (created if needed).
    A different older file is kept as batch_settings_<date>.json.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FILENAME
    if path.is_file():
        try:
            old = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            old = None
        if old is not None and _same(old, doc):
            return path  # nothing changed: keep the original date
        stamp = str(old.get('saved', '') if old else '')
        stamp = (
            stamp.replace(':', '').replace('-', '').replace('T', '-')
            or datetime.now().strftime('%Y%m%d-%H%M%S')
        )
        archive = out_dir / f'batch_settings_{stamp}.json'
        n = 2
        while archive.exists():
            archive = out_dir / f'batch_settings_{stamp}_{n}.json'
            n += 1
        path.replace(archive)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(doc, indent=2), encoding='utf-8')
    tmp.replace(path)
    return path


def load_settings(path) -> dict:
    """Read and check a settings file. Raises ValueError with a clear message."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding='utf-8'))
    except OSError as e:
        raise ValueError(f'Cannot read {path}: {e}') from e
    except ValueError as e:
        raise ValueError(f'{path.name} is not a valid json file: {e}') from e
    if not isinstance(doc, dict) or doc.get('format_version') is None:
        raise ValueError(f'{path.name} is not a w-seg batch settings file.')
    if doc['format_version'] > FORMAT_VERSION:
        raise ValueError(
            f'{path.name} was written by a newer version of the plugin '
            f'(format {doc["format_version"]}); update the plugin to use it.'
        )
    missing = [k for k in REQUIRED if k not in doc]
    if missing:
        raise ValueError(f'{path.name} lacks: {", ".join(missing)}.')
    if not isinstance(doc['stages'], dict):
        raise ValueError(f'{path.name}: "stages" must be a table of on/off values.')
    return doc


def apply_loaded(settings: dict, doc: dict) -> dict:
    """A copy of the run's settings with the loaded file's values in place."""
    out = dict(settings)
    for key in OVERRIDE_KEYS:
        out[key] = copy.deepcopy(doc[key])
    return out


def model_differences(doc, model_name, model_cfg) -> list[str]:
    """What differs between the model of the file and the loaded one (empty: same)."""
    saved = doc.get('model') or {}
    diffs = []
    if saved.get('name') and model_name and saved['name'] != model_name:
        diffs.append(f'weights: "{saved["name"]}" saved, "{model_name}" loaded')
    cfg = saved.get('config')
    if cfg is not None and model_cfg is not None:
        norm = lambda x: json.dumps(  # noqa: E731
            json.loads(json.dumps(x, default=_default)), sort_keys=True
        )
        if norm(cfg) != norm(model_cfg):
            diffs.append('the model configuration (outputs / preprocessing) differs')
    return diffs


def describe(doc) -> str:
    """One or two lines for the Batch tab."""
    on = [k for k, v in doc['stages'].items() if v]
    sp = doc['sp_params'].get('algo_name', '?')
    gasp = doc['gasp_params'].get('linkage', '?')
    px = doc['morph_params'].get('pixel_spacing', '?')
    return (
        f'saved {doc.get("saved", "?")} (plugin {doc.get("plugin_version", "?")}); '
        f'stages: {", ".join(on) or "none"}; superpixels: {sp}; '
        f'GASP: {gasp}; pixel size {px}'
    )
