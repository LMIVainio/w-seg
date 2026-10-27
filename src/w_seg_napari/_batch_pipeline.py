"""
Headless pipeline for the batch tab.

The batch tab takes a snapshot of the settings in tabs 1 and 4 on the main
thread (see ``collect_batch_settings`` in _widget.py) and hands it to
``run_batch``, a generator that runs in a background thread. It yields small
event tuples:

    ('progress', percent, message)   update the progress bar
    ('render', image_name, results)  show the results of one image in the viewer
    ('failed', image_name, message)  one image could not be processed

and returns (number of images processed, [(image name, error), ...]).

Every stage uses the same functions as the interactive widgets, with the
settings the widgets had when the batch was started.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from w_seg_napari._affinity import gather, layer_payloads
from w_seg_napari._batch_io import (
    load_image,
    output_dir,
    session_path,
    table_path,
    to_model_input_range,
)

STAGES = (
    'predict',
    'superpixels',
    'gasp',
    'morphology',
    'classification',
    'export',
)

# errors that mean "this image failed", not "the program is broken"
IMAGE_ERRORS = (RuntimeError, ValueError, TypeError, OSError, KeyError)


class _AbortShim:
    """What measure_cell_morphology expects of its worker (it checks for an abort)."""

    def __init__(self, is_cancelled):
        self._is_cancelled = is_cancelled

    @property
    def abort_requested(self):
        return bool(self._is_cancelled())

    @property
    def _abort_requested(self):
        return bool(self._is_cancelled())


# ---------------------------------------------------------------------
# Session files (what a batch run leaves next to the tables)
# ---------------------------------------------------------------------
def save_session(path, arrays: dict, attrs: dict):
    import h5py

    with h5py.File(path, 'w') as f:
        for key, val in arrays.items():
            f.create_dataset(key, data=val, compression='gzip')
        for key, val in attrs.items():
            f.attrs[key] = val


def load_session(path) -> dict:
    import h5py

    with h5py.File(path, 'r') as f:
        return {key: f[key][:] for key in f}


def _require(store: dict, *keys, stage: str):
    missing = [k for k in keys if k not in store]
    if missing:
        raise ValueError(
            f"Stage '{stage}' is switched off and no saved result for "
            f'{", ".join(missing)} exists. Run the earlier stages first.'
        )


def affinity_set(aff, cfg, source):
    """AffinitySet for a stitched prediction (the same objects the widgets build from layers)."""
    layers = [
        SimpleNamespace(name=p['name'], data=p['data'], metadata=p['metadata'])
        for p in layer_payloads(aff, cfg, source)
    ]
    return gather(layers, source)


def _signal(kind, sem, rgb):
    """The map an optional 'Signal layer' stands for (see SuperpixelWidget.snapshot_params)."""
    if kind is None:
        return None
    if kind == 'prediction':
        return sem
    from skimage.color import rgb2gray

    return rgb2gray(to_model_input_range(rgb)).astype(np.float32)


# ---------------------------------------------------------------------
# One image
# ---------------------------------------------------------------------
def process_image(image_path, settings: dict, is_cancelled):
    """
    Generator for one image. Yields ('sub', message, fraction_done) and returns
    a dict with the results (None when cancelled).
    """
    from w_seg_napari._gasp_widget import GaspWidget
    from w_seg_napari._morphology_widget import measure_cell_morphology
    from w_seg_napari._superpixel_widget import SuperpixelWidget
    from w_seg_napari._wsegutils import process_large_image

    name = Path(image_path).name
    source = Path(name).stem
    stages = settings['stages']
    cfg = settings['model_cfg']
    n_stages = max(1, sum(bool(stages.get(s)) for s in STAGES))
    done = 0

    def sub(message):
        return ('sub', message, done / n_stages)

    sess_path = session_path(settings['out_dir'], name)
    store = {}
    if (
        not all(stages.get(s) for s in ('predict', 'superpixels', 'gasp'))
        and sess_path.is_file()
    ):
        store = load_session(sess_path)  # earlier results of skipped stages

    rgb = load_image(image_path)
    results = {'raw': rgb}

    # 1. Prediction ------------------------------------------------------
    if stages.get('predict'):
        yield sub('Predicting cell walls and affinities')
        # (3, H, W) float image; process_large_image converts it to a tensor
        tensor = np.ascontiguousarray(
            to_model_input_range(rgb).transpose(2, 0, 1)
        )
        pre = cfg['preprocessing']
        sem, aff = yield from process_large_image(
            image=tensor,
            model=settings['model'],
            patch_size=settings['patch_size'],
            stride=settings['stride'],
            device=settings['device'],
            live_view=False,
            is_cancelled=is_cancelled,
            hue_encoding=pre['hue_encoding'],
            enable_lab=pre['enable_lab'],
            output_names=(cfg['outputs']['wall'], cfg['outputs']['affinity']),
        )
        if sem is None:
            return None
        store['pred_sem'] = np.squeeze(sem).astype(np.float32)
        store['pred_aff'] = np.clip(aff, 0.0, 1.0).astype(np.float32)
        done += 1
    else:
        _require(store, 'pred_sem', 'pred_aff', stage='predict')
    sem = store['pred_sem']
    aset = affinity_set(store['pred_aff'], cfg, source)
    results['pred_sem'] = sem
    results['affinity_layers'] = layer_payloads(store['pred_aff'], cfg, source)

    if is_cancelled():
        return None

    # 2. Superpixels -----------------------------------------------------
    if stages.get('superpixels'):
        yield sub('Superpixels')
        p = settings['sp_params']
        store['sp_labels'] = SuperpixelWidget.run_algorithm(
            p,
            sem_data=sem,
            aset=aset,
            felz_signal=_signal(p['felz_signal_kind'], sem, rgb),
            smutex_sem=_signal(p['smutex_signal_kind'], sem, rgb),
            mutex_sem=_signal(p['mutex_signal_kind'], sem, rgb),
        )
        done += 1
    if 'sp_labels' in store:
        results['sp_labels'] = store['sp_labels']

    if is_cancelled():
        return None

    # 3. GASP ------------------------------------------------------------
    if stages.get('gasp'):
        yield sub('GASP')
        _require(store, 'sp_labels', stage='superpixels')
        store['gasp_labels'] = GaspWidget.run_gasp(
            settings['gasp_params'], aset, store['sp_labels']
        )
        done += 1
    if 'gasp_labels' in store:
        results['gasp_labels'] = store['gasp_labels']

    if is_cancelled():
        return None

    # 4. Morphology ------------------------------------------------------
    df = None
    labels_measured = None
    if stages.get('morphology'):
        key = (
            'gasp_labels'
            if settings['morph_labels'] == 'gasp'
            else 'sp_labels'
        )
        _require(
            store, key, stage='gasp' if key == 'gasp_labels' else 'superpixels'
        )
        wall_signal = (
            sem
            if settings['morph_signal'] == 'prediction'
            else _signal('image', sem, rgb)
        )
        params = dict(settings['morph_params'])
        params['center_coords'] = (
            None  # found per image when radial symmetry is on
        )
        gen = measure_cell_morphology(
            labels_layer=store[key],
            cell_wall_signal=wall_signal,
            yolo_image=rgb,
            worker=_AbortShim(is_cancelled),
            verbose=False,
            **params,
        )
        try:
            while True:
                step = next(gen)
                completed, total, msg = step
                yield (
                    'sub',
                    msg,
                    (done + completed / max(total, 1)) / n_stages,
                )
        except StopIteration as stop:
            df, labels_measured, _class_names = stop.value
        if is_cancelled():
            return None
        done += 1
        df = df.copy()
        df.insert(0, 'source_file', name)

    # 5. Classification ----------------------------------------------------
    if stages.get('classification') and df is not None:
        yield sub('Classifying cells')
        clf = settings['classifier']
        if clf is None or not clf.is_trained:
            raise ValueError('No trained or loaded cell classifier.')
        df, cell_ids, preds = clf.predict(df)
        store['class_labels'] = clf.generate_prediction_layer(
            labels_measured, cell_ids, preds
        )
        done += 1
    if 'class_labels' in store and stages.get('classification'):
        results['class_labels'] = store['class_labels']

    # 6. Export ----------------------------------------------------------
    out_dir = settings['out_dir']
    if stages.get('export') and df is not None:
        yield sub('Writing table')
        df.to_csv(table_path(out_dir, name), index=False)
        done += 1
    save_session(
        sess_path,
        store,
        {
            'source_file': name,
            'model_config_json': json.dumps(cfg),
            'fe_config_json': json.dumps(
                settings['morph_params'], default=str
            ),
        },
    )
    results['table'] = df
    return results


# ---------------------------------------------------------------------
# A folder
# ---------------------------------------------------------------------
def run_batch(files, dirpath, settings: dict, is_cancelled):
    """Process the image files of a folder one after the other (see module doc)."""
    settings = dict(settings)
    out = output_dir(dirpath)
    out.mkdir(parents=True, exist_ok=True)
    settings['out_dir'] = out

    total = len(files)
    n_ok, failed = 0, []
    for i, name in enumerate(files):
        if is_cancelled():
            break
        yield ('progress', int(100 * i / total), f'({i + 1}/{total}) {name}')

        gen = process_image(Path(dirpath) / name, settings, is_cancelled)
        results = None
        try:
            while True:
                _, message, frac = next(gen)
                yield (
                    'progress',
                    int(100 * (i + frac) / total),
                    f'({i + 1}/{total}) {name}: {message}',
                )
        except StopIteration as stop:
            results = stop.value
        except IMAGE_ERRORS as e:
            failed.append((name, f'{type(e).__name__}: {e}'))
            yield ('failed', name, str(e))
            continue

        if results is None:  # cancelled while working on this image
            break
        n_ok += 1
        if settings.get('render'):
            yield ('render', name, results)

    yield ('progress', 100, 'Done' if not is_cancelled() else 'Cancelled')
    return n_ok, failed
