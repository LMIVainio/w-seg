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
    find_session_dir,
    load_image,
    output_dir,
    session_path,
    table_path,
    to_model_input_range,
    to_rgb,
)
from w_seg_napari._batch_settings import save_settings as save_settings_file
from w_seg_napari._geometry_fields import geometry_payloads, masked_direction
from w_seg_napari._session_io import (
    CORRECTED,
    has_dataset,
    load_session,
    read_attrs,
    save_session,
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


# Session files live in _session_io (save_session / load_session).


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


NO_GEOMETRY = (
    'The seeded EDT watershed with this seed source needs the geometry head, '
    'and no geometry prediction is available (the model has none, or the '
    'prediction stage was skipped and the session has no geometry). Choose a '
    'model with a geometry head, run the predict stage, or pick another '
    'algorithm.'
)


_SEED_NAMES = {
    'skeleton': 'predicted skeleton (geo_skel)',
    'edge_distance': 'predicted edge distance (geo_dist)',
    'prediction': 'predicted wall map (pred_)',
}
_RIDGE_NAMES = {
    'convergence': 'ridges where the direction field converges',
    'sato': 'Sato ridges of the edge distance',
    'both': 'convergence x Sato ridges',
}
NEEDS_GEOMETRY_HEAD = (
    'needs a model with a geometry head, and the loaded model has none'
)


def describe_seed_source(params, has_geometry=None):
    """
    What the batch will use as seeds, as shown in the Batch tab.

    params         the Superpixel tab's snapshot_params()
    has_geometry   True / False when the model is known, None when it is not
    Returns {'text': str or None, 'problem': str or None}. text is None when
    the algorithm has no seeds (only the seeded EDT watershed does); problem
    is set when the batch could not run with these settings. It mirrors
    seed_arguments, which does the real work for every image.
    """
    if params.get('algo_idx') != 5:
        return {'text': None, 'problem': None}
    ridge = params.get('ridge_method')
    kind = params.get('seed_kind')
    layer = params.get('seed_layer') or 'no layer'
    if ridge:
        what = _RIDGE_NAMES.get(ridge, ridge)
        uses_geo = True
    elif kind in ('skeleton', 'edge_distance'):
        what = _SEED_NAMES[kind]
        uses_geo = True
    elif kind == 'prediction':
        what = _SEED_NAMES[kind]
        uses_geo = False
    else:
        return {
            'text': f"layer '{layer}'",
            'problem': (
                f"The seed layer '{layer}' exists for one image only, so a "
                'batch cannot use it. Choose a geo_skel_ or geo_dist_ layer, '
                'a pred_ layer or a ridge source in the Superpixel tab.'
            ),
        }
    detail = f'{what}, threshold {params.get("seed_threshold", 0.5):g}'
    if not ridge and kind in _SEED_NAMES and layer != 'no layer':
        detail += f" (from layer '{layer}')"
    text = f'{detail}; computed for every image'
    problem = None
    if uses_geo and has_geometry is False:
        problem = f'These seeds ({what}) {NEEDS_GEOMETRY_HEAD}.'
    return {'text': text, 'problem': problem}


def seed_arguments(params, store, sem):
    """
    Extra inputs of the seeded EDT watershed (algorithm 5) for one image, taken
    from the stored predictions the same way the interactive widget takes them
    from the viewer layers. Other algorithms get nothing extra.
    """
    if params.get('algo_idx') != 5:
        return {}
    geo = store.get('pred_geo')
    if params.get('ridge_method'):
        if geo is None:
            raise ValueError(NO_GEOMETRY)
        return {'edge_distance': geo[0], 'direction': masked_direction(geo)}
    kind = params.get('seed_kind')
    if kind in ('skeleton', 'edge_distance'):
        if geo is None:
            raise ValueError(NO_GEOMETRY)
        return {'seed_data': geo[3] if kind == 'skeleton' else geo[0]}
    if kind == 'prediction':
        return {'seed_data': sem}
    raise ValueError(
        'The selected seed layer exists for one image only, so a batch cannot '
        'use it. Choose a geo_skel_ or geo_dist_ layer, a pred_ layer or one '
        'of the ridge sources.'
    )


# ---------------------------------------------------------------------
# Resuming: which images are already done
# ---------------------------------------------------------------------
# the dataset of the session that each stage leaves behind
STAGE_RESULTS = {
    'predict': 'pred_sem',
    'superpixels': 'sp_labels',
    'gasp': 'gasp_labels',
    'classification': 'class_labels',
}


def is_done(image_name, out_dir, stages: dict) -> bool:
    """
    True when the outputs of the selected stages already exist for this
    image: the session holds every result those stages make, and the table is
    there if the table is exported. (The session is written last, so a run
    that was interrupted does not count as done.)
    """
    sess = session_path(out_dir, image_name)
    if not sess.is_file():
        return False
    if stages.get('export') and not table_path(out_dir, image_name).is_file():
        return False
    try:
        return all(
            has_dataset(sess, key)
            for stage, key in STAGE_RESULTS.items()
            if stages.get(stage)
        )
    except OSError:  # unreadable file: process it again
        return False


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
    sessions_mode = settings.get('input_mode') == 'sessions'
    all_labels = all(stages.get(s) for s in ('predict', 'superpixels', 'gasp'))
    store = {}
    if sessions_mode or (not all_labels and sess_path.is_file()):
        store = load_session(sess_path)  # earlier results of skipped stages
    notes = []
    recompute_labels = bool(stages.get('superpixels') or stages.get('gasp'))
    if (
        sess_path.is_file()
        and recompute_labels
        and has_dataset(sess_path, CORRECTED)
    ):
        notes.append(
            f'{name}: new labels were computed, so the corrected labels '
            'saved earlier in its session were discarded.'
        )
    if recompute_labels:
        store.pop(CORRECTED, None)
    if any(stages.get(s) for s in ('predict', 'superpixels', 'gasp')):
        store.pop('error_map', None)  # belongs to the old labels

    source_name = name
    if sessions_mode:
        # the image lives in the session (a hand-made session has no predictions)
        if 'raw' not in store:
            raise ValueError('This session holds no image (raw).')
        rgb = to_rgb(store['raw'])
        if all_labels:
            store = {}  # everything is recomputed from the image
        try:
            source_name = read_attrs(sess_path).get('source_file') or name
        except (OSError, KeyError):
            pass
    else:
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
        geometry_name = cfg['outputs'].get('geometry')  # wasp_v4 models only
        out = yield from process_large_image(
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
            geometry_name=geometry_name,
        )
        sem, aff, geo = out if geometry_name else (*out, None)
        if sem is None:
            return None
        if geo is not None:
            store['pred_geo'] = np.asarray(geo, dtype=np.float32)
        else:
            store.pop('pred_geo', None)  # an old session's geometry is stale
        store['pred_sem'] = np.squeeze(sem).astype(np.float32)
        store['pred_aff'] = np.clip(aff, 0.0, 1.0).astype(np.float32)
        done += 1
    else:
        _require(store, 'pred_sem', 'pred_aff', stage='predict')
    sem = store['pred_sem']
    aset = affinity_set(store['pred_aff'], cfg, source)
    results['pred_sem'] = sem
    results['affinity_layers'] = layer_payloads(store['pred_aff'], cfg, source)
    if 'pred_geo' in store:
        results['geometry_layers'] = geometry_payloads(
            store['pred_geo'], source
        )

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
            **seed_arguments(p, store, sem),
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
        wanted = settings['morph_labels']
        if wanted == 'corrected':
            if CORRECTED in store:
                key = CORRECTED
            else:
                key = 'gasp_labels'
                notes.append(
                    f'{name}: no corrected labels saved, measured the GASP '
                    'labels instead.'
                )
        else:
            key = 'gasp_labels' if wanted == 'gasp' else 'sp_labels'
        _require(
            store,
            key,
            stage='gasp' if key == 'gasp_labels' else 'superpixels',
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
        df.insert(0, 'source_file', source_name)
        df.insert(
            1,
            'labels_source',
            {CORRECTED: 'corrected', 'gasp_labels': 'gasp'}.get(
                key, 'superpixels'
            ),
        )

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

    # 5b. Likely segmentation errors (optional model from "Train error model")
    bundle = settings.get('error_bundle')
    if stages.get('morphology'):
        store.pop('error_map', None)
    if bundle is not None and df is not None:
        yield sub('Flagging likely segmentation errors')
        from w_seg_napari._segmentation_errors import flag_labels

        scores, emap = flag_labels(
            store[key],
            sem,
            bundle,
            settings['morph_params'].get('pixel_spacing') or 1.0,
        )
        df = df.merge(scores, on='label_id', how='left')
        store['error_map'] = emap
        results['error_map'] = emap
        results['n_flagged'] = int(scores['flagged'].sum())

    # 6. Export ----------------------------------------------------------
    out_dir = settings['out_dir']
    if stages.get('export') and df is not None:
        yield sub('Writing table')
        df.to_csv(table_path(out_dir, name), index=False)
        done += 1
    # Keep the original image in the session so manual corrections can be made
    # later against the exact input the labels were computed from.
    store['raw'] = rgb
    save_session(
        sess_path,
        store,
        {
            'source_file': source_name,
            'model_config_json': json.dumps(cfg),
            'fe_config_json': json.dumps(
                settings['morph_params'], default=str
            ),
        },
    )
    results['table'] = df
    results['notes'] = notes
    return results


# ---------------------------------------------------------------------
# A folder
# ---------------------------------------------------------------------
def run_batch(files, dirpath, settings: dict, is_cancelled):
    """Process the image files of a folder one after the other (see module doc)."""
    settings = dict(settings)
    if settings.get('input_mode') == 'sessions':
        out = find_session_dir(dirpath)
    else:
        out = output_dir(dirpath)
    out.mkdir(parents=True, exist_ok=True)
    settings['out_dir'] = out

    # the settings come first, before any image is touched
    if settings.get('save_settings') and settings.get('settings_doc'):
        path = save_settings_file(out, settings['settings_doc'])
        yield ('progress', 0, f'Settings saved to {path.name}')

    total = len(files)
    n_ok, failed = 0, []
    for i, name in enumerate(files):
        if is_cancelled():
            break
        yield ('progress', int(100 * i / total), f'({i + 1}/{total}) {name}')
        if settings.get('skip_done') and is_done(
            name, out, settings['stages']
        ):
            yield ('skipped', name)
            continue

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
        for note in results.get('notes', []):
            yield ('note', note)
        if 'n_flagged' in results:
            yield (
                'note',
                (
                    f'{name}: {results["n_flagged"]} cell(s) flagged as '
                    'likely segmentation errors (columns p_error, '
                    'error_type).'
                ),
            )
        if settings.get('render'):
            yield ('render', name, results)

    yield ('progress', 100, 'Done' if not is_cancelled() else 'Cancelled')
    return n_ok, failed
