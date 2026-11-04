"""Batch orchestration with the heavy parts (model, algorithms) replaced by fakes."""

import sys
import types

import numpy as np
import pandas as pd
import tifffile

import w_seg_napari._batch_pipeline as bp
from w_seg_napari._model_config import legacy_config, make_config

CALLS = []
SEED_ARGS = {}
SEED_KEYS = ('seed_data', 'edge_distance', 'direction')


def _install_fakes(monkeypatch):
    CALLS.clear()

    def process_large_image(image, model, is_cancelled=None, **kw):
        CALLS.append(('predict', image.shape))
        geometry = kw.get('geometry_name')
        if is_cancelled and is_cancelled():
            return (None, None, None) if geometry else (None, None)
        _, h, w = image.shape
        n_aff = 8 if geometry else 2
        out = (
            np.full((1, h, w), 0.5, np.float32),
            np.full((n_aff, h, w), 0.9, np.float32),
        )
        if geometry:
            geo = np.zeros((4, h, w), np.float32)
            geo[0] = 0.8  # edge distance
            geo[1] = 1.0  # dir_y
            geo[3] = 0.6  # skeleton
            out = (*out, geo)
        return out
        yield  # makes this a generator, like the real function

    class SP:
        @classmethod
        def run_algorithm(cls, params, sem_data=None, aset=None, **kw):
            CALLS.append(('superpixels', aset.describe()))
            SEED_ARGS.clear()
            SEED_ARGS.update({k: v for k, v in kw.items() if k in SEED_KEYS})
            lab = np.ones(sem_data.shape, np.uint64)
            lab[:, sem_data.shape[1] // 2 :] = 2
            return lab

    class GASP:
        @classmethod
        def run_gasp(cls, params, aset, sp):
            CALLS.append(('gasp',))
            return sp

    def measure(labels_layer, worker=None, **kw):
        CALLS.append(('morphology', worker.abort_requested, labels_layer))
        yield 1, 2, 'half way'
        df = pd.DataFrame({'label_id': [1, 2], 'area': [10.0, 20.0]})
        return df, labels_layer, {}

    for name, attrs in {
        'w_seg_napari._wsegutils': {
            'process_large_image': process_large_image
        },
        'w_seg_napari._superpixel_widget': {'SuperpixelWidget': SP},
        'w_seg_napari._gasp_widget': {'GaspWidget': GASP},
        'w_seg_napari._morphology_widget': {
            'measure_cell_morphology': measure
        },
    }.items():
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)

    # in-memory session files (the h5 reading/writing is plain h5py)
    sessions = {}

    def fake_save(path, arrays, attrs):
        sessions[str(path)] = dict(arrays)
        path.write_text('session')  # the file has to exist, like the real one

    monkeypatch.setattr(bp, 'save_session', fake_save)
    monkeypatch.setattr(
        bp, 'load_session', lambda path: dict(sessions[str(path)])
    )
    monkeypatch.setattr(
        bp,
        'has_dataset',
        lambda path, key: key in sessions.get(str(path), {}),
    )
    return sessions


def _settings(**stages):
    all_on = dict.fromkeys(bp.STAGES, True)
    all_on['classification'] = False
    all_on.update(stages)
    return {
        'stages': all_on,
        'render': False,
        'morph_labels': 'gasp',
        'morph_signal': 'prediction',
        'model': object(),
        'model_cfg': legacy_config(),
        'device': 'cpu',
        'patch_size': 256,
        'stride': 128,
        'sp_params': {
            'felz_signal_kind': None,
            'smutex_signal_kind': None,
            'mutex_signal_kind': None,
        },
        'gasp_params': {},
        'morph_params': {'pixel_spacing': 1.0},
        'classifier': None,
    }


def _make_images(folder, names):
    for n in names:
        tifffile.imwrite(folder / n, np.zeros((16, 20, 3), np.uint8))


def _drain(gen):
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as stop:
        return events, stop.value


def test_full_run_writes_table_and_reports_progress(tmp_path, monkeypatch):
    _install_fakes(monkeypatch)
    _make_images(tmp_path, ['a.tif', 'b.tif'])
    events, (n_ok, failed) = _drain(
        bp.run_batch(['a.tif', 'b.tif'], tmp_path, _settings(), lambda: False)
    )
    assert n_ok == 2 and failed == []
    assert (tmp_path / 'wseg_batch' / 'a_morphology.csv').is_file()
    assert (tmp_path / 'wseg_batch' / 'b_morphology.csv').is_file()
    percents = [e[1] for e in events if e[0] == 'progress']
    assert percents == sorted(percents) and percents[-1] == 100
    stage_order = [c[0] for c in CALLS[:4]]
    assert stage_order == ['predict', 'superpixels', 'gasp', 'morphology']
    table = pd.read_csv(tmp_path / 'wseg_batch' / 'a_morphology.csv')
    assert list(table['source_file'].unique()) == ['a.tif']


def test_one_bad_image_does_not_stop_the_batch(tmp_path, monkeypatch):
    _install_fakes(monkeypatch)
    _make_images(tmp_path, ['a.tif'])
    (tmp_path / 'bad.tif').write_text('not an image')
    events, (n_ok, failed) = _drain(
        bp.run_batch(
            ['a.tif', 'bad.tif'], tmp_path, _settings(), lambda: False
        )
    )
    assert n_ok == 1
    assert [f[0] for f in failed] == ['bad.tif']
    assert any(e[0] == 'failed' for e in events)


def test_cancel_stops_before_the_next_image(tmp_path, monkeypatch):
    _install_fakes(monkeypatch)
    _make_images(tmp_path, ['a.tif', 'b.tif', 'c.tif'])
    first_result = tmp_path / 'wseg_batch' / 'a_morphology.csv'

    # "the user presses cancel" as soon as the first image is finished
    _, (n_ok, failed) = _drain(
        bp.run_batch(
            ['a.tif', 'b.tif', 'c.tif'],
            tmp_path,
            _settings(),
            lambda: first_result.exists(),
        )
    )
    assert n_ok == 1 and failed == []
    assert not (tmp_path / 'wseg_batch' / 'b_morphology.csv').exists()


def test_cancel_inside_an_image_discards_it(tmp_path, monkeypatch):
    _install_fakes(monkeypatch)
    _make_images(tmp_path, ['a.tif'])
    _, (n_ok, _) = _drain(
        bp.run_batch(
            ['a.tif'],
            tmp_path,
            _settings(),
            lambda: any(c[0] == 'gasp' for c in CALLS),
        )
    )
    assert n_ok == 0
    assert not (tmp_path / 'wseg_batch' / 'a_morphology.csv').exists()


def test_skipped_stages_use_the_saved_session(tmp_path, monkeypatch):
    sessions = _install_fakes(monkeypatch)
    _make_images(tmp_path, ['a.tif'])
    _drain(bp.run_batch(['a.tif'], tmp_path, _settings(), lambda: False))
    CALLS.clear()
    s = _settings(predict=False, superpixels=False, gasp=False)
    _drain(bp.run_batch(['a.tif'], tmp_path, s, lambda: False))
    assert [c[0] for c in CALLS] == ['morphology']
    assert sessions  # the session file is kept up to date


def test_skipped_stage_without_saved_result_is_an_error(tmp_path, monkeypatch):
    _install_fakes(monkeypatch)
    _make_images(tmp_path, ['a.tif'])
    s = _settings(predict=False)
    _, (n_ok, failed) = _drain(
        bp.run_batch(['a.tif'], tmp_path, s, lambda: False)
    )
    assert n_ok == 0 and 'no saved result' in failed[0][1]


def test_render_event_carries_the_layers(tmp_path, monkeypatch):
    _install_fakes(monkeypatch)
    _make_images(tmp_path, ['a.tif'])
    s = _settings()
    s['render'] = True
    events, _ = _drain(bp.run_batch(['a.tif'], tmp_path, s, lambda: False))
    render = [e for e in events if e[0] == 'render'][0]
    assert render[1] == 'a.tif'
    assert {'raw', 'pred_sem', 'sp_labels', 'gasp_labels'} <= set(render[2])
    assert render[2]['affinity_layers']


def _geometry_settings(**sp):
    s = _settings()
    s['model_cfg'] = make_config(ranges=(6, 32), geometry=True)
    s['sp_params'] = {
        'algo_idx': 5,
        'felz_signal_kind': None,
        'smutex_signal_kind': None,
        'mutex_signal_kind': None,
        'ridge_method': None,
        'seed_kind': 'skeleton',
        **sp,
    }
    return s


def test_geometry_model_feeds_the_seeded_watershed(tmp_path, monkeypatch):
    sessions = _install_fakes(monkeypatch)
    _make_images(tmp_path, ['a.tif'])

    # seeds from the skeleton channel
    _, (n_ok, failed) = _drain(
        bp.run_batch(['a.tif'], tmp_path, _geometry_settings(), lambda: False)
    )
    assert n_ok == 1 and failed == []
    assert SEED_ARGS['seed_data'].shape == (16, 20)
    assert np.allclose(SEED_ARGS['seed_data'], 0.6)  # the skeleton channel
    # the geometry is kept in the session for later stages
    session = next(iter(sessions.values()))
    assert session['pred_geo'].shape == (4, 16, 20)

    # edge distance as seed layer
    _drain(
        bp.run_batch(
            ['a.tif'],
            tmp_path,
            _geometry_settings(seed_kind='edge_distance'),
            lambda: False,
        )
    )
    assert np.allclose(SEED_ARGS['seed_data'], 0.8)

    # ridge sources get edge distance and direction (zero where unset)
    _drain(
        bp.run_batch(
            ['a.tif'],
            tmp_path,
            _geometry_settings(ridge_method='convergence'),
            lambda: False,
        )
    )
    assert SEED_ARGS['edge_distance'].shape == (16, 20)
    assert SEED_ARGS['direction'].shape == (16, 20, 2)
    assert np.allclose(SEED_ARGS['direction'][..., 0], 1.0)


def test_seeded_watershed_errors_are_clear_per_image(tmp_path, monkeypatch):
    _install_fakes(monkeypatch)
    _make_images(tmp_path, ['a.tif'])
    # a seed layer that only exists for one image
    _, (n_ok, failed) = _drain(
        bp.run_batch(
            ['a.tif'],
            tmp_path,
            _geometry_settings(seed_kind='other'),
            lambda: False,
        )
    )
    assert n_ok == 0 and 'one image only' in failed[0][1]

    # a model without geometry head cannot give geometry seeds
    s = _geometry_settings()
    s['model_cfg'] = legacy_config()
    _, (n_ok, failed) = _drain(
        bp.run_batch(['a.tif'], tmp_path, s, lambda: False)
    )
    assert n_ok == 0 and 'geometry head' in failed[0][1]

    # ... but a prediction map as seeds works with any model
    s = _geometry_settings(seed_kind='prediction')
    s['model_cfg'] = legacy_config()
    _, (n_ok, failed) = _drain(
        bp.run_batch(['a.tif'], tmp_path, s, lambda: False)
    )
    assert n_ok == 1 and failed == []


def test_describe_seed_source():
    from w_seg_napari._batch_pipeline import describe_seed_source

    base = {'algo_idx': 5, 'seed_threshold': 0.5}
    # other algorithms have no seed line
    assert describe_seed_source({'algo_idx': 1}) == {'text': None, 'problem': None}

    r = describe_seed_source({**base, 'seed_kind': 'skeleton', 'seed_layer': 'geo_skel_a'}, True)
    assert r['problem'] is None
    assert 'skeleton' in r['text'] and 'geo_skel_a' in r['text'] and 'every image' in r['text']

    r = describe_seed_source({**base, 'seed_kind': 'skeleton', 'seed_layer': 'geo_skel_a'}, False)
    assert 'geometry head' in r['problem']
    # unknown model: no complaint
    assert describe_seed_source({**base, 'seed_kind': 'edge_distance', 'seed_layer': 'g'}, None)['problem'] is None

    # a wall-prediction seed does not need the geometry head
    r = describe_seed_source({**base, 'seed_kind': 'prediction', 'seed_layer': 'pred_a'}, False)
    assert r['problem'] is None

    # ridge sources win over the layer and need the geometry head
    r = describe_seed_source({**base, 'ridge_method': 'both', 'seed_kind': 'other'}, False)
    assert 'Sato' in r['text'] and r['problem']

    # a layer that exists for one image only
    for kind in ('labels', 'other', None):
        r = describe_seed_source({**base, 'seed_kind': kind, 'seed_layer': 'my_seeds'}, True)
        assert 'my_seeds' in r['problem'] and 'one image only' in r['problem']


def test_describe_matches_seed_arguments():
    # whatever describe_seed_source accepts, seed_arguments must also accept
    import numpy as np

    from w_seg_napari._batch_pipeline import describe_seed_source, seed_arguments

    geo = np.zeros((4, 8, 8), np.float32)
    sem = np.zeros((8, 8), np.float32)
    for params in (
        {'algo_idx': 5, 'seed_kind': 'skeleton'},
        {'algo_idx': 5, 'seed_kind': 'edge_distance'},
        {'algo_idx': 5, 'seed_kind': 'prediction'},
        {'algo_idx': 5, 'ridge_method': 'sato'},
    ):
        assert describe_seed_source(params, True)['problem'] is None
        assert seed_arguments(params, {'pred_geo': geo}, sem)
    for kind in ('labels', 'other'):
        params = {'algo_idx': 5, 'seed_kind': kind}
        assert describe_seed_source(params, True)['problem']
        try:
            seed_arguments(params, {'pred_geo': geo}, sem)
        except ValueError:
            pass
        else:
            raise AssertionError('seed_arguments should refuse this')


# --------------------------------------------------------------- corrections
def _morph_only(**extra):
    s = _settings(predict=False, superpixels=False, gasp=False, **extra)
    s['morph_labels'] = 'corrected'
    return s


def _with_correction(tmp_path, sessions, name='a.tif'):
    key = str(tmp_path / 'wseg_batch' / (name.split('.')[0] + '_session.h5'))
    labels = np.full((16, 20), 7, np.int32)
    sessions[key][bp.CORRECTED] = labels
    return key, labels


def test_morphology_uses_the_corrected_labels(tmp_path, monkeypatch):
    sessions = _install_fakes(monkeypatch)
    _make_images(tmp_path, ['a.tif'])
    _drain(bp.run_batch(['a.tif'], tmp_path, _settings(), lambda: False))
    _, labels = _with_correction(tmp_path, sessions)
    CALLS.clear()
    events, _ = _drain(
        bp.run_batch(['a.tif'], tmp_path, _morph_only(), lambda: False)
    )
    assert CALLS[0][0] == 'morphology'
    assert (CALLS[0][2] == labels).all()  # the corrected labels were measured
    table = pd.read_csv(tmp_path / 'wseg_batch' / 'a_morphology.csv')
    assert list(table['labels_source'].unique()) == ['corrected']
    assert not [e for e in events if e[0] == 'note']


def test_image_without_corrections_falls_back_and_says_so(
    tmp_path, monkeypatch
):
    _install_fakes(monkeypatch)
    _make_images(tmp_path, ['a.tif'])
    _drain(bp.run_batch(['a.tif'], tmp_path, _settings(), lambda: False))
    events, _ = _drain(
        bp.run_batch(['a.tif'], tmp_path, _morph_only(), lambda: False)
    )
    notes = [e[1] for e in events if e[0] == 'note']
    assert len(notes) == 1 and 'a.tif' in notes[0]
    table = pd.read_csv(tmp_path / 'wseg_batch' / 'a_morphology.csv')
    assert list(table['labels_source'].unique()) == ['gasp']


def test_recomputing_labels_discards_stale_corrections(tmp_path, monkeypatch):
    sessions = _install_fakes(monkeypatch)
    _make_images(tmp_path, ['a.tif'])
    _drain(bp.run_batch(['a.tif'], tmp_path, _settings(), lambda: False))
    key, _ = _with_correction(tmp_path, sessions)
    events, _ = _drain(
        bp.run_batch(['a.tif'], tmp_path, _settings(), lambda: False)
    )
    assert bp.CORRECTED not in sessions[key]
    assert any('discarded' in e[1] for e in events if e[0] == 'note')


def test_morphology_only_run_keeps_corrections(tmp_path, monkeypatch):
    sessions = _install_fakes(monkeypatch)
    _make_images(tmp_path, ['a.tif'])
    _drain(bp.run_batch(['a.tif'], tmp_path, _settings(), lambda: False))
    key, _ = _with_correction(tmp_path, sessions)
    _drain(bp.run_batch(['a.tif'], tmp_path, _morph_only(), lambda: False))
    assert bp.CORRECTED in sessions[key]


# ------------------------------------------------------------ error flagging
def _fake_flagger(monkeypatch):
    seen = {}

    def flag_labels(labels, wall, bundle, pixel_size=1.0):
        seen['labels'] = labels
        scores = pd.DataFrame(
            {
                'label_id': [1, 2],
                'p_error': [0.9, 0.1],
                'error_type': ['merged', 'ok'],
                'flagged': [True, False],
            }
        )
        return scores, np.where(labels == 1, 0.9, 0.1).astype(np.float32)

    mod = types.ModuleType('w_seg_napari._segmentation_errors')
    mod.flag_labels = flag_labels
    monkeypatch.setitem(sys.modules, 'w_seg_napari._segmentation_errors', mod)
    return seen


def test_error_model_adds_columns_map_and_note(tmp_path, monkeypatch):
    sessions = _install_fakes(monkeypatch)
    _fake_flagger(monkeypatch)
    _make_images(tmp_path, ['a.tif'])
    s = _settings()
    s['error_bundle'] = object()
    events, _ = _drain(bp.run_batch(['a.tif'], tmp_path, s, lambda: False))
    table = pd.read_csv(tmp_path / 'wseg_batch' / 'a_morphology.csv')
    assert list(table['error_type']) == ['merged', 'ok']
    assert list(table['flagged']) == [True, False]
    (sess,) = sessions.values()
    assert sess['error_map'].shape == sess['gasp_labels'].shape
    assert any('1 cell(s) flagged' in e[1] for e in events if e[0] == 'note')


def test_no_error_model_no_columns(tmp_path, monkeypatch):
    _install_fakes(monkeypatch)
    _make_images(tmp_path, ['a.tif'])
    _drain(bp.run_batch(['a.tif'], tmp_path, _settings(), lambda: False))
    table = pd.read_csv(tmp_path / 'wseg_batch' / 'a_morphology.csv')
    assert 'p_error' not in table.columns


def test_errors_are_scored_on_the_labels_that_were_measured(
    tmp_path, monkeypatch
):
    sessions = _install_fakes(monkeypatch)
    seen = _fake_flagger(monkeypatch)
    _make_images(tmp_path, ['a.tif'])
    _drain(bp.run_batch(['a.tif'], tmp_path, _settings(), lambda: False))
    key, labels = _with_correction(tmp_path, sessions)
    s = _morph_only()
    s['error_bundle'] = object()
    _drain(bp.run_batch(['a.tif'], tmp_path, s, lambda: False))
    assert (seen['labels'] == labels).all()
