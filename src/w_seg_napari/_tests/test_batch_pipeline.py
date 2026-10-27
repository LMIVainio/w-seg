"""Batch orchestration with the heavy parts (model, algorithms) replaced by fakes."""

import sys
import types

import numpy as np
import pandas as pd
import tifffile

import w_seg_napari._batch_pipeline as bp
from w_seg_napari._model_config import legacy_config

CALLS = []


def _install_fakes(monkeypatch):
    CALLS.clear()

    def process_large_image(image, model, is_cancelled=None, **kw):
        CALLS.append(('predict', image.shape))
        if is_cancelled and is_cancelled():
            return None, None
        _, h, w = image.shape
        return np.full((1, h, w), 0.5, np.float32), np.full(
            (2, h, w), 0.9, np.float32
        )
        yield  # makes this a generator, like the real function

    class SP:
        @classmethod
        def run_algorithm(cls, params, sem_data=None, aset=None, **kw):
            CALLS.append(('superpixels', aset.describe()))
            lab = np.ones(sem_data.shape, np.uint64)
            lab[:, sem_data.shape[1] // 2 :] = 2
            return lab

    class GASP:
        @classmethod
        def run_gasp(cls, params, aset, sp):
            CALLS.append(('gasp',))
            return sp

    def measure(labels_layer, worker=None, **kw):
        CALLS.append(('morphology', worker.abort_requested))
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
