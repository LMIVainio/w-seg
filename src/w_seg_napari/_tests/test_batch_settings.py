import json

import numpy as np
import pytest

import w_seg_napari._batch_settings as bs


def _settings(**over):
    s = {
        'stages': {'predict': True, 'gasp': True, 'export': True},
        'morph_labels': 'gasp',
        'morph_signal': 'prediction',
        'sp_params': {'algo_name': 'Seeded Mutex', 'sigma': np.float32(1.5)},
        'gasp_params': {'linkage': 'average', 'ranges': 'all'},
        'morph_params': {
            'pixel_spacing': 0.5,
            'center_coords': (10, 20),
            'YOLO_weights': '',
        },
        'patch_size': 256,
        'stride': 128,
        'device': 'cuda',
        'model_cfg': {'outputs': {'wall': 'w'}, 'offsets': [(0, 1), (1, 0)]},
        'input_mode': 'images',
        'skip_done': True,
        'model': object(),  # never saved
    }
    s.update(over)
    return s


def test_document_is_plain_json_and_leaves_out_the_model_object():
    doc = bs.settings_document(_settings(), 'weights_a', 'rf', None)
    text = json.dumps(doc)
    assert 'object' not in text
    assert doc['sp_params']['sigma'] == 1.5  # numpy value became a number
    assert doc['morph_params']['center_coords'] == [10, 20]
    assert doc['model']['name'] == 'weights_a' and doc['error_model'] is None


def test_unsaveable_value_is_found_when_the_document_is_made():
    with pytest.raises(TypeError):
        bs.settings_document(_settings(sp_params={'bad': object()}))


def test_save_load_round_trip(tmp_path):
    doc = bs.settings_document(_settings(), 'w', 'rf')
    path = bs.save_settings(tmp_path / 'wseg_batch', doc)
    assert path.name == bs.FILENAME and path.parent.is_dir()
    back = bs.load_settings(path)
    assert back == doc


def test_same_settings_keep_the_file_different_ones_archive_the_old(tmp_path):
    out = tmp_path
    first = bs.settings_document(_settings(), 'w')
    bs.save_settings(out, first)
    bs.save_settings(out, bs.settings_document(_settings(), 'w'))  # same
    assert sorted(p.name for p in out.glob('*.json')) == [bs.FILENAME]
    changed = bs.settings_document(_settings(stride=64), 'w')
    bs.save_settings(out, changed)
    names = sorted(p.name for p in out.glob('*.json'))
    assert bs.FILENAME in names and len(names) == 2
    archived = next(iter(out.glob('batch_settings_*.json')))
    assert json.loads(archived.read_text())['stride'] == 128
    assert bs.load_settings(out / bs.FILENAME)['stride'] == 64


def test_apply_loaded_replaces_only_the_processing_settings():
    doc = bs.settings_document(_settings(stride=64, device='cpu'), 'w')
    run = _settings(stride=999, device='cuda', skip_done=False)
    out = bs.apply_loaded(run, doc)
    assert out['stride'] == 64
    assert out['gasp_params'] == doc['gasp_params']
    assert (
        out['device'] == 'cuda' and out['skip_done'] is False
    )  # not replaced
    out['gasp_params']['linkage'] = 'x'
    assert doc['gasp_params']['linkage'] == 'average'  # a copy


@pytest.mark.parametrize(
    'content, match',
    [
        ('not json', 'not a valid json'),
        ('{"a": 1}', 'not a w-seg batch settings file'),
        ('{"format_version": 99}', 'newer version'),
        ('{"format_version": 1}', 'lacks'),
    ],
)
def test_bad_files_give_clear_errors(tmp_path, content, match):
    p = tmp_path / 'x.json'
    p.write_text(content)
    with pytest.raises(ValueError, match=match):
        bs.load_settings(p)


def test_model_differences():
    doc = bs.settings_document(_settings(), 'a')
    cfg = _settings()['model_cfg']
    assert (
        bs.model_differences(doc, 'a', cfg) == []
    )  # tuples vs lists are fine
    assert 'weights' in bs.model_differences(doc, 'b', cfg)[0]
    other = {'outputs': {'wall': 'other'}, 'offsets': []}
    assert 'configuration' in bs.model_differences(doc, 'a', other)[0]


def test_describe_mentions_the_main_choices():
    text = bs.describe(bs.settings_document(_settings(), 'a'))
    assert 'Seeded Mutex' in text and 'average' in text and '0.5' in text
