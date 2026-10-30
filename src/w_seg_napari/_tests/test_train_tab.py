import pytest

from w_seg_napari._train_args import (
    build_args,
    count_tifs,
    parse_ranges,
    safe_name,
)


def test_safe_name_and_ranges():
    assert safe_name(' my model/1 ') == 'my_model_1'
    assert safe_name('???') == 'model'
    assert parse_ranges('6, 32  8') == [6, 8, 32]
    with pytest.raises(ValueError):
        parse_ranges('0 5')
    with pytest.raises(ValueError):
        parse_ranges('')


def test_count_tifs_and_args(tmp_path):
    (tmp_path / 'a.tif').write_text('x')
    (tmp_path / 'b.tiff').write_text('x')
    (tmp_path / 'c.png').write_text('x')
    assert count_tifs(tmp_path) == 2
    assert count_tifs('') == 0
    p = {
        'train_dir': 'T', 'val_dir': 'V', 'out_root': 'O', 'name': 'n',
        'stop_file': 'S', 'steps': 10, 'batch_size': 2, 'eval_every': 5,
        'lr': 0.001, 'warmup': 1, 'val_samples': 8, 'ranges': [6, 32],
        'zoom': (0.5, 1.5), 'tissue_bias': 0.7, 'workers': 0, 'seed': 0,
    }  # fmt: skip
    a = build_args(p)
    assert a[a.index('--ranges') + 1 : a.index('--ranges') + 3] == ['6', '32']
    assert a[a.index('--zoom') + 1 : a.index('--zoom') + 3] == ['0.5', '1.5']
    assert all(isinstance(x, str) for x in a)


def test_init_from_is_passed_on():
    base = {
        'train_dir': 'T', 'val_dir': 'V', 'out_root': 'O', 'name': 'n',
        'stop_file': 'S', 'steps': 10, 'batch_size': 2, 'eval_every': 5,
        'lr': 0.001, 'warmup': 1, 'val_samples': 8, 'ranges': [6, 32],
        'zoom': (0.5, 1.5), 'tissue_bias': 0.7, 'workers': 0, 'seed': 0,
    }  # fmt: skip
    assert '--init-from' not in build_args(base)
    a = build_args({**base, 'init_from': '/m/x'})
    assert a[a.index('--init-from') + 1] == '/m/x'


def test_geometry_flag_only_for_new_models():
    base = {
        'train_dir': 'T', 'val_dir': 'V', 'out_root': 'O', 'name': 'n',
        'stop_file': 'S', 'steps': 10, 'batch_size': 2, 'eval_every': 5,
        'lr': 0.001, 'warmup': 1, 'val_samples': 8, 'ranges': [6, 32],
        'zoom': (0.5, 1.5), 'tissue_bias': 0.7, 'workers': 0, 'seed': 0,
    }  # fmt: skip
    assert '--geometry' in build_args({**base, 'geometry': True})
    assert '--geometry' not in build_args(base)
    # a continued model keeps its own architecture
    both = build_args({**base, 'geometry': True, 'init_from': '/m'})
    assert '--geometry' not in both and '--init-from' in both
