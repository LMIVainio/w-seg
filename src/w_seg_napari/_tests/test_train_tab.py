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


def test_auto_batch_size_is_passed_on():
    p = {
        'train_dir': 'T', 'val_dir': 'V', 'out_root': 'O', 'name': 'n',
        'stop_file': 'S', 'steps': 10, 'batch_size': 0, 'eval_every': 5,
        'lr': 0.001, 'warmup': 1, 'val_samples': 8, 'ranges': [6, 32],
        'zoom': (0.5, 1.5), 'tissue_bias': 0.7, 'workers': 0, 'seed': 0,
    }  # fmt: skip
    a = build_args(p)
    assert a[a.index('--batch-size') + 1] == '0'  # 0 = automatic


def test_find_max_batch_and_safe_batch():
    from w_seg_napari._training.autobatch import find_max_batch, safe_batch

    for limit_fit in (1, 2, 3, 7, 8, 13, 40, 127, 128, 500):
        calls = []

        def fits(b, n=limit_fit, calls=calls):
            calls.append(b)
            return b <= n

        got = find_max_batch(fits, start=2, limit=128)
        assert got == min(limit_fit, 128), (limit_fit, got)
        assert len(calls) <= 16  # a handful of probes, not a scan
    assert find_max_batch(lambda b: False) == 0  # not even one fits
    # the margin leaves headroom and keeps even sizes from 4 up
    assert safe_batch(0) == 0 and safe_batch(1) == 1 and safe_batch(2) == 1
    assert safe_batch(10) == 8 and safe_batch(40) == 34
    assert safe_batch(128) % 2 == 0 and safe_batch(128) <= 128 * 0.85


def test_loss_options_are_passed_on_and_can_be_switched_off():
    p = {
        'train_dir': 'T', 'val_dir': 'V', 'out_root': 'O', 'name': 'n',
        'stop_file': 'S', 'steps': 10, 'batch_size': 2, 'eval_every': 5,
        'lr': 0.001, 'warmup': 1, 'val_samples': 8, 'ranges': [6, 32],
        'zoom': (0.5, 1.5), 'tissue_bias': 0.7, 'workers': 0, 'seed': 0,
    }  # fmt: skip
    assert '--w-dot' not in build_args(p)  # the trainer's defaults apply
    a = build_args({**p, 'w_dot': 0.0, 'hard_neg': 0.0, 'w_interior': 0.5})
    assert a[a.index('--w-dot') + 1] == '0.0'  # 0 switches the term off
    assert a[a.index('--hard-neg') + 1] == '0.0'
    assert a[a.index('--w-interior') + 1] == '0.5'
    assert '--aff-floor' not in a


def test_mixed_precision_and_accumulation_args():
    p = {
        'train_dir': 'T', 'val_dir': 'V', 'out_root': 'O', 'name': 'n',
        'stop_file': 'S', 'steps': 10, 'batch_size': 4, 'eval_every': 5,
        'lr': 0.001, 'warmup': 1, 'val_samples': 8, 'ranges': [6, 32],
        'zoom': (0.5, 1.5), 'tissue_bias': 0.7, 'workers': 0, 'seed': 0,
    }  # fmt: skip
    plain = build_args(p)
    assert '--mixed-precision' not in plain and '--accum' not in plain
    assert '--accum' not in build_args({**p, 'accum': 1})
    a = build_args({**p, 'accum': 4, 'mixed_precision': True})
    assert '--mixed-precision' in a
    assert a[a.index('--accum') + 1] == '4'
