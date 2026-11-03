import os

os.environ.setdefault('KERAS_BACKEND', 'torch')

import numpy as np
import pytest
import tifffile

from w_seg_napari._train_args import build_args
from w_seg_napari._training.data import photometric_jitter
from w_seg_napari._training.data_stats import (
    collect_stats,
    flag_outliers,
    robust_z,
    spread_summary,
)


def _x(seed=0):
    return np.random.default_rng(seed).uniform(-0.8, 0.8, (64, 64, 6)).astype(
        np.float32
    )


@pytest.mark.parametrize('strength', ['mild', 'strong'])
def test_jitter_keeps_shape_dtype_range(strength):
    out = photometric_jitter(_x(), np.random.default_rng(1), strength)
    assert out.shape == (64, 64, 6) and out.dtype == np.float32
    assert out.min() >= -1 and out.max() <= 1


def test_strong_changes_more_than_mild():
    d = {}
    for s in ('mild', 'strong'):
        diffs = []
        for i in range(30):
            x = _x(i)
            diffs.append(
                np.abs(photometric_jitter(x.copy(), np.random.default_rng(i), s) - x).mean()
            )
        d[s] = np.mean(diffs)
    assert d['strong'] > 1.5 * d['mild']


def test_hue_rotation_keeps_chroma_pairs_consistent():
    # with identical a,b and chroma pairs, strong jitter without blur/noise
    # must rotate both pairs by the same angle: norms stay in step
    from w_seg_napari._training.data import _rotate_pair

    x = _x()
    x[..., 3:5] = x[..., 1:3]
    _rotate_pair(x, 1, 0.2)
    _rotate_pair(x, 3, 0.2)
    assert np.allclose(x[..., 1:3], x[..., 3:5], atol=1e-6)


def test_bad_strength():
    with pytest.raises(ValueError):
        photometric_jitter(_x(), np.random.default_rng(0), 'huge')


def test_build_args_aug_and_stem():
    base = dict(
        train_dir='a', val_dir='b', out_root='c', name='n', stop_file='s',
        steps=1, batch_size=2, eval_every=1, lr=1e-4, warmup=0,
        val_samples=8, ranges=[6, 32], zoom=(0.5, 1.5), tissue_bias=0.7,
        workers=0, seed=0,
    )  # fmt: skip
    a = build_args(base)
    assert '--aug' not in a and '--input-stem' not in a
    a = build_args({**base, 'aug': 'strong', 'input_stem': 16})
    assert a[a.index('--aug') + 1] == 'strong'
    assert a[a.index('--input-stem') + 1] == '16'


def test_stem_starts_as_identity_and_trains():
    import keras

    from w_seg_napari._training.model import (
        Generator,
        add_input_stem,
        has_input_stem,
    )

    base = Generator(
        input_shape=(32, 32, 6), widths=(4, 8, 8, 8, 8), deep_supervision=False
    )
    wrapped = add_input_stem(base, hidden=8)
    assert has_input_stem(wrapped) and not has_input_stem(base)
    x = np.random.default_rng(0).uniform(-1, 1, (1, 32, 32, 6)).astype('float32')
    a = base(x, training=False)
    b = wrapped(x, training=False)
    for k in a:
        assert np.allclose(
            keras.ops.convert_to_numpy(a[k]), keras.ops.convert_to_numpy(b[k]), atol=1e-5
        )
    n_extra = wrapped.count_params() - base.count_params()
    assert 0 < n_extra < 2000


def test_stem_survives_save_and_load(tmp_path):
    import keras

    from w_seg_napari._training.model import (
        Generator,
        add_input_stem,
        has_input_stem,
    )

    m = add_input_stem(
        Generator(input_shape=(32, 32, 6), widths=(4, 8, 8, 8, 8), deep_supervision=False),
        hidden=8,
    )
    path = str(tmp_path / 'm.keras')
    m.save(path)
    m2 = keras.saving.load_model(path, compile=False)
    assert has_input_stem(m2)
    x = np.zeros((1, 32, 32, 6), 'float32')
    o1, o2 = m(x, training=False), m2(x, training=False)
    for k in o1:
        assert np.allclose(
            keras.ops.convert_to_numpy(o1[k]), keras.ops.convert_to_numpy(o2[k]), atol=1e-5
        )


def _slide(path, brightness=1.0, blur=0.0, seed=0):
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    H = W = 400
    inst = np.zeros((H, W), np.int32)
    k = 1
    for y in range(0, H, 40):
        for x in range(0, W, 40):
            inst[y + 3 : y + 37, x + 3 : x + 37] = k
            k += 1
    wall = (inst == 0).astype(np.uint8)
    gray = np.where(wall > 0, 60.0, 200.0) * brightness
    gray = ndi.gaussian_filter(gray, blur) if blur else gray
    rgb = np.stack([gray, gray * 0.9, gray * 0.8]) + rng.normal(0, 2, (3, H, W))
    st = np.concatenate(
        [np.clip(rgb, 0, 255).astype(np.uint8), wall[None], inst[None].astype(np.uint8)]
    )
    tifffile.imwrite(path, st, photometric='minisblack')


def test_stats_flag_the_odd_slide(tmp_path):
    for i in range(6):
        _slide(tmp_path / f's{i}.tif', brightness=1.0 + 0.01 * i, seed=i)
    _slide(tmp_path / 'dark_blurry.tif', brightness=0.5, blur=3.0, seed=9)
    rows = collect_stats(str(tmp_path), n_crops=4)
    assert len(rows) == 7
    flagged = flag_outliers(rows)
    assert 'dark_blurry.tif' in flagged
    assert not any(f.startswith('s') for f in flagged)
    names = {m for m, _ in flagged['dark_blurry.tif']}
    assert 'median_gray' in names and 'sharpness' in names
    assert set(spread_summary(rows)) >= {'contrast', 'sharpness'}


def test_robust_z_zero_spread():
    assert np.all(robust_z([3, 3, 3, 3]) == 0)
