import numpy as np

from w_seg_napari._training.targets import compute_affinities, compute_geometry
from w_seg_napari._training.viz import (
    direction_to_rgb,
    make_progress_figure,
    overlay_skeleton,
)

OFFSETS = [
    (0, 6),
    (6, 0),
    (6, 6),
    (6, -6),
    (0, 32),
    (32, 0),
    (32, 32),
    (32, -32),
]


def _inst(n=128):
    inst = np.zeros((n, n), np.int32)
    inst[4:60, 4:60] = 1
    inst[4:60, 66:124] = 2
    inst[66:124, 4:124] = 3
    return inst


def _sample(with_geo, rng):
    inst = _inst()
    n = inst.shape[0]
    pad = 32
    padded = np.pad(inst, pad)
    wall = (inst == 0).astype(np.float32)
    aff = compute_affinities(padded, OFFSETS, n, pad)
    y = [wall[..., None], aff]
    geo = compute_geometry(padded, n, pad)
    sample = {
        'index': 3,
        'x': rng.uniform(-1, 1, (n, n, 6)).astype(np.float32),
        'pred_wall': wall,
        'pred_aff': aff + 0.0,
    }
    if with_geo:
        y.append(geo)
        sample['pred_geo'] = geo + rng.normal(0, 0.05, geo.shape).astype(
            np.float32
        )
    sample['y'] = np.concatenate(y, axis=-1)
    return sample


def _history():
    keys = ('step', 'train_total', 'train_sem', 'train_aff')
    h = {k: [1.0, 0.9, 0.8, 0.7] for k in keys}
    for k in ('eval_step', 'val_total', 'val_sem', 'val_aff', 'val_dice'):
        h[k] = [1, 3]
    h['eval_step'] = [1, 3]
    return h


def test_direction_colours():
    vec = np.zeros((2, 2, 2))
    vec[0, 0] = (0, 1)  # dir_x = +1
    vec[0, 1] = (1, 0)  # dir_y = +1
    rgb = direction_to_rgb(vec, valid=np.array([[True, True], [False, False]]))
    assert rgb[0, 0, 0] == 1.0 and rgb[0, 0, 2] == 0.5  # red end
    assert rgb[0, 1, 2] == 1.0 and rgb[0, 1, 0] == 0.5  # blue end
    assert np.allclose(rgb[1], 0.5)  # invalid -> grey


def test_overlay_skeleton():
    base = np.zeros((3, 3, 3))
    sk = np.zeros((3, 3))
    sk[1, 1] = 1
    out = overlay_skeleton(base, sk)
    assert out[1, 1, 0] == 1.0 and out[0, 0].sum() == 0


def test_progress_figure_with_and_without_geometry():
    rng = np.random.default_rng(0)
    for with_geo in (False, True):
        fig = make_progress_figure(
            _history(),
            _sample(with_geo, rng),
            _sample(with_geo, rng),
            OFFSETS,
            [6, 32],
        )
        n_axes = len(fig.axes)
        assert n_axes > 0
        if with_geo:
            titles = [a.get_title() for a in fig.axes]
            assert 'skeleton: prediction' in titles
            assert 'direction: truth' in titles
