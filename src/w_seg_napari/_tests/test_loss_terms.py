"""
The interior-error, hard-negative and dot terms of the training loss.

Needs Keras. Where Keras is not installed (this test environment), a small
numpy stand-in for the few ops the loss uses is put in place for the duration
of the test, so the logic (shapes, opening, switches) is still checked.
"""

import importlib
import sys
import types
from contextlib import contextmanager

import numpy as np
from scipy import ndimage as ndi


def _fake_keras():
    ops = types.SimpleNamespace(
        convert_to_tensor=lambda x: np.asarray(x, dtype=np.float32),
        clip=np.clip,
        sum=np.sum,
        mean=np.mean,
        abs=np.abs,
        maximum=np.maximum,
        sqrt=np.sqrt,
        relu=lambda x: np.maximum(x, 0),
        stop_gradient=lambda x: x,
        cast=lambda x, dtype: np.asarray(x).astype(dtype),
        pad=lambda x, pad_width, mode='constant', constant_values=0.0: np.pad(
            x, pad_width, mode='constant', constant_values=constant_values
        ),
    )

    def max_pool(x, size, strides=1, padding='valid'):
        assert strides == 1
        r = size // 2
        filt = ndi.maximum_filter(
            x, size=(1, size, size, 1), mode='constant', cval=-np.inf
        )
        return filt if padding == 'same' else filt[:, r:-r, r:-r, :]

    ops.max_pool = max_pool
    return types.SimpleNamespace(ops=ops)


@contextmanager
def _loss_modules():
    try:
        import keras  # noqa: F401

        real = True
    except ImportError:
        real = False
    saved = {
        k: sys.modules.get(k)
        for k in (
            'keras',
            'torch',
            'w_seg_napari._training.steps',
            'w_seg_napari._training.losses',
        )
    }
    if not real:
        sys.modules['keras'] = _fake_keras()
        fake_torch = types.ModuleType('torch')
        fake_torch.no_grad = lambda: lambda fn: fn  # decorator in steps.py
        sys.modules['torch'] = fake_torch
        sys.modules.pop('w_seg_napari._training.steps', None)
        sys.modules.pop('w_seg_napari._training.losses', None)
    try:
        yield (
            importlib.import_module('w_seg_napari._training.steps'),
            importlib.import_module('w_seg_napari._training.losses'),
        )
    finally:
        if not real:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v


def _batch():
    """Three lumens separated by 2 px walls, and a model output for it."""
    n = 48
    wall = np.zeros((n, n), np.float32)
    wall[::16, :] = 1
    wall[:, ::16] = 1
    wall[1::16, :] = 1
    wall[:, 1::16] = 1
    y_sem = wall[None, :, :, None]
    y_aff = (1 - y_sem).repeat(2, axis=-1)  # 1 inside cells (simplified)
    y = np.concatenate([y_sem, y_aff], axis=-1)
    w = (0.1 + 10 * np.exp(-(ndi.distance_transform_edt(1 - wall) ** 2) / 50))[
        None, :, :, None
    ].astype(np.float32)
    clean = {
        'semantic_head': np.clip(y_sem * 0.95, 0, 1).astype(np.float32),
        'affinity_head': np.clip(y_aff * 0.95, 0, 1).astype(np.float32),
    }
    return y.astype(np.float32), w, clean


def _with_dots(outputs, size=2):
    out = {k: v.copy() for k, v in outputs.items()}
    for cy, cx in ((24, 24), (24, 40), (40, 24)):  # inside lumens
        out['semantic_head'][0, cy : cy + size, cx : cx + size, 0] = 0.9
    return out


def test_soft_open_removes_dots_keeps_walls():
    with _loss_modules() as (steps, _):
        p = np.zeros((1, 30, 30, 1), np.float32)
        p[0, 10:12, 10:12, 0] = 1.0  # a 2 x 2 dot
        p[0, 20:26, 3:28, 0] = 1.0  # a 6 px thick line
        p[0, 3:6, 3:28, 0] = 1.0  # a 3 px thick line: thinner than the window
        opened = np.asarray(steps.soft_open(p, 2))
        assert opened[0, 10:12, 10:12, 0].max() == 0  # dot gone
        assert (opened[0, 20:26, 3:28, 0] == 1).all()  # thick line survives
        assert opened[0, 3:6, 3:28, 0].max() == 0  # thin specks do not
        assert opened.shape == p.shape


def test_dot_term_and_switches():
    with _loss_modules() as (steps, losses):
        ftl = losses.WeightedFocalTverskyLoss()
        y, w, clean = _batch()
        dotty = _with_dots(clean)

        _, parts_clean = steps.compute_loss(clean, y, w, ftl)
        _, parts_dotty = steps.compute_loss(dotty, y, w, ftl)
        assert float(parts_clean['int_dots']) < 1e-6
        assert float(parts_dotty['int_dots']) > 10 * float(
            parts_clean['int_dots']
        )

        # w_dot switches the penalty on and off exactly
        off = {'w_dot': 0.0, 'hard_neg': 0.0}
        t_off, p_off = steps.compute_loss(dotty, y, w, ftl, **off)
        t_on, p_on = steps.compute_loss(
            dotty, y, w, ftl, w_dot=10.0, hard_neg=0.0
        )
        assert np.isclose(
            float(t_on) - float(t_off),
            10.0 * float(p_on['int_dots']),
            rtol=1e-4,
        )
        # the metric is logged either way
        assert float(p_off['int_dots']) == float(p_on['int_dots'])

        # hard negatives: false wall pixels weigh more, clean outputs do not
        t_hn0, _ = steps.compute_loss(
            dotty, y, w, ftl, w_dot=0.0, hard_neg=0.0
        )
        t_hn, _ = steps.compute_loss(dotty, y, w, ftl, w_dot=0.0, hard_neg=2.0)
        assert float(t_hn) > float(t_hn0)
        c0, _ = steps.compute_loss(clean, y, w, ftl, w_dot=0.0, hard_neg=0.0)
        c2, _ = steps.compute_loss(clean, y, w, ftl, w_dot=0.0, hard_neg=2.0)
        assert abs(float(c2) - float(c0)) < 0.02 * float(c0)


def test_larger_blobs_are_not_dots():
    with _loss_modules() as (steps, losses):
        ftl = losses.WeightedFocalTverskyLoss()
        y, w, clean = _batch()
        small = _with_dots(clean, size=2)
        big = _with_dots(clean, size=7)  # bigger than the opening window
        d_small = float(steps.compute_loss(small, y, w, ftl)[1]['int_dots'])
        d_big = float(steps.compute_loss(big, y, w, ftl)[1]['int_dots'])
        assert d_small > 0 and d_big < d_small
