"""Tests for the affinity helpers (numpy only; no napari or GPU needed)."""

from types import SimpleNamespace

import numpy as np
import pytest

from w_seg_napari import _affinity as A
from w_seg_napari._model_config import legacy_config, make_config

rng = np.random.default_rng(0)


def blocky_labels(size=96, cell=16, wall=3):
    yy, xx = np.mgrid[0:size, 0:size]
    in_wall = (yy % cell < wall) | (xx % cell < wall)
    labels = (yy // cell) * 100 + (xx // cell) + 1
    return np.where(in_wall, 0, labels)


def true_affinities(labels, offsets):
    """Brute-force ground truth with the training convention."""
    h, w = labels.shape
    out = np.zeros((len(offsets), h, w), dtype=np.float32)
    for i, (dy, dx) in enumerate(offsets):
        for y in range(h):
            for x in range(w):
                y2, x2 = y + dy, x + dx
                if 0 <= y2 < h and 0 <= x2 < w:
                    a, b = labels[y, x], labels[y2, x2]
                    out[i, y, x] = 1.0 if (a == b and a > 0) else 0.0
    return out


def fake_layers(payloads):
    return [SimpleNamespace(**p) for p in payloads]


def test_shift_matches_brute_force():
    a = rng.random((9, 11)).astype(np.float32)
    for dy, dx in [(0, 1), (1, 0), (-2, 3), (3, -4), (0, 0)]:
        out = A.shift(a, dy, dx)
        for y in range(9):
            for x in range(11):
                y2, x2 = y + dy, x + dx
                expect = (
                    a[y2, x2] if (0 <= y2 < 9 and 0 <= x2 < 11) else a[y, x]
                )
                assert out[y, x] == expect


def test_layer_payloads_new_model():
    cfg = make_config(ranges=(6, 32))
    aff = rng.random((8, 20, 30)).astype(np.float32)
    payloads = A.layer_payloads(aff, cfg, 'img')
    assert [p['name'] for p in payloads] == ['aff_k6_img', 'aff_k32_img']
    assert payloads[0]['data'].shape == (4, 20, 30)
    meta = payloads[1]['metadata'][A.META_KEY]
    assert meta['offsets'] == [[0, 32], [32, 0], [32, 32], [32, -32]]
    assert np.array_equal(payloads[1]['data'], aff[4:8])


def test_layer_payloads_legacy_and_errors():
    aff = rng.random((2, 10, 10)).astype(np.float32)
    payloads = A.layer_payloads(aff, legacy_config(), 'img')
    assert [p['name'] for p in payloads] == ['affH_img', 'affV_img']
    assert payloads[0]['data'].shape == (10, 10)
    with pytest.raises(ValueError):
        A.layer_payloads(
            aff, make_config(), 'img'
        )  # 2 channels, config says 8


def test_gather_round_trip_and_ordering():
    cfg = make_config(ranges=(6, 32))
    aff = rng.random((8, 12, 14)).astype(np.float32)
    layers = fake_layers(A.layer_payloads(aff, cfg, 'img'))[
        ::-1
    ]  # wrong order on purpose
    aset = A.gather(layers, 'img')
    assert aset.offsets == [tuple(o) for o in cfg['affinity']['offsets']]
    assert np.allclose(aset.data, aff)
    assert aset.ranges() == [6, 32] and not aset.legacy
    assert A.gather(layers, 'other') is None
    assert A.find_sources(
        [lay.name for lay in layers] + ['pred_img', 'x']
    ) == ['img']


def test_gather_without_metadata_uses_names():
    aff = rng.random((4, 8, 8)).astype(np.float32)
    layer = SimpleNamespace(name='aff_k6_img', data=aff, metadata={})
    aset = A.gather([layer], 'img')
    assert aset.offsets == [(0, 6), (6, 0), (6, 6), (6, -6)]
    h = SimpleNamespace(name='affH_img', data=aff[0], metadata={})
    v = SimpleNamespace(name='affV_img', data=aff[1], metadata={})
    old = A.gather([v, h], 'img', legacy_k=8)
    assert old.offsets == [(0, 8), (8, 0)] and old.legacy


def test_gather_errors():
    bad = SimpleNamespace(
        name='aff_k6_img', data=np.zeros((3, 4, 4)), metadata={}
    )
    with pytest.raises(ValueError):
        A.gather([bad], 'img')
    a = SimpleNamespace(name='affH_img', data=np.zeros((4, 4)), metadata={})
    b = SimpleNamespace(name='affV_img', data=np.zeros((5, 5)), metadata={})
    with pytest.raises(ValueError):
        A.gather([a, b], 'img')


def test_select_ranges():
    cfg = make_config(ranges=(6, 32))
    aff = rng.random((8, 6, 6)).astype(np.float32)
    aset = A.gather(fake_layers(A.layer_payloads(aff, cfg, 'i')), 'i')
    assert aset.select('shortest').ranges() == [6]
    assert aset.select('longest').ranges() == [32]
    assert aset.select('all') is aset
    assert len(aset.select('longest').offsets) == 4


def make_set(labels, ranges=(3, 8)):
    cfg = make_config(ranges=ranges)
    offsets = [tuple(o) for o in cfg['affinity']['offsets']]
    aff = true_affinities(labels, offsets)
    layers = fake_layers(A.layer_payloads(aff, cfg, 'img'))
    return A.gather(layers, 'img'), offsets


def test_boundary_is_high_on_walls_low_inside():
    labels = blocky_labels()
    aset, _ = make_set(labels)
    b = A.boundary_magnitude(aset)  # symmetric (new model)
    wall = labels == 0
    # pixels at least 4 px from any wall: rows/cols 7..10 within each 16 px cell
    gy, gx = np.mgrid[0:96, 0:96]
    centre = (
        (gy % 16 >= 7) & (gy % 16 <= 10) & (gx % 16 >= 7) & (gx % 16 <= 10)
    ) & ~wall
    assert b[wall].mean() > 0.9
    assert b[centre].mean() < 0.05


def test_symmetric_boundary_is_more_centred_than_one_sided():
    labels = blocky_labels()
    aset, _ = make_set(labels)
    wall = labels == 0
    sym = A.boundary_magnitude(aset, symmetric=True)
    one = A.boundary_magnitude(aset, symmetric=False)
    # wall pixels: symmetric map marks (at least) as many of them as strongly
    assert (sym[wall] > 0.5).mean() >= (one[wall] > 0.5).mean()


def test_legacy_boundary_matches_old_formula():
    labels = blocky_labels()
    offs = [(0, 5), (5, 0)]
    aff = true_affinities(labels, offs)
    cfg = legacy_config()
    cfg['affinity']['offsets'] = [list(o) for o in offs]
    aset = A.gather(fake_layers(A.layer_payloads(aff, cfg, 'img')), 'img')
    assert aset.legacy
    old = np.sqrt(0.5 * ((1 - aff[0]) ** 2 + (1 - aff[1]) ** 2))
    assert np.allclose(A.boundary_magnitude(aset), old, atol=1e-6)


def test_gasp_inputs_offsets_and_shapes():
    labels = blocky_labels()
    aset, offsets = make_set(labels)
    offs3, affs = A.gasp_inputs(aset)
    assert affs.shape == (8, 1, 96, 96) and affs.dtype == np.float32
    assert offs3 == [[0, dy, dx] for dy, dx in offsets]
    # Convention check: an affinity of 1 only ever joins pixels of the SAME cell.
    for (_z, dy, dx), channel in zip(offs3, affs[:, 0]):
        ys, xs = np.nonzero(channel > 0.5)
        ok = (ys + dy >= 0) & (ys + dy < 96) & (xs + dx >= 0) & (xs + dx < 96)
        ys, xs = ys[ok], xs[ok]
        assert np.all(labels[ys, xs] == labels[ys + dy, xs + dx])
        assert np.all(labels[ys, xs] > 0)


def test_mutex_inputs():
    labels = blocky_labels()
    aset, offsets = make_set(labels)
    wall = (labels == 0).astype(np.float32)
    affs, offs = A.mutex_inputs(aset, wall)
    assert affs.shape == (10, 96, 96) and offs[:2] == [[0, 1], [1, 0]]
    assert offs[2:] == [list(o) for o in offsets]
    # attractive channels carry the wall map as boundary probability
    assert (
        affs[0][labels == 0].min() == 1.0 and affs[0][8:12, 8:12].max() == 0.0
    )
    # repulsive channels are 1 - affinity
    assert np.allclose(affs[2:], 1.0 - aset.data)
    # without a wall map the boundary comes from the affinities
    affs2, _ = A.mutex_inputs(aset)
    assert affs2.shape == affs.shape
    with pytest.raises(ValueError):
        A.mutex_inputs(aset, np.zeros((5, 5)))
