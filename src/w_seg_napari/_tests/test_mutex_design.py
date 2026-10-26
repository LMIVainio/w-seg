"""
Design test for the new mutex watershed inputs.

elf is not needed: a small reference implementation of the mutex watershed
(Wolf et al., 2018) is used, with the same input convention as
elf.segmentation.mutex_watershed.mutex_watershed: the first two channels are
attractive and are given as boundary probability (inverted internally), all
other channels are repulsive.
"""

import numpy as np

from w_seg_napari import _affinity as A
from w_seg_napari._model_config import make_config
from w_seg_napari._tests.test_affinity import (
    blocky_labels,
    fake_layers,
    true_affinities,
)


def reference_mws(affs, offsets, mask, n_attractive=2):
    c, h, w = affs.shape
    weights = affs.copy()
    weights[:n_attractive] = 1.0 - weights[:n_attractive]
    edges = []  # (weight, is_attractive, u, v)
    idx = np.arange(h * w).reshape(h, w)
    for ch, (dy, dx) in enumerate(offsets):
        ys, xs = np.mgrid[0:h, 0:w]
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < h) & (x2 >= 0) & (x2 < w)
        ok &= mask
        ok[ok] &= mask[y2[ok], x2[ok]]
        for u, v, wt in zip(idx[ok], idx[y2[ok], x2[ok]], weights[ch][ok]):
            edges.append((wt, ch < n_attractive, u, v))
    edges.sort(key=lambda e: -e[0])

    parent = list(range(h * w))
    mutex = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for _wt, attractive, u, v in edges:
        a, b = find(u), find(v)
        if a == b:
            continue
        if attractive:
            if b in mutex.get(a, ()):
                continue
            parent[a] = b
            for m in mutex.pop(a, set()):
                mutex[m].discard(a)
                mutex[m].add(b)
                mutex.setdefault(b, set()).add(m)
        else:
            mutex.setdefault(a, set()).add(b)
            mutex.setdefault(b, set()).add(a)
    flat = np.array([find(i) for i in range(h * w)]).reshape(h, w)
    out = np.where(mask, flat, -1)
    return out


def same_partition(pred, truth, mask):
    """Every true cell is exactly one predicted segment."""
    for lab in np.unique(truth[mask]):
        region = (truth == lab) & mask
        if len(np.unique(pred[region])) != 1:
            return False
    ids = [pred[(truth == lab) & mask][0] for lab in np.unique(truth[mask])]
    return len(set(ids)) == len(ids)


def make_set(labels, ranges):
    cfg = make_config(ranges=ranges)
    offsets = [tuple(o) for o in cfg['affinity']['offsets']]
    aff = true_affinities(labels, offsets)
    return A.gather(fake_layers(A.layer_payloads(aff, cfg, 'img')), 'img'), aff


def test_mutex_recovers_cells_from_true_affinities():
    labels = blocky_labels(size=48, cell=16, wall=3)
    aset, _ = make_set(labels, (3, 8))
    wall = (labels == 0).astype(np.float32)
    affs, offsets = A.mutex_inputs(aset, wall)
    mask = wall < 0.5
    pred = reference_mws(affs, offsets, mask)
    assert same_partition(pred, labels, mask)


def test_mutex_without_wall_map_uses_affinity_boundary():
    labels = blocky_labels(size=48, cell=16, wall=3)
    aset, _ = make_set(labels, (3, 8))
    boundary = A.boundary_magnitude(aset)
    mask = boundary < 0.5
    affs, offsets = A.mutex_inputs(aset)
    pred = reference_mws(affs, offsets, mask)
    truth_mask = mask & (labels > 0)
    assert same_partition(pred, labels, truth_mask)


def test_mutex_survives_noisy_affinities():
    labels = blocky_labels(size=48, cell=16, wall=3)
    aset, _ = make_set(labels, (3, 8))
    rng = np.random.default_rng(3)
    noisy = np.clip(
        aset.data + rng.normal(0, 0.15, aset.data.shape), 0, 1
    ).astype(np.float32)
    noisy_set = A.AffinitySet('img', aset.offsets, noisy)
    wall = np.clip(
        (labels == 0) + rng.normal(0, 0.1, labels.shape), 0, 1
    ).astype(np.float32)
    affs, offsets = A.mutex_inputs(noisy_set, wall)
    mask = wall < 0.5
    pred = reference_mws(affs, offsets, mask)
    truth_mask = mask & (labels > 0)
    # no segment may span two true cells (under-segmentation), most cells intact
    for seg in np.unique(pred[truth_mask]):
        assert len(np.unique(labels[(pred == seg) & truth_mask])) == 1
