"""
Affinity layouts for the plugin.

Which napari layers hold which affinity channels, and how the decoders
(superpixels, GASP, mutex watershed) turn them into their inputs.

Pure numpy: no napari, Qt or torch, so it can be tested on its own.

Conventions (same as the training code)
---------------------------------------
* An affinity channel belongs to an offset (dy, dx). At pixel p it is 1 when p
  and p + (dy, dx) lie in the same cell, otherwise 0 (different cell, or one of
  the two pixels is wall).
* New models write one napari layer PER RANGE, named ``aff_k<range>_<source>``,
  with 4 channels (E, S, SE, SW) on the first axis. The layer's
  ``metadata['w_seg_affinity']`` lists the offsets, so any layout works.
* Old models write two layers ``affH_<source>`` and ``affV_<source>``
  (offsets (0, k) and (k, 0), k = 8 unless told otherwise).
"""

import re

import numpy as np

META_KEY = 'w_seg_affinity'
DEFAULT_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

_NEW_NAME = re.compile(r'^aff_k(\d+)_(.+)$')
_LEGACY_NAME = re.compile(r'^aff([HV])_(.+)$')


def offset_range(offset):
    """Range of an offset in pixels along its longest axis (diagonals count k, not k*sqrt(2))."""
    return max(abs(int(offset[0])), abs(int(offset[1])))


# ---------------------------------------------------------------------
# Container for the affinity channels of one prediction
# ---------------------------------------------------------------------
class AffinitySet:
    """All affinity channels belonging to one prediction. ``offsets[i]`` belongs to ``data[i]``."""

    def __init__(self, source, offsets, data, legacy=False):
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 3 or data.shape[0] != len(offsets):
            raise ValueError(
                f'Affinity data of shape {data.shape} does not match {len(offsets)} offsets'
            )
        self.source = source
        self.offsets = [(int(dy), int(dx)) for dy, dx in offsets]
        self.data = data
        self.legacy = legacy

    @property
    def shape(self):
        return self.data.shape[1:]

    def ranges(self):
        return sorted({offset_range(o) for o in self.offsets})

    def channels(self, k):
        """Channel indices whose offsets have range k."""
        return [i for i, o in enumerate(self.offsets) if offset_range(o) == k]

    def select(self, which='all'):
        """A new AffinitySet with 'all', the 'shortest' or the 'longest' range only."""
        if which == 'all' or len(self.ranges()) == 1:
            return self
        ks = self.ranges()
        k = ks[0] if which == 'shortest' else ks[-1]
        idx = self.channels(k)
        return AffinitySet(
            self.source,
            [self.offsets[i] for i in idx],
            self.data[idx],
            self.legacy,
        )

    def describe(self):
        ks = self.ranges()
        return f'{len(self.offsets)} channels, ranges {ks}'


# ---------------------------------------------------------------------
# From the model output to napari layers
# ---------------------------------------------------------------------
def layer_payloads(final_aff, cfg, source):
    """
    Split the stitched affinity array [n, H, W] into napari layers.

    Returns a list of dicts with keys name, data, metadata, ready for
    ``viewer.add_image(**p, colormap='gray', blending='additive')``.
    """
    aff = np.clip(np.asarray(final_aff, dtype=np.float32), 0.0, 1.0)
    a = cfg['affinity']
    offsets = [tuple(o) for o in a['offsets']]
    if aff.shape[0] != len(offsets):
        raise ValueError(
            f'Model produced {aff.shape[0]} affinity channels but its config lists {len(offsets)} offsets'
        )

    if cfg.get('legacy'):
        names = [f'affH_{source}', f'affV_{source}']
        return [
            {
                'name': names[i],
                'data': aff[i],
                'metadata': {
                    META_KEY: {
                        'source': source,
                        'range': offset_range(offsets[i]),
                        'offsets': [list(offsets[i])],
                    }
                },
            }
            for i in range(len(offsets))
        ]

    per = len(a['directions'])
    payloads = []
    for r, k in enumerate(a['ranges']):
        lo, hi = r * per, (r + 1) * per
        payloads.append(
            {
                'name': f'aff_k{k}_{source}',
                'data': aff[lo:hi],
                'metadata': {
                    META_KEY: {
                        'source': source,
                        'range': int(k),
                        'offsets': [list(o) for o in offsets[lo:hi]],
                        'direction_names': list(a['direction_names']),
                    }
                },
            }
        )
    return payloads


# ---------------------------------------------------------------------
# From napari layers to decoder inputs
# ---------------------------------------------------------------------
def find_sources(layer_names):
    """Sorted names of all predictions that have affinity layers."""
    found = set()
    for name in layer_names:
        for pattern in (_NEW_NAME, _LEGACY_NAME):
            m = pattern.match(name)
            if m:
                found.add(m.group(m.lastindex))
    return sorted(found)


def gather(layers, source, legacy_k=8):
    """
    Collect every affinity layer of ``source`` into one AffinitySet (shortest
    range first). ``layers`` are napari layers (or anything with .name, .data,
    .metadata). Returns None when the source has no affinity layers.

    Offsets come from the layer metadata; if it is missing (for example after
    the layers were saved and reloaded) the layer name is used: ``aff_k6_*`` is
    taken as the default 4 directions at range 6, ``affH_*``/``affV_*`` as
    (0, legacy_k) and (legacy_k, 0).
    """
    parts, legacy_only = [], True
    for lay in layers:
        name = lay.name
        meta = (getattr(lay, 'metadata', None) or {}).get(META_KEY)

        m = _NEW_NAME.match(name)
        if m and m.group(2) == source:
            data = np.asarray(lay.data, dtype=np.float32)
            if data.ndim == 2:
                data = data[None]
            if meta:
                offsets = [tuple(o) for o in meta['offsets']]
            else:
                k = int(m.group(1))
                offsets = [(dy * k, dx * k) for dy, dx in DEFAULT_DIRECTIONS]
            if data.shape[0] != len(offsets):
                raise ValueError(
                    f'Layer {name}: {data.shape[0]} channels but {len(offsets)} offsets'
                )
            parts.append((offsets, data))
            legacy_only = False
            continue

        m = _LEGACY_NAME.match(name)
        if m and m.group(2) == source:
            data = np.squeeze(np.asarray(lay.data, dtype=np.float32))
            if data.ndim != 2:
                raise ValueError(
                    f'Layer {name}: expected a 2D array, got shape {data.shape}'
                )
            if meta:
                offsets = [tuple(meta['offsets'][0])]
            else:
                offsets = (
                    [(0, legacy_k)] if m.group(1) == 'H' else [(legacy_k, 0)]
                )
            parts.append((offsets, data[None]))

    if not parts:
        return None

    parts.sort(key=lambda p: (offset_range(p[0][0]), p[0][0]))
    shapes = {p[1].shape[1:] for p in parts}
    if len(shapes) != 1:
        raise ValueError(
            f'Affinity layers of {source!r} have different shapes: {sorted(shapes)}'
        )
    offsets = [o for p in parts for o in p[0]]
    data = np.concatenate([p[1] for p in parts], axis=0)
    return AffinitySet(source, offsets, data, legacy=legacy_only)


# ---------------------------------------------------------------------
# Decoder inputs
# ---------------------------------------------------------------------
def shift(a, dy, dx):
    """
    out[y, x] = a[y + dy, x + dx] on the last two axes; where that pixel lies
    outside the image the pixel keeps its own value.
    """
    h, w = a.shape[-2:]
    out = a.copy()
    out[..., max(-dy, 0) : h + min(-dy, 0), max(-dx, 0) : w + min(-dx, 0)] = a[
        ..., max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)
    ]
    return out


def boundary_magnitude(aset, symmetric=None):
    """
    Boundary strength in [0, 1] (0 = inside a cell, 1 = wall), from the shortest range.

    Affinities are one-sided (pixel p versus p + offset). With symmetric=True the
    opposite direction is added as well (pixel p versus p - offset), which
    centres the boundary on the wall. Default: symmetric for new models, as-is
    (exactly the old formula) for old two-channel models.
    """
    if symmetric is None:
        symmetric = not aset.legacy
    idx = aset.channels(aset.ranges()[0])
    parts = []
    for i in idx:
        a = np.clip(aset.data[i], 0.0, 1.0)
        parts.append(1.0 - a)
        if symmetric:
            dy, dx = aset.offsets[i]
            parts.append(1.0 - shift(a, -dy, -dx))
    return np.sqrt(np.mean(np.square(parts), axis=0)).astype(np.float32)


def gasp_inputs(aset):
    """
    (offsets, affinities) for elf's GaspFromAffinities: offsets as [z, y, x]
    with z = 0, affinities as [n, 1, H, W].
    """
    offsets = [[0, dy, dx] for dy, dx in aset.offsets]
    affs = np.ascontiguousarray(aset.data[:, None, :, :], dtype=np.float32)
    return offsets, affs


def mutex_inputs(aset, wall=None):
    """
    Inputs for elf's mutex_watershed, which treats the FIRST TWO channels as
    attractive (it inverts them itself) and all remaining ones as repulsive.

    * channels 0, 1: nearest-neighbour edges (0, 1) and (1, 0), given as
      boundary probability. The boundary is the predicted wall map when given,
      otherwise the boundary magnitude of the shortest affinity range. An edge
      is as strong a boundary as its stronger pixel.
    * all other channels: every affinity channel as a repulsive edge, weight
      1 - affinity (probability that the two pixels are in different cells).

    Returns (affs [2 + n, H, W] float32, offsets [[dy, dx], ...]).

    The affinity offsets themselves are not used as attractive edges: with only
    long edges the pixel graph splits into independent sub-lattices.
    """
    if wall is None:
        boundary = boundary_magnitude(aset)
    else:
        boundary = np.clip(
            np.squeeze(np.asarray(wall, dtype=np.float32)), 0.0, 1.0
        )
        if boundary.shape != aset.shape:
            raise ValueError(
                f'Wall map shape {boundary.shape} does not match the affinities {aset.shape}'
            )
    att_h = np.maximum(boundary, shift(boundary, 0, 1))
    att_v = np.maximum(boundary, shift(boundary, 1, 0))
    rep = 1.0 - np.clip(aset.data, 0.0, 1.0)
    affs = np.concatenate([att_h[None], att_v[None], rep], axis=0)
    offsets = [[0, 1], [1, 0]] + [list(o) for o in aset.offsets]
    return np.ascontiguousarray(affs, dtype=np.float32), offsets
