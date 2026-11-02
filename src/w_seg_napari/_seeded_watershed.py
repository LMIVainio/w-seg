"""
Seeded EDT watershed (plain numpy / scipy / skimage, no napari).

Seeds come from any map or label image: the predicted skeleton, the predicted
edge distance (high in the middle of each cell), a Labels layer, ... Every
connected piece of a seed becomes ONE cell, and a watershed on the
distance-to-wall map grows the seeds to the cell walls.

Compared with seeding from distance maxima this gives one seed per piece you
choose, so with a skeleton a long cell with several distance maxima stays one
cell. Weak spots:
  * where a wall is missing, seeds of neighbouring cells can touch and merge;
  * a thresholded edge-distance map has one seed per distance peak, so a
    dumbbell-shaped cell is split in two (the skeleton keeps it whole).
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as cc_label
from skimage.morphology import skeletonize
from skimage.segmentation import relabel_sequential, watershed


def map_seeds(
    values,
    cell_mask,
    threshold=0.5,
    min_seed_px=5,
    thin=False,
):
    """
    Connected pieces of ``values > threshold`` as integer seed labels [H, W].

    values       any map: skeleton probability, edge distance, ...
    cell_mask    bool [H, W], only pixels inside it count
    threshold    pixels above this belong to a seed
    min_seed_px  pieces with fewer pixels are dropped as noise
    thin         reduce every piece to a one-pixel line (keeps connectivity)
    """
    sk = (np.asarray(values) > threshold) & np.asarray(cell_mask, dtype=bool)
    if thin and sk.any():
        sk = skeletonize(sk)
    seeds = cc_label(sk, connectivity=2)  # diagonal neighbours are connected
    if min_seed_px > 1 and seeds.max() > 0:
        sizes = np.bincount(seeds.ravel())
        small = sizes < min_seed_px
        small[0] = False
        seeds[small[seeds]] = 0
    return seeds.astype(np.int32)


def label_seeds(seed_labels, cell_mask, min_seed_px=1):
    """
    Existing labels as seeds: each label is cut to the cell interior, and a
    label that falls apart there keeps only its connected pieces as separate
    seeds. Returns int32 labels [H, W].
    """
    seed_labels = np.squeeze(np.asarray(seed_labels))
    inside = np.where(np.asarray(cell_mask, dtype=bool), seed_labels, 0)
    out = np.zeros(inside.shape, dtype=np.int32)
    next_id = 1
    for lab in np.unique(inside):
        if lab == 0:
            continue
        pieces = cc_label(inside == lab, connectivity=2)
        for k in range(1, pieces.max() + 1):
            m = pieces == k
            if m.sum() >= min_seed_px:
                out[m] = next_id
                next_id += 1
    return out


def seeded_edt_watershed(
    wall,
    seed_map=None,
    wall_threshold=0.35,
    sigma=1.0,
    seed_threshold=0.5,
    min_seed_px=5,
    fill_unseeded=True,
    min_cell_size=15,
    compactness=0.0,
    thin_seeds=False,
    seed_labels=None,
):
    """
    Cell labels from the wall prediction and a seed map or seed labels.

    wall                 wall probability [H, W] (high on walls)
    seed_map             any map [H, W] (skeleton, edge distance, ...); its
                         connected pieces above seed_threshold are the seeds
    seed_labels          alternatively an integer label image used as seeds
                         (seed_map is then ignored)
    wall_threshold       cell interior = smoothed wall probability below this
    sigma                Gaussian smoothing of the wall map before thresholding
    seed_threshold       see map_seeds
    min_seed_px          see map_seeds
    thin_seeds           see map_seeds (thin = True)
    fill_unseeded        interior regions without any seed (the skeleton
                         missed the cell) become a cell of their own when
                         they are larger than min_cell_size; otherwise they
                         stay unlabelled
    min_cell_size        labels with this many pixels or fewer are removed
    compactness          passed to the watershed (0 = plain watershed)

    Returns uint64 labels [H, W]; 0 = wall / background.
    """
    wall = np.squeeze(np.asarray(wall)).astype(np.float32)
    given = seed_labels if seed_labels is not None else seed_map
    if given is None:
        raise ValueError('Give a seed map or seed labels.')
    given = np.squeeze(np.asarray(given))
    if wall.shape != given.shape:
        raise ValueError(
            f'Wall {wall.shape} and seeds {given.shape} must have the same shape.'
        )
    if sigma > 0:
        wall = ndi.gaussian_filter(wall, sigma=sigma)
    cell_mask = wall < wall_threshold
    if not cell_mask.any():
        return np.zeros(wall.shape, dtype=np.uint64)

    distance = ndi.distance_transform_edt(cell_mask)
    if seed_labels is not None:
        seeds = label_seeds(given, cell_mask, min_seed_px)
    else:
        seeds = map_seeds(
            given.astype(np.float32),
            cell_mask,
            seed_threshold,
            min_seed_px,
            thin_seeds,
        )

    if fill_unseeded:
        regions = cc_label(cell_mask, connectivity=1)
        seeded = np.unique(regions[seeds > 0])
        sizes = np.bincount(regions.ravel())
        orphan = np.ones(len(sizes), dtype=bool)
        orphan[0] = False
        orphan[seeded] = False
        orphan &= sizes > max(min_cell_size, 0)
        if orphan.any():
            # one seed per orphan region: its deepest point
            next_id = int(seeds.max()) + 1
            for rid in np.nonzero(orphan)[0]:
                m = regions == rid
                idx = np.argmax(np.where(m, distance, -1.0))
                seeds[np.unravel_index(idx, seeds.shape)] = next_id
                next_id += 1

    if seeds.max() == 0:
        return np.zeros(wall.shape, dtype=np.uint64)

    labels = watershed(
        -distance, markers=seeds, mask=cell_mask, compactness=compactness
    )
    if min_cell_size > 0:
        sizes = np.bincount(labels.ravel())
        small = sizes <= min_cell_size
        small[0] = False
        labels[small[labels]] = 0
    labels = relabel_sequential(labels)[0]
    return labels.astype(np.uint64)
