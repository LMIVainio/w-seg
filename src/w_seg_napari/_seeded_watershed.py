"""
Skeleton-seeded EDT watershed (plain numpy / scipy / skimage, no napari).

The geometry head predicts each cell's skeleton. Every connected piece of the
skeleton becomes ONE seed, and a watershed on the distance-to-wall map grows
the seeds to the cell walls. Compared with seeding from distance maxima this
gives exactly one seed per cell, also for long cells with several maxima.

Known weak spot: where a wall is missing, the skeletons of two neighbouring
cells can touch and become one seed, which merges the cells.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as cc_label
from skimage.morphology import skeletonize
from skimage.segmentation import relabel_sequential, watershed


def skeleton_seeds(
    skeleton,
    cell_mask,
    threshold=0.5,
    min_seed_px=5,
    thin=True,
):
    """
    Connected skeleton pieces as integer seed labels [H, W].

    skeleton     predicted skeleton probability [H, W]
    cell_mask    bool [H, W], only skeleton pixels inside it count
    threshold    skeleton probability above which a pixel is skeleton
    min_seed_px  pieces with fewer pixels are dropped as noise
    thin         re-skeletonize the thresholded map so that thick or blobby
                 predictions become one-pixel lines (does not change topology)
    """
    sk = (np.asarray(skeleton) > threshold) & np.asarray(cell_mask, dtype=bool)
    if thin and sk.any():
        sk = skeletonize(sk)
    seeds = cc_label(sk, connectivity=2)  # diagonal neighbours are connected
    if min_seed_px > 1 and seeds.max() > 0:
        sizes = np.bincount(seeds.ravel())
        small = sizes < min_seed_px
        small[0] = False
        seeds[small[seeds]] = 0
    return seeds.astype(np.int32)


def seeded_edt_watershed(
    wall,
    skeleton,
    wall_threshold=0.35,
    sigma=1.0,
    skeleton_threshold=0.5,
    min_seed_px=5,
    fill_unseeded=True,
    min_cell_size=15,
    compactness=0.0,
):
    """
    Cell labels from the wall prediction and the predicted skeleton.

    wall                 wall probability [H, W] (high on walls)
    skeleton             skeleton probability [H, W]
    wall_threshold       cell interior = smoothed wall probability below this
    sigma                Gaussian smoothing of the wall map before thresholding
    skeleton_threshold   see skeleton_seeds
    min_seed_px          see skeleton_seeds
    fill_unseeded        interior regions without any seed (the skeleton
                         missed the cell) become a cell of their own when
                         they are larger than min_cell_size; otherwise they
                         stay unlabelled
    min_cell_size        labels with this many pixels or fewer are removed
    compactness          passed to the watershed (0 = plain watershed)

    Returns uint64 labels [H, W]; 0 = wall / background.
    """
    wall = np.squeeze(np.asarray(wall)).astype(np.float32)
    skeleton = np.squeeze(np.asarray(skeleton)).astype(np.float32)
    if wall.shape != skeleton.shape:
        raise ValueError(
            f'Wall {wall.shape} and skeleton {skeleton.shape} must have the same shape.'
        )
    if sigma > 0:
        wall = ndi.gaussian_filter(wall, sigma=sigma)
    cell_mask = wall < wall_threshold
    if not cell_mask.any():
        return np.zeros(wall.shape, dtype=np.uint64)

    distance = ndi.distance_transform_edt(cell_mask)
    seeds = skeleton_seeds(
        skeleton, cell_mask, skeleton_threshold, min_seed_px
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
