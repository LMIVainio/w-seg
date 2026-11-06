"""
Pick labels by size and by isolation (pure numpy / scipy, no Qt).

    label_areas     id and pixel area of every label
    size_ids        labels smaller than a lower or larger than an upper limit
    cell_groups     the group each label belongs to: labels that touch, or
                    lie within `gap` pixels of each other, form one group
    isolated_ids    labels in small groups, or outside the largest group
"""

import numpy as np


def label_areas(labels):
    """(ids, areas): every label id above 0 with its pixel count."""
    counts = np.bincount(np.asarray(labels).ravel().astype(np.int64))
    ids = np.nonzero(counts)[0]
    ids = ids[ids > 0]
    return ids, counts[ids]


def size_ids(labels, lower=0, upper=0):
    """
    Label ids with an area below `lower` or above `upper` (pixels).
    A limit of 0 is off.
    """
    ids, areas = label_areas(labels)
    hit = np.zeros(len(ids), bool)
    if lower > 0:
        hit |= areas < lower
    if upper > 0:
        hit |= areas > upper
    return ids[hit]


def cell_groups(labels, gap=2):
    """
    Group the labels: two labels belong together when they touch or are at
    most `gap` pixels apart (gap=2 reaches across a one-pixel wall).
    Returns (ids, group_of, n_cells, area): the label ids, the group number
    of each, and the cell count and pixel area of every group.
    """
    from scipy import ndimage as ndi

    labels = np.asarray(labels)
    filled = labels > 0
    if gap > 1:
        # pixels within gap / 2 of a cell join the cells that reach them
        near = ndi.distance_transform_edt(~filled) <= gap / 2.0
        closed = filled | near
    else:
        closed = filled
    comp, n_comp = ndi.label(closed, structure=np.ones((3, 3), bool))
    ids, areas = label_areas(labels)
    # the group of a label is the one covering most of its pixels
    lab = labels[filled].astype(np.int64)
    grp = comp[filled].astype(np.int64)
    width = n_comp + 1
    keys, counts = np.unique(lab * width + grp, return_counts=True)
    best = {}
    for key, n in zip(keys, counts, strict=True):
        cell, g = divmod(int(key), width)
        if cell not in best or n > best[cell][0]:
            best[cell] = (n, g)
    group_of = np.array([best[int(i)][1] for i in ids], dtype=np.int64)
    n_cells = np.bincount(group_of, minlength=width)
    area = np.bincount(group_of, weights=areas, minlength=width)
    return ids, group_of, n_cells, area


def isolated_ids(labels, gap=2, max_cells=0, all_but_largest=False):
    """
    Labels that sit in a small group (at most `max_cells` cells; 1 = single
    isolated cells; 0 = off) and, with all_but_largest, every label outside
    the group with the largest area (the one root to keep).
    Returns (ids, info) with info = {'groups': number of groups, 'kept':
    cells in the largest group}.
    """
    ids, group_of, n_cells, area = cell_groups(labels, gap)
    if len(ids) == 0:
        return ids, {'groups': 0, 'kept': 0}
    hit = np.zeros(len(ids), bool)
    if max_cells > 0:
        hit |= n_cells[group_of] <= max_cells
    largest = int(np.argmax(area))
    if all_but_largest:
        hit |= group_of != largest
    info = {
        'groups': int((n_cells > 0).sum()),
        'kept': int(n_cells[largest]),
    }
    return ids[hit], info
