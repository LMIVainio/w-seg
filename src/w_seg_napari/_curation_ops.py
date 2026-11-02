"""Pure numpy label-editing operations for the Curation tab (no Qt, no napari)."""

import numpy as np
from scipy import ndimage
from skimage.draw import line as draw_line
from skimage.measure import label as cc_label


def _bbox(mask, pad=0, shape=None):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    y0, y1 = max(ys.min() - pad, 0), ys.max() + 1 + pad
    x0, x1 = max(xs.min() - pad, 0), xs.max() + 1 + pad
    if shape is not None:
        y1, x1 = min(y1, shape[0]), min(x1, shape[1])
    return slice(int(y0), int(y1)), slice(int(x0), int(x1))


def touching_groups(labels, ids, gap=1):
    """Split `ids` into groups of labels that touch each other.

    Two labels touch when they are no more than `gap` pixels apart (gap=1 means
    directly adjacent, 2 tolerates a one-pixel wall between them). Returns a
    list of id lists, in the order the ids were given; a label that touches
    nothing forms a group of its own.
    """
    ids = [int(i) for i in dict.fromkeys(ids)]
    if len(ids) < 2:
        return [ids] if ids else []
    sel = np.isin(labels, ids)
    box = _bbox(sel, pad=int(gap) + 1, shape=labels.shape)
    sub = labels[box]
    structure = np.ones((3, 3), bool)
    iterations = max(1, int(gap))
    masks = {i: sub == i for i in ids}
    grown = {
        i: ndimage.binary_dilation(
            m, structure=structure, iterations=iterations
        )
        for i, m in masks.items()
    }
    parent = {i: i for i in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for n, a in enumerate(ids):
        for b in ids[n + 1 :]:
            if np.any(grown[a] & masks[b]):
                parent[find(b)] = find(a)
    groups = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def choose_target(ids, rule):
    if rule == 'Lowest ID':
        return min(ids)
    if rule == 'Highest ID':
        return max(ids)
    return ids[0]  # 'First Selected ID'


def merge_label_groups(
    labels, groups, rule='Lowest ID', fill_holes=True, close_gap=0
):
    """Merge each group of ids (in place) into one label. Returns the target ids.

    Groups with a single id are left untouched. With close_gap > 0 the
    background between the merged cells (a wall up to about 2 * close_gap
    pixels thick) is given to the merged cell too, never taking pixels of
    other labels.
    """
    targets = []
    for ids in groups:
        if len(ids) < 2:
            continue
        target = choose_target(list(ids), rule)
        mask = np.isin(labels, ids)
        if fill_holes:
            mask = ndimage.binary_fill_holes(mask)
        labels[np.isin(labels, ids)] = 0
        if close_gap > 0:
            k = int(close_gap) + 1
            padded = np.pad(mask, k)
            closed = ndimage.binary_closing(
                padded,
                structure=np.ones((3, 3), bool),
                iterations=int(close_gap),
            )[k:-k, k:-k]
            if fill_holes:
                closed = ndimage.binary_fill_holes(closed)
            mask = mask | (closed & (labels == 0))
        labels[mask] = target
        targets.append(int(target))
    return targets


def border_table(labels, gap=1):
    """
    How long is the border between every pair of touching labels?

    Border length = number of pixel pairs (a, b) next to each other along the
    rows or columns, a in one label and b in the other. With gap=1 the labels
    must be directly adjacent; with gap=2 a one-pixel wall (background)
    between them is crossed as well, and so on.

    Returns a list of dicts, one per touching pair, with keys
        a, b          label ids (a < b)
        border        border length in pixels
        area_a/_b     areas
        perimeter_a/_b   boundary pixels of each label
        fraction      border / perimeter of the SMALLER label (capped at 1);
                      1 means the smaller cell is enclosed by the other one
    sorted by decreasing border length.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError('Labels must be a 2D array.')
    gap = max(1, int(gap))
    h, w = labels.shape
    big = int(labels.max()) + 1
    counts = {}
    for d in range(1, gap + 1):
        for axis in (0, 1):
            if (h if axis == 0 else w) <= d:
                continue
            if axis == 0:
                a, b = labels[: h - d], labels[d:]
                mids = [labels[k : h - d + k] for k in range(1, d)]
            else:
                a, b = labels[:, : w - d], labels[:, d:]
                mids = [labels[:, k : w - d + k] for k in range(1, d)]
            ok = (a > 0) & (b > 0) & (a != b)
            for m in mids:
                ok &= m == 0  # only background between the two labels
            if not ok.any():
                continue
            lo = np.minimum(a[ok], b[ok]).astype(np.int64)
            hi = np.maximum(a[ok], b[ok]).astype(np.int64)
            keys, n = np.unique(lo * big + hi, return_counts=True)
            for key, c in zip(keys, n):
                counts[int(key)] = counts.get(int(key), 0) + int(c)
    if not counts:
        return []

    area = np.bincount(labels.ravel(), minlength=big)
    # boundary pixel = has a 4-neighbour with a different label
    edge = np.zeros(labels.shape, bool)
    edge[:-1] |= labels[:-1] != labels[1:]
    edge[1:] |= labels[1:] != labels[:-1]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    perimeter = np.bincount(labels[edge].ravel(), minlength=big)

    table = []
    for key, n in counts.items():
        a, b = divmod(key, big)
        small = a if area[a] <= area[b] else b
        table.append(
            {
                'a': a,
                'b': b,
                'border': n,
                'area_a': int(area[a]),
                'area_b': int(area[b]),
                'perimeter_a': int(perimeter[a]),
                'perimeter_b': int(perimeter[b]),
                'fraction': min(1.0, n / max(int(perimeter[small]), 1)),
            }
        )
    table.sort(key=lambda r: -r['border'])
    return table


def groups_by_border(
    labels,
    min_border=10,
    min_fraction=0.0,
    max_small_area=0,
    gap=1,
):
    """
    Groups of labels to merge because they share a long enough border.

    A pair of touching labels qualifies when
        border length >= min_border   (pixels)
        and border / perimeter of the smaller label >= min_fraction (0 = off)
        and, if max_small_area > 0, the smaller label has at most that area
    Qualifying pairs are chained: if A-B and B-C qualify, A, B and C form one
    group. Returns (groups, qualifying pairs); groups is a list of id lists.
    """
    table = border_table(labels, gap)
    chosen = []
    for r in table:
        small_area = min(r['area_a'], r['area_b'])
        if r['border'] < min_border:
            continue
        if r['fraction'] < min_fraction:
            continue
        if max_small_area > 0 and small_area > max_small_area:
            continue
        chosen.append(r)
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in chosen:
        parent[find(r['b'])] = find(r['a'])
    groups = {}
    for x in list(parent):
        groups.setdefault(find(x), []).append(int(x))
    return [sorted(g) for g in groups.values()], chosen


def cut_cell_with_line(labels, p0, p1, min_size=1):
    """Cut the cell crossed by the line p0 -> p1 into pieces (in place).

    p0, p1 are (row, col) array coordinates. The cell is the label most often
    under the line. The line pixels are removed, the cell falls apart into
    connected pieces, and the line pixels are given back to the nearest piece
    so no gap is left. The biggest piece keeps the original id; the others get
    new ids. Pieces smaller than `min_size` are not split off.
    Returns (ok, message, new_ids).
    """
    h, w = labels.shape
    r0, c0 = (int(round(v)) for v in p0)
    r1, c1 = (int(round(v)) for v in p1)
    rr, cc = draw_line(r0, c0, r1, c1)
    keep = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    rr, cc = rr[keep], cc[keep]
    if len(rr) == 0:
        return False, 'The line is outside the image.', []
    under = labels[rr, cc]
    under = under[under > 0]
    if len(under) == 0:
        return False, 'The line does not cross a cell.', []
    ids, counts = np.unique(under, return_counts=True)
    target = int(ids[np.argmax(counts)])

    box = _bbox(labels == target, pad=1, shape=labels.shape)
    sub = labels[box]
    cell = sub == target
    cut = np.zeros_like(cell)
    ry, rx = rr - box[0].start, cc - box[1].start
    inbox = (ry >= 0) & (ry < cell.shape[0]) & (rx >= 0) & (rx < cell.shape[1])
    cut[ry[inbox], rx[inbox]] = True
    cut &= cell
    pieces, n = cc_label(cell & ~cut, connectivity=1, return_num=True)
    if n < 2:
        return False, 'The line does not divide the cell into two parts.', []
    sizes = ndimage.sum(np.ones_like(pieces), pieces, index=range(1, n + 1))
    small = [i + 1 for i, s in enumerate(sizes) if s < min_size]
    for i in small:  # too small: fold back into the cell's other pieces
        pieces[pieces == i] = 0
    remaining = sorted(set(np.unique(pieces)) - {0})
    if len(remaining) < 2:
        return False, 'The pieces would be smaller than the minimum size.', []
    # Everything of the cell that has no piece yet (line, tiny bits) joins the nearest piece
    nearest = ndimage.distance_transform_edt(
        pieces == 0, return_distances=False, return_indices=True
    )
    filled = pieces[nearest[0], nearest[1]]
    filled[~cell] = 0
    sizes = {p: int((filled == p).sum()) for p in remaining}
    biggest = max(sizes, key=sizes.get)
    next_id = int(labels.max()) + 1
    new_ids = []
    out = sub.copy()
    out[cell] = 0
    out[filled == biggest] = target
    for p in remaining:
        if p == biggest:
            continue
        out[filled == p] = next_id
        new_ids.append(next_id)
        next_id += 1
    labels[box] = out
    return True, f'Cell {target} split into {len(remaining)} parts.', new_ids


def inherit_parent_labels(child, parent, unmatched='keep'):
    """Give every child label the parent label it overlaps most.

    Counts overlap area per (child, parent) pair, ignoring parent background
    (0). A child inside one parent label takes that label; a child touching
    several takes the one with the largest overlap. Children that overlap no
    parent label keep their own value ('keep') or become 0 ('zero').
    """
    if child.shape != parent.shape:
        raise ValueError(
            f'Child {child.shape} and parent {parent.shape} must be the same size.'
        )
    out = np.zeros_like(parent) if unmatched == 'zero' else child.copy()
    if unmatched != 'zero':
        out = out.astype(np.result_type(child.dtype, parent.dtype))
    inside = (child > 0) & (parent > 0)
    if not inside.any():
        return out
    c = child[inside].astype(np.int64)
    p = parent[inside].astype(np.int64)
    n_parent = int(p.max()) + 1
    keys, counts = np.unique(c * n_parent + p, return_counts=True)
    kc, kp = keys // n_parent, keys % n_parent
    order = np.lexsort((-counts, kc))  # per child, largest overlap first
    kc, kp = kc[order], kp[order]
    first = np.r_[True, kc[1:] != kc[:-1]]
    best = dict(zip(kc[first].tolist(), kp[first].tolist(), strict=True))
    lut_ids = np.array(list(best), dtype=np.int64)
    lut_vals = np.array(list(best.values()), dtype=out.dtype)
    sorter = np.argsort(lut_ids)
    pos = np.searchsorted(lut_ids, child.astype(np.int64), sorter=sorter)
    pos = np.clip(pos, 0, len(lut_ids) - 1)
    hit = (lut_ids[sorter][pos] == child) & (child > 0)
    out[hit] = lut_vals[sorter][pos][hit]
    return out
