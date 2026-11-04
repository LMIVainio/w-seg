"""
Instance segmentation scores: how many cells did the tool get right?

A predicted cell is a hit when it overlaps a reference cell with IoU (intersection
over union) of at least the threshold; every cell is matched at most once.
Unmatched reference cells are also classified, so you can see which kind of
error dominates:

    split     the cell is cut into two or more predicted pieces
    merged    the cell is part of a predicted cell that also covers another cell
    missed    (almost) nothing predicted there
    outline   found, but the outline is too far off for the threshold

Used by scripts/instance_f1.py (predicted labels against corrected labels of
a session). Needs numpy and pandas only.
"""

import numpy as np
import pandas as pd

THRESHOLDS = (0.5, 0.75, 0.9)
PIECE_FRACTION = 0.2  # a piece must cover this share of a cell to count as a split


def overlaps(pred, ref):
    """
    Pairs of overlapping cells (label 0 is background and is ignored).
    Returns ref_ids, pred_ids, intersection, area_ref (dict), area_pred (dict).
    """
    pred = np.asarray(pred).ravel()
    ref = np.asarray(ref).ravel()
    if pred.shape != ref.shape:
        raise ValueError('The two label images must have the same shape.')
    r_ids, r_inv = np.unique(ref, return_inverse=True)
    p_ids, p_inv = np.unique(pred, return_inverse=True)
    area_r = np.bincount(r_inv, minlength=len(r_ids))
    area_p = np.bincount(p_inv, minlength=len(p_ids))
    code = r_inv.astype(np.int64) * len(p_ids) + p_inv
    pairs, inter = np.unique(code, return_counts=True)
    ri, pi = pairs // len(p_ids), pairs % len(p_ids)
    keep = (r_ids[ri] != 0) & (p_ids[pi] != 0)
    ref_area = {int(i): int(a) for i, a in zip(r_ids, area_r) if i != 0}
    pred_area = {int(i): int(a) for i, a in zip(p_ids, area_p) if i != 0}
    return (
        r_ids[ri][keep].astype(np.int64),
        p_ids[pi][keep].astype(np.int64),
        inter[keep].astype(np.int64),
        ref_area,
        pred_area,
    )


def match(ref_ids, pred_ids, inter, ref_area, pred_area, threshold):
    """One-to-one matches {ref_id: (pred_id, iou)}, best overlaps first."""
    iou = np.array(
        [
            n / (ref_area[r] + pred_area[p] - n)
            for r, p, n in zip(ref_ids, pred_ids, inter)
        ]
    )
    matched, used_p = {}, set()
    for k in np.argsort(-iou, kind='stable'):
        if iou[k] < threshold:
            break
        r, p = int(ref_ids[k]), int(pred_ids[k])
        if r in matched or p in used_p:
            continue
        matched[r] = (p, float(iou[k]))
        used_p.add(p)
    return matched


def _scores(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def classify_cells(ref_ids, pred_ids, inter, ref_area, pred_area, matched):
    """One row per reference cell: what happened to it at the first threshold."""
    by_ref, by_pred = {}, {}
    for r, p, n in zip(ref_ids, pred_ids, inter):
        by_ref.setdefault(int(r), []).append((int(p), int(n)))
        by_pred.setdefault(int(p), []).append((int(r), int(n)))
    rows = []
    for r, area in ref_area.items():
        if r in matched:
            status, best_p, best_iou = 'matched', matched[r][0], matched[r][1]
        else:
            cands = by_ref.get(r, [])
            pieces = [p for p, n in cands if n / area >= PIECE_FRACTION]
            best_p, n_best = max(cands, key=lambda c: c[1], default=(0, 0))
            best_iou = (
                n_best / (area + pred_area[best_p] - n_best) if best_p else 0.0
            )
            others = [
                q
                for q, n in by_pred.get(best_p, [])
                if q != r and n / pred_area[best_p] >= PIECE_FRACTION
            ]
            if len(pieces) >= 2:
                status = 'split'
            elif best_p and n_best / area >= 0.5 and others:
                status = 'merged'
            elif n_best / area < 0.5:
                status = 'missed'
            else:
                status = 'outline'
        rows.append(
            {
                'ref_id': r,
                'area': area,
                'status': status,
                'pred_id': best_p,
                'iou': round(float(best_iou), 3),
            }
        )
    return pd.DataFrame(rows)


def evaluate(pred, ref, thresholds=THRESHOLDS):
    """
    Scores of predicted labels against reference labels.
    Returns (scores, cells): scores is a dict {threshold: {tp, fp, fn,
    precision, recall, f1}} plus 'n_ref', 'n_pred' and 'mean_iou' (of the
    matches at the first threshold); cells is the per-cell DataFrame.
    """
    ref_ids, pred_ids, inter, ref_area, pred_area = overlaps(pred, ref)
    scores = {}
    first = None
    for t in thresholds:
        m = match(ref_ids, pred_ids, inter, ref_area, pred_area, t)
        tp = len(m)
        scores[t] = _scores(tp, len(pred_area) - tp, len(ref_area) - tp)
        if first is None:
            first = m
    scores['n_ref'] = len(ref_area)
    scores['n_pred'] = len(pred_area)
    scores['mean_iou'] = (
        float(np.mean([v[1] for v in first.values()])) if first else 0.0
    )
    cells = classify_cells(
        ref_ids, pred_ids, inter, ref_area, pred_area, first
    )
    return scores, cells


def summary_row(name, scores, cells, thresholds=THRESHOLDS):
    row = {'image': name, 'n_ref': scores['n_ref'], 'n_pred': scores['n_pred']}
    for t in thresholds:
        s = scores[t]
        row.update(
            {
                f'F1@{t}': round(s['f1'], 4),
                f'precision@{t}': round(s['precision'], 4),
                f'recall@{t}': round(s['recall'], 4),
                f'tp@{t}': s['tp'],
                f'fp@{t}': s['fp'],
                f'fn@{t}': s['fn'],
            }
        )
    row['mean_iou_of_matches'] = round(scores['mean_iou'], 4)
    counts = cells['status'].value_counts()
    for k in ('split', 'merged', 'missed', 'outline'):
        row[k] = int(counts.get(k, 0))
    return row


def pooled(rows, thresholds=THRESHOLDS):
    """Totals over all images: F1 from the summed counts (not a mean of means)."""
    df = pd.DataFrame(rows)
    out = {'image': 'ALL (pooled)'}
    out['n_ref'] = int(df['n_ref'].sum())
    out['n_pred'] = int(df['n_pred'].sum())
    for t in thresholds:
        s = _scores(
            int(df[f'tp@{t}'].sum()),
            int(df[f'fp@{t}'].sum()),
            int(df[f'fn@{t}'].sum()),
        )
        out.update(
            {
                f'F1@{t}': round(s['f1'], 4),
                f'precision@{t}': round(s['precision'], 4),
                f'recall@{t}': round(s['recall'], 4),
                f'tp@{t}': s['tp'],
                f'fp@{t}': s['fp'],
                f'fn@{t}': s['fn'],
            }
        )
    out['mean_iou_of_matches'] = round(df['mean_iou_of_matches'].mean(), 4)
    for k in ('split', 'merged', 'missed', 'outline'):
        out[k] = int(df[k].sum())
    return out
