"""
Validation metrics for the wall prediction, numpy only (no torch, no Qt).

Plain Dice at 0.5 hardly tells whether the walls are good: walls are 1-3 px
thin, so a one-pixel shift costs a lot, and a gap in a wall (which merges two
cells) costs almost nothing. ValidationMetrics adds, on the same pass:

  dice            pooled over all validation crops, threshold 0.5
  dice_best/thr   the best pooled Dice over thresholds 0.30 ... 0.70
  precision/recall at 0.5 (the focal Tversky loss favours recall)
  dice_img/_sd    mean and spread of the Dice of single crops
  tol_f1          wall pixels count as right when within `tol` px of the
                  true wall (precision and recall both measured that way)
  cldice          Dice of the wall skeletons: rewards unbroken walls
  cell_f1         cells (regions between walls) found with IoU > 0.5; this is
                  the closest to the final result: a gap that merges two cells
                  makes both count as missed

The true cells are the regions between the true walls, so the metric needs
nothing but the wall mask. Cells that touch the crop border and specks
smaller than `min_cell` pixels are left out.
"""

import numpy as np

THRESHOLDS = tuple(np.round(np.arange(0.30, 0.701, 0.05), 2))


def _ratio(a, b, empty=1.0):
    return float(a) / float(b) if b else empty


def _f1(precision, recall):
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def _near(mask, tol):
    """The mask grown by `tol` pixels (a square, so diagonals count too)."""
    from scipy import ndimage as ndi

    if tol <= 0:
        return mask
    return ndi.maximum_filter(mask, size=2 * tol + 1)


def _skeleton(mask):
    from skimage.morphology import skeletonize

    return skeletonize(mask)


def cell_counts(true_wall, pred_wall, min_cell=16, iou_min=0.5):
    """
    (matched, missed, extra) cells: regions between walls, truth against
    prediction, matched when their IoU is above `iou_min`.
    """
    from scipy import ndimage as ndi

    t, nt = ndi.label(~true_wall)
    p, npr = ndi.label(~pred_wall)

    def usable(lab, n):
        area = np.bincount(lab.ravel(), minlength=n + 1)
        edge = np.unique(
            np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        )
        ok = area >= min_cell
        ok[0] = False
        ok[edge] = False
        return area, ok

    at, ok_t = usable(t, nt)
    ap, ok_p = usable(p, npr)
    pair, count = np.unique(
        t.astype(np.int64).ravel() * (npr + 1) + p.ravel(),
        return_counts=True,
    )
    i, j = pair // (npr + 1), pair % (npr + 1)
    keep = (i > 0) & (j > 0)
    i, j, count = i[keep], j[keep], count[keep]
    iou = count / (at[i] + ap[j] - count)
    hit = (iou > iou_min) & ok_t[i] & ok_p[j]  # IoU > 0.5: one match at most
    matched = int(hit.sum())
    return (
        matched,
        int(ok_t.sum()) - matched,
        int(ok_p.sum()) - matched,
    )


class ValidationMetrics:
    """Add crops with add(prob, truth); read the numbers with result()."""

    def __init__(self, tol=1, min_cell=16, thresholds=THRESHOLDS):
        self.tol, self.min_cell = tol, min_cell
        self.thresholds = tuple(thresholds)
        self.sums = np.zeros((len(self.thresholds), 3))  # tp, fp, fn
        self.img_dice = []
        self.tol_sums = np.zeros(
            4
        )  # near-hit pred, n pred, near-hit true, n true
        self.cl_sums = np.zeros(
            4
        )  # skeleton of pred on true, n, ... on pred, n
        self.cells = np.zeros(3)

    def add(self, prob, truth):
        """prob: wall probability (H, W); truth: wall mask (H, W)."""
        prob = np.asarray(prob, np.float32)
        truth = np.asarray(truth) > 0.5
        for k, thr in enumerate(self.thresholds):
            pred = prob > thr
            tp = np.count_nonzero(pred & truth)
            self.sums[k] += (
                tp,
                np.count_nonzero(pred) - tp,
                np.count_nonzero(truth) - tp,
            )
        pred = prob > 0.5
        n_p, n_t = np.count_nonzero(pred), np.count_nonzero(truth)
        tp = np.count_nonzero(pred & truth)
        self.img_dice.append(_ratio(2 * tp, n_p + n_t))

        self.tol_sums += (
            np.count_nonzero(pred & _near(truth, self.tol)),
            n_p,
            np.count_nonzero(truth & _near(pred, self.tol)),
            n_t,
        )
        if n_p and n_t:
            sk_p, sk_t = _skeleton(pred), _skeleton(truth)
            self.cl_sums += (
                np.count_nonzero(sk_p & truth),
                np.count_nonzero(sk_p),
                np.count_nonzero(sk_t & pred),
                np.count_nonzero(sk_t),
            )
        self.cells += cell_counts(truth, pred, self.min_cell)

    def result(self):
        tp, fp, fn = self.sums.T
        dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        best = int(np.argmax(dice))
        mid = int(np.argmin(np.abs(np.array(self.thresholds) - 0.5)))
        tol_p = _ratio(self.tol_sums[0], self.tol_sums[1])
        tol_r = _ratio(self.tol_sums[2], self.tol_sums[3])
        cl_p = _ratio(self.cl_sums[0], self.cl_sums[1], 0.0)
        cl_r = _ratio(self.cl_sums[2], self.cl_sums[3], 0.0)
        matched, missed, extra = self.cells
        cell_p = _ratio(matched, matched + extra, 0.0)
        cell_r = _ratio(matched, matched + missed, 0.0)
        no_cells = not (matched or missed or extra)  # nothing to find or miss
        img = np.asarray(self.img_dice or [0.0])
        return {
            'dice_best': float(dice[best]),
            'thr_best': float(self.thresholds[best]),
            'precision': _ratio(tp[mid], tp[mid] + fp[mid], 0.0),
            'recall': _ratio(tp[mid], tp[mid] + fn[mid], 0.0),
            'dice_img': float(img.mean()),
            'dice_img_sd': float(img.std()),
            'tol_f1': _f1(tol_p, tol_r),
            'cldice': _f1(cl_p, cl_r),
            'cell_f1': 1.0 if no_cells else _f1(cell_p, cell_r),
        }


def selection_score(history, key='val_cell_f1', last=3):
    """Mean of the last `last` evaluations: a steadier score than one crop set."""
    values = history[key][-max(int(last), 1) :]
    return float(np.mean(values)) if values else 0.0
