"""
Flag cells that were probably segmented wrongly (merged, split, spurious).

Idea: the cells you correct in napari tell what the tool gets wrong. Every
predicted cell of a corrected session is compared with the corrected labels
and gets one of

    ok        matches a corrected cell (IoU >= 0.5)
    merged    contains two or more corrected cells (under-segmented)
    fragment  is a piece of a larger corrected cell (over-segmented)
    spurious  lies mostly where the corrected labels have no cell
    outline   found but traced too loosely: left out of the training

A gradient boosting model learns to tell these apart from features that need
only the predicted labels and the wall prediction (no morphology run):

    inside the cell      wall signal in the interior (a merged cell has a
                         wall line through it), distance-map peaks, shape
    on its boundary      wall signal along the outline, and on the border
                         shared with each neighbour (a fragment has a weak
                         border towards the rest of its cell)
    context              size relative to the image and to the neighbours

Result for each cell: p_error (probability that it is not ok), the most
likely error type and a flag (p_error above the threshold chosen so that
about 90 % of the real errors are flagged on held-out slides).

Needs numpy, pandas, scipy, scikit-image and scikit-learn.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import expand_labels

from w_seg_napari._instance_metrics import match, overlaps

BAD = ('merged', 'fragment', 'spurious')
STATUSES = ('ok', *BAD, 'outline')
MODEL_VERSION = 1
CHANGE_WORDS = {
    'ok': 'unchanged',
    'merged': 'split into smaller cells',  # the tool had joined several cells
    'fragment': 'merged into a larger cell',  # the tool had cut a cell apart
    'spurious': 'deleted',
    'outline': 'outline adjusted',
}
MIN_CONTACT_PX = 3  # shorter shared borders are corners, not neighbours

FEATURES = (
    'area_um2',
    'perimeter_um',
    'solidity',
    'extent',
    'eccentricity',
    'circularity',
    'aspect',
    'n_edt_peaks',
    'edt_max_um',
    'inner_wall_mean',
    'inner_wall_p95',
    'inner_wall_max',
    'border_wall_mean',
    'border_wall_p10',
    'inner_over_border',
    'n_neighbours',
    'contact_wall_min',
    'contact_wall_mean',
    'weakest_contact_um',
    'weakest_nb_area_ratio',
    'area_rel_image',
    'area_rel_neighbours',
    'touches_border',
)


# ------------------------------------------------------------ the labels
def label_predicted_cells(pred, ref, threshold=0.5, piece=0.2):
    """DataFrame label_id, status (see STATUSES), iou of every predicted cell."""
    ref_ids, pred_ids, inter, ref_area, pred_area = overlaps(pred, ref)
    matched = match(ref_ids, pred_ids, inter, ref_area, pred_area, threshold)
    iou_of = {p: v for _r, (p, v) in matched.items()}
    by_pred = {}
    for r, p, n in zip(ref_ids, pred_ids, inter):
        by_pred.setdefault(int(p), []).append((int(r), int(n)))
    rows = []
    for p, area in pred_area.items():
        hits = by_pred.get(p, [])
        covered = sum(n for _r, n in hits)
        # corrected cells that lie mostly inside this cell: it is their union
        inside = [
            r for r, n in hits if n / ref_area[r] >= 0.5 and n / area >= 0.05
        ]
        big = [r for r, n in hits if n / area >= piece]
        r_best, n_best = max(hits, key=lambda h: h[1], default=(0, 0))
        if len(inside) >= 2:
            status = 'merged'
        elif p in iou_of:
            status = 'ok'
        elif covered / area < 0.5:
            status = 'spurious'
        elif len(big) >= 2:
            status = 'merged'
        elif n_best / area >= 0.5 and n_best / ref_area[r_best] < threshold:
            status = 'fragment'
        else:
            status = 'outline'
        rows.append((p, status, iou_of.get(p, 0.0)))
    return pd.DataFrame(rows, columns=['label_id', 'status', 'iou'])


# ---------------------------------------------------------- the features
def _contacts(labels, wall, grow=2):
    """Per pair of touching cells: mean wall on the shared border, its length."""
    ex = expand_labels(labels, distance=grow)
    parts = []
    for sl_a, sl_b in (
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
    ):
        a, b = ex[sl_a], ex[sl_b]
        m = (a != b) & (a > 0) & (b > 0)
        w = 0.5 * (wall[sl_a][m] + wall[sl_b][m])
        lo = np.minimum(a[m], b[m])
        hi = np.maximum(a[m], b[m])
        parts.append(pd.DataFrame({'a': lo, 'b': hi, 'w': w}))
    both = pd.concat(parts, ignore_index=True)
    if both.empty:
        return pd.DataFrame(columns=['a', 'b', 'wall', 'n'])
    g = both.groupby(['a', 'b'])['w'].agg(['mean', 'size']).reset_index()
    g.columns = ['a', 'b', 'wall', 'n']
    return g[g['n'] >= MIN_CONTACT_PX]


def _axis(rp, which):
    # renamed in scikit-image 0.26 (axis_major_length)
    if hasattr(rp, f'axis_{which}_length'):
        return getattr(rp, f'axis_{which}_length')
    return getattr(rp, f'{which}_axis_length')


def cell_features(labels, wall, pixel_size=1.0):
    """
    One row per cell with the FEATURES (plus label_id).

    labels      2D int labels (0 = background)
    wall        2D wall probability (0-1), same shape
    pixel_size  micrometres per pixel
    """
    labels = np.asarray(labels)
    wall = np.asarray(wall, np.float32)
    if labels.shape != wall.shape or labels.ndim != 2:
        raise ValueError(
            'labels and wall must be 2D images of the same shape.'
        )
    px = float(pixel_size or 1.0)
    h, w = labels.shape
    rows = {}
    for rp in regionprops(labels):
        r0, c0, r1, c1 = rp.bbox
        mask = np.pad(rp.image, 1)
        wall_c = np.pad(wall[r0:r1, c0:c1], 1, mode='edge')
        inner_mask = ndi.binary_erosion(mask, iterations=2)
        contour = mask & ~ndi.binary_erosion(mask)
        edt = ndi.distance_transform_edt(mask)
        emax = float(edt.max())
        peaks = h_maxima(edt, max(1.0, 0.2 * emax))
        n_peaks = int(ndi.label(peaks & mask)[1])
        inner = wall_c[inner_mask]
        border = wall_c[contour]
        perim = rp.perimeter
        row = {
            'area_um2': rp.area * px * px,
            'perimeter_um': perim * px,
            'solidity': rp.solidity,
            'extent': rp.extent,
            'eccentricity': rp.eccentricity,
            'circularity': 4 * np.pi * rp.area / perim**2
            if perim > 0
            else 0.0,
            'aspect': (
                minor / major if (major := _axis(rp, 'major')) > 0 else 1.0
            )
            if (minor := _axis(rp, 'minor')) is not None
            else 1.0,
            'n_edt_peaks': n_peaks,
            'edt_max_um': emax * px,
            'inner_wall_mean': float(inner.mean()) if inner.size else np.nan,
            'inner_wall_p95': (
                float(np.percentile(inner, 95)) if inner.size else np.nan
            ),
            'inner_wall_max': float(inner.max()) if inner.size else np.nan,
            'border_wall_mean': float(border.mean())
            if border.size
            else np.nan,
            'border_wall_p10': (
                float(np.percentile(border, 10)) if border.size else np.nan
            ),
            'touches_border': int(r0 == 0 or c0 == 0 or r1 == h or c1 == w),
            '_area': rp.area,
        }
        row['inner_over_border'] = (
            row['inner_wall_p95'] / row['border_wall_mean']
            if border.size and inner.size and row['border_wall_mean'] > 1e-6
            else np.nan
        )
        rows[rp.label] = row
    df = pd.DataFrame.from_dict(rows, orient='index')
    df.index.name = 'label_id'
    if df.empty:
        return df.reset_index().reindex(columns=['label_id', *FEATURES])

    # neighbours: strength and length of the shared borders
    contacts = _contacts(labels, wall)
    nb = {}
    for a, b, wl, n in contacts[['a', 'b', 'wall', 'n']].itertuples(
        index=False
    ):
        nb.setdefault(int(a), []).append((int(b), float(wl), int(n)))
        nb.setdefault(int(b), []).append((int(a), float(wl), int(n)))
    area = df['_area']
    cols = {
        k: []
        for k in (
            'n_neighbours',
            'contact_wall_min',
            'contact_wall_mean',
            'weakest_contact_um',
            'weakest_nb_area_ratio',
            'area_rel_neighbours',
        )
    }
    for lab in df.index:
        lst = nb.get(int(lab), [])
        cols['n_neighbours'].append(len(lst))
        if lst:
            weakest = min(lst, key=lambda t: t[1])
            cols['contact_wall_min'].append(weakest[1])
            cols['contact_wall_mean'].append(
                float(np.mean([t[1] for t in lst]))
            )
            cols['weakest_contact_um'].append(weakest[2] * px)
            cols['weakest_nb_area_ratio'].append(area[lab] / area[weakest[0]])
            cols['area_rel_neighbours'].append(
                area[lab] / float(np.median([area[t[0]] for t in lst]))
            )
        else:
            for k in list(cols)[1:]:
                cols[k].append(np.nan)
    for k, v in cols.items():
        df[k] = v
    df['area_rel_image'] = area / float(area.median())
    df = df.drop(columns='_area').reset_index()
    return df[['label_id', *FEATURES]]


# ------------------------------------------------------ training material
def table_from_session(path, pred_key='gasp_labels'):
    """
    Training rows of one corrected session: features + status + slide.
    Returns None when the session has no saved corrections.
    """
    from w_seg_napari._session_io import (
        CORRECTED,
        load_session,
        read_attrs,
    )

    store = load_session(path)
    if (
        CORRECTED not in store
        or pred_key not in store
        or 'pred_sem' not in store
    ):
        return None
    attrs = read_attrs(path)
    try:
        px = float(
            json.loads(attrs.get('fe_config_json', '{}')).get('pixel_spacing')
            or 1.0
        )
    except (ValueError, TypeError):
        px = 1.0
    pred = store[pred_key]
    feats = cell_features(pred, np.squeeze(store['pred_sem']), px)
    status = label_predicted_cells(pred, store[CORRECTED])
    table = feats.merge(status, on='label_id')
    table['slide'] = Path(path).name.removesuffix('_session.h5')
    # what you did to the cell, in plain words
    table['change'] = table['status'].map(CHANGE_WORDS)
    return table


# ------------------------------------------------------------ the model
def _make_model(classifier, seed=0, hyperparameters=None):
    """The chosen classifier from the shared list, class-balanced."""
    from w_seg_napari._classifiers import TextLabelClassifier

    hp = {'balanced': True, 'seed': seed}
    hp.update(hyperparameters or {})
    return TextLabelClassifier(classifier, hp)


def _p_error(model, X):
    proba = model.predict_proba(X)
    ok = list(model.classes_).index('ok')
    return 1.0 - proba[:, ok]


def _threshold_for_recall(y_bad, score, recall=0.9):
    """Largest threshold that still flags `recall` of the real errors."""
    bad_scores = np.sort(score[y_bad == 1])
    if bad_scores.size == 0:
        return 0.5
    k = int(np.floor((1 - recall) * bad_scores.size))
    return float(bad_scores[k])


def train_error_model(
    table,
    n_splits=5,
    seed=0,
    classifier='Gradient Boosting',
    hyperparameters=None,
):
    """
    Fit the error model on a table from table_from_session (several slides).
    Returns (bundle, report): bundle goes to save_error_model; report holds
    the cross-validated scores (whole slides held out when there are 2+).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    t = table[table['status'] != 'outline'].reset_index(drop=True)
    y = t['status'].to_numpy()
    y_bad = np.isin(y, BAD).astype(int)
    if y_bad.sum() < 10 or (1 - y_bad).sum() < 10:
        raise ValueError(
            f'Need at least 10 wrong and 10 correct cells to train; got '
            f'{int(y_bad.sum())} wrong, {int((1 - y_bad).sum())} correct. '
            'Correct and save more images first.'
        )
    X = t[list(FEATURES)]
    slides = t['slide'].to_numpy()
    n_slides = len(np.unique(slides))
    if n_slides >= 2:
        folds = GroupKFold(n_splits=min(n_splits, n_slides)).split(
            X, y_bad, slides
        )
        grouping = 'slides'
    else:
        folds = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        ).split(X, y_bad)
        grouping = 'cells (only one slide: optimistic)'
    oof = np.full(len(t), np.nan)
    oof_rf = np.full(len(t), np.nan)
    for tr, te in folds:
        if y_bad[tr].sum() == 0 or y_bad[tr].sum() == len(tr):
            continue
        m = _make_model(classifier, seed, hyperparameters).fit(
            X.iloc[tr], y[tr]
        )
        oof[te] = _p_error(m, X.iloc[te])
        rf = RandomForestClassifier(
            300,
            min_samples_leaf=3,
            class_weight='balanced_subsample',
            random_state=seed,
            n_jobs=-1,
        ).fit(X.iloc[tr].fillna(-1), y[tr])
        oof_rf[te] = _p_error(rf, X.iloc[te].fillna(-1))
    seen = ~np.isnan(oof)
    if not seen.any():
        raise ValueError('Cross-validation had no usable fold.')
    rule = np.abs(np.log(t['area_rel_image'].clip(lower=1e-3))).to_numpy()
    thr = _threshold_for_recall(y_bad[seen], oof[seen])
    flagged = oof[seen] >= thr
    report = {
        'grouping': grouping,
        'n_cells': int(len(t)),
        'n_wrong': int(y_bad.sum()),
        'wrong_share': float(y_bad.mean()),
        'average_precision': float(
            average_precision_score(y_bad[seen], oof[seen])
        ),
        'average_precision_random_forest': float(
            average_precision_score(y_bad[seen], oof_rf[seen])
        ),
        'average_precision_size_rule': float(
            average_precision_score(y_bad[seen], rule[seen])
        ),
        'classifier': classifier,
        'chance_level': float(y_bad[seen].mean()),
        'threshold': thr,
        'recall_at_threshold': float(flagged[y_bad[seen] == 1].mean()),
        'precision_at_threshold': float(y_bad[seen][flagged].mean())
        if flagged.any()
        else 0.0,
        'flagged_share': float(flagged.mean()),
    }
    final = _make_model(classifier, seed, hyperparameters).fit(X, y)
    bundle = {
        'version': MODEL_VERSION,
        'model': final,
        'features': list(FEATURES),
        'classes': [str(c) for c in final.classes_],
        'threshold': thr,
        'report': report,
        'created': datetime.now().isoformat(timespec='seconds'),
    }
    return bundle, report


def read_morphology(table, session_paths, pred_key='gasp_labels'):
    """
    Attach the batch's morphology columns (prefix `morph_`) to a training
    table, matched on slide and label_id. Slides without a morphology csv
    next to their session, or whose csv was measured on other labels than
    the predicted ones, are left out. Returns (table, slides_used).
    """
    want = {'gasp_labels': 'gasp', 'sp_labels': 'sp'}[pred_key]
    parts, used = [], []
    for path in session_paths:
        path = Path(path)
        slide = path.name.removesuffix('_session.h5')
        csv = path.with_name(f'{slide}_morphology.csv')
        if not csv.exists():
            continue
        m = pd.read_csv(csv)
        if 'label_id' not in m.columns:
            continue
        if 'labels_source' in m.columns and set(m['labels_source']) != {want}:
            continue
        keep = m.select_dtypes('number').drop(
            columns=[c for c in m.columns if c in _NOT_MORPH], errors='ignore'
        )
        keep = keep.add_prefix('morph_').rename(
            columns={'morph_label_id': 'label_id'}
        )
        keep['label_id'] = m['label_id'].to_numpy()
        keep['slide'] = slide
        parts.append(keep)
        used.append(slide)
    if not parts:
        return table.iloc[0:0].copy(), []
    morph = pd.concat(parts, ignore_index=True)
    joined = table.merge(morph, on=['slide', 'label_id'], how='inner')
    return joined, used


_NOT_MORPH = {
    'id',
    'label_id',
    'centroid-0',
    'centroid-1',
    'centroid-2',
    'class',
    'class_id',
    'class_name',
    'predicted_class',
}


def morphology_gain(
    table,
    classifier='Gradient Boosting',
    hyperparameters=None,
    n_splits=5,
    seed=0,
):
    """
    Does adding morphology columns (those starting with `morph_`) to the
    error features help on slides the model has not seen?
    Returns the held-out average precision with and without them.
    """
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold

    t = table[table['status'] != 'outline'].reset_index(drop=True)
    y = t['status'].to_numpy()
    y_bad = np.isin(y, BAD).astype(int)
    slides = t['slide'].to_numpy()
    if len(np.unique(slides)) < 2 or y_bad.sum() < 10:
        raise ValueError(
            'Need at least 2 slides with morphology tables and 10 wrong cells.'
        )
    extra = [c for c in t.columns if c.startswith('morph_')]
    out = {'n_cells': int(len(t)), 'n_morphology_columns': len(extra)}
    for key, cols in (
        ('without', list(FEATURES)),
        ('with', list(FEATURES) + extra),
    ):
        oof = np.full(len(t), np.nan)
        folds = GroupKFold(n_splits=min(n_splits, len(set(slides)))).split(
            t[cols], y_bad, slides
        )
        for tr, te in folds:
            if y_bad[tr].sum() == 0:
                continue
            m = _make_model(classifier, seed, hyperparameters).fit(
                t.iloc[tr][cols], y[tr]
            )
            oof[te] = _p_error(m, t.iloc[te][cols])
        seen = ~np.isnan(oof)
        out[key] = float(average_precision_score(y_bad[seen], oof[seen]))
    out['chance_level'] = float(y_bad.mean())
    return out


def save_error_model(bundle, path):
    import joblib

    joblib.dump(bundle, path)


def load_error_model(path):
    import joblib

    try:
        bundle = joblib.load(path)
    except Exception as e:  # unreadable or not a joblib file
        raise ValueError(f'Cannot read the error model {path}: {e}') from e
    if not isinstance(bundle, dict) or bundle.get('version') != MODEL_VERSION:
        raise ValueError(
            'This is not a w-seg error model (or a different version).'
        )
    missing = [f for f in bundle['features'] if f not in FEATURES]
    if missing:
        raise ValueError(f'The error model needs unknown features: {missing}')
    return bundle


def score_cells(bundle, feats):
    """DataFrame label_id, p_error, error_type, flagged for a features table."""
    if feats.empty:
        return pd.DataFrame(
            columns=['label_id', 'p_error', 'error_type', 'flagged']
        )
    model = bundle['model']
    X = feats.reindex(columns=bundle['features'])
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    ok = classes.index('ok')
    p_err = 1.0 - proba[:, ok]
    bad_idx = [i for i, c in enumerate(classes) if c != 'ok']
    kind = np.array(classes, dtype=object)[
        np.array(bad_idx)[np.argmax(proba[:, bad_idx], axis=1)]
    ]
    return pd.DataFrame(
        {
            'label_id': feats['label_id'].to_numpy(),
            'p_error': p_err.round(4),
            'error_type': np.where(p_err >= bundle['threshold'], kind, 'ok'),
            'flagged': p_err >= bundle['threshold'],
        }
    )


def error_map(labels, scores):
    """Float image: every cell filled with its p_error (0 on the background)."""
    labels = np.asarray(labels)
    lut = np.zeros(int(labels.max()) + 1, np.float32)
    lut[scores['label_id'].to_numpy().astype(int)] = scores[
        'p_error'
    ].to_numpy()
    return lut[labels]


def flag_labels(labels, wall, bundle, pixel_size=1.0):
    """features -> scores -> (scores DataFrame, error map) in one call."""
    feats = cell_features(labels, wall, pixel_size)
    scores = score_cells(bundle, feats)
    return scores, error_map(labels, scores)
