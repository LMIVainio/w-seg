"""Train the model that flags likely segmentation errors.

    python scripts/error_model.py SESSIONS --out error_model.joblib

SESSIONS  *_session.h5 files or folders holding them (the batch's wseg_batch
          folder, or the image folder that contains it). Only sessions with
          corrected labels saved from napari (Batch tab, "Save Session") are
          used: every predicted cell is compared with your corrected labels,
          and the model learns which cells you had to fix (merged, fragment,
          spurious) from features of the cell, its wall signal and its
          neighbours.

With --table the cell-by-cell list of your changes is also written (deleted /
split into smaller cells / merged into a larger cell / outline adjusted).

The report shows how well the model ranks errors on slides it has not seen
(whole slides are held out), next to a size-only rule and a random forest.
Use the .joblib file in the Batch tab ("Flag likely segmentation errors").
Only review images you really went through: an unreviewed image teaches the
model that every cell is fine.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from w_seg_napari._classifiers import SUPPORTED_CLASSIFIERS
from w_seg_napari._segmentation_errors import (
    morphology_gain,
    read_morphology,
    save_error_model,
    table_from_session,
    train_error_model,
)


def find_sessions(paths):
    found = []
    for p in map(Path, paths):
        if p.is_dir():
            found += sorted(p.glob('*_session.h5'))
            found += sorted(p.glob('wseg_batch/*_session.h5'))
        else:
            found.append(p)
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('sessions', nargs='+')
    ap.add_argument('--out', default='error_model.joblib')
    ap.add_argument('--pred', choices=['gasp', 'sp'], default='gasp')
    ap.add_argument(
        '--table',
        help='write one row per predicted cell (label_id, slide, change, '
        'status, iou and all features) to this csv; the `change` column '
        'says what you did to each cell',
    )
    ap.add_argument(
        '--classifier',
        choices=list(SUPPORTED_CLASSIFIERS),
        default='Gradient Boosting',
        help='algorithm of the error model (same list as the Cell '
        'Morphology tab; XGBoost and LightGBM only if installed)',
    )
    ap.add_argument('--n-estimators', type=int, help='trees / boosting rounds')
    ap.add_argument('--max-depth', type=int)
    ap.add_argument('--learning-rate', type=float)
    ap.add_argument(
        '--compare-morphology',
        action='store_true',
        help="also test whether adding the batch's morphology csv columns "
        '(<image>_morphology.csv next to each session) improves the score '
        'on held-out slides. Only a report: the saved model never uses them',
    )
    args = ap.parse_args(argv)
    hp = {
        k: v
        for k, v in {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
        }.items()
        if v is not None
    }

    key = 'gasp_labels' if args.pred == 'gasp' else 'sp_labels'
    tables = []
    used_paths = []
    for path in find_sessions(args.sessions):
        t = table_from_session(path, key)
        if t is None:
            print(f'skipped {path.name}: no corrected labels saved')
            continue
        tables.append(t)
        used_paths.append(path)
        counts = t['change'].value_counts().to_dict()
        print(f'{t["slide"].iloc[0]}: {len(t)} cells {counts}')
    if not tables:
        sys.exit('No session with saved corrections found.')
    table = pd.concat(tables, ignore_index=True)
    if args.table:
        table.to_csv(args.table, index=False)

    try:
        bundle, r = train_error_model(
            table, classifier=args.classifier, hyperparameters=hp
        )
    except ValueError as e:
        sys.exit(str(e))
    save_error_model(bundle, args.out)

    print(
        f'\nTrained on {r["n_cells"]} cells, {r["n_wrong"]} wrong '
        f'({r["wrong_share"]:.0%}). Held out: {r["grouping"]}.'
    )
    print(
        'Average precision (higher is better; chance = '
        f'{r["chance_level"]:.2f}):'
    )
    print(
        f'  {args.classifier} (used)'.ljust(28)
        + f'{r["average_precision"]:.3f}'
    )
    print(
        f'  random forest             {r["average_precision_random_forest"]:.3f}'
    )
    print(
        f'  size-only rule            {r["average_precision_size_rule"]:.3f}'
    )
    print(
        f'\nFlag threshold {r["threshold"]:.3f}: finds '
        f'{r["recall_at_threshold"]:.0%} of the wrong cells, '
        f'{r["precision_at_threshold"]:.0%} of flagged cells are wrong, '
        f'{r["flagged_share"]:.0%} of all cells flagged.'
    )
    if args.compare_morphology:
        joined, used = read_morphology(table, used_paths, key)
        print('\nMorphology columns as extra features:')
        if len(used) < 2:
            print(
                '  need morphology csv files (same labels as the model '
                'predicts) for at least 2 slides; found '
                f'{len(used)}.'
            )
        else:
            try:
                g = morphology_gain(joined, args.classifier, hp)
                print(
                    f'  {len(used)} slides, {g["n_morphology_columns"]} '
                    'columns. Held-out average precision:\n'
                    f'    error features only       {g["without"]:.3f}\n'
                    f'    + morphology columns      {g["with"]:.3f}'
                )
            except ValueError as e:
                print(f'  {e}')
    print(f'\nSaved {args.out}')


if __name__ == '__main__':
    main()
