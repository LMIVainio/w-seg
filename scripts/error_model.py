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

from w_seg_napari._segmentation_errors import (
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
    args = ap.parse_args(argv)

    key = 'gasp_labels' if args.pred == 'gasp' else 'sp_labels'
    tables = []
    for path in find_sessions(args.sessions):
        t = table_from_session(path, key)
        if t is None:
            print(f'skipped {path.name}: no corrected labels saved')
            continue
        tables.append(t)
        counts = t['change'].value_counts().to_dict()
        print(f'{t["slide"].iloc[0]}: {len(t)} cells {counts}')
    if not tables:
        sys.exit('No session with saved corrections found.')
    table = pd.concat(tables, ignore_index=True)
    if args.table:
        table.to_csv(args.table, index=False)

    try:
        bundle, r = train_error_model(table)
    except ValueError as e:
        sys.exit(str(e))
    save_error_model(bundle, args.out)

    print(f'\nTrained on {r["n_cells"]} cells, {r["n_wrong"]} wrong '
          f'({r["wrong_share"]:.0%}). Held out: {r["grouping"]}.')
    print('Average precision (higher is better; chance = '
          f'{r["chance_level"]:.2f}):')
    print(f'  gradient boosting (used)  {r["average_precision"]:.3f}')
    print(f'  random forest             {r["average_precision_random_forest"]:.3f}')
    print(f'  size-only rule            {r["average_precision_size_rule"]:.3f}')
    print(f'\nFlag threshold {r["threshold"]:.3f}: finds '
          f'{r["recall_at_threshold"]:.0%} of the wrong cells, '
          f'{r["precision_at_threshold"]:.0%} of flagged cells are wrong, '
          f'{r["flagged_share"]:.0%} of all cells flagged.')
    print(f'\nSaved {args.out}')


if __name__ == '__main__':
    main()
