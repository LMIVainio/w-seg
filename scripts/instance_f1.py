"""How many cells did the tool get right?  F1 at IoU 0.5 / 0.75 / 0.9.

Compares the labels a batch run produced with the labels you corrected and
saved ("Save Session" in the Batch tab):

    python scripts/instance_f1.py SESSIONS [--pred gasp|sp] [--out result]

SESSIONS  *_session.h5 files, or folders holding them (the batch's
          wseg_batch folder, or the image folder that contains it).
          Sessions without saved corrections are skipped.

Reading the result: the corrected labels start as a copy of the predicted
ones, so F1 measures how much of the tool's output survived your review. Only
check images you really went through; an unreviewed image scores 1.0.
The columns split / merged / missed / outline count the reference cells that
were not matched at IoU 0.5 by kind of error.

Writes <out>.xlsx (per image + pooled, and one sheet of all cells).
"""

import argparse
from pathlib import Path

import pandas as pd

from w_seg_napari._instance_metrics import (
    THRESHOLDS,
    evaluate,
    pooled,
    summary_row,
)
from w_seg_napari._session_io import CORRECTED, load_session


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
    ap.add_argument(
        '--pred',
        choices=['gasp', 'sp'],
        default='gasp',
        help='which batch labels to score (default gasp)',
    )
    ap.add_argument('--out', default='instance_f1')
    args = ap.parse_args(argv)

    key = 'gasp_labels' if args.pred == 'gasp' else 'sp_labels'
    rows, cell_tables = [], []
    for path in find_sessions(args.sessions):
        store = load_session(path)
        name = path.name.removesuffix('_session.h5')
        if CORRECTED not in store or key not in store:
            print(
                f'skipped {name}: no {CORRECTED if CORRECTED not in store else key}'
            )
            continue
        scores, cells = evaluate(store[key], store[CORRECTED], THRESHOLDS)
        rows.append(summary_row(name, scores, cells))
        cells.insert(0, 'image', name)
        cell_tables.append(cells)
        print(f'{name}: F1@0.5 = {rows[-1]["F1@0.5"]:.3f}')
    if not rows:
        raise SystemExit('No session with saved corrections found.')

    table = pd.DataFrame([*rows, pooled(rows)])
    print()
    print(table.iloc[[-1]].T.to_string(header=False))
    with pd.ExcelWriter(f'{args.out}.xlsx') as xw:
        table.to_excel(xw, sheet_name='per image', index=False)
        pd.concat(cell_tables).to_excel(xw, sheet_name='cells', index=False)
    print(f'\nWrote {args.out}.xlsx')


if __name__ == '__main__':
    main()
