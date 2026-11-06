"""Train the Cell Morphology classifier from batch sessions.

    python scripts/train_cell_classifier.py SESSIONS --out cell_classifier.joblib

SESSIONS  *_session.h5 files or folders holding them (the batch's wseg_batch
          folder, or the image folder that contains it).

For each session the script needs
  * the batch's table  <image>_morphology.csv  next to the session (every
    measurement, and the YOLO class probabilities prob_* if the batch ran
    YOLO), and
  * a labels layer named  classifier_labels  saved with the session (Batch
    tab "Save Session", tick the layer under "Also keep these layers"):
    paint the class ids on the cells you want to teach, the same number for
    the same kind of cell in every image.

Every cell of the table gets a new column `classifier_label` (the class
painted on most of the cell, 0 = not painted) and the painted cells train the
classifier. Because the YOLO probabilities are among the columns, the
classifier can combine "YOLO thinks this is xylem" with the cell's shape,
wall and neighbour measurements, e.g. to tell vessels from xylem parenchyma.

The score is measured on whole images the model has not seen. Load the
.joblib file in the Cell Morphology tab ("Load Model"), or pick it in the
Batch tab.

The classes painted in the layer should be painted over the cells of the
labels the table was measured on. Run the batch with "corrected" cell labels
for the morphology stage (Batch tab, sessions as input) to measure your
corrected cells; --require-corrected insists on it.
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from w_seg_napari._classifiers import SUPPORTED_CLASSIFIERS
from w_seg_napari._session_classifier import (
    LAYER_NAME,
    morphology_config,
    session_table,
    train_from_tables,
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
    ap.add_argument('--out', default='cell_classifier.joblib')
    ap.add_argument(
        '--layer',
        default=LAYER_NAME,
        help=f'name of the painted class layer (default {LAYER_NAME})',
    )
    ap.add_argument(
        '--classifier',
        choices=[c for c in SUPPORTED_CLASSIFIERS if c != 'XGBoost'],
        default='Random Forest',
    )
    ap.add_argument('--n-estimators', type=int, default=100)
    ap.add_argument('--max-depth', type=int, default=15)
    ap.add_argument(
        '--min-coverage',
        type=float,
        default=0.0,
        help='a cell needs at least this share (0-1) of its pixels painted',
    )
    ap.add_argument(
        '--require-corrected',
        action='store_true',
        help='only use tables that were measured on the corrected labels',
    )
    ap.add_argument(
        '--table',
        help='write the cell table with the new classifier_label column '
        '(default: next to --out)',
    )
    args = ap.parse_args(argv)

    paths = find_sessions(args.sessions)
    if not paths:
        sys.exit('No *_session.h5 files found.')
    tables, used, skipped = [], [], []
    for path in paths:
        t, why = session_table(
            path, args.layer, args.require_corrected, args.min_coverage
        )
        if t is None:
            skipped.append(f'  {path.name}: {why}')
            continue
        tables.append(t)
        used.append(path)
        n = int((t['classifier_label'] > 0).sum())
        print(
            f'{path.name}: {n} painted of {len(t)} cells '
            f'(table measured on {t.attrs["labels_source"]} labels)'
        )
    if skipped:
        print('\nSkipped:\n' + '\n'.join(skipped))
    if not tables:
        sys.exit('\nNothing to train on.')

    hp = {'n_estimators': args.n_estimators, 'max_depth': args.max_depth}
    try:
        payload, r = train_from_tables(
            tables,
            args.classifier,
            hp,
            morphology_config(used),
        )
    except ValueError as e:
        sys.exit(f'\n{e}')

    print(
        f'\n{r["n_cells"]} painted cells from {r["n_slides"]} images, '
        f'{r["n_features"]} measurements ({r["yolo_columns"]} YOLO '
        'probability columns)'
    )
    if r['yolo_columns'] == 0:
        print(
            '  note: the tables have no YOLO probabilities (prob_*); the '
            'classifier uses morphology only.'
        )
    for c, n in r['classes'].items():
        print(f'  class {c}: {n} cells')
    h = r.get('held_out')
    if h:
        print(
            f'\nOn images the model has not seen ({h["cells"]} cells): '
            f'accuracy {h["accuracy"]:.3f}, macro F1 {h["macro_f1"]:.3f}'
        )
        for c, v in h['per_class'].items():
            print(
                f'  class {c}: precision {v["precision"]:.2f}, '
                f'recall {v["recall"]:.2f} ({v["cells"]} cells)'
            )
    else:
        print(
            '\nOnly one image: no held-out score. Paint at least two images '
            'for an honest estimate.'
        )

    out = Path(args.out)
    joblib.dump(payload, out)
    table_path = Path(args.table) if args.table else out.with_suffix('.csv')
    pd.concat(tables, ignore_index=True).to_csv(table_path, index=False)
    print(f'\nSaved {out}\nCell table with classifier_label: {table_path}')


if __name__ == '__main__':
    main()
