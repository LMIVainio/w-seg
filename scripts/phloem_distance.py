"""Primary phloem pole distance for a batch of images, optionally checked
against manual measurements.

    python scripts/phloem_distance.py TABLES --phloem-class 3 [options]

TABLES   a folder of per-image tables written by the plugin's batch run
         (<image>_morphology.csv), or one csv / xlsx with a `source_file`
         column (or the single-image export; then the file name is the image).

Example (phloem = class 3, xylem = class 4, manual values in an Excel file):

    python scripts/phloem_distance.py batch_out/ --phloem-class 3 \\
        --xylem-class 4 --pixel-size 0.5 \\
        --manual manual.xlsx --manual-image-col Image --manual-value-col Distance_um

Phloem poles in steps (most reliable class first; run with --list-classes to
see the class names in the tables):

    python scripts/phloem_distance.py batch_out/ --class-col yolo_pred_tissue \\
        --phloem-order "phloem pole parenchymous cells" "primary phloem" \\
        "sieve element" --xylem-class "primary xylem" \\
        --sessions batch_out/wseg_batch --plots-dir checks/

Mixed magnifications: give the pixel size per image with --scales (a table
with the image name in the first column and um per pixel in the second).
Writes <out>.xlsx (results, comparison, summary) and <out>_comparison.png,
plus one check picture per image with --plots-dir.
"""

import argparse
import os

import numpy as np
import pandas as pd

from w_seg_napari._phloem_distance import (
    compare_with_manual,
    find_sessions,
    image_key,
    load_tables,
    measure_all,
    plot_comparison,
    plot_image,
    plot_overlay,
    pooling_levels,
    read_manual_poles,
    session_image,
    unmatched_classes,
)


def read_table(path, sheet=None):
    if str(path).lower().endswith('.csv'):
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=sheet or 0)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('tables')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        '--phloem-class',
        nargs='+',
        help='class id(s) / name(s) of the primary phloem cells, all used '
        'at once',
    )
    g.add_argument(
        '--phloem-order',
        nargs='+',
        help='phloem-type classes, most reliable first (e.g. the phloem '
        'pole parenchyma class, then primary phloem, then sieve element). '
        'The first class alone is tried; if its cells do not form two clear '
        'clusters the next class is added, and so on. An image where no '
        'step works is marked "unclear".',
    )
    p.add_argument(
        '--min-sep',
        type=float,
        default=4.0,
        help='two clusters are clear when the distance between their '
        'centres is at least this many times the mean distance of the '
        'cells to their own centre (default 4)',
    )
    p.add_argument(
        '--min-pole-cells',
        type=int,
        default=2,
        help='cells a pole needs to count as a cluster (default 2)',
    )
    p.add_argument(
        '--drop-unclear',
        action='store_true',
        help='leave the distance empty for images where no step gave two '
        'clear clusters (default: measured from the best step, flagged)',
    )
    p.add_argument(
        '--list-classes',
        action='store_true',
        help='print the values of the class column and exit',
    )
    p.add_argument(
        '--sessions',
        default=None,
        help='batch results folder (wseg_batch) with the <image>_session.h5 '
        'files: with --plots-dir a picture of the image with the cluster '
        'centroids and lines is saved for each image',
    )
    p.add_argument(
        '--xylem-class',
        nargs='+',
        default=None,
        help='class id(s) of the primary xylem cells (optional)',
    )
    p.add_argument('--class-col', default='predicted_class')
    p.add_argument('--sheet', default=None, help='xlsx sheet of the tables')
    p.add_argument(
        '--pixel-size', type=float, default=None, help='um per pixel'
    )
    p.add_argument(
        '--scales',
        default=None,
        help='table: image name, um per pixel (overrides --pixel-size)',
    )
    p.add_argument(
        '--trim',
        type=float,
        default=3.0,
        help='drop cells farther than this x median spread from their '
        'pole and refit (0 = off)',
    )
    p.add_argument(
        '--manual', default=None, help='xlsx / csv with manual values'
    )
    p.add_argument('--manual-sheet', default=None)
    p.add_argument('--manual-image-col', default=None)
    p.add_argument('--manual-value-col', default=None)
    p.add_argument(
        '--use',
        choices=['distance', 'distance_xylem'],
        default=None,
        help='which measurement to compare (default: xylem-corrected '
        'when xylem classes are given)',
    )
    p.add_argument('--out', default='phloem_distance')
    p.add_argument('--plots-dir', default=None)
    a = p.parse_args()

    tables = load_tables(a.tables, a.sheet)
    if a.list_classes:
        counts = pd.concat(
            [t[a.class_col] for t in tables.values() if a.class_col in t]
        ).value_counts()
        print(counts.to_string())
        return
    wanted = (a.phloem_class or a.phloem_order) + (a.xylem_class or [])
    missing = unmatched_classes(tables, a.class_col, wanted)
    if missing:
        have = pd.concat(
            [t[a.class_col] for t in tables.values() if a.class_col in t]
        ).value_counts()
        raise SystemExit(
            f'No cell has the class {missing} in column {a.class_col!r}. '
            f'The column holds:\n{have.to_string()}\n'
            'Use these names exactly (or just their number, e.g. 10).'
        )
    manual_poles = {}
    sessions = find_sessions(a.sessions) if a.sessions else {}
    for name in tables:
        session = sessions.get(image_key(name))
        if session is not None:
            poles = read_manual_poles(session)
            if poles is not None:
                manual_poles[name] = poles
    if manual_poles:
        print(f'hand-placed phloem poles used for: {sorted(manual_poles)}')
    scales = None
    if a.scales:
        sc = read_table(a.scales)
        scales = dict(zip(sc.iloc[:, 0], sc.iloc[:, 1]))
    results, details = measure_all(
        tables,
        um_per_px=a.pixel_size,
        scales=scales,
        manual=manual_poles,
        phloem_classes=a.phloem_class,
        phloem_levels=pooling_levels(a.phloem_order)
        if a.phloem_order
        else None,
        min_sep=a.min_sep,
        min_pole_cells=a.min_pole_cells,
        class_col=a.class_col,
        xylem_classes=a.xylem_class,
        trim=a.trim,
    )
    if a.drop_unclear:
        bad = results['phloem_level'] == 'unclear'
        cols = [c for c in results.columns if c.startswith('distance')]
        results.loc[bad, cols] = np.nan
    unit = 'um' if results['um_per_px'].notna().any() else 'px'
    use = a.use or ('distance_xylem' if a.xylem_class else 'distance')
    use_col = f'{use}_{unit}'

    sheets = {'results': results}
    summary = [
        ('images', len(results)),
        ('with a flag', int((results['flags'] != '').sum())),
        ('compared column', use_col),
    ]
    merged = None
    if a.manual:
        if not (a.manual_image_col and a.manual_value_col):
            raise SystemExit(
                '--manual needs --manual-image-col and --manual-value-col'
            )
        manual = read_table(a.manual, a.manual_sheet)
        merged, stats, unmatched = compare_with_manual(
            results, manual, a.manual_image_col, a.manual_value_col, use_col
        )
        sheets['comparison'] = merged
        summary += list(stats.items())
        if unmatched:
            summary.append(('unmatched names', ', '.join(map(str, unmatched))))
        if stats['n'] >= 2:
            plot_comparison(merged, stats, a.out + '_comparison.png', use_col)
        print(
            f'compared {stats["n"]} images; '
            + ', '.join(f'{k}={v:.3g}' for k, v in stats.items() if k != 'n')
        )
        if unmatched:
            print('names without a partner:', unmatched)
    sheets['summary'] = pd.DataFrame(summary, columns=['item', 'value'])

    with pd.ExcelWriter(a.out + '.xlsx') as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
    if a.plots_dir:
        os.makedirs(a.plots_dir, exist_ok=True)
        no_image = []
        for _, row in results.iterrows():
            if not np.isfinite(row['distance_px']):
                continue
            name = row['image']
            out = os.path.join(a.plots_dir, f'{name}.png')
            session = sessions.get(image_key(name))
            if session is not None:
                plot_overlay(
                    name, row, details[name], session_image(session), out
                )
            else:
                no_image.append(name)
                plot_image(name, row, details[name], out)
        if a.sessions and no_image:
            print('no session image for (plain plot made):', no_image)
    print(
        results[
            [
                'image',
                'phloem_level',
                'n_phloem_cells',
                'distance_px',
                f'{use}_{unit}',
                'flags',
            ]
        ].to_string(index=False)
    )
    print(f'\nWrote {a.out}.xlsx')


if __name__ == '__main__':
    main()
