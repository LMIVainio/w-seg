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
    load_tables,
    measure_all,
    plot_comparison,
    plot_image,
)


def read_table(path, sheet=None):
    if str(path).lower().endswith('.csv'):
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=sheet or 0)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('tables')
    p.add_argument('--phloem-class', nargs='+', required=True,
                   help='class id(s) / name(s) of the primary phloem cells')
    p.add_argument('--xylem-class', nargs='+', default=None,
                   help='class id(s) of the primary xylem cells (optional)')
    p.add_argument('--class-col', default='predicted_class')
    p.add_argument('--sheet', default=None, help='xlsx sheet of the tables')
    p.add_argument('--pixel-size', type=float, default=None, help='um per pixel')
    p.add_argument('--scales', default=None,
                   help='table: image name, um per pixel (overrides --pixel-size)')
    p.add_argument('--trim', type=float, default=3.0,
                   help='drop cells farther than this x median spread from their '
                   'pole and refit (0 = off)')
    p.add_argument('--manual', default=None, help='xlsx / csv with manual values')
    p.add_argument('--manual-sheet', default=None)
    p.add_argument('--manual-image-col', default=None)
    p.add_argument('--manual-value-col', default=None)
    p.add_argument('--use', choices=['distance', 'distance_xylem'], default=None,
                   help='which measurement to compare (default: xylem-corrected '
                   'when xylem classes are given)')
    p.add_argument('--out', default='phloem_distance')
    p.add_argument('--plots-dir', default=None)
    a = p.parse_args()

    tables = load_tables(a.tables, a.sheet)
    scales = None
    if a.scales:
        sc = read_table(a.scales)
        scales = dict(zip(sc.iloc[:, 0], sc.iloc[:, 1]))
    results, details = measure_all(
        tables,
        um_per_px=a.pixel_size,
        scales=scales,
        phloem_classes=a.phloem_class,
        class_col=a.class_col,
        xylem_classes=a.xylem_class,
        trim=a.trim,
    )
    unit = 'um' if results['um_per_px'].notna().any() else 'px'
    use = a.use or ('distance_xylem' if a.xylem_class else 'distance')
    use_col = f'{use}_{unit}'

    sheets = {'results': results}
    summary = [('images', len(results)),
               ('with a flag', int((results['flags'] != '').sum())),
               ('compared column', use_col)]
    merged = None
    if a.manual:
        if not (a.manual_image_col and a.manual_value_col):
            raise SystemExit('--manual needs --manual-image-col and --manual-value-col')
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
        print(f'compared {stats["n"]} images; ' + ', '.join(
            f'{k}={v:.3g}' for k, v in stats.items() if k != 'n'))
        if unmatched:
            print('names without a partner:', unmatched)
    sheets['summary'] = pd.DataFrame(summary, columns=['item', 'value'])

    with pd.ExcelWriter(a.out + '.xlsx') as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
    if a.plots_dir:
        os.makedirs(a.plots_dir, exist_ok=True)
        for _, row in results.iterrows():
            if np.isfinite(row['distance_px']):
                plot_image(row['image'], row, details[row['image']],
                           os.path.join(a.plots_dir, f'{row["image"]}.png'))
    print(results[['image', 'n_phloem_cells', 'distance_px', f'{use}_{unit}', 'flags']].to_string(index=False))
    print(f'\nWrote {a.out}.xlsx')


if __name__ == '__main__':
    main()
