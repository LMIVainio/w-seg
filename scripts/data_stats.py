"""Compare the images of a training folder (colour, contrast, sharpness).

    python scripts/data_stats.py D:/data/train [D:/data/val ...] [--out stats]

Pool several folders to see the variation the model has to deal with. Writes
<out>.csv and <out>.png and prints a summary plus the slides that stand out.
Use the spread column to choose the augmentation: a relative spread (IQR /
median) above about 0.2 for brightness, contrast or sharpness, or clear
white-balance differences, is a reason for --aug strong.
"""

import argparse
import os

from w_seg_napari._training.data_stats import (
    METRICS,
    collect_stats,
    flag_outliers,
    save_figure,
    spread_summary,
    write_csv,
)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('folders', nargs='+', help='folders of 5-plane tifs')
    p.add_argument('--out', default='slide_stats', help='output file stem')
    p.add_argument('--crops', type=int, default=12)
    a = p.parse_args()

    rows = []
    for folder in a.folders:
        part = collect_stats(folder, n_crops=a.crops)
        for r in part:
            r['file'] = (
                f'{os.path.basename(os.path.normpath(folder))}/{r["file"]}'
            )
        rows += part
    if not rows:
        raise SystemExit('No .tif files found.')

    write_csv(rows, a.out + '.csv')
    save_figure(rows, a.out + '.png')
    print(f'{len(rows)} slides\n')
    print(f'{"metric":18s} {"median":>10s} {"spread":>8s}  meaning')
    for m, (med, spread) in spread_summary(rows).items():
        print(f'{m:18s} {med:10.3g} {spread:8.2f}  {METRICS[m]}')
    flagged = flag_outliers(rows)
    print('\nSlides that stand out:' if flagged else '\nNo outlier slides.')
    for f, items in flagged.items():
        print(f'  {f}: ' + ', '.join(f'{m} ({z:+.1f})' for m, z in items))
    print(f'\nWrote {a.out}.csv and {a.out}.png')


if __name__ == '__main__':
    main()
