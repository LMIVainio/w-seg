"""Add hand-corrected cell classes to the batch's morphology tables.

In napari: open the image's session (Batch tab, "Open session"), copy the
layer `yolo_labels` to a new Labels layer named `yolo_class`, paint the right
class number over the cells that are wrong (10 = 10_primary_phloem_pole etc.,
use the paint brush, not the bucket: the bucket also fills touching cells of
the same class), and save the session with `yolo_class` ticked under "Also
keep these layers". Then:

    python scripts/apply_class_layer.py wseg_batch/

For every <image>_morphology.csv that has a session with a `yolo_class` layer
the columns `yolo_corrected` (the corrected class; the prediction for cells
you did not touch) and `yolo_edited` (True for the cells you changed) are
added to the table. Use --class-col yolo_corrected in phloem_distance.py or
the cell-count notebook to work with the corrected classes. Re-running the
batch's morphology stage rewrites the table, so run this again afterwards.

The same layer is the training layer of the YOLO classifier trainer.
"""

import argparse
from pathlib import Path

import numpy as np

from w_seg_napari import _class_layer as cl
from w_seg_napari import _session_io as sio


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('folder', help='batch folder with the tables and sessions')
    p.add_argument('--layer', default=cl.CLASS_LAYER)
    p.add_argument('--sessions', default=None, help='default: the same folder')
    p.add_argument('--class-col', default='yolo_pred_tissue')
    p.add_argument('--column', default='yolo_corrected')
    p.add_argument(
        '--min-changed',
        type=float,
        default=0.2,
        help='a cell counts as edited when this share of it was painted '
        'with a class other than the original (default 0.2)',
    )
    a = p.parse_args()

    folder = Path(a.folder)
    sess_dir = Path(a.sessions) if a.sessions else folder
    tables = sorted(folder.glob('*_morphology.csv'))
    if not tables:
        raise SystemExit(f'No *_morphology.csv in {folder}')
    n_done = 0
    for table in tables:
        name = table.name[: -len('_morphology.csv')]
        session = sess_dir / f'{name}{sio.SESSION_SUFFIX}'
        if not session.is_file():
            continue
        layers = {x['name']: x for x in sio.read_extras(session)}
        if a.layer not in layers:
            continue
        head, df = cl.read_table_keep_header(table)
        source = (
            str(df['labels_source'].iloc[0])
            if 'labels_source' in df.columns and len(df)
            else 'corrected'
        )
        labels = sio.read_dataset(
            session, cl.LABELS_DATASET.get(source, 'labels_corrected')
        )
        if labels is None:
            print(f'{name}: the session has no {source} labels, skipped')
            continue
        original = sio.read_dataset(session, 'yolo_labels')
        layer = np.squeeze(np.asarray(layers[a.layer]['data']))
        classes = cl.corrected_classes(
            labels, layer, original, min_changed=a.min_changed
        )
        out, unknown = cl.add_corrected_column(
            df, classes, a.class_col, a.column
        )
        cl.write_table_keep_header(table, head, out)
        n_edit = int(out['yolo_edited'].sum())
        print(f'{name}: {n_edit} cell(s) changed')
        if unknown:
            print(f'  class number(s) {unknown} are not in the table: ignored')
        n_done += 1
    print(f'{n_done} table(s) updated')


if __name__ == '__main__':
    main()
