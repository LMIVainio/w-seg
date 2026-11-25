"""Put a crude tissue mask into session files, as a napari layer to edit.

    python scripts/make_tissue_mask.py SESSION_OR_FOLDER [...] [--source auto]

For every session a layer called `tissue_mask` (a Labels layer, 1 = tissue) is
made from the cell labels of the session: where the cells are, with the gaps
between them and the holes of deleted cells closed. It is stored in the
session (extra/) and shows up as a layer when you open the session in napari.
Paint it into shape (1 = tissue, erase what is not tissue: peeling layer,
secondary roots, debris) and save the edits of the session. The tissue mask
training (scripts/train_tissue_mask.py --sessions ...) uses this layer.

SESSION_OR_FOLDER  *_session.h5 files, or folders: the batch's wseg_batch
                   folder, or the image folder that contains it.
--source           which labels to start from: corrected (the corrected
                   labels), gasp (the automatic labels), auto = corrected if
                   the session has them, else gasp. The automatic labels also
                   cover whatever should be left out, so there is more to erase.
--grow             how far gaps are closed, in median cell diameters (1.0)
--size             the long side of the layer in pixels (1024): small enough
                   to paint on comfortably; it is placed over the full image
--overwrite        replace a tissue_mask layer that is already in the session.
                   Without it such sessions are left alone, so your painting
                   is never lost by running this again.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from w_seg_napari import _tissue_mask as tm
from w_seg_napari._session_io import (
    CORRECTED,
    SESSION_SUFFIX,
    editable_labels,
    read_extras,
    save_edits,
)


def session_files(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            sub = p / 'wseg_batch'
            files += sorted(
                (sub if sub.is_dir() else p).glob('*' + SESSION_SUFFIX)
            )
        elif p.is_file():
            files.append(p)
        else:
            print(f'Not found: {p}', file=sys.stderr)
    return files


def read_labels(path, source):
    """(labels, origin) from the session, or (None, reason)."""
    import h5py

    keys = {
        'corrected': (CORRECTED,),
        'gasp': ('gasp_labels', 'sp_labels'),
        'auto': (CORRECTED, 'gasp_labels', 'sp_labels'),
    }[source]
    with h5py.File(path, 'r') as f:
        store = {k: f[k][()] for k in keys if k in f}
    if not store:
        return None, f'no {source} labels in the session'
    labels, origin = editable_labels(store)
    return np.squeeze(labels), origin


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('sessions', nargs='+')
    p.add_argument(
        '--source', choices=('auto', 'corrected', 'gasp'), default='auto'
    )
    p.add_argument('--grow', type=float, default=1.0)
    p.add_argument('--size', type=int, default=1024)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()

    files = session_files(a.sessions)
    if not files:
        sys.exit('No session files found.')
    made = kept = failed = 0
    for path in files:
        name = path.name.removesuffix(SESSION_SUFFIX)
        try:
            existing = read_extras(path)
            has = any(q['name'] == tm.LAYER_NAME for q in existing)
            if has and not a.overwrite:
                print(f'  {name}: already has a tissue_mask layer, left alone')
                kept += 1
                continue
            labels, origin = read_labels(path, a.source)
            if labels is None:
                print(f'  {name}: skipped, {origin}')
                failed += 1
                continue
            mask = tm.make_target_from_labels(
                labels, a.size, grow=a.grow, squash=False
            )
            layer = tm.mask_to_layer(mask, labels.shape)
            others = [q for q in existing if q['name'] != tm.LAYER_NAME]
            layer['order'] = max([q.get('order', 0) for q in others] + [0]) + 1
            save_edits(path, extras=[*others, layer])
            print(
                f'  {name}: tissue_mask from the {origin} labels, '
                f'{mask.mean():.0%} of the image is tissue'
            )
            made += 1
        except Exception as e:
            print(f'  {name}: failed, {e}')
            failed += 1
    print(f'{made} made, {kept} left alone, {failed} skipped or failed')


if __name__ == '__main__':
    main()
