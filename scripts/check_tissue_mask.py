"""See what the tissue mask model does on your images.

    python scripts/check_tissue_mask.py MODEL_FOLDER IMAGE_OR_SESSION [...] \
        [--out mask_check] [--threshold 0.5]

MODEL_FOLDER  the folder with tissue_mask.keras + tissue_mask.json (the output
              of train_tissue_mask.py, or an installed model bundle)
IMAGE_OR_SESSION  image files, *_session.h5 files, or folders of either

For every image a picture <out>/<name>.png is written: the image with the
predicted tissue outline in red and, for a session with a painted
`tissue_mask` layer, your mask in green. Prints, per image:

  tissue    share of the image that is tissue by the model
  Dice/IoU  against your painted mask (sessions that have one)
  tiles     share of the 256 px tiles that the prediction would skip
            (tile_filter with the same rule as the batch tab)
  ms        time of the mask model

Put the images the model did NOT train on in here (a held-out session, the
frozen benchmark), otherwise the scores are too good.
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'torch')

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from w_seg_napari import _tissue_mask as tm  # noqa: E402
from w_seg_napari._batch_io import load_image, to_rgb  # noqa: E402

IMAGE_EXTS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}


def inputs(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            sub = p / 'wseg_batch'
            folder = sub if sub.is_dir() else p
            files += sorted(folder.glob('*_session.h5'))
            files += sorted(
                f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS
            )
        elif p.is_file():
            files.append(p)
    return files


def read_one(path):
    """(name, rgb, painted mask or None)."""
    if path.name.endswith('_session.h5'):
        import h5py

        with h5py.File(path, 'r') as f:
            if 'raw' not in f:
                raise ValueError('it holds no image (raw)')
            rgb = to_rgb(f['raw'][()])
        return (
            path.name.removesuffix('_session.h5'),
            rgb,
            tm.read_tissue_layer(path),
        )
    return path.stem, to_rgb(load_image(path)), None


def outline(picture, mask, colour):
    mask8 = cv2.resize(
        mask.astype(np.uint8),
        (picture.shape[1], picture.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    contours, _ = cv2.findContours(
        mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(picture, contours, -1, colour, 3)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('model')
    p.add_argument('images', nargs='+')
    p.add_argument('--out', default='mask_check')
    p.add_argument('--threshold', type=float, default=None)
    p.add_argument('--tile', type=int, default=256)
    p.add_argument('--min-tile', type=float, default=0.01)
    a = p.parse_args()

    import torch

    folder = tm.find_tissue_mask(a.model)
    if folder is None:
        sys.exit(
            f'No tissue mask model (tissue_mask.keras/.json) in {a.model}'
        )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = tm.load_tissue_mask(folder, device)
    files = inputs(a.images)
    if not files:
        sys.exit('No images found.')
    Path(a.out).mkdir(parents=True, exist_ok=True)

    dices = []
    for path in files:
        try:
            name, rgb, painted = read_one(path)
            t0 = time.time()
            look = tm.tissue_lookup(model, cfg, rgb, device, a.threshold)
            ms = (time.time() - t0) * 1000
        except Exception as e:
            print(f'{path.name}: failed, {e}')
            continue
        H, W = rgb.shape[:2]
        ys = range(0, H, a.tile)
        xs = range(0, W, a.tile)
        kept = sum(
            look.keep_tile(y, x, a.tile, a.min_tile) for y in ys for x in xs
        )
        total = len(ys) * len(xs)
        line = (
            f'{name}: tissue {look.mask.mean():.0%}, '
            f'tiles skipped {1 - kept / total:.0%}, {ms:.0f} ms'
        )
        step = max(1, int(np.ceil(max(H, W) / 1400)))
        picture = np.ascontiguousarray(rgb[::step, ::step])
        if picture.dtype != np.uint8:
            top = float(picture.max()) or 1.0
            picture = (picture.astype(np.float32) / top * 255).astype(np.uint8)
        outline(picture, look.mask, (255, 0, 0))
        if painted is not None:
            truth = (
                cv2.resize(
                    (np.squeeze(painted) > 0).astype(np.uint8),
                    (look.mask.shape[1], look.mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )
            inter = float((truth & look.mask).sum())
            d = (2 * inter + 1) / (truth.sum() + look.mask.sum() + 1)
            j = (inter + 1) / ((truth | look.mask).sum() + 1)
            dices.append(d)
            line += f', Dice {d:.3f}, IoU {j:.3f}'
            outline(picture, truth, (0, 200, 0))
        cv2.imwrite(
            str(Path(a.out) / f'{name}.png'),
            cv2.cvtColor(picture, cv2.COLOR_RGB2BGR),
        )
        print(line)
    if dices:
        print(
            f'mean Dice {np.mean(dices):.3f}, worst {min(dices):.3f} '
            f'over {len(dices)} image(s)'
        )
    print(f'Pictures in {a.out}/  (red = model, green = your mask)')


if __name__ == '__main__':
    main()
