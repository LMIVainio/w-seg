"""Train the coarse tissue mask model (whole image at 512 x 512 -> tissue).

    python scripts/train_tissue_mask.py TRAIN_FOLDER [TRAIN_FOLDER ...] \
        --out tissue_mask_run [--masks PAINTED_DIR] [--epochs 60]

TRAIN_FOLDER  folders of the 5-plane training tifs (planes 0-2 = RGB, 3 = wall
              mask), the same files the cell-wall model is trained on.

The target is made automatically: tissue = "contains cell walls", with the
gaps between cells closed (--close, in pixels of a 1024 px picture). To keep a
peeling layer or debris OUT of the tissue, paint those images by hand:

    --masks DIR   binary pictures named <slide name>.png / .tif (any non-zero
                  pixel = tissue). They replace the automatic target of the
                  slide of the same name; other slides keep the automatic one.

Every 5th slide (--val-every) is held out; the best epoch by validation Dice
is kept. Writes <out>/tissue_mask.keras and tissue_mask.json. Pack them with a
model:  python scripts/pack_model_zip.py MODEL_DIR --name NAME --tissue-mask <out>
Only 20-50 slides are needed and a few minutes of training.
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'torch')

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402

from w_seg_napari import _tissue_mask as tm  # noqa: E402


def load_slide(path, close, masks_dir):
    """(rgb [STORE, STORE, 3] float16 in -1..1, target [STORE, STORE] bool)."""
    try:
        stack = tifffile.memmap(path)
    except Exception:
        stack = tifffile.imread(path)
    if stack.ndim != 3 or stack.shape[0] < 4:
        raise ValueError(f'{path}: expected a planar stack with >= 4 planes')
    _, H, W = stack.shape
    step = max(1, int(max(H, W) // (2 * tm.STORE)))
    rgb = np.moveaxis(np.asarray(stack[:3, ::step, ::step]), 0, -1)
    x = tm.prepare_input(rgb, tm.STORE).astype(np.float16)

    stem = Path(path).stem
    painted = None
    if masks_dir:
        for ext in ('.png', '.tif', '.tiff'):
            f = Path(masks_dir) / f'{stem}{ext}'
            if f.is_file():
                painted = (
                    tifffile.imread(f)
                    if ext != '.png'
                    else cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                )
                break
    if painted is not None:
        painted = np.squeeze(painted)
        if painted.ndim == 3:
            painted = painted.max(axis=0 if painted.shape[0] < 5 else -1)
        y = cv2.resize(
            (painted > 0).astype(np.uint8),
            (tm.STORE, tm.STORE),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    else:
        y = tm.make_target(stack[3], tm.STORE, close=close)
    return x, y, painted is not None


def augment(x, y, rng, out=tm.SIZE):
    """Random crop (zoom), flips, rotation by 90 degrees, colour jitter."""
    S = x.shape[0]
    h = int(S * rng.uniform(0.55, 1.0))
    w = int(S * rng.uniform(0.55, 1.0))
    y0 = rng.randint(0, S - h)
    x0 = rng.randint(0, S - w)
    xi = x[y0 : y0 + h, x0 : x0 + w].astype(np.float32)
    yi = y[y0 : y0 + h, x0 : x0 + w].astype(np.uint8)
    xi = cv2.resize(xi, (out, out), interpolation=cv2.INTER_AREA)
    yi = cv2.resize(yi, (out, out), interpolation=cv2.INTER_NEAREST)
    if rng.random() < 0.5:
        xi, yi = xi[:, ::-1], yi[:, ::-1]
    if rng.random() < 0.5:
        xi, yi = xi[::-1], yi[::-1]
    k = rng.randint(0, 3)
    xi, yi = np.rot90(xi, k), np.rot90(yi, k)
    gain = np.array([rng.uniform(0.9, 1.1) for _ in range(3)], np.float32)
    xi = xi * rng.uniform(0.85, 1.15) * gain + rng.uniform(-0.1, 0.1)
    return np.clip(xi, -1, 1).astype(np.float32), yi.astype(np.float32)


def val_pair(x, y, out=tm.SIZE):
    xi = cv2.resize(
        x.astype(np.float32), (out, out), interpolation=cv2.INTER_AREA
    )
    yi = cv2.resize(
        y.astype(np.uint8), (out, out), interpolation=cv2.INTER_NEAREST
    )
    return xi, yi.astype(np.float32)


def dice_iou(prob_bool, truth_bool):
    inter = float((prob_bool & truth_bool).sum())
    a, b = float(prob_bool.sum()), float(truth_bool.sum())
    union = a + b - inter
    return (2 * inter + 1) / (a + b + 1), (inter + 1) / (union + 1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('folders', nargs='+')
    p.add_argument('--out', default='tissue_mask')
    p.add_argument('--masks', default=None)
    p.add_argument('--close', type=int, default=8)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--val-every', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    import torch

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)

    paths = []
    for f in a.folders:
        paths += sorted(
            str(q)
            for q in Path(f).iterdir()
            if q.suffix.lower() in ('.tif', '.tiff')
        )
    if not paths:
        sys.exit('No tif files found.')
    print(f'{len(paths)} slides')
    slides, painted_n = [], 0
    for i, path in enumerate(paths, 1):
        try:
            x, y, painted = load_slide(path, a.close, a.masks)
        except Exception as e:
            print(f'  skipped {path}: {e}')
            continue
        slides.append((Path(path).name, x, y))
        painted_n += painted
        print(
            f'  {i}/{len(paths)} {Path(path).name}  tissue {y.mean():.0%}'
            + ('  (painted mask)' if painted else '')
        )
    if painted_n:
        print(f'{painted_n} slides use a painted mask')

    val = [
        s
        for i, s in enumerate(slides)
        if a.val_every and i % a.val_every == a.val_every - 1
    ]
    train = [s for s in slides if s not in val] or slides
    if not val:
        print(
            'Too few slides for a validation set: the best epoch cannot be chosen.'
        )
    print(f'train {len(train)}, validation {len(val)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = tm.build_tissue_unet()
    model.to(device)
    print(f'{model.count_params():,} weights on {device}')
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_per_epoch = max(10, len(train) * 4 // a.batch_size)
    total = a.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(s / total, 1.0)))
        ),
    )
    bce = torch.nn.BCEWithLogitsLoss()

    def loss_fn(logits, y):
        prob = torch.sigmoid(logits)
        inter = (prob * y).sum(dim=(1, 2, 3))
        dice = 1 - (2 * inter + 1) / (
            prob.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + 1
        )
        return bce(logits, y) + dice.mean()

    best, best_epoch = -1.0, -1
    for epoch in range(1, a.epochs + 1):
        model.train()
        running = 0.0
        for _ in range(steps_per_epoch):
            pairs = [
                augment(*rng.choice(train)[1:], rng)
                for _ in range(a.batch_size)
            ]
            xb = torch.from_numpy(np.stack([q[0] for q in pairs])).to(device)
            yb = torch.from_numpy(
                np.stack([q[1] for q in pairs])[..., None]
            ).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb, training=True), yb)
            loss.backward()
            opt.step()
            sched.step()
            running += float(loss.detach())
        msg = f'epoch {epoch:3d}  loss {running / steps_per_epoch:.4f}'
        score = -running
        if val:
            model.eval()
            d, j = [], []
            with torch.no_grad():
                for _, x, y in val:
                    xi, yi = val_pair(x, y)
                    out = model(
                        torch.from_numpy(xi[None]).to(device), training=False
                    )
                    pb = torch.sigmoid(out)[0, ..., 0].cpu().numpy() > 0.5
                    di, ji = dice_iou(pb, yi > 0.5)
                    d.append(di)
                    j.append(ji)
            score = float(np.mean(d))
            msg += f'  val Dice {score:.3f}  IoU {np.mean(j):.3f}  (worst Dice {min(d):.3f})'
        print(msg, flush=True)
        if score > best:
            best, best_epoch = score, epoch
            tm.save_tissue_mask(model, a.out)
    print(f'Best epoch {best_epoch} (score {best:.3f}); saved in {a.out}')


if __name__ == '__main__':
    main()
