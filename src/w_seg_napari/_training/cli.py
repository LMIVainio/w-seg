"""Train the W-net from the command line or from the plugin's Train tab.

    python -m w_seg_napari._training --train-dir A --val-dir B --out-root DIR --name run1

A run is written to <out-root>/<name>/ :

    model.keras, config.json   the best model (highest validation Dice): a
                               bundle the plugin can load directly
    run/progress.png           example crops + loss curves, rewritten at every evaluation
    run/log.csv                one row per evaluation
    run/run_settings.json      every setting of the run
    run/last/                  the latest model

The plugin starts this as a separate process and reads lines that begin with
"@@" (JSON) for its progress bar. Creating the file given by --stop-file ends
the run cleanly after the current step.
"""

import os

os.environ.setdefault('KERAS_BACKEND', 'torch')  # before importing keras

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import keras
import numpy as np
import torch

from w_seg_napari._model_config import (
    check_bundle,
    load_bundle,
    make_config,
    save_bundle,
)
from w_seg_napari._training.data import LargeTiffCellDataset
from w_seg_napari._training.losses import WeightedFocalTverskyLoss
from w_seg_napari._training.model import Generator
from w_seg_napari._training.steps import train_step, validate


def emit(event, **kw):
    """One machine-readable progress line for the plugin."""
    print('@@' + json.dumps({'event': event, **kw}), flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--train-dir', required=True, help='folder of 5-plane TIFFs'
    )
    p.add_argument('--val-dir', required=True, help='folder of 5-plane TIFFs')
    p.add_argument(
        '--out-root', required=True, help='folder that will hold <name>/'
    )
    p.add_argument('--name', default='run')
    p.add_argument('--stop-file', default=None)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--eval-every', type=int, default=200)
    p.add_argument('--val-samples', type=int, default=64)
    p.add_argument('--ranges', type=int, nargs='+', default=[6, 32])
    p.add_argument('--zoom', type=float, nargs=2, default=[0.5, 1.5])
    p.add_argument(
        '--widths', type=int, nargs='+', default=[32, 64, 128, 256, 512]
    )
    p.add_argument('--tissue-bias', type=float, default=0.7)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--keep-history', action='store_true')
    p.add_argument(
        '--init-from',
        default=None,
        help='model bundle folder to continue training from',
    )
    return p.parse_args(argv)


def denoise(t):
    """Same bilateral blur as the plugin (input [1, 3, h, w] in 0..1)."""
    import kornia.filters as K

    return K.bilateral_blur(
        t, kernel_size=(5, 5), sigma_color=0.1, sigma_space=(1.5, 1.5)
    )


def infinite(loader):
    while True:
        yield from loader


def pick_informative(dataset, rng, min_wall=0.03, tries=40):
    """A random (index, crop) that shows at least `min_wall` wall pixels."""
    for _ in range(tries):
        index = rng.randrange(len(dataset))
        crop = dataset[index]
        if float(crop[1][..., 0].mean()) >= min_wall:
            break
    return index, crop


@torch.no_grad()
def predict_sample(model, index, crop, device):
    x, y, _ = crop
    out = model(x[None].float().to(device), training=False)

    def to_np(t):
        return keras.ops.convert_to_numpy(t)[0]

    return {
        'index': index,
        'x': x.numpy(),
        'y': y.numpy(),
        'pred_wall': to_np(out['semantic_head'])[..., 0],
        'pred_aff': to_np(out['affinity_head']),
    }


def unique_dir(root, name):
    out = os.path.join(root, name)
    n = 2
    while os.path.exists(os.path.join(out, 'config.json')):
        out = os.path.join(root, f'{name}_{n}')
        n += 1
    return out


def main(argv=None):
    args = parse_args(argv)
    import matplotlib

    matplotlib.use('Agg')
    from w_seg_napari._training.viz import make_progress_figure

    random.seed(args.seed)
    np.random.seed(args.seed)
    keras.utils.set_random_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bundle_dir = unique_dir(args.out_root, args.name)
    run_dir = os.path.join(bundle_dir, 'run')
    os.makedirs(os.path.join(run_dir, 'history'), exist_ok=True)
    emit('start', bundle_dir=bundle_dir, run_dir=run_dir, device=device)

    base_model = None
    if args.init_from:
        # Continue from an existing bundle: its config (ranges, preprocessing)
        # wins over the command-line values, because the weights depend on it.
        base_model, cfg = load_bundle(args.init_from)
        check_bundle(base_model, cfg)
        args.ranges = list(cfg['affinity']['ranges'])
        for pt in Path(args.init_from).glob('*.pt'):  # keep the paired YOLO
            shutil.copy2(pt, bundle_dir)
    else:
        cfg = make_config(ranges=tuple(args.ranges))
    offsets = cfg['affinity']['offsets']
    pre = cfg['preprocessing']

    common = {
        'offsets': offsets,
        'denoise_fn': denoise,
        'tissue_bias': args.tissue_bias,
        'enable_lab': pre['enable_lab'],
        'hue_encoding': pre['hue_encoding'],
    }
    emit(
        'status',
        message='Preparing training data (first run builds a cache)...',
    )
    train_ds = LargeTiffCellDataset(
        args.train_dir,
        samples_per_epoch=args.batch_size * 100,
        scale_range=tuple(args.zoom),
        **common,
    )
    val_ds = LargeTiffCellDataset(
        args.val_dir,
        samples_per_epoch=args.val_samples,
        augment=False,
        photometric=False,
        seed=1234,
        **common,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=device == 'cuda',
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False
    )
    batches = infinite(train_loader)

    model = (
        base_model
        if base_model is not None
        else Generator(n_affinity=len(offsets), widths=tuple(args.widths))
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )

    def lr_factor(step):  # linear warm-up, then cosine decay to 5 %
        if step < args.warmup:
            return (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    ftl = WeightedFocalTverskyLoss()

    if base_model is not None:
        base = validate(
            model, val_loader, ftl, device, max_batches=len(val_loader)
        )
        emit(
            'status',
            message=f'Starting from {os.path.basename(args.init_from)}: '
            f'validation Dice before training {base["dice"]:.3f}',
        )
        print(f'Base model validation Dice: {base["dice"]:.4f}', flush=True)

    settings = dict(
        vars(args),
        device=device,
        n_parameters=int(model.count_params()),
        offsets=offsets,
        loss={'alpha': ftl.alpha, 'beta': ftl.beta, 'gamma': ftl.gamma},
        optimizer='AdamW weight_decay=1e-4, linear warm-up + cosine to 5 %',
    )
    with open(os.path.join(run_dir, 'run_settings.json'), 'w') as f:
        json.dump(settings, f, indent=2)
    print(
        f'Run "{args.name}": {settings["n_parameters"]:,} parameters on {device}; '
        f'{len(offsets)} affinity channels; output in {bundle_dir}/',
        flush=True,
    )

    history = {
        k: []
        for k in (
            'step', 'train_total', 'train_sem', 'train_aff',
            'eval_step', 'val_total', 'val_sem', 'val_aff', 'val_dice',
        )
    }  # fmt: skip
    log_path = os.path.join(run_dir, 'log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            [
                'step',
                'train_total',
                'val_total',
                'val_sem',
                'val_aff',
                'val_dice',
                'lr',
                'minutes',
            ]
        )

    rng = random.Random(args.seed)
    best_dice, t0, step = -1.0, time.time(), 0
    fig = None

    def evaluate_and_draw():
        nonlocal best_dice, fig
        import matplotlib.pyplot as plt

        val = validate(
            model, val_loader, ftl, device, max_batches=len(val_loader)
        )
        for k in ('total', 'sem', 'aff', 'dice'):
            history[f'val_{k}'].append(val[k])
        history['eval_step'].append(step)

        recent = float(np.mean(history['train_total'][-args.eval_every :]))
        if val['dice'] > best_dice:
            best_dice = val['dice']
            save_bundle(model, bundle_dir, cfg)  # the best model IS the bundle
        save_bundle(model, os.path.join(run_dir, 'last'), cfg)

        minutes = (time.time() - t0) / 60
        lr = scheduler.get_last_lr()[0]
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow(
                [step, f'{recent:.4f}', f'{val["total"]:.4f}', f'{val["sem"]:.4f}',
                 f'{val["aff"]:.4f}', f'{val["dice"]:.4f}', f'{lr:.2e}', f'{minutes:.1f}']
            )  # fmt: skip
        print(
            f'step {step:6d} | train {recent:.3f} | val {val["total"]:.3f} '
            f'(wall {val["sem"]:.3f}, aff {val["aff"]:.3f}) | val Dice {val["dice"]:.3f} '
            f'| best {best_dice:.3f} | {minutes:.1f} min',
            flush=True,
        )

        train_sample = predict_sample(
            model, *pick_informative(train_ds, rng), device
        )
        val_sample = predict_sample(
            model, *pick_informative(val_ds, rng), device
        )
        if fig is not None:
            plt.close(fig)
        fig = make_progress_figure(
            history, train_sample, val_sample, offsets, args.ranges,
            title=f'{args.name} | step {step} | lr {lr:.1e}',
        )  # fmt: skip
        progress_png = os.path.join(run_dir, 'progress.png')
        fig.savefig(progress_png, dpi=80)
        if args.keep_history:
            fig.savefig(
                os.path.join(run_dir, 'history', f'step_{step:06d}.png'),
                dpi=60,
            )
        emit(
            'eval', step=step, steps=args.steps, val_dice=val['dice'],
            best_dice=best_dice, val_total=val['total'], train_total=recent,
            minutes=minutes, progress_png=progress_png,
        )  # fmt: skip

    def stop_requested():
        return bool(args.stop_file) and os.path.exists(args.stop_file)

    stopped = False
    try:
        while step < args.steps:
            if stop_requested():
                stopped = True
                break
            parts = train_step(model, optimizer, next(batches), ftl, device)
            scheduler.step()
            step += 1
            history['step'].append(step)
            history['train_total'].append(parts['total'])
            history['train_sem'].append(parts['sem'])
            history['train_aff'].append(parts['aff'])
            if not np.isfinite(parts['total']):
                emit(
                    'error',
                    message=f'Loss became {parts["total"]} at step {step}.',
                )
                sys.exit(
                    f'Loss became {parts["total"]} at step {step}; stopping.'
                )
            if step % 10 == 0:
                emit('step', step=step, steps=args.steps, loss=parts['total'])
            if step == 1 or step % args.eval_every == 0:
                evaluate_and_draw()
    except KeyboardInterrupt:
        stopped = True
    if stopped:
        print('\nStopped: saving the current state.', flush=True)
    if history['step'] and (
        not history['eval_step'] or history['eval_step'][-1] != step
    ):
        evaluate_and_draw()
    print(
        f'Done. Best val Dice {best_dice:.4f}. Files in {bundle_dir}/',
        flush=True,
    )
    emit(
        'done',
        bundle_dir=bundle_dir,
        best_dice=best_dice,
        stopped=stopped,
        step=step,
    )
