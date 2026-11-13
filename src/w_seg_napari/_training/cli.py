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
import gc
import json
import math
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import keras
import numpy as np
import torch

from w_seg_napari._model_config import (
    check_bundle,
    has_geometry,
    load_bundle,
    make_config,
    save_bundle,
)
from w_seg_napari._training.autobatch import find_max_batch, safe_batch
from w_seg_napari._training.data import LargeTiffCellDataset
from w_seg_napari._training.losses import WeightedFocalTverskyLoss
from w_seg_napari._training.metrics import selection_score
from w_seg_napari._training.model import (
    Generator,
    add_input_stem,
    has_input_stem,
)
from w_seg_napari._training.steps import (
    compute_loss,
    forward,
    train_step,
    validate,
)


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
    p.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='0 = automatic: the largest batch that fits in GPU memory',
    )
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
        '--geometry',
        action='store_true',
        help='add the geometry head (architecture wasp_v4)',
    )
    p.add_argument(
        '--mixed-precision',
        action='store_true',
        help='run the layers in bfloat16 (needs a GPU with bf16 support); '
        'about half the activation memory',
    )
    p.add_argument(
        '--accum',
        type=int,
        default=1,
        help='gradient accumulation: batches per weight update (the update '
        'then uses batch-size x accum images)',
    )
    p.add_argument(
        '--aff-floor',
        type=float,
        default=1.0,
        help='minimum weight of the affinity loss (the EDT map is ~0.1 inside '
        'cells; 0 restores the old behaviour)',
    )
    p.add_argument(
        '--w-interior',
        type=float,
        default=0.5,
        help='weight of the interior term (false wall + affinity holes deep '
        'inside cells); 0 switches it off',
    )
    p.add_argument(
        '--hard-neg',
        type=float,
        default=2.0,
        help='extra loss weight for wall-free pixels that the network calls '
        'wall (hard negatives); 0 switches it off',
    )
    p.add_argument(
        '--w-dot',
        type=float,
        default=10.0,
        help='weight of the dot penalty (small wall-like specks inside '
        'cells, found with a morphological opening); 0 switches it off',
    )
    p.add_argument(
        '--w-cross',
        type=float,
        default=0.0,
        help="weight of the skeleton-on-wall penalty (the geometry head's "
        'cell skeleton must not lie on a true wall); geometry models only; '
        '0 switches it off',
    )
    p.add_argument(
        '--skel-boost',
        type=float,
        default=0.0,
        help='extra loss weight on the centre line of the true walls, as a '
        'multiple of the normal weight (a missed stretch of it is a gap in a '
        'wall); training data only; 0 switches it off',
    )
    p.add_argument(
        '--dot-radius',
        type=int,
        default=2,
        help='specks thinner than 2 * radius + 1 pixels count as dots',
    )
    p.add_argument(
        '--interior-margin',
        type=int,
        default=3,
        help='pixels from any wall that count as "deep inside a cell"',
    )
    p.add_argument(
        '--select',
        choices=['cells', 'dice'],
        default='cells',
        help="which score picks the best checkpoint: 'cells' = the F1 of the "
        'cells found between the predicted walls (smoothed over the last '
        "evaluations), 'dice' = the plain wall Dice at 0.5",
    )
    p.add_argument(
        '--select-smooth',
        type=int,
        default=3,
        help='evaluations averaged for the checkpoint score (1 = no smoothing)',
    )
    p.add_argument(
        '--aug',
        choices=['mild', 'strong'],
        default='mild',
        help="photometric augmentation: 'strong' adds gamma, stain strength, "
        'white balance, hue rotation, blur and sharpening (for data from '
        'different scanners / stainings)',
    )
    p.add_argument(
        '--input-stem',
        type=int,
        default=0,
        help='width of a learnable input stem (0 = none, 16 is a good start); '
        'starts as the identity, so --init-from weights stay valid',
    )
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


def check_mixed_precision(
    model, ftl, loss_kw, device, patch, in_ch, n_targets
):
    """
    True when bfloat16 training works here: a GPU with bf16 support and one
    real forward + backward pass in bf16 that runs without an error.
    Otherwise says why not and returns False (training then uses float32).
    """
    why = None
    if device != 'cuda':
        why = 'there is no GPU'
    elif torch.cuda.get_device_capability()[0] < 8:
        # bf16 is native from the Ampere generation (compute capability 8.0)
        why = (
            'this GPU is older than the Ampere generation (no native bfloat16)'
        )
    else:
        try:
            x = torch.rand(1, patch, patch, in_ch, device=device)
            y = torch.rand(1, patch, patch, n_targets, device=device)
            w = torch.ones(1, patch, patch, 1, device=device)
            total, _ = compute_loss(
                forward(model, x, True, True), y, w, ftl, **loss_kw
            )
            total.backward()
        except (RuntimeError, TypeError, ValueError) as e:
            why = f'a test step failed ({str(e)[:120]})'
        finally:
            for p in model.parameters():
                p.grad = None
            torch.cuda.empty_cache()
    if why:
        msg = f'Mixed precision switched off: {why}. Training in float32.'
        print(msg, flush=True)
        emit('status', message=msg)
        return False
    emit('status', message='Mixed precision (bfloat16) is on.')
    return True


def auto_batch_size(
    model, ftl, loss_kw, device, patch, in_ch, n_targets, amp=False
):
    """
    Largest comfortable batch size for this GPU, found by running real
    forward + backward passes on random data (no weight update).
    """
    if device != 'cuda':
        emit('status', message='No GPU: using batch size 4.')
        return 4
    free, total = torch.cuda.mem_get_info()
    gb = 1024**3
    emit(
        'status',
        message=f'GPU memory: {free / gb:.1f} GB free of {total / gb:.1f} GB. '
        'Finding the largest batch size...',
    )

    # What counts as "fits": no out-of-memory error AND the peak memory of the
    # step stays within 80 % of the free memory minus the optimizer's state
    # (AdamW keeps two float32 values per weight, which this probe does not
    # allocate). Without the second rule a size that only just squeezes in can
    # still fail once real training adds the optimizer, the validation pass and
    # allocator fragmentation.
    base = torch.cuda.memory_allocated()
    budget = 0.8 * free - 8 * int(model.count_params())

    def fits(b):
        ok = True
        torch.cuda.reset_peak_memory_stats()
        try:
            x = torch.rand(b, patch, patch, in_ch, device=device)
            y = torch.rand(b, patch, patch, n_targets, device=device)
            w = torch.ones(b, patch, patch, 1, device=device)
            total_loss, _ = compute_loss(
                forward(model, x, True, amp), y, w, ftl, **loss_kw
            )
            total_loss.backward()
            ok = torch.cuda.max_memory_allocated() - base <= budget
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if 'out of memory' not in str(e).lower():
                raise
            ok = False
        finally:
            for p in model.parameters():
                p.grad = None
            x = y = w = total_loss = None
            gc.collect()
            torch.cuda.empty_cache()
        return ok

    best = find_max_batch(fits)
    if best == 0:
        raise RuntimeError(
            'Not even a batch of 1 fits in GPU memory. Close other programs '
            'that use the GPU, or train with smaller --widths.'
        )
    chosen = safe_batch(best)
    emit(
        'status',
        message=f'Largest batch that fits: {best}; training with {chosen}.',
    )
    print(
        f'Auto batch size: largest {best}, using {chosen} '
        f'({free / gb:.1f} GB free of {total / gb:.1f} GB)',
        flush=True,
    )
    return chosen


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

    extra = {}
    if 'geometry_head' in out:
        extra['pred_geo'] = to_np(out['geometry_head'])
    return {
        **extra,
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
        cfg['base_model'] = cfg.get('name') or os.path.basename(args.init_from)
        cfg['name'] = args.name
        cfg['created'] = datetime.now(timezone.utc).strftime(
            '%Y-%m-%d %H:%M UTC'
        )
        args.ranges = list(cfg['affinity']['ranges'])
        for pt in Path(args.init_from).glob('*.pt'):  # keep the paired YOLO
            shutil.copy2(pt, bundle_dir)
    else:
        cfg = make_config(
            ranges=tuple(args.ranges), name=args.name, geometry=args.geometry
        )
    offsets = cfg['affinity']['offsets']
    pre = cfg['preprocessing']
    geometry = cfg['geometry'] if has_geometry(cfg) else None
    args.geometry = geometry is not None

    common = {
        'offsets': offsets,
        'denoise_fn': denoise,
        'tissue_bias': args.tissue_bias,
        'enable_lab': pre['enable_lab'],
        'hue_encoding': pre['hue_encoding'],
        'geometry': geometry,
    }
    model = (
        base_model
        if base_model is not None
        else Generator(
            n_affinity=len(offsets),
            widths=tuple(args.widths),
            geometry=geometry is not None,
        )
    )
    if has_input_stem(model):
        cfg['architecture'].setdefault('input_stem', True)
    elif args.input_stem > 0:
        model = add_input_stem(model, args.input_stem)
        cfg['architecture']['input_stem'] = int(args.input_stem)
    ftl = WeightedFocalTverskyLoss()
    loss_kw = {
        'aff_floor': args.aff_floor,
        'w_interior': args.w_interior,
        'interior_margin': args.interior_margin,
        'hard_neg': args.hard_neg,
        'w_dot': args.w_dot,
        'w_cross': args.w_cross,
        'dot_radius': args.dot_radius,
    }
    n_targets = 1 + len(offsets) + (4 if geometry is not None else 0)
    probe = (
        cfg['input']['patch_size'],
        cfg['input']['channels'],
        n_targets,
    )
    amp = False
    if args.mixed_precision:
        amp = check_mixed_precision(model, ftl, loss_kw, device, *probe)
    if (
        args.batch_size <= 0
    ):  # automatic: as large as the free GPU memory allows
        args.batch_size = auto_batch_size(
            model, ftl, loss_kw, device, *probe, amp=amp
        )
    emit(
        'status',
        message='Preparing training data (first run builds a cache)...',
    )
    train_ds = LargeTiffCellDataset(
        args.train_dir,
        samples_per_epoch=args.batch_size * 100,
        scale_range=tuple(args.zoom),
        photometric_strength=args.aug,
        skel_boost=args.skel_boost,
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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )

    def lr_factor(step):  # linear warm-up, then cosine decay to 5 %
        if step < args.warmup:
            return (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    if base_model is not None:
        base = validate(
            model,
            val_loader,
            ftl,
            device,
            max_batches=len(val_loader),
            amp=amp,
            **loss_kw,
        )
        emit(
            'status',
            message=f'Starting from {os.path.basename(args.init_from)}: '
            f'validation Dice before training {base["dice"]:.3f}',
        )
        print(f'Base model validation Dice: {base["dice"]:.4f}', flush=True)
        if base['dice'] < 0.05:
            msg = (
                'The base model scores a Dice of about 0 on this validation '
                'set. It may be untrained (for example the saved model of a '
                'very short test run) or from different data.'
            )
            print(msg, flush=True)
            emit('status', message=msg)

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
            'val_int_wall', 'val_int_aff', 'val_int_dots',
            'train_geo', 'val_geo', 'train_interior', 'val_interior',
            'val_dice_best', 'val_thr_best', 'val_precision', 'val_recall',
            'val_dice_img', 'val_dice_img_sd', 'val_tol_f1', 'val_cldice',
            'val_cell_f1', 'val_score', 'val_skel_on_wall',
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
                'val_int_wall',
                'val_int_aff',
                'val_int_dots',
                'val_dice_best',
                'val_thr_best',
                'val_precision',
                'val_recall',
                'val_dice_img',
                'val_dice_img_sd',
                'val_tol_f1',
                'val_cldice',
                'val_cell_f1',
                'val_score',
                'val_skel_on_wall',
                'lr',
                'minutes',
            ]
        )

    rng = random.Random(args.seed)
    best_dice, best_loss, t0, step = -1.0, float('inf'), time.time(), 0
    best_score = -1.0
    fig = None

    def evaluate_and_draw():
        nonlocal best_dice, best_loss, best_score, fig
        import matplotlib.pyplot as plt

        val = validate(
            model,
            val_loader,
            ftl,
            device,
            max_batches=len(val_loader),
            amp=amp,
            **loss_kw,
        )
        for k in (
            'total',
            'sem',
            'aff',
            'dice',
            'int_wall',
            'int_aff',
            'int_dots',
        ):
            history[f'val_{k}'].append(val[k])
        for k in (
            'dice_best', 'thr_best', 'precision', 'recall', 'dice_img',
            'dice_img_sd', 'tol_f1', 'cldice', 'cell_f1', 'skel_on_wall',
        ):  # fmt: skip
            history[f'val_{k}'].append(val[k])
        history['val_geo'].append(val['geo'])
        history['val_interior'].append(val['int_wall'] + val['int_aff'])
        history['eval_step'].append(step)

        recent = float(np.mean(history['train_total'][-args.eval_every :]))
        # Best = highest score (cell F1 averaged over the last evaluations, or
        # the plain Dice); at an equal score (for example 0 early on) the
        # lower validation loss wins, so the saved model is never just the
        # step-1 one.
        key = 'val_cell_f1' if args.select == 'cells' else 'val_dice'
        score = selection_score(history, key, args.select_smooth)
        history['val_score'].append(score)
        if score > best_score or (
            score == best_score and val['total'] < best_loss
        ):
            best_score, best_loss = score, val['total']
            best_dice = val['dice']
            cfg['training'] = {
                'steps_done': step,
                'steps_planned': args.steps,
                'val_dice': round(float(val['dice']), 4),
                'val_cell_f1': round(float(val['cell_f1']), 4),
                'selected_by': args.select,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'loss': dict(loss_kw),
                'mixed_precision': amp,
                'accumulation_steps': args.accum,
                'augmentation': args.aug,
                'train_files': sorted(
                    p.name for p in Path(args.train_dir).glob('*.tif*')
                ),
            }
            save_bundle(model, bundle_dir, cfg)  # the best model IS the bundle
        save_bundle(model, os.path.join(run_dir, 'last'), cfg)

        minutes = (time.time() - t0) / 60
        lr = scheduler.get_last_lr()[0]
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow(
                [step, f'{recent:.4f}', f'{val["total"]:.4f}', f'{val["sem"]:.4f}',
                 f'{val["aff"]:.4f}', f'{val["dice"]:.4f}',
                 f'{val["int_wall"]:.4f}', f'{val["int_aff"]:.4f}',
                 f'{val["int_dots"]:.5f}',
                 f'{val["dice_best"]:.4f}', f'{val["thr_best"]:.2f}',
                 f'{val["precision"]:.4f}', f'{val["recall"]:.4f}',
                 f'{val["dice_img"]:.4f}', f'{val["dice_img_sd"]:.4f}',
                 f'{val["tol_f1"]:.4f}', f'{val["cldice"]:.4f}',
                 f'{val["cell_f1"]:.4f}', f'{score:.4f}',
                 f'{val["skel_on_wall"]:.4f}',
                 f'{lr:.2e}', f'{minutes:.1f}']
            )  # fmt: skip
        print(
            f'step {step:6d} | train {recent:.3f} | val {val["total"]:.3f} '
            f'(wall {val["sem"]:.3f}, aff {val["aff"]:.3f}) | val Dice {val["dice"]:.3f} '
            f'| interior: wall {val["int_wall"]:.4f}, aff-holes {val["int_aff"]:.3f}, dots {val["int_dots"]:.4f} '
            f'| P {val["precision"]:.3f} R {val["recall"]:.3f} '
            f'| Dice@{val["thr_best"]:.2f} {val["dice_best"]:.3f} '
            f'| img {val["dice_img"]:.3f}+-{val["dice_img_sd"]:.3f} '
            f'| tol-F1 {val["tol_f1"]:.3f} | clDice {val["cldice"]:.3f} '
            f'| cell-F1 {val["cell_f1"]:.3f} (score {score:.3f}) '
            + (
                ''
                if np.isnan(val['skel_on_wall'])
                else f'| skeleton on wall {val["skel_on_wall"]:.3f} '
            )
            + f'| best {best_dice:.3f} | {minutes:.1f} min',
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
        fig.savefig(progress_png, dpi=150)
        if args.keep_history:
            fig.savefig(
                os.path.join(run_dir, 'history', f'step_{step:06d}.png'),
                dpi=100,
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
            parts = train_step(
                model,
                optimizer,
                next(batches),
                ftl,
                device,
                accum=args.accum,
                next_batch=lambda: next(batches),
                amp=amp,
                **loss_kw,
            )
            scheduler.step()
            step += 1
            history['step'].append(step)
            history['train_total'].append(parts['total'])
            history['train_sem'].append(parts['sem'])
            history['train_aff'].append(parts['aff'])
            history['train_geo'].append(parts['geo'])
            history['train_interior'].append(
                parts['int_wall'] + parts['int_aff']
            )
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
