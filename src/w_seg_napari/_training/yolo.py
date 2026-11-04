"""Train the YOLO cell-type classifier and attach it to a W-seg model bundle.

    python -m w_seg_napari._training.yolo --dataset DIR --out-root OUT --name tissue \\
        [--attach-to BUNDLE --out-bundle NEW_BUNDLE]

DIR holds train/ and val/ with one folder per class (the Train tab writes it
with _yolo_data.build_dataset). A run is written to <out-root>/<name>/ :

    <name>.pt             the best classifier (highest validation top-1 accuracy)
    yolo_result.json      accuracy, class names, crop settings, hyper-parameters
    results.csv           one row per epoch (from Ultralytics)
    ultralytics/          the complete Ultralytics run folder

With --attach-to the best classifier is also written, together with a copy of
that W-seg bundle, to --out-bundle (config.json gets the crop settings and the
class names). The source bundle is never changed.

Like the W-net trainer, this prints lines that begin with "@@" (JSON) for the
plugin's progress bar, and stops cleanly after the current epoch when the file
given by --stop-file appears.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def emit(event, **kw):
    print('@@' + json.dumps({'event': event, **kw}), flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--dataset', required=True, help='folder with train/ and val/'
    )
    p.add_argument('--out-root', required=True)
    p.add_argument('--name', default='yolo_classifier')
    p.add_argument('--stop-file', default=None)
    p.add_argument(
        '--base',
        default='yolo11n-cls.pt',
        help='Ultralytics classification weights to start from (a name that '
        'is downloaded, or a path, e.g. an earlier classifier to continue)',
    )
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=256)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument(
        '--device', default='auto', help="'auto', 'cpu' or a GPU index"
    )
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--lr0', type=float, default=1e-3)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=0)
    # augmentation (cell crops have no preferred orientation)
    p.add_argument('--degrees', type=float, default=180.0)
    p.add_argument('--scale', type=float, default=0.3)
    p.add_argument('--flipud', type=float, default=0.5)
    p.add_argument('--fliplr', type=float, default=0.5)
    p.add_argument('--hsv-h', type=float, default=0.015)
    p.add_argument('--hsv-s', type=float, default=0.3)
    p.add_argument('--hsv-v', type=float, default=0.3)
    p.add_argument('--erasing', type=float, default=0.2)
    p.add_argument(
        '--info-json',
        default=None,
        help='json file with what the dataset builder reports (crop settings, '
        'class folders, counts); copied into yolo_result.json and the bundle',
    )
    p.add_argument('--attach-to', default=None, help='W-seg bundle folder')
    p.add_argument(
        '--out-bundle', default=None, help='folder of the new bundle'
    )
    args = p.parse_args(argv)
    if args.attach_to and not args.out_bundle:
        p.error('--attach-to needs --out-bundle')
    return args


def pick_device(choice):
    if choice != 'auto':
        return choice
    import torch

    return 0 if torch.cuda.is_available() else 'cpu'


def epoch_summary(trainer):
    """Numbers of the epoch that just finished, from an Ultralytics trainer."""
    m = getattr(trainer, 'metrics', {}) or {}
    return {
        'epoch': int(trainer.epoch) + 1,
        'epochs': int(trainer.epochs),
        'top1': float(m.get('metrics/accuracy_top1', float('nan'))),
        'top5': float(m.get('metrics/accuracy_top5', float('nan'))),
        'val_loss': float(m.get('val/loss', float('nan'))),
    }


def main(argv=None):
    args = parse_args(argv)
    from ultralytics import YOLO

    dataset = Path(args.dataset)
    for split in ('train', 'val'):
        if not (dataset / split).is_dir():
            raise FileNotFoundError(
                f"{dataset} must contain '{split}/' folders"
            )

    out_dir = Path(args.out_root) / args.name
    n = 2
    while (out_dir / 'yolo_result.json').exists():
        out_dir = Path(args.out_root) / f'{args.name}_{n}'
        n += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    if args.info_json:
        with open(args.info_json) as f:
            info = json.load(f)
    device = pick_device(args.device)
    emit('start', out_dir=str(out_dir), device=str(device))

    model = YOLO(args.base)
    best = {'top1': -1.0}

    def on_epoch_end(trainer):
        s = epoch_summary(trainer)
        best['top1'] = max(best['top1'], s['top1'])
        emit('progress', best_top1=best['top1'], **s)
        if args.stop_file and os.path.exists(args.stop_file):
            trainer.stop = True  # Ultralytics ends after this epoch

    model.add_callback('on_fit_epoch_end', on_epoch_end)
    hyper = {
        'epochs': args.epochs, 'imgsz': args.imgsz, 'batch': args.batch,
        'workers': args.workers, 'patience': args.patience, 'lr0': args.lr0,
        'dropout': args.dropout, 'optimizer': 'AdamW', 'seed': args.seed,
        'degrees': args.degrees, 'scale': args.scale, 'flipud': args.flipud,
        'fliplr': args.fliplr, 'hsv_h': args.hsv_h, 'hsv_s': args.hsv_s,
        'hsv_v': args.hsv_v, 'erasing': args.erasing, 'mosaic': 0.0,
    }  # fmt: skip
    model.train(
        data=str(dataset.resolve()),
        device=device,
        project=str((out_dir / 'ultralytics').resolve()),
        name='run',
        exist_ok=True,
        **hyper,
    )

    trainer = model.trainer
    best_pt = Path(trainer.best)
    if not best_pt.is_file():  # stopped before the first checkpoint
        best_pt = Path(trainer.last)
    weights = out_dir / f'{args.name}.pt'
    shutil.copy2(best_pt, weights)
    results_csv = Path(trainer.save_dir) / 'results.csv'
    if results_csv.is_file():
        shutil.copy2(results_csv, out_dir / 'results.csv')

    trained = YOLO(str(weights))
    names = {str(k): v for k, v in sorted(trained.names.items())}
    result = {
        'created': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        'weights': str(weights),
        'base': args.base,
        'best_val_top1': best['top1'],
        'class_names': names,  # model output index -> folder name
        'hyper_parameters': hyper,
        'dataset': {
            k: info[k]
            for k in (
                'split_mode',
                'val_images',
                'counts',
                'n_cells',
                'folders',
            )
            if k in info
        },
        'crop': info.get('crop'),
    }
    with open(out_dir / 'yolo_result.json', 'w') as f:
        json.dump(result, f, indent=2)

    bundle = None
    if args.attach_to:
        from w_seg_napari._model_config import attach_yolo

        bundle = attach_yolo(
            args.attach_to,
            weights,
            args.out_bundle,
            {
                'crop': info.get('crop') or {},
                'class_names': names,
                'class_ids': info.get('folders', {}),
                'training': {
                    'created': result['created'],
                    'base': args.base,
                    'best_val_top1': best['top1'],
                    'split_mode': info.get('split_mode'),
                    'epochs_planned': args.epochs,
                },
            },
        )
    emit(
        'done',
        weights=str(weights),
        result=str(out_dir / 'yolo_result.json'),
        bundle=str(bundle) if bundle else None,
        best_top1=best['top1'],
    )


if __name__ == '__main__':
    main(sys.argv[1:])
