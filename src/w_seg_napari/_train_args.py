"""Qt-free helpers for the Train tab (kept separate so they can be tested)."""

import re
from pathlib import Path


def count_tifs(folder):
    p = Path(folder)
    if not folder or not p.is_dir():
        return 0
    return len(list(p.glob('*.tif')) + list(p.glob('*.tiff')))


def safe_name(name):
    """A folder-name-safe model name."""
    name = re.sub(r'[^\w.-]+', '_', name.strip()).strip('._')
    return name or 'model'


def build_args(p):
    """Command-line arguments (after ``-m w_seg_napari._training``) for settings dict p."""
    args = [
        '--train-dir', p['train_dir'],
        '--val-dir', p['val_dir'],
        '--out-root', p['out_root'],
        '--name', p['name'],
        '--stop-file', p['stop_file'],
        '--steps', str(p['steps']),
        '--batch-size', str(p['batch_size']),
        '--eval-every', str(p['eval_every']),
        '--lr', str(p['lr']),
        '--warmup', str(p['warmup']),
        '--val-samples', str(p['val_samples']),
        '--ranges', *[str(r) for r in p['ranges']],
        '--zoom', str(p['zoom'][0]), str(p['zoom'][1]),
        '--tissue-bias', str(p['tissue_bias']),
        '--workers', str(p['workers']),
        '--seed', str(p['seed']),
    ]  # fmt: skip
    # loss options (the trainer's own defaults apply when a key is missing)
    for key, flag in (
        ('aff_floor', '--aff-floor'),
        ('w_interior', '--w-interior'),
        ('hard_neg', '--hard-neg'),
        ('w_dot', '--w-dot'),
    ):
        if p.get(key) is not None:
            args += [flag, str(p[key])]
    if p.get('aug') in ('mild', 'strong'):
        args += ['--aug', p['aug']]
    if int(p.get('input_stem') or 0) > 0:
        args += ['--input-stem', str(int(p['input_stem']))]
    if p.get('mixed_precision'):
        args.append('--mixed-precision')
    if int(p.get('accum') or 1) > 1:
        args += ['--accum', str(int(p['accum']))]
    if p.get('geometry') and not p.get('init_from'):
        args.append('--geometry')
    if p.get('init_from'):
        args += ['--init-from', p['init_from']]
    return args


def parse_ranges(text):
    vals = [int(v) for v in re.split(r'[,\s]+', text.strip()) if v]
    if not vals or any(v < 1 for v in vals):
        raise ValueError('Affinity ranges must be positive whole numbers.')
    return sorted(set(vals))


def build_yolo_args(p):
    """Command-line arguments (after ``-m w_seg_napari._training.yolo``) for settings dict p."""
    args = [
        '--dataset', p['dataset'],
        '--out-root', p['out_root'],
        '--name', p['name'],
        '--stop-file', p['stop_file'],
        '--base', p['base'],
        '--epochs', str(p['epochs']),
        '--imgsz', str(p['imgsz']),
        '--batch', str(p['batch']),
        '--patience', str(p['patience']),
        '--seed', str(p['seed']),
    ]  # fmt: skip
    if p.get('workers') is not None:
        args += ['--workers', str(p['workers'])]
    if p.get('info_json'):
        args += ['--info-json', p['info_json']]
    if p.get('attach_to'):
        args += [
            '--attach-to',
            p['attach_to'],
            '--out-bundle',
            p['out_bundle'],
        ]
    return args


def unique_bundle_dir(root, name):
    """root/name, or root/name_2, root/name_3 ... the first one that is free."""
    root = Path(root)
    out = root / name
    n = 2
    while out.exists():
        out = root / f'{name}_{n}'
        n += 1
    return out
