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
    if p.get('init_from'):
        args += ['--init-from', p['init_from']]
    return args


def parse_ranges(text):
    vals = [int(v) for v in re.split(r'[,\s]+', text.strip()) if v]
    if not vals or any(v < 1 for v in vals):
        raise ValueError('Affinity ranges must be positive whole numbers.')
    return sorted(set(vals))
