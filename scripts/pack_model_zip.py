"""
Pack a trained model into one zip for the plugin (GitHub release asset).

    python scripts/pack_model_zip.py runs/run1/best --name aspen_tolblue_v1 \
        --yolo path/to/yolov11_Populus_TolBlu.pt --radial-symmetry no

The zip holds model.keras, config.json (with the paired YOLO file and the
radial-symmetry preset written into its "companions" section) and the YOLO file.
Upload the zip to a release, then add one line to MODEL_ZIP_URL_MAP in
_wsegutils.py:   'aspen_tolblue_v1': 'https://github.com/.../aspen_tolblue_v1.zip'

Only the Python standard library is needed.
"""

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bundle', help='folder with model.keras and config.json')
    ap.add_argument('--name', required=True, help='name of the model / zip')
    ap.add_argument('--yolo', help='YOLO cell-type classifier (.pt) to pair')
    ap.add_argument(
        '--radial-symmetry',
        choices=['yes', 'no', 'keep'],
        default='keep',
        help='preset of the radial symmetry option when the model is loaded',
    )
    ap.add_argument('--out', default='.', help='folder for the zip')
    args = ap.parse_args()

    src = Path(args.bundle)
    for f in ('model.keras', 'config.json'):
        if not (src / f).is_file():
            raise SystemExit(f'{src / f} not found')

    cfg = json.loads((src / 'config.json').read_text())
    comp = cfg.setdefault('companions', {})
    comp.setdefault('yolo', None)
    comp.setdefault('radial_symmetry', None)
    if args.yolo:
        comp['yolo'] = Path(args.yolo).name
    if args.radial_symmetry != 'keep':
        comp['radial_symmetry'] = args.radial_symmetry == 'yes'

    out = Path(args.out) / f'{args.name}.zip'
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(src / 'model.keras', tmp / 'model.keras')
        (tmp / 'config.json').write_text(json.dumps(cfg, indent=2))
        files = ['model.keras', 'config.json']
        if args.yolo:
            shutil.copy(args.yolo, tmp / Path(args.yolo).name)
            files.append(Path(args.yolo).name)
        with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(tmp / f, f)
    print(f'Wrote {out}  (companions: {comp})')


if __name__ == '__main__':
    main()
