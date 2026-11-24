"""Is a benchmark image (or part of it) already in the training data?

    python scripts/check_overlap.py CANDIDATES --against TRAIN [TRAIN ...]
           [--out overlap] [--write-split benchmark_split.json]

CANDIDATES  image files, or folders of images, that you want to keep out of
            training (the benchmark / test set)
--against   folders (or files) with everything the models are trained or
            validated on: image tifs, png/jpg, or the 5-plane training tifs
            (the first three planes are the colour image)

Names do not matter here, only the pixels. Every candidate is compared with
every training image in three ways:

  identical   the same pixels (sha1 of the colour image)
  duplicate   the same picture after resizing, recompression or a brightness
              change (perceptual hash, Hamming distance <= --hash-dist)
  overlap     a shared region, e.g. the candidate is a crop of a larger
              training slide, or the other way round (ORB feature matches
              that agree on one shift/scale/rotation, >= --min-inliers)

The overlap test is best effort: it needs a few hundred pixels of shared
tissue at the same magnification, and it can also fire on two pictures of
the same plant imaged next to each other. Look at the pairs it reports
(--save-matches DIR writes a picture of each) and decide yourself.
A "same_name" note only means that a training file has the same file name.

Plants imaged twice are not detected: make sure the benchmark plants are not
in the training data (different plants, ideally different days).

Writes <out>.csv with every pair that was flagged, prints a summary, and with
--write-split saves the candidates that are clean, with their fingerprints
(file name, sha1, hash, size). Keep that file with the results: the same check
can be repeated later, with a new training folder, against the same images.
Exit status 1 when anything was flagged.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
WORK_MAX = 6000  # longest side kept for the overlap test
HASH_SIZE = 16  # 16 x 16 hash = 256 bits


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def list_images(paths):
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files += sorted(
                f
                for f in p.rglob('*')
                if f.is_file()
                and f.suffix.lower() in IMAGE_EXTS
                and not f.name.startswith('.')
            )
        elif p.is_file():
            files.append(p)
        else:
            print(f'Not found: {p}', file=sys.stderr)
    return files


def open_array(path):
    """The pixel array of an image file, memory-mapped when possible."""
    if path.suffix.lower() in ('.tif', '.tiff'):
        import tifffile

        try:
            return tifffile.memmap(path)
        except Exception:
            return tifffile.imread(path)
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError('cannot read the image')
    if arr.ndim == 3 and arr.shape[-1] >= 3:  # OpenCV gives BGR(A)
        arr = arr[..., [2, 1, 0]]
    return arr


def colour_view(arr):
    """(H, W, 3) or (H, W) view of the colour image inside an array.

    Handles H x W, H x W x C, C x H x W and the 5-plane training tifs
    (planes first; the first three are the colour image).
    """
    arr = np.asarray(arr) if not isinstance(arr, np.memmap) else arr
    arr = arr.squeeze() if arr.ndim > 3 else arr
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f'unsupported shape {arr.shape}')
    if arr.shape[0] <= 8 and arr.shape[0] < min(arr.shape[1:]):
        return np.moveaxis(arr[: min(3, arr.shape[0])], 0, -1)
    if arr.shape[-1] in (1, 3, 4):
        return arr[..., :3] if arr.shape[-1] != 1 else arr[..., 0]
    raise ValueError(f'cannot tell the colour channels of shape {arr.shape}')


def to_gray_u8(view, step=1):
    """8-bit gray of a (strided) colour view, stretched to its own range."""
    v = view[::step, ::step]
    v = np.asarray(v)
    if v.ndim == 3:
        v = v[..., :3].astype(np.float32).mean(axis=-1)
    else:
        v = v.astype(np.float32)
    lo, hi = np.percentile(v[::7, ::7], (1, 99))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((v - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def sha1_of(view, rows=512):
    h = hashlib.sha1()
    h.update(str(view.shape).encode())
    h.update(str(view.dtype).encode())
    for r in range(0, view.shape[0], rows):
        h.update(np.ascontiguousarray(view[r : r + rows]).tobytes())
    return h.hexdigest()


def dhash(gray, size=HASH_SIZE):
    """Difference hash: does each pixel of a size+1 x size thumbnail
    get brighter to the right? Robust to resizing and brightness."""
    small = cv2.resize(
        gray, (size + 1, size), interpolation=cv2.INTER_AREA
    ).astype(np.int16)
    return (small[:, 1:] > small[:, :-1]).flatten()


class Item:
    """Everything the checks need from one image."""

    def __init__(self, path, want_sha=True):
        self.path = Path(path)
        view = colour_view(open_array(self.path))
        self.shape = (int(view.shape[0]), int(view.shape[1]))
        self.sha1 = sha1_of(view) if want_sha else ''
        step = max(1, int(np.ceil(max(self.shape) / WORK_MAX)))
        work = to_gray_u8(view, step)
        self.work_scale = work.shape[1] / self.shape[1]  # work px per full px
        self.work = work
        self.hash = dhash(work)
        self._kp = {}

    @property
    def name(self):
        return self.path.name

    def orb(self, factor):
        """Keypoints and descriptors at `factor` (work px per full px)."""
        key = round(factor, 4)
        if key not in self._kp:
            rel = factor / self.work_scale
            img = self.work
            if rel < 0.999:
                w = max(32, int(round(img.shape[1] * rel)))
                h = max(32, int(round(img.shape[0] * rel)))
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            orb = cv2.ORB_create(nfeatures=4000)
            kp, desc = orb.detectAndCompute(img, None)
            self._kp[key] = (kp, desc, img)
        return self._kp[key]


# --------------------------------------------------------------------------
# comparing
# --------------------------------------------------------------------------
def pair_factor(a, b):
    """Common scale (work px per full px) for comparing two images: as
    detailed as possible, but never more than WORK_MAX on the larger one."""
    big = max(max(a.shape), max(b.shape))
    small = min(max(a.shape), max(b.shape))
    f = min(1.0, 3000.0 / big)
    f = max(f, min(1.0, 800.0 / small))  # keep the smaller one detailed
    return min(f, WORK_MAX / big, a.work_scale, b.work_scale)


def feature_overlap(a, b, min_inliers):
    """(inliers, matches picture or None) of the best similarity transform
    between the features of two images; inliers = 0 when none was found."""
    f = pair_factor(a, b)
    ka, da, _ = a.orb(f)
    kb, db, _ = b.orb(f)
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        return 0, None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = []
    for pair in matcher.knnMatch(da, db, k=2):
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            good.append(pair[0])
    if len(good) < min_inliers:
        return 0, None
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    matrix, mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if matrix is None:
        return 0, None
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    inliers = int(mask.sum())
    if not 0.2 < scale < 5.0 or inliers < min_inliers:
        return 0, None
    return inliers, (good, mask.ravel().astype(bool), f)


def save_match_picture(a, b, info, path):
    good, mask, f = info
    ka, _, ga = a.orb(f)
    kb, _, gb = b.orb(f)
    inl = [m for m, ok in zip(good, mask) if ok]
    pic = cv2.drawMatches(
        ga,
        ka,
        gb,
        kb,
        inl[:60],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(path), pic)


def compare(cand, ref, hash_dist, min_inliers, crops):
    """List of (kind, detail, picture_info) for one candidate/reference pair."""
    found = []
    if cand.sha1 and cand.sha1 == ref.sha1:
        found.append(('identical', 'same pixels', None))
        return found
    dist = int((cand.hash != ref.hash).sum())
    if dist <= hash_dist:
        found.append(('duplicate', f'hash distance {dist}', None))
        return found
    if crops:
        n, info = feature_overlap(cand, ref, min_inliers)
        if n:
            found.append(('overlap', f'{n} matching features', info))
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('candidates', nargs='+')
    p.add_argument('--against', nargs='+', required=True)
    p.add_argument('--out', default='overlap', help='output file stem')
    p.add_argument('--write-split', default=None, metavar='JSON')
    p.add_argument('--hash-dist', type=int, default=24)
    p.add_argument('--min-inliers', type=int, default=30)
    p.add_argument('--no-crops', action='store_true')
    p.add_argument('--save-matches', default=None, metavar='DIR')
    a = p.parse_args()

    cands = list_images(a.candidates)
    refs = list_images(a.against)
    if not cands or not refs:
        sys.exit('Need at least one candidate and one training image.')
    # a candidate listed among the training files is a mistake
    cand_paths = {c.resolve() for c in cands}
    refs = [r for r in refs if r.resolve() not in cand_paths]

    print(f'{len(cands)} candidate(s), {len(refs)} training image(s)')
    items = []
    for c in cands:
        try:
            items.append(Item(c))
        except Exception as e:
            print(f'Skipped {c}: {e}', file=sys.stderr)
    ref_items = []
    for i, r in enumerate(refs, 1):
        try:
            ref_items.append(Item(r))
        except Exception as e:
            print(f'Skipped {r}: {e}', file=sys.stderr)
        if i % 20 == 0:
            print(f'  read {i}/{len(refs)} training images')

    if a.save_matches:
        os.makedirs(a.save_matches, exist_ok=True)

    rows = []
    flagged = set()
    for c in items:
        for r in ref_items:
            hits = compare(c, r, a.hash_dist, a.min_inliers, not a.no_crops)
            same_name = c.path.stem == r.path.stem
            if not hits and same_name:
                hits = [('same_name', 'file name only', None)]
            for kind, detail, info in hits:
                flagged.add(c.path)
                rows.append(
                    {
                        'candidate': c.name,
                        'training_image': str(r.path),
                        'kind': kind,
                        'detail': detail,
                        'same_name': same_name,
                    }
                )
                if info is not None and a.save_matches:
                    out = (
                        Path(a.save_matches)
                        / f'{c.path.stem}__{r.path.stem}.png'
                    )
                    save_match_picture(c, r, info, out)

    import pandas as pd

    table = pd.DataFrame(
        rows,
        columns=[
            'candidate',
            'training_image',
            'kind',
            'detail',
            'same_name',
        ],
    )
    table.to_csv(f'{a.out}.csv', index=False)

    clean = [c for c in items if c.path not in flagged]
    print()
    if table.empty:
        print('No overlap found.')
    else:
        print(table.to_string(index=False))
        print(
            f'\n{len(flagged)} of {len(items)} candidate(s) flagged '
            f'(details in {a.out}.csv); {len(clean)} clean.'
        )
    if a.write_split:
        doc = {
            'created': datetime.now().isoformat(timespec='seconds'),
            'checked_against': [str(Path(x).resolve()) for x in a.against],
            'n_training_images_checked': len(ref_items),
            'benchmark': [
                {
                    'file': c.name,
                    'sha1': c.sha1,
                    'dhash': ''.join('1' if x else '0' for x in c.hash),
                    'shape': list(c.shape),
                }
                for c in clean
            ],
        }
        with open(a.write_split, 'w') as f:
            json.dump(doc, f, indent=2)
        print(f'Wrote {len(clean)} clean image(s) to {a.write_split}')
    sys.exit(1 if flagged else 0)


if __name__ == '__main__':
    main()
