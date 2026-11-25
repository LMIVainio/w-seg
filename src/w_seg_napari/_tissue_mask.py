"""
Coarse tissue mask: where is the tissue, on the whole image at 512 x 512?

A small U-Net looks at the WHOLE image squashed to 512 x 512 and says, for
every pixel, "tissue" or "not tissue" (empty slide, debris, peeling layer ...).
Because it sees the whole picture it can tell a peeling layer from the tissue,
which a 256 px patch of the cell-wall model cannot. The mask is used to

  * skip the empty tiles during prediction (speed),
  * drop cells that lie outside the tissue (the "in / out" decision, with one
    fixed threshold),
  * cut an image with several roots into one box per root.

The model is optional and is stored in the model bundle (zip) next to
model.keras: tissue_mask.keras + tissue_mask.json.

Only numpy / scipy / OpenCV are needed here, except for building, saving and
loading the network (keras, imported when used).
"""

import json
from pathlib import Path

import numpy as np

SIZE = 512  # what the network sees
STORE = 1024  # what the training keeps in memory (random crops come from it)
MODEL_FILE = 'tissue_mask.keras'
CONFIG_FILE = 'tissue_mask.json'


# =====================================================================
# Input and target
# =====================================================================
def _float01(arr):
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    arr = arr.astype(np.float32)
    top = float(arr.max()) if arr.size else 1.0
    if top > 255.0:
        return arr / top
    return arr / 255.0 if top > 1.0 else arr


def prepare_input(rgb, size=SIZE):
    """(H, W, 3) image of any size and dtype -> (size, size, 3) float32 in
    [-1, 1]: per-channel 1-99 percentile stretch of the whole image, then the
    image is squashed to a square (the aspect ratio is not kept, as in the
    training)."""
    import cv2

    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    rgb = rgb[..., :3]
    step = max(1, int(max(rgb.shape[:2]) // (4 * size)))
    small = _float01(rgb[::step, ::step])
    if small.shape[0] != size or small.shape[1] != size:
        small = cv2.resize(small, (size, size), interpolation=cv2.INTER_AREA)
    flat = small.reshape(-1, 3)
    lo = np.percentile(flat, 1, axis=0)
    hi = np.percentile(flat, 99, axis=0)
    out = (small - lo) / np.maximum(hi - lo, 1e-6)
    return (np.clip(out, 0.0, 1.0) * 2.0 - 1.0).astype(np.float32)


def _block_any(mask2d, block, rows=2048):
    """One boolean per block x block pixels: True if the block holds any
    True pixel. mask2d can be a memory map (read in strips)."""
    H, W = mask2d.shape
    out = np.zeros((-(-H // block), -(-W // block)), dtype=bool)
    rows = max(block, (rows // block) * block)
    for y0 in range(0, H, rows):
        chunk = np.asarray(mask2d[y0 : y0 + rows]) > 0
        h, w = chunk.shape
        ph, pw = (-h) % block, (-w) % block
        if ph or pw:
            chunk = np.pad(chunk, ((0, ph), (0, pw)))
        blocks = chunk.reshape(
            chunk.shape[0] // block, block, chunk.shape[1] // block, block
        ).any(axis=(1, 3))
        out[y0 // block : y0 // block + blocks.shape[0]] = blocks
    return out


def make_target(wall, size=STORE, close=8, min_area=0.002):
    """Tissue target (size, size) bool from the wall plane of a training slide.

    "Tissue" = "contains cell walls": the wall mask is reduced to blocks, the
    gaps between cells are closed (`close` = radius in pixels of the size x
    size picture), holes are filled and specks smaller than `min_area` of the
    picture are removed. A peeling layer that is annotated like the rest of
    the tissue is therefore tissue here: paint its masks by hand to change that.
    """
    import cv2
    from scipy import ndimage as ndi

    H, W = wall.shape
    block = max(1, int(np.ceil(max(H, W) / size)))
    coarse = _block_any(wall, block).astype(np.float32)
    mask = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_AREA)
    mask = mask > 0.25
    if close > 0:
        r = int(close)
        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
        disk = (xx * xx + yy * yy) <= r * r
        padded = np.pad(mask, r)  # so that the closing does not eat the edge
        mask = ndi.binary_closing(padded, structure=disk)[r:-r, r:-r]
    mask = ndi.binary_fill_holes(mask)
    lab, n = ndi.label(mask)
    if n:
        sizes = np.bincount(lab.ravel())
        small = sizes < min_area * size * size
        small[0] = False
        mask = mask & ~small[lab]
    return mask


# =====================================================================
# The mask in use
# =====================================================================
class TissueLookup:
    """The mask of one image, answering questions in full-image coordinates.

    prob: (h, w) tissue probabilities from the network (low resolution).
    shape: (H, W) of the image the mask belongs to.
    """

    def __init__(self, prob, shape, threshold=0.5):
        prob = np.asarray(prob, dtype=np.float32)
        self.prob = prob
        self.shape = (int(shape[0]), int(shape[1]))
        self.threshold = float(threshold)
        self.mask = prob >= self.threshold
        ii = np.zeros(
            (self.mask.shape[0] + 1, self.mask.shape[1] + 1), dtype=np.int64
        )
        ii[1:, 1:] = self.mask.astype(np.int64).cumsum(0).cumsum(1)
        self._ii = ii

    def fraction(self, y0, x0, y1, x1):
        """Share of the box (full-image pixels) that is tissue."""
        H, W = self.shape
        h, w = self.mask.shape
        a0 = int(np.clip(np.floor(y0 / H * h), 0, h - 1))
        b0 = int(np.clip(np.floor(x0 / W * w), 0, w - 1))
        a1 = int(np.clip(np.ceil(y1 / H * h), a0 + 1, h))
        b1 = int(np.clip(np.ceil(x1 / W * w), b0 + 1, w))
        total = self._ii[a1, b1] - self._ii[a0, b1] - self._ii[a1, b0]
        total += self._ii[a0, b0]
        return float(total) / ((a1 - a0) * (b1 - b0))

    def keep_tile(self, y, x, size, min_fraction=0.01):
        """False for a tile that is (almost) all background."""
        return self.fraction(y, x, y + size, x + size) >= min_fraction

    def full_mask(self):
        """The mask at the full image size (bool)."""
        import cv2

        H, W = self.shape
        up = cv2.resize(
            self.prob, (W, H), interpolation=cv2.INTER_LINEAR
        )  # smooth edge, then the same threshold
        return up >= self.threshold

    def outside_cells(self, labels, min_inside=0.5):
        """Ids of the cells of which less than `min_inside` of the area lies
        in the tissue, and each cell's inside fraction: (ids, {id: fraction})."""
        labels = np.asarray(labels)
        if labels.shape != self.shape:
            raise ValueError(
                f'labels {labels.shape} do not fit the mask {self.shape}'
            )
        inside = self.full_mask()
        n = int(labels.max()) + 1
        flat = labels.ravel()
        area = np.bincount(flat, minlength=n)
        ins = np.bincount(flat, weights=inside.ravel(), minlength=n)
        frac = np.divide(ins, np.maximum(area, 1))
        ids = [
            int(i)
            for i in np.nonzero(area)[0]
            if i != 0 and frac[i] < min_inside
        ]
        return ids, {int(i): float(frac[i]) for i in np.nonzero(area)[0]}


def split_roots(
    lookup, min_relative=0.1, min_area=0.005, pad=0.02, merge_touching=True
):
    """One box (y0, x0, y1, x1) in full-image pixels per separate piece of
    tissue, left to right.

    A piece must be at least `min_relative` of the biggest piece and `min_area`
    of the image. Boxes are padded by `pad` of the image side; boxes that
    overlap after padding are joined (so a root that the mask cuts in two is
    not split). The image itself is returned when there is one piece or none.
    """
    from scipy import ndimage as ndi

    H, W = lookup.shape
    h, w = lookup.mask.shape
    lab, n = ndi.label(lookup.mask)
    if n == 0:
        return [(0, 0, H, W)]
    sizes = np.bincount(lab.ravel())[1:]
    keep = [
        i + 1
        for i, s in enumerate(sizes)
        if s >= min_relative * sizes.max() and s >= min_area * h * w
    ]
    boxes = []
    for i, sl in zip(range(1, n + 1), ndi.find_objects(lab)):
        if i not in keep:
            continue
        y0, y1 = sl[0].start / h * H, sl[0].stop / h * H
        x0, x1 = sl[1].start / w * W, sl[1].stop / w * W
        py, px = pad * H, pad * W
        boxes.append(
            [
                max(0.0, y0 - py),
                max(0.0, x0 - px),
                min(float(H), y1 + py),
                min(float(W), x1 + px),
            ]
        )
    if merge_touching:
        merged = True
        while merged and len(boxes) > 1:
            merged = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    a, b = boxes[i], boxes[j]
                    if (
                        a[0] < b[2]
                        and b[0] < a[2]
                        and a[1] < b[3]
                        and b[1] < a[3]
                    ):
                        boxes[i] = [
                            min(a[0], b[0]),
                            min(a[1], b[1]),
                            max(a[2], b[2]),
                            max(a[3], b[3]),
                        ]
                        del boxes[j]
                        merged = True
                        break
                if merged:
                    break
    boxes.sort(key=lambda b: (b[1], b[0]))
    out = [
        (
            int(round(b[0])),
            int(round(b[1])),
            int(np.ceil(b[2])),
            int(np.ceil(b[3])),
        )
        for b in boxes
    ]
    return out or [(0, 0, H, W)]


# =====================================================================
# Network, saving, loading, running
# =====================================================================
def build_tissue_unet(size=SIZE, widths=(16, 32, 64, 128, 192)):
    """Compact U-Net (about 1 M weights) for the whole picture. It works at
    1/2 resolution and up, because a coarse mask needs no more. The output
    is a logit (apply a sigmoid)."""
    from keras import layers

    from w_seg_napari._training.model import conv_norm_act, res_block

    inputs = layers.Input(shape=(size, size, 3))
    x = conv_norm_act(inputs, widths[0], 3, stride=2)  # size / 2
    skips = [x]
    for w in widths[1:]:
        x = res_block(conv_norm_act(x, w, 3, stride=2), w)
        skips.append(x)
    x = skips.pop()  # bottleneck, size / 2^len(widths)
    for skip in reversed(skips):
        x = layers.UpSampling2D(2, interpolation='bilinear')(x)
        x = layers.Concatenate()([x, skip])
        x = res_block(conv_norm_act(x, skip.shape[-1], 1), skip.shape[-1])
    x = layers.UpSampling2D(2, interpolation='bilinear')(x)
    logits = layers.Conv2D(1, 1, dtype='float32', name='tissue')(x)
    import keras

    return keras.Model(inputs, logits, name='tissue-mask-unet')


def default_config(size=SIZE, threshold=0.5):
    return {
        'type': 'tissue_mask',
        'version': 1,
        'size': int(size),
        'threshold': float(threshold),
        'output': 'logits',
        'preprocessing': 'percentile_rgb_squash',
    }


def save_tissue_mask(model, folder, config=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model.save(str(folder / MODEL_FILE))
    cfg = config or default_config(model.input_shape[1])
    (folder / CONFIG_FILE).write_text(json.dumps(cfg, indent=2))


def find_tissue_mask(folder):
    """Folder of a model bundle -> folder if it holds a tissue mask model."""
    if folder is None:
        return None
    folder = Path(folder)
    if (folder / MODEL_FILE).is_file() and (folder / CONFIG_FILE).is_file():
        return folder
    return None


def load_tissue_mask(folder, device=None):
    """(model, config) of the tissue mask model in a bundle folder."""
    import keras

    folder = Path(folder)
    cfg = json.loads((folder / CONFIG_FILE).read_text())
    if cfg.get('type') != 'tissue_mask':
        raise ValueError(f'{folder / CONFIG_FILE} is not a tissue mask config')
    model = keras.saving.load_model(str(folder / MODEL_FILE), compile=False)
    if device is not None:
        model.to(device)
    model.eval()
    return model, cfg


def predict_tissue(model, cfg, rgb, device=None):
    """Tissue probability (size, size) of a whole image."""
    import torch

    size = int(cfg['size'])
    x = prepare_input(rgb, size)[None]
    tensor = torch.from_numpy(x)
    if device is not None:
        tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor, training=False)
    logits = np.asarray(logits.detach().cpu().numpy(), dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-logits[0, ..., 0]))


def tissue_lookup(model, cfg, rgb, device=None, threshold=None):
    """TissueLookup of an image, ready to use."""
    prob = predict_tissue(model, cfg, rgb, device)
    thr = cfg.get('threshold', 0.5) if threshold is None else threshold
    return TissueLookup(prob, rgb.shape[:2], thr)
