"""
Headless end-to-end check of a trained model bundle through the PLUGIN code
(no napari window needed):

    python scripts/check_model_in_plugin.py runs/run1/best
    python scripts/check_model_in_plugin.py runs/run1/best --image my_section.tif

Stages (stops at the first failure):
  1 load the bundle with the plugin's loader
  2 predict a whole image with the plugin's sliding-window code
  3 build the napari layer payloads and read them back like the widgets do
  4 superpixels (local affinity watershed)
  5 GASP
  6 mutex watershed
Writes plugin_check.png next to where you run it.
"""

import argparse
import os
import sys
import traceback
from types import SimpleNamespace

os.environ.setdefault('KERAS_BACKEND', 'torch')

import numpy as np  # noqa: E402

STATE = {}


def stage_load(args):
    import w_seg_napari._wsegutils as w

    STATE['w'] = w
    gen = w.load_my_generator(w.device, args.bundle)
    cfg = w.get_model_config(gen)
    STATE.update(gen=gen, cfg=cfg)
    a = cfg['affinity']
    print(
        f'   ranges {a["ranges"]}, directions {a["direction_names"]}, '
        f'{len(a["offsets"])} channels, hue encoding {cfg["preprocessing"]["hue_encoding"]}'
    )


def stage_predict(args):
    import torch

    w, gen, cfg = STATE['w'], STATE['gen'], STATE['cfg']
    if args.image:
        import tifffile

        img = tifffile.imread(args.image)
        if (
            img.ndim == 3
            and img.shape[0] in (3, 4)
            and img.shape[-1] not in (3, 4)
        ):
            img = np.moveaxis(img, 0, -1)
        img = img[..., :3]
    else:
        # synthetic tissue-like picture: bright cells separated by dark walls
        yy, xx = np.mgrid[0:512, 0:512]
        walls = (yy % 40 < 5) | (xx % 40 < 5)
        img = np.where(
            walls[..., None], [90, 40, 120], [215, 225, 235]
        ).astype(np.float32)
        img += np.random.default_rng(0).normal(0, 6, img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    pre = cfg['preprocessing']
    sem, aff = w.process_large_image(
        tensor,
        gen,
        patch_size=256,
        stride=128,
        device=w.device,
        live_view=False,
        hue_encoding=pre['hue_encoding'],
        enable_lab=pre['enable_lab'],
        output_names=(cfg['outputs']['wall'], cfg['outputs']['affinity']),
    )
    H, W = img.shape[:2]
    assert sem.shape == (1, H, W), sem.shape
    assert aff.shape == (len(cfg['affinity']['offsets']), H, W), aff.shape
    assert np.isfinite(sem).all() and np.isfinite(aff).all()
    print(
        f'   image {W}x{H}; wall {sem.shape}, affinities {aff.shape}; '
        f'wall mean {sem.mean():.2f}'
    )
    STATE.update(img=img, sem=sem[0], aff=aff)


def stage_layers(args):
    from w_seg_napari import _affinity as A

    payloads = A.layer_payloads(STATE['aff'], STATE['cfg'], 'check')
    for p in payloads:
        print(f'   layer {p["name"]}: data {p["data"].shape}')
    layers = [SimpleNamespace(**p) for p in payloads]
    layers.append(
        SimpleNamespace(name='pred_check', data=STATE['sem'], metadata={})
    )
    assert A.find_sources([lay.name for lay in layers]) == ['check']
    aset = A.gather(layers, 'check')
    assert aset.offsets == [
        tuple(o) for o in STATE['cfg']['affinity']['offsets']
    ]
    assert np.allclose(aset.data, np.clip(STATE['aff'], 0, 1))
    STATE['aset'] = aset
    print(f'   read back: {aset.describe()}')


def stage_superpixels(args):
    from w_seg_napari import _affinity as A
    from w_seg_napari._superpixel_widget import SuperpixelWidget

    sp = SuperpixelWidget._compute_affinity_watershed(
        None,
        None,
        sem_mask=STATE['sem'],
        compactness=0.0,
        min_distance=7,
        boundary_map=A.boundary_magnitude(STATE['aset']),
        threshold=0.35,
        aff_threshold=0.5,
        min_cell_size=15,
        sigma=1.0,
        aff_sigma=1.0,
        min_peak_height=2.0,
    )
    print(f'   {int(sp.max())} superpixels')
    assert sp.shape == STATE['sem'].shape and sp.max() > 0
    STATE['sp'] = sp


def stage_gasp(args):
    from w_seg_napari._gasp_widget import GaspWidget

    for which in ('all', 'shortest'):
        aset = STATE['aset'].select(which)
        cells = GaspWidget._compute_gasp_from_set(
            aset, superpixels=STATE['sp'], linkage='average', beta_bias=0.9
        )
        assert cells is not None and cells.shape == STATE['sp'].shape
        print(
            f'   ranges {which}: {len(np.unique(cells))} segments from {int(STATE["sp"].max())} superpixels'
        )
        STATE['gasp_' + which] = cells


def stage_mutex(args):
    from w_seg_napari._superpixel_widget import SuperpixelWidget

    labels = SuperpixelWidget._compute_mutex_multi(
        STATE['aset'], semantic_map=STATE['sem'], strides=2, min_cell_size=15
    )
    assert labels is not None and labels.shape == STATE['sem'].shape
    print(f'   {int(labels.max())} segments')
    STATE['mutex'] = labels


def write_picture():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from skimage.color import label2rgb

    fig, ax = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    ax[0].imshow(STATE['img'])
    ax[0].set_title('input')
    ax[1].imshow(STATE['sem'], cmap='gray', vmin=0, vmax=1)
    ax[1].set_title('wall prediction')
    if 'gasp_all' in STATE:
        ax[2].imshow(label2rgb(STATE['gasp_all'], bg_label=0))
        ax[2].set_title('GASP, all ranges')
    if 'mutex' in STATE:
        ax[3].imshow(label2rgb(STATE['mutex'], bg_label=0))
        ax[3].set_title('mutex watershed')
    for a in ax:
        a.axis('off')
    fig.savefig('plugin_check.png', dpi=70)
    print('picture written to plugin_check.png')


STAGES = [
    ('load the model bundle with the plugin loader', stage_load),
    ('predict a whole image (sliding window)', stage_predict),
    ('napari layer payloads and read-back', stage_layers),
    ('superpixels (local affinity watershed)', stage_superpixels),
    ('GASP (needs elf)', stage_gasp),
    ('mutex watershed (needs elf)', stage_mutex),
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'bundle', help='model bundle folder (model.keras + config.json)'
    )
    parser.add_argument(
        '--image', help='optional RGB image (TIFF); default: synthetic picture'
    )
    args = parser.parse_args()
    for n, (title, fn) in enumerate(STAGES, 1):
        print(f'[{n}/{len(STAGES)}] {title}')
        try:
            fn(args)
        except Exception:
            traceback.print_exc()
            print(f'\nFAILED at stage {n}: {title}')
            if n >= 3:
                try:
                    write_picture()
                except Exception:
                    pass
            sys.exit(1)
        print('   PASS')
    write_picture()
    print('\nALL STAGES PASSED')
