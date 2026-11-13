"""
Progress figure for training: example crops (train + validation) with the
network's predictions, and the loss / Dice curves.

Uses plain numpy + matplotlib, so it can be tested without Keras.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import lab2rgb

from w_seg_napari._geometry_fields import direction_to_rgb


def input_to_rgb(x):
    """[H, W, 6] network input (channels 0-2 = LAB scaled to [-1, 1]) -> RGB for display."""
    c = np.asarray(x[..., :3], dtype=np.float64) * 0.5 + 0.5
    lab = np.stack(
        [
            c[..., 0] * 100.0,
            c[..., 1] * 255.0 - 128.0,
            c[..., 2] * 255.0 - 128.0,
        ],
        axis=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # out-of-gamut LAB values are clipped
        return np.clip(lab2rgb(lab), 0, 1)


def smooth(values, window):
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return v
    window = max(1, min(window, len(v)))
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, v[0]), v])
    return np.convolve(padded, kernel, mode='valid')


def _show(ax, img, title, cmap=None):
    ax.imshow(
        img,
        cmap=cmap,
        vmin=0 if cmap else None,
        vmax=1 if cmap else None,
        interpolation='nearest',
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def sample_row(axes, sample, aff_channels, aff_labels, row_name):
    """
    sample: dict with x [H,W,6], y [H,W,1+n], pred_wall [H,W], pred_aff [H,W,n].
    aff_channels: affinity channel indices to show (0-based within the affinities).
    """
    _show(axes[0], input_to_rgb(sample['x']), f'{row_name}: input')
    _show(axes[1], sample['y'][..., 0], 'wall: truth', 'gray')
    _show(axes[2], sample['pred_wall'], 'wall: prediction', 'gray')
    for k, (ch, label) in enumerate(zip(aff_channels, aff_labels)):
        _show(
            axes[3 + 2 * k],
            sample['y'][..., 1 + ch],
            f'{label}: truth',
            'gray',
        )
        _show(
            axes[4 + 2 * k],
            sample['pred_aff'][..., ch],
            f'{label}: prediction',
            'gray',
        )


def overlay_skeleton(rgb, skeleton, threshold=0.5):
    """Paint skeleton pixels (> threshold) bright red on an RGB image."""
    out = np.array(rgb, dtype=np.float64, copy=True)
    out[np.asarray(skeleton) > threshold] = (1.0, 0.1, 0.1)
    return out


def geometry_row(axes, sample, n_aff):
    """
    Geometry panels (needs sample['pred_geo'] [H,W,4] and the 4 geometry
    channels after the affinities in sample['y']): edge distance, direction
    vector field, skeleton, each truth and prediction, then the input with the
    predicted skeleton on top.
    """
    truth = sample['y'][..., 1 + n_aff : 5 + n_aff]
    pred = sample['pred_geo']
    t_valid = np.abs(truth[..., 1:3]).sum(-1) > 0.5
    # Predictions exist everywhere; only show a direction where the network
    # itself thinks it is inside a cell.
    p_valid = pred[..., 0] > 0.05
    _show(axes[0], truth[..., 0], 'edge distance: truth', 'magma')
    _show(
        axes[1],
        np.clip(pred[..., 0], 0, 1),
        'edge distance: prediction',
        'magma',
    )
    _show(
        axes[2], direction_to_rgb(truth[..., 1:3], t_valid), 'direction: truth'
    )
    _show(
        axes[3],
        direction_to_rgb(pred[..., 1:3], p_valid),
        'direction: prediction',
    )
    _show(axes[4], truth[..., 3], 'skeleton: truth', 'gray')
    _show(axes[5], np.clip(pred[..., 3], 0, 1), 'skeleton: prediction', 'gray')
    if len(axes) > 6:
        _show(
            axes[6],
            overlay_skeleton(input_to_rgb(sample['x']), pred[..., 3]),
            'input + predicted skeleton',
        )


def make_progress_figure(
    history, train_sample, val_sample, offsets, ranges, title=''
):
    """
    history keys: step, train_total, train_sem, train_aff (one per step),
                  eval_step, val_total, val_sem, val_aff, val_dice (one per eval).
    Shows the first-direction (E) affinity channel of every range.
    """
    n_ranges = len(ranges)
    per_range = len(offsets) // n_ranges
    aff_channels = [r * per_range for r in range(n_ranges)]
    aff_labels = [f'affinity E, k={ranges[r]}' for r in range(n_ranges)]
    has_geo = 'pred_geo' in train_sample and 'pred_geo' in val_sample
    ncols = max(3 + 2 * n_ranges, 7 if has_geo else 0)
    rows_per_sample = 2 if has_geo else 1
    n_img_rows = 2 * rows_per_sample

    fig = plt.figure(
        figsize=(2.3 * ncols, 2.5 * n_img_rows + 3.8), layout='constrained'
    )
    # Separate sub-figures: the curve axes' tick labels must not distort the image grid
    top, bottom = fig.subfigures(
        2, 1, height_ratios=[n_img_rows * 2.5, 3.8], hspace=0.02
    )
    gs = top.add_gridspec(n_img_rows, ncols)
    n_aff = len(offsets)
    for r, (sample, name) in enumerate(
        [(train_sample, 'train'), (val_sample, 'val')]
    ):
        row = r * rows_per_sample
        axes = [top.add_subplot(gs[row, c]) for c in range(ncols)]
        for ax in axes[3 + 2 * n_ranges :]:
            ax.axis('off')  # unused cells in the first row
        sample_row(
            axes[: 3 + 2 * n_ranges],
            sample,
            aff_channels,
            aff_labels,
            f'{name} #{sample.get("index", "?")}',
        )
        if has_geo:
            geo_axes = [top.add_subplot(gs[row + 1, c]) for c in range(ncols)]
            geometry_row(geo_axes, sample, n_aff)
            for ax in geo_axes[7:]:
                ax.axis('off')

    ax_loss, ax_parts, ax_dice = bottom.subplots(
        1, 3, gridspec_kw={'width_ratios': [1.3, 1.3, 1]}
    )

    steps = np.asarray(history['step'])
    window = max(5, len(steps) // 40)
    ax_loss.plot(steps, history['train_total'], color='tab:blue', alpha=0.2)
    ax_loss.plot(
        steps,
        smooth(history['train_total'], window),
        color='tab:blue',
        label='train',
    )
    ax_loss.plot(
        history['eval_step'],
        history['val_total'],
        'o-',
        color='tab:orange',
        label='validation',
    )
    ax_loss.set_yscale('log')
    ax_loss.set_title('total loss (log scale)', fontsize=11)

    # Wall and affinity always; geometry and the interior term only when the
    # run has them (a model without geometry head logs geo = 0).
    train_parts = [
        ('train_sem', 'tab:blue', 'wall, train'),
        ('train_aff', 'tab:green', 'affinity, train'),
        ('train_geo', 'tab:brown', 'geometry, train'),
        ('train_interior', 'tab:olive', 'interior, train'),
    ]
    val_parts = [
        ('val_sem', 'tab:orange', 'wall, val'),
        ('val_aff', 'tab:red', 'affinity, val'),
        ('val_geo', 'tab:pink', 'geometry, val'),
        ('val_interior', 'tab:cyan', 'interior, val'),
    ]
    for key, color, label in train_parts:
        values = history.get(key)
        if values is not None and len(values) == len(steps) and np.any(values):
            ax_parts.plot(
                steps, smooth(values, window), color=color, label=label
            )
    for key, color, label in val_parts:
        values = history.get(key)
        if (
            values is not None
            and len(values) == len(history['eval_step'])
            and np.any(values)
        ):
            ax_parts.plot(
                history['eval_step'], values, 'o--', color=color, label=label
            )
    ax_parts.set_yscale('log')
    ax_parts.set_title('loss parts', fontsize=11)

    ax_dice.plot(
        history['eval_step'], history['val_dice'], 'o-', color='tab:purple'
    )
    if history['val_dice']:
        best = int(np.argmax(history['val_dice']))
        ax_dice.axvline(history['eval_step'][best], color='gray', ls=':')
        ax_dice.set_title(
            f'validation Dice (purple, best {history["val_dice"][best]:.3f}) and others',
            fontsize=11,
        )
    for key, color, label in (
        ('val_tol_f1', 'tab:green', 'tolerance F1'),
        ('val_cldice', 'tab:brown', 'clDice'),
        ('val_cell_f1', 'tab:red', 'cell F1'),
    ):
        values = history.get(key)
        if values and len(values) == len(history['eval_step']):
            ax_dice.plot(
                history['eval_step'], values, 'o--', color=color, label=label,
                ms=3,
            )  # fmt: skip
    if len(ax_dice.lines) > 1:
        ax_dice.legend(fontsize=8)
    ax_dice.set_ylim(0, 1)

    for ax in (ax_loss, ax_parts, ax_dice):
        ax.set_xlabel('step', fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=10)
    ax_loss.legend(fontsize=9)
    ax_parts.legend(fontsize=8, ncol=2)
    if has_geo:
        title = (
            (f'{title}\n' if title else '')
            + 'direction key: x = red (right) / cyan (left), y = blue (down) / yellow (up), grey = none'
        )
    if title:
        fig.suptitle(title, fontsize=13)
    return fig
