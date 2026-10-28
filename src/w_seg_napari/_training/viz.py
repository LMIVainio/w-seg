"""
Progress figure for training: example crops (train + validation) with the
network's predictions, and the loss / Dice curves.

Uses plain numpy + matplotlib, so it can be tested without Keras.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import lab2rgb


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
    ax.set_title(title, fontsize=8)
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
    ncols = 3 + 2 * n_ranges

    fig = plt.figure(
        figsize=(2.1 * ncols, 2.4 * 2 + 3.6), layout='constrained'
    )
    # Separate sub-figures: the curve axes' tick labels must not distort the image grid
    top, bottom = fig.subfigures(
        2, 1, height_ratios=[2 * 2.4, 3.6], hspace=0.02
    )
    gs = top.add_gridspec(2, ncols)
    for r, (sample, name) in enumerate(
        [(train_sample, 'train'), (val_sample, 'val')]
    ):
        axes = [top.add_subplot(gs[r, c]) for c in range(ncols)]
        sample_row(
            axes,
            sample,
            aff_channels,
            aff_labels,
            f'{name} #{sample.get("index", "?")}',
        )

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
    ax_loss.set_title('total loss (log scale)', fontsize=9)

    for key, color, label in (
        ('train_sem', 'tab:blue', 'wall, train'),
        ('train_aff', 'tab:green', 'affinity, train'),
    ):
        ax_parts.plot(
            steps, smooth(history[key], window), color=color, label=label
        )
    for key, color, label in (
        ('val_sem', 'tab:orange', 'wall, val'),
        ('val_aff', 'tab:red', 'affinity, val'),
    ):
        ax_parts.plot(
            history['eval_step'], history[key], 'o--', color=color, label=label
        )
    ax_parts.set_yscale('log')
    ax_parts.set_title('loss parts', fontsize=9)

    ax_dice.plot(
        history['eval_step'], history['val_dice'], 'o-', color='tab:purple'
    )
    if history['val_dice']:
        best = int(np.argmax(history['val_dice']))
        ax_dice.axvline(history['eval_step'][best], color='gray', ls=':')
        ax_dice.set_title(
            f'validation Dice (best {history["val_dice"][best]:.3f})',
            fontsize=9,
        )
    ax_dice.set_ylim(0, 1)

    for ax in (ax_loss, ax_parts, ax_dice):
        ax.set_xlabel('step', fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    for ax in (ax_loss, ax_parts):
        ax.legend(fontsize=7)
    if title:
        fig.suptitle(title, fontsize=10)
    return fig
