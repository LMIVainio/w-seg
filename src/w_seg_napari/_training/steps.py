"""
Plain supervised training for the W-net (no discriminator, no runner class).

Batch layout everywhere (decide it once, in the Dataset, and assert it):
    x : [B, H, W, 6]   input channels
    y : [B, H, W, 3]   channel 0 = semantic mask, channels 1-2 = affinities
    w : [B, H, W, 1]   EDT boundary weight map

The generator returns a dict:
    'semantic_head', 'affinity_head', and (optionally) 'aux_semantic_*'.
"""

import keras
import torch


def compute_loss(outputs, y, w, ftl, w_aff=1.0, w_aux=0.25):
    """
    Total loss = semantic FTL
               + w_aff * weighted-L1 on affinities
               + w_aux * sum of FTL on the auxiliary (deep supervision) heads.

    ftl is a WeightedFocalTverskyLoss from wseg_losses.py (the fixed version).
    Returns (total, dict of the individual parts for logging).
    """
    y_sem, y_aff = y[..., :1], y[..., 1:]

    sem = ftl(y_sem, outputs['semantic_head'], w)
    aff = keras.ops.mean(w * keras.ops.abs(y_aff - outputs['affinity_head']))

    aux = 0.0
    for name, pred in outputs.items():
        if name.startswith('aux_semantic_'):
            aux = aux + ftl(y_sem, pred, w)  # same mask, same weights

    total = sem + w_aff * aff + w_aux * aux
    parts = {'total': total, 'sem': sem, 'aff': aff, 'aux': aux}
    return total, parts


def train_step(generator, optimizer, batch, ftl, device):
    """One optimisation step. `optimizer` is a torch optimizer, as in your loop."""
    x, y, w = (t.float().to(device) for t in batch)

    optimizer.zero_grad()
    outputs = generator(x, training=True)
    total, parts = compute_loss(outputs, y, w, ftl)
    total.backward()
    optimizer.step()

    return {
        k: float(v.detach()) if hasattr(v, 'detach') else float(v)
        for k, v in parts.items()
    }


@torch.no_grad()
def validate(generator, val_loader, ftl, device, max_batches=20):
    """
    Average loss and Dice (semantic head, threshold 0.5) on a FIXED validation
    set. Call this every N steps and keep the checkpoint with the best Dice.
    """
    sums = {'total': 0.0, 'sem': 0.0, 'aff': 0.0, 'aux': 0.0, 'dice': 0.0}
    n = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y, w = (t.float().to(device) for t in batch)
        outputs = generator(x, training=False)  # training=False: no dropout
        _, parts = compute_loss(outputs, y, w, ftl)

        pred = (outputs['semantic_head'] > 0.5).float()
        true = y[..., :1]
        inter = (pred * true).sum()
        dice = (2 * inter + 1e-6) / (pred.sum() + true.sum() + 1e-6)

        for k, v in parts.items():
            sums[k] += float(v)
        sums['dice'] += float(dice)
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


def overfit_one_batch(generator, optimizer, batch, ftl, device, steps=200):
    """
    Sanity test: the loss on ONE fixed batch should fall close to 0.
    If it cannot, something is wrong with the model, the loss or the data
    (layout, target channels, weights) -- fix that before any real training.
    """
    for step in range(1, steps + 1):
        parts = train_step(generator, optimizer, batch, ftl, device)
        if step == 1 or step % 20 == 0:
            print(
                f'step {step:4d} | total {parts["total"]:.4f} '
                f'(sem {parts["sem"]:.3f}, aff {parts["aff"]:.3f}, '
                f'aux {parts["aux"]:.3f})'
            )
