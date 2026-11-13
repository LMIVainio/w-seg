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

from w_seg_napari._training.metrics import ValidationMetrics


def geometry_loss(pred, target, ftl):
    """
    Loss of the geometry head. pred/target are [B, H, W, 4]:
    edge distance, dir_y, dir_x, skeleton.
      * distance: L1 on pixels inside cells (target distance > 0)
      * direction: 1 - cosine, on pixels where a direction is defined
        (the target vector is zero elsewhere)
      * skeleton: the same focal Tversky loss as the wall head
    Returns (total, dict of parts).
    """
    ops = keras.ops
    d_t, dir_t, sk_t = target[..., :1], target[..., 1:3], target[..., 3:4]
    d_p, dir_p, sk_p = pred[..., :1], pred[..., 1:3], pred[..., 3:4]

    cell = ops.cast(d_t > 0, 'float32')
    dist = ops.sum(cell * ops.abs(d_p - d_t)) / ops.maximum(ops.sum(cell), 1.0)

    valid = ops.cast(
        ops.sum(ops.abs(dir_t), axis=-1, keepdims=True) > 0.5, 'float32'
    )
    norm_p = ops.sqrt(ops.sum(dir_p * dir_p, axis=-1, keepdims=True) + 1e-6)
    cos = ops.sum(dir_p * dir_t, axis=-1, keepdims=True) / norm_p
    direction = ops.sum(valid * (1.0 - cos)) / ops.maximum(ops.sum(valid), 1.0)

    skel = ftl(sk_t, sk_p, ops.ones_like(sk_t))
    total = dist + direction + skel
    return total, {'geo_dist': dist, 'geo_dir': direction, 'geo_skel': skel}


def interior_errors(outputs, y, n_aff, margin=3):
    """
    Errors deep inside cells, where the boundary-weighted losses barely look.
    A pixel is "interior" when no wall pixel lies within `margin` px of it.
      int_wall : mean predicted wall probability on interior pixels
                 (true value 0, so this is the false-wall rate)
      int_aff  : mean (1 - predicted affinity) over the affinities whose target
                 is 1 and whose pixel is interior ("holes" inside a cell)
    Both are 0 for a perfect model. Returns (int_wall, int_aff).
    """
    ops = keras.ops
    y_sem = y[..., :1]
    k = 2 * int(margin) + 1
    near_wall = ops.max_pool(y_sem, k, strides=1, padding='same')
    interior = 1.0 - near_wall  # 1 where no wall within `margin` px
    n_int = ops.maximum(ops.sum(interior), 1.0)

    int_wall = ops.sum(interior * outputs['semantic_head']) / n_int

    y_aff = y[..., 1 : 1 + n_aff]
    pos = interior * y_aff  # target 1 and interior
    int_aff = ops.sum(pos * (1.0 - outputs['affinity_head'])) / ops.maximum(
        ops.sum(pos), 1.0
    )
    return int_wall, int_aff


def soft_open(p, radius):
    """
    Grey-scale opening (erosion then dilation, square window of side
    2 * radius + 1) of a [B, H, W, 1] map. Structures thinner than the window,
    such as dots, disappear; larger ones survive. Differentiable.
    """
    ops = keras.ops
    r = int(radius)
    k = 2 * r + 1
    pad = ((0, 0), (r, r), (r, r), (0, 0))
    # constant padding with the neutral value of each step (1 for a minimum,
    # 0 for a maximum) works on every Keras backend and keeps the borders fair
    eroded = -ops.max_pool(
        -ops.pad(p, pad, mode='constant', constant_values=1.0),
        k,
        strides=1,
        padding='valid',
    )
    return ops.max_pool(
        ops.pad(eroded, pad, mode='constant', constant_values=0.0),
        k,
        strides=1,
        padding='valid',
    )


def interior_dots(outputs, y, margin=3, radius=2):
    """
    Dot-like false walls inside cells: the part of the predicted wall map that
    an opening removes (small blobs and thin specks), averaged over the
    interior pixels (no true wall within `margin` px). 0 for a perfect model.
    """
    ops = keras.ops
    pred = outputs['semantic_head']
    dots = ops.relu(pred - soft_open(pred, radius))
    near_wall = ops.max_pool(
        y[..., :1], 2 * int(margin) + 1, strides=1, padding='same'
    )
    interior = 1.0 - near_wall
    return ops.sum(interior * dots) / ops.maximum(ops.sum(interior), 1.0)


def compute_loss(
    outputs,
    y,
    w,
    ftl,
    w_aff=1.0,
    w_aux=0.25,
    w_geo=0.5,
    aff_floor=1.0,
    w_interior=0.5,
    interior_margin=3,
    hard_neg=2.0,
    w_dot=10.0,
    dot_radius=2,
):
    """
    Total loss = semantic FTL
               + w_aff * weighted-L1 on affinities
               + w_interior * (interior false-wall + interior affinity holes)
               + w_dot * dot-like false walls inside cells
               + w_aux * sum of FTL on the auxiliary (deep supervision) heads
               + w_geo * geometry loss (only when the model has a geometry head).

    y is [wall, affinities (n), geometry (4, optional)]; n is read from the
    model's affinity output. ftl is a WeightedFocalTverskyLoss.
    Returns (total, dict of the individual parts for logging).
    """
    n_aff = int(outputs['affinity_head'].shape[-1])
    y_sem = y[..., :1]
    y_aff = y[..., 1 : 1 + n_aff]

    # Hard negatives: wall-free pixels that the network currently calls wall
    # get extra weight (added to the EDT map, which is ~10 next to walls and
    # ~0.1 in a cell). The extra weight follows the prediction but is not
    # differentiated, so it only steers where the loss looks.
    if hard_neg > 0:
        p_now = keras.ops.stop_gradient(outputs['semantic_head'])
        w_sem = w + hard_neg * p_now * (1.0 - y_sem)
    else:
        w_sem = w
    sem = ftl(y_sem, outputs['semantic_head'], w_sem)
    # Flatter weights for the affinities: the EDT map is ~10 next to a wall and
    # ~0.1 inside a cell, so an error deep in a cell was almost free. Raising
    # the floor to `aff_floor` (a few times the interior value) keeps the
    # emphasis on walls but makes interior holes cost something. The mean weight
    # stays about the same, so the overall scale of the term is unchanged.
    w_a = keras.ops.maximum(w, aff_floor)
    aff = keras.ops.mean(w_a * keras.ops.abs(y_aff - outputs['affinity_head']))

    aux = 0.0
    for name, pred in outputs.items():
        if name.startswith('aux_semantic_'):
            aux = aux + ftl(y_sem, pred, w)  # same mask, same weights

    int_wall, int_aff = interior_errors(outputs, y, n_aff, interior_margin)
    interior = int_wall + int_aff

    dots = interior_dots(outputs, y, interior_margin, dot_radius)

    total = (
        sem + w_aff * aff + w_aux * aux + w_interior * interior + w_dot * dots
    )
    parts = {
        'int_dots': dots,
        'sem': sem,
        'aff': aff,
        'aux': aux,
        'int_wall': int_wall,
        'int_aff': int_aff,
    }

    geo = 0.0
    if 'geometry_head' in outputs:
        if y.shape[-1] < 1 + n_aff + 4:
            raise ValueError(
                'The model has a geometry head but the batch has no geometry targets.'
            )
        geo, geo_parts = geometry_loss(
            outputs['geometry_head'], y[..., 1 + n_aff : 5 + n_aff], ftl
        )
        total = total + w_geo * geo
        parts.update(geo_parts)
    parts['geo'] = geo
    parts['total'] = total
    return total, parts


def forward(generator, x, training, amp=False):
    """
    Model forward pass. With amp the layers run in bfloat16 (weights stay
    float32), which roughly halves the activation memory; the outputs are
    converted back to float32 so the loss is computed at full precision.
    """
    if not amp:
        return generator(x, training=training)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        outputs = generator(x, training=training)
    return {k: keras.ops.cast(v, 'float32') for k, v in outputs.items()}


def mean_parts(parts_list):
    """Average a list of {name: value} dicts (the parts of several micro-batches)."""
    return {
        k: sum(p[k] for p in parts_list) / len(parts_list)
        for k in parts_list[0]
    }


def train_step(
    generator,
    optimizer,
    batch,
    ftl,
    device,
    accum=1,
    next_batch=None,
    amp=False,
    **loss_kw,
):
    """
    One optimisation step. `optimizer` is a torch optimizer, as in your loop.

    Gradient accumulation: with accum > 1 the step is made of `accum`
    micro-batches (the given one, then accum - 1 more from `next_batch()`).
    Every micro-batch contributes loss / accum to the gradient, which sums up
    to the gradient of the average loss, and the weights are updated once. The
    memory needed is that of ONE micro-batch.
    """
    accum = max(1, int(accum))
    optimizer.zero_grad()
    parts_list = []
    for k in range(accum):
        b = batch if k == 0 else next_batch()
        x, y, w = (t.float().to(device) for t in b)
        outputs = forward(generator, x, True, amp)
        total, parts = compute_loss(outputs, y, w, ftl, **loss_kw)
        (total / accum).backward()
        parts_list.append(
            {
                key: float(v.detach()) if hasattr(v, 'detach') else float(v)
                for key, v in parts.items()
            }
        )
    optimizer.step()
    return mean_parts(parts_list)


@torch.no_grad()
def validate(
    generator,
    val_loader,
    ftl,
    device,
    max_batches=20,
    amp=False,
    extra_metrics=True,
    **loss_kw,
):
    """
    Average loss and Dice (semantic head, threshold 0.5) on a FIXED validation
    set. Call this every N steps and keep the checkpoint with the best Dice.
    """
    sums = {
        'total': 0.0,
        'sem': 0.0,
        'aff': 0.0,
        'aux': 0.0,
        'geo': 0.0,
        'int_wall': 0.0,
        'int_aff': 0.0,
        'int_dots': 0.0,
        'dice': 0.0,
    }
    n = 0
    metrics = ValidationMetrics() if extra_metrics else None
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y, w = (t.float().to(device) for t in batch)
        outputs = forward(generator, x, False, amp)  # False: no dropout
        _, parts = compute_loss(outputs, y, w, ftl, **loss_kw)

        pred = (outputs['semantic_head'] > 0.5).float()
        true = y[..., :1]
        inter = (pred * true).sum()
        dice = (2 * inter + 1e-6) / (pred.sum() + true.sum() + 1e-6)

        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        sums['dice'] += float(dice)
        n += 1
        if metrics is not None:
            probs = outputs['semantic_head'][..., 0].float().cpu().numpy()
            truths = true[..., 0].cpu().numpy()
            for prob, truth in zip(probs, truths, strict=True):
                metrics.add(prob, truth)
    out = {k: v / max(n, 1) for k, v in sums.items()}
    if metrics is not None:
        out.update(metrics.result())
    return out


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
