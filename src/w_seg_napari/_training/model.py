"""
W-shaped segmentation network (dual-stream encoder: standard + separable conv).

Changes vs. the pix2pix-derived version:
  * residual double-conv blocks at every scale (encoder, bottleneck, decoder)
  * cross-stream fusion (with optional attention) after every encoder level
  * GroupNorm instead of InstanceNorm
  * SpatialDropout2D(0.1) instead of Dropout(0.5)
  * deep supervision: auxiliary semantic heads at 32/64/128 px decoder levels
  * He-normal init (instead of N(0, 0.02))
  * output layers forced to float32 (safe with mixed precision)
"""

import keras
from keras import layers, ops

INIT = 'he_normal'
GROUPS = 8  # all channel counts used below are divisible by 8


# =====================================================================
# Building blocks
# =====================================================================
def norm():
    return layers.GroupNormalization(groups=GROUPS, epsilon=1e-5)


def act():
    return layers.LeakyReLU(0.2)


def conv_norm_act(
    x, filters, size=3, stride=1, separable=False, dilation=1, activation=True
):
    """Conv (standard or separable) -> GroupNorm -> LeakyReLU."""
    if separable:
        x = layers.SeparableConv2D(
            filters,
            size,
            strides=stride,
            padding='same',
            dilation_rate=dilation,
            depthwise_initializer=INIT,
            pointwise_initializer=INIT,
            use_bias=False,
        )(x)
    else:
        x = layers.Conv2D(
            filters,
            size,
            strides=stride,
            padding='same',
            dilation_rate=dilation,
            kernel_initializer=INIT,
            use_bias=False,
        )(x)
    x = norm()(x)
    if activation:
        x = act()(x)
    return x


def res_block(x, filters, separable=False, dilation=1):
    """Residual double-conv block: (conv-norm-act, conv-norm) + shortcut."""
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, 1, kernel_initializer=INIT, use_bias=False
        )(x)
    y = conv_norm_act(x, filters, 3, separable=separable, dilation=dilation)
    y = conv_norm_act(
        y, filters, 3, separable=separable, dilation=dilation, activation=False
    )
    y = layers.add([shortcut, y])
    return act()(y)


def attention_gate(gated_signal, skip_signal, inter_filters):
    """Additive attention gate (Attention U-Net). Both inputs same H x W."""
    theta_x = layers.Conv2D(
        inter_filters,
        1,
        padding='same',
        kernel_initializer=INIT,
        use_bias=False,
    )(skip_signal)
    phi_g = layers.Conv2D(
        inter_filters,
        1,
        padding='same',
        kernel_initializer=INIT,
        use_bias=False,
    )(gated_signal)
    x = layers.Activation('relu')(layers.add([theta_x, phi_g]))
    psi = layers.Conv2D(
        1, 1, padding='same', activation='sigmoid', use_bias=True
    )(x)
    return layers.multiply([skip_signal, psi])


def cross_fuse(a, b, filters, use_attention=True):
    """
    Exchange information between the standard stream (a) and the separable
    stream (b). Each stream receives a residual update computed from both.

    With use_attention=True each stream is first gated by the other one
    (a is gated by b, b is gated by a) so the fusion only passes the
    parts of the other stream that are relevant to this stream.
    """
    if use_attention:
        inter = max(filters // 2, 8)
        a_g = attention_gate(b, a, inter)
        b_g = attention_gate(a, b, inter)
    else:
        a_g, b_g = a, b

    cat = layers.Concatenate(axis=-1)([a_g, b_g])
    a_upd = layers.Conv2D(filters, 1, kernel_initializer=INIT, use_bias=False)(
        cat
    )
    b_upd = layers.Conv2D(filters, 1, kernel_initializer=INIT, use_bias=False)(
        cat
    )
    a_out = layers.add([a, a_upd])
    b_out = layers.add([b, b_upd])
    return a_out, b_out


def upsample(x, filters, use_resize_conv=False):
    """Transposed conv (default) or bilinear resize + 3x3 conv."""
    if use_resize_conv:
        x = layers.UpSampling2D(2, interpolation='bilinear')(x)
        x = layers.Conv2D(
            filters, 3, padding='same', kernel_initializer=INIT, use_bias=False
        )(x)
    else:
        x = layers.Conv2DTranspose(
            filters,
            4,
            strides=2,
            padding='same',
            kernel_initializer=INIT,
            use_bias=False,
        )(x)
    x = norm()(x)
    return layers.ReLU()(x)


# =====================================================================
# Losses (use the same names in compile())
# =====================================================================
@keras.saving.register_keras_serializable(package='w_seg')
def dice_bce_loss(y_true, y_pred, smooth=1.0):
    y_true = ops.cast(y_true, y_pred.dtype)
    bce = ops.mean(keras.losses.binary_crossentropy(y_true, y_pred))
    axes = [1, 2, 3]
    inter = ops.sum(y_true * y_pred, axis=axes)
    denom = ops.sum(y_true, axis=axes) + ops.sum(y_pred, axis=axes)
    dice = 1.0 - (2.0 * inter + smooth) / (denom + smooth)
    return bce + ops.mean(dice)


# =====================================================================
# Generator
# =====================================================================
def Generator(
    input_shape=(256, 256, 6),
    widths=(32, 64, 128, 256, 512),
    cross_attention=True,
    deep_supervision=True,
    spatial_dropout=0.1,
    use_resize_conv=False,
    n_affinity=8,
):
    """
    widths: channels at 256, 128, 64, 32, 16 px. widths[-1] is the bottleneck.
    n_affinity: number of affinity channels = len(offsets) in the config
        (default 8 = 4 directions x 2 ranges; the old model used 2).
    Returns a model whose outputs are a dict:
        semantic_head (1 channel), affinity_head (n_affinity channels)
        (+ aux_semantic_32 / _64 / _128 when deep_supervision=True)
    """
    inputs = layers.Input(shape=input_shape)
    n_levels = len(widths)

    # --- 1. Stem (full resolution) + first fusion -----------------------
    a = res_block(conv_norm_act(inputs, widths[0], 3), widths[0])
    b = res_block(
        conv_norm_act(inputs, widths[0], 3, separable=True),
        widths[0],
        separable=True,
    )
    a, b = cross_fuse(a, b, widths[0], cross_attention)

    skips_a, skips_b = [a], [b]

    # --- 2. Parallel encoders with fusion at every scale -----------------
    for f in widths[1:]:
        a = conv_norm_act(a, f, 4, stride=2)
        a = res_block(a, f)
        b = conv_norm_act(b, f, 4, stride=2, separable=True)
        b = res_block(b, f, separable=True)
        a, b = cross_fuse(a, b, f, cross_attention)
        skips_a.append(a)
        skips_b.append(b)

    # --- 3. Bottleneck -------------------------------------------------
    f_bn = widths[-1]
    x = layers.Concatenate(axis=-1)([skips_a[-1], skips_b[-1]])
    x = conv_norm_act(x, f_bn, 1)
    for d in (2, 4):
        x = res_block(x, f_bn, dilation=d)

    # Skips excluding the bottleneck, deepest first
    skips_a = list(reversed(skips_a[:-1]))
    skips_b = list(reversed(skips_b[:-1]))

    # --- 4. Decoder --------------------------------------------------------
    aux_outputs = {}
    n_stages = n_levels - 1
    for i in range(n_stages):
        f = skips_a[i].shape[-1]
        up = upsample(x, f, use_resize_conv)

        inter = max(f // 2, 8)
        gate_a = attention_gate(up, skips_a[i], inter)
        gate_b = attention_gate(up, skips_b[i], inter)

        merged = layers.Concatenate(axis=-1)([up, gate_a, gate_b])
        x = res_block(merged, f)
        if spatial_dropout and i < 2:  # only the two deepest decoder stages
            x = layers.SpatialDropout2D(spatial_dropout)(x)

        # Deep supervision on every stage except the last (full-res) one
        if deep_supervision and i < n_stages - 1:
            scale = 2 ** (n_stages - 1 - i)
            side = layers.Conv2D(
                1,
                1,
                activation='sigmoid',
                dtype='float32',
                kernel_initializer=INIT,
            )(x)
            side = layers.UpSampling2D(
                scale,
                interpolation='bilinear',
                dtype='float32',
                name=f'aux_semantic_{input_shape[0] // scale}',
            )(side)
            aux_outputs[f'aux_semantic_{input_shape[0] // scale}'] = side

    # --- 5. Output heads ------------------------------------------------
    x_sem = layers.Conv2D(
        32, 3, padding='same', activation='relu', kernel_initializer=INIT
    )(x)
    semantic = layers.Conv2D(
        1,
        1,
        activation='sigmoid',
        dtype='float32',
        kernel_initializer=INIT,
        name='semantic_head',
    )(x_sem)

    x_aff = layers.Conv2D(
        32, 3, padding='same', activation='relu', kernel_initializer=INIT
    )(x)
    affinity = layers.Conv2D(
        n_affinity,
        1,
        activation='sigmoid',
        dtype='float32',
        kernel_initializer=INIT,
        name='affinity_head',
    )(x_aff)

    outputs = {'semantic_head': semantic, 'affinity_head': affinity}
    outputs.update(aux_outputs)

    return keras.Model(inputs, outputs, name='WASP-Generator-V3')


def compile_model(model, lr=2e-4):
    """Example compile: Dice+BCE on semantic maps, BCE on affinities."""
    losses = {
        'semantic_head': dice_bce_loss,
        'affinity_head': 'binary_crossentropy',
    }
    weights = {'semantic_head': 1.0, 'affinity_head': 1.0}
    for name in model.output_names:
        if name.startswith('aux_semantic_'):
            losses[name] = dice_bce_loss
            weights[name] = 0.25  # small weight for the auxiliary heads
    model.compile(
        optimizer=keras.optimizers.AdamW(lr, weight_decay=1e-4),
        loss=losses,
        loss_weights=weights,
    )
    return model


if __name__ == '__main__':
    m = Generator()
    m.summary()
