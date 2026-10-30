"""Where does the W-net spend its time and memory?

    python scripts/profile_model.py [--batch 8] [--steps 10] [--no-bf16]

For several variants of the architecture it prints
  * parameters and multiply-adds (MACs) per resolution of the convolutions
  * milliseconds for one training step (forward + backward) and peak GPU memory,
    in float32 and, if the GPU supports it, in bfloat16 mixed precision.
MACs only count convolutions; the memory-bound layers (GroupNorm, attention
gates, concatenations) show up in the timing instead. Compare the variants to
see which part of the network is worth cutting.
"""

import os

os.environ.setdefault('KERAS_BACKEND', 'torch')

import argparse
import time
from collections import defaultdict

import keras
import torch

from w_seg_napari._training.model import Generator

VARIANTS = {
    'baseline': {},
    'no cross-attention': {'cross_attention': False},
    'no deep supervision': {'deep_supervision': False},
    'no cross-attn, no deep sup.': {
        'cross_attention': False,
        'deep_supervision': False,
    },
    'widths x0.75': {'widths': (24, 48, 96, 192, 384)},
    'widths x0.5': {'widths': (16, 32, 64, 128, 256)},
    'with geometry head': {'geometry': True},
}


def conv_macs(layer):
    """(macs, params) of one conv-like layer, or None."""
    name = type(layer).__name__
    try:
        out = layer.output.shape
        inp = layer.input.shape
    except (AttributeError, ValueError, RuntimeError):
        return None
    if name in (
        'Conv2D',
        'Conv2DTranspose',
        'SeparableConv2D',
        'DepthwiseConv2D',
    ):
        h, w = out[1], out[2]
        k = layer.kernel_size[0] * layer.kernel_size[1]
        cin = inp[-1]
        if name == 'Conv2D':
            macs = h * w * k * cin * layer.filters / layer.groups
        elif name == 'Conv2DTranspose':
            macs = inp[1] * inp[2] * k * cin * layer.filters
        elif name == 'SeparableConv2D':
            macs = h * w * k * cin + h * w * cin * layer.filters
        else:
            macs = h * w * k * cin
        return macs, layer.count_params(), int(h)
    return None


def table_by_resolution(model):
    macs, params = defaultdict(float), defaultdict(int)
    for layer in model.layers:
        r = conv_macs(layer)
        if r:
            macs[r[2]] += r[0]
            params[r[2]] += r[1]
    total = sum(macs.values())
    print('  resolution    GMACs   share   params')
    for res in sorted(macs, reverse=True):
        print(
            f'  {res:>5} px  {macs[res] / 1e9:8.2f}  {100 * macs[res] / total:5.1f}%  '
            f'{params[res]:>10,}'
        )
    print(
        f'  total        {total / 1e9:8.2f}          {model.count_params():>10,}'
    )


def time_step(model, batch, steps, device):
    x = torch.rand(batch, 256, 256, 6, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    def one():
        opt.zero_grad()
        out = model(x, training=True)
        loss = sum(keras.ops.mean(v) for v in out.values())
        loss.backward()
        opt.step()

    for _ in range(3):  # warm-up (kernel selection, allocation)
        one()
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(steps):
        one()
    if device == 'cuda':
        torch.cuda.synchronize()
    ms = 1000 * (time.time() - t0) / steps
    mem = (
        torch.cuda.max_memory_allocated() / 2**30 if device == 'cuda' else 0.0
    )
    return ms, mem


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--no-bf16', action='store_true')
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('No GPU found: timings on the CPU are not representative.')
    policies = ['float32']
    if (
        device == 'cuda'
        and not args.no_bf16
        and torch.cuda.is_bf16_supported()
    ):
        policies.append('mixed_bfloat16')

    for policy in policies:
        keras.config.set_dtype_policy(policy)
        print(f'\n===== precision: {policy}, batch {args.batch} =====')
        results = []
        for name, kwargs in VARIANTS.items():
            model = Generator(**kwargs)
            if policy == 'float32' and name == 'baseline':
                print('\nbaseline convolutions by resolution:')
                table_by_resolution(model)
            ms, mem = time_step(model, args.batch, args.steps, device)
            results.append((name, model.count_params(), ms, mem))
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()
        print(
            f'\n  {"variant":30s} {"params":>12s} {"ms/step":>9s} {"GB peak":>8s}'
        )
        base_ms = results[0][2]
        for name, n, ms, mem in results:
            print(
                f'  {name:30s} {n:>12,} {ms:9.1f} {mem:8.2f}  ({100 * ms / base_ms:.0f}% of baseline)'
            )


if __name__ == '__main__':
    main()
