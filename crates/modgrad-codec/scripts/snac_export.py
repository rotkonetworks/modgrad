#!/usr/bin/env python3
"""
Export hubertsiuzdak/snac_24khz decoder weights to a format the modgrad
SNAC port can read.

PyTorch SNAC uses `torch.nn.utils.parametrizations.weight_norm` on every
Conv1d / ConvTranspose1d, which stores `parametrizations.weight.original0`
(g, scalar per output channel) and `parametrizations.weight.original1`
(v, full weight tensor). The effective weight is

    w_effective = (g / ||v||) * v

We fuse that here so the runtime never sees the parametrisation and just
runs plain convolution math.

Output:
  - `snac24k.safetensors`        — all fused tensors in safetensors format
  - `snac24k.manifest.json`      — tensor name + shape map for the Rust loader

Run (on bkk07, where the venv already has torch + snac + safetensors):
  cd /root/norpheus
  venv/bin/python snac_export.py /root/norpheus/snac_export
"""
import json
import os
import sys
from collections import OrderedDict

import torch
from snac import SNAC

try:
    from safetensors.torch import save_file
except ImportError:
    print("error: pip install safetensors", file=sys.stderr)
    raise


def fuse_weight_norm(state_dict: "OrderedDict[str, torch.Tensor]") -> "OrderedDict[str, torch.Tensor]":
    """Collapse `*.parametrizations.weight.original0` + `original1` into
    a single `*.weight` entry. Leaves non-parametrised tensors alone."""
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    g_keys: dict[str, str] = {}   # base_name -> g key
    v_keys: dict[str, str] = {}   # base_name -> v key

    for k in state_dict.keys():
        if k.endswith(".parametrizations.weight.original0"):
            base = k[: -len(".parametrizations.weight.original0")]
            g_keys[base] = k
        elif k.endswith(".parametrizations.weight.original1"):
            base = k[: -len(".parametrizations.weight.original1")]
            v_keys[base] = k

    fused_bases = set(g_keys) & set(v_keys)
    handled = set()
    for base in fused_bases:
        g = state_dict[g_keys[base]]
        v = state_dict[v_keys[base]]
        # weight_norm: effective = g * v / ||v|| (norm over all axes except 0)
        # but PyTorch's parametrisation stores g per output channel,
        # broadcast across (in_channels * kernel_size). Verify shapes.
        flat_v = v.reshape(v.shape[0], -1)
        norm_v = flat_v.norm(dim=1, keepdim=True).clamp_min(1e-8)
        normalized = (v.reshape(v.shape[0], -1) / norm_v).reshape_as(v)
        # g may be shape (out_ch,) or (out_ch, 1, 1) — match v's leading dim.
        if g.dim() == 1:
            g_bcast = g.view([-1] + [1] * (v.dim() - 1))
        else:
            g_bcast = g
        fused = g_bcast * normalized
        out[base + ".weight"] = fused.contiguous().float()
        handled.add(g_keys[base])
        handled.add(v_keys[base])

    for k, t in state_dict.items():
        if k in handled:
            continue
        # Skip pure parametrisation metadata that we already fused.
        if ".parametrizations.weight.original" in k:
            continue
        out[k] = t.float()
    return out


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/root/norpheus/snac_export"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/4] loading SNAC 24khz...", flush=True)
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

    print(f"[2/4] dumping state dict ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)...", flush=True)
    raw = OrderedDict(model.state_dict())
    print(f"  raw keys: {len(raw)}")

    print(f"[3/4] fusing weight_norm...", flush=True)
    fused = fuse_weight_norm(raw)
    print(f"  fused keys: {len(fused)}")

    # Quick sanity: print the unique top-level submodule names + a sample
    # of decoder-side tensors so we can verify which keys land where.
    by_module: dict[str, int] = {}
    for k in fused.keys():
        head = ".".join(k.split(".")[:2])
        by_module[head] = by_module.get(head, 0) + 1
    print("  top-level modules (tensor count):")
    for m, n in sorted(by_module.items()):
        print(f"    {m:40s}  {n}")

    print(f"[4/4] writing outputs to {out_dir}/", flush=True)
    safetensors_path = os.path.join(out_dir, "snac24k.safetensors")
    save_file(fused, safetensors_path)
    size_mb = os.path.getsize(safetensors_path) / 1024 / 1024
    print(f"  {safetensors_path}: {size_mb:.1f} MB")

    manifest = {
        "model": "hubertsiuzdak/snac_24khz",
        "fused_weight_norm": True,
        "tensors": {
            k: {"shape": list(t.shape), "dtype": str(t.dtype)}
            for k, t in fused.items()
        },
    }
    manifest_path = os.path.join(out_dir, "snac24k.manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  {manifest_path}: {os.path.getsize(manifest_path)} bytes")

    print("\ndone.")


if __name__ == "__main__":
    main()
