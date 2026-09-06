"""GLM-5.3-Flash single-layer oracle.

Drives ONE real decoder layer of the checkpoint through the pinned
`transformers` reference (glm5_next @ 5.16.1), from the real weights on
`model-drive`, and dumps every boundary tensor.

Nothing here transcribes the reference: it *is* the reference. The only
LARQL-side code is the shard reader and the FP8 dequantiser, and the
dequantiser is checked against transformers' own `Fp8Dequantize` in
`--check-dequant` mode.
"""
import argparse, json, os, struct, sys, glob
import numpy as np
import torch
from safetensors import safe_open

CKPT = "/Volumes/model-drive/models/GLM-5.3-Flash"


def shard_map(ckpt):
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))
    return idx["weight_map"]


def dequant_fp8(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Block dequant, transcribed from transformers Fp8Dequantize._dequantize_one.

    The block size is derived from the SCALE GRID, never from
    `quantization_config.weight_block_size` — the same checkpoint may ship
    different grids for different tensors.
    """
    q = w.to(torch.float32)
    rows, cols = q.shape[-2:]
    sr, sc = scale.shape[-2:]
    if rows % sr or cols % sc:
        raise ValueError(f"weight {tuple(q.shape)} not divisible by scale grid {(sr, sc)}")
    bm, bn = rows // sr, cols // sc
    s = scale.to(torch.float32).reshape(-1, sr, sc).unsqueeze(-1).unsqueeze(2)
    return (q.reshape(-1, sr, bm, sc, bn) * s).reshape(q.shape)


def load_layer(ckpt, prefix, dtype=torch.float32, skip=None):
    """Every tensor under `prefix`, FP8 pairs folded, keyed relative to it.

    `skip` is a predicate on the FULL tensor name. It exists because a
    GLM sparse layer's prefix covers its 288-expert bank: 6.78 GiB of
    FP8, which is 27 GB once widened to f32, and a caller that only wants
    the layer's attention and router must be able to say so rather than
    discover it through the OOM killer.
    """
    wm = shard_map(ckpt)
    # The prefix must land on a SEPARATOR. A bare `...layers.1` is a
    # prefix of `...layers.10` through `...layers.19` as well, so the
    # naive `startswith` silently loads eleven layers — including sparse
    # ones, whose 288-expert banks are ~300 GB once widened to f32. It
    # was found by the OOM killer, with no output, on the second layer.
    #
    # Same rule the Rust side uses for `per_expert_bank_prefix`: a prefix
    # is only a prefix if the divergence falls on a `.` boundary.
    def under(n):
        return n == prefix or n.startswith(prefix + ".")

    names = [n for n in wm if under(n) and not (skip and skip(n))]
    by_shard = {}
    for n in names:
        by_shard.setdefault(wm[n], []).append(n)
    raw = {}
    for shard, ns in sorted(by_shard.items()):
        with safe_open(os.path.join(ckpt, shard), framework="pt") as f:
            for n in ns:
                raw[n] = f.get_tensor(n)
    out = {}
    for n, t in raw.items():
        if n.endswith("_scale_inv"):
            continue
        sib = n[: -len(".weight")] + ".weight_scale_inv" if n.endswith(".weight") else n + "_scale_inv"
        key = n[len(prefix):].lstrip(".")
        if sib in raw:
            out[key] = dequant_fp8(t, raw[sib]).to(dtype)
        else:
            out[key] = t.to(dtype) if t.is_floating_point() else t
    return out



# The checkpoint's tensor dialect is NOT the reference module's parameter
# layout. Two declared differences, both read off the reference itself:
#
#   * `conv1d.weight` is ONE depthwise conv over `cat([q, k, v], dim=-1)`
#     (`Glm5NextTextLinearAttention.forward`), while the checkpoint ships
#     `q_conv1d` / `k_conv1d` / `v_conv1d` separately. Order is q, k, v —
#     taken from the `torch.cat` that feeds it, not assumed.
#   * the decay gate lives in a `forget_gate` submodule
#     (`Glm5NextTextForgetGate`), while the checkpoint keeps its four
#     tensors flat on `self_attn`.
#   * mHC parameters are `hc_{site}_{p}` in the checkpoint and
#     `{site}_hc.{p}` on the module.
def remap_checkpoint_to_module(sd: dict) -> dict:
    out = dict(sd)
    convs = [f"self_attn.{a}_conv1d.weight" for a in ("q", "k", "v")]
    if all(c in out for c in convs):
        out["self_attn.conv1d.weight"] = torch.cat([out.pop(c) for c in convs], dim=0)
    for leaf in ("A_log", "dt_bias", "f_a_proj.weight", "f_b_proj.weight"):
        src = f"self_attn.{leaf}"
        if src in out:
            out[f"self_attn.forget_gate.{leaf}"] = out.pop(src)
    for site in ("attn", "ffn"):
        for p in ("fn", "base", "scale"):
            src = f"hc_{site}_{p}"
            if src in out:
                out[f"{site}_hc.{p}"] = out.pop(src)
    return out


def build_config(ckpt):
    sys.path.insert(0, os.path.dirname(__file__))
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
    cfg = Glm5NextConfig.from_pretrained(ckpt)
    return cfg.text_config if hasattr(cfg, "text_config") else cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--positions", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    tcfg = build_config(args.ckpt)
    tcfg._attn_implementation = "eager"

    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer

    prefix = f"model.language_model.layers.{args.layer}"
    print(f"loading {prefix} …", flush=True)
    sd = load_layer(args.ckpt, prefix, dtype)
    print(f"  {len(sd)} tensors, {sum(v.numel()*v.element_size() for v in sd.values())/2**30:.2f} GiB @ {args.dtype}")

    layer = Glm5NextTextDecoderLayer(tcfg, args.layer).to(dtype)
    sd = remap_checkpoint_to_module(sd)
    missing, unexpected = layer.load_state_dict(sd, strict=True)
    print(f"  loaded strict: missing={list(missing)} unexpected={list(unexpected)}")
    layer.eval()

    B, S, H, D = 1, args.positions, tcfg.hc_mult, tcfg.hidden_size
    streams = torch.randn(B, S, H, D, dtype=dtype, generator=torch.Generator().manual_seed(args.seed)) * 0.02

    rec = {}
    hooks = []

    def cap(name):
        def fn(_m, _i, o):
            t = o[0] if isinstance(o, tuple) else o
            if torch.is_tensor(t):
                rec[name] = t.detach().float()
        return fn

    for n, m in layer.named_modules():
        if n and len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(cap(n)))

    with torch.no_grad():
        out, topk = layer(hidden_states=streams, attention_mask=None)
    for h in hooks:
        h.remove()

    rec["__input_streams__"] = streams.float()
    rec["__output_streams__"] = out.float()
    print(f"\nboundaries captured: {len(rec)}")
    for k in sorted(rec):
        t = rec[k]
        print(f"  {k:52s} {str(tuple(t.shape)):22s} "
              f"mean={t.mean():+.6e} std={t.std():+.6e} absmax={t.abs().max():.6e}")

    if args.out:
        np.savez(args.out, **{k: v.numpy() for k, v in rec.items()})
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
