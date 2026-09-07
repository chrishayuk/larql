"""Bit-exactness gate for LARQL's fine-grained FP8 codec.

Runs `transformers.integrations.finegrained_fp8.Fp8Dequantize` — the
loader GLM-5.3-Flash's own `config.json` selects — over real tensors from
the real checkpoint, and compares the result against the f32 dump
produced by:

    cargo run --release -p larql-models --example fp8_dequant_probe

The bar is BIT-EXACT, not "close". Both sides do one f32 multiply per
element in the same order, so any difference is a real disagreement about
the format — a transposed scale grid, a wrong tile, a divide instead of a
multiply — and not accumulated error. A tolerance here would hide exactly
the failures the gate exists to catch.
"""
import argparse, json, os, struct, subprocess, sys, tempfile
import numpy as np
import torch
from transformers.integrations.finegrained_fp8 import Fp8Dequantize

CKPT = "/Volumes/model-drive/models/GLM-5.3-Flash"
# One tensor of each kind GLM ships in this format, so a defect specific
# to a shape or a grid cannot hide behind a passing sibling.
DEFAULT_TENSORS = [
    "model.language_model.layers.0.mlp.gate_proj.weight",    # dense FFN  [12288, 4096]
    "model.language_model.layers.0.mlp.down_proj.weight",    # transposed [4096, 12288]
    "model.language_model.layers.3.mlp.experts.0.up_proj.weight",   # routed expert
    "model.language_model.layers.3.mlp.shared_experts.down_proj.weight",
    "model.language_model.layers.3.self_attn.q_a_proj.weight",      # MLA q-LoRA down
    "model.language_model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
]


def shard_of(ckpt, name):
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))
    return idx["weight_map"][name]


def reference(path, name):
    """transformers' own dequantiser, at f32."""
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        w = f.get_tensor(name)
        s = f.get_tensor(name.removesuffix(".weight") + ".weight_scale_inv")
    op = Fp8Dequantize(hf_quantizer=None)
    return op._dequantize_one(w, s, output_dtype=torch.float32).numpy()


def corrupt(got, want, shard, name, how):
    """Introduce exactly the defect the format most invites.

    Each arm is a real mis-reading someone could ship, not random noise:
    the point is that the gate distinguishes THESE, not that it can see
    an arbitrary perturbation.
    """
    from safetensors import safe_open
    with safe_open(shard, framework="pt") as f:
        w = f.get_tensor(name)
        s = f.get_tensor(name.removesuffix(".weight") + ".weight_scale_inv")
    q = w.to(torch.float32).numpy()
    sr, sc = s.shape
    rows, cols = q.shape
    bm, bn = rows // sr, cols // sc
    sn = s.numpy()
    if how == "transpose-grid":
        if sr != sc:
            print(f"  (self-test transpose-grid needs a square grid; {sr}x{sc} — using divide)")
            how = "divide"
        else:
            sn = sn.T
    if how == "divide":
        grid = np.repeat(np.repeat(1.0 / sn, bm, axis=0), bn, axis=1)
    elif how == "wrong-tile":
        # Read the grid with the axes of the tile swapped — right shape,
        # wrong mapping, and only detectable on a non-square tiling.
        grid = np.repeat(np.repeat(sn, bm, axis=0), bn, axis=1)
        grid = np.roll(grid, bn, axis=1)
    elif how == "flip-one-bit":
        out = got.copy()
        out.view(np.uint32)[0] ^= 1
        return out
    else:
        grid = np.repeat(np.repeat(sn, bm, axis=0), bn, axis=1)
    return (q * grid).astype("<f4").reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--probe", default="target/release/examples/fp8_dequant_probe")
    ap.add_argument("--tensors", nargs="*", default=DEFAULT_TENSORS)
    ap.add_argument(
        "--self-test",
        choices=["transpose-grid", "divide", "wrong-tile", "flip-one-bit"],
        help="Corrupt the candidate before comparing, to prove this gate "
             "can FAIL. A comparison that has never returned the other "
             "answer is not evidence.",
    )
    args = ap.parse_args()

    failures = 0
    for name in args.tensors:
        shard = os.path.join(args.ckpt, shard_of(args.ckpt, name))
        with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as tmp:
            out = tmp.name
        try:
            subprocess.run([args.probe, shard, name, out], check=True)
            got = np.fromfile(out, dtype="<f4")
        finally:
            os.unlink(out)

        want = reference(shard, name).astype("<f4").reshape(-1)
        if args.self_test:
            got = corrupt(got, want, shard, name, args.self_test)
        if got.shape != want.shape:
            print(f"FAIL {name}: shape {got.shape} vs {want.shape}")
            failures += 1
            continue

        # Bit-level comparison: NaN must match NaN, and -0.0 must not pass
        # for +0.0. Comparing the raw bit patterns is the only check that
        # says "identical" rather than "numerically indistinguishable".
        same = got.view(np.uint32) == want.view(np.uint32)
        n_bad = int((~same).sum())
        if n_bad:
            i = int(np.argmax(~same))
            print(f"FAIL {name}: {n_bad}/{got.size} values differ; "
                  f"first at {i}: {got[i]!r} vs {want[i]!r}")
            failures += 1
        else:
            print(f"ok   {name}: {got.size:,} values bit-identical  "
                  f"(absmax {np.abs(want).max():.6g})")

    print()
    if failures:
        print(f"{failures}/{len(args.tensors)} tensors FAILED")
        sys.exit(1)
    if args.self_test:
        print(f"UNEXPECTED: self-test `{args.self_test}` did not make the gate fail — "
              f"the comparison is not actually checking anything")
        sys.exit(1)
    print(f"all {len(args.tensors)} tensors bit-identical to the reference")


if __name__ == "__main__":
    main()
