"""Smoke test: the GLM-5.3-Flash KDA oracle reconstructs and answers.

Passing this is the definition of "the oracle environment is rebuilt".
It proves four things in order, and each one can fail on its own:

  1. the pinned `transformers` imports and exposes `glm5_next`;
  2. the reference SOURCE hashes to what was judged (`verify_sources.sh`
     is the standalone form; repeated here so one command suffices);
  3. one real GLM KDA layer strict-loads from the real weights;
  4. it runs, and emits the boundaries PHYSICAL-1 scores — the decay
     gate and the layer output at every position.

Usage:
    tools/oracles/glm5/bootstrap.sh
    $LARQL_GLM_ORACLE/bin/python tools/oracles/glm5/smoke.py [--out fixture.npz]

`--out` writes the reference trajectory the Metal arm is compared to.
Without it this is a pure liveness check.
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))

CKPT = "/Volumes/model-drive/models/GLM-5.3-Flash"
# Layer 0 is a declared KDA layer: GLM's `kda_layers` is 0-indexed and
# covers 0..44 with the 11 DSA layers removed, and 0 is not among them.
LAYER = 0


def check_sources(models_dir, recorded):
    """The oracle is an authority only if its source is the judged one."""
    want = {}
    for line in open(recorded):
        h, _, name = line.strip().partition("  ")
        if name.startswith("glm5_next/"):
            want[name] = h
    assert want, f"no glm5_next rows in {recorded}"
    for name, expected in sorted(want.items()):
        p = os.path.join(models_dir, name)
        got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        assert got == expected, (
            f"{name} hashes {got}, recorded {expected} — the pinned reference "
            f"changed under this environment; re-judge before trusting any GLM "
            f"parity number produced with it"
        )
    print(f"  reference sources OK ({len(want)} files)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--positions", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--raw-dir",
        default=None,
        help="write input/output/decay as flat little-endian f32 so a Rust "
        "arm can read them without an npz reader",
    )
    args = ap.parse_args()

    print("1. environment")
    import transformers

    print(f"  transformers {transformers.__version__}, torch {torch.__version__}")
    assert transformers.__version__ == "5.16.1", (
        f"pinned oracle is transformers 5.16.1, found {transformers.__version__}"
    )
    models_dir = os.path.join(os.path.dirname(transformers.__file__), "models")

    print("2. reference sources")
    check_sources(models_dir, os.path.join(REPO, "scripts", "glm_reference_sources.sha256"))

    print("3. one real GLM KDA layer")
    if not os.path.isdir(args.ckpt):
        sys.exit(
            f"  checkpoint not mounted at {args.ckpt} — steps 1 and 2 passed, so the "
            f"ENVIRONMENT is rebuilt; mount the weights to complete the smoke test"
        )
    from glm_layer_oracle import build_config, load_layer
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextLinearAttention

    cfg = build_config(args.ckpt)
    cfg._attn_implementation = "eager"
    prefix = f"model.language_model.layers.{args.layer}.self_attn"
    sd = load_layer(args.ckpt, prefix, torch.float32)
    # The checkpoint's dialect is not the module's layout, and both
    # differences are load-bearing (funnel doc, F1/F2):
    #   - three per-stream `*_conv1d` tensors are ONE conv over
    #     cat([q, k, v]);
    #   - the decay-gate operands live under `forget_gate`.
    sd["conv1d.weight"] = torch.cat(
        [sd.pop(f"{a}_conv1d.weight") for a in ("q", "k", "v")], dim=0
    )
    for leaf in ("A_log", "dt_bias", "f_a_proj.weight", "f_b_proj.weight"):
        sd[f"forget_gate.{leaf}"] = sd.pop(leaf)
    attn = Glm5NextTextLinearAttention(cfg, args.layer).to(torch.float32)
    attn.load_state_dict(sd, strict=True)
    attn.eval()
    n = sum(v.numel() for v in sd.values())
    print(f"  strict-loaded {len(sd)} tensors, {n/1e9:.3f} B params")

    print("4. boundaries")
    hidden = cfg.hidden_size
    g = torch.Generator().manual_seed(args.seed)
    # Unit-RMS, matching the scale of the normalised hidden states KDA
    # consumes. The SAME sequence must be fed to the Metal arm.
    x = torch.randn(1, args.positions, hidden, generator=g)
    x = x / x.pow(2).mean(-1, keepdim=True).sqrt()

    rec = {}
    h = attn.forget_gate.register_forward_hook(
        lambda _m, _i, o: rec.__setitem__(
            "decay", (o[0] if isinstance(o, tuple) else o).detach().float()
        )
    )
    with torch.no_grad():
        out = attn(hidden_states=x)
    h.remove()
    out = out[0] if isinstance(out, tuple) else out

    rec["input"] = x.float()
    rec["output"] = out.detach().float()
    for k in sorted(rec):
        t = rec[k]
        assert torch.isfinite(t).all(), f"{k} is not finite"
        print(
            f"  {k:8s} {str(tuple(t.shape)):20s} mean={t.mean():+.6e} "
            f"std={t.std():+.6e} absmax={t.abs().max():.6e}"
        )
    assert rec["output"].abs().max() > 0, "the layer emitted all zeros"

    if args.raw_dir:
        os.makedirs(args.raw_dir, exist_ok=True)
        for k, t in rec.items():
            t.numpy().astype("<f4").tofile(os.path.join(args.raw_dir, f"{k}.f32"))
        with open(os.path.join(args.raw_dir, "meta.txt"), "w") as f:
            f.write(
                f"layer {args.layer}\npositions {args.positions}\nseed {args.seed}\n"
                f"hidden {hidden}\n"
            )
        print(f"\nwrote raw f32 to {args.raw_dir}")

    if args.out:
        np.savez(args.out, **{k: v.numpy() for k, v in rec.items()},
                 layer=np.array([args.layer]), positions=np.array([args.positions]),
                 seed=np.array([args.seed]))
        print(f"\nwrote {args.out}")
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
