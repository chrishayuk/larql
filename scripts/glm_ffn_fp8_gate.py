"""Real GLM-5.3-Flash dense FFN through LARQL's fine-grained FP8 kernel.

Takes the layer-0 boundaries the pinned upstream reference produced
(`scripts/glm_layer_oracle.py`), feeds the reference's OWN
`post_attention_layernorm` output into LARQL's `FusedFp8Block` over the
checkpoint's own FP8 bytes, and compares every projection.

What this is and is not: it qualifies the ARITHMETIC of a real GLM dense
FFN read natively from FP8 — no dequantised weight image exists at any
point. It does not go through a VINDEX3 container, because GLM's plan is
not yet admissible; container carriage is qualified separately on a
synthetic FP8 checkpoint through the real encode path.

The bar is not bit-exactness here, and the reason is worth stating: the
reference materialises f32 weights and calls into BLAS, while the kernel
accumulates one tile at a time. Same values, different summation order,
so the two agree to f32 rounding over a 4096-term dot and not beyond.
Bit-exactness is claimed for the DECODE (`glm_fp8_dequant_gate.py`) —
which is the part where a disagreement would mean a format error rather
than an ordering one.
"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np

CKPT = "/Volumes/model-drive/models/GLM-5.3-Flash"
# Relative-error bar. Set from the arithmetic, not tuned: an f32 dot over
# 4096 terms accumulates ~sqrt(4096)*2^-24 ~ 4e-6 of relative rounding,
# and reassociating it (BLAS blocks vs 128-wide tiles) moves the result by
# about that. Anything above this is a real disagreement.
TOL = 2e-5


def rel(a, b):
    """Error normalised by the ROW's scale, not elementwise.

    An elementwise relative error on a vector whose entries span orders of
    magnitude reports the smallest entry's rounding as if it were the
    result's error.
    """
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def reference_mlp(ckpt, layer, x, gain):
    """The reference `Glm5NextTextMLP`, on real weights, at a gain that
    trips the clamp. Built here rather than read from the npz because the
    npz was produced at an activation scale where the clamp is inert."""
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from glm_layer_oracle import load_layer, build_config
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMLP

    tcfg = build_config(ckpt)
    prefix = f"model.language_model.layers.{layer}.mlp"
    sd = load_layer(ckpt, prefix, torch.float32)
    mlp = Glm5NextTextMLP(tcfg).to(torch.float32)
    mlp.load_state_dict(sd, strict=True)
    mlp.eval()
    xt = torch.from_numpy((x * gain).astype("float32")).view(1, 1, -1)
    L = mlp.swiglu_limit
    with torch.no_grad():
        gate = mlp.gate_proj(xt)
        up = mlp.up_proj(xt)
        n_gate = int((gate > L).sum())
        n_up = int((up.abs() > L).sum())
        n_below = int((gate < -L).sum())
        out = mlp(xt)

        # Is the ASYMMETRY load-bearing, or would a symmetric gate clamp
        # agree? Measured rather than assumed, because the answer is
        # counter-intuitive: `silu(-10)` is already -4.5e-4 and
        # `silu(-46)` is -1.3e-19, so clamping the gate's lower tail
        # changes almost nothing. The asymmetry is REAL in the source and
        # nearly INERT in the arithmetic, and only a measurement
        # distinguishes those.
        def down(gc, uc):
            return mlp.down_proj(torch.nn.functional.silu(gc) * uc).view(-1).numpy()

        shipped = down(gate.clamp(min=None, max=L), up.clamp(-L, L))
        symmetric = down(gate.clamp(-L, L), up.clamp(-L, L))
        asym = float(np.linalg.norm(shipped - symmetric) / np.linalg.norm(shipped))
    return out.view(-1).numpy(), n_gate, n_up, float(gate.abs().max()), n_below, asym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--npz", required=True, help="from scripts/glm_layer_oracle.py")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--position", type=int, default=0)
    ap.add_argument("--probe", default="target/release/examples/glm_ffn_fp8_probe")
    ap.add_argument(
        "--clamp-control",
        type=float,
        default=None,
        metavar="GAIN",
        help="Scale the input by GAIN so `swiglu_limit` actually bites, run the "
             "REFERENCE MLP on the same input, and compare. Without this the "
             "clamp is never exercised: at the oracle's own activation scale "
             "the gate peaks around 0.23 against a limit of 10.",
    )
    args = ap.parse_args()

    ref = np.load(args.npz)
    prefix = f"model.language_model.layers.{args.layer}"
    idx = json.load(open(os.path.join(args.ckpt, "model.safetensors.index.json")))
    shard = os.path.join(args.ckpt, idx["weight_map"][f"{prefix}.mlp.gate_proj.weight"])

    # The FFN's input is the reference's own post-attention norm output —
    # taken from the oracle rather than recomputed, so this measures the
    # FFN alone and not everything upstream of it.
    x = ref["post_attention_layernorm"][0, args.position].astype("<f4")

    if args.clamp_control is not None:
        gain = args.clamp_control
        want_down, n_gate, n_up, peak, n_below, asym = reference_mlp(
            args.ckpt, args.layer, x, gain
        )
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "x.f32"), "wb") as f:
                f.write((x * gain).astype("<f4").tobytes())
            subprocess.run([args.probe, shard, prefix, os.path.join(d, "x.f32"), d], check=True)
            got_down = np.fromfile(os.path.join(d, "down_proj.f32"), dtype="<f4")

        print(f"\nCLAMP CONTROL — input x{gain:g}, gate peaks at {peak:.3g} "
              f"against swiglu_limit\n")
        print(f"  reference clamped {n_gate} gate and {n_up} up values "
              f"({n_below} gate values sit below -limit)")
        print(f"  a SYMMETRIC gate clamp would move the output by rel {asym:.3e} — "
              f"the asymmetry is {'load-bearing' if asym > TOL else 'nearly inert'} here, "
              f"because silu(-limit) is already ~-4.5e-4 and silu(-peak) ~0")
        if n_gate == 0 and n_up == 0:
            print("  UNEXERCISED: the clamp never fired, so this run qualifies nothing")
            sys.exit(1)
        e = rel(got_down, want_down)
        print(f"  {'ok  ' if e <= TOL else 'FAIL'} down_proj  rel {e:.3e}")
        if e > TOL:
            sys.exit(1)
        print("\nthe asymmetric swiglu clamp agrees with the reference where it BITES")
        return

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "x.f32"), "wb") as f:
            f.write(x.tobytes())
        subprocess.run(
            [args.probe, shard, prefix, os.path.join(d, "x.f32"), d], check=True
        )
        got = {
            n: np.fromfile(os.path.join(d, f"{n}.f32"), dtype="<f4")
            for n in ("gate_proj", "up_proj", "act_fn", "down_proj")
        }

    want = {
        "gate_proj": ref["mlp.gate_proj"][0, args.position],
        "up_proj": ref["mlp.up_proj"][0, args.position],
        # `mlp.act_fn` is the module's output — `silu(clamped gate)`
        # alone. The GLU product is formed after it, outside any module,
        # so no hook sees it and it is checked through `down_proj`.
        "act_fn": ref["mlp.act_fn"][0, args.position],
        "down_proj": ref["mlp.down_proj"][0, args.position],
    }

    print(f"\nGLM layer {args.layer}, position {args.position} — dense FFN, "
          f"native FP8, no dequantised weight image\n")
    worst = 0.0
    for n in ("gate_proj", "up_proj", "act_fn", "down_proj"):
        e = rel(got[n], want[n])
        worst = max(worst, e)
        flag = "ok  " if e <= TOL else "FAIL"
        print(f"  {flag} {n:11s} {got[n].size:>6} values   rel {e:.3e}   "
              f"absmax(ref) {np.abs(want[n]).max():.6g}")

    print(f"\nworst {worst:.3e} against a bar of {TOL:.0e}")
    if worst > TOL:
        print("DISAGREEMENT — this is not a summation-order effect")
        sys.exit(1)
    print("real GLM dense FFN agrees with the upstream reference")


if __name__ == "__main__":
    main()
