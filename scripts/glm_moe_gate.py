"""Real GLM-5.3-Flash sparse MoE against the pinned upstream reference.

Two arms, compared independently so a failure names its own half:

  ROUTER   router logits -> selected expert IDs -> post-selection weights
  BRANCH   the whole routed FFN (288 experts, top-8, native FP8) and the
           shared expert, each against the reference's own value

The router is the dangerous half and is checked in pieces, because every
ingredient here is one that produces fluent-but-wrong output if taken for
the neighbouring one: **sigmoid** not softmax; a correction bias that
shifts SELECTION but must never shift WEIGHTING; top-8; renormalisation;
and a routed scale of 2.5 applied to the routed branch only.

Two properties that trip naive comparisons, both handled explicitly:

  * `torch.topk(..., sorted=False)` — the reference's expert ORDER is
    undefined, so ids are compared as a SET.
  * the checkpoint ships per-expert `gate_proj`/`up_proj` while the
    reference module holds a fused `gate_up_proj`; the mapping is
    `cat([gate, up], dim=0)`, read off `_apply_gate`'s own `chunk(2)`.
"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np
import torch

CKPT = "/Volumes/model-drive/models/GLM-5.3-Flash"
TOL = 5e-5


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def build_reference_moe(ckpt, layer, tcfg):
    """The reference `Glm5NextTextMoE` on this layer's real weights.

    The checkpoint's per-expert matrices are fused into the module's
    `gate_up_proj` exactly as `_apply_gate`'s `chunk(2, dim=-1)` implies:
    gate rows first, then up.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from glm_layer_oracle import load_layer
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMoE

    prefix = f"model.language_model.layers.{layer}.mlp"
    sd = load_layer(ckpt, prefix, torch.float32)
    n = tcfg.num_local_experts
    gate = torch.stack([sd[f"experts.{e}.gate_proj.weight"] for e in range(n)])
    up = torch.stack([sd[f"experts.{e}.up_proj.weight"] for e in range(n)])
    down = torch.stack([sd[f"experts.{e}.down_proj.weight"] for e in range(n)])

    moe = Glm5NextTextMoE(tcfg).to(torch.float32)
    state = {
        "experts.gate_up_proj": torch.cat([gate, up], dim=1),
        "experts.down_proj": down,
        "gate.weight": sd["gate.weight"],
        "gate.e_score_correction_bias": sd["gate.e_score_correction_bias"],
        "shared_experts.gate_proj.weight": sd["shared_experts.gate_proj.weight"],
        "shared_experts.up_proj.weight": sd["shared_experts.up_proj.weight"],
        "shared_experts.down_proj.weight": sd["shared_experts.down_proj.weight"],
    }
    missing, unexpected = moe.load_state_dict(state, strict=True)
    moe.eval()
    return moe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--positions", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--probe", default="target/release/examples/glm_moe_probe")
    ap.add_argument(
        "--control",
        choices=["no-renorm", "no-scale", "gpt-oss-glu", "no-clamp"],
        help="Perturb a DECLARED routing or gating fact and require the "
             "comparison to FAIL. `no-renorm` and `no-scale` keep the selected "
             "experts and change only their weights, so a gate that merely "
             "agreed on top-k could not pass them. `gpt-oss-glu` is the defect "
             "this rung found: GPT-OSS's (u+1)*g*sigmoid(a*g) served for GLM's "
             "clamped SwiGLU. (The bias-corrected selection vs unbiased "
             "weighting split has its own unit control, "
             "`kimi_router::Mutation::GatherBiasedWeights`, over the same "
             "router code.)",
    )
    ap.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help="Scale the input so `swiglu_limit` actually bites. At the "
             "residual scale the gate peaks far below 10, so `--control "
             "no-clamp` is INERT there — which is a fact about the fixture, "
             "not a failure of the control, and the gate says so.",
    )
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from glm_layer_oracle import build_config

    tcfg = build_config(args.ckpt)
    moe = build_reference_moe(args.ckpt, args.layer, tcfg)

    g = torch.Generator().manual_seed(args.seed)
    x = torch.randn(1, args.positions, tcfg.hidden_size, generator=g) * 0.02 * args.gain

    with torch.no_grad():
        logits, weights, ids = moe.gate(x)
        want_total = moe(x).view(args.positions, -1).numpy()
        want_shared = moe.shared_experts(x).view(args.positions, -1).numpy()
        # The probe computes the ROUTED branch alone; the reference's
        # `forward` is `experts(x) + shared_experts(x)`, an exact sum, so
        # the routed half is recovered by subtraction. Compared apart on
        # purpose: a disagreement then names which branch it is in.
        want_routed = want_total - want_shared
    print(f"reference: {tcfg.num_local_experts} experts, top-{tcfg.num_experts_per_tok}, "
          f"scale {tcfg.routed_scaling_factor}, norm_topk={tcfg.norm_topk_prob}, "
          f"scoring={tcfg.scoring_func}\n")

    env = dict(
        os.environ,
        GLM_EXPERTS=str(tcfg.num_local_experts),
        GLM_TOP_K=str(tcfg.num_experts_per_tok),
        GLM_MOE_INTERMEDIATE=str(tcfg.moe_intermediate_size),
        GLM_ROUTED_SCALE=repr(tcfg.routed_scaling_factor),
        GLM_SWIGLU_LIMIT=repr(tcfg.swiglu_limit),
    )
    if args.control:
        env["GLM_MOE_CONTROL"] = args.control

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "x.f32"), "wb") as f:
            f.write(x.view(-1).numpy().astype("<f4").tobytes())
        subprocess.run(
            [args.probe, args.ckpt, str(args.layer), os.path.join(d, "x.f32"), d],
            check=True, env=env,
        )
        got = np.fromfile(os.path.join(d, "routed.f32"), dtype="<f4").reshape(args.positions, -1)

    # The routed branch INCLUDES the shared expert, as the reference's own
    # `forward` composes it. Reported together and separately so a
    # disagreement can be attributed to one side.
    print("routed branch (288 experts, top-8, native FP8; shared expert compared apart):")
    worst = 0.0
    for p in range(args.positions):
        e = rel(got[p], want_routed[p])
        worst = max(worst, e)
        sel = sorted(ids[p].tolist())
        print(f"  {'ok  ' if e <= TOL else 'FAIL'} position {p}  rel {e:.3e}   "
              f"experts {sel}")
        print(f"       weights {np.sort(weights[p].numpy())[::-1].round(4).tolist()}")

    print(f"\n  branch magnitudes: |routed| {np.linalg.norm(want_routed):.4f}   "
          f"|shared| {np.linalg.norm(want_shared):.4f}   "
          f"|total| {np.linalg.norm(want_total):.4f}")
    print(f"\nworst {worst:.3e} against a bar of {TOL:.0e}")

    if args.control:
        if worst <= TOL:
            print(f"control `{args.control}` is INERT at gain {args.gain:g} — it moved "
                  f"nothing, so this run qualifies nothing about it")
            if args.control in ("no-clamp",):
                print("  (expected: the clamp only bites well above the residual scale; "
                      "re-run with --gain 200)")
            sys.exit(1)
        print(f"control `{args.control}` fires: the gate can return the other answer")
        return
    if worst > TOL:
        print("DISAGREEMENT")
        sys.exit(1)
    print("real GLM sparse MoE agrees with the upstream reference")


if __name__ == "__main__":
    main()
