#!/usr/bin/env python3
"""Export a real-weight complete-KDA-layer parity fixture.

Loads one layer's 15 KDA operands, its two norms, and its router — routes
on the seeded input FIRST, then loads only the selected experts plus the
shared one (same sparsity discipline as `kimi_moe_export.py`).

    python scripts/kimi_kda_layer_export.py <checkpoint> --layer 1 --out DIR
    LARQL_KIMI_KDA_LAYER_FIXTURE=DIR cargo test -p larql-vindex --lib kimi_kda_layer_real
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import kda_reference as kda_ref
import kimi_kda_layer_reference as layer_ref
import kimi_moe_reference as moe_ref
from kda_fixture import load_layer as load_kda_operands
from kimi_moe_export import read_tensors, write, write_bf16

INPUT_SEED = 20260827


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--layer", type=int, default=1, help="a KDA layer that is ALSO routed")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads((args.checkpoint / "config.json").read_text())
    hidden = cfg["hidden_size"]
    eps = cfg["rms_norm_eps"]
    experts_n = cfg["num_experts"]
    top_k = cfg["num_experts_per_token"]
    moe_renormalize = cfg["moe_renormalize"]
    branch_scale = cfg["routed_scaling_factor"]
    kda_layers = {i - 1 for i in cfg["linear_attn_config"]["kda_layers"]}
    assert args.layer in kda_layers, f"layer {args.layer} is not a KDA layer"

    kda_w = load_kda_operands(args.checkpoint, args.layer)
    print(f"layer {args.layer}: KDA {kda_w.num_heads} heads x {kda_w.head_dim}, hidden {hidden}")

    prefix = f"model.layers.{args.layer}."
    norms = read_tensors(
        args.checkpoint,
        {prefix + "input_layernorm.weight", prefix + "post_attention_layernorm.weight"},
    )
    input_norm_w = norms[prefix + "input_layernorm.weight"]
    post_attn_norm_w = norms[prefix + "post_attention_layernorm.weight"]

    moe_prefix = prefix + "block_sparse_moe."
    router_tensors = read_tensors(
        args.checkpoint,
        {moe_prefix + "gate.weight", moe_prefix + "gate.e_score_correction_bias"},
    )
    router = moe_ref.RouterWeights(
        weight=router_tensors[moe_prefix + "gate.weight"],
        e_score_correction_bias=router_tensors[moe_prefix + "gate.e_score_correction_bias"],
    )

    torch.manual_seed(INPUT_SEED)
    x = torch.randn(hidden, dtype=torch.float32) * 0.02

    # Run the norm + attention half directly (not through
    # `kda_decoder_layer_forward`, which also needs the expert set this
    # loop exists to discover) to learn which experts the router selects,
    # loading only those — same sparsity discipline as `kimi_moe_export.py`.
    residual = x
    h = layer_ref.rms_norm(x, input_norm_w, eps)
    attn = kda_ref.kda_forward(h.unsqueeze(0), kda_w)
    after_attention = residual + attn["output"][0]
    post_normed = layer_ref.rms_norm(after_attention, post_attn_norm_w, eps)
    routed = moe_ref.route(post_normed, router, top_k, moe_renormalize, branch_scale)
    ids = routed["ids"].tolist()
    print(f"  selected experts {sorted(ids)}")

    expert_names = {
        f"{moe_prefix}experts.{i}.{suf}.weight" for i in ids for suf in ("w1", "w2", "w3")
    }
    shared_names = {
        f"{moe_prefix}shared_experts.{suf}.weight" for suf in ("gate_proj", "up_proj", "down_proj")
    }
    tensors = read_tensors(args.checkpoint, expert_names | shared_names)
    experts = {
        i: moe_ref.ExpertWeights(
            w1=tensors[f"{moe_prefix}experts.{i}.w1.weight"],
            w2=tensors[f"{moe_prefix}experts.{i}.w2.weight"],
            w3=tensors[f"{moe_prefix}experts.{i}.w3.weight"],
        )
        for i in ids
    }
    shared = moe_ref.ExpertWeights(
        w1=tensors[f"{moe_prefix}shared_experts.gate_proj.weight"],
        w2=tensors[f"{moe_prefix}shared_experts.down_proj.weight"],
        w3=tensors[f"{moe_prefix}shared_experts.up_proj.weight"],
    )

    result = layer_ref.kda_decoder_layer_forward(
        x, input_norm_w, post_attn_norm_w, eps, kda_w, router,
        experts, shared, top_k, moe_renormalize, branch_scale,
    )
    # Sanity: the two-pass probe above must agree with the real run's own
    # routing decision, or this fixture is internally inconsistent.
    assert result["moe"]["router"]["ids"].tolist() == ids

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": str(args.checkpoint),
        "layer": args.layer,
        "hidden": hidden,
        "rms_eps": eps,
        "num_heads": kda_w.num_heads,
        "head_dim": kda_w.head_dim,
        "moe_intermediate_size": cfg["moe_intermediate_size"],
        "num_shared_experts": cfg["num_shared_experts"],
        "experts": experts_n,
        "top_k": top_k,
        "moe_renormalize": moe_renormalize,
        "routed_scaling_factor": branch_scale,
        "selected_ids_order": ids,
        "files": {},
    }

    def dump(name: str, t: torch.Tensor) -> None:
        manifest["files"][name] = write(args.out / f"{name}.f32", t)

    def dump_bf16(name: str, t: torch.Tensor) -> None:
        # P4a (experts) / P4c-4 (KDA q/k/v/o_proj): BF16 code units, never
        # F32 — real checkpoint tensors are already bf16-native (the f32
        # upcast `read_tensors` did is lossless), so truncating back needs
        # no separate quantisation step.
        manifest["files"][name] = write_bf16(args.out / f"{name}.bf16", t)

    #: `KdaWeights`'s four widest operands (P4c-4) — dumped bf16, like the
    #: expert weights; everything else in the dataclass stays f32.
    kda_bf16_fields = {"q_proj", "k_proj", "v_proj", "o_proj"}

    dump("input", x)
    dump("input_norm_weight", input_norm_w)
    dump("post_attention_norm_weight", post_attn_norm_w)
    for name in kda_ref.KdaWeights.__dataclass_fields__:
        t = getattr(kda_w, name)
        (dump_bf16 if name in kda_bf16_fields else dump)(f"kda_{name}", t)
    dump("router_weight", router.weight)
    dump("router_bias", router.e_score_correction_bias)
    for i in ids:
        e = experts[i]
        dump_bf16(f"expert{i}_w1", e.w1)
        dump_bf16(f"expert{i}_w2", e.w2)
        dump_bf16(f"expert{i}_w3", e.w3)
    dump_bf16("shared_w1", shared.w1)
    dump_bf16("shared_w2", shared.w2)
    dump_bf16("shared_w3", shared.w3)

    dump("out_input_normed", result["input_normed"])
    dump("out_attention_output", result["attention"]["output"][0])
    dump("out_after_attention", result["after_attention"])
    dump("out_post_attention_normed", result["post_attention_normed"])
    for idx, out in enumerate(result["moe"]["expert_outputs"]):
        dump(f"out_expert_output_{idx}", out)
    dump("out_routed_sum", result["moe"]["routed_sum"])
    dump("out_shared_output", result["moe"]["shared_output"])
    dump("out_layer_output", result["output"])

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(p.stat().st_size for p in args.out.glob("*.f32")) / 2**20 + sum(p.stat().st_size for p in args.out.glob("*.bf16")) / 2**20
    print(f"layer output absmax {result['output'].abs().max():.6f}")
    print(f"{total:.1f} MiB written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
