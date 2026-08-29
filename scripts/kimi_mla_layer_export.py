#!/usr/bin/env python3
"""Export a real-weight complete-MLA-layer parity fixture.

Loads one FULL-ATTENTION layer's five MLA operands, its two norms, and
its router — routes on the real post-attention-normed hidden state at
EVERY position of a short sequence FIRST, then loads only the union of
selected experts across all positions plus the shared one (same
sparsity discipline as `kimi_moe_export.py`/`kimi_kda_layer_export.py`).

Three positions, not one: a single cached position cannot exercise MLA's
attention math at all (softmax over one score is 1.0 regardless of its
value) — see `exec::mla`'s own doc comment.

    python scripts/kimi_mla_layer_export.py <checkpoint> --layer 3 --out DIR
    LARQL_KIMI_MLA_LAYER_FIXTURE=DIR cargo test -p larql-vindex --lib kimi_mla_layer_real
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import kimi_mla_layer_reference as layer_ref
import kimi_mla_reference as mla_ref
import kimi_moe_reference as moe_ref
from kimi_moe_export import read_tensors, write, write_bf16

INPUT_SEED = 20260828
POSITIONS = 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--layer", type=int, default=3, help="a full-attention (MLA) layer")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads((args.checkpoint / "config.json").read_text())
    hidden = cfg["hidden_size"]
    eps = cfg["rms_norm_eps"]
    experts_n = cfg["num_experts"]
    top_k = cfg["num_experts_per_token"]
    moe_renormalize = cfg["moe_renormalize"]
    branch_scale = cfg["routed_scaling_factor"]
    full_attn_layers = {i - 1 for i in cfg["linear_attn_config"]["full_attn_layers"]}
    assert args.layer in full_attn_layers, f"layer {args.layer} is not a full-attention layer"

    geometry = mla_ref.MlaGeometry(
        num_heads=cfg["num_attention_heads"],
        kv_lora_rank=cfg["kv_lora_rank"],
        qk_nope_head_dim=cfg["qk_nope_head_dim"],
        qk_rope_head_dim=cfg["qk_rope_head_dim"],
        v_head_dim=cfg["v_head_dim"],
    )

    prefix = f"model.layers.{args.layer}."
    attn_prefix = prefix + "self_attn."
    mla_tensors = read_tensors(args.checkpoint, {
        attn_prefix + "q_proj.weight",
        attn_prefix + "kv_a_proj_with_mqa.weight",
        attn_prefix + "kv_a_layernorm.weight",
        attn_prefix + "kv_b_proj.weight",
        attn_prefix + "o_proj.weight",
    })
    mla_w = mla_ref.MlaWeights(
        q_proj=mla_tensors[attn_prefix + "q_proj.weight"],
        kv_a_proj=mla_tensors[attn_prefix + "kv_a_proj_with_mqa.weight"],
        kv_a_norm=mla_tensors[attn_prefix + "kv_a_layernorm.weight"],
        kv_b_proj=mla_tensors[attn_prefix + "kv_b_proj.weight"],
        o_proj=mla_tensors[attn_prefix + "o_proj.weight"],
    )
    print(f"layer {args.layer}: MLA {geometry.num_heads} heads, "
          f"q_head_dim {geometry.q_head_dim}, kv_lora_rank {geometry.kv_lora_rank}, hidden {hidden}")

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
    x = torch.randn(POSITIONS, hidden, dtype=torch.float32) * 0.02

    # Probe: norm + real MLA attention + residual + norm, at every
    # position, to learn which experts EACH position selects — loading
    # only the union, same sparsity discipline as the MoE-only export.
    residual = x
    h = layer_ref.rms_norm(x, input_norm_w, eps)
    attn = mla_ref.mla_forward(h, mla_w, geometry)
    after_attention = residual + attn["output"]
    post_normed = layer_ref.rms_norm(after_attention, post_attn_norm_w, eps)
    ids_per_position = [
        moe_ref.route(post_normed[p], router, top_k, moe_renormalize, branch_scale)["ids"].tolist()
        for p in range(POSITIONS)
    ]
    union_ids = sorted({i for ids in ids_per_position for i in ids})
    print(f"  selected experts per position: {ids_per_position}")
    print(f"  union: {len(union_ids)} experts")

    expert_names = {
        f"{moe_prefix}experts.{i}.{suf}.weight" for i in union_ids for suf in ("w1", "w2", "w3")
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
        for i in union_ids
    }
    shared = moe_ref.ExpertWeights(
        w1=tensors[f"{moe_prefix}shared_experts.gate_proj.weight"],
        w2=tensors[f"{moe_prefix}shared_experts.down_proj.weight"],
        w3=tensors[f"{moe_prefix}shared_experts.up_proj.weight"],
    )

    result = layer_ref.mla_decoder_layer_forward(
        x, input_norm_w, post_attn_norm_w, eps, mla_w, geometry, router,
        experts, shared, top_k, moe_renormalize, branch_scale,
    )
    # Sanity: the probe above must agree with the real run's own routing
    # decision at every position, or this fixture is internally
    # inconsistent.
    for p in range(POSITIONS):
        assert result["moe"][p]["router"]["ids"].tolist() == ids_per_position[p], p

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": str(args.checkpoint),
        "layer": args.layer,
        "hidden": hidden,
        "rms_eps": eps,
        "num_heads": geometry.num_heads,
        "kv_lora_rank": geometry.kv_lora_rank,
        "qk_nope_head_dim": geometry.qk_nope_head_dim,
        "qk_rope_head_dim": geometry.qk_rope_head_dim,
        "v_head_dim": geometry.v_head_dim,
        "kv_a_norm_eps": mla_ref.KV_A_NORM_EPS,
        "moe_intermediate_size": cfg["moe_intermediate_size"],
        "num_shared_experts": cfg["num_shared_experts"],
        "experts": experts_n,
        "top_k": top_k,
        "moe_renormalize": moe_renormalize,
        "routed_scaling_factor": branch_scale,
        "positions": POSITIONS,
        "selected_ids_per_position": ids_per_position,
        "selected_ids_union_order": union_ids,
        "files": {},
    }

    def dump(name: str, t: torch.Tensor) -> None:
        manifest["files"][name] = write(args.out / f"{name}.f32", t)

    def dump_bf16(name: str, t: torch.Tensor) -> None:
        # P4a: expert weights are BF16 code units, never F32 — real
        # checkpoint tensors are already bf16-native (the f32 upcast
        # `read_tensors` did is lossless), so truncating back needs no
        # separate quantisation step.
        manifest["files"][name] = write_bf16(args.out / f"{name}.bf16", t)

    for p in range(POSITIONS):
        dump(f"input_{p}", x[p])
    dump("input_norm_weight", input_norm_w)
    dump("post_attention_norm_weight", post_attn_norm_w)
    dump("mla_q_proj", mla_w.q_proj)
    dump("mla_kv_a_proj", mla_w.kv_a_proj)
    dump("mla_kv_a_norm", mla_w.kv_a_norm)
    dump("mla_kv_b_proj", mla_w.kv_b_proj)
    dump("mla_o_proj", mla_w.o_proj)
    dump("router_weight", router.weight)
    dump("router_bias", router.e_score_correction_bias)
    for i in union_ids:
        e = experts[i]
        dump_bf16(f"expert{i}_w1", e.w1)
        dump_bf16(f"expert{i}_w2", e.w2)
        dump_bf16(f"expert{i}_w3", e.w3)
    dump_bf16("shared_w1", shared.w1)
    dump_bf16("shared_w2", shared.w2)
    dump_bf16("shared_w3", shared.w3)

    for p in range(POSITIONS):
        dump(f"out_input_normed_{p}", result["input_normed"][p])
        dump(f"out_attention_output_{p}", result["attention"]["output"][p])
        dump(f"out_after_attention_{p}", result["after_attention"][p])
        dump(f"out_post_attention_normed_{p}", result["post_attention_normed"][p])
        for idx, out in enumerate(result["moe"][p]["expert_outputs"]):
            dump(f"out_expert_output_{p}_{idx}", out)
        dump(f"out_routed_sum_{p}", result["moe"][p]["routed_sum"])
        dump(f"out_shared_output_{p}", result["moe"][p]["shared_output"])
        dump(f"out_layer_output_{p}", result["output"][p])

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(p.stat().st_size for p in args.out.glob("*.f32")) / 2**20 + sum(p.stat().st_size for p in args.out.glob("*.bf16")) / 2**20
    print(f"layer output absmax {result['output'].abs().max():.6f}")
    print(f"{total:.1f} MiB written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
