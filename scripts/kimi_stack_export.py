#!/usr/bin/env python3
"""Export a real-weight, complete 27-layer Kimi Linear stack parity
fixture — three real positions through every layer.

Layer by layer, in order, REAL forward: each layer's own attention runs
on the REAL hidden state its predecessor actually produced (not a probe
input), routes on that real state, loads only the union of experts its
three positions select, computes the real FFN output, and the result
becomes the next layer's input. This is the only correct way to build
this fixture — a layer's routing decision depends on the true upstream
hidden state, so there is no cross-layer analogue of the single-layer
"probe first, load second" trick; sparsity is still honoured PER LAYER.

Layer 0 is dense (`first_k_dense_replace=1`): `KimiMLP` with
`intermediate_size=config.intermediate_size` (9216 here — NOT
`moe_intermediate_size`, a genuinely different, wider value) and
`gate_proj`/`up_proj`/`down_proj` weight names — NOT the routed experts'
`w1`/`w2`/`w3`. Read from the checkpoint's own `modeling_kimi.py`
`KimiSparseMoeBlock.__init__`/`KimiDecoderLayer.__init__`, not assumed
identical to the routed-expert shape.

    python scripts/kimi_stack_export.py ~/chris-models/Kimi-Linear-48B-A3B-Instruct --out DIR
    LARQL_KIMI_STACK_FIXTURE=DIR cargo test -p larql-vindex --lib stack_real
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import kda_reference as kda_ref
import kimi_mla_reference as mla_ref
import kimi_moe_reference as moe_ref
from kda_fixture import load_layer as load_kda_operands
from kimi_mla_layer_reference import rms_norm
from kimi_moe_export import read_tensors, write, write_bf16

INPUT_SEED = 20260830
POSITIONS = 3

#: `KdaWeights`'s four widest operands (P4c-4) — dumped bf16 rather than
#: f32, same reasoning `kimi_kda_layer_export.py` already applies.
KDA_BF16_FIELDS = {"q_proj", "k_proj", "v_proj", "o_proj"}


def layer_geometry_from_config(cfg: dict) -> tuple:
    """The facts every caller of [`run_layers`] needs from `config.json`,
    read once so `kimi_stack_export.py`'s own `main()` and
    `kimi_token_export.py` (which prepends embedding lookup and appends
    the final norm + `lm_head`) read them identically."""
    num_layers = cfg["num_hidden_layers"]
    kda_layers = {i - 1 for i in cfg["linear_attn_config"]["kda_layers"]}
    mla_layers = {i - 1 for i in cfg["linear_attn_config"]["full_attn_layers"]}
    dense_layers = set(range(cfg["first_k_dense_replace"]))
    assert dense_layers == {0}, "this script assumes exactly layer 0 is dense"
    assert kda_layers | mla_layers == set(range(num_layers))
    assert not (kda_layers & mla_layers)
    mla_geometry = mla_ref.MlaGeometry(
        num_heads=cfg["num_attention_heads"], kv_lora_rank=cfg["kv_lora_rank"],
        qk_nope_head_dim=cfg["qk_nope_head_dim"], qk_rope_head_dim=cfg["qk_rope_head_dim"],
        v_head_dim=cfg["v_head_dim"],
    )
    return num_layers, kda_layers, mla_layers, dense_layers, mla_geometry


def run_layers(
    checkpoint: Path, num_layers: int, kda_layers: set, mla_layers: set, dense_layers: set,
    mla_geometry, eps: float, top_k: int, moe_renormalize: bool, branch_scale: float,
    positions: int, h: torch.Tensor, dump, dump_bf16, manifest: dict,
) -> tuple:
    """Run `h` (`[positions, hidden]`) through every real layer, in
    order, dumping every weight and boundary via `dump(name, tensor)` and
    appending each layer's own manifest entry to `manifest["layers"]`.
    Expert weights (routed, shared, and layer 0's dense `KimiMLP`) go
    through `dump_bf16(name, tensor)` instead — P4a: `exec::kimi_moe_
    block::ExpertWeights` is BF16 code units, never F32.

    SEQUENTIAL, layer by layer: a layer's routing decision depends on the
    TRUE hidden state its predecessor produced, so there is no
    cross-layer analogue of the single-layer "probe first, load second"
    trick — each layer is computed for real before the next layer's
    routing is even decided, and only that layer's own union-of-
    `positions` experts are loaded.

    Returns `(h, total_expert_loads)` — the final `[positions, hidden]`
    output and the sum of every MoE layer's union-of-experts count, for
    the caller's own summary print.
    """
    total_expert_loads = 0
    for i in range(num_layers):
        prefix = f"model.layers.{i}."
        norms = read_tensors(
            checkpoint,
            {prefix + "input_layernorm.weight", prefix + "post_attention_layernorm.weight"},
        )
        input_norm_w = norms[prefix + "input_layernorm.weight"]
        post_norm_w = norms[prefix + "post_attention_layernorm.weight"]
        dump(f"layer{i}_input_norm_weight", input_norm_w)
        dump(f"layer{i}_post_norm_weight", post_norm_w)

        residual = h
        h_normed = rms_norm(h, input_norm_w, eps)

        if i in kda_layers:
            kind = "kda"
            kda_w = load_kda_operands(checkpoint, i)
            if manifest["kda_num_heads"] is None:
                manifest["kda_num_heads"] = kda_w.num_heads
                manifest["kda_head_dim"] = kda_w.head_dim
            attn_output = kda_ref.kda_forward(h_normed, kda_w)["output"]
            for name in kda_ref.KdaWeights.__dataclass_fields__:
                t = getattr(kda_w, name)
                # `KdaWeights`'s four widest operands (P4c-4) — dumped
                # bf16, like the expert/dense-FFN weights below; everything
                # else in the dataclass stays f32.
                (dump_bf16 if name in KDA_BF16_FIELDS else dump)(f"layer{i}_kda_{name}", t)
        else:
            kind = "mla"
            attn_prefix = prefix + "self_attn."
            mla_tensors = read_tensors(checkpoint, {
                attn_prefix + "q_proj.weight", attn_prefix + "kv_a_proj_with_mqa.weight",
                attn_prefix + "kv_a_layernorm.weight", attn_prefix + "kv_b_proj.weight",
                attn_prefix + "o_proj.weight",
            })
            mla_w = mla_ref.MlaWeights(
                q_proj=mla_tensors[attn_prefix + "q_proj.weight"],
                kv_a_proj=mla_tensors[attn_prefix + "kv_a_proj_with_mqa.weight"],
                kv_a_norm=mla_tensors[attn_prefix + "kv_a_layernorm.weight"],
                kv_b_proj=mla_tensors[attn_prefix + "kv_b_proj.weight"],
                o_proj=mla_tensors[attn_prefix + "o_proj.weight"],
            )
            attn_output = mla_ref.mla_forward(h_normed, mla_w, mla_geometry)["output"]
            for name in mla_ref.MlaWeights.__dataclass_fields__:
                dump(f"layer{i}_mla_{name}", getattr(mla_w, name))

        after_attention = residual + attn_output
        post_normed = rms_norm(after_attention, post_norm_w, eps)

        layer_manifest = {"kind": kind, "dense": i in dense_layers}

        if i in dense_layers:
            dense_tensors = read_tensors(checkpoint, {
                prefix + "mlp.gate_proj.weight", prefix + "mlp.up_proj.weight", prefix + "mlp.down_proj.weight",
            })
            dense_w = moe_ref.ExpertWeights(
                w1=dense_tensors[prefix + "mlp.gate_proj.weight"],
                w2=dense_tensors[prefix + "mlp.down_proj.weight"],
                w3=dense_tensors[prefix + "mlp.up_proj.weight"],
            )
            dump_bf16(f"layer{i}_dense_w1", dense_w.w1)
            dump_bf16(f"layer{i}_dense_w2", dense_w.w2)
            dump_bf16(f"layer{i}_dense_w3", dense_w.w3)
            ffn_output = torch.stack([
                moe_ref.expert_forward(post_normed[p], dense_w) for p in range(positions)
            ])
        else:
            moe_prefix = prefix + "block_sparse_moe."
            router_tensors = read_tensors(checkpoint, {
                moe_prefix + "gate.weight", moe_prefix + "gate.e_score_correction_bias",
            })
            router = moe_ref.RouterWeights(
                weight=router_tensors[moe_prefix + "gate.weight"],
                e_score_correction_bias=router_tensors[moe_prefix + "gate.e_score_correction_bias"],
            )
            dump(f"layer{i}_router_weight", router.weight)
            dump(f"layer{i}_router_bias", router.e_score_correction_bias)

            ids_per_position = [
                moe_ref.route(post_normed[p], router, top_k, moe_renormalize, branch_scale)["ids"].tolist()
                for p in range(positions)
            ]
            union_ids = sorted({e for ids in ids_per_position for e in ids})
            total_expert_loads += len(union_ids)

            expert_names = {
                f"{moe_prefix}experts.{e}.{suf}.weight" for e in union_ids for suf in ("w1", "w2", "w3")
            }
            shared_names = {
                f"{moe_prefix}shared_experts.{suf}.weight" for suf in ("gate_proj", "up_proj", "down_proj")
            }
            tensors = read_tensors(checkpoint, expert_names | shared_names)
            experts = {
                e: moe_ref.ExpertWeights(
                    w1=tensors[f"{moe_prefix}experts.{e}.w1.weight"],
                    w2=tensors[f"{moe_prefix}experts.{e}.w2.weight"],
                    w3=tensors[f"{moe_prefix}experts.{e}.w3.weight"],
                )
                for e in union_ids
            }
            shared = moe_ref.ExpertWeights(
                w1=tensors[f"{moe_prefix}shared_experts.gate_proj.weight"],
                w2=tensors[f"{moe_prefix}shared_experts.down_proj.weight"],
                w3=tensors[f"{moe_prefix}shared_experts.up_proj.weight"],
            )
            for e in union_ids:
                dump_bf16(f"layer{i}_expert{e}_w1", experts[e].w1)
                dump_bf16(f"layer{i}_expert{e}_w2", experts[e].w2)
                dump_bf16(f"layer{i}_expert{e}_w3", experts[e].w3)
            dump_bf16(f"layer{i}_shared_w1", shared.w1)
            dump_bf16(f"layer{i}_shared_w2", shared.w2)
            dump_bf16(f"layer{i}_shared_w3", shared.w3)

            moe_runs = [
                moe_ref.moe_block_forward(
                    post_normed[p], router, experts, shared, top_k, moe_renormalize, branch_scale,
                )
                for p in range(positions)
            ]
            # Sanity: the routing probe above must agree with the real run.
            for p in range(positions):
                assert moe_runs[p]["router"]["ids"].tolist() == ids_per_position[p], (i, p)
            ffn_output = torch.stack([m["output"] for m in moe_runs])

            layer_manifest["selected_ids_per_position"] = ids_per_position
            layer_manifest["selected_ids_union_order"] = union_ids

        layer_output = after_attention + ffn_output

        for p in range(positions):
            dump(f"layer{i}_out_input_residual_{p}", residual[p])
            dump(f"layer{i}_out_attention_output_{p}", attn_output[p])
            dump(f"layer{i}_out_post_attention_residual_{p}", after_attention[p])
            dump(f"layer{i}_out_ffn_output_{p}", ffn_output[p])
            dump(f"layer{i}_out_layer_output_{p}", layer_output[p])

        manifest["layers"].append(layer_manifest)
        print(f"  layer {i:>2} [{kind}{'/dense' if i in dense_layers else ''}]: "
              f"output absmax {layer_output.abs().max():.6f}")

        h = layer_output

    return h, total_expert_loads


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads((args.checkpoint / "config.json").read_text())
    hidden = cfg["hidden_size"]
    eps = cfg["rms_norm_eps"]
    experts_n = cfg["num_experts"]
    top_k = cfg["num_experts_per_token"]
    moe_renormalize = cfg["moe_renormalize"]
    branch_scale = cfg["routed_scaling_factor"]
    dense_intermediate = cfg["intermediate_size"]
    moe_intermediate = cfg["moe_intermediate_size"]
    num_shared = cfg["num_shared_experts"]
    num_layers, kda_layers, mla_layers, dense_layers, mla_geometry = layer_geometry_from_config(cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": str(args.checkpoint),
        "hidden": hidden, "rms_eps": eps,
        "kda_num_heads": None, "kda_head_dim": None, "kda_conv_kernel": cfg["linear_attn_config"]["short_conv_kernel_size"],
        "mla_num_heads": mla_geometry.num_heads, "mla_kv_lora_rank": mla_geometry.kv_lora_rank,
        "mla_qk_nope_head_dim": mla_geometry.qk_nope_head_dim, "mla_qk_rope_head_dim": mla_geometry.qk_rope_head_dim,
        "mla_v_head_dim": mla_geometry.v_head_dim, "mla_kv_a_norm_eps": mla_ref.KV_A_NORM_EPS,
        "experts": experts_n, "top_k": top_k, "moe_intermediate_size": moe_intermediate,
        "num_shared_experts": num_shared, "dense_intermediate_size": dense_intermediate,
        "moe_renormalize": moe_renormalize, "routed_scaling_factor": branch_scale,
        "num_layers": num_layers, "mla_layers": sorted(mla_layers), "dense_layers": sorted(dense_layers),
        "positions": POSITIONS, "files": {}, "layers": [],
    }

    def dump(name: str, t: torch.Tensor) -> None:
        manifest["files"][name] = write(args.out / f"{name}.f32", t)

    def dump_bf16(name: str, t: torch.Tensor) -> None:
        manifest["files"][name] = write_bf16(args.out / f"{name}.bf16", t)

    torch.manual_seed(INPUT_SEED)
    h = torch.randn(POSITIONS, hidden, dtype=torch.float32) * 0.02
    for p in range(POSITIONS):
        dump(f"input_{p}", h[p])

    h, total_expert_loads = run_layers(
        args.checkpoint, num_layers, kda_layers, mla_layers, dense_layers, mla_geometry,
        eps, top_k, moe_renormalize, branch_scale, POSITIONS, h, dump, dump_bf16, manifest,
    )

    for p in range(POSITIONS):
        dump(f"final_output_{p}", h[p])

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(p.stat().st_size for p in args.out.glob("*.f32")) / 2**20 + sum(p.stat().st_size for p in args.out.glob("*.bf16")) / 2**20
    print(f"\n{total_expert_loads} total (layer, expert) loads across {num_layers} layers")
    print(f"final stack output absmax {h.abs().max():.6f}")
    print(f"{total:.1f} MiB written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
