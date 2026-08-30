#!/usr/bin/env python3
"""Export a real-weight MULTI-LAYER chained parity fixture.

Rung 5e's fixture. The point is not two layers of weights — it is that
layer 2's input is layer 1's OUTPUT, so layer 2's routing decision is
dynamically downstream of layer 1. A device path that quietly re-used the
original host input, or read a stale buffer, selects different experts
and fails.

Each layer's experts are discovered by routing first and loaded second —
the same sparsity discipline `kimi_kda_layer_export.py` uses — and the
two layers' expert sets are exported separately because Kimi's experts
are per-layer tensors, not a shared bank.

    python scripts/kimi_two_layer_export.py <checkpoint> --layers 1 2 4 5 --out DIR
    LARQL_KIMI_TWO_LAYER_FIXTURE=DIR \
      cargo test -p larql-vindex --features gpu --release --lib kimi_two_layer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import kda_reference as kda_ref
import kimi_kda_layer_reference as layer_ref
import kimi_mla_layer_reference as mla_layer_ref
import kimi_mla_reference as mla_ref
import kimi_moe_reference as moe_ref
from kda_fixture import load_layer as load_kda_operands
from kimi_moe_export import read_tensors, write, write_bf16

INPUT_SEED = 20260828
#: `KdaWeights`'s four widest operands are BF16 code units on disk; the
#: rest of the dataclass stays f32.
KDA_BF16_FIELDS = {"q_proj", "k_proj", "v_proj", "o_proj"}


def load_layer_weights(checkpoint: Path, layer: int, kind: str, geom):
    """Every operand one layer needs except its experts, which depend on
    a route that does not exist yet."""
    prefix = f"model.layers.{layer}."
    moe_prefix = prefix + "block_sparse_moe."
    norms = read_tensors(
        checkpoint,
        {prefix + "input_layernorm.weight", prefix + "post_attention_layernorm.weight"},
    )
    router_tensors = read_tensors(
        checkpoint,
        {moe_prefix + "gate.weight", moe_prefix + "gate.e_score_correction_bias"},
    )
    if kind == "kda":
        attention = {"kind": "kda", "w": load_kda_operands(checkpoint, layer)}
    else:
        a = prefix + "self_attn."
        t = read_tensors(checkpoint, {
            a + "q_proj.weight",
            a + "kv_a_proj_with_mqa.weight",
            a + "kv_a_layernorm.weight",
            a + "kv_b_proj.weight",
            a + "o_proj.weight",
        })
        attention = {
            "kind": "mla",
            "w": mla_ref.MlaWeights(
                q_proj=t[a + "q_proj.weight"],
                kv_a_proj=t[a + "kv_a_proj_with_mqa.weight"],
                kv_a_norm=t[a + "kv_a_layernorm.weight"],
                kv_b_proj=t[a + "kv_b_proj.weight"],
                o_proj=t[a + "o_proj.weight"],
            ),
            "geom": geom,
        }
    return {
        "attention": attention,
        "input_norm": norms[prefix + "input_layernorm.weight"],
        "post_norm": norms[prefix + "post_attention_layernorm.weight"],
        "router": moe_ref.RouterWeights(
            weight=router_tensors[moe_prefix + "gate.weight"],
            e_score_correction_bias=router_tensors[moe_prefix + "gate.e_score_correction_bias"],
        ),
        "prefix": moe_prefix,
    }


def attention_out(w, h):
    """One layer's attention output for a `[1, hidden]` normed input."""
    a = w["attention"]
    if a["kind"] == "kda":
        return kda_ref.kda_forward(h, a["w"])["output"][0]
    return mla_ref.mla_forward(h, a["w"], a["geom"])["output"][0]


def route_for(w, x, eps, top_k, renorm, scale):
    """Which experts this layer selects on `x`, without loading any."""
    h = layer_ref.rms_norm(x, w["input_norm"], eps)
    after = x + attention_out(w, h.unsqueeze(0))
    post = layer_ref.rms_norm(after, w["post_norm"], eps)
    return moe_ref.route(post, w["router"], top_k, renorm, scale)["ids"].tolist()


def load_experts(checkpoint: Path, moe_prefix: str, ids):
    names = {f"{moe_prefix}experts.{i}.{s}.weight" for i in ids for s in ("w1", "w2", "w3")}
    names |= {f"{moe_prefix}shared_experts.{s}.weight" for s in ("gate_proj", "up_proj", "down_proj")}
    t = read_tensors(checkpoint, names)
    experts = {
        i: moe_ref.ExpertWeights(
            w1=t[f"{moe_prefix}experts.{i}.w1.weight"],
            w2=t[f"{moe_prefix}experts.{i}.w2.weight"],
            w3=t[f"{moe_prefix}experts.{i}.w3.weight"],
        )
        for i in ids
    }
    shared = moe_ref.ExpertWeights(
        w1=t[f"{moe_prefix}shared_experts.gate_proj.weight"],
        w2=t[f"{moe_prefix}shared_experts.down_proj.weight"],
        w3=t[f"{moe_prefix}shared_experts.up_proj.weight"],
    )
    return experts, shared


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads((args.checkpoint / "config.json").read_text())
    hidden = cfg["hidden_size"]
    eps = cfg["rms_norm_eps"]
    top_k = cfg["num_experts_per_token"]
    renorm = cfg["moe_renormalize"]
    scale = cfg["routed_scaling_factor"]
    kda_layers = {i - 1 for i in cfg["linear_attn_config"]["kda_layers"]}
    dense_below = cfg["first_k_dense_replace"]
    geom = mla_ref.MlaGeometry(
        num_heads=cfg["num_attention_heads"],
        kv_lora_rank=cfg["kv_lora_rank"],
        qk_nope_head_dim=cfg["qk_nope_head_dim"],
        qk_rope_head_dim=cfg["qk_rope_head_dim"],
        v_head_dim=cfg["v_head_dim"],
    )
    kinds = ["kda" if i in kda_layers else "mla" for i in args.layers]
    for layer in args.layers:
        assert layer >= dense_below, f"layer {layer} is dense, not routed"

    torch.manual_seed(INPUT_SEED)
    x0 = torch.randn(hidden, dtype=torch.float32) * 0.02

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": str(args.checkpoint),
        "layers": args.layers,
        "hidden": hidden,
        "rms_eps": eps,
        "moe_intermediate_size": cfg["moe_intermediate_size"],
        "experts": cfg["num_experts"],
        "top_k": top_k,
        "moe_renormalize": renorm,
        "routed_scaling_factor": scale,
        "selected_ids_order": {},
        "files": {},
    }

    def dump(name, t):
        manifest["files"][name] = write(args.out / f"{name}.f32", t)

    def dump_bf16(name, t):
        manifest["files"][name] = write_bf16(args.out / f"{name}.bf16", t)

    dump("input", x0)
    x = x0
    manifest["kinds"] = kinds
    manifest["mla"] = {
        "num_heads": geom.num_heads,
        "kv_lora_rank": geom.kv_lora_rank,
        "qk_nope_head_dim": geom.qk_nope_head_dim,
        "qk_rope_head_dim": geom.qk_rope_head_dim,
        "v_head_dim": geom.v_head_dim,
        "kv_a_norm_eps": cfg.get("rms_norm_eps", eps),
    }
    for pos, (layer, kind) in enumerate(zip(args.layers, kinds)):
        w = load_layer_weights(args.checkpoint, layer, kind, geom)
        if kind == "kda" and "num_heads" not in manifest:
            manifest["num_heads"] = w["attention"]["w"].num_heads
            manifest["head_dim"] = w["attention"]["w"].head_dim
        ids = route_for(w, x, eps, top_k, renorm, scale)
        print(f"layer {layer} (position {pos}): selects {sorted(ids)}")
        experts, shared = load_experts(args.checkpoint, w["prefix"], ids)
        if kind == "kda":
            result = layer_ref.kda_decoder_layer_forward(
                x, w["input_norm"], w["post_norm"], eps, w["attention"]["w"], w["router"],
                experts, shared, top_k, renorm, scale,
            )
        else:
            result = mla_layer_ref.mla_decoder_layer_forward(
                x.unsqueeze(0), w["input_norm"], w["post_norm"], eps,
                w["attention"]["w"], geom, w["router"],
                experts, shared, top_k, renorm, scale,
            )
        # The MLA layer reference is sequence-shaped (it runs `[T,
        # hidden]`), so normalise both kinds to one position's boundaries
        # before anything reads them.
        if kind != "kda":
            result = {
                "input_normed": result["input_normed"][0],
                "attention": {"output": result["attention"]["output"][0]},
                "after_attention": result["after_attention"][0],
                "post_attention_normed": result["post_attention_normed"][0],
                "moe": result["moe"][0],
                "output": result["output"][0],
            }
        # The probe and the real run must agree, or the fixture is
        # internally inconsistent.
        assert result["moe"]["router"]["ids"].tolist() == ids

        p = f"l{pos}_"
        manifest["selected_ids_order"][str(pos)] = ids
        dump(p + "input_norm_weight", w["input_norm"])
        dump(p + "post_attention_norm_weight", w["post_norm"])
        if kind == "kda":
            for name in kda_ref.KdaWeights.__dataclass_fields__:
                t = getattr(w["attention"]["w"], name)
                (dump_bf16 if name in KDA_BF16_FIELDS else dump)(f"{p}kda_{name}", t)
        else:
            mw = w["attention"]["w"]
            for name, t in [
                ("q_proj", mw.q_proj), ("kv_a_proj", mw.kv_a_proj),
                ("kv_b_proj", mw.kv_b_proj), ("o_proj", mw.o_proj),
            ]:
                dump_bf16(f"{p}mla_{name}", t)
            dump(p + "mla_kv_a_norm", mw.kv_a_norm)
        dump(p + "router_weight", w["router"].weight)
        dump(p + "router_bias", w["router"].e_score_correction_bias)
        for i in ids:
            e = experts[i]
            dump_bf16(f"{p}expert{i}_w1", e.w1)
            dump_bf16(f"{p}expert{i}_w2", e.w2)
            dump_bf16(f"{p}expert{i}_w3", e.w3)
        dump_bf16(p + "shared_w1", shared.w1)
        dump_bf16(p + "shared_w2", shared.w2)
        dump_bf16(p + "shared_w3", shared.w3)

        dump(p + "out_input_normed", result["input_normed"])
        dump(p + "out_attention_output", result["attention"]["output"]
             if kind != "kda" else result["attention"]["output"][0])
        dump(p + "out_after_attention", result["after_attention"])
        dump(p + "out_post_attention_normed", result["post_attention_normed"])
        dump(p + "out_routed_sum", result["moe"]["routed_sum"])
        dump(p + "out_shared_output", result["moe"]["shared_output"])
        dump(p + "out_layer_output", result["output"])
        x = result["output"]

    # The proof the chain is real: consecutive layers route differently,
    # so a device path that fed a layer the wrong hidden state cannot
    # pass by accident.
    routes = [manifest["selected_ids_order"][str(i)] for i in range(len(args.layers))]
    manifest["routes_differ"] = all(
        sorted(a) != sorted(b) for a, b in zip(routes, routes[1:])
    )
    overlaps = [len(set(a) & set(b)) for a, b in zip(routes, routes[1:])]
    print(f"consecutive routes all differ: {manifest['routes_differ']}; overlaps {overlaps}")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(p.stat().st_size for p in args.out.iterdir()) / 2**20
    print(f"final output absmax {x.abs().max():.6f}")
    print(f"{total:.1f} MiB written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
