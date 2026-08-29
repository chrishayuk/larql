#!/usr/bin/env python3
"""Generate the COMMITTED `kimi_stack_oracle.json` fixture for
`exec::stack`'s own parity test.

Synthetic, tiny widths, REAL topology: the 27-layer kind sequence below
is copied from the checkpoint's own `config.json`
(`linear_attn_config.kda_layers`/`full_attn_layers`,
`first_k_dense_replace`), not invented — this is what makes the test
prove LAYER ORDERING, not just "27 layers of something".

    python scripts/kimi_stack_oracle_export.py > \
        crates/larql-vindex/src/format/vindex3/opplan/exec/tests/kimi_stack_oracle.json
"""

from __future__ import annotations

import json

import torch

import kda_reference as kda_ref
import kimi_mla_reference as mla_ref
import kimi_moe_reference as moe_ref
import kimi_stack_reference as stack_ref

SEED = 20260829
POSITIONS = 3
HIDDEN = 6

#: Real Kimi Linear topology (0-indexed), from `config.json`'s
#: `linear_attn_config` — not invented. `kda_layers` (1-indexed)
#: `[1..27] minus full_attn_layers`; `full_attn_layers` (1-indexed)
#: `[4, 8, 12, 16, 20, 24, 27]`. `first_k_dense_replace=1` excludes only
#: layer 0 from MoE routing.
MLA_LAYERS = {3, 7, 11, 15, 19, 23, 26}
NUM_LAYERS = 27
DENSE_LAYERS = {0}

KDA_HEADS, KDA_HEAD_DIM, KDA_CONV = 2, 3, 4
MLA_HEADS, MLA_LORA, MLA_NOPE, MLA_ROPE, MLA_V = 2, 5, 2, 3, 4
EXPERTS, TOP_K, INTER = 4, 2, 5
ROUTED_SCALING_FACTOR = 2.446
MOE_RENORMALIZE = True
EPS = 1e-5


def lst(t: torch.Tensor) -> list:
    return [round(v, 8) for v in t.detach().flatten().tolist()]


def rand(*shape: int, scale: float = 0.3) -> torch.Tensor:
    return torch.randn(*shape) * scale


def bf16_exact(t: torch.Tensor) -> torch.Tensor:
    """Round-trip through bf16 (PyTorch's own round-to-nearest-even), so
    the RETURNED f32 tensor already has zero in its lower 16 mantissa
    bits — a value REAL bf16 weights actually take, and the exact value
    `exec::kimi_moe_block`'s BF16 `matvec` will compute from (Rust
    truncates rather than re-rounds; see `stack_real.rs`'s own
    `read_bf16` doc comment for why that recovers the SAME bits, not an
    independent rounding, once the source is already bf16-exact)."""
    return t.detach().to(torch.bfloat16).to(torch.float32)


def kda_weights() -> kda_ref.KdaWeights:
    # P4c-4: q/k/v/o_proj are bf16-exact from construction — `KdaWeights`'s
    # four widest operands, the checkpoint's own representation, same
    # reasoning `expert_weights()` already applies below. Everything else
    # (conv, gates, recurrence, gated norm) stays plain f32.
    h, d, k = KDA_HEADS, KDA_HEAD_DIM, KDA_CONV
    width = h * d
    return kda_ref.KdaWeights(
        q_proj=bf16_exact(rand(width, HIDDEN)),
        k_proj=bf16_exact(rand(width, HIDDEN)),
        v_proj=bf16_exact(rand(width, HIDDEN)),
        q_conv1d=rand(width, 1, k), k_conv1d=rand(width, 1, k), v_conv1d=rand(width, 1, k),
        f_a_proj=rand(d, HIDDEN), f_b_proj=rand(width, d),
        g_a_proj=rand(d, HIDDEN), g_b_proj=rand(width, d),
        b_proj=rand(h, HIDDEN), a_log=rand(h) - 1.0, dt_bias=rand(width),
        o_norm=torch.rand(d) * 0.5 + 0.75, o_proj=bf16_exact(rand(HIDDEN, width)),
    )


def mla_weights() -> mla_ref.MlaWeights:
    g = mla_geometry()
    return mla_ref.MlaWeights(
        q_proj=rand(g.num_heads * g.q_head_dim, HIDDEN),
        kv_a_proj=rand(g.kv_lora_rank + g.qk_rope_head_dim, HIDDEN),
        kv_a_norm=torch.rand(g.kv_lora_rank) * 0.5 + 0.75,
        kv_b_proj=rand(g.num_heads * (g.qk_nope_head_dim + g.v_head_dim), g.kv_lora_rank),
        o_proj=rand(HIDDEN, g.num_heads * g.v_head_dim),
    )


def mla_geometry() -> mla_ref.MlaGeometry:
    return mla_ref.MlaGeometry(
        num_heads=MLA_HEADS, kv_lora_rank=MLA_LORA,
        qk_nope_head_dim=MLA_NOPE, qk_rope_head_dim=MLA_ROPE, v_head_dim=MLA_V,
    )


def expert_weights() -> moe_ref.ExpertWeights:
    # P4a: expert weights are bf16-exact from construction — the
    # checkpoint's own representation, never an f32 value this format
    # would have to round away later.
    return moe_ref.ExpertWeights(
        w1=bf16_exact(rand(INTER, HIDDEN)),
        w2=bf16_exact(rand(HIDDEN, INTER)),
        w3=bf16_exact(rand(INTER, HIDDEN)),
    )


def router_weights() -> moe_ref.RouterWeights:
    return moe_ref.RouterWeights(
        weight=rand(EXPERTS, HIDDEN), e_score_correction_bias=rand(EXPERTS, scale=0.1),
    )


def build_layer(i: int):
    norms = (torch.rand(HIDDEN) * 0.5 + 0.75, torch.rand(HIDDEN) * 0.5 + 0.75)
    if i in MLA_LAYERS:
        return stack_ref.MlaLayerSpec(
            attn_weights=mla_weights(), attn_geometry=mla_geometry(),
            input_norm=norms[0], post_norm=norms[1],
            router=router_weights(), experts={e: expert_weights() for e in range(EXPERTS)},
            shared=expert_weights(),
        )
    if i in DENSE_LAYERS:
        return stack_ref.KdaLayerSpec(
            dense=True, attn_weights=kda_weights(), input_norm=norms[0], post_norm=norms[1],
            ffn_dense=expert_weights(),
        )
    return stack_ref.KdaLayerSpec(
        dense=False, attn_weights=kda_weights(), input_norm=norms[0], post_norm=norms[1],
        router=router_weights(), experts={e: expert_weights() for e in range(EXPERTS)},
        shared=expert_weights(),
    )


def dump_kda(w: kda_ref.KdaWeights) -> dict:
    return {name: lst(getattr(w, name)) for name in kda_ref.KdaWeights.__dataclass_fields__}


def dump_mla(w: mla_ref.MlaWeights) -> dict:
    return {name: lst(getattr(w, name)) for name in mla_ref.MlaWeights.__dataclass_fields__}


def dump_expert(w: moe_ref.ExpertWeights) -> dict:
    return {"w1": lst(w.w1), "w2": lst(w.w2), "w3": lst(w.w3)}


def dump_router(r: moe_ref.RouterWeights) -> dict:
    return {"weight": lst(r.weight), "bias": lst(r.e_score_correction_bias)}


def dump_layer(i: int, spec) -> dict:
    out = {
        "kind": "mla" if i in MLA_LAYERS else "kda",
        "dense": i in DENSE_LAYERS,
        "input_norm": lst(spec.input_norm),
        "post_norm": lst(spec.post_norm),
    }
    if i in MLA_LAYERS:
        out["attn_weights"] = dump_mla(spec.attn_weights)
        out["router"] = dump_router(spec.router)
        out["experts"] = {str(e): dump_expert(w) for e, w in spec.experts.items()}
        out["shared"] = dump_expert(spec.shared)
    elif i in DENSE_LAYERS:
        out["attn_weights"] = dump_kda(spec.attn_weights)
        out["ffn_dense"] = dump_expert(spec.ffn_dense)
    else:
        out["attn_weights"] = dump_kda(spec.attn_weights)
        out["router"] = dump_router(spec.router)
        out["experts"] = {str(e): dump_expert(w) for e, w in spec.experts.items()}
        out["shared"] = dump_expert(spec.shared)
    return out


def main() -> int:
    torch.manual_seed(SEED)
    layers = [build_layer(i) for i in range(NUM_LAYERS)]
    x = torch.randn(POSITIONS, HIDDEN) * 0.2

    result = stack_ref.stack_forward(x, layers, EPS, TOP_K, MOE_RENORMALIZE, ROUTED_SCALING_FACTOR)

    fixture = {
        "hidden": HIDDEN,
        "rms_eps": EPS,
        "kda_num_heads": KDA_HEADS, "kda_head_dim": KDA_HEAD_DIM, "kda_conv_kernel": KDA_CONV,
        "mla_num_heads": MLA_HEADS, "mla_kv_lora_rank": MLA_LORA,
        "mla_qk_nope_head_dim": MLA_NOPE, "mla_qk_rope_head_dim": MLA_ROPE, "mla_v_head_dim": MLA_V,
        "mla_kv_a_norm_eps": mla_ref.KV_A_NORM_EPS,
        "experts": EXPERTS, "top_k": TOP_K, "inter": INTER,
        "moe_renormalize": MOE_RENORMALIZE, "routed_scaling_factor": ROUTED_SCALING_FACTOR,
        "num_layers": NUM_LAYERS,
        "mla_layers": sorted(MLA_LAYERS),
        "dense_layers": sorted(DENSE_LAYERS),
        "positions": POSITIONS,
        "input": [lst(x[p]) for p in range(POSITIONS)],
        "layers": [dump_layer(i, layers[i]) for i in range(NUM_LAYERS)],
        # Per (layer, position) boundaries — what `exec::stack`'s trace
        # names explicitly.
        "boundaries": {
            "input_residual": [
                [lst(result["layers"][i]["input_residual"][p]) for p in range(POSITIONS)]
                for i in range(NUM_LAYERS)
            ],
            "attention_output": [
                [lst(result["layers"][i]["attention_output"][p]) for p in range(POSITIONS)]
                for i in range(NUM_LAYERS)
            ],
            "post_attention_residual": [
                [lst(result["layers"][i]["post_attention_residual"][p]) for p in range(POSITIONS)]
                for i in range(NUM_LAYERS)
            ],
            "ffn_output": [
                [lst(result["layers"][i]["ffn_output"][p]) for p in range(POSITIONS)]
                for i in range(NUM_LAYERS)
            ],
            "layer_output": [
                [lst(result["layers"][i]["output"][p]) for p in range(POSITIONS)]
                for i in range(NUM_LAYERS)
            ],
        },
        "final_output": [lst(result["output"][p]) for p in range(POSITIONS)],
    }
    print(json.dumps(fixture))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
