#!/usr/bin/env python3
"""Torch transcription of the mixed Kimi Linear stack — the P3d-j parity
oracle for `exec::stack`.

Nothing new is transcribed here either: every layer's own attention
(`kda_reference.kda_forward`, `kimi_mla_reference.mla_forward`) and FFN
(`kimi_moe_reference.expert_forward`/`moe_block_forward`) are already
proven pieces. This composes them per `KimiDecoderLayer.forward`'s
residual/norm shape, once per layer, in layer order.

**Whole-sequence per layer, not depth-first per position** — the
opposite computational order `exec::stack::stack_forward` uses (one full
depth-first stack pass per position, state carried in Rust). Both
`kda_forward` and `mla_forward` are natively whole-sequence functions
(`[T, hidden]` in, `[T, hidden]` out, causal internally), so this
reference computes "layer 0 for all T positions, then layer 1 for all T
positions, ..." — mathematically identical to "all 27 layers for
position 0, then all 27 layers for position 1, ..." for any stack whose
layers are each internally causal (which is exactly what makes
autoregressive decode and teacher-forced whole-sequence scoring agree in
general). Comparing the two different orders is a STRONGER gate than
reproducing one order twice: it is impossible for both sides to share an
order-dependent bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import kda_reference as kda_ref
import kimi_mla_reference as mla_ref
import kimi_moe_reference as moe_ref
from kimi_mla_layer_reference import rms_norm


@dataclass
class KdaLayerSpec:
    dense: bool
    attn_weights: kda_ref.KdaWeights
    input_norm: torch.Tensor
    post_norm: torch.Tensor
    ffn_dense: moe_ref.ExpertWeights | None = None
    router: moe_ref.RouterWeights | None = None
    experts: dict | None = None
    shared: moe_ref.ExpertWeights | None = None


@dataclass
class MlaLayerSpec:
    attn_weights: mla_ref.MlaWeights
    attn_geometry: mla_ref.MlaGeometry
    input_norm: torch.Tensor
    post_norm: torch.Tensor
    router: moe_ref.RouterWeights
    experts: dict
    shared: moe_ref.ExpertWeights


def stack_layer_forward(
    x: torch.Tensor, spec, eps: float, top_k: int, moe_renormalize: bool, routed_scaling_factor: float,
) -> dict:
    """`x` is `[T, hidden]`. One layer, whichever kind `spec` names."""
    t = x.shape[0]
    residual = x
    h = rms_norm(x, spec.input_norm, eps)

    if isinstance(spec, KdaLayerSpec):
        kind = "kda"
        attn_output = kda_ref.kda_forward(h, spec.attn_weights)["output"]
    else:
        kind = "mla"
        attn_output = mla_ref.mla_forward(h, spec.attn_weights, spec.attn_geometry)["output"]
    after_attention = residual + attn_output

    post_normed = rms_norm(after_attention, spec.post_norm, eps)

    if isinstance(spec, KdaLayerSpec) and spec.dense:
        ffn_output = torch.stack([moe_ref.expert_forward(post_normed[p], spec.ffn_dense) for p in range(t)])
    else:
        moe_runs = [
            moe_ref.moe_block_forward(
                post_normed[p], spec.router, spec.experts, spec.shared,
                top_k, moe_renormalize, routed_scaling_factor,
            )
            for p in range(t)
        ]
        ffn_output = torch.stack([m["output"] for m in moe_runs])

    output = after_attention + ffn_output
    return {
        "kind": kind,
        "input_residual": x,
        "attention_output": attn_output,
        "post_attention_residual": after_attention,
        "ffn_output": ffn_output,
        "output": output,
    }


def stack_forward(
    x: torch.Tensor, layers: list, eps: float, top_k: int, moe_renormalize: bool, routed_scaling_factor: float,
) -> dict:
    """`x` is `[T, hidden]` — the initial hidden states, ALL layers deep.
    Returns every layer's own trace plus the final `[T, hidden]` output."""
    h = x
    traces = []
    for spec in layers:
        trace = stack_layer_forward(h, spec, eps, top_k, moe_renormalize, routed_scaling_factor)
        traces.append(trace)
        h = trace["output"]
    return {"layers": traces, "output": h}
