#!/usr/bin/env python3
"""Torch transcription of one complete Kimi Linear MLA decoder layer —
the P3d-i parity oracle for `kimi_mla_layer.rs`.

Composes two ALREADY-TRANSCRIBED pieces (`kimi_mla_reference.mla_forward`,
`kimi_moe_reference.moe_block_forward`) with the SAME residual/norm
structure `kimi_kda_layer_reference.py` already proved for the KDA
family — `KimiDecoderLayer.forward` does not change shape between
attention families, only which `self_attn` runs. Nothing new is
transcribed here beyond that composition.

Unlike the KDA layer (one token in, one token out), MLA's attention is
naturally sequence-shaped: `mla_forward` runs the whole `[T, hidden]`
input at once. Routing is still PER TOKEN — `KimiSparseMoeBlock` has no
notion of a batch dimension sharing one decision — so this composes the
batched attention half with a per-position loop over the MoE half.
"""

from __future__ import annotations

import torch

import kimi_mla_reference as mla_ref
import kimi_moe_reference as moe_ref


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """`KimiRMSNorm.forward` — f32 throughout, no weight offset. Row-wise,
    so it applies identically whether `x` is one token or a sequence."""
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    return weight.float() * (x * torch.rsqrt(variance + eps))


def mla_decoder_layer_forward(
    x: torch.Tensor,
    input_norm_weight: torch.Tensor,
    post_attention_norm_weight: torch.Tensor,
    norm_eps: float,
    mla_weights: mla_ref.MlaWeights,
    mla_geometry: mla_ref.MlaGeometry,
    router: moe_ref.RouterWeights,
    experts: dict[int, moe_ref.ExpertWeights],
    shared: moe_ref.ExpertWeights,
    top_k: int,
    moe_renormalize: bool,
    routed_scaling_factor: float,
) -> dict:
    """`x` is `[T, hidden]`. `experts` must already contain every id ANY
    position selects — the caller's job (see `kimi_mla_layer_export.py`'s
    two-pass routing probe), not this function's. Returns every named
    boundary, per position where the boundary is itself per-position."""
    t = x.shape[0]
    residual = x
    h = rms_norm(x, input_norm_weight, norm_eps)

    attn = mla_ref.mla_forward(h, mla_weights, mla_geometry)  # [T, hidden] in, out
    after_attention = residual + attn["output"]

    post_normed = rms_norm(after_attention, post_attention_norm_weight, norm_eps)

    moe_per_position = [
        moe_ref.moe_block_forward(
            post_normed[p], router, experts, shared, top_k, moe_renormalize, routed_scaling_factor,
        )
        for p in range(t)
    ]
    moe_output = torch.stack([m["output"] for m in moe_per_position])
    output = after_attention + moe_output

    return {
        "input_normed": h,
        "attention": attn,
        "after_attention": after_attention,
        "post_attention_normed": post_normed,
        "moe": moe_per_position,
        "output": output,
    }
