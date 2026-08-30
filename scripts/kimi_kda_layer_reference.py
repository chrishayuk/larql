#!/usr/bin/env python3
"""Torch transcription of one complete Kimi Linear KDA decoder layer — the
P3d-h parity oracle for `kimi_kda_layer.rs`.

Composes two ALREADY-TRANSCRIBED pieces (`kda_reference.kda_forward`,
`kimi_moe_reference.moe_block_forward`) with the residual/norm structure
`KimiDecoderLayer.forward` states directly in the checkpoint's own
`modeling_kimi.py`:

    residual = x
    h = input_layernorm(x)
    h = self_attn(h)                # kda_reference.kda_forward
    h = residual + h
    residual = h
    h = post_attention_layernorm(h)
    h = block_sparse_moe(h)         # kimi_moe_reference.moe_block_forward
    h = residual + h
    return h

Nothing new is transcribed here except `KimiRMSNorm` itself (`x *
rsqrt(mean(x^2) + eps) * weight`, no offset — the same formula
`kda_reference.py`'s own `o_norm`-adjacent statistic uses, just without
the sigmoid gate).
"""

from __future__ import annotations

import torch

import kda_reference as kda_ref
import kimi_moe_reference as moe_ref


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """`KimiRMSNorm.forward` — f32 throughout, no weight offset."""
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    return weight.float() * (x * torch.rsqrt(variance + eps))


def kda_decoder_layer_forward(
    x: torch.Tensor,
    input_norm_weight: torch.Tensor,
    post_attention_norm_weight: torch.Tensor,
    norm_eps: float,
    kda_weights: kda_ref.KdaWeights,
    router: moe_ref.RouterWeights,
    experts: dict[int, moe_ref.ExpertWeights],
    shared: moe_ref.ExpertWeights,
    top_k: int,
    moe_renormalize: bool,
    routed_scaling_factor: float,
) -> dict:
    """`x` is `[hidden]` (one token). Returns every named boundary."""
    residual = x
    h = rms_norm(x, input_norm_weight, norm_eps)

    attn = kda_ref.kda_forward(h.unsqueeze(0), kda_weights)  # [1, hidden] in
    attn_output = attn["output"][0]  # back to [hidden]
    after_attention = residual + attn_output

    post_normed = rms_norm(after_attention, post_attention_norm_weight, norm_eps)
    block = moe_ref.moe_block_forward(
        post_normed, router, experts, shared, top_k, moe_renormalize, routed_scaling_factor,
    )
    output = after_attention + block["output"]

    return {
        "input_normed": h,
        "attention": attn,
        "after_attention": after_attention,
        "post_attention_normed": post_normed,
        "moe": block,
        "output": output,
    }
