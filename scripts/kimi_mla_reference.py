#!/usr/bin/env python3
"""Torch transcription of `KimiMLAAttention.forward`, generalised over
sequence length and exposing every internal boundary per position.

Pinned to the checkpoint's own `modeling_kimi.py` (sha256
`d79b365e3737…`, the same file `kda_reference.py` pins) rather than
imported: the module's top-level `try/except ImportError: raise` on `fla`
re-raises unconditionally on this Mac (Triton/CUDA), the same wall the
KDA oracle already hit.

The one fact this file exists to make executable rather than merely
documented: `mla_use_nope=True` is asserted in `__init__`, and `forward`
never calls a rotary embedding on `q_rope`/`k_rot` at all — "RoPE" in
`qk_rope_head_dim` is DeepSeek's inherited field NAME for the shape, not
a claim this family rotates it. `kv_rot` here is exactly the SPLIT
component, untouched.

Also load-bearing and easy to get wrong by analogy with the layer's own
two norms: `kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank)` in
`__init__` passes no `eps`, so it uses `KimiRMSNorm`'s class DEFAULT
(`1e-6`) — NOT `config.rms_norm_eps` (`1e-5`), what `input_layernorm`/
`post_attention_layernorm` use. `KV_A_NORM_EPS` names that fact so it
cannot be silently assumed equal to the layer eps.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: `KimiRMSNorm`'s class default (`eps=1e-6`), what `kv_a_layernorm`
#: actually runs at — its constructor passes no override.
KV_A_NORM_EPS = 1e-6

#: Boundaries this transcription exposes per position, in execution
#: order. Named here so a fixture and a report cannot drift on what one
#: means — same convention `kda_reference.py`'s own `BOUNDARIES` sets.
BOUNDARIES = (
    "q_proj", "compressed_kv", "kv_a_normed", "kv_b",
    "attn_weights", "attn_value", "output",
)

#: The same ladder with Kimi-K3's OUTPUT GATE in it
#: (`mla_use_output_gate: true`, `modeling_kimi_linear.py` L398-401,
#: L470-472): `output_gate = sigmoid(g_proj(x))` and
#: `gated_value = attn_value * output_gate`, both between the aggregation
#: and `o_proj`. Kimi Linear's ungated ladder above is unchanged.
GATED_BOUNDARIES = BOUNDARIES[:-1] + ("output_gate", "gated_value", "output")

#: Named defects of the output gate, for K3-REP-GATE-1's controls — each
#: perturbs the real forward at one point, never a copy of it.
GATE_MUTATIONS = (
    "none",
    # `gated_value := attn_value`: the gate is not applied.
    "gate_omitted",
    # the raw pre-activation multiplied in, no sigmoid.
    "sigmoid_omitted",
    # every cached position's `v` gated by ITS OWN position's gate before
    # the weighted sum, and nothing gated after — a placement defect: the
    # reference gates the AGGREGATE by the query position's gate.
    "gate_on_values_before_aggregation",
)


@dataclass
class MlaWeights:
    """One layer's five operands, already in f32. `num_heads` cannot be
    read back out of these shapes alone (`q_proj` rows are
    `Hq*q_head_dim`, `kv_b_proj` rows are `Hq*(nope+v_head_dim)` — either
    needs nope/rope/v_head_dim to invert), so every caller states it
    explicitly via `MlaGeometry` instead."""
    q_proj: torch.Tensor       # [Hq*q_head_dim, hidden]
    kv_a_proj: torch.Tensor    # [kv_lora_rank+rope, hidden]
    kv_a_norm: torch.Tensor    # [kv_lora_rank]
    kv_b_proj: torch.Tensor    # [Hq*(nope+v_head_dim), kv_lora_rank]
    o_proj: torch.Tensor       # [hidden, Hq*v_head_dim]


@dataclass
class MlaGeometry:
    num_heads: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

    @property
    def q_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    return weight.float() * (x * torch.rsqrt(variance + eps))


def mla_forward(
    x: torch.Tensor,
    w: MlaWeights,
    g: MlaGeometry,
    *,
    g_proj: torch.Tensor | None = None,
    mutation: str = "none",
) -> dict:
    """`x` is `[T, hidden]`. Returns every boundary, per position, plus
    the final `[T, hidden]` output — causal, `mla_use_nope` (no rotation
    anywhere), exactly `KimiMLAAttention.forward` at batch size 1.

    `g_proj` (`[Hq*v_head_dim, hidden]`) adds Kimi-K3's output gate
    (`GATED_BOUNDARIES`); `None` is Kimi Linear's ungated block and
    returns exactly `BOUNDARIES`. `mutation` names one of
    `GATE_MUTATIONS` and is only meaningful with a gate.
    """
    if mutation not in GATE_MUTATIONS:
        raise ValueError(f"unknown gate mutation {mutation!r}")
    if g_proj is None and mutation != "none":
        raise ValueError("gate mutations need a gate")
    t, hidden = x.shape
    h, nope, rope, v_hd, lora = (
        g.num_heads, g.qk_nope_head_dim, g.qk_rope_head_dim, g.v_head_dim, g.kv_lora_rank,
    )
    q_head_dim = g.q_head_dim
    scaling = q_head_dim ** -0.5

    q_full = x @ w.q_proj.T  # [T, h*q_head_dim]
    q = q_full.view(t, h, q_head_dim)
    q_nope, q_rope = torch.split(q, [nope, rope], dim=-1)

    compressed_kv = x @ w.kv_a_proj.T  # [T, lora+rope]
    k_pass_raw, k_rot = torch.split(compressed_kv, [lora, rope], dim=-1)

    kv_a_normed = rms_norm(k_pass_raw, w.kv_a_norm, KV_A_NORM_EPS)  # [T, lora]
    kv_b = kv_a_normed @ w.kv_b_proj.T  # [T, h*(nope+v_hd)]
    kv_b_heads = kv_b.view(t, h, nope + v_hd)
    k_nope, v = torch.split(kv_b_heads, [nope, v_hd], dim=-1)

    # k_rot is the SAME vector for every head at a position (MQA-style),
    # never rotated: `key = concat(k_nope, k_rot.expand(heads))`.
    key = torch.cat([k_nope, k_rot.unsqueeze(1).expand(-1, h, -1)], dim=-1)  # [T,h,q_head_dim]
    query = torch.cat([q_nope, q_rope], dim=-1)  # [T,h,q_head_dim]

    scores = torch.einsum("thd,shd->hts", query, key).float() * scaling  # [h,T,T]
    causal = torch.full((t, t), float("-inf")).triu(diagonal=1)
    scores = scores + causal
    weights = torch.softmax(scores, dim=-1)  # [h,T,T]
    gated = {}
    if g_proj is not None:
        # L470-472: the gate reads the block INPUT `hidden_states`, never
        # the attention output; its sigmoid multiplies the AGGREGATE.
        pre = (x @ g_proj.T).float()  # [T, h*v_hd]
        output_gate = pre if mutation == "sigmoid_omitted" else torch.sigmoid(pre)
        if mutation == "gate_on_values_before_aggregation":
            v = v * output_gate.view(t, h, v_hd)
    attn_value = torch.einsum("hts,shd->thd", weights, v)  # [T,h,v_hd]

    pre_o_proj = attn_value.reshape(t, h * v_hd)
    if g_proj is not None:
        if mutation not in ("gate_omitted", "gate_on_values_before_aggregation"):
            pre_o_proj = pre_o_proj * output_gate
        gated = {"output_gate": output_gate, "gated_value": pre_o_proj}
    output = pre_o_proj @ w.o_proj.T  # [T, hidden]

    return {
        **gated,
        "q_proj": q_full,
        "compressed_kv": compressed_kv,
        "kv_a_normed": kv_a_normed,
        "kv_b": kv_b,
        "attn_weights": weights,  # [h, T, T] — per query position T, causal row [0..=t]
        "attn_value": attn_value.reshape(t, h * v_hd),
        "output": output,
    }
