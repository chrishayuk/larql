#!/usr/bin/env python3
"""Torch transcription of Kimi Delta Attention — the P3d parity oracle.

`fla` is Triton/CUDA and does not run on this machine, so the oracle for KDA
parity has to be a transcription. That makes provenance the first concern:
this file follows **the call the checkpoint's own `modeling_kimi.py` makes**,
not the signature upstream `fla` currently offers. The two have drifted —
today's `fused_kda_gate(g, A_log, dt_bias, lower_bound, …)` against the
checkpoint's `fused_kda_gate(g, A_log, head_dim, g_bias=dt_bias)` — and
reading the third positional argument as `dt_bias` would silently substitute
a head width for a bias.

Transcribed from, and pinned by sha256 (see `fla/SOURCES.sha256` beside the
cached copies):

    fla/ops/kda/naive.py            naive_recurrent_kda  — the recurrence
    fla/ops/kda/gate.py             naive_kda_gate       — the decay gate
    fla/modules/conv/short_conv.py  ShortConvolution     — depthwise causal
    fla/modules/fused_norm_gate.py  rms_norm_gated       — sigmoid-gated RMS

**`gate_lower_bound` is deliberately not an input.** Kimi Linear declares
`-5.0` and reads it nowhere — the field appears in neither
`modeling_kimi.py` nor `configuration_kimi.py` — and the gate call selects
the softplus form, not the `lower_bound · sigmoid(...)` form the same
upstream function also offers. Applying it would compute a different decay
envelope with every shape still closing. `assert_gate_bound_is_inert` pins
that.

Every intermediate is returned, not just the output: a recurrence that is
wrong in one factor still produces a plausible final tensor, and the point
of a parity ladder is to name *which* boundary moved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

#: `rms_norm_eps` of the Kimi Linear checkpoint.
RMS_EPS = 1e-5

#: Boundaries the parity ladder compares, in execution order. Named here so
#: a fixture and a report cannot drift on what "stage 6" means.
BOUNDARIES = (
    "q_proj", "k_proj", "v_proj",
    "q_conv", "k_conv", "v_conv",
    "q_norm", "k_norm",
    "f_lowrank", "g_decay", "beta",
    "recurrent_out", "o_gate", "o_norm", "output",
)


@dataclass
class KdaWeights:
    """One layer's fifteen operands, already in f32."""
    q_proj: torch.Tensor      # [H*D, hidden]
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    q_conv1d: torch.Tensor    # [H*D, 1, kernel]
    k_conv1d: torch.Tensor
    v_conv1d: torch.Tensor
    f_a_proj: torch.Tensor    # [D, hidden]
    f_b_proj: torch.Tensor    # [H*D, D]
    #: The output gate's LOW-RANK pair (Kimi Linear, GLM-5.3-Flash). `None`
    #: on both when the layer carries the FULL-RANK form instead — Kimi-K3's
    #: `linear_attn_config.use_full_rank_gate: true`, one `g_proj` of
    #: `[H*D, hidden]` (`modeling_kimi_linear.py` L531-537, L651-654). The
    #: form is passed to `kda_forward` as `g_proj=`; the dataclass keeps its
    #: fifteen named operands so every exporter that iterates its fields
    #: stays valid.
    g_a_proj: torch.Tensor | None
    g_b_proj: torch.Tensor | None
    b_proj: torch.Tensor      # [H, hidden]
    a_log: torch.Tensor       # [H] (stored [1,1,H,1])
    dt_bias: torch.Tensor     # [H*D]
    o_norm: torch.Tensor      # [D]
    o_proj: torch.Tensor      # [hidden, H*D]

    @property
    def num_heads(self) -> int:
        return self.b_proj.shape[0]

    @property
    def head_dim(self) -> int:
        return self.q_proj.shape[0] // self.num_heads


def short_conv(x: torch.Tensor, weight: torch.Tensor, state: torch.Tensor | None):
    """Depthwise causal convolution followed by SiLU.

    `x` is `[T, C]`, `weight` is `[C, 1, K]`. Causality comes from padding
    `K-1` on the LEFT only and taking the first `T` outputs — padding both
    sides, as a symmetric `padding=K-1` would, lets position `t` read `t+1`.

    `state` is the previous `K-1` inputs, so a continuation produces exactly
    what one pass over the concatenation would.
    """
    kernel = weight.shape[-1]
    seq = x.transpose(0, 1)                              # [C, T]
    prefix = state if state is not None else seq.new_zeros(seq.shape[0], kernel - 1)
    padded = torch.cat([prefix, seq], dim=-1)            # [C, K-1+T]
    out = F.conv1d(padded.unsqueeze(0), weight, groups=seq.shape[0]).squeeze(0)
    new_state = padded[:, -(kernel - 1):].clone()
    return F.silu(out.transpose(0, 1)), new_state


def kda_gate(g_pre: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor,
             num_heads: int, head_dim: int) -> torch.Tensor:
    """`g = -exp(A_log) * softplus(g_pre + dt_bias)`, per (head, dim).

    `A_log` is per HEAD and broadcasts across the head's dims; `dt_bias` is
    per CHANNEL, `[H*D]`. That asymmetry is the operator's signature — Gated
    DeltaNet's `dt_bias` is per head — so it is written out rather than
    folded into one reshape.
    """
    g = g_pre.float().view(-1, num_heads, head_dim) + dt_bias.view(num_heads, head_dim)
    return -a_log.float().view(num_heads, 1).exp() * F.softplus(g)


def recurrent_kda(q, k, v, g, beta, state=None):
    """`naive_recurrent_kda`, transcribed.

    q/k/v are `[T, H, D]`, `g` is `[T, H, D]` (log-space decay), `beta` is
    `[T, H]`. Returns `[T, H, D]` and the final `[H, D, D]` state.

    The delta rule is the middle line: the write is `beta·k` against the
    prediction ERROR `v - kᵀS`, not against `v`. Writing `v` directly is the
    single most plausible wrong transcription, and it matches at `T = 1`
    from a zero state — which is exactly why the ladder starts at 1 and does
    not stop there.
    """
    T, H, D = q.shape
    scale = D ** -0.5
    S = torch.zeros(H, D, D, dtype=torch.float32) if state is None else state.clone().float()
    out = torch.zeros(T, H, D, dtype=torch.float32)
    for t in range(T):
        q_t, k_t, v_t = q[t].float() * scale, k[t].float(), v[t].float()
        S = S * g[t].float().unsqueeze(-1).exp()
        err = v_t - (k_t.unsqueeze(-1) * S).sum(-2)
        S = S + torch.einsum('hk,hv->hkv', beta[t].float().unsqueeze(-1) * k_t, err)
        out[t] = torch.einsum('hk,hkv->hv', q_t, S)
    return out, S


#: Named defects of the OUTPUT GATE, for K3-REP-GATE-1's controls. Each
#: perturbs the real forward below at exactly one point — never a
#: hand-rolled copy — so the reference and its controls cannot drift.
GATE_MUTATIONS = (
    "none",
    # `o_gate := 0` — sigmoid(0) = 0.5 everywhere: the gate is skipped.
    "gate_skipped",
    # gate applied to the recurrent output BEFORE the RMS norm instead of
    # after it (`FusedRMSNormGated`'s norm-then-gate order inverted).
    "gate_before_norm",
    # the raw pre-activation multiplied in, no sigmoid.
    "sigmoid_omitted",
    # the gate applied to `v` before the recurrence and not after it — a
    # placement defect: the gate belongs after aggregation.
    "gate_on_value_before_recurrence",
)


def kda_forward(
    x: torch.Tensor,
    w: KdaWeights,
    state=None,
    conv_state=None,
    *,
    g_proj: torch.Tensor | None = None,
    mutation: str = "none",
) -> dict:
    """One KDA attention block. Returns every boundary in `BOUNDARIES`.

    `g_proj` selects the output gate's FORM: `None` runs the low-rank pair
    `w.g_a_proj`/`w.g_b_proj` (Kimi Linear); a `[H*D, hidden]` tensor runs
    Kimi-K3's full-rank gate, `g = g_proj(x)` (`modeling_kimi_linear.py`
    L651-654), and the pair must then be absent. Nothing else in the block
    changes between the two forms — the sigmoid, the gated RMS norm and
    `o_proj` are identical — which is what the full-rank fixture exists to
    pin. `mutation` names one of `GATE_MUTATIONS`.
    """
    if mutation not in GATE_MUTATIONS:
        raise ValueError(f"unknown gate mutation {mutation!r}")
    if (g_proj is None) == (w.g_a_proj is None or w.g_b_proj is None):
        raise ValueError("exactly one output-gate form: the g_a/g_b pair OR g_proj")
    H, D = w.num_heads, w.head_dim
    b = {}
    b["q_proj"] = x @ w.q_proj.T
    b["k_proj"] = x @ w.k_proj.T
    b["v_proj"] = x @ w.v_proj.T

    cs = conv_state or (None, None, None)
    b["q_conv"], cs_q = short_conv(b["q_proj"], w.q_conv1d, cs[0])
    b["k_conv"], cs_k = short_conv(b["k_proj"], w.k_conv1d, cs[1])
    b["v_conv"], cs_v = short_conv(b["v_proj"], w.v_conv1d, cs[2])

    q = b["q_conv"].view(-1, H, D)
    k = b["k_conv"].view(-1, H, D)
    v = b["v_conv"].view(-1, H, D)
    # `use_qk_l2norm_in_kernel=True`: q and k are L2-normalised per head
    # INSIDE the kernel, so it never appears in the modeling file.
    b["q_norm"] = q = F.normalize(q.float(), p=2, dim=-1)
    b["k_norm"] = k = F.normalize(k.float(), p=2, dim=-1)

    b["f_lowrank"] = (x @ w.f_a_proj.T) @ w.f_b_proj.T
    b["g_decay"] = kda_gate(b["f_lowrank"], w.a_log, w.dt_bias, H, D)
    b["beta"] = torch.sigmoid((x @ w.b_proj.T).float())

    # The output gate depends on `x` alone, so it is computed here, before
    # the recurrence, exactly as the executor does; the reference applies
    # it after (L651-654 sit after the recurrence call, but read only
    # `hidden_states`).
    if g_proj is None:
        o_gate = ((x @ w.g_a_proj.T) @ w.g_b_proj.T).view(-1, H, D)
    else:
        o_gate = (x @ g_proj.T).view(-1, H, D)
    if mutation == "gate_skipped":
        o_gate = torch.zeros_like(o_gate)
    b["o_gate"] = o_gate
    if mutation == "gate_on_value_before_recurrence":
        v = v * torch.sigmoid(o_gate.float())

    b["recurrent_out"], final_state = recurrent_kda(q, k, v, b["g_decay"], b["beta"], state)

    if mutation == "gate_before_norm":
        gated = b["recurrent_out"] * torch.sigmoid(b["o_gate"].float())
        b["o_norm"] = gated * torch.rsqrt(gated.pow(2).mean(-1, keepdim=True) + RMS_EPS) * w.o_norm
    else:
        normed = b["recurrent_out"] * torch.rsqrt(
            b["recurrent_out"].pow(2).mean(-1, keepdim=True) + RMS_EPS)
        if mutation == "sigmoid_omitted":
            b["o_norm"] = normed * w.o_norm * b["o_gate"].float()
        elif mutation == "gate_on_value_before_recurrence":
            b["o_norm"] = normed * w.o_norm
        else:
            b["o_norm"] = normed * w.o_norm * torch.sigmoid(b["o_gate"].float())
    b["output"] = b["o_norm"].reshape(-1, H * D) @ w.o_proj.T

    b["_state"] = final_state
    b["_conv_state"] = (cs_q, cs_k, cs_v)
    return b
