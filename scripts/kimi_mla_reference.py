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

#: The SAME number for `q_a_layernorm`, and deliberately its own name.
#:
#: `q_a_layernorm = KimiRMSNorm(self.q_lora_rank)` (`modeling_kimi_
#: linear.py` L368) also passes no `eps`, so it also runs at the class
#: default — not `config.rms_norm_eps` (`1e-5`), which the layer's own
#: two norms use.
#:
#: They are equal because they share a CAUSE (one class default), not
#: because they share an AUTHORITY. A single constant for both would make
#: a coincidence into a contract, and a family that overrode one and not
#: the other would then be silently wrong. Two names, one value, and the
#: fixture exports both so a consumer cannot collapse them either.
Q_A_NORM_EPS = 1e-6

#: The layer epsilon the q-A norm does NOT use. Here only so the
#: `layer_eps_for_q_a` mutant reads a real declared number rather than an
#: arbitrary perturbation.
LAYER_NORM_EPS = 1e-5

#: Boundaries this transcription exposes per position, in execution
#: order. Named here so a fixture and a report cannot drift on what one
#: means — same convention `kda_reference.py`'s own `BOUNDARIES` sets.
#: `q_states` rather than `q_proj`: under the low-rank query form
#: nothing computes a `q_proj` at all, and the reference's own variable
#: name for the query leaving either form is `q_states` (L419/421). The
#: rename is the only key that moves; every VALUE the two pre-existing
#: arms export is unchanged, which their export asserts.
BOUNDARIES = (
    "q_states", "compressed_kv", "kv_a_normed", "kv_b",
    "attn_weights", "attn_value", "output",
)

#: The ladder with Kimi-K3's FACTORISED query in it (`q_lora_rank:
#: 1536`, `modeling_kimi_linear.py` L364-372, L419): `q_a_proj` into the
#: rank, `q_a_layernorm` over the rank, `q_b_proj` back out to
#: `Hq*q_head_dim` — and then the identical split every form performs.
#:
#: `q_pass`/`q_rot` are exported for BOTH forms' sake even though the
#: split is form-independent: a wrong split produces finite attention
#: over correctly-shaped tensors, and without the decomposed components
#: it surfaces as an attention difference with no way to localise it.
Q_LORA_BOUNDARIES = (
    "q_a", "q_a_normed", "q_b", "q_states", "q_pass", "q_rot",
    "compressed_kv", "kv_a_normed", "kv_b",
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


#: Named defects of the QUERY path, for K3-MLA-Q-LORA-1's controls. Each
#: perturbs the real forward below at exactly one point — never a
#: hand-rolled copy — so the reference and its controls cannot drift.
#:
#: The mutants that need DIFFERENT OPERANDS rather than a different code
#: path (an ordinary `q_proj` substituted, a `q_b` built at the wrong
#: rank, a `q_b` whose columns are `hidden`) are not here: the caller
#: supplies those operands to this same unmodified forward, which is the
#: same rule by another route.
QUERY_MUTATIONS = (
    "none",
    # No `q_a_layernorm` at all: `q_b(q_a(x))`.
    "q_a_norm_omitted",
    # The layer's own `rms_norm_eps` (1e-5) instead of the class default
    # the constructor actually leaves in place (1e-6). THE likeliest
    # silent defect in this rung, and probably a small delta — which is
    # why it is scored at the boundary it is caught at, not at `output`.
    "layer_eps_for_q_a",
    # The norm computed AND REPORTED, but `q_b` fed the un-normed `q_a`.
    # Output-identical to `q_a_norm_omitted` and trace-DIFFERENT: this is
    # the control for the TRACE, proving `q_a_normed` is the value the
    # next stage consumed and not a separately-computed display.
    "q_b_fed_pre_norm",
    # Within each head, the first `qk_rope_head_dim` entries taken as the
    # rope component instead of the last.
    "split_rope_first",
    # The flat `[Hq*q_head_dim]` vector cut into one `[Hq*nope]` block
    # and one `[Hq*rope]` block instead of split per head.
    "split_flat",
)

#: Query mutations whose effect is confined to the query ladder and
#: therefore CANNOT move a KV-side boundary. Exported so a consumer can
#: assert the non-query side stayed put under each of them.
QUERY_ONLY = tuple(m for m in QUERY_MUTATIONS if m != "none")


@dataclass
class QLoraWeights:
    """The factorised query's three operands, already in f32.

    The rank cannot be read back out of these shapes without knowing
    which axis is which — `q_a_proj` is `[rank, hidden]` and `q_b_proj`
    is `[Hq*q_head_dim, rank]`, and on a fixture where `hidden` happened
    to equal `Hq*q_head_dim` the pair would be ambiguous — so the caller
    states it, exactly as `MlaGeometry` states `num_heads`."""
    q_a_proj: torch.Tensor     # [rank, hidden]
    q_a_norm: torch.Tensor     # [rank]
    q_b_proj: torch.Tensor     # [Hq*q_head_dim, rank]
    rank: int


@dataclass
class MlaWeights:
    """One layer's five operands, already in f32. `num_heads` cannot be
    read back out of these shapes alone (`q_proj` rows are
    `Hq*q_head_dim`, `kv_b_proj` rows are `Hq*(nope+v_head_dim)` — either
    needs nope/rope/v_head_dim to invert), so every caller states it
    explicitly via `MlaGeometry` instead."""
    #: `[Hq*q_head_dim, hidden]`. `None` under the low-rank query form,
    #: where the reference's `__init__` never constructs it (L364-376 is
    #: an `if/else`: a q-LoRA layer has no `q_proj` attribute at all).
    q_proj: torch.Tensor | None  # [Hq*q_head_dim, hidden]
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


def query_states(
    x: torch.Tensor,
    w: MlaWeights,
    query_form: str,
    q_lora: QLoraWeights | None,
    query_mutation: str,
) -> tuple[torch.Tensor, dict]:
    """The query leaving either form, plus the low-rank form's own
    boundaries.

    `query_form` is STATED, never inferred from which operands happen to
    be present: the reference branches on `config.q_lora_rank is not
    None` (L364, L418), a declaration, and a transcription that instead
    sniffed the weights would be answering a different question — and
    would silently pick a form for a checkpoint that shipped both.

    Returns `(q_states, extras)`; `extras` is empty under `"direct"`.
    """
    if query_form == "direct":
        if w.q_proj is None:
            raise ValueError("the direct query form needs `q_proj`")
        if q_lora is not None:
            raise ValueError(
                "the direct query form was stated and q-LoRA operands were supplied; "
                "the reference constructs one form or the other, never both"
            )
        if query_mutation not in ("none", "split_rope_first", "split_flat"):
            raise ValueError(
                f"{query_mutation!r} perturbs the factorisation, which the direct form "
                "does not have"
            )
        return x @ w.q_proj.T, {}

    if query_form != "low_rank":
        raise ValueError(f"unknown query form {query_form!r}")
    if q_lora is None:
        raise ValueError("the low-rank query form needs its three operands")
    if w.q_proj is not None:
        raise ValueError(
            "the low-rank query form was stated and a `q_proj` was supplied; "
            "the reference constructs one form or the other, never both"
        )

    # L419: q_b_proj(q_a_layernorm(q_a_proj(hidden_states))).
    q_a = x @ q_lora.q_a_proj.T                                    # [T, rank]
    eps = LAYER_NORM_EPS if query_mutation == "layer_eps_for_q_a" else Q_A_NORM_EPS
    q_a_normed = rms_norm(q_a, q_lora.q_a_norm, eps)               # [T, rank]
    if query_mutation == "q_a_norm_omitted":
        q_a_normed = q_a
    # The norm stays REPORTED and only its consumer changes, which is the
    # whole point of this arm: it is output-identical to omitting the
    # norm and trace-different.
    into_b = q_a if query_mutation == "q_b_fed_pre_norm" else q_a_normed
    q_b = into_b @ q_lora.q_b_proj.T                                # [T, Hq*q_head_dim]
    return q_b, {"q_a": q_a, "q_a_normed": q_a_normed, "q_b": q_b}


def split_query(
    q_full: torch.Tensor, g: MlaGeometry, query_mutation: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """L422-424: view per head, then split `[nope | rope]` on the last
    axis. Form-independent — both query forms reach here with the same
    object — and the two wrong cuts below both produce finite attention
    over correctly-shaped tensors, which is why they are controls."""
    t = q_full.shape[0]
    h, nope, rope = g.num_heads, g.qk_nope_head_dim, g.qk_rope_head_dim
    if query_mutation == "split_flat":
        # One `[Hq*nope]` block then one `[Hq*rope]` block: every element
        # is still visited exactly once, so no shape or bounds check can
        # see it.
        return (
            q_full[:, : h * nope].reshape(t, h, nope),
            q_full[:, h * nope :].reshape(t, h, rope),
        )
    q = q_full.view(t, h, g.q_head_dim)
    if query_mutation == "split_rope_first":
        q_rope, q_nope = torch.split(q, [rope, nope], dim=-1)
        return q_nope, q_rope
    return torch.split(q, [nope, rope], dim=-1)


def mla_forward(
    x: torch.Tensor,
    w: MlaWeights,
    g: MlaGeometry,
    *,
    query_form: str,
    q_lora: QLoraWeights | None = None,
    query_mutation: str = "none",
    g_proj: torch.Tensor | None = None,
    mutation: str = "none",
) -> dict:
    """`x` is `[T, hidden]`. Returns every boundary, per position, plus
    the final `[T, hidden]` output — causal, `mla_use_nope` (no rotation
    anywhere), exactly `KimiMLAAttention.forward` at batch size 1.

    `query_form` is REQUIRED and states which query the layer builds —
    `"direct"` (one `q_proj`) or `"low_rank"` (`q_a_proj` ->
    `q_a_layernorm` -> `q_b_proj`, with `q_lora` supplying the three).
    It is never inferred from which operands were passed; see
    `query_states`. `query_mutation` names one of `QUERY_MUTATIONS`.

    `g_proj` (`[Hq*v_head_dim, hidden]`) adds Kimi-K3's output gate
    (`GATED_BOUNDARIES`); `None` is Kimi Linear's ungated block and
    returns exactly `BOUNDARIES`. `mutation` names one of
    `GATE_MUTATIONS` and is only meaningful with a gate.

    The two query forms differ ONLY in how `q_states` is produced.
    Everything from the split onward — the KV path, the un-rotated shared
    rope-K, the attention, the gate, `o_proj` — reads the same code here
    whichever form ran, which is what makes an equivalence fixture
    (two forms constructed to yield the same `q_states`) a real check on
    the rest of the block rather than a tautology.
    """
    if mutation not in GATE_MUTATIONS:
        raise ValueError(f"unknown gate mutation {mutation!r}")
    if g_proj is None and mutation != "none":
        raise ValueError("gate mutations need a gate")
    if query_mutation not in QUERY_MUTATIONS:
        raise ValueError(f"unknown query mutation {query_mutation!r}")
    t, hidden = x.shape
    h, nope, rope, v_hd, lora = (
        g.num_heads, g.qk_nope_head_dim, g.qk_rope_head_dim, g.v_head_dim, g.kv_lora_rank,
    )
    q_head_dim = g.q_head_dim
    scaling = q_head_dim ** -0.5

    q_full, q_extras = query_states(x, w, query_form, q_lora, query_mutation)
    q_nope, q_rope = split_query(q_full, g, query_mutation)

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
        **q_extras,
        "q_pass": q_nope.reshape(t, -1),
        "q_rot": q_rope.reshape(t, -1),
        "q_states": q_full,
        "compressed_kv": compressed_kv,
        "kv_a_normed": kv_a_normed,
        "kv_b": kv_b,
        "attn_weights": weights,  # [h, T, T] — per query position T, causal row [0..=t]
        "attn_value": attn_value.reshape(t, h * v_hd),
        "output": output,
    }
