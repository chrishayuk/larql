#!/usr/bin/env python3
"""Torch transcription of Kimi Linear's MoE router + block — the P3d-g
parity oracle for `kimi_router.rs` / `kimi_moe_block.rs`.

Importing `modeling_kimi.py` directly is not possible on this machine: its
top-level `try/except ImportError: raise ImportError(...)` for `fla`
re-raises rather than falling back, and `fla` is Triton/CUDA (same finding
KDA's oracle already recorded). So this is a transcription, the same
posture `kda_reference.py` takes — but the router/MoE-block classes touch
no `fla` code at all (`KimiMoEGate`, `KimiBlockSparseMLP`, `KimiMLP`,
`KimiSparseMoeBlock.moe_infer` are plain `nn.Linear` + sigmoid + topk +
SiLU), so the transcription risk here is far lower than KDA's recurrence
was.

Transcribed from the checkpoint's own `modeling_kimi.py`
(sha256 `d79b365e3737…`, same file KDA's oracle pins):

    class KimiMoEGate.forward           lines ~651-693
    class KimiBlockSparseMLP.forward    lines ~258-262
    class KimiMLP.forward               lines ~279-282
    KimiSparseMoeBlock.forward/moe_infer lines ~733-780

**Deliberately not modelled**: expert groups (`num_expert_group` /
`topk_group`). The reference's group-topk masking is a no-op at the
identity values (`num_expert_group=1`, `topk_group=1`) admission requires
(`plan/tests/moe_spellings.rs::more_than_one_expert_group_blocks`), so
this oracle asserts those values and skips the masking machinery rather
than transcribing dead code.

Every intermediate is returned, not just the final output — a router
that is wrong in one factor still produces a plausible top-k, and the
point of the ladder is to name which stage moved.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

#: Router boundaries, in execution order.
ROUTER_BOUNDARIES = (
    "logits", "scores", "selection_scores", "gathered_weights",
    "normalized_weights", "weights",
)


@dataclass
class RouterWeights:
    weight: torch.Tensor              # [experts, hidden]
    e_score_correction_bias: torch.Tensor  # [experts]


def route(
    x: torch.Tensor,
    w: RouterWeights,
    top_k: int,
    moe_renormalize: bool,
    routed_scaling_factor: float,
) -> dict:
    """`KimiMoEGate.forward`, sigmoid branch, `num_expert_group == 1`.

    `x`: `[hidden]`. Returns every boundary plus `ids` (`LongTensor
    [top_k]`) and `weights` (`[top_k]`).
    """
    logits = F.linear(x.float(), w.weight.float(), None)
    scores = logits.sigmoid()
    selection_scores = scores + w.e_score_correction_bias.float()

    # `num_expert_group == 1`: group-topk selects the one group that
    # exists, masking nothing — `tmp_scores == selection_scores`.
    _, topk_idx = torch.topk(selection_scores, k=top_k, dim=-1, sorted=False)
    gathered_weights = scores.gather(0, topk_idx)

    if top_k > 1 and moe_renormalize:
        denom = gathered_weights.sum(dim=-1, keepdim=True) + 1e-20
        normalized_weights = gathered_weights / denom
    else:
        normalized_weights = gathered_weights

    weights = normalized_weights * routed_scaling_factor

    return {
        "logits": logits,
        "scores": scores,
        "selection_scores": selection_scores,
        "ids": topk_idx,
        "gathered_weights": gathered_weights,
        "normalized_weights": normalized_weights,
        "weights": weights,
    }


@dataclass
class ExpertWeights:
    """`w1` (gate), `w3` (up), `w2` (down) — the checkpoint's own naming,
    confirmed against `KimiBlockSparseMLP.__init__`'s inline comment."""
    w1: torch.Tensor  # [inter, hidden]
    w2: torch.Tensor  # [hidden, inter]
    w3: torch.Tensor  # [inter, hidden]


def expert_forward(x: torch.Tensor, w: ExpertWeights) -> torch.Tensor:
    """`KimiBlockSparseMLP.forward` / `KimiMLP.forward` — identical shape,
    `act_fn(w1(x)) * w3(x)` then `w2(...)`, `act_fn = silu`."""
    gate = F.linear(x.float(), w.w1.float(), None)
    up = F.linear(x.float(), w.w3.float(), None)
    h = F.silu(gate) * up
    return F.linear(h, w.w2.float(), None)


def moe_block_forward(
    x: torch.Tensor,
    router: RouterWeights,
    experts: dict[int, ExpertWeights],
    shared: ExpertWeights,
    top_k: int,
    moe_renormalize: bool,
    routed_scaling_factor: float,
) -> dict:
    """`KimiSparseMoeBlock.forward` + `moe_infer`, specialised to ONE
    token: `moe_infer`'s sort/argsort/scatter dance is a batched
    implementation of "for each selected expert, run it on this token,
    weight by that token's weight for that expert, sum" — which is
    exactly what this computes directly. `experts` must have an entry for
    every id `route()` selects; a missing one raises `KeyError`, the
    equivalent of `moe_infer` indexing `self.experts[i]` on an
    unpopulated `ModuleList`.
    """
    r = route(x, router, top_k, moe_renormalize, routed_scaling_factor)
    ids = r["ids"].tolist()
    weights = r["weights"].tolist()

    expert_outputs = [expert_forward(x, experts[i]) for i in ids]
    routed_sum = sum(w * out for w, out in zip(weights, expert_outputs))

    shared_output = expert_forward(x, shared)
    # `y = y + self.shared_experts(identity)` — summed, never scaled.
    output = routed_sum + shared_output

    return {
        "router": r,
        "expert_outputs": expert_outputs,
        "routed_sum": routed_sum,
        "shared_output": shared_output,
        "output": output,
    }
