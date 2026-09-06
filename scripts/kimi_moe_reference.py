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


#: Boundaries the LatentMoE ladder exposes, in execution order. The
#: routed branch and the shared branch are witnessed SEPARATELY and the
#: final sum only after both, so a difference can be attributed to one
#: side rather than to "the block".
LATENT_BOUNDARIES = (
    "router_input", "router_weights",
    "latent", "routed_sum", "routed_normed", "routed_out",
    "shared_input", "shared_output",
    "output",
)

#: Named defects of the LATENT WRAPPER. Each perturbs the real forward
#: below at exactly one point — never a hand-rolled copy.
#:
#: The first three are the PLACEMENT controls, and they exist because
#: every tensor shape in this block stays valid while each of them is
#: wrong: the router reads the un-projected hidden (L818 precedes L822),
#: the norm sees the weighted aggregate (L830 follows `moe_infer`), and
#: the shared branch is outside the bottleneck (L836-838).
#: **Only the norm is a silent defect.** Measured while building this
#: file, and it corrects an assumption worth stating: of the three
#: placement facts, only the norm's placement can be got wrong with every
#: shape still closing.
#:
#:   router before down      SHAPE-PROTECTED — the router is
#:                           `[experts, hidden]`; the latent does not fit
#:   shared outside          SHAPE-PROTECTED — the shared expert is
#:                           hidden-wide, and `[latent] + [hidden]` does
#:                           not add
#:   down/up swapped         SHAPE-PROTECTED whenever latent != hidden
#:   norm after aggregation  NOT protected. `norm(sum w_i e_i)`,
#:                           `sum w_i norm(e_i)` and `sum norm(w_i e_i)`
#:                           are all `[latent]` and all type-check.
#:
#: So the mutation set below is the norm's, and the shape-protected facts
#: get POSITIVE witnesses instead: the export asserts `router_weights`,
#: `router_ids`, `shared_input` and `shared_output` bit-identical across
#: the two forms. Recorded as unreachable, never as covered.
LATENT_MUTATIONS = (
    "none",
    # `sum_i w_i * norm(e_i)` instead of `norm(sum_i w_i * e_i)`.
    "norm_per_expert",
    # `sum_i norm(w_i * e_i)` — normalised after weighting, before the sum.
    "norm_after_weighting",
    # The flag declared and the norm skipped.
    "norm_omitted",
    # `KimiRMSNorm`'s class default instead of the declared layer eps.
    #
    # THE ANTI-MEMORY ARM. Kimi-K3's two low-rank ATTENTION norms
    # (`q_a_layernorm` L368, `kv_a_layernorm` L383) pass no `eps` and so
    # run at 1e-6 — the only two such sites in the file. This one passes
    # `eps=config.rms_norm_eps` EXPLICITLY (L811-813). Anyone
    # generalising the attention finding computes this mutant instead of
    # the reference.
    "class_default_eps",
)

#: `KimiRMSNorm`'s class default — what this norm does NOT run at, kept
#: here so `class_default_eps` names a real declared number rather than
#: an arbitrary perturbation.
CLASS_DEFAULT_EPS = 1e-6


@dataclass
class LatentWrapper:
    """Kimi-K3's routed-branch bottleneck (`routed_expert_hidden_size`).

    `norm` is `None` when `latent_moe_use_norm` is false or absent — and
    the reference gates its construction on the wrapper existing at all
    (L803 encloses L810), so a norm without a wrapper is not a state this
    dataclass can describe."""
    down: torch.Tensor            # [latent, hidden]
    up: torch.Tensor              # [hidden, latent]
    norm: torch.Tensor | None     # [latent]
    norm_eps: float               # config.rms_norm_eps — the LAYER's


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """`KimiRMSNorm.forward` (L232-236): f32 internally, cast back, affine
    applied in the input dtype."""
    dtype = x.dtype
    f = x.float()
    f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
    return weight * f.to(dtype)


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


def latent_moe_block_forward(
    x: torch.Tensor,
    router: RouterWeights,
    experts: dict[int, ExpertWeights],
    shared: ExpertWeights,
    top_k: int,
    moe_renormalize: bool,
    routed_scaling_factor: float,
    *,
    form: str,
    latent: LatentWrapper | None = None,
    mutation: str = "none",
) -> dict:
    """`KimiSparseMoeBlock.forward` with Kimi-K3's LATENT routed branch
    (`routed_expert_hidden_size`), specialised to one token.

    `form` is STATED — `"ordinary"` or `"latent"` — never inferred from
    whether a wrapper was supplied. The reference branches on
    `getattr(config, 'routed_expert_hidden_size', None) is not None`
    (L776), a declaration; a transcription that sniffed the operands
    would be answering a different question, and would answer it silently
    for a checkpoint that shipped both.

    The claim this function exists to make checkable:

        LatentMoE changes only the routed branch's representation and
        geometry. It does not change routing, the expert's internal
        operator, or the shared-expert branch.

    So the router runs on `x` before any projection, the experts are the
    same `expert_forward` both arms use, and the shared branch reads `x`
    — and the export asserts all three as BIT equalities across the two
    forms rather than leaving them to be read off the code.

    NOTE the expert operator here is the SiLU-GLU one the two pre-existing
    arms use, not Kimi-K3's SiTU. That is deliberate isolation: LatentMoE
    is orthogonal to the activation (K3-ACT-1's cell), and varying both at
    once would make this arm test two things and localise neither.
    """
    if form not in ("ordinary", "latent"):
        raise ValueError(f"unknown routed-branch form {form!r}")
    if mutation not in LATENT_MUTATIONS:
        raise ValueError(f"unknown latent mutation {mutation!r}")
    if (form == "latent") != (latent is not None):
        raise ValueError(
            "the latent form needs a wrapper and the ordinary form must not be given one; "
            "the reference constructs one or the other, never both"
        )
    if form == "ordinary" and mutation != "none":
        raise ValueError(f"{mutation!r} perturbs the wrapper, which this form does not have")

    # L818: the router reads the BLOCK INPUT, before any projection. The
    # mutation is the only way this line sees anything else.
    router_input = x
    r = route(router_input, router, top_k, moe_renormalize, routed_scaling_factor)
    ids = r["ids"].tolist()
    weights = r["weights"].tolist()

    # L822: the experts' input. Ordinary = the block input; latent = the
    # down-projection of it.
    if form == "latent":
        expert_input = torch.nn.functional.linear(x.float(), latent.down.float(), None)
    else:
        expert_input = x

    expert_outputs = [expert_forward(expert_input, experts[i]) for i in ids]

    # `moe_infer` weights each expert's output and sums. The two norm
    # placement mutants move the norm INSIDE this aggregation, which is
    # exactly what the post-aggregate reference does not do.
    if form == "latent" and latent.norm is not None and mutation == "norm_per_expert":
        terms = [w * rms_norm(o, latent.norm, latent.norm_eps)
                 for w, o in zip(weights, expert_outputs)]
    elif form == "latent" and latent.norm is not None and mutation == "norm_after_weighting":
        terms = [rms_norm(w * o, latent.norm, latent.norm_eps)
                 for w, o in zip(weights, expert_outputs)]
    else:
        terms = [w * o for w, o in zip(weights, expert_outputs)]
    routed_sum = sum(terms)

    routed_normed = None
    routed = routed_sum
    if form == "latent":
        # L830-831: the norm sees the AGGREGATE, one vector per token.
        if latent.norm is not None and mutation not in (
            "norm_omitted", "norm_per_expert", "norm_after_weighting"
        ):
            eps = CLASS_DEFAULT_EPS if mutation == "class_default_eps" else latent.norm_eps
            routed_normed = rms_norm(routed_sum, latent.norm, eps)
            routed = routed_normed

    # L836-838: the shared branch reads the BLOCK INPUT and is added
    # AFTER the up-projection.
    shared_input = x
    shared_output = expert_forward(shared_input, shared)

    if form == "latent":
        routed_out = torch.nn.functional.linear(routed.float(), latent.up.float(), None)
        output = routed_out + shared_output
    else:
        routed_out = routed
        output = routed_out + shared_output

    return {
        "router_input": router_input,
        "router_weights": r["weights"],
        "router_ids": r["ids"],
        "latent": expert_input,
        "expert_outputs": expert_outputs,
        "routed_sum": routed_sum,
        "routed_normed": routed_normed,
        "routed_out": routed_out,
        "shared_input": shared_input,
        "shared_output": shared_output,
        "output": output,
    }
