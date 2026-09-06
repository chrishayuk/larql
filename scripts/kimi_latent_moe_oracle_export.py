#!/usr/bin/env python3
"""Generate the committed `kimi_latent_moe_oracle.json` fixture — the
K3-LATENTMOE-1 parity oracle for Kimi-K3's LATENT routed branch.

    python3 scripts/kimi_latent_moe_oracle_export.py > \
        crates/larql-vindex/src/format/vindex3/opplan/exec/tests/kimi_latent_moe_oracle.json

Two ARMS of the same block on the same input and the same operands,
differing ONLY in the routed branch's representation:

    ordinary   experts run in the block's own hidden space
    latent     routed_expert_down_proj -> experts -> RMSNorm -> up_proj

The claim the fixture exists to make checkable:

    LatentMoE changes only the routed branch's representation and
    geometry. It does not change routing, the expert's internal operator,
    or the shared-expert branch.

So three quantities are asserted BIT-IDENTICAL across the arms rather
than described — `router_weights`, `shared_input`, `shared_output` — and
any downstream difference therefore belongs solely to the bottleneck.

The geometry is hostile to accidental K3-specific arithmetic. K3 happens
to satisfy `hidden / 2 == latent` (7168 / 2 == 3584) and a build that
derived the latent that way would be right on exactly one model; here
`hidden / 2 != latent`, and hidden, latent and the expert's inner width
are pairwise distinct, so a wrong-width mutant is a real measurement
rather than an assertion about constants.
"""

from __future__ import annotations

import json
import sys

import torch

import kimi_moe_reference as ref

SEED = 20260906

#: Pairwise distinct, and `hidden // 2 == 5 != 7 == latent` so the
#: forbidden derivation is not accidentally satisfied.
HIDDEN = 10
LATENT = 7
INTER = 4
EXPERTS = 5
TOP_K = 2

#: Kimi-K3's declared values for the two leaves under test.
ROUTED_SCALING_FACTOR = 1.0
MOE_RENORMALIZE = True
#: `config.rms_norm_eps` — the LAYER's epsilon, which
#: `routed_expert_norm` passes EXPLICITLY (L811-813).
LAYER_NORM_EPS = 1e-5

#: Scale of the expert weights. Chosen, and asserted below, so that
#: `mean(routed_sum^2)` lands in `AGGREGATE_BAND` — the band where the
#: difference between `1e-5` and `1e-6` is MATERIAL at `routed_normed`.
#: At a variance of 1.0 the two epsilons move `rsqrt` by ~5e-6 and the
#: anti-memory control would read as noise on any fixture.
EXPERT_SCALE = 0.18

#: `mean(routed_sum^2)` must sit here for the epsilon control to be
#: readable. A band on the SIGNAL, not on the control's delta: the
#: fixture is judged on whether it can SEE the fact.
AGGREGATE_BAND = {"min_mean_square": 1e-5, "max_mean_square": 1e-2}

#: The router must not be degenerate. One expert at weight 1.0 makes
#: every aggregation-placement control read the same number, because
#: `sum_i w_i f(e_i)` and `f(sum_i w_i e_i)` coincide at a single term.
ROUTER_BAND = {"max_weight": 0.9, "min_weight": 0.1}

#: Which boundary each mutation is CAUGHT at. Scored there rather than at
#: `output`, because a defect that is real and small at its own boundary
#: is still a defect — and scoring everything at the end of the ladder is
#: how a true fact gets recorded as inert.
CAUGHT_AT = {
    "norm_per_expert": "routed_sum",
    "norm_after_weighting": "routed_sum",
    "norm_omitted": "routed_out",
    "class_default_eps": "routed_normed",
}

#: Invariants across the two forms. Bit equality, not tolerance: each is
#: a quantity the bottleneck cannot reach.
CROSS_FORM_IDENTICAL = ("router_weights", "shared_input", "shared_output")


def lst(t: torch.Tensor) -> list:
    return [round(v, 8) for v in t.detach().flatten().tolist()]


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def build():
    torch.manual_seed(SEED)
    router = ref.RouterWeights(
        weight=torch.randn(EXPERTS, HIDDEN) * 0.4,
        e_score_correction_bias=torch.randn(EXPERTS) * 0.2,
    )
    # The experts are built at the LATENT width for the latent arm and at
    # HIDDEN for the ordinary one: the wrapper changes the space the
    # experts live in, which is the half of this rung that is geometry
    # rather than wrapper.
    def bank(width: int, seed_shift: int) -> dict[int, ref.ExpertWeights]:
        return {
            i: ref.ExpertWeights(
                w1=torch.randn(INTER, width) * EXPERT_SCALE,
                w2=torch.randn(width, INTER) * EXPERT_SCALE,
                w3=torch.randn(INTER, width) * EXPERT_SCALE,
            )
            for i in range(EXPERTS)
        }

    latent_experts = bank(LATENT, 0)
    ordinary_experts = bank(HIDDEN, 1)
    # The shared branch is HIDDEN-wide in both arms — it never enters the
    # bottleneck — and is drawn once so the two arms share it exactly.
    shared = ref.ExpertWeights(
        w1=torch.randn(INTER, HIDDEN) * EXPERT_SCALE,
        w2=torch.randn(HIDDEN, INTER) * EXPERT_SCALE,
        w3=torch.randn(INTER, HIDDEN) * EXPERT_SCALE,
    )
    wrapper = ref.LatentWrapper(
        down=torch.randn(LATENT, HIDDEN) * 0.3,
        up=torch.randn(HIDDEN, LATENT) * 0.3,
        norm=torch.rand(LATENT) * 0.5 + 0.75,  # near 1.0, never zero
        norm_eps=LAYER_NORM_EPS,
    )
    x = torch.randn(HIDDEN) * 0.5
    return router, latent_experts, ordinary_experts, shared, wrapper, x


def run(form, experts, router, shared, wrapper, x, mutation="none"):
    return ref.latent_moe_block_forward(
        x,
        router,
        experts,
        shared,
        TOP_K,
        MOE_RENORMALIZE,
        ROUTED_SCALING_FACTOR,
        form=form,
        latent=wrapper if form == "latent" else None,
        mutation=mutation,
    )


def bands(result: dict) -> dict:
    """Measure and ASSERT both bands before any control is scored."""
    mean_square = float(result["routed_sum"].pow(2).mean())
    assert (
        AGGREGATE_BAND["min_mean_square"] <= mean_square <= AGGREGATE_BAND["max_mean_square"]
    ), (
        f"mean(routed_sum^2) = {mean_square:.3e} is outside {AGGREGATE_BAND}; the layer "
        "epsilon is not material on this fixture and `class_default_eps` would read as noise"
    )
    weights = result["router_weights"]
    hi, lo = float(weights.max()), float(weights.min())
    assert hi <= ROUTER_BAND["max_weight"] and lo >= ROUTER_BAND["min_weight"], (
        f"router weights {weights.tolist()} are degenerate; at a single dominant expert the "
        "aggregation-placement controls coincide with the reference"
    )
    return {
        "routed_sum_mean_square": mean_square,
        "aggregate_limits": AGGREGATE_BAND,
        "router_max_weight": hi,
        "router_min_weight": lo,
        "router_limits": ROUTER_BAND,
    }


def cross_form(latent: dict, ordinary: dict) -> dict:
    """The three quantities the bottleneck cannot reach, asserted as BIT
    equalities. Any downstream difference then belongs solely to the
    routed branch's representation."""
    for name in CROSS_FORM_IDENTICAL:
        assert torch.equal(latent[name], ordinary[name]), (
            f"{name} differs between the routed-branch forms — LatentMoE must change only "
            "the routed branch's representation and geometry"
        )
    assert torch.equal(latent["router_ids"], ordinary["router_ids"]), (
        "the two forms selected different experts; routing must not see the bottleneck"
    )
    return {
        "bit_identical": list(CROSS_FORM_IDENTICAL) + ["router_ids"],
        "note": (
            "asserted with torch.equal at export. The router reads the block input before "
            "any projection, and the shared branch reads it too and is added after the "
            "up-projection, so neither can depend on how the routed branch is represented."
        ),
    }


def boundaries(result: dict) -> dict:
    out = {}
    for name in ref.LATENT_BOUNDARIES:
        if name == "router_input":
            out[name] = lst(result["router_input"])
        elif result.get(name) is None:
            out[name] = None
        else:
            out[name] = lst(result[name])
    out["expert_outputs"] = [lst(o) for o in result["expert_outputs"]]
    out["router_ids"] = result["router_ids"].tolist()
    return out


def main() -> int:
    router, latent_experts, ordinary_experts, shared, wrapper, x = build()
    latent = run("latent", latent_experts, router, shared, wrapper, x)
    ordinary = run("ordinary", ordinary_experts, router, shared, None, x)

    fixture = {
        "_source": "moonshotai/Kimi-K3 modeling_kimi_linear.py L769-841 (KimiSparseMoeBlock)",
        "_generator": "scripts/kimi_latent_moe_oracle_export.py, from scripts/kimi_moe_reference.py",
        "_only_the_norm_is_silent": (
            "Measured while building this fixture, and it corrects a natural assumption: "
            "of LatentMoE's three placement facts only the NORM's can be got wrong with "
            "every shape still closing. The router is [experts, hidden] and the shared "
            "expert is hidden-wide, so routing on the latent or running the shared branch "
            "through the bottleneck both fail on shape at the reference's own geometry. "
            "Those two get positive cross-form witnesses instead of mutants; the norm gets "
            "four mutants, because it is the one that can be silently wrong."
        ),
        "_claim": (
            "LatentMoE changes only the routed branch's representation and geometry; it does "
            "not change routing, the expert's internal operator, or the shared-expert branch."
        ),
        "hidden": HIDDEN,
        "latent": LATENT,
        "intermediate": INTER,
        "experts": EXPERTS,
        "top_k": TOP_K,
        "moe_renormalize": MOE_RENORMALIZE,
        "routed_scaling_factor": ROUTED_SCALING_FACTOR,
        # Both epsilons exported, so the Rust side never reconstructs what
        # the negative control meant.
        "latent_norm_eps": LAYER_NORM_EPS,
        "class_default_eps_not_used_here": ref.CLASS_DEFAULT_EPS,
        "geometry_note": (
            f"hidden // 2 == {HIDDEN // 2} != {LATENT} == latent, on purpose: K3 satisfies "
            "hidden/2 == latent and a build deriving the width that way would be right on "
            "exactly one model. hidden, latent and intermediate are pairwise distinct."
        ),
        "input": lst(x),
        "weights": {
            "router": lst(router.weight),
            "router_bias": lst(router.e_score_correction_bias),
            "routed_expert_down_proj": lst(wrapper.down),
            "routed_expert_norm": lst(wrapper.norm),
            "routed_expert_up_proj": lst(wrapper.up),
            "shared_w1": lst(shared.w1),
            "shared_w2": lst(shared.w2),
            "shared_w3": lst(shared.w3),
            "latent_experts": {
                str(i): {"w1": lst(e.w1), "w2": lst(e.w2), "w3": lst(e.w3)}
                for i, e in latent_experts.items()
            },
            "ordinary_experts": {
                str(i): {"w1": lst(e.w1), "w2": lst(e.w2), "w3": lst(e.w3)}
                for i, e in ordinary_experts.items()
            },
        },
        "bands": bands(latent),
        "cross_form": cross_form(latent, ordinary),
        "structurally_unreachable": {
            "router_reads_the_latent": (
                "the router matrix is [experts, hidden], so feeding it the [latent] vector "
                "fails on shape. This placement fact is SHAPE-PROTECTED and cannot be a "
                "silent defect, unlike the norm's placement and the shared branch's. Its "
                "positive witness is `cross_form`: router_weights and router_ids are "
                "bit-identical between the two forms, so the router saw the same input. "
                "Recorded as unreachable, never as covered."
            ),
            "shared_reads_the_latent": (
                "the shared expert's w1 is [inter, hidden], so feeding it the [latent] "
                "vector fails on shape. Shape-protected; witnessed positively by "
                "`cross_form`'s bit-identical shared_input and shared_output."
            ),
            "shared_added_before_up_proj": (
                f"summing the [{LATENT}] routed aggregate with the [{HIDDEN}] shared "
                "contribution does not add. Shape-protected."
            ),
            "down_up_swapped": (
                f"down is [{LATENT}, {HIDDEN}] and up is [{HIDDEN}, {LATENT}]; transposing "
                "them into each other's slots fails on shape whenever latent != hidden, "
                "which this fixture and K3 both satisfy."
            ),
            "norm_after_up_proj": (
                f"applying the norm to the expanded [{HIDDEN}] vector instead of the "
                f"[{LATENT}] aggregate needs latent == hidden to be expressible; this "
                "fixture keeps them distinct on purpose."
            ),
        },
        "arms": {
            "latent": boundaries(latent),
            "ordinary": boundaries(ordinary),
        },
    }

    controls = {}
    for mutation in ref.LATENT_MUTATIONS:
        if mutation == "none":
            continue
        mutant = run("latent", latent_experts, router, shared, wrapper, x, mutation)
        at = CAUGHT_AT[mutation]
        deltas = {}
        for name in ref.LATENT_BOUNDARIES:
            a, b = mutant.get(name), latent.get(name)
            deltas[name] = None if (a is None or b is None) else rel_l2(a, b)
        here = deltas[at]
        assert here is not None and here > 1e-3, (
            f"control {mutation} reads {here} at `{at}` — it cannot tell that defect from "
            "the reference on this fixture"
        )
        print(
            f"latent control {mutation:28s} caught at {at:14s} {here:.3e}"
            f"  output {deltas['output']:.3e}",
            file=sys.stderr,
        )
        controls[mutation] = {
            "caught_at": at,
            "delta_rel_l2": deltas,
            "boundaries": boundaries(mutant),
        }

    # The two aggregation-placement mutants must be DISTINCT from each
    # other as well as from the reference: `sum w*norm(e)` and
    # `sum norm(w*e)` are different wrong answers, and a fixture where
    # they coincided would witness one defect while appearing to witness
    # two.
    per_expert = controls["norm_per_expert"]["delta_rel_l2"]["routed_sum"]
    after_weight = controls["norm_after_weighting"]["delta_rel_l2"]["routed_sum"]
    assert abs(per_expert - after_weight) > 1e-6, (
        "the two norm-placement controls coincide on this fixture; one of them is not the "
        f"defect it claims to be ({per_expert} vs {after_weight})"
    )

    fixture["controls"] = controls
    json.dump(fixture, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
