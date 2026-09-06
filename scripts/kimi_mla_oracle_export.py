#!/usr/bin/env python3
"""Generate the tiny, COMMITTED `kimi_mla_oracle.json` fixture for
`exec::mla`'s own parity test.

Synthetic on purpose, same reasoning `kda_oracle.json` (2 heads x 4)
already established for the KDA operator: the arithmetic is identical at
any width, small enough to commit is small enough to actually get run,
and this operator's real-checkpoint width (32 heads, 512-wide latent) is
its own SEPARATE env-gated real-weight gate
(`kimi_mla_layer_export.py`/`kimi_mla_layer_real.rs`) — proving indexing
and stride at scale, not the formula.

    python scripts/kimi_mla_oracle_export.py > \
        crates/larql-vindex/src/format/vindex3/opplan/exec/tests/kimi_mla_oracle.json

K3-REP-GATE-1 adds a SECOND arm, never an overwrite of the first: the same
geometry, seed, every operand and the input, plus Kimi-K3's OUTPUT GATE
(`mla_use_output_gate: true`) — one `g_proj` of `[Hq*v_head_dim, hidden]`
read from the block input, `sigmoid`, multiplied into the aggregated
value before `o_proj` (`modeling_kimi_linear.py` L470-472). Every
boundary up to `attn_value` is IDENTICAL between the arms; the gated arm
adds `output_gate` and `gated_value` and changes `output`. `hidden` (7)
differs from `Hq*v_head_dim` (10) on purpose, so a gate applied after
`o_proj` cannot even be expressed here. The arm ships its named controls
(`kimi_mla_reference.GATE_MUTATIONS`) with measured deltas and asserts
the gate band first.

    python scripts/kimi_mla_oracle_export.py --output-gate > \
        crates/larql-vindex/src/format/vindex3/opplan/exec/tests/kimi_mla_oracle_output_gate.json

K3-MLA-Q-LORA-1 adds a THIRD arm, again never an overwrite: Kimi-K3's
FACTORISED query (`q_lora_rank: 1536`), where `q_a_proj` compresses the
block input to the rank, `q_a_layernorm` normalises over the rank, and
`q_b_proj` expands back to `Hq*q_head_dim` (`modeling_kimi_linear.py`
L364-372, L419). The arm is UNGATED on purpose — it isolates the query
change and nothing else — and its `equivalence` block shows directly
that the output gate is query-independent rather than leaving that to be
assumed.

    python scripts/kimi_mla_oracle_export.py --q-lora > \
        crates/larql-vindex/src/format/vindex3/opplan/exec/tests/kimi_mla_oracle_q_lora.json

The two pre-existing arms change by exactly ONE key: the boundary
`q_proj` is renamed `q_states`, because under the factorised form
nothing computes a `q_proj` and `q_states` is the reference's own name
for the query leaving either form (L419/421). Every VALUE they export is
unchanged, which is checkable by regenerating them and diffing.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

import kimi_mla_reference as ref

SEED = 20260828
POSITIONS = 3

#: Small enough to commit, large enough that heads, nope/rope split and
#: the asymmetric v_head_dim are all genuinely exercised. Every width
#: below is DIFFERENT from every other (`hidden` included) so a
#: transposed axis or swapped slice is NOT invisible here the way
#: `kda_oracle.json`'s own doc warns a too-small fixture can be — see
#: `mla_parity.rs::geometry_matches_kimis_ratios_not_a_symmetric_placeholder`.
GEOMETRY = ref.MlaGeometry(
    num_heads=2, kv_lora_rank=6, qk_nope_head_dim=3, qk_rope_head_dim=4, v_head_dim=5,
)
HIDDEN = 7

#: Scale of the output gate's rows — chosen so the pre-activations sit
#: inside `GATE_BAND` on this fixture, and checked rather than hoped.
OUTPUT_GATE_SCALE = 1.5

#: The non-saturation band (K3-REP-GATE-1 freeze, D10): |g| <= 4 keeps
#: sigmoid inside [0.018, 0.982]; every head reaches |g| >= 0.5 somewhere.
GATE_BAND = {"max_abs": 4.0, "min_head_max_abs": 0.5}

#: Floor under a control's relative-L2 delta on `output`; below it the
#: control is inert on this fixture and is reported, never listed as caught.
CONTROL_FLOOR = 1e-3


def lst(t: torch.Tensor) -> list:
    return [round(v, 8) for v in t.detach().flatten().tolist()]


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def gate_band(pre: torch.Tensor, heads: int, v_hd: int) -> dict:
    """Band on the `[T, Hq*v_head_dim]` pre-activations, per head."""
    per_head = [float(pre[:, h * v_hd:(h + 1) * v_hd].abs().max()) for h in range(heads)]
    measured = {"max_abs": float(pre.abs().max()), "per_head_max_abs": per_head}
    assert measured["max_abs"] <= GATE_BAND["max_abs"], f"gate saturated: {measured}"
    assert min(per_head) >= GATE_BAND["min_head_max_abs"], f"gate near-constant: {measured}"
    return {**measured, "limits": GATE_BAND}


def per_position(result: dict, name: str, positions: int) -> list:
    if name == "attn_weights":
        return [lst(result[name][:, p, : p + 1]) for p in range(positions)]
    return [lst(result[name][p]) for p in range(positions)]


#: The factorised query's rank. Distinct from EVERY other width in this
#: fixture (`num_heads` 2, `nope` 3, `rope` 4, `v_head_dim` 5,
#: `kv_lora_rank` 6, `hidden` 7, `Hq*q_head_dim` 14) so a transposed axis
#: or a borrowed width is not invisible here. In particular `rank !=
#: hidden`, which makes `norm_before_q_a` — the norm applied to the block
#: input instead of to `q_a` — structurally inexpressible rather than
#: numerically caught; that is recorded as unreachable, never as covered.
Q_LORA_RANK = 9

#: Scale of `q_a_proj`'s rows. Chosen, and asserted below, so that
#: `mean(q_a^2)` lands in `Q_A_BAND` — the band where the epsilon is
#: MATERIAL. At variance ~1e-4 the difference between 1e-6 and 1e-5 moves
#: `rsqrt` by ~0.5%; at variance 1.0 it moves it by ~5e-6, and the
#: `layer_eps_for_q_a` control would read as noise on any fixture — a
#: blind instrument rather than a passing test.
Q_A_PROJ_SCALE = 0.04

#: `mean(q_a^2)` must sit here for the epsilon control to be readable.
#: Stated as a band on the SIGNAL, not on the control's delta, so the
#: fixture is judged on whether it can see the fact rather than on
#: whether it produced the answer wanted.
Q_A_BAND = {"min_mean_square": 1e-5, "max_mean_square": 1e-2}

#: Attention must not be saturated before any query control is scored: a
#: softmax pinned at 1.0 makes every upstream query defect invisible
#: downstream.
#:
#: Measured over rows with at least TWO visible positions only. Row 0 of
#: a causal softmax has one visible position and its weight is therefore
#: exactly 1.0 by construction — including it would make this band
#: unsatisfiable on every fixture, which is the shape of gate that gets
#: reinterpreted rather than met.
ATTENTION_BAND = {"max_weight": 0.98, "measured_over": "query positions >= 1"}

#: Which boundary each query mutation is CAUGHT at. The floor is applied
#: there rather than at `output`, because a defect that is real and small
#: at its own boundary is still a defect — and scoring everything at
#: `output` is how a true fact gets recorded as inert.
CAUGHT_AT = {
    "q_a_norm_omitted": "q_a_normed",
    "layer_eps_for_q_a": "q_a_normed",
    "q_b_fed_pre_norm": "q_b",
    "split_rope_first": "q_pass",
    "split_flat": "q_pass",
}

#: Boundaries no query mutation may move — the whole non-query side of
#: the block. Asserted bit-identical under every query control.
NON_QUERY_BOUNDARIES = ("compressed_kv", "kv_a_normed", "kv_b")


def q_lora_weights(g: ref.MlaGeometry, rank: int) -> ref.QLoraWeights:
    return ref.QLoraWeights(
        q_a_proj=torch.randn(rank, HIDDEN) * Q_A_PROJ_SCALE,
        q_a_norm=torch.rand(rank) * 0.5 + 0.75,  # near 1.0, never zero
        q_b_proj=torch.randn(g.num_heads * g.q_head_dim, rank) * 0.3,
        rank=rank,
    )


def bands(result: dict) -> dict:
    """Measure and ASSERT the two bands this arm's controls depend on,
    before any control is scored."""
    mean_square = float(result["q_a"].pow(2).mean())
    assert Q_A_BAND["min_mean_square"] <= mean_square <= Q_A_BAND["max_mean_square"], (
        f"mean(q_a^2) = {mean_square:.3e} is outside {Q_A_BAND}; the q-A epsilon is not "
        "material on this fixture and `layer_eps_for_q_a` would read as noise"
    )
    weights = result["attn_weights"]  # [h, T, T], causal
    spreadable = [
        float(weights[:, p, : p + 1].max()) for p in range(1, weights.shape[1])
    ]
    assert spreadable, "a one-position fixture cannot witness attention spread at all"
    max_weight = max(spreadable)
    assert max_weight <= ATTENTION_BAND["max_weight"], (
        f"attention saturated at {max_weight:.4f}; every query defect below it would be "
        "invisible at `attn_value` and `output`"
    )
    return {
        "q_a_mean_square": mean_square,
        "q_a_limits": Q_A_BAND,
        "attn_max_weight": max_weight,
        "attn_limits": ATTENTION_BAND,
    }


def operand_substitutions(
    g: ref.MlaGeometry,
    w: ref.MlaWeights,
    ql: ref.QLoraWeights,
    x: torch.Tensor,
    result: dict,
) -> dict:
    """Controls that need different OPERANDS rather than a different code
    path, run through the same unmodified forward.

    `q_b_columns_from_hidden` is the column-count trap made numeric: a
    `q_b` shaped `[Hq*q_head_dim, hidden]` fed the block input directly.
    It has the same ROW count as the real one — which is the trap — and
    only its column count says it is a different operand.
    """
    out = {}

    independent = ref.MlaWeights(
        q_proj=torch.randn(g.num_heads * g.q_head_dim, HIDDEN) * 0.3,
        kv_a_proj=w.kv_a_proj,
        kv_a_norm=w.kv_a_norm,
        kv_b_proj=w.kv_b_proj,
        o_proj=w.o_proj,
    )
    substituted = ref.mla_forward(x, independent, g, query_form="direct")
    out["q_proj_substituted"] = {
        "q_proj": lst(independent.q_proj),
        "delta_rel_l2": {
            n: rel_l2(substituted[n], result[n]) for n in ("q_states", "output")
        },
    }

    wrong = q_lora_weights(g, Q_LORA_RANK + 3)
    at_wrong_rank = ref.mla_forward(x, w, g, query_form="low_rank", q_lora=wrong)
    out["wrong_rank"] = {
        "rank": wrong.rank,
        "delta_rel_l2": {
            n: rel_l2(at_wrong_rank[n], result[n]) for n in ("q_states", "output")
        },
    }

    columns_wrong = ref.MlaWeights(
        q_proj=torch.randn(g.num_heads * g.q_head_dim, HIDDEN) * 0.3,
        kv_a_proj=w.kv_a_proj,
        kv_a_norm=w.kv_a_norm,
        kv_b_proj=w.kv_b_proj,
        o_proj=w.o_proj,
    )
    from_hidden = ref.mla_forward(x, columns_wrong, g, query_form="direct")
    out["q_b_columns_from_hidden"] = {
        "note": (
            "a q_b with `hidden` columns instead of `rank`, consuming the block input "
            "directly. Same ROW count as the real q_b — that is the trap — and a "
            "different operation entirely."
        ),
        "delta_rel_l2": {
            n: rel_l2(from_hidden[n], result[n]) for n in ("q_states", "output")
        },
    }

    for name, arm in out.items():
        assert arm["delta_rel_l2"]["output"] >= CONTROL_FLOOR, (
            f"operand substitution {name} is inert on this fixture"
        )
    return out


def equivalence(
    g: ref.MlaGeometry,
    w: ref.MlaWeights,
    ql: ref.QLoraWeights,
    x: torch.Tensor,
    low: dict,
) -> dict:
    """The P14 arm: a `q_proj` constructed so the DIRECT form yields this
    fixture's low-rank `q_states`, then everything else compared.

    `W = S @ pinv(X)` with `X = x.T` `[hidden, T]` and `S = q_states.T`
    reproduces `S` exactly when `X` has full column rank, which needs
    `T <= hidden` — 3 <= 7 here, asserted.

    What this can and cannot claim, stated rather than blurred: the
    KV-side boundaries and the output gate read the block input alone and
    are therefore BIT-identical between the forms, which is asserted as
    bit equality. `q_states` agrees only to the residual of a
    pseudoinverse solve, so everything downstream of it — attention,
    value, output — agrees to that residual propagated, which is MEASURED
    and exported rather than asserted as a bit equality it does not have.
    """
    assert x.shape[0] <= HIDDEN, "the solve needs at least as many columns as positions"
    q_proj = low["q_states"].T @ torch.linalg.pinv(x.T)  # [Hq*q_head_dim, hidden]
    direct_w = ref.MlaWeights(
        q_proj=q_proj,
        kv_a_proj=w.kv_a_proj,
        kv_a_norm=w.kv_a_norm,
        kv_b_proj=w.kv_b_proj,
        o_proj=w.o_proj,
    )
    direct = ref.mla_forward(x, direct_w, g, query_form="direct")

    for name in NON_QUERY_BOUNDARIES:
        assert torch.equal(direct[name], low[name]), (
            f"{name} differs between the query forms — it reads the block input alone "
            "and cannot depend on how the query was built"
        )
    # The same with the output gate on, so the gate is shown to be
    # query-independent too. K3's real MLA layer carries both.
    g_proj = torch.randn(g.num_heads * g.v_head_dim, HIDDEN) * OUTPUT_GATE_SCALE
    gated_low = ref.mla_forward(x, w, g, query_form="low_rank", q_lora=ql, g_proj=g_proj)
    gated_direct = ref.mla_forward(x, direct_w, g, query_form="direct", g_proj=g_proj)
    assert torch.equal(gated_low["output_gate"], gated_direct["output_gate"]), (
        "the output gate differs between the query forms — it reads the block input"
    )

    return {
        "q_proj": lst(q_proj),
        "note": (
            "a q_proj solved so the direct form reproduces this arm's q_states. The KV "
            "boundaries and the output gate are bit-identical across the forms; the "
            "query-dependent ones agree to the solve's residual, measured below."
        ),
        "bit_identical": list(NON_QUERY_BOUNDARIES) + ["output_gate"],
        "residual_rel_l2": {
            name: rel_l2(direct[name], low[name])
            for name in ("q_states", "attn_weights", "attn_value", "output")
        },
    }


def export_q_lora(g: ref.MlaGeometry, base: ref.MlaWeights, x: torch.Tensor) -> int:
    """The third arm: Kimi-K3's factorised query, ungated.

    Ungated on purpose — this arm isolates the query change and nothing
    else. K3's real MLA layer carries the output gate as well, and the
    two are independent because the gate reads the block input rather
    than the query; `equivalence` shows that directly instead of leaving
    it to be assumed.
    """
    ql = q_lora_weights(g, Q_LORA_RANK)
    # The low-rank form has NO `q_proj`: the reference's `__init__` is an
    # if/else and never constructs one (L364-376).
    w = ref.MlaWeights(
        q_proj=None,
        kv_a_proj=base.kv_a_proj,
        kv_a_norm=base.kv_a_norm,
        kv_b_proj=base.kv_b_proj,
        o_proj=base.o_proj,
    )
    result = ref.mla_forward(x, w, g, query_form="low_rank", q_lora=ql)

    fixture = {
        "num_heads": g.num_heads,
        "kv_lora_rank": g.kv_lora_rank,
        "qk_nope_head_dim": g.qk_nope_head_dim,
        "qk_rope_head_dim": g.qk_rope_head_dim,
        "v_head_dim": g.v_head_dim,
        "hidden": HIDDEN,
        "q_lora_rank": Q_LORA_RANK,
        "kv_a_norm_eps": ref.KV_A_NORM_EPS,
        # Its OWN key, equal to the kv one and never derived from it.
        "q_a_norm_eps": ref.Q_A_NORM_EPS,
        "layer_norm_eps_not_used_by_q_a": ref.LAYER_NORM_EPS,
        "weights": {
            "q_a_proj": lst(ql.q_a_proj),
            "q_a_norm": lst(ql.q_a_norm),
            "q_b_proj": lst(ql.q_b_proj),
            "kv_a_proj": lst(w.kv_a_proj),
            "kv_a_norm": lst(w.kv_a_norm),
            "kv_b_proj": lst(w.kv_b_proj),
            "o_proj": lst(w.o_proj),
        },
        "positions": POSITIONS,
        "input": [lst(x[p]) for p in range(POSITIONS)],
        "bands": bands(result),
        "boundaries": {
            name: per_position(result, name, POSITIONS) for name in ref.Q_LORA_BOUNDARIES
        },
        "structurally_unreachable": {
            "norm_before_q_a": (
                f"needs hidden == rank to be expressible; this fixture is {HIDDEN} != "
                f"{Q_LORA_RANK} on purpose. Recorded as unreachable, never as covered."
            )
        },
    }

    controls = {}
    for mutation in ref.QUERY_MUTATIONS:
        if mutation == "none":
            continue
        mutant = ref.mla_forward(
            x, w, g, query_form="low_rank", q_lora=ql, query_mutation=mutation
        )
        for name in NON_QUERY_BOUNDARIES:
            assert torch.equal(mutant[name], result[name]), (
                f"query control {mutation} moved {name}, which the query cannot reach"
            )
        deltas = {
            name: rel_l2(mutant[name], result[name]) for name in ref.Q_LORA_BOUNDARIES
        }
        at = CAUGHT_AT[mutation]
        inert = deltas[at] < CONTROL_FLOOR
        print(
            f"query control {mutation:20s} caught at {at:12s} {deltas[at]:.3e}"
            f"  output {deltas['output']:.3e}{'  INERT' if inert else ''}",
            file=sys.stderr,
        )
        controls[mutation] = {
            "caught_at": at,
            "delta_rel_l2": deltas,
            "inert_at_its_own_boundary": inert,
            "boundaries": {
                name: per_position(mutant, name, POSITIONS)
                for name in ref.Q_LORA_BOUNDARIES
            },
        }
        if mutation == "q_b_fed_pre_norm":
            # The trace property, asserted rather than described: the norm
            # is REPORTED unchanged and only its consumer changed.
            assert deltas["q_a_normed"] == 0.0, (
                "q_b_fed_pre_norm must leave the reported q_a_normed untouched — that "
                "is the whole control"
            )
            omitted = ref.mla_forward(
                x,
                w,
                g,
                query_form="low_rank",
                q_lora=ql,
                query_mutation="q_a_norm_omitted",
            )
            assert torch.equal(mutant["output"], omitted["output"]), (
                "q_b_fed_pre_norm and q_a_norm_omitted must be output-identical; if "
                "they are not, one of them is not the defect it claims to be"
            )
            controls[mutation]["output_identical_to"] = "q_a_norm_omitted"

    assert not any(c["inert_at_its_own_boundary"] for c in controls.values()), (
        "a control inert at its own boundary cannot be listed as a control"
    )
    fixture["controls"] = controls
    fixture["operand_substitutions"] = operand_substitutions(g, w, ql, x, result)
    fixture["equivalence"] = equivalence(g, w, ql, x, result)
    print(json.dumps(fixture, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-gate",
        action="store_true",
        help="export the K3 output-gate arm instead of Kimi Linear's ungated one",
    )
    parser.add_argument(
        "--q-lora",
        action="store_true",
        help="export the K3 factorised-query arm (q_a_proj -> q_a_layernorm -> q_b_proj)",
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    g = GEOMETRY
    w = ref.MlaWeights(
        q_proj=torch.randn(g.num_heads * g.q_head_dim, HIDDEN) * 0.3,
        kv_a_proj=torch.randn(g.kv_lora_rank + g.qk_rope_head_dim, HIDDEN) * 0.3,
        kv_a_norm=torch.rand(g.kv_lora_rank) * 0.5 + 0.75,  # near 1.0, never zero
        kv_b_proj=torch.randn(g.num_heads * (g.qk_nope_head_dim + g.v_head_dim), g.kv_lora_rank) * 0.3,
        o_proj=torch.randn(HIDDEN, g.num_heads * g.v_head_dim) * 0.3,
    )
    x = torch.randn(POSITIONS, HIDDEN) * 0.2

    # The gated arm draws its gate AFTER everything the ungated arm draws,
    # so every shared operand and the input are bit-identical across arms.
    g_proj = None
    if args.output_gate:
        g_proj = torch.randn(g.num_heads * g.v_head_dim, HIDDEN) * OUTPUT_GATE_SCALE
    if args.q_lora:
        return export_q_lora(g, w, x)
    result = ref.mla_forward(x, w, g, query_form="direct", g_proj=g_proj)

    fixture = {
        "num_heads": g.num_heads,
        "kv_lora_rank": g.kv_lora_rank,
        "qk_nope_head_dim": g.qk_nope_head_dim,
        "qk_rope_head_dim": g.qk_rope_head_dim,
        "v_head_dim": g.v_head_dim,
        "hidden": HIDDEN,
        "kv_a_norm_eps": ref.KV_A_NORM_EPS,
        "weights": {
            "q_proj": lst(w.q_proj), "kv_a_proj": lst(w.kv_a_proj),
            "kv_a_norm": lst(w.kv_a_norm), "kv_b_proj": lst(w.kv_b_proj),
            "o_proj": lst(w.o_proj),
        },
        "positions": POSITIONS,
        "input": [lst(x[p]) for p in range(POSITIONS)],
        # Per-position boundaries. `attn_weights`/`attn_value`/`output` at
        # position p depend on positions [0..=p] (causal) — sliced from
        # the full-sequence run, exactly what step-by-step KV-cache
        # decoding must reproduce.
        "boundaries": {
            "q_states": [lst(result["q_states"][p]) for p in range(POSITIONS)],
            "compressed_kv": [lst(result["compressed_kv"][p]) for p in range(POSITIONS)],
            "kv_a_normed": [lst(result["kv_a_normed"][p]) for p in range(POSITIONS)],
            "kv_b": [lst(result["kv_b"][p]) for p in range(POSITIONS)],
            # weights[h, p, 0..=p] flattened head-major, causal row only.
            "attn_weights": [
                lst(result["attn_weights"][:, p, : p + 1]) for p in range(POSITIONS)
            ],
            "attn_value": [lst(result["attn_value"][p]) for p in range(POSITIONS)],
            "output": [lst(result["output"][p]) for p in range(POSITIONS)],
        },
    }
    if args.output_gate:
        fixture["weights"]["g_proj"] = lst(g_proj)
        fixture["output_gate"] = True
        fixture["boundaries"] = {
            name: per_position(result, name, POSITIONS) for name in ref.GATED_BOUNDARIES
        }
        # The band is measured and asserted BEFORE any control is scored.
        fixture["gate_band"] = gate_band((x @ g_proj.T).float(), g.num_heads, g.v_head_dim)
        controls = {}
        for mutation in ref.GATE_MUTATIONS:
            if mutation == "none":
                continue
            mutant = ref.mla_forward(
                x, w, g, query_form="direct", g_proj=g_proj, mutation=mutation
            )
            named = ("output_gate", "gated_value", "output")
            deltas = {name: rel_l2(mutant[name], result[name]) for name in named}
            inert = deltas["output"] < CONTROL_FLOOR
            print(
                f"control {mutation:34s} output_gate {deltas['output_gate']:.3e}"
                f"  gated_value {deltas['gated_value']:.3e}  output {deltas['output']:.3e}"
                f"{'  INERT' if inert else ''}",
                file=sys.stderr,
            )
            controls[mutation] = {
                "delta_rel_l2": deltas,
                "inert_on_this_fixture": inert,
                # The mutant's own boundaries, so the Rust mutant can be
                # required to EQUAL the oracle's wrong answer rather than
                # merely differ from the right one.
                "boundaries": {name: per_position(mutant, name, POSITIONS) for name in named},
            }
        assert not any(c["inert_on_this_fixture"] for c in controls.values()), (
            "an inert control cannot be listed as a control"
        )
        fixture["controls"] = controls
    print(json.dumps(fixture, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
