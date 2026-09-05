#!/usr/bin/env python3
"""Golden values for the attention-residual topology (K3-ATTNRES-1).

A faithful transcription of Kimi-K3's OWN reference,
`modeling_kimi_linear.py` (the TEXT model's file; `modeling_kimi_k3.py`
is the multimodal wrapper and never touches the mechanism):

  * `_apply_attn_res` (~L1075) — the reduction itself: concatenate the
    block snapshots with the prefix sum, RMS-normalise the candidates
    **for scoring only**, dot them against ONE learned score vector
    (`norm.weight * proj.weight.squeeze(0)`), softmax, and mix the RAW
    candidates.
  * `KimiDecoderLayer._forward_attn_residual` (~L973) — the block
    schedule: an attention-site reduce guarded on a NON-EMPTY snapshot
    set, then the boundary snapshot of the ENTERING state with a prefix
    reset, then the sublayer, then an UNCONDITIONAL mlp-site reduce.
  * `KimiLinearModel.forward` (~L1188) and `_apply_output_attn_res`
    (~L1226) — the snapshot set starts EMPTY (`new_zeros(tokens, 0, d)`)
    and the exit reduction runs before the final norm.

Every ordering detail below is OBSERVED in that source, not inferred
from tensor names or from the topology's description.

## What this artefact is for

The traversal transition of K3-ATTNRES-1 freezes against this file. No
Rust arithmetic exists for the topology yet, and none may be written
until an instrument that can REJECT a wrong implementation exists —
otherwise the first implementation becomes its own reference. Each named
ordering mistake therefore ships with a control that is run here and
whose divergence from the reference is measured and recorded, so a
reader can see which defects this oracle can catch and which it cannot.

## The geometry is deliberately awkward

`hidden = 5` is neither a power of two nor equal to any count in the
fixture; `block_size = 3` over `7` layers puts boundaries at layers 0, 3
and 6, so the exit mixes FOUR candidates and there are two boundary
layers after layer 0 — which is what makes the snapshot-source controls
observable at all (at layer 0 they are not; see `_verify`). Candidate
magnitudes span two orders, and every layer carries its own scale, so no
control can coincide with the reference by averaging.

    scripts/attn_res_oracle_export.py            > the oracle json
    scripts/attn_res_oracle_export.py --explain  # the schedule, to stderr

The script asserts every property and every control before it writes
anything; a non-zero exit means the transcription and its own claims
disagree.
"""

from __future__ import annotations

import json
import sys

import torch

# ── Fixture geometry ────────────────────────────────────────────────
# Kimi-K3 itself declares attn_res_block_size = 12 over 93 layers, which
# is the same schedule at a size no test can carry. 3 over 7 keeps the
# THREE structural cases the schedule has: a boundary at layer 0 (no
# snapshots yet), ordinary layers, and boundaries after layer 0 (where
# the old/new snapshot-set distinction becomes observable).
HIDDEN = 5
POSITIONS = 3
LAYERS = 7
BLOCK_SIZE = 3
# K3's own `rms_norm_eps`. It enters the SCORE only.
NORM_EPS = 1e-5
SEED = 20260905

# Per-layer BRANCH scales spanning 20x, so no two layers are the same
# problem and a site that read another layer's operands could not pass by
# looking similar. These set the spread of the CANDIDATES.
LAYER_SCALES = [0.2, 0.5, 1.1, 2.0, 3.4, 0.8, 4.0]

# The score pair's scale, and it is the single most load-bearing number in
# this fixture. `score = sum_h k_h * (norm_h * proj_h)` over an
# RMS-normalised `k`, so |k_h| ~ 1 and the score spread is set entirely by
# `|norm * proj| * hidden`. The first draft drew both factors at the layer
# scales above and produced probabilities of `[1.0, 4.2e-11]` — a softmax
# saturated to one-hot, on which EVERY candidate-set control is invisible:
# adding or removing a candidate that carries 4e-11 of the mass changes
# nothing a comparison can see. The oracle looked like it was rejecting
# nothing because the mechanism was right, when in fact it had been
# blinded. `_verify` now asserts non-degeneracy so a regenerated fixture
# cannot lose the property silently.
SCORE_NORM_JITTER = 0.3
SCORE_PROJ_SCALE = 0.15
# No site's distribution may be flatter than this or more peaked than
# this; outside the band a control can hide.
MIN_PROB_FLOOR = 5e-3
MAX_PROB_CEILING = 0.98

# ── Mutations: one per named ordering mistake ───────────────────────
NONE = "none"
ATTN_SITE_OVER_NEW_SNAPSHOTS = "attn_site_over_new_snapshots"
SNAPSHOT_IS_MIXED_VECTOR = "snapshot_is_mixed_vector"
SNAPSHOT_AFTER_ATTENTION = "snapshot_after_attention"
LAYER0_ATTENTION_SITE_RUNS = "layer0_attention_site_runs"
MLP_SITE_SKIPPED_AT_LAYER_0 = "mlp_site_skipped_at_layer_0"
MLP_SITE_GUARDED_ON_NONEMPTY = "mlp_site_guarded_on_nonempty"
MIX_OVER_NORMALISED_CANDIDATES = "mix_over_normalised_candidates"
SCORE_WITHOUT_RMSNORM = "score_without_rmsnorm"
EXIT_SKIPPED = "exit_skipped"
EXIT_USES_A_LAYER_PAIR = "exit_uses_a_layer_pair"


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """`KimiRMSNorm.forward`, transcribed: f32 internally, cast back."""
    v = x.float()
    variance = v.pow(2).mean(-1, keepdim=True)
    return (v * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    mix_over_normalised: bool = False,
    score_without_rmsnorm: bool = False,
):
    """`_apply_attn_res`, line for line.

    `prefix_sum` is `[tokens, hidden]`, `block_residual` is
    `[tokens, blocks, hidden]`. Returns the mixed vector and the probs,
    the second only so the witness can record what the reference does
    not return.

    The two keyword arguments are the CONTROLS for the two things this
    function decides and a transcription could get wrong: which tensor
    the probabilities are applied to, and whether the score sees the
    normalised candidates. Both default to the reference.
    """
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + eps)
    score_weight = norm_weight.float() * proj_weight.squeeze(0).float()
    scored = v_float if score_without_rmsnorm else k
    scores = (scored * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    # The reference mixes `v_float` — the RAW candidates. `k` is the
    # scoring form and nothing else reads it.
    mixed_over = k if mix_over_normalised else v_float
    hidden_states = torch.matmul(probs, mixed_over).squeeze(1)
    return hidden_states.to(v.dtype), probs.squeeze(1)


def branch(x: torch.Tensor, norm_weight: torch.Tensor, weight: torch.Tensor, eps: float):
    """The sublayer stand-in: pre-norm, then a linear map, then tanh.

    What the branch COMPUTES is not what this oracle is about; what
    matters is that it reads the MIXED vector and returns `[hidden]`, so
    a traversal that fed it the prefix sum instead diverges here.
    """
    return torch.tanh(rms_norm(x, norm_weight, eps) @ weight.T)


def run_stack(w: dict, x0: torch.Tensor, mutation: str = NONE):
    """The whole schedule, with one mutation applied.

    Transcribed from `KimiLinearModel.forward` and
    `KimiDecoderLayer._forward_attn_residual`. Every branch below is the
    reference's own; the mutation arms are marked.
    """
    tokens = x0.shape[0]
    # `hidden_states.new_zeros(tokens, 0, hidden)` — the set starts EMPTY.
    blocks = x0.new_zeros(tokens, 0, HIDDEN)
    hidden_states = x0
    witness = []

    for layer in range(LAYERS):
        lw = w["layers"][layer]
        record = {"layer": layer, "prefix_in": hidden_states.clone()}
        prefix_sum = hidden_states
        is_boundary = layer % BLOCK_SIZE == 0
        record["is_boundary"] = is_boundary
        record["snapshots_before"] = int(blocks.shape[1])

        # --- MUTATION: append the boundary snapshot BEFORE the
        # --- attention-site reduce, so that reduce sees the NEW set.
        if mutation == ATTN_SITE_OVER_NEW_SNAPSHOTS and is_boundary:
            blocks = torch.cat([blocks, prefix_sum.unsqueeze(1)], dim=1)

        # The attention site. Guarded on a NON-EMPTY snapshot set, which
        # is why layer 0 has no attention-site reduction at all.
        run_attention_site = blocks.shape[1] > 0
        if mutation == LAYER0_ATTENTION_SITE_RUNS:
            # --- MUTATION: drop the guard; the site always runs.
            run_attention_site = True
        if run_attention_site:
            hidden_states, probs = apply_attn_res(
                prefix_sum,
                blocks,
                lw["attn_res_proj"],
                lw["attn_res_norm"],
                NORM_EPS,
                mix_over_normalised=(mutation == MIX_OVER_NORMALISED_CANDIDATES),
                score_without_rmsnorm=(mutation == SCORE_WITHOUT_RMSNORM),
            )
            record["attention_site"] = {
                "ran": True,
                "candidate_count": int(blocks.shape[1] + 1),
                "snapshots_reduced_over": int(blocks.shape[1]),
                "softmax_probs": probs.clone(),
                "mixed_vector": hidden_states.clone(),
            }
        else:
            record["attention_site"] = {"ran": False, "candidate_count": 0}

        # The boundary event: snapshot the ENTERING state, reset the prefix.
        if is_boundary and mutation != ATTN_SITE_OVER_NEW_SNAPSHOTS:
            snapshot = prefix_sum
            if mutation == SNAPSHOT_IS_MIXED_VECTOR:
                # --- MUTATION: snapshot the attention site's OUTPUT.
                snapshot = hidden_states
            if mutation != SNAPSHOT_AFTER_ATTENTION:
                blocks = torch.cat([blocks, snapshot.unsqueeze(1)], dim=1)
                record["snapshot_event"] = {
                    "taken": True,
                    "source": "entering_prefix_state",
                    "value": snapshot.clone(),
                }
        if is_boundary:
            prefix_sum = None
        record.setdefault("snapshot_event", {"taken": is_boundary, "source": "mutated"})
        if not is_boundary:
            record["snapshot_event"] = {"taken": False}

        # The attention sublayer, on the MIXED vector.
        attn_out = branch(hidden_states, lw["input_norm"], lw["attn_weight"], NORM_EPS)
        record["attention_site"]["branch_output"] = attn_out.clone()
        prefix_sum = attn_out if prefix_sum is None else prefix_sum + attn_out

        # --- MUTATION: snapshot AFTER the attention instead of before it.
        if mutation == SNAPSHOT_AFTER_ATTENTION and is_boundary:
            blocks = torch.cat([blocks, prefix_sum.unsqueeze(1)], dim=1)
            record["snapshot_event"] = {
                "taken": True,
                "source": "post_attention_prefix",
                "value": prefix_sum.clone(),
            }

        record["attention_site"]["prefix_out"] = prefix_sum.clone()
        record["snapshots_after"] = int(blocks.shape[1])

        # The mlp site. UNCONDITIONAL in the reference — no guard at all.
        run_mlp_site = True
        if mutation == MLP_SITE_SKIPPED_AT_LAYER_0 and layer == 0:
            # --- MUTATION: a reader who thought the layer-0 site is skipped.
            run_mlp_site = False
        if mutation == MLP_SITE_GUARDED_ON_NONEMPTY:
            # --- MUTATION: give the mlp site the attention site's guard.
            run_mlp_site = blocks.shape[1] > 0
        mlp_prefix_in = prefix_sum
        if run_mlp_site:
            hidden_states, probs = apply_attn_res(
                prefix_sum,
                blocks,
                lw["mlp_res_proj"],
                lw["mlp_res_norm"],
                NORM_EPS,
                mix_over_normalised=(mutation == MIX_OVER_NORMALISED_CANDIDATES),
                score_without_rmsnorm=(mutation == SCORE_WITHOUT_RMSNORM),
            )
            record["mlp_site"] = {
                "ran": True,
                "candidate_count": int(blocks.shape[1] + 1),
                "snapshots_reduced_over": int(blocks.shape[1]),
                "softmax_probs": probs.clone(),
                "mixed_vector": hidden_states.clone(),
            }
        else:
            hidden_states = prefix_sum
            record["mlp_site"] = {"ran": False, "candidate_count": 0}
        record["mlp_site"]["prefix_in"] = mlp_prefix_in.clone()

        mlp_out = branch(hidden_states, lw["post_attention_norm"], lw["mlp_weight"], NORM_EPS)
        record["mlp_site"]["branch_output"] = mlp_out.clone()
        prefix_sum = prefix_sum + mlp_out
        record["mlp_site"]["prefix_out"] = prefix_sum.clone()

        hidden_states = prefix_sum
        witness.append(record)

    # The exit. Required under the declaration, and it reads the SHIPPED
    # output pair.
    exit_record = {"snapshots": int(blocks.shape[1])}
    if mutation == EXIT_SKIPPED:
        # --- MUTATION: no exit reduction at all.
        final = hidden_states
        exit_record["ran"] = False
    else:
        pair = w["exit"]
        if mutation == EXIT_USES_A_LAYER_PAIR:
            # --- MUTATION: the exit reads layer 0's mlp pair instead of
            # --- the model-level output pair the checkpoint ships.
            pair = {
                "proj": w["layers"][0]["mlp_res_proj"],
                "norm": w["layers"][0]["mlp_res_norm"],
            }
        final, probs = apply_attn_res(
            hidden_states,
            blocks,
            pair["proj"],
            pair["norm"],
            NORM_EPS,
            mix_over_normalised=(mutation == MIX_OVER_NORMALISED_CANDIDATES),
            score_without_rmsnorm=(mutation == SCORE_WITHOUT_RMSNORM),
        )
        exit_record.update(
            {
                "ran": True,
                "candidate_count": int(blocks.shape[1] + 1),
                "softmax_probs": probs.clone(),
                "mixed_vector": final.clone(),
            }
        )
    exit_record["prefix_in"] = hidden_states.clone()

    return {"witness": witness, "exit": exit_record, "final": final, "snapshots": blocks.clone()}


def delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max())


def _verify(w: dict, x0: torch.Tensor, ref: dict) -> dict:
    """Prove the six properties, and measure every control.

    Raises on any disagreement: this artefact must not be able to ship
    while its own claims are false.
    """
    witness, controls = ref["witness"], {}

    def run(m):
        return run_stack(w, x0, m)

    # --- 0. the instrument can see at all --------------------------
    # Checked FIRST, because every property below is scored by comparing
    # outputs and a saturated softmax makes those comparisons blind. A
    # distribution that has collapsed onto one candidate carries no
    # information about the others, so adding, removing or reordering
    # them changes nothing measurable — and the oracle then reports every
    # control as passing while rejecting none of them. Recorded as an
    # assertion rather than a comment because it is a property of the
    # FIXTURE, and a regeneration could lose it without any other test
    # noticing.
    extremes = []
    for rec in witness:
        for key in ("attention_site", "mlp_site"):
            s = rec[key]
            if s["ran"]:
                extremes.append((rec["layer"], key, s["softmax_probs"]))
    extremes.append((LAYERS, "exit", ref["exit"]["softmax_probs"]))
    worst_max = max(float(p.max()) for _, _, p in extremes)
    worst_min = min(float(p.min()) for _, _, p in extremes)
    assert worst_max <= MAX_PROB_CEILING, ("softmax saturated", worst_max)
    assert worst_min >= MIN_PROB_FLOOR, ("softmax starved", worst_min)

    # --- 1. the attention site reduces over the OLD snapshot set ----
    for rec in witness:
        if rec["is_boundary"] and rec["attention_site"]["ran"]:
            assert (
                rec["attention_site"]["snapshots_reduced_over"] == rec["snapshots_before"]
            ), rec["layer"]
            assert rec["snapshots_after"] == rec["snapshots_before"] + 1, rec["layer"]
    boundary_after_zero = [r for r in witness if r["is_boundary"] and r["layer"] > 0]
    assert boundary_after_zero, "the fixture must have a boundary after layer 0"
    controls[ATTN_SITE_OVER_NEW_SNAPSHOTS] = run(ATTN_SITE_OVER_NEW_SNAPSHOTS)

    # --- 2. the boundary snapshot is the ENTERING prefix state ------
    snaps = [r for r in witness if r["snapshot_event"]["taken"]]
    assert len(snaps) == len(range(0, LAYERS, BLOCK_SIZE)), len(snaps)
    for rec in snaps:
        assert torch.equal(rec["snapshot_event"]["value"], rec["prefix_in"]), rec["layer"]
    controls[SNAPSHOT_IS_MIXED_VECTOR] = run(SNAPSHOT_IS_MIXED_VECTOR)
    controls[SNAPSHOT_AFTER_ATTENTION] = run(SNAPSHOT_AFTER_ATTENTION)

    # --- 3. layer 0 skips the attention-site reduction ---------------
    assert witness[0]["attention_site"]["ran"] is False
    assert witness[0]["attention_site"]["candidate_count"] == 0
    for rec in witness[1:]:
        assert rec["attention_site"]["ran"] is True, rec["layer"]
    controls[LAYER0_ATTENTION_SITE_RUNS] = run(LAYER0_ATTENTION_SITE_RUNS)

    # --- 4. layer 0 still performs the mlp-site reduction ------------
    for rec in witness:
        assert rec["mlp_site"]["ran"] is True, rec["layer"]
    controls[MLP_SITE_SKIPPED_AT_LAYER_0] = run(MLP_SITE_SKIPPED_AT_LAYER_0)
    controls[MLP_SITE_GUARDED_ON_NONEMPTY] = run(MLP_SITE_GUARDED_ON_NONEMPTY)

    # --- 5. the mix is over RAW candidates; RMS-norm scores only -----
    controls[MIX_OVER_NORMALISED_CANDIDATES] = run(MIX_OVER_NORMALISED_CANDIDATES)
    controls[SCORE_WITHOUT_RMSNORM] = run(SCORE_WITHOUT_RMSNORM)

    # --- 6. the exit is mandatory and uses the SHIPPED output pair ---
    assert ref["exit"]["ran"] is True
    assert ref["exit"]["candidate_count"] == ref["exit"]["snapshots"] + 1
    controls[EXIT_SKIPPED] = run(EXIT_SKIPPED)
    controls[EXIT_USES_A_LAYER_PAIR] = run(EXIT_USES_A_LAYER_PAIR)

    # Every control's divergence from the reference, measured.
    measured = {name: delta(out["final"], ref["final"]) for name, out in controls.items()}

    # The controls that MUST move the numbers.
    rejecting = [
        ATTN_SITE_OVER_NEW_SNAPSHOTS,
        SNAPSHOT_IS_MIXED_VECTOR,
        SNAPSHOT_AFTER_ATTENTION,
        MLP_SITE_SKIPPED_AT_LAYER_0,
        MIX_OVER_NORMALISED_CANDIDATES,
        SCORE_WITHOUT_RMSNORM,
        EXIT_SKIPPED,
        EXIT_USES_A_LAYER_PAIR,
    ]
    for name in rejecting:
        assert measured[name] > 1e-3, (name, measured[name])

    # The controls that are NUMERICALLY INERT, and must be shown to be.
    # These are not weak controls; they are the honest statement that
    # two of the six properties cannot be caught by comparing outputs,
    # and are witness-structural claims only.
    for name in (LAYER0_ATTENTION_SITE_RUNS, MLP_SITE_GUARDED_ON_NONEMPTY):
        assert measured[name] == 0.0, (name, measured[name])

    # ...and the structural difference each of them DOES make.
    assert controls[LAYER0_ATTENTION_SITE_RUNS]["witness"][0]["attention_site"]["ran"] is True
    assert controls[LAYER0_ATTENTION_SITE_RUNS]["witness"][0]["attention_site"][
        "candidate_count"
    ] == 1
    assert all(
        r["mlp_site"]["ran"] for r in controls[MLP_SITE_GUARDED_ON_NONEMPTY]["witness"]
    ), "the mlp guard never fires: no site in this schedule sees an empty set"

    return {"deltas": measured, "prob_band": [worst_min, worst_max]}


def schedule_table(witness: list, exit_record: dict) -> list[str]:
    """The block schedule as OBSERVED, one line per event in order."""
    lines = []
    for rec in witness:
        layer = rec["layer"]
        a, m = rec["attention_site"], rec["mlp_site"]
        if a["ran"]:
            lines.append(
                f"layer {layer:<2}  attn: reduce over {a['snapshots_reduced_over']} "
                f"snapshot(s) + prefix = {a['candidate_count']} candidates"
            )
        else:
            lines.append(f"layer {layer:<2}  attn: NO reduction (snapshot set empty)")
        if rec["snapshot_event"]["taken"]:
            lines.append(
                f"          snapshot: entering prefix state -> "
                f"{rec['snapshots_before']} becomes {rec['snapshots_after']}; prefix reset"
            )
        lines.append(
            f"          mlp:  reduce over {m['snapshots_reduced_over']} snapshot(s) "
            f"+ prefix = {m['candidate_count']} candidates"
        )
    lines.append(
        f"exit      reduce over {exit_record['snapshots']} snapshot(s) + prefix "
        f"= {exit_record['candidate_count']} candidates, then the final norm"
    )
    return lines


def main() -> None:
    torch.manual_seed(SEED)
    explain = "--explain" in sys.argv

    # Per-position tilt so no two positions are the same problem, and a
    # magnitude spread so a wrong candidate set cannot average its way
    # to the right answer.
    tilt = torch.tensor([0.4, 1.0, 2.6]).view(POSITIONS, 1)
    x0 = torch.randn(POSITIONS, HIDDEN) * tilt

    # The score pairs are drawn SMALL and the branch weights at the layer
    # scales: the first decides whether the softmax can discriminate, the
    # second decides whether the candidates differ enough to be worth
    # discriminating between. Conflating them is what saturated the first
    # draft.
    w = {
        "layers": [
            {
                "attn_res_norm": torch.randn(HIDDEN) * SCORE_NORM_JITTER + 1.0,
                "attn_res_proj": torch.randn(1, HIDDEN) * SCORE_PROJ_SCALE,
                "mlp_res_norm": torch.randn(HIDDEN) * SCORE_NORM_JITTER + 1.0,
                "mlp_res_proj": torch.randn(1, HIDDEN) * SCORE_PROJ_SCALE,
                "input_norm": torch.randn(HIDDEN) * 0.3 + 1.0,
                "post_attention_norm": torch.randn(HIDDEN) * 0.3 + 1.0,
                "attn_weight": torch.randn(HIDDEN, HIDDEN) * s,
                "mlp_weight": torch.randn(HIDDEN, HIDDEN) * (0.8 * s),
            }
            for s in LAYER_SCALES[:LAYERS]
        ],
        "exit": {
            "norm": torch.randn(HIDDEN) * SCORE_NORM_JITTER + 1.0,
            "proj": torch.randn(1, HIDDEN) * SCORE_PROJ_SCALE,
        },
    }

    ref = run_stack(w, x0, NONE)
    verified = _verify(w, x0, ref)
    measured, prob_band = verified["deltas"], verified["prob_band"]

    if explain:
        for line in schedule_table(ref["witness"], ref["exit"]):
            print(line, file=sys.stderr)

    def flat(t) -> list:
        return [round(v, 9) for v in t.double().flatten().tolist()]

    def site(s: dict) -> dict:
        out = {"ran": s["ran"], "candidate_count": s["candidate_count"]}
        if s["ran"]:
            out["snapshots_reduced_over"] = s["snapshots_reduced_over"]
            out["softmax_probs"] = flat(s["softmax_probs"])
            out["mixed_vector"] = flat(s["mixed_vector"])
        if "prefix_in" in s:
            out["prefix_in"] = flat(s["prefix_in"])
        out["branch_output"] = flat(s["branch_output"])
        out["prefix_out"] = flat(s["prefix_out"])
        return out

    doc = {
        "_comment": (
            "Attention-residual topology oracle (K3-ATTNRES-1). Generated by "
            "scripts/attn_res_oracle_export.py, a torch transcription of Kimi-K3's own "
            "modeling_kimi_linear.py: _apply_attn_res (~L1075), "
            "KimiDecoderLayer._forward_attn_residual (~L973) and "
            "KimiLinearModel.forward / _apply_output_attn_res (~L1188, ~L1226). "
            "Per-token shapes are [positions, hidden]; snapshots are "
            "[positions, blocks, hidden]. Regenerate rather than hand-edit."
        ),
        "geometry": {
            "hidden": HIDDEN,
            "positions": POSITIONS,
            "layers": LAYERS,
            "block_size": BLOCK_SIZE,
            "norm_eps": NORM_EPS,
            "seed": SEED,
            "layer_scales": LAYER_SCALES[:LAYERS],
            "boundary_layers": list(range(0, LAYERS, BLOCK_SIZE)),
            "snapshots_at_exit": int(ref["snapshots"].shape[1]),
        },
        "weights": {
            "layers": [
                {
                    "attn_res_norm": flat(lw["attn_res_norm"]),
                    "attn_res_proj": flat(lw["attn_res_proj"]),
                    "mlp_res_norm": flat(lw["mlp_res_norm"]),
                    "mlp_res_proj": flat(lw["mlp_res_proj"]),
                    "input_norm": flat(lw["input_norm"]),
                    "post_attention_norm": flat(lw["post_attention_norm"]),
                    "attn_weight": flat(lw["attn_weight"]),
                    "mlp_weight": flat(lw["mlp_weight"]),
                }
                for lw in w["layers"]
            ],
            "exit": {"norm": flat(w["exit"]["norm"]), "proj": flat(w["exit"]["proj"])},
        },
        "input": {"embedded": flat(x0)},
        "schedule": schedule_table(ref["witness"], ref["exit"]),
        "witness": [
            {
                "layer": rec["layer"],
                "is_boundary": rec["is_boundary"],
                "prefix_in": flat(rec["prefix_in"]),
                "snapshots_before": rec["snapshots_before"],
                "snapshots_after": rec["snapshots_after"],
                "snapshot_event": (
                    {
                        "taken": True,
                        "source": rec["snapshot_event"]["source"],
                        "value": flat(rec["snapshot_event"]["value"]),
                    }
                    if rec["snapshot_event"]["taken"]
                    else {"taken": False}
                ),
                "attention_site": site(rec["attention_site"]),
                "mlp_site": site(rec["mlp_site"]),
            }
            for rec in ref["witness"]
        ],
        "exit": {
            "ran": ref["exit"]["ran"],
            "snapshots": ref["exit"]["snapshots"],
            "candidate_count": ref["exit"]["candidate_count"],
            "prefix_in": flat(ref["exit"]["prefix_in"]),
            "softmax_probs": flat(ref["exit"]["softmax_probs"]),
            "mixed_vector": flat(ref["exit"]["mixed_vector"]),
        },
        "final": flat(ref["final"]),
        "controls": {
            "_comment": (
                "One control per named ordering mistake, each run through the whole "
                "schedule. `max_abs_delta` is against the reference's final vector. "
                "The two at 0.0 are NOT weak controls: they are the honest statement "
                "that those properties cannot be caught by comparing outputs, because "
                "a softmax over one candidate is exactly the identity and the mlp "
                "site's guard never fires. They are witness-structural claims, and "
                "the witness records what each of them changes instead."
            ),
            "max_abs_delta": {name: measured[name] for name in sorted(measured)},
            "numerically_inert": [LAYER0_ATTENTION_SITE_RUNS, MLP_SITE_GUARDED_ON_NONEMPTY],
            "softmax_probability_band": {
                "_comment": (
                    "The smallest and largest probability any site or the exit assigns "
                    "to a candidate. This is the fixture's ability to SEE: on a "
                    "saturated distribution every candidate-set control is invisible, "
                    "because a candidate carrying ~0 of the mass can be added or "
                    "removed without changing the mix. The first draft of this oracle "
                    "read [4.2e-11, 1.0] and rejected nothing while appearing to pass."
                ),
                "min": prob_band[0],
                "max": prob_band[1],
                "floor": MIN_PROB_FLOOR,
                "ceiling": MAX_PROB_CEILING,
            },
        },
    }
    json.dump(doc, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
