"""Pure comparison logic for the HF-equivalence harness (`hf_equivalence.py`).

Kept free of torch and of any LARQL invocation so the method itself is unit
tested (`test_equivalence_metrics.py`) and versioned: any change to what is
computed here, or to the gates below, bumps `HARNESS_VERSION`, so a later
re-run can tell a methodology change from a model regression.

Metric (fixed before any run):
  max |Δ logprob| at a position = max over the FULL vocabulary of
  |log_softmax(a) - log_softmax(b)|, each log_softmax computed from that
  side's own raw logits upcast to float64. Both sides must have the same
  vocabulary width; a difference is refused, never truncated silently.

Ordering gate (fixed before any run):
  The reference ordering is transformers f32's. An arm's top-k at a
  position PASSES when its ordered ids equal the reference's. A rank where
  they differ is a TIE_SWAP when the arm's token there is inside the
  reference's top-(k+1) and its reference logprob is within the arm's tie
  tolerance of the reference token at that rank; any other difference is a
  FAIL. A tie swap is reported, never counted as a pass.
"""

from __future__ import annotations

import numpy as np

HARNESS_VERSION = "1.1.1"

#: Gate: LARQL reference (f32 interpreter) vs transformers f32.
REFERENCE_MAX_ABS_LOGPROB = 1e-3
#: Ranks gated for ordered agreement, and ranks recorded (k + 1, so the
#: margin below the last gated rank is on the record too).
GATED_TOP_K = 5
RECORDED_TOP_K = GATED_TOP_K + 1
#: Near-tie tolerance per arm, in reference-arm logprob units. The
#: reference arm's equals its error gate: a swap below it is not
#: resolvable at that precision. Production's is the documented
#: reduced-precision allowance; it gates ordering only, never error size.
TIE_TOLERANCE = {"reference": REFERENCE_MAX_ABS_LOGPROB, "production": 5e-2}
#: Production error above this is explained in the record, not gated.
PRODUCTION_EXPLAIN_ABOVE = 0.1

#: Informational, never gated: the error restricted to tokens the
#: reference gives at least this logprob. The full-vocabulary maximum is
#: dominated by extreme-tail tokens (logprob below about -50), where f32
#: summation order alone moves the value; this separates that from error
#: on tokens that carry probability mass.
MASS_BEARING_LOGPROB = -20.0

PASS = "pass"
TIE_SWAP = "tie_swap"
FAIL = "fail"


def log_softmax64(logits: np.ndarray) -> np.ndarray:
    """log_softmax over the last axis, in float64."""
    x = np.asarray(logits, dtype=np.float64)
    shifted = x - x.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def max_abs_logprob(a_logits: np.ndarray, b_logits: np.ndarray) -> np.ndarray:
    """Per-position max |Δ logprob| over the full vocabulary.

    Accepts `[vocab]` or `[positions, vocab]`; returns a scalar array or one
    value per position.
    """
    a = np.atleast_2d(a_logits)
    b = np.atleast_2d(b_logits)
    if a.shape != b.shape:
        raise ValueError(f"vocabulary/position shapes differ: {a.shape} vs {b.shape}")
    out = np.abs(log_softmax64(a) - log_softmax64(b)).max(axis=-1)
    return out if np.ndim(a_logits) == 2 else out[0]


def max_abs_logprob_mass_bearing(
    reference_lp: np.ndarray, arm_lp: np.ndarray, floor: float = MASS_BEARING_LOGPROB
) -> float:
    """max |Δ logprob| over tokens whose reference logprob is >= `floor`."""
    keep = np.asarray(reference_lp) >= floor
    return float(np.abs(np.asarray(reference_lp)[keep] - np.asarray(arm_lp)[keep]).max())


def top_k(logprobs: np.ndarray, k: int = RECORDED_TOP_K) -> list[tuple[int, float]]:
    """Ordered `(id, logprob)` for the k largest, ties broken by lower id."""
    lp = np.asarray(logprobs, dtype=np.float64)
    idx = np.lexsort((np.arange(lp.size), -lp))[:k]
    return [(int(i), float(lp[i])) for i in idx]


def order_verdict(
    reference_lp: np.ndarray, arm_lp: np.ndarray, tolerance: float, k: int = GATED_TOP_K
) -> tuple[str, list[dict]]:
    """Ordered top-k agreement of `arm_lp` against `reference_lp`.

    Returns the verdict and one entry per differing rank, so a tie swap or
    failure names the tokens and the reference margin that decided it.
    """
    ref = top_k(reference_lp, k + 1)
    ref_ids = [i for i, _ in ref]
    arm_ids = [i for i, _ in top_k(arm_lp, k)]
    ref_lp = np.asarray(reference_lp, dtype=np.float64)
    diffs = []
    verdict = PASS
    for rank in range(k):
        want, got = ref_ids[rank], arm_ids[rank]
        if want == got:
            continue
        margin = abs(float(ref_lp[want]) - float(ref_lp[got]))
        tie = got in ref_ids and margin < tolerance
        diffs.append(
            {
                "rank": rank + 1,
                "reference_id": want,
                "arm_id": got,
                "reference_margin": margin,
                "status": TIE_SWAP if tie else FAIL,
            }
        )
        if not tie:
            verdict = FAIL
        elif verdict == PASS:
            verdict = TIE_SWAP
    return verdict, diffs


def worst(verdicts) -> str:
    """The most severe of a set of verdicts (fail > tie_swap > pass)."""
    order = {PASS: 0, TIE_SWAP: 1, FAIL: 2}
    return max(verdicts, key=order.__getitem__, default=PASS)


def stage(status: str, evidence) -> dict:
    """One ladder stage: PASS / FAIL / PENDING with its evidence."""
    assert status in ("PASS", "FAIL", "PENDING"), status
    return {"status": status, "evidence": evidence}


def append_run(record: dict | None, run: dict, identity: dict) -> dict:
    """Append `run` to an equivalence record, never rewriting a past run.

    A record is keyed by the checkpoint it speaks for; appending a run for a
    different checkpoint is refused. `ever_failed` names every stage that
    has failed in any run, so a later pass cannot erase a failure.
    """
    if record is None:
        record = {"schema": "larql.equivalence/1", "model": identity, "runs": []}
    elif record["model"] != identity:
        raise ValueError(f"record is for {record['model']}, not {identity}")
    record["runs"].append(run)
    failed = set(record.get("ever_failed", []))
    for name, result in run["ladder"].items():
        if result["status"] == "FAIL":
            failed.add(name)
    record["ever_failed"] = sorted(failed)
    return record
