#!/usr/bin/env python3
"""CAR-1A paired carrier effects and one familywide held-out max-t gate."""
from __future__ import annotations

import numpy as np

from gwstate1_analysis import subject_effects_from_logits

CANDIDATES = 19
SEED = 27022033
REPLICATES = 10_000


def effects_from_replay(rows: list[dict], replay: dict, proximal: np.ndarray,
                        terminal: np.ndarray, split: str) -> tuple[list[str], np.ndarray]:
    if replay.get("candidate_ids") != list(range(CANDIDATES)):
        raise ValueError("CAR-1A candidate ID order changed")
    return subject_effects_from_logits(rows, proximal, terminal,
                                       replay["candidate_ids"], split,
                                       replay["row_indices"])


def exact_effect_from_state1(rows: list[dict], replay: dict, proximal: np.ndarray,
                             terminal: np.ndarray, split: str) -> tuple[list[str], np.ndarray]:
    context_ids = replay["context_ids"]
    if 255 not in context_ids:
        raise ValueError("STATE-1 all-natural denominator absent")
    position = context_ids.index(255)
    subjects, values = subject_effects_from_logits(
        rows, proximal[:, position:position + 1], terminal[:, position:position + 1],
        [255], split, replay["row_indices"])
    return subjects, values[:, 0]


def adjudicate_split(candidate_effects: np.ndarray, exact_effects: np.ndarray,
                     seed: int = SEED, replicates: int = REPLICATES) -> dict:
    candidates = np.asarray(candidate_effects, dtype=np.float64)
    exact = np.asarray(exact_effects, dtype=np.float64)
    if (candidates.shape != (15, CANDIDATES, 2, 2)
        or exact.shape != (15, 2, 2)
        or not np.isfinite(candidates).all() or not np.isfinite(exact).all()
        or replicates < 2):
        raise ValueError("invalid CAR-1A held-out subject-effect geometry")
    point = candidates.mean(axis=0)
    denominator = exact.mean(axis=0)[0]
    rng = np.random.default_rng(seed)
    draws = rng.integers(15, size=(replicates, 15))
    weights = np.stack([(draws == i).sum(axis=1) for i in range(15)], axis=1) / 15
    sampled = np.einsum("bs,scmz->bcmz", weights, candidates)
    denominator_draws = np.einsum("bs,sz->bz", weights, exact[:, 0])
    stable = bool(np.all(denominator > 0)
                  and np.all(denominator_draws > 0)
                  and np.all(np.quantile(denominator_draws, 0.025, axis=0) > 0))
    if not stable:
        return {"exact_denominator_stable": False, "candidates": {},
                "max_t_critical": None}
    retention = point[:, 0] / denominator
    retention_draws = sampled[:, :, 0] / denominator_draws[:, None]
    claim_point = np.concatenate((retention.reshape(-1), point[:, 1].reshape(-1)))
    claim_draws = np.concatenate((retention_draws.reshape(replicates, -1),
                                  sampled[:, :, 1].reshape(replicates, -1)), axis=1)
    se = claim_draws.std(axis=0, ddof=1)
    if not np.isfinite(se).all() or np.any(se <= 0):
        raise ValueError("zero or nonfinite CAR-1A simultaneous-family standard error")
    critical = float(np.quantile(((claim_point - claim_draws) / se).max(axis=1), 0.95))
    lower = claim_point - critical * se
    retention_lower = lower[:CANDIDATES * 2].reshape(CANDIDATES, 2)
    terminal_lower = lower[CANDIDATES * 2:].reshape(CANDIDATES, 2)
    results = {}
    for candidate in range(CANDIDATES):
        clears = bool(np.all(retention[candidate] >= 0.80)
                      and np.all(retention_lower[candidate] >= 0.50)
                      and np.all(terminal_lower[candidate] > 0))
        results[str(candidate)] = {
            "proximal_retention": retention[candidate].tolist(),
            "proximal_simultaneous_lower95": retention_lower[candidate].tolist(),
            "terminal_effect": point[candidate, 1].tolist(),
            "terminal_simultaneous_lower95": terminal_lower[candidate].tolist(),
            "gate_pass": clears,
        }
    return {
        "exact_denominator_stable": True,
        "exact_proximal_effect": denominator.tolist(),
        "exact_pointwise_lower95": np.quantile(denominator_draws, 0.025, axis=0).tolist(),
        "candidates": results, "max_t_critical": critical,
        "bootstrap_seed": seed, "bootstrap_replicates": replicates,
    }


def summarize(validation: dict, test: dict, costs: list[dict]) -> dict:
    if len(costs) != CANDIDATES or [item["candidate_id"] for item in costs] != list(range(CANDIDATES)):
        raise ValueError("CAR-1A cost grid changed")
    if not validation["exact_denominator_stable"] or not test["exact_denominator_stable"]:
        return {"status": "invalid_exact_denominator", "clearing_candidates": [],
                "smallest_clearing_candidate": None}
    def clears(candidate: int) -> bool:
        return (validation["candidates"][str(candidate)]["gate_pass"]
                and test["candidates"][str(candidate)]["gate_pass"])
    if not clears(0):
        return {"status": "invalid_exact_context128_control", "clearing_candidates": [],
                "smallest_clearing_candidate": None}
    clearing = [i for i in range(2, CANDIDATES) if clears(i)]
    def total_bytes(candidate: int) -> int:
        row = costs[candidate]
        return row["per_row_bytes"] * 666 + row["shared_model_bytes"]
    smallest = min(clearing, key=lambda i: (total_bytes(i), i)) if clearing else None
    return {"status": "valid", "clearing_candidates": clearing,
            "smallest_clearing_candidate": smallest,
            "smallest_total_bytes_at_666_rows": total_bytes(smallest) if smallest is not None else None,
            "source_requires_natural_prefix": True}
