"""Frozen capital-cohort effect reduction and 104-readout simultaneous gate."""
from __future__ import annotations

import numpy as np

PAIRS = ((11, 8), (12, 8), (13, 8), (11, 17), (12, 18), (13, 19))
SEED = 27022034


def probabilities(logits: np.ndarray, z: bool) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite readout")
    if z:
        scale = x.std(axis=-1, keepdims=True)
        if np.any(scale <= 0):
            raise ValueError("zero-variance z-score readout")
        x = (x - x.mean(axis=-1, keepdims=True)) / scale
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def js(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    middle = (a + b) * 0.5
    def term(p):
        ratio = np.ones_like(p)
        np.divide(p, middle, out=ratio, where=p > 0)
        return np.sum(p * np.log(ratio), axis=-1)
    return (term(a) + term(b)) * 0.5


def group_js(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    a, b, c = (values[groups[:, i]] for i in range(3))
    return (js(a, b) + js(a, c) + js(b, c)) / 3


def effects(rows: list[dict], proximal: np.ndarray, terminal: np.ndarray,
            split: str) -> tuple[list[str], np.ndarray]:
    """Rows are the complete fresh cohort; one capital edge / three prompts per subject."""
    p, t = np.asarray(proximal), np.asarray(terminal)
    if (p.ndim != 4 or p.shape != t.shape or p.shape[0] != len(rows)
        or p.shape[2] != 2 or not np.isfinite(p).all() or not np.isfinite(t).all()):
        raise ValueError("invalid paired readout shape")
    subjects = sorted({r["subject_id"] for r in rows if r["split"] == split})
    if not subjects:
        raise ValueError("empty split")
    groups = [[i for i, r in enumerate(rows) if r["subject_id"] == subject] for subject in subjects]
    if any(len(g) != 3 or {rows[i]["split"] for i in g} != {split}
           or {rows[i]["template"] for i in g} != {"canonical", "question", "alternate"}
           or {rows[i]["relation"] for i in g} != {"capital"} for g in groups):
        raise ValueError("capital population is not three templates per split-isolated subject")
    donors = []
    for group in groups:
        donor_group = []
        for i in group:
            j = rows[i]["donor_index"]
            if (not isinstance(j, int) or not 0 <= j < len(rows) or j == i
                or rows[j]["subject_id"] == rows[i]["subject_id"]
                or rows[j]["split"] != split or rows[j]["template"] != rows[i]["template"]):
                raise ValueError("invalid matched control")
            donor_group.append(j)
        donors.append(donor_group)
    groups, donors = np.array(groups), np.array(donors)
    result = np.empty((len(subjects), p.shape[1], 2, 2))
    for surface, logits in enumerate((p, t)):
        for z in range(2):
            values = probabilities(logits, bool(z))
            same, control = group_js(values, groups), group_js(values, donors)
            result[:, :, surface, z] = same[:, :, 0] - same[:, :, 1] - control[:, :, 0] + control[:, :, 1]
    return subjects, result


def adjudicate(values: np.ndarray, reference: np.ndarray, replicates: int = 10000) -> dict:
    values, reference = np.asarray(values, dtype=float), np.asarray(reference, dtype=float)
    n = len(values)
    if (values.shape != (n, 20, 2, 2) or reference.shape != (n, 2, 2) or n < 4
        or not np.isfinite(values).all() or not np.isfinite(reference).all() or replicates < 2):
        raise ValueError("invalid subject effect geometry")
    point = values.mean(axis=0)
    denominator = reference[:, 0].mean(axis=0)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(n, size=(replicates, n))
    weights = np.stack([(draws == i).sum(axis=1) for i in range(n)], axis=1) / n
    sampled = np.einsum("bs,scmz->bcmz", weights, values)
    denom_draws = weights @ reference[:, 0]
    if (np.any(denominator <= 0) or np.any(denom_draws <= 0)
        or np.any(np.quantile(denom_draws, 0.025, axis=0) <= 0)):
        return {"status": "invalid_exact_denominator", "candidates": {}, "comparisons": {}}
    retention = point[:, 0] / denominator
    retention_draws = sampled[:, :, 0] / denom_draws[:, None]
    differences = np.stack([point[a] - point[b] for a, b in PAIRS])
    difference_draws = np.stack([sampled[:, a] - sampled[:, b] for a, b in PAIRS], axis=1)
    claim = np.concatenate((retention.ravel(), point[:, 1].ravel(), differences.ravel()))
    samples = np.concatenate((retention_draws.reshape(replicates, -1),
                              sampled[:, :, 1].reshape(replicates, -1),
                              difference_draws.reshape(replicates, -1)), axis=1)
    if claim.size != 104:
        raise ValueError("simultaneous family changed")
    se = samples.std(axis=0, ddof=1)
    if not np.isfinite(se).all():
        raise ValueError("nonfinite bootstrap standard error")
    active = se > 1e-15
    standardized = np.zeros_like(samples)
    standardized[:, active] = (claim[active] - samples[:, active]) / se[active]
    critical = max(0.0, float(np.quantile(standardized.max(axis=1), 0.95)))
    lower = claim - critical * se
    # Degenerate coordinates cannot support any positive lower-bound claim.
    lower[~active] = np.minimum(claim[~active], 0)
    prox_lower, term_lower = lower[:40].reshape(20, 2), lower[40:80].reshape(20, 2)
    difference_lower = lower[80:].reshape(6, 2, 2)
    candidates = {}
    for i in range(20):
        candidates[str(i)] = {
            "proximal_retention": retention[i].tolist(),
            "proximal_lower95": prox_lower[i].tolist(),
            "terminal_effect": point[i, 1].tolist(), "terminal_lower95": term_lower[i].tolist(),
            "gate_pass": bool(np.all(retention[i] >= 0.8) and np.all(prox_lower[i] >= 0.5)
                              and np.all(term_lower[i] > 0)),
        }
    comparisons = {f"{a}-{b}": {"effect_difference": differences[i].tolist(),
                                  "lower95": difference_lower[i].tolist(),
                                  "gate_pass": bool(np.all(difference_lower[i] > 0))}
                   for i, (a, b) in enumerate(PAIRS)}
    return {"status": "valid" if candidates["0"]["gate_pass"] else "invalid_exact_context128_control",
            "subjects": n, "bootstrap_seed": SEED, "bootstrap_replicates": replicates,
            "exact_proximal_effect": denominator.tolist(), "max_t_critical": critical,
            "degenerate_readout_indices": np.flatnonzero(~active).tolist(),
            "simultaneous_readouts": 104, "candidates": candidates, "comparisons": comparisons}


def summarize(validation: dict, test: dict) -> dict:
    if any(r["status"] != "valid" for r in (validation, test)):
        return {"status": "invalid", "clearing_sources": [], "informative_delta_sources": [],
                "split_status": {"validation": validation["status"], "test": test["status"]}}
    clearing = [i for i in range(6, 16) if all(r["candidates"][str(i)]["gate_pass"] for r in (validation, test))]
    informative = [i for i in (11, 12, 13) if i in clearing and all(
        r["comparisons"][pair]["gate_pass"] for r in (validation, test) for pair in (f"{i}-8", f"{i}-{i + 6}"))]
    return {"status": "valid", "clearing_sources": clearing, "informative_delta_sources": informative,
            "verdict": "carrier_source_supported" if clearing else "no_registered_carrier_source",
            "scope": "fresh capital subjects at the oracle-H1/natural-KV interface",
            "efficiency_claim": False}
