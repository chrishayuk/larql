#!/usr/bin/env python3
"""Train-only STATE-1 selection and frozen-family held-out adjudication primitives.

Input effects have axes [subject, context, proximal/terminal, raw/zscored].
Each effect is the control-adjusted target-H1 minus donor-H1 contrast after
averaging the three relation edges within a subject. Context IDs are
carrier * 128 + a seven-head bit mask ordered H0,H2,H3,H4,H5,H6,H7.
"""
from __future__ import annotations

import numpy as np
from gwv2_adjudicate import distributions, group_js

HEADS = (0, 2, 3, 4, 5, 6, 7)
EXACT = 255
SENTINELS = (0, 127, 128, 255)
SEED = 27022033
REPLICATES = 10_000


def subject_effects_from_logits(execution_rows: list[dict], proximal: np.ndarray,
                                terminal: np.ndarray, context_ids: list[int],
                                split: str, row_indices: list[int] | None = None) -> tuple[list[str], np.ndarray]:
    """Reduce paired replay logits to control-adjusted subject H1 contrasts.

    Logits have axes [row, context, donor/target H1, candidate token]. The
    paired before-readout cancels, so only post-intervention logits are needed.
    Contexts are processed in small chunks to bound transient f64 memory.
    """
    if split not in ("train", "validation", "test") or len(set(context_ids)) != len(context_ids):
        raise ValueError("invalid split or repeated context")
    if any(not 0 <= i < 256 for i in context_ids):
        raise ValueError("invalid context ID")
    if row_indices is None:
        row_indices = list(range(len(execution_rows)))
    if len(set(row_indices)) != len(row_indices) or any(i < 0 or i >= len(execution_rows) for i in row_indices):
        raise ValueError("invalid replay row indices")
    tensor_position = {row: position for position, row in enumerate(row_indices)}
    tensors = [np.asarray(proximal), np.asarray(terminal)]
    if (any(t.ndim != 4 or t.shape[:3] != (len(row_indices), len(context_ids), 2)
            or not np.isfinite(t).all() for t in tensors)
        or tensors[0].shape != tensors[1].shape):
        raise ValueError("invalid paired replay-logit geometry")
    indices = [i for i in row_indices if execution_rows[i]["row"]["split"] == split]
    tensor_indices = [tensor_position[i] for i in indices]
    local = {global_index: position for position, global_index in enumerate(indices)}
    lookup = {item["row"]["edge_id"]: i for i, item in enumerate(execution_rows)}
    groups = {}
    for i in indices:
        row = execution_rows[i]["row"]
        key = (row["subject_id"], row["semantic_edge"]["relation"])
        groups.setdefault(key, []).append(local[i])
    keys = sorted(groups)
    if not keys or any(len(groups[key]) != 3 for key in keys):
        raise ValueError("incomplete three-prompt semantic-edge groups")
    facts = np.array([groups[key] for key in keys], dtype=np.intp)
    controls = []
    for group in facts:
        paired = []
        for row_index in group:
            row = execution_rows[indices[row_index]]["row"]
            matches = [control for control in row["control_ids"]
                       if control["kind"] == "same_relation_different_subject"]
            if len(matches) != 1 or matches[0]["paired_edge_id"] not in lookup:
                raise ValueError("missing frozen matched control")
            donor = lookup[matches[0]["paired_edge_id"]]
            if donor not in local:
                raise ValueError("cross-split matched control")
            paired.append(local[donor])
        controls.append(paired)
    controls = np.array(controls, dtype=np.intp)
    subjects = sorted({key[0] for key in keys})
    edge_indices = [[i for i, key in enumerate(keys) if key[0] == subject] for subject in subjects]
    if any(len(group) != 3 for group in edge_indices):
        raise ValueError("a subject is missing a factual relation")
    result = np.empty((len(subjects), len(context_ids), 2, 2), dtype=np.float64)
    for start in range(0, len(context_ids), 16):
        stop = min(start + 16, len(context_ids))
        for surface, tensor in enumerate(tensors):
            for metric in range(2):
                probs = distributions(tensor[tensor_indices, start:stop], z=bool(metric))
                same = group_js(probs, facts)
                control = group_js(probs, controls)
                paired_effect = (same[:, :, 0] - same[:, :, 1]
                                 - control[:, :, 0] + control[:, :, 1])
                for subject_index, group in enumerate(edge_indices):
                    result[subject_index, start:stop, surface, metric] = paired_effect[group].mean(axis=0)
    return subjects, result


def context_heads(context: int) -> tuple[int, ...]:
    if not 0 <= context < 256:
        raise ValueError("invalid STATE-1 context")
    return tuple(head for bit, head in enumerate(HEADS) if context % 128 & (1 << bit))


def _effects(values: np.ndarray, full_surface: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 4 or values.shape[2:] != (2, 2)
        or (full_surface and values.shape[1] != 256)
        or not np.isfinite(values).all()):
        raise ValueError("invalid finite STATE-1 subject-effect surface")
    return values


def select_train(subject_effects: np.ndarray, natural_head_norms: np.ndarray) -> dict:
    """Freeze one minimum eligible context per carrier branch using train only."""
    values = _effects(subject_effects)
    norms = np.asarray(natural_head_norms, dtype=np.float64)
    if norms.shape != (7,) or not np.isfinite(norms).all() or np.any(norms < 0):
        raise ValueError("seven finite nonnegative effective-W_O head norms required")
    point = values.mean(axis=0)
    denominator = point[EXACT, 0]
    if np.any(denominator <= 0):
        raise ValueError("nonpositive all-natural train H1 effect")
    retention = point[:, 0] / denominator
    eligible = (np.all(retention >= 0.80, axis=1)
                & np.all(point[:, 0] > 0, axis=1)
                & np.all(point[:, 1] > 0, axis=1))
    selected = []
    for carrier in range(2):
        candidates = [i for i in range(carrier * 128, (carrier + 1) * 128) if eligible[i]]
        def rank(i):
            heads = context_heads(i)
            head_norm = sum(norms[HEADS.index(h)] for h in heads)
            return (len(heads), -float(retention[i].min()), head_norm, heads)
        selected.append(min(candidates, key=rank) if candidates else None)
    return {"carrier_branch_selected": selected,
            "eligible_contexts": int(eligible.sum()),
            "exact_proximal_effect": denominator.tolist(),
            "selected_train_retention": [retention[i].tolist() if i is not None else None for i in selected]}


def frozen_family(selected: list[int | None]) -> tuple[int, ...]:
    if len(selected) != 2 or any(i is not None and not 0 <= i < 256 for i in selected):
        raise ValueError("two frozen carrier-branch selections required")
    if selected[0] is not None and selected[0] >= 128:
        raise ValueError("donor-carrier selection crosses branch")
    if selected[1] is not None and selected[1] < 128:
        raise ValueError("natural-carrier selection crosses branch")
    return tuple(sorted(set(SENTINELS).union(i for i in selected if i is not None)))


def adjudicate_split(subject_effects: np.ndarray, selected: list[int | None],
                     seed: int = SEED, replicates: int = REPLICATES,
                     context_ids: list[int] | None = None) -> dict:
    """One subject-clustered max-t family over every verdict-bearing context."""
    values = _effects(subject_effects, full_surface=context_ids is None)
    if values.shape[0] != 15 or replicates < 2:
        raise ValueError("held-out split requires 15 subjects and multiple draws")
    family = frozen_family(selected)
    if context_ids is None:
        local = values[:, family]
    else:
        if tuple(context_ids) != family or values.shape[1] != len(context_ids):
            raise ValueError("held-out replay differs from the frozen simultaneous family")
        local = values
    point = local.mean(axis=0)
    rng = np.random.default_rng(seed)
    draws = rng.integers(values.shape[0], size=(replicates, values.shape[0]))
    weights = np.stack([(draws == i).sum(axis=1) for i in range(values.shape[0])], axis=1)
    weights = weights / values.shape[0]
    sampled = np.einsum("bs,scmz->bcmz", weights, local)
    exact_index = family.index(EXACT)
    denominator = point[exact_index, 0]
    denominator_draws = sampled[:, exact_index, 0]
    stable = bool(np.all(denominator > 0)
                  and np.all(denominator_draws > 0)
                  and np.all(np.quantile(denominator_draws, 0.025, axis=0) > 0))
    if not stable:
        return {"family": list(family), "exact_stable": False,
                "contexts": {}, "max_t_critical": None}
    retention = point[:, 0] / denominator
    retention_draws = sampled[:, :, 0] / denominator_draws[:, None]
    terminal = point[:, 1]
    terminal_draws = sampled[:, :, 1]
    # Exact/all-natural proximal retention is identically 1; its SE is zero.
    retained_indices = [i for i, context in enumerate(family) if context != EXACT]
    claim_point = np.concatenate((retention[retained_indices].reshape(-1), terminal.reshape(-1)))
    claim_draws = np.concatenate((retention_draws[:, retained_indices].reshape(replicates, -1),
                                  terminal_draws.reshape(replicates, -1)), axis=1)
    standard_error = claim_draws.std(axis=0, ddof=1)
    if not np.isfinite(standard_error).all() or np.any(standard_error <= 0):
        raise ValueError("zero or nonfinite STATE-1 simultaneous-family standard error")
    critical = float(np.quantile(((claim_point - claim_draws) / standard_error).max(axis=1), 0.95))
    lower = claim_point - critical * standard_error
    retention_lower = np.ones_like(retention)
    terminal_lower = lower[len(retained_indices) * 2:].reshape(len(family), 2)
    retention_lower[retained_indices] = lower[:len(retained_indices) * 2].reshape(-1, 2)
    contexts = {}
    for index, context in enumerate(family):
        passes = bool(np.all(retention[index] >= 0.80)
                      and np.all(retention_lower[index] >= 0.50)
                      and np.all(terminal_lower[index] > 0))
        contexts[str(context)] = {
            "carrier": "natural" if context >= 128 else "donor",
            "natural_non_h1_heads": list(context_heads(context)),
            "proximal_retention": retention[index].tolist(),
            "proximal_simultaneous_lower95": retention_lower[index].tolist(),
            "terminal_effect": terminal[index].tolist(),
            "terminal_simultaneous_lower95": terminal_lower[index].tolist(),
            "gate_pass": passes,
        }
    return {"family": list(family), "exact_stable": True,
            "exact_proximal_effect": denominator.tolist(),
            "exact_pointwise_lower95": np.quantile(denominator_draws, 0.025, axis=0).tolist(),
            "contexts": contexts, "max_t_critical": critical,
            "bootstrap_seed": seed, "bootstrap_replicates": replicates}


def verdict(validation: dict, test: dict, selected: list[int | None]) -> str:
    """Apply the ordered, frozen classes only after both splits are available."""
    if not validation["exact_stable"] or not test["exact_stable"]:
        return "no_stable_context"
    def clears(context):
        return all(result["contexts"].get(str(context), {}).get("gate_pass", False)
                   for result in (validation, test))
    if not clears(EXACT):
        return "no_stable_context"
    if clears(0):
        return "edge_only"
    if clears(128):
        return "carrier_plus_edge"
    if any(i is not None and len(context_heads(i)) <= 3 and clears(i) for i in selected):
        return "local_head_circuit"
    return "broad_context"
