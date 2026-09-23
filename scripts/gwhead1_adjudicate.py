#!/usr/bin/env python3
"""Adjudicate frozen GW-HEAD-1 held-out necessity and sufficiency."""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from gwconv1_adjudicate import mean_pairwise_js, softmax
from gwhead1_preregister import SCHEMA as PREREG_SCHEMA
from gwhead1_select import OUTPUT_SCHEMA as SELECTION_SCHEMA
from gwsup1_preregister import canonical_hash, sha

SCHEMA = "larql.gwhead1.adjudication.v1"
CAPTURE_SCHEMA = "larql.gwhead1.natural-capture.v1"
HELDOUT_SCHEMA = "larql.gwhead1.heldout-arms.v1"
SEED = 1_398_104_331
RESAMPLES = 10_000
ROWS = 426
HELDOUT_ROWS = 171
CANDIDATES = 126
HIDDEN = 2560
RELATIONS = ("capital", "currency", "language", "hypernym")
SPLITS = ("validation", "test")
METRICS = ("candidate_raw", "candidate_zscore", "carrier")
SEMANTIC_METRICS = ("candidate_raw", "candidate_zscore")
DENOMINATOR_EPS = 1e-12
PRIMARY_ARMS = (
    "I_full", "I_ref", "I_S_global", "I_full_minus_S_global",
    "I_S_relation", "I_full_minus_S_relation", "I_identity",
)


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def descriptor(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in manifest["artifacts"] if item["path"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one artifact {name}")
    return matches[0]


def checked_memmap(
    root: Path, manifest: dict[str, Any], name: str, shape: tuple[int, ...]
) -> np.memmap:
    item = descriptor(manifest, name)
    path = root / name
    if sha(path) != item["sha256"]:
        raise ValueError(f"artifact hash mismatch: {path}")
    if path.stat().st_size != math.prod(shape) * 4:
        raise ValueError(f"artifact shape mismatch: {path}")
    return np.memmap(path, dtype="<f4", mode="r", shape=shape)


def zscore_distribution(logits: np.ndarray) -> np.ndarray:
    std = np.std(logits, axis=-1, keepdims=True)
    if np.any(std == 0) or np.any(~np.isfinite(std)):
        raise ValueError("z-scored candidate readout has zero or non-finite variance")
    return softmax((logits - np.mean(logits, axis=-1, keepdims=True)) / std)


def pairwise_cosine(states: np.ndarray) -> float:
    normalized = states / np.linalg.norm(states, axis=1, keepdims=True)
    if np.any(~np.isfinite(normalized)):
        raise ValueError("carrier cosine has a zero or non-finite norm")
    return float(
        np.mean(
            [np.sum(normalized[left] * normalized[right]) for left, right in combinations(range(3), 2)]
        )
    )


def draw_indices(relations: np.ndarray, stratified: bool) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    strata = (
        [np.flatnonzero(relations == relation) for relation in sorted(set(relations))]
        if stratified
        else [np.arange(len(relations))]
    )
    return np.stack(
        [
            np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in strata])
            for _ in range(RESAMPLES)
        ]
    )


def interval(values: np.ndarray, samples: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or np.any(~np.isfinite(values)):
        raise ValueError("interval requires a nonempty finite vector")
    draws = np.mean(values[samples], axis=1)
    return {
        "estimate": float(np.mean(values)),
        "lower_95": float(np.percentile(draws, 2.5)),
        "upper_95": float(np.percentile(draws, 97.5)),
        "resamples": RESAMPLES,
        "seed": SEED,
    }


def causal_summary(
    effects: np.ndarray,
    arm_index: dict[str, int],
    subset: str,
    full_minus: str,
    samples: np.ndarray,
    effect_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    full = effects[:, arm_index["I_full"]]
    reference = effects[:, arm_index["I_ref"]]
    selected = effects[:, arm_index[subset]]
    ablated = effects[:, arm_index[full_minus]]
    denominator = float(np.mean(full) - np.mean(reference))
    if denominator <= DENOMINATOR_EPS:
        raise ValueError(f"non-positive or unstable sufficiency denominator: {denominator}")
    necessity = full - ablated
    numerator_draws = np.mean((selected - reference)[samples], axis=1)
    denominator_draws = np.mean((full - reference)[samples], axis=1)
    if np.any(~np.isfinite(denominator_draws)) or np.any(denominator_draws <= DENOMINATOR_EPS):
        raise ValueError(
            "bootstrap produced a non-positive sufficiency denominator "
            f"(estimate={denominator}, minimum_draw={float(np.min(denominator_draws))}, "
            f"nonpositive_draws={int(np.sum(denominator_draws <= DENOMINATOR_EPS))})"
        )
    ratio_draws = numerator_draws / denominator_draws
    return {
        "effects": {
            name: interval(effects[:, index], samples)
            for name in (effect_names or tuple(arm_index))
            for index in (arm_index[name],)
        },
        "necessity": interval(necessity, samples),
        "sufficiency": {
            "estimate": float((np.mean(selected) - np.mean(reference)) / denominator),
            "lower_95": float(np.percentile(ratio_draws, 2.5)),
            "upper_95": float(np.percentile(ratio_draws, 97.5)),
            "resamples": RESAMPLES,
            "seed": SEED,
            "denominator": denominator,
        },
    }


def safe_causal_summary(
    effects: np.ndarray,
    arm_index: dict[str, int],
    subset: str,
    full_minus: str,
    samples: np.ndarray,
    effect_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Report a frozen descriptive arm while preserving denominator refusal."""
    try:
        result = causal_summary(
            effects, arm_index, subset, full_minus, samples, effect_names
        )
        result["estimable"] = True
        return result
    except ValueError as error:
        if "denominator" not in str(error):
            raise
        return {
            "estimable": False,
            "refusal": str(error),
            "effects": {
                name: interval(effects[:, index], samples)
                for name in (effect_names or tuple(arm_index))
                for index in (arm_index[name],)
            },
        }


def build(args: argparse.Namespace) -> dict[str, Any]:
    prereg = json.loads(args.preregistration.read_text())
    selection = json.loads(args.selection.read_text())
    candidates = json.loads(args.candidates.read_text())
    capture = json.loads(args.capture_manifest.read_text())
    heldout = json.loads(args.heldout_manifest.read_text())
    if prereg.get("schema") != PREREG_SCHEMA or prereg.get("status") != "frozen_pre_execution":
        raise ValueError("not a frozen GW-HEAD-1 preregistration")
    if canonical_hash(prereg, "preregistration_sha256") != prereg["preregistration_sha256"]:
        raise ValueError("GW-HEAD-1 preregistration identity mismatch")
    if selection.get("schema") != SELECTION_SCHEMA or selection.get("status") != "frozen_pre_heldout":
        raise ValueError("not a frozen GW-HEAD-1 selection")
    if canonical_hash(selection, "selection_sha256") != selection["selection_sha256"]:
        raise ValueError("GW-HEAD-1 selection identity mismatch")
    if capture.get("schema") != CAPTURE_SCHEMA or heldout.get("schema") != HELDOUT_SCHEMA:
        raise ValueError("not GW-HEAD-1 execution artifacts")
    if heldout.get("status") != "heldout_execution_complete_pre_adjudication":
        raise ValueError("held-out execution is not sealed pre-adjudication")
    identities = {
        prereg["preregistration_sha256"],
        selection["preregistration_sha256"],
        capture["preregistration_sha256"],
        heldout["preregistration_sha256"],
    }
    if len(identities) != 1 or heldout["selection_sha256"] != selection["selection_sha256"]:
        raise ValueError("GW-HEAD-1 authorities are not frozen together")
    if heldout["natural_capture_sha256"] != sha(args.capture_manifest):
        raise ValueError("held-out execution binds a different natural capture")
    if capture["candidate_identity_sha256"] != candidates["candidate_identity_sha256"]:
        raise ValueError("candidate identity changed")
    metrics_authority = selection["authorities"]["train_subset_metrics"]
    metrics_path = (args.selection.parent / metrics_authority["path"]).resolve()
    if sha(metrics_path) != metrics_authority["sha256"]:
        raise ValueError("train subset metrics authority changed")
    train_metrics = json.loads(metrics_path.read_text())
    if canonical_hash(train_metrics, "artifact_sha256") != metrics_authority["artifact_sha256"]:
        raise ValueError("train subset metrics identity changed")
    arm_count = int(heldout["shape"]["arms"])
    expected_parity = {
        "full_carrier_bit_mismatches": 0,
        "full_proximal_logit_bit_mismatches": 0,
        "full_terminal_logit_bit_mismatches": 0,
        "identity_terminal_logit_bit_mismatches": 0,
        "intervention_firings": HELDOUT_ROWS * (arm_count - 1),
        "expected_intervention_firings": HELDOUT_ROWS * (arm_count - 1),
    }
    if heldout["parity"] != expected_parity:
        raise ValueError("held-out execution parity or firing count failed")
    if heldout["held_out_outcomes_used_for_selection"] is not False:
        raise ValueError("selection used held-out outcomes")

    capture_root = args.capture_manifest.parent
    heldout_root = args.heldout_manifest.parent
    natural_rows_path = capture_root / "natural-rows.jsonl"
    heldout_rows_path = heldout_root / "heldout-rows.jsonl"
    if sha(natural_rows_path) != descriptor(capture, "natural-rows.jsonl")["sha256"]:
        raise ValueError("natural rows hash mismatch")
    if sha(heldout_rows_path) != descriptor(heldout, "heldout-rows.jsonl")["sha256"]:
        raise ValueError("held-out rows hash mismatch")
    natural_rows = jsonl(natural_rows_path)
    heldout_rows = jsonl(heldout_rows_path)
    if len(natural_rows) != ROWS or len(heldout_rows) != HELDOUT_ROWS:
        raise ValueError("GW-HEAD-1 row count changed")
    original_rows = np.asarray([int(row["original_row"]) for row in heldout_rows])
    for index, (row, original) in enumerate(zip(heldout_rows, original_rows)):
        natural = natural_rows[original]
        if row["heldout_row"] != index or row["edge_id"] != natural["edge_id"]:
            raise ValueError("held-out/natural row order differs")
        if row["split"] not in SPLITS or row["split"] != natural["split"]:
            raise ValueError("held-out split differs from natural capture")

    arms = heldout["arms"]
    arm_index = {name: index for index, name in enumerate(arms)}
    expected_arms = {
        "I_full", "I_ref", "I_S_global", "I_full_minus_S_global",
        "I_S_relation", "I_full_minus_S_relation", "I_identity",
    }
    if not expected_arms.issubset(arm_index) or len(arm_index) != len(arms):
        raise ValueError("held-out arm universe changed")
    before_logits_all = checked_memmap(
        capture_root, capture, "candidate-before-logits.f32", (ROWS, CANDIDATES)
    )
    before_carriers_all = checked_memmap(
        capture_root, capture, "carrier-before.f32", (ROWS, HIDDEN)
    )
    proximal_logits = checked_memmap(
        heldout_root, heldout, "heldout-proximal-logits.f32",
        (HELDOUT_ROWS, len(arms), CANDIDATES),
    )
    carriers = checked_memmap(
        heldout_root, heldout, "heldout-carriers.f32", (HELDOUT_ROWS, len(arms), HIDDEN)
    )
    terminal_logits = checked_memmap(
        heldout_root, heldout, "heldout-terminal-logits.f32",
        (HELDOUT_ROWS, len(arms), CANDIDATES),
    )
    before_logits = np.asarray(before_logits_all[original_rows], dtype=float)
    before_carriers = np.asarray(before_carriers_all[original_rows], dtype=float)

    edge_lookup = {row["edge_id"]: index for index, row in enumerate(heldout_rows)}
    if len(edge_lookup) != HELDOUT_ROWS:
        raise ValueError("duplicate held-out execution row")
    controls = {
        item["edge_id"]: item["control_edge_id"]
        for item in candidates["different_destination_controls"]
    }
    groups = [
        group for group in candidates["prompt_family_groups"]
        if group["edge_ids"][0] in edge_lookup
    ]
    if len(groups) != 57:
        raise ValueError("held-out semantic-edge count changed")
    fact_rows = np.asarray([[edge_lookup[edge] for edge in group["edge_ids"]] for group in groups])
    control_rows = np.asarray(
        [[edge_lookup[controls[edge]] for edge in group["edge_ids"]] for group in groups]
    )
    splits = np.asarray([heldout_rows[rows[0]]["split"] for rows in fact_rows])
    relations = np.asarray([group["relation"] for group in groups])
    if {split: int(np.sum(splits == split)) for split in SPLITS} != {"validation": 29, "test": 28}:
        raise ValueError("held-out semantic-edge split count changed")

    before_distributions = {
        "candidate_raw": softmax(before_logits),
        "candidate_zscore": zscore_distribution(before_logits),
    }
    proximal_distributions = {
        "candidate_raw": softmax(np.asarray(proximal_logits, dtype=float)),
        "candidate_zscore": zscore_distribution(np.asarray(proximal_logits, dtype=float)),
    }
    terminal_distributions = {
        "candidate_raw": softmax(np.asarray(terminal_logits, dtype=float)),
        "candidate_zscore": zscore_distribution(np.asarray(terminal_logits, dtype=float)),
    }
    proximal_effects = {name: np.empty((len(groups), len(arms))) for name in METRICS}
    terminal_effects = {
        name: np.empty((len(groups), len(arms))) for name in SEMANTIC_METRICS
    }
    levels: dict[str, Any] = {name: [] for name in METRICS}
    for group_index, (fact, control) in enumerate(zip(fact_rows, control_rows)):
        for name in SEMANTIC_METRICS:
            fact_before = mean_pairwise_js(before_distributions[name][fact])
            control_before = mean_pairwise_js(before_distributions[name][control])
            fact_after = mean_pairwise_js(proximal_distributions[name][fact])
            control_after = mean_pairwise_js(proximal_distributions[name][control])
            proximal_effects[name][group_index] = (
                (fact_before - fact_after) - (control_before - control_after)
            )
            fact_terminal = mean_pairwise_js(terminal_distributions[name][fact])
            control_terminal = mean_pairwise_js(terminal_distributions[name][control])
            terminal_effects[name][group_index] = control_terminal - fact_terminal
            levels[name].append({
                "same_fact_before": float(fact_before),
                "matched_control_before": float(control_before),
            })
        fact_before_cos = pairwise_cosine(before_carriers[fact])
        control_before_cos = pairwise_cosine(before_carriers[control])
        for arm in range(len(arms)):
            proximal_effects["carrier"][group_index, arm] = (
                pairwise_cosine(np.asarray(carriers[fact, arm], dtype=float)) - fact_before_cos
                - pairwise_cosine(np.asarray(carriers[control, arm], dtype=float)) + control_before_cos
            )
        levels["carrier"].append({
            "same_fact_before": fact_before_cos,
            "matched_control_before": control_before_cos,
        })

    held_out: dict[str, Any] = {}
    relation_breakdown: dict[str, Any] = {}
    for split in SPLITS:
        split_mask = splits == split
        split_relations = relations[split_mask]
        samples = draw_indices(split_relations, True)
        held_out[split] = {
            "semantic_edges": int(np.sum(split_mask)),
            "global_subset": {
                metric: safe_causal_summary(
                    proximal_effects[metric][split_mask], arm_index,
                    "I_S_global", "I_full_minus_S_global", samples,
                    PRIMARY_ARMS,
                )
                for metric in METRICS
            },
            "terminal_global_subset": {
                metric: safe_causal_summary(
                    terminal_effects[metric][split_mask], arm_index,
                    "I_S_global", "I_full_minus_S_global", samples,
                    PRIMARY_ARMS,
                )
                for metric in SEMANTIC_METRICS
            },
        }
        relation_breakdown[split] = {}
        for relation in RELATIONS:
            mask = split_mask & (relations == relation)
            relation_samples = draw_indices(relations[mask], False)
            relation_breakdown[split][relation] = {
                "semantic_edges": int(np.sum(mask)),
                "selected_heads": selection["scopes"][relation]["selected"]["heads"],
                "proximal": {
                    metric: safe_causal_summary(
                        proximal_effects[metric][mask], arm_index,
                        "I_S_relation", "I_full_minus_S_relation", relation_samples,
                        PRIMARY_ARMS,
                    )
                    for metric in METRICS
                },
                "terminal": {
                    metric: safe_causal_summary(
                        terminal_effects[metric][mask], arm_index,
                        "I_S_relation", "I_full_minus_S_relation", relation_samples,
                        PRIMARY_ARMS,
                    )
                    for metric in SEMANTIC_METRICS
                },
            }

    replication = heldout["parity"] == expected_parity
    necessity = all(
        held_out[split]["global_subset"][metric].get("estimable", False)
        and held_out[split]["global_subset"][metric]["necessity"]["lower_95"] > 0
        for split in SPLITS for metric in SEMANTIC_METRICS
    )
    sufficiency = all(
        held_out[split]["global_subset"][metric].get("estimable", False)
        and held_out[split]["global_subset"][metric]["sufficiency"]["lower_95"] >= 0.50
        for split in SPLITS for metric in SEMANTIC_METRICS
    )
    structural = (
        capture["reconstruction"]["result"] == "pass"
        and capture["reconstruction"]["max_head_sum_relative_l2"]
        <= prereg["exact_decomposition"]["raw_max_relative_l2"]
        and capture["reconstruction"]["max_applied_delta_relative_l2"]
        <= prereg["exact_decomposition"]["applied_max_relative_l2"]
    )
    conditions = {
        "structural": bool(structural),
        "replication": replication,
        "necessity": necessity,
        "sufficiency": sufficiency,
    }
    conditions["causal_head_subset"] = all(conditions.values())
    global_heads = selection["scopes"]["global"]["selected"]["heads"]
    if len(global_heads) != 1:
        raise ValueError("the frozen global subset is not a singleton")
    selected_head = int(global_heads[0])
    singleton_train_rows = [
        row for row in train_metrics["scopes"]["global"]["subsets"]
        if len(row["heads"]) == 1
    ]
    if len(singleton_train_rows) != 8:
        raise ValueError("train surface does not contain all singleton controls")
    selected_norm = next(
        float(row["total_natural_contribution_norm"])
        for row in singleton_train_rows if row["heads"] == global_heads
    )
    nearest_norm_row = min(
        (row for row in singleton_train_rows if row["heads"] != global_heads),
        key=lambda row: (
            abs(float(row["total_natural_contribution_norm"]) - selected_norm),
            row["heads"],
        ),
    )
    conditions["concentrated_causal_head_subset"] = (
        conditions["causal_head_subset"] and len(global_heads) <= 4
    )
    terminal_conditions = {
        "necessity": all(
            held_out[split]["terminal_global_subset"][metric].get("estimable", False)
            and held_out[split]["terminal_global_subset"][metric]["necessity"]["lower_95"] > 0
            for split in SPLITS for metric in SEMANTIC_METRICS
        ),
        "sufficiency": all(
            held_out[split]["terminal_global_subset"][metric].get("estimable", False)
            and held_out[split]["terminal_global_subset"][metric]["sufficiency"]["lower_95"] >= 0.50
            for split in SPLITS for metric in SEMANTIC_METRICS
        ),
    }
    terminal_conditions["transport_supported"] = all(terminal_conditions.values())

    specificity: dict[str, Any] = {}
    singleton_arms = [f"I_S_head_{head}" for head in range(8)]
    singleton_ablations = [f"I_full_minus_head_{head}" for head in range(8)]
    if all(name in arm_index for name in (*singleton_arms, *singleton_ablations)):
        for split in SPLITS:
            split_mask = splits == split
            samples = draw_indices(relations[split_mask], True)
            specificity[split] = {}
            for metric in METRICS:
                head_results = {
                    str(head): causal_summary(
                        proximal_effects[metric][split_mask],
                        arm_index,
                        singleton_arms[head],
                        singleton_ablations[head],
                        samples,
                        ("I_full", "I_ref", singleton_arms[head], singleton_ablations[head]),
                    )
                    for head in range(8)
                }
                ordered = sorted(
                    range(8),
                    key=lambda head: (
                        -head_results[str(head)]["sufficiency"]["estimate"], head
                    ),
                )
                specificity[split][metric] = {
                    "heads": head_results,
                    "sufficiency_rank_descending": ordered,
                    "selected_head_rank": ordered.index(selected_head) + 1,
                }
            if "I_zero_S_global" in arm_index and "I_only_S_global" in arm_index:
                specificity[split]["zero_controls"] = {
                    metric: {
                        "zero_selected_effect": interval(
                            proximal_effects[metric][split_mask, arm_index["I_zero_S_global"]],
                            samples,
                        ),
                        "retain_only_selected_effect": interval(
                            proximal_effects[metric][split_mask, arm_index["I_only_S_global"]],
                            samples,
                        ),
                    }
                    for metric in METRICS
                }

    edge_metrics = []
    for index, group in enumerate(groups):
        edge_metrics.append({
            "subject": group["subject"],
            "relation": group["relation"],
            "target": group["target"],
            "edge_ids": group["edge_ids"],
            "split": str(splits[index]),
            "proximal": {
                metric: {arm: float(proximal_effects[metric][index, arm_index[arm]]) for arm in arms}
                for metric in METRICS
            },
            "terminal": {
                metric: {arm: float(terminal_effects[metric][index, arm_index[arm]]) for arm in arms}
                for metric in SEMANTIC_METRICS
            },
        })
    args.output.mkdir(parents=True, exist_ok=True)
    edge_path = args.output / "heldout-edge-metrics.jsonl"
    edge_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in edge_metrics))
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "adjudicated_before_example_inspection",
        "adjudication_sha256": None,
        "authorities": {
            "preregistration_sha256": prereg["preregistration_sha256"],
            "selection_sha256": selection["selection_sha256"],
            "train_subset_metrics_sha256": metrics_authority["sha256"],
            "train_subset_metrics_identity": metrics_authority["artifact_sha256"],
            "candidate_identity_sha256": candidates["candidate_identity_sha256"],
            "natural_capture_manifest_sha256": sha(args.capture_manifest),
            "heldout_manifest_sha256": sha(args.heldout_manifest),
            "edge_metrics_sha256": sha(edge_path),
        },
        "selection": {
            "global_heads": global_heads,
            "selected_train_contribution_norm": selected_norm,
            "nearest_train_contribution_norm_singleton": {
                "heads": nearest_norm_row["heads"],
                "total_natural_contribution_norm": nearest_norm_row[
                    "total_natural_contribution_norm"
                ],
            },
            "relation_heads": {
                relation: selection["scopes"][relation]["selected"]["heads"]
                for relation in RELATIONS
            },
            "held_out_outcomes_used_for_selection": False,
        },
        "held_out": held_out,
        "relation_breakdown": relation_breakdown,
        "heldout_specificity_controls": specificity,
        "gate": {
            "conditions": conditions,
            "verdict": (
                "causal_head_subset_supported"
                if conditions["causal_head_subset"]
                else "full_conjunctive_gate_not_met"
            ),
        },
        "terminal_transport": {
            "estimand": "matched-control terminal alignment: control JS minus same-fact JS",
            "conditions": terminal_conditions,
            "claim_boundary": "mandatory qualification; not part of proximal GW-HEAD-1 selection or gate",
        },
        "carrier_role": "qualification only; cannot rescue either conjunctive candidate metric",
        "source_key_inferred": False,
        "examples_inspected_before_adjudication": False,
        "artifacts": {
            "edge_metrics": {"path": edge_path.name, "rows": len(edge_metrics), "sha256": sha(edge_path)}
        },
    }
    report["adjudication_sha256"] = canonical_hash(report, "adjudication_sha256")
    report_path = args.output / "adjudication.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args)
    print(json.dumps({
        "schema": report["schema"],
        "status": report["status"],
        "adjudication_sha256": report["adjudication_sha256"],
        "selection": report["selection"],
        "gate": report["gate"],
        "terminal_transport": report["terminal_transport"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
