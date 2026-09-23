#!/usr/bin/env python3
"""Adjudicate frozen GW-KEY-1 held-out K/V source-role causality."""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from gwconv1_adjudicate import mean_pairwise_js, softmax
from gwsup1_preregister import canonical_hash, sha
from gwkey1_preregister import SCHEMA as PREREG_SCHEMA
from gwkey1_select import OUTPUT_SCHEMA as SELECTION_SCHEMA

SCHEMA = "larql.gwkey1.adjudication.v1"
HEAD_SCHEMA = "larql.gwhead1.natural-capture.v1"
HELDOUT_SCHEMA = "larql.gwkey1.heldout-arms.v1"
SEED = 1_398_114_331
RESAMPLES = 10_000
ROWS = 426
HELDOUT_ROWS = 171
CANDIDATES = 126
HIDDEN = 2560
SPLITS = ("validation", "test")
RELATIONS = ("capital", "currency", "language", "hypernym")
SEMANTIC_METRICS = ("candidate_raw", "candidate_zscore")
METRICS = (*SEMANTIC_METRICS, "carrier")
HELDOUT_FAMILIES = ("V", "joint")
DENOMINATOR_EPS = 1e-12


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
    if sha(path) != item["sha256"] or path.stat().st_size != math.prod(shape) * 4:
        raise ValueError(f"invalid artifact {path}")
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
            [
                np.sum(normalized[left] * normalized[right])
                for left, right in combinations(range(3), 2)
            ]
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
            np.concatenate(
                [rng.choice(indices, size=len(indices), replace=True) for indices in strata]
            )
            for _ in range(RESAMPLES)
        ]
    )


def interval(values: np.ndarray, samples: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
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
    family: str,
    samples: np.ndarray,
) -> dict[str, Any]:
    full = effects[:, arm_index["I_full"]]
    reference = effects[:, arm_index[f"{family}_I_ref"]]
    selected = effects[:, arm_index[f"{family}_I_S"]]
    ablated = effects[:, arm_index[f"{family}_I_full_minus_S"]]
    denominator_values = full - reference
    denominator = float(np.mean(denominator_values))
    if denominator <= DENOMINATOR_EPS:
        raise ValueError(f"non-positive sufficiency denominator: {denominator}")
    denominator_draws = np.mean(denominator_values[samples], axis=1)
    if np.any(denominator_draws <= DENOMINATOR_EPS):
        raise ValueError(
            "bootstrap produced a non-positive sufficiency denominator "
            f"(minimum={float(np.min(denominator_draws))})"
        )
    numerator_draws = np.mean((selected - reference)[samples], axis=1)
    ratio_draws = numerator_draws / denominator_draws
    names = (
        "I_full",
        f"{family}_I_ref",
        f"{family}_I_S",
        f"{family}_I_full_minus_S",
        "I_identity",
    )
    return {
        "estimable": True,
        "effects": {name: interval(effects[:, arm_index[name]], samples) for name in names},
        "necessity": interval(full - ablated, samples),
        "sufficiency": {
            "estimate": float(np.mean(selected - reference) / denominator),
            "lower_95": float(np.percentile(ratio_draws, 2.5)),
            "upper_95": float(np.percentile(ratio_draws, 97.5)),
            "denominator": denominator,
            "resamples": RESAMPLES,
            "seed": SEED,
        },
    }


def safe_summary(
    effects: np.ndarray,
    arm_index: dict[str, int],
    family: str,
    samples: np.ndarray,
) -> dict[str, Any]:
    try:
        return causal_summary(effects, arm_index, family, samples)
    except ValueError as error:
        if "denominator" not in str(error):
            raise
        return {"estimable": False, "refusal": str(error)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    prereg = json.loads(args.preregistration.read_text())
    selection = json.loads(args.selection.read_text())
    candidates = json.loads(args.candidates.read_text())
    head = json.loads(args.head_capture.read_text())
    heldout = json.loads(args.heldout_manifest.read_text())
    if prereg.get("schema") != PREREG_SCHEMA or prereg.get("status") != "frozen_pre_execution":
        raise ValueError("not a frozen GW-KEY-1 preregistration")
    if canonical_hash(prereg, "preregistration_sha256") != prereg["preregistration_sha256"]:
        raise ValueError("GW-KEY-1 preregistration identity mismatch")
    if selection.get("schema") != SELECTION_SCHEMA or selection.get("status") != "frozen_pre_heldout":
        raise ValueError("not a frozen GW-KEY-1 selection")
    if canonical_hash(selection, "selection_sha256") != selection["selection_sha256"]:
        raise ValueError("GW-KEY-1 selection identity mismatch")
    if head.get("schema") != HEAD_SCHEMA or heldout.get("schema") != HELDOUT_SCHEMA or heldout.get("status") != "complete_frozen_heldout":
        raise ValueError("not complete GW-KEY-1 held-out artifacts")
    if len(
        {
            prereg["preregistration_sha256"],
            selection["preregistration_sha256"],
            heldout["preregistration_sha256"],
        }
    ) != 1 or heldout["selection_sha256"] != selection["selection_sha256"]:
        raise ValueError("GW-KEY-1 authorities are not frozen together")
    if heldout["selection_file_sha256"] != sha(args.selection) or heldout["gwhead1_natural_capture_sha256"] != sha(args.head_capture):
        raise ValueError("GW-KEY-1 held-out execution binds different authorities")
    if candidates["candidate_identity_sha256"] != prereg["authorities"]["candidate_identity"]:
        raise ValueError("candidate identity changed")
    expected_parity = {
        "full_carrier_bit_mismatches": 0,
        "full_proximal_bit_mismatches": 0,
        "full_terminal_bit_mismatches": 0,
        "identity_terminal_bit_mismatches": 0,
    }
    if heldout["parity"] != expected_parity or heldout["intervention_firings"] != heldout["expected_intervention_firings"]:
        raise ValueError("GW-KEY-1 held-out parity or intervention firing count failed")

    head_root = args.head_capture.parent
    heldout_root = args.heldout_manifest.parent
    natural_rows_path = head_root / "natural-rows.jsonl"
    heldout_rows_path = heldout_root / "heldout-rows.jsonl"
    if sha(natural_rows_path) != descriptor(head, "natural-rows.jsonl")["sha256"] or sha(heldout_rows_path) != descriptor(heldout, "heldout-rows.jsonl")["sha256"]:
        raise ValueError("GW-KEY-1 row artifact hash mismatch")
    natural_rows = jsonl(natural_rows_path)
    heldout_rows = jsonl(heldout_rows_path)
    if len(natural_rows) != ROWS or len(heldout_rows) != HELDOUT_ROWS:
        raise ValueError("GW-KEY-1 held-out row count changed")
    original_rows = np.asarray([int(row["original_row"]) for row in heldout_rows])
    for index, (row, original) in enumerate(zip(heldout_rows, original_rows)):
        if row["heldout_row"] != index or row["edge_id"] != natural_rows[original]["edge_id"] or row["split"] not in SPLITS:
            raise ValueError("GW-KEY-1 held-out row order changed")

    arms = heldout["arms"]
    arm_index = {name: index for index, name in enumerate(arms)}
    expected_arms = {"I_full", "I_identity"} | {
        f"{family}_{suffix}"
        for family in HELDOUT_FAMILIES
        for suffix in ("I_ref", "I_S", "I_full_minus_S")
    }
    if set(arm_index) != expected_arms or len(arm_index) != len(arms):
        raise ValueError("GW-KEY-1 held-out arm universe changed")
    before_logits_all = checked_memmap(
        head_root, head, "candidate-before-logits.f32", (ROWS, CANDIDATES)
    )
    before_carriers_all = checked_memmap(
        head_root, head, "carrier-before.f32", (ROWS, HIDDEN)
    )
    proximal_logits = checked_memmap(
        heldout_root,
        heldout,
        "heldout-proximal-logits.f32",
        (HELDOUT_ROWS, len(arms), CANDIDATES),
    )
    carriers = checked_memmap(
        heldout_root,
        heldout,
        "heldout-carriers.f32",
        (HELDOUT_ROWS, len(arms), HIDDEN),
    )
    terminal_logits = checked_memmap(
        heldout_root,
        heldout,
        "heldout-terminal-logits.f32",
        (HELDOUT_ROWS, len(arms), CANDIDATES),
    )
    before_logits = np.asarray(before_logits_all[original_rows], dtype=float)
    before_carriers = np.asarray(before_carriers_all[original_rows], dtype=float)

    edge_lookup = {row["edge_id"]: index for index, row in enumerate(heldout_rows)}
    controls = {
        item["edge_id"]: item["control_edge_id"]
        for item in candidates["different_destination_controls"]
    }
    groups = [
        group
        for group in candidates["prompt_family_groups"]
        if group["edge_ids"][0] in edge_lookup
    ]
    if len(groups) != 57:
        raise ValueError("GW-KEY-1 held-out semantic-edge count changed")
    fact_rows = np.asarray(
        [[edge_lookup[edge] for edge in group["edge_ids"]] for group in groups]
    )
    control_rows = np.asarray(
        [[edge_lookup[controls[edge]] for edge in group["edge_ids"]] for group in groups]
    )
    splits = np.asarray([heldout_rows[rows[0]]["split"] for rows in fact_rows])
    relations = np.asarray([group["relation"] for group in groups])
    if {split: int(np.sum(splits == split)) for split in SPLITS} != {"validation": 29, "test": 28}:
        raise ValueError("GW-KEY-1 held-out split count changed")

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
    proximal_effects = {
        metric: np.empty((len(groups), len(arms))) for metric in METRICS
    }
    terminal_effects = {
        metric: np.empty((len(groups), len(arms))) for metric in SEMANTIC_METRICS
    }
    for group_index, (fact, control) in enumerate(zip(fact_rows, control_rows)):
        for metric in SEMANTIC_METRICS:
            fact_before = mean_pairwise_js(before_distributions[metric][fact])
            control_before = mean_pairwise_js(before_distributions[metric][control])
            proximal_effects[metric][group_index] = (
                fact_before
                - np.asarray(
                    [mean_pairwise_js(proximal_distributions[metric][fact, arm]) for arm in range(len(arms))]
                )
                - control_before
                + np.asarray(
                    [mean_pairwise_js(proximal_distributions[metric][control, arm]) for arm in range(len(arms))]
                )
            )
            terminal_effects[metric][group_index] = np.asarray(
                [mean_pairwise_js(terminal_distributions[metric][control, arm]) for arm in range(len(arms))]
            ) - np.asarray(
                [mean_pairwise_js(terminal_distributions[metric][fact, arm]) for arm in range(len(arms))]
            )
        fact_before_cos = pairwise_cosine(before_carriers[fact])
        control_before_cos = pairwise_cosine(before_carriers[control])
        for arm in range(len(arms)):
            proximal_effects["carrier"][group_index, arm] = (
                pairwise_cosine(np.asarray(carriers[fact, arm], dtype=float))
                - fact_before_cos
                - pairwise_cosine(np.asarray(carriers[control, arm], dtype=float))
                + control_before_cos
            )

    held_out: dict[str, Any] = {}
    relation_breakdown: dict[str, Any] = {}
    for split in SPLITS:
        mask = splits == split
        samples = draw_indices(relations[mask], True)
        held_out[split] = {
            "semantic_edges": int(np.sum(mask)),
            "proximal": {
                family: {
                    metric: safe_summary(
                        proximal_effects[metric][mask], arm_index, family, samples
                    )
                    for metric in METRICS
                }
                for family in HELDOUT_FAMILIES
            },
            "terminal": {
                family: {
                    metric: safe_summary(
                        terminal_effects[metric][mask], arm_index, family, samples
                    )
                    for metric in SEMANTIC_METRICS
                }
                for family in HELDOUT_FAMILIES
            },
        }
        relation_breakdown[split] = {}
        for relation in RELATIONS:
            relation_mask = mask & (relations == relation)
            relation_samples = draw_indices(relations[relation_mask], False)
            relation_breakdown[split][relation] = {
                "semantic_edges": int(np.sum(relation_mask)),
                "proximal": {
                    family: {
                        metric: safe_summary(
                            proximal_effects[metric][relation_mask],
                            arm_index,
                            family,
                            relation_samples,
                        )
                        for metric in METRICS
                    }
                    for family in HELDOUT_FAMILIES
                },
            }

    gates: dict[str, Any] = {}
    terminal_gates: dict[str, Any] = {}
    for family in HELDOUT_FAMILIES:
        necessity = all(
            held_out[split]["proximal"][family][metric].get("estimable", False)
            and held_out[split]["proximal"][family][metric]["necessity"]["lower_95"] > 0
            for split in SPLITS
            for metric in SEMANTIC_METRICS
        )
        sufficiency = all(
            held_out[split]["proximal"][family][metric].get("estimable", False)
            and held_out[split]["proximal"][family][metric]["sufficiency"]["lower_95"] >= 0.50
            for split in SPLITS
            for metric in SEMANTIC_METRICS
        )
        gates[family] = {
            "necessity": necessity,
            "sufficiency": sufficiency,
            "pass": necessity and sufficiency,
        }
        terminal_necessity = all(
            held_out[split]["terminal"][family][metric].get("estimable", False)
            and held_out[split]["terminal"][family][metric]["necessity"]["lower_95"] > 0
            for split in SPLITS
            for metric in SEMANTIC_METRICS
        )
        terminal_sufficiency = all(
            held_out[split]["terminal"][family][metric].get("estimable", False)
            and held_out[split]["terminal"][family][metric]["sufficiency"]["lower_95"] >= 0.50
            for split in SPLITS
            for metric in SEMANTIC_METRICS
        )
        terminal_gates[family] = {
            "necessity": terminal_necessity,
            "sufficiency": terminal_sufficiency,
            "transport_supported": terminal_necessity and terminal_sufficiency,
        }
    gates["K"] = {
        "necessity": False,
        "sufficiency": False,
        "pass": False,
        "train_stage_refusal": selection["families"]["K"]["refusal"],
        "train_denominator": selection["families"]["K"]["denominator"],
    }
    terminal_gates["K"] = {
        "transport_supported": False,
        "not_executed": "no admissible train-selected K subset",
    }

    edge_metrics = []
    for index, group in enumerate(groups):
        edge_metrics.append(
            {
                "subject": group["subject"],
                "relation": group["relation"],
                "target": group["target"],
                "edge_ids": group["edge_ids"],
                "split": str(splits[index]),
                "proximal": {
                    metric: {
                        arm: float(proximal_effects[metric][index, arm_index[arm]])
                        for arm in arms
                    }
                    for metric in METRICS
                },
                "terminal": {
                    metric: {
                        arm: float(terminal_effects[metric][index, arm_index[arm]])
                        for arm in arms
                    }
                    for metric in SEMANTIC_METRICS
                },
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    edge_path = args.output / "heldout-edge-metrics.jsonl"
    edge_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in edge_metrics)
    )
    compact = bool(selection["compact_joint"])
    candidate_read_path = gates["joint"]["pass"] and compact
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "adjudicated_before_example_inspection",
        "adjudication_sha256": None,
        "authorities": {
            "preregistration_sha256": prereg["preregistration_sha256"],
            "selection_sha256": selection["selection_sha256"],
            "candidate_identity_sha256": candidates["candidate_identity_sha256"],
            "gwhead1_natural_capture_sha256": sha(args.head_capture),
            "heldout_manifest_sha256": sha(args.heldout_manifest),
            "edge_metrics_sha256": sha(edge_path),
        },
        "selection": {
            family: selection["families"][family]["selected"]
            for family in ("K", *HELDOUT_FAMILIES)
        },
        "held_out": held_out,
        "relation_breakdown": relation_breakdown,
        "gate": {
            "families": gates,
            "compact_joint": compact,
            "candidate_read_path": candidate_read_path,
            "verdict": (
                "compact_candidate_read_path_supported"
                if candidate_read_path
                else "joint_conjunctive_gate_not_met"
            ),
        },
        "terminal_transport": {
            "families": terminal_gates,
            "estimand": "matched-control terminal alignment: control JS minus same-fact JS",
            "claim_boundary": "mandatory qualification; not part of proximal source-role gate",
        },
        "carrier_role": "qualification only; cannot rescue either conjunctive candidate metric",
        "natural_query_cost_established": False,
        "efficient_walk_established": False,
        "examples_inspected_before_adjudication": False,
        "artifacts": {
            "edge_metrics": {
                "path": edge_path.name,
                "rows": len(edge_metrics),
                "sha256": sha(edge_path),
            }
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
    parser.add_argument("--head-capture", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "status": report["status"],
                "adjudication_sha256": report["adjudication_sha256"],
                "selection": report["selection"],
                "gate": report["gate"],
                "terminal_transport": report["terminal_transport"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
