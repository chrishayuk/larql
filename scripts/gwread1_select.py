#!/usr/bin/env python3
"""Compute train-only GW-READ-1 effects and seal Q/K/V primaries."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from gwconv1_adjudicate import mean_pairwise_js, softmax
from gwsup1_preregister import canonical_hash, sha


ROWS = 426
TRAIN_ROWS = 255
CANDIDATES = 126


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def descriptor(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in manifest["artifacts"] if item["path"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one artifact {name}")
    return matches[0]


def checked_memmap(
    root: Path,
    manifest: dict[str, Any],
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.memmap:
    item = descriptor(manifest, name)
    path = root / name
    shape = expected_shape or tuple(item["shape"])
    if sha(path) != item["sha256"] or path.stat().st_size != math.prod(shape) * 4:
        raise ValueError(f"artifact changed: {path}")
    return np.memmap(path, dtype="<f4", mode="r", shape=shape)


def checked_u8(root: Path, manifest: dict[str, Any], name: str) -> np.ndarray:
    item = descriptor(manifest, name)
    path = root / name
    shape = tuple(item["shape"])
    if sha(path) != item["sha256"] or path.stat().st_size != math.prod(shape):
        raise ValueError(f"artifact changed: {path}")
    values = np.fromfile(path, dtype=np.uint8).reshape(shape)
    if np.any(values > 1):
        raise ValueError(f"invalid availability mask: {path}")
    return values


def zscore_distribution(logits: np.ndarray) -> np.ndarray:
    std = np.std(logits, axis=-1, keepdims=True)
    if np.any(std == 0) or np.any(~np.isfinite(std)):
        raise ValueError("z-scored readout has zero or non-finite variance")
    return softmax((logits - np.mean(logits, axis=-1, keepdims=True)) / std)


def adjusted(
    before: np.ndarray,
    after: np.ndarray,
    fact_rows: np.ndarray,
    control_rows: np.ndarray,
) -> np.ndarray:
    fact_before = np.asarray([mean_pairwise_js(before[rows]) for rows in fact_rows])
    fact_after = np.asarray([mean_pairwise_js(after[rows]) for rows in fact_rows])
    control_before = np.asarray([mean_pairwise_js(before[rows]) for rows in control_rows])
    control_after = np.asarray([mean_pairwise_js(after[rows]) for rows in control_rows])
    return (fact_before - fact_after) - (control_before - control_after)


def cost(candidate_id: str, arm: str, average_subject_tokens: float, average_source_tokens: float) -> dict[str, Any]:
    cached = "train-mean" in candidate_id or candidate_id.startswith("k/") or candidate_id.startswith("v/cache-")
    diagnostic = candidate_id == "v/same-fact-cross-prompt-cycle"
    if cached:
        dynamic_flops = 0
        dynamic_bytes = HEAD_BYTES = 256 * 4
        if arm == "K":
            dynamic_bytes = int(math.ceil(average_source_tokens * HEAD_BYTES))
        elif arm == "V":
            dynamic_bytes = int(math.ceil(average_subject_tokens * HEAD_BYTES))
        return {
            "cost_class": "cached/materialized",
            "dynamic_matvec_equivalent_flops": dynamic_flops,
            "dynamic_bytes_touched": dynamic_bytes,
            "prefix_end_layer": None,
            "diagnostic_only": diagnostic,
        }
    if candidate_id == "v/entity-only-L24":
        return {
            "cost_class": "per-query computed",
            "dynamic_matvec_equivalent_flops": 25_000_000_000_000,
            "dynamic_bytes_touched": 25_000_000_000,
            "prefix_end_layer": 24,
            "execution_stream": "entity-only",
            "diagnostic_only": False,
        }
    match = re.search(r"layer(\d+)", candidate_id)
    if not match:
        return {
            "cost_class": "diagnostic",
            "dynamic_matvec_equivalent_flops": 2**63 - 1,
            "dynamic_bytes_touched": 2**63 - 1,
            "prefix_end_layer": None,
            "diagnostic_only": True,
        }
    layer = int(match.group(1))
    rank_match = re.search(r"-r(\d+)$", candidate_id)
    predictor_flops = 0 if rank_match is None else 2 * 256 * int(rank_match.group(1))
    return {
        "cost_class": "per-query computed",
        "dynamic_matvec_equivalent_flops": (layer + 1) * 1_000_000_000_000 + predictor_flops,
        "dynamic_bytes_touched": (layer + 1) * 1_000_000_000 + predictor_flops * 2,
        "prefix_end_layer": layer,
        "ordering_ledger": "monotone frozen prefix units; physical plan ledger required before READ-1E cost adjudication",
        "diagnostic_only": False,
    }


def pareto(rows: list[dict[str, Any]]) -> list[str]:
    frontier: list[str] = []
    for row in rows:
        dominated = False
        for other in rows:
            if row is other:
                continue
            no_worse = (
                other["minimum_retention"] >= row["minimum_retention"]
                and other["cost"]["dynamic_matvec_equivalent_flops"] <= row["cost"]["dynamic_matvec_equivalent_flops"]
                and other["cost"]["dynamic_bytes_touched"] <= row["cost"]["dynamic_bytes_touched"]
            )
            strict = (
                other["minimum_retention"] > row["minimum_retention"]
                or other["cost"]["dynamic_matvec_equivalent_flops"] < row["cost"]["dynamic_matvec_equivalent_flops"]
                or other["cost"]["dynamic_bytes_touched"] < row["cost"]["dynamic_bytes_touched"]
            )
            if no_worse and strict:
                dominated = True
                break
        if not dominated:
            frontier.append(row["id"])
    return sorted(frontier)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--head-capture", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    if args.metrics.exists() or args.selection.exists():
        raise ValueError("GW-READ-1 train selection output already exists")

    protocol = json.loads(args.protocol.read_text())
    candidates = json.loads(args.candidates.read_text())
    replay = json.loads(args.replay.read_text())
    head = json.loads(args.head_capture.read_text())
    groups = json.loads(args.groups.read_text())
    if protocol["schema"] != "larql.gwread1.protocol.v2" or candidates["schema"] != "larql.gwread1.candidate-artifact.v1" or replay["schema"] != "larql.gwread1.train-candidate-replay.v1":
        raise ValueError("inadmissible GW-READ-1 selection authority")
    if replay["status"] != "complete_train_only_pre_selection" or replay["validation_or_test_executed"] is not False:
        raise ValueError("selection input is not train-only")
    if replay["parity"] != {"exact_gwkey_candidate_logit_bit_mismatches": 0}:
        raise ValueError("train replay parity failed")
    if replay["candidate_artifact_sha256"] != sha(args.candidates) or replay["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError("selection authorities disagree")

    replay_root = args.replay.parent
    head_root = args.head_capture.parent
    train_rows = jsonl(replay_root / "train-rows.jsonl")
    natural_rows = jsonl(head_root / "natural-rows.jsonl")
    if len(train_rows) != TRAIN_ROWS or len(natural_rows) != ROWS:
        raise ValueError("selection row count changed")
    original_rows = np.asarray([row["original_row"] for row in train_rows], dtype=int)
    before_all = checked_memmap(
        head_root, head, "candidate-before-logits.f32", (ROWS, CANDIDATES)
    )
    before_logits = np.asarray(before_all[original_rows], dtype=float)
    before = {
        "raw": softmax(before_logits),
        "zscored": zscore_distribution(before_logits),
    }
    edge_lookup = {row["edge_id"]: index for index, row in enumerate(train_rows)}
    controls = {item["edge_id"]: item["control_edge_id"] for item in groups["different_destination_controls"]}
    train_groups = [group for group in groups["prompt_family_groups"] if group["edge_ids"][0] in edge_lookup]
    if len(train_groups) != 85:
        raise ValueError("train semantic-edge group count changed")
    fact_rows = np.asarray([[edge_lookup[edge] for edge in group["edge_ids"]] for group in train_groups])
    control_rows = np.asarray([[edge_lookup[controls[edge]] for edge in group["edge_ids"]] for group in train_groups])
    relations = np.asarray([group["relation"] for group in train_groups])
    factual = relations != "hypernym"

    source_counts = [natural_rows[index]["capture_position"] + 1 for index in original_rows]
    input_rows = jsonl((args.protocol.parent / protocol["authorities"]["input_rows"]["path"]).resolve())
    role_rows = jsonl((args.protocol.parent / protocol["authorities"]["source_roles"]["path"]).resolve())
    subject_counts = [len(role_rows[index]["roles"]["subject_entity"]) for index in original_rows]
    avg_sources = float(np.mean(source_counts))
    avg_subjects = float(np.mean(subject_counts))

    candidate_root = args.candidates.parent
    population_availability = {
        "Q": checked_u8(candidate_root, candidates, "q-availability.u8"),
        "K": checked_u8(candidate_root, candidates, "k-availability.u8"),
        "V": checked_u8(candidate_root, candidates, "v-availability.u8"),
    }

    arm_files = {"Q": "train-q-logits.f32", "K": "train-k-logits.f32", "V": "train-v-logits.f32"}
    arm_results: dict[str, Any] = {}
    for arm, filename in arm_files.items():
        logits = checked_memmap(replay_root, replay, filename)
        ids = candidates["candidates"][arm]
        coverage = replay["coverage_rows"][arm]
        if logits.shape != (TRAIN_ROWS, len(ids) + 1, CANDIDATES):
            raise ValueError(f"{arm} replay shape changed")
        effects: list[dict[str, Any]] = []
        all_per_edge: dict[str, list[np.ndarray]] = {"raw": [], "zscored": []}
        for arm_index in range(len(ids) + 1):
            arm_logits = np.asarray(logits[:, arm_index, :], dtype=float)
            after = {"raw": softmax(arm_logits), "zscored": zscore_distribution(arm_logits)}
            for surface in ("raw", "zscored"):
                all_per_edge[surface].append(adjusted(before[surface], after[surface], fact_rows, control_rows))
        exact = {surface: float(np.mean(all_per_edge[surface][0][factual])) for surface in ("raw", "zscored")}
        if any(value <= 0 for value in exact.values()):
            raise ValueError(f"{arm} exact factual denominator is non-positive")
        for candidate_index, candidate_id in enumerate(ids):
            effect = {surface: float(np.mean(all_per_edge[surface][candidate_index + 1][factual])) for surface in ("raw", "zscored")}
            retention = {surface: effect[surface] / exact[surface] for surface in ("raw", "zscored")}
            candidate_cost = cost(candidate_id, arm, avg_subjects, avg_sources)
            complete = coverage[candidate_index] == TRAIN_ROWS
            population_coverage = int(population_availability[arm][candidate_index].sum())
            population_size = int(population_availability[arm].shape[1])
            population_complete = population_coverage == population_size
            eligible_for_e = not candidate_cost["diagnostic_only"]
            eligible = (
                complete
                and population_complete
                and eligible_for_e
                and min(retention.values()) >= 0.8
            )
            effects.append({
                "id": candidate_id,
                "coverage_rows": coverage[candidate_index],
                "complete_coverage": complete,
                "frozen_population_coverage": population_coverage,
                "frozen_population_size": population_size,
                "frozen_population_complete": population_complete,
                "eligible_for_E": eligible_for_e,
                "E": effect,
                "retention": retention,
                "minimum_retention": min(retention.values()),
                "E_by_relation": {
                    relation: {surface: float(np.mean(all_per_edge[surface][candidate_index + 1][relations == relation])) for surface in ("raw", "zscored")}
                    for relation in ("capital", "currency", "language", "hypernym")
                },
                "cost": candidate_cost,
                "train_eligible": eligible,
            })
        eligible_rows = [row for row in effects if row["train_eligible"]]
        selected = None
        if eligible_rows:
            selected_row = min(
                eligible_rows,
                key=lambda row: (
                    row["cost"]["dynamic_matvec_equivalent_flops"],
                    row["cost"]["dynamic_bytes_touched"],
                    -row["minimum_retention"],
                    row["id"],
                ),
            )
            selected = selected_row["id"]
        arm_results[arm] = {
            "exact_E_factual": exact,
            "candidates": effects,
            "eligible_count": len(eligible_rows),
            "pareto_frontier": pareto(eligible_rows),
            "selected": selected,
        }

    metrics: dict[str, Any] = {
        "schema": "larql.gwread1.train-metrics.v1",
        "status": "complete_train_only",
        "artifact_sha256": None,
        "protocol_sha256": protocol["protocol_sha256"],
        "authorities": {
            "candidate_artifact": {"path": str(args.candidates), "sha256": sha(args.candidates)},
            "train_replay": {"path": str(args.replay), "sha256": sha(args.replay)},
            "head_capture": {"path": str(args.head_capture), "sha256": sha(args.head_capture)},
            "groups": {"path": str(args.groups), "sha256": sha(args.groups)},
        },
        "split": "train",
        "heldout_rows": 0,
        "semantic_edges": len(train_groups),
        "factual_semantic_edges": int(factual.sum()),
        "effects": arm_results,
        "cost_warning": "selection ordering uses frozen monotone prefix units; physical plan bytes/FLOPs and measured latency remain mandatory before READ-1E cost adjudication",
    }
    metrics["artifact_sha256"] = canonical_hash(metrics, "artifact_sha256")
    args.metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    if any(arm_results[arm]["selected"] is None for arm in ("Q", "K", "V")):
        status = "selection_refused_missing_train_eligible_component"
    else:
        status = "frozen_pre_heldout"
    selection: dict[str, Any] = {
        "schema": "larql.gwread1.selection.v1",
        "status": status,
        "selection_sha256": None,
        "protocol_sha256": protocol["protocol_sha256"],
        "authorities": {
            "train_metrics": {"path": str(args.metrics), "sha256": sha(args.metrics), "identity": metrics["artifact_sha256"]},
            "candidate_artifact": {"path": str(args.candidates), "sha256": sha(args.candidates)},
            "train_replay": {"path": str(args.replay), "sha256": sha(args.replay)},
        },
        "selection_rule": protocol["selection"],
        "selected": {arm: arm_results[arm]["selected"] for arm in ("Q", "K", "V")},
        "pareto_frontier": {arm: arm_results[arm]["pareto_frontier"] for arm in ("Q", "K", "V")},
        "heldout_rows_seen": 0,
        "composition_tuned": False,
    }
    selection["selection_sha256"] = canonical_hash(selection, "selection_sha256")
    args.selection.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
