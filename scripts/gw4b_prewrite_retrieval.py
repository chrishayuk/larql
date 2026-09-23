#!/usr/bin/env python3
"""Evaluate pre-write carrier keys against frozen GW-4A write families."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gw4a_write_families import artifact, quantiles, read_bound, sha256_file
from gw4b_preregister import validate as validate_preregistration


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("key contains a zero or non-finite vector")
    return rows / norms


def load_vectors(preregistration_path: Path, preregistration: dict) -> tuple[list[dict], np.ndarray, np.ndarray, dict]:
    root = preregistration_path.parent
    authorities = preregistration["authorities"]
    report_path = (root / authorities["gw4a_report"]["path"]).resolve()
    report = json.loads(report_path.read_text())
    metadata = report["site_evidence"]

    sealed_path = (root / authorities["sealed_gw0"]["path"]).resolve()
    sealed = json.loads(sealed_path.read_text())
    census_path = sealed_path.parent / sealed["census"]["path"]
    if sha256_file(census_path) != sealed["census"]["sha256"]:
        raise ValueError("sealed census hash mismatch")
    by_edge = {row["edge_id"]: row for row in jsonl(census_path)}
    artifact_root = (sealed_path.parent / sealed["artifact_root"]).resolve()
    hidden = preregistration["cohort"]["hidden_size"]
    tolerance = preregistration["query_contract"]["norm_relative_tolerance"]
    before_rows = []
    delta_rows = []
    norm_errors = []
    evidence = []
    for site in metadata:
        edge_id = site["edge_id"]
        row = by_edge[edge_id]
        record_item = artifact(row, "execution_record")
        record = json.loads(read_bound(artifact_root, record_item))
        matches = [
            index
            for index, value in enumerate(record["sites"])
            if value["layer"] == site["layer"] and value["site"] == "ffn"
        ]
        if len(matches) != 1 or matches[0] == 0:
            raise ValueError(f"{edge_id}: FFN site is not uniquely preceded")
        position = matches[0]
        previous = record["sites"][position - 1]
        if previous["site"] != "attention" or previous["layer"] != site["layer"]:
            raise ValueError(f"{edge_id}: FFN carrier-before is not the same-layer attention output")

        before_item = artifact(row, "carrier-before")
        delta_item = artifact(row, "delta")
        if delta_item["sha256"] != site["delta_sha256"]:
            raise ValueError(f"{edge_id}: delta differs from GW-4A evidence")
        expected_shape = [len(record["sites"]), hidden]
        before = np.frombuffer(read_bound(artifact_root, before_item), dtype="<f4")
        delta = np.frombuffer(read_bound(artifact_root, delta_item), dtype="<f4")
        if before_item.get("shape") != expected_shape or delta_item.get("shape") != expected_shape:
            raise ValueError(f"{edge_id}: artifact shape mismatch")
        if before.size != math.prod(expected_shape) or delta.size != math.prod(expected_shape):
            raise ValueError(f"{edge_id}: artifact length mismatch")
        before_vector = before.reshape(expected_shape)[position].astype(np.float64)
        delta_vector = delta.reshape(expected_shape)[position].astype(np.float64)
        before_norm = float(np.linalg.norm(before_vector))
        expected_norm = float(previous["carrier_norm"])
        error = abs(before_norm - expected_norm) / max(expected_norm, 1e-30)
        if error > tolerance:
            raise ValueError(f"{edge_id}: carrier-before norm mismatch ({error})")
        if abs(float(np.linalg.norm(delta_vector)) - site["write_norm"]) / site["write_norm"] > tolerance:
            raise ValueError(f"{edge_id}: delta norm mismatch")
        before_rows.append(before_vector)
        delta_rows.append(delta_vector)
        norm_errors.append(error)
        evidence.append(
            {
                "edge_id": edge_id,
                "carrier_before_sha256": before_item["sha256"],
                "delta_sha256": delta_item["sha256"],
                "execution_record_sha256": record_item["sha256"],
                "carrier_before_norm_relative_error": error,
            }
        )
    return metadata, normalize(np.stack(before_rows)), normalize(np.stack(delta_rows)), {
        "gw4a_report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        "sealed_manifest": {"path": str(sealed_path), "sha256": sha256_file(sealed_path)},
        "census": {"path": str(census_path), "sha256": sha256_file(census_path)},
        "carrier_before_norm_relative_error": quantiles(norm_errors),
        "site_evidence": evidence,
    }


def bootstrap_p05(values_by_fact: dict[tuple, list[float]], rng: np.random.Generator, trials: int) -> dict:
    blocks = np.array([np.median(values) for values in values_by_fact.values()], dtype=np.float64)
    draws = [float(np.median(rng.choice(blocks, size=len(blocks), replace=True))) for _ in range(trials)]
    return quantiles(draws)


def make_relation_keys(metadata: list[dict], before: np.ndarray, widths: list[int]) -> tuple[dict, dict]:
    train = [index for index, row in enumerate(metadata) if row["split"] == "train"]
    relations = sorted({row["relation"] for row in metadata})
    centered = np.empty_like(before)
    means = {}
    bases = {}
    for relation in relations:
        train_rows = [index for index in train if metadata[index]["relation"] == relation]
        all_rows = [index for index, row in enumerate(metadata) if row["relation"] == relation]
        mean = before[train_rows].mean(axis=0)
        means[relation] = mean
        centered[all_rows] = normalize(before[all_rows] - mean)
        _, _, right = np.linalg.svd(centered[train_rows], full_matrices=False)
        bases[relation] = right[: max(1, len(train_rows) - 1)]
    keys = {"raw_carrier": before, "relation_centered_full": centered}
    effective = {}
    for width in widths:
        matrix = np.zeros((len(metadata), width), dtype=np.float64)
        relation_widths = {}
        for relation in relations:
            rows = [index for index, row in enumerate(metadata) if row["relation"] == relation]
            rank = min(width, bases[relation].shape[0])
            relation_widths[relation] = rank
            projected = centered[rows] @ bases[relation][:rank].T
            projected = normalize(projected)
            matrix[np.ix_(rows, range(rank))] = projected
        name = f"relation_centered_pca_{width}"
        keys[name] = matrix
        effective[name] = relation_widths
    return keys, {"means": means, "bases": bases, "effective_widths": effective}


def evaluate_arm(
    name: str,
    key: np.ndarray,
    key_width: int,
    metadata: list[dict],
    deltas: np.ndarray,
    candidate_ks: list[int],
    oracle_k: int,
    rng_seed: int,
    random_trials: int,
    bootstrap_trials: int,
    state_only: bool = False,
) -> dict:
    train = [index for index, row in enumerate(metadata) if row["split"] == "train"]
    assessment = [index for index, row in enumerate(metadata) if row["split"] != "train"]
    dot = key @ key.T
    delta_dot = deltas @ deltas.T
    per_k = {}
    for k in candidate_ks:
        best = []
        gaps = []
        fractions = []
        global_fractions = []
        top1_hits = 0
        top4_hits = 0
        top4_total = 0
        returned_total = 0
        dot_products = 0
        random_expected = []
        advantages = []
        advantages_by_fact: dict[tuple, list[float]] = defaultdict(list)
        refused = []
        arm_rng = np.random.default_rng(rng_seed + 1009 * k + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
        for query in assessment:
            structural = [
                index
                for index in train
                if metadata[index]["layer"] == metadata[query]["layer"]
                and metadata[index]["relation"] == metadata[query]["relation"]
                and metadata[index]["fact"] != metadata[query]["fact"]
            ]
            if not structural:
                refused.append(metadata[query]["edge_id"])
                continue
            universe = [
                index
                for index in train
                if metadata[index]["layer"] == metadata[query]["layer"]
                and metadata[index]["fact"] != metadata[query]["fact"]
                and (state_only or metadata[index]["relation"] == metadata[query]["relation"])
            ]
            count = min(k, len(universe))
            ranked = sorted(universe, key=lambda index: (-float(dot[query, index]), metadata[index]["edge_id"]))
            retrieved = ranked[:count]
            oracle = sorted(
                structural,
                key=lambda index: (-float(delta_dot[query, index]), metadata[index]["edge_id"]),
            )
            oracle_best = float(delta_dot[query, oracle[0]])
            retrieved_best = max(float(delta_dot[query, index]) for index in retrieved)
            best.append(retrieved_best)
            gaps.append(oracle_best - retrieved_best)
            fractions.append(count / len(universe))
            global_fractions.append(count / len(train))
            top1_hits += oracle[0] in retrieved
            neighbourhood = set(oracle[: min(oracle_k, len(oracle))])
            top4_hits += len(neighbourhood.intersection(retrieved))
            top4_total += len(neighbourhood)
            returned_total += count
            dot_products += len(universe)
            random_best = []
            for _ in range(random_trials):
                sample = arm_rng.choice(universe, size=count, replace=False)
                random_best.append(max(float(delta_dot[query, index]) for index in sample))
            expected_random = float(np.mean(random_best))
            random_expected.append(expected_random)
            advantage = retrieved_best - expected_random
            advantages.append(advantage)
            advantages_by_fact[tuple(metadata[query]["fact"])].append(advantage)
        eligible = len(assessment) - len(refused)
        bootstrap = bootstrap_p05(advantages_by_fact, arm_rng, bootstrap_trials) if eligible else {"n": 0}
        per_k[str(k)] = {
            "assessment_rows": len(assessment),
            "eligible_queries": eligible,
            "query_coverage": eligible / len(assessment),
            "refused_edge_ids": refused,
            "mean_candidate_fraction": float(np.mean(fractions)) if fractions else None,
            "mean_global_training_fraction": float(np.mean(global_fractions)) if global_fractions else None,
            "best_retrieved_delta_cosine": quantiles(best),
            "matched_random_expected_best_delta_cosine": quantiles(random_expected),
            "advantage_over_matched_random": quantiles(advantages),
            "oracle_gap": quantiles(gaps),
            "oracle_top1_exemplar_retention": top1_hits / len(assessment),
            "oracle_top4_exemplar_recall": top4_hits / top4_total if top4_total else None,
            "advantage_over_matched_random_block_bootstrap": bootstrap,
            "costs": {
                "returned_candidates": returned_total,
                "key_dot_products": dot_products,
                "key_bytes_touched": (dot_products + eligible) * key_width * 4,
                "returned_delta_bytes_touched": returned_total * deltas.shape[1] * 4,
            },
        }
    return {"name": name, "key_width": key_width, "state_only_ablation": state_only, "operating_points": per_k}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"GW-4B output already exists: {args.output}")
    validation = validate_preregistration(args.preregistration)
    preregistration = json.loads(args.preregistration.read_text())
    operating = preregistration["operating_points"]
    metadata, before, deltas, inputs = load_vectors(args.preregistration, preregistration)
    keys, fitted = make_relation_keys(metadata, before, operating["pca_widths"])
    arms = []
    for name, key in keys.items():
        width = before.shape[1] if not name.startswith("relation_centered_pca_") else int(name.rsplit("_", 1)[1])
        arms.append(
            evaluate_arm(
                name,
                key,
                width,
                metadata,
                deltas,
                operating["candidate_k"],
                operating["oracle_neighbourhood_k"],
                operating["random_seed"],
                operating["random_trials"],
                operating["bootstrap_trials"],
            )
        )
    arms.append(
        evaluate_arm(
            "state_only_ablation",
            before,
            before.shape[1],
            metadata,
            deltas,
            operating["candidate_k"],
            operating["oracle_neighbourhood_k"],
            operating["random_seed"],
            operating["random_trials"],
            operating["bootstrap_trials"],
            state_only=True,
        )
    )

    gate = preregistration["progression_gate"]
    passing = []
    for arm in arms:
        if arm["state_only_ablation"]:
            continue
        for k, point in arm["operating_points"].items():
            bootstrap = point["advantage_over_matched_random_block_bootstrap"]
            if (
                point["query_coverage"] >= gate["minimum_query_coverage"]
                and point["mean_candidate_fraction"] <= gate["maximum_mean_candidate_fraction"]
                and point["best_retrieved_delta_cosine"]["median"] >= gate["minimum_median_best_delta_cosine"]
                and point["oracle_gap"]["median"] <= gate["maximum_median_oracle_gap"]
                and bootstrap["p05"] >= gate["minimum_block_bootstrap_p05_advantage_over_matched_random"]
            ):
                passing.append({"arm": arm["name"], "candidate_k": int(k)})

    train_rows = sum(row["split"] == "train" for row in metadata)
    scalar_bytes = operating["bytes_per_scalar"]
    metadata_bytes = operating["metadata_bytes_per_index_row"]
    index_accounting = {}
    for arm in arms:
        width = arm["key_width"]
        index_accounting[arm["name"]] = {
            "key_bytes": train_rows * width * scalar_bytes,
            "metadata_bytes": train_rows * metadata_bytes,
            "projection_basis_bytes": 0,
        }
        if arm["name"].startswith("relation_centered_pca_"):
            effective = fitted["effective_widths"][arm["name"]]
            index_accounting[arm["name"]]["projection_basis_bytes"] = sum(
                before.shape[1] * rank * scalar_bytes for rank in effective.values()
            ) + len(effective) * before.shape[1] * scalar_bytes
        elif arm["name"] == "relation_centered_full":
            index_accounting[arm["name"]]["projection_basis_bytes"] = (
                len(fitted["means"]) * before.shape[1] * scalar_bytes
            )

    report = {
        "schema": "larql.gw4b.prewrite-retrieval-report.v1",
        "claim_boundary": preregistration["claim_boundary"],
        "preregistration": {
            "path": str(args.preregistration),
            "identity_sha256": validation["preregistration_sha256"],
            "file_sha256": sha256_file(args.preregistration),
        },
        "inputs": inputs,
        "denominators": {
            "sites": len(metadata),
            "training_rows": train_rows,
            "assessment_rows": len(metadata) - train_rows,
            "causal_status": {"untested": len(metadata)},
        },
        "effective_pca_widths": fitted["effective_widths"],
        "arms": arms,
        "index_accounting": index_accounting,
        "decision": {
            "passing_operating_points": passing,
            "open_gw4c": bool(passing),
            "otherwise": "retire direct carrier-before exemplar indexing and move to multi-site trajectories or semantic basins",
        },
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
