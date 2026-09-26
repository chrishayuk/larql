#!/usr/bin/env python3
"""Evaluate reusable FFN write families over the sealed GW-0/GW-0B corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gw4a_preregister import validate as validate_preregistration


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def quantiles(values: Iterable[float]) -> dict:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"n": 0}

    def at(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "n": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": at(0.5),
        "p05": at(0.05),
        "p95": at(0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def artifact(row: dict, kind: str) -> dict:
    matches = [item for item in row["artifacts"] if item.get("kind") == kind]
    if len(matches) != 1:
        raise ValueError(f"{row['edge_id']}: expected one {kind} artifact")
    return matches[0]


def read_bound(root: Path, item: dict) -> bytes:
    path = root / item["path"]
    payload = path.read_bytes()
    if len(payload) != item["bytes"] or "sha256:" + hashlib.sha256(payload).hexdigest() != item["sha256"]:
        raise ValueError(f"sealed artifact mismatch: {path}")
    return payload


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC requires positive and negative pairs")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[labels].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def balanced_accuracy(truth: list[str], prediction: list[str | None], relations: list[str]) -> dict:
    recalls = {}
    for relation in relations:
        indices = [index for index, value in enumerate(truth) if value == relation]
        if indices:
            recalls[relation] = sum(prediction[index] == relation for index in indices) / len(indices)
    return {
        "value": sum(recalls.values()) / len(recalls) if recalls else None,
        "per_relation_recall": recalls,
        "rows": len(truth),
    }


def load_sites(preregistration_path: Path, preregistration: dict) -> tuple[list[dict], np.ndarray, dict]:
    root = preregistration_path.parent
    authorities = preregistration["authorities"]
    sealed_path = (root / authorities["sealed_gw0"]["path"]).resolve()
    sealed = json.loads(sealed_path.read_text())
    census_path = sealed_path.parent / sealed["census"]["path"]
    if sha256_file(census_path) != sealed["census"]["sha256"]:
        raise ValueError("sealed census hash mismatch")
    census = jsonl(census_path)
    by_edge = {row["edge_id"]: row for row in census}
    artifact_root = (sealed_path.parent / sealed["artifact_root"]).resolve()

    reconciliation_path = (root / authorities["gw0b_reconciliation"]["path"]).resolve()
    reconciliation = json.loads(reconciliation_path.read_text())
    attribution_root = (root / authorities["gw0b_reconciliation"]["attribution_root"]).resolve()
    attributions = {}
    attribution_files = []
    for item in reconciliation["input_artifacts"]["attributions"]:
        path = attribution_root / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"attribution hash mismatch: {path}")
        rows = jsonl(path)
        if len(rows) != item["rows"]:
            raise ValueError(f"attribution row count mismatch: {path}")
        attribution_files.append(item)
        for row in rows:
            if row["status"] != "reconstructed" or row["edge_id"] in attributions:
                raise ValueError("GW-4A requires unique reconstructed attributions")
            attributions[row["edge_id"]] = row

    hidden = preregistration["cohort"]["hidden_size"]
    tolerance = preregistration["vector_rule"]["write_norm_relative_tolerance"]
    metadata = []
    vectors = []
    for edge_id, attribution in sorted(attributions.items()):
        row = by_edge[edge_id]
        layer = attribution["site"]["layer"]
        record_item = artifact(row, "execution_record")
        record = json.loads(read_bound(artifact_root, record_item))
        if record["edge_id"] != edge_id:
            raise ValueError(f"{edge_id}: execution record binding mismatch")
        matching = [
            index
            for index, site in enumerate(record["sites"])
            if site["layer"] == layer and site["site"] == "ffn"
        ]
        if len(matching) != 1:
            raise ValueError(f"{edge_id}: FFN site is not uniquely represented")
        delta_item = artifact(row, "delta")
        delta = np.frombuffer(read_bound(artifact_root, delta_item), dtype="<f4")
        shape = delta_item.get("shape", [len(record["sites"]), hidden])
        if shape != [len(record["sites"]), hidden] or delta.size != math.prod(shape):
            raise ValueError(f"{edge_id}: delta geometry mismatch")
        vector = delta.reshape(shape)[matching[0]].astype(np.float64)
        if not np.isfinite(vector).all():
            raise ValueError(f"{edge_id}: non-finite delta")
        norm = float(np.linalg.norm(vector))
        recorded_norm = float(record["sites"][matching[0]]["write_norm"])
        relative_error = abs(norm - recorded_norm) / max(recorded_norm, 1e-30)
        if norm == 0 or relative_error > tolerance:
            raise ValueError(f"{edge_id}: write norm mismatch ({relative_error})")
        semantic = row["semantic_edge"]
        fact = (semantic["subject"], semantic["relation"], semantic["target"])
        metadata.append(
            {
                "edge_id": edge_id,
                "edge_family_id": row["edge_family_id"],
                "fact": fact,
                "relation": semantic["relation"],
                "subject": semantic["subject"],
                "target": semantic["target"],
                "prompt_family": semantic["prompt_semantic_family"],
                "split": row["split"],
                "layer": layer,
                "site": "ffn",
                "write_norm": norm,
                "recorded_write_norm": recorded_norm,
                "write_norm_relative_error": relative_error,
                "delta_sha256": delta_item["sha256"],
                "execution_record_sha256": record_item["sha256"],
                "causal_status": row["causal_status"],
            }
        )
        vectors.append(vector / norm)
    matrix = np.stack(vectors)
    if len(metadata) != preregistration["cohort"]["eligible_sites"]:
        raise ValueError("loaded site count differs from preregistration")
    return metadata, matrix, {
        "sealed_manifest": {"path": str(sealed_path), "sha256": sha256_file(sealed_path)},
        "census": {"path": str(census_path), "sha256": sha256_file(census_path)},
        "reconciliation": {
            "path": str(reconciliation_path),
            "sha256": sha256_file(reconciliation_path),
        },
        "attributions": attribution_files,
    }


def prompt_repeatability(metadata: list[dict], cosine: np.ndarray, rng: np.random.Generator, trials: int) -> dict:
    exact_positive = []
    all_positive = []
    control = []
    positive_blocks: dict[tuple, list[float]] = defaultdict(list)
    control_blocks: dict[tuple, list[float]] = defaultdict(list)
    for left in range(len(metadata)):
        for right in range(left + 1, len(metadata)):
            a, b = metadata[left], metadata[right]
            if a["prompt_family"] == b["prompt_family"]:
                continue
            score = float(cosine[left, right])
            if a["fact"] == b["fact"]:
                all_positive.append(score)
                if a["layer"] == b["layer"]:
                    exact_positive.append(score)
                    positive_blocks[a["fact"]].append(score)
            elif a["relation"] == b["relation"] and a["layer"] == b["layer"]:
                control.append(score)
                key = tuple(sorted((a["fact"], b["fact"])))
                control_blocks[key].append(score)
    positive_values = np.array([np.median(values) for values in positive_blocks.values()])
    control_values = np.array([np.median(values) for values in control_blocks.values()])
    gaps = []
    if len(positive_values) and len(control_values):
        for _ in range(trials):
            p = rng.choice(positive_values, size=len(positive_values), replace=True)
            c = rng.choice(control_values, size=len(control_values), replace=True)
            gaps.append(float(np.median(p) - np.median(c)))
    exact = quantiles(exact_positive)
    control_summary = quantiles(control)
    gap = exact.get("median", 0.0) - control_summary.get("median", 0.0)
    return {
        "same_fact_different_prompt_exact_layer": exact,
        "same_fact_different_prompt_all_layers_descriptive": quantiles(all_positive),
        "same_relation_different_fact_exact_layer_control": control_summary,
        "median_gap": gap,
        "block_bootstrap_gap": quantiles(gaps),
        "fact_blocks": len(positive_blocks),
        "control_fact_pair_blocks": len(control_blocks),
    }


def relation_neighbourhood(
    metadata: list[dict], cosine: np.ndarray, rng: np.random.Generator, trials: int
) -> dict:
    pairs = []
    for left in range(len(metadata)):
        for right in range(left + 1, len(metadata)):
            a, b = metadata[left], metadata[right]
            if a["layer"] == b["layer"] and a["fact"] != b["fact"]:
                pairs.append((left, right, float(cosine[left, right])))
    scores = np.array([item[2] for item in pairs])
    labels = np.array(
        [metadata[left]["relation"] == metadata[right]["relation"] for left, right, _ in pairs]
    )
    observed = roc_auc(scores, labels)

    blocks_by_stratum: dict[tuple[int, int], list[tuple]] = defaultdict(list)
    rows_by_block: dict[tuple[int, tuple], list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        key = (row["layer"], row["fact"])
        rows_by_block[key].append(index)
    for (layer, fact), indices in rows_by_block.items():
        blocks_by_stratum[(layer, len(indices))].append((layer, fact))
    original_label = {
        block: metadata[indices[0]]["relation"] for block, indices in rows_by_block.items()
    }
    null = []
    for _ in range(trials):
        permuted = {}
        for blocks in blocks_by_stratum.values():
            labels_for_blocks = [original_label[block] for block in blocks]
            rng.shuffle(labels_for_blocks)
            permuted.update(zip(blocks, labels_for_blocks))
        row_labels = [permuted[(row["layer"], row["fact"])] for row in metadata]
        permuted_pairs = np.array([row_labels[left] == row_labels[right] for left, right, _ in pairs])
        if permuted_pairs.any() and (~permuted_pairs).any():
            null.append(roc_auc(scores, permuted_pairs))
    return {
        "exact_layer_pair_auc": observed,
        "same_relation_pairs": int(labels.sum()),
        "different_relation_pairs": int((~labels).sum()),
        "same_relation_cosine": quantiles(scores[labels]),
        "different_relation_cosine": quantiles(scores[~labels]),
        "layer_and_block_size_stratified_permutation": {
            "trials": len(null),
            "auc": quantiles(null),
            "one_sided_empirical_p": (1 + sum(value >= observed for value in null)) / (len(null) + 1),
        },
    }


def layer_prior(metadata: list[dict], train_indices: list[int], query_indices: list[int]) -> list[str | None]:
    counts: dict[int, Counter] = defaultdict(Counter)
    for index in train_indices:
        counts[metadata[index]["layer"]][metadata[index]["relation"]] += 1
    predictions = []
    for index in query_indices:
        layer_counts = counts.get(metadata[index]["layer"])
        predictions.append(
            sorted(layer_counts, key=lambda relation: (-layer_counts[relation], relation))[0]
            if layer_counts
            else None
        )
    return predictions


def neighbour_retrieval(metadata: list[dict], cosine: np.ndarray, relations: list[str]) -> dict:
    train = [index for index, row in enumerate(metadata) if row["split"] == "train"]
    assessment = [index for index, row in enumerate(metadata) if row["split"] != "train"]
    truth = [metadata[index]["relation"] for index in assessment]
    predictions = []
    ranked_families = []
    refusals = []
    for query in assessment:
        candidates = [
            index
            for index in train
            if metadata[index]["layer"] == metadata[query]["layer"]
            and metadata[index]["fact"] != metadata[query]["fact"]
        ]
        if not candidates:
            predictions.append(None)
            ranked_families.append([])
            refusals.append(metadata[query]["edge_id"])
            continue
        family_scores: dict[str, float] = {}
        for candidate in candidates:
            relation = metadata[candidate]["relation"]
            family_scores[relation] = max(
                family_scores.get(relation, -math.inf), float(cosine[query, candidate])
            )
        ranked = sorted(family_scores, key=lambda relation: (-family_scores[relation], relation))
        predictions.append(ranked[0])
        ranked_families.append(ranked)
    top_k = {
        str(k): {
            "retention": sum(target in ranked[:k] for target, ranked in zip(truth, ranked_families))
            / len(truth),
            "mean_candidate_fraction": sum(min(k, len(ranked)) / len(relations) for ranked in ranked_families)
            / len(truth),
        }
        for k in range(1, len(relations) + 1)
    }
    prior_predictions = layer_prior(metadata, train, assessment)
    accuracy = balanced_accuracy(truth, predictions, relations)
    prior = balanced_accuracy(truth, prior_predictions, relations)
    return {
        "training_rows": len(train),
        "assessment_rows": len(assessment),
        "query_coverage": 1.0 - len(refusals) / len(assessment),
        "refused_edge_ids": refusals,
        "top_k": top_k,
        "top1_balanced_accuracy": accuracy,
        "layer_prior_balanced_accuracy": prior,
        "improvement_over_layer_prior": accuracy["value"] - prior["value"],
    }


def relation_subspaces(
    metadata: list[dict], vectors: np.ndarray, relations: list[str], ranks: list[int], rng: np.random.Generator
) -> dict:
    train = [index for index, row in enumerate(metadata) if row["split"] == "train"]
    assessment = [index for index, row in enumerate(metadata) if row["split"] != "train"]
    truth = [metadata[index]["relation"] for index in assessment]
    bases = {}
    for relation in relations:
        rows = vectors[[index for index in train if metadata[index]["relation"] == relation]]
        _, _, right = np.linalg.svd(rows, full_matrices=False)
        bases[relation] = right
    prior_predictions = layer_prior(metadata, train, assessment)
    prior = balanced_accuracy(truth, prior_predictions, relations)
    outputs = []
    hidden = vectors.shape[1]
    for requested_rank in ranks:
        effective = {relation: min(requested_rank, bases[relation].shape[0]) for relation in relations}
        predictions = []
        margins = []
        correct_projection = []
        wrong_projection = []
        for index in assessment:
            scores = {
                relation: float(np.square(bases[relation][: effective[relation]] @ vectors[index]).sum())
                for relation in relations
            }
            ranked = sorted(scores, key=lambda relation: (-scores[relation], relation))
            predictions.append(ranked[0])
            correct = scores[metadata[index]["relation"]]
            wrong = max(value for relation, value in scores.items() if relation != metadata[index]["relation"])
            correct_projection.append(correct)
            wrong_projection.append(wrong)
            margins.append(correct - wrong)

        random_bases = {}
        for relation in relations:
            gaussian = rng.normal(size=(hidden, effective[relation]))
            q, _ = np.linalg.qr(gaussian)
            random_bases[relation] = q.T
        random_predictions = []
        random_correct = []
        for index in assessment:
            scores = {
                relation: float(np.square(random_bases[relation] @ vectors[index]).sum())
                for relation in relations
            }
            random_predictions.append(sorted(scores, key=lambda relation: (-scores[relation], relation))[0])
            random_correct.append(scores[metadata[index]["relation"]])
        accuracy = balanced_accuracy(truth, predictions, relations)
        outputs.append(
            {
                "requested_rank": requested_rank,
                "effective_rank_by_relation": effective,
                "balanced_accuracy": accuracy,
                "layer_prior_balanced_accuracy": prior,
                "improvement_over_layer_prior": accuracy["value"] - prior["value"],
                "correct_projection_mass": quantiles(correct_projection),
                "strongest_wrong_projection_mass": quantiles(wrong_projection),
                "correct_minus_wrong_projection": quantiles(margins),
                "matched_rank_random": {
                    "balanced_accuracy": balanced_accuracy(truth, random_predictions, relations),
                    "correct_projection_mass": quantiles(random_correct),
                },
            }
        )
    return {"training_rows": len(train), "assessment_rows": len(assessment), "ranks": outputs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"GW-4A output already exists: {args.output}")

    validation = validate_preregistration(args.preregistration)
    preregistration = json.loads(args.preregistration.read_text())
    operating = preregistration["operating_points"]
    rng = np.random.default_rng(operating["random_seed"])
    metadata, vectors, inputs = load_sites(args.preregistration, preregistration)
    cosine = vectors @ vectors.T

    prompt = prompt_repeatability(metadata, cosine, rng, operating["bootstrap_trials"])
    neighbourhood = relation_neighbourhood(
        metadata, cosine, rng, operating["permutation_trials"]
    )
    relations = preregistration["cohort"]["semantic_relations"]
    retrieval = neighbour_retrieval(metadata, cosine, relations)
    subspaces = relation_subspaces(
        metadata, vectors, relations, operating["subspace_ranks"], rng
    )

    random_cosines = []
    for index, row in enumerate(metadata):
        direction = rng.normal(size=vectors.shape[1])
        direction /= np.linalg.norm(direction)
        matched_energy = direction * row["write_norm"]
        random_cosines.append(float(vectors[index] @ (matched_energy / np.linalg.norm(matched_energy))))

    gates = preregistration["gates"]
    prompt_gate = gates["prompt_stable_fact_direction"]
    prompt_pass = (
        prompt["same_fact_different_prompt_exact_layer"]["median"]
        >= prompt_gate["minimum_exact_layer_same_fact_median_cosine"]
        and prompt["median_gap"] >= prompt_gate["minimum_median_gap_over_same_relation_control"]
    )
    direction_gate = gates["relation_direction"]
    direction_pass = (
        neighbourhood["exact_layer_pair_auc"] >= direction_gate["minimum_exact_layer_pair_auc"]
        and neighbourhood["layer_and_block_size_stratified_permutation"]["one_sided_empirical_p"]
        <= direction_gate["maximum_permutation_p"]
    )
    retrieval_gate = gates["neighbour_retrieval"]
    retrieval_pass = (
        retrieval["top1_balanced_accuracy"]["value"]
        >= retrieval_gate["minimum_top1_balanced_accuracy"]
        and retrieval["improvement_over_layer_prior"]
        >= retrieval_gate["minimum_improvement_over_layer_prior"]
        and retrieval["query_coverage"] >= retrieval_gate["minimum_query_coverage"]
    )
    subspace_gate = gates["shared_subspace"]
    qualifying_subspaces = [
        item["requested_rank"]
        for item in subspaces["ranks"]
        if item["balanced_accuracy"]["value"] >= subspace_gate["minimum_balanced_accuracy"]
        and item["improvement_over_layer_prior"]
        >= subspace_gate["minimum_improvement_over_layer_prior"]
        and item["correct_minus_wrong_projection"]["mean"]
        >= subspace_gate["minimum_mean_correct_minus_wrong_projection"]
    ]
    direct_relation_pass = direction_pass and retrieval_pass
    open_gw4b = prompt_pass or direct_relation_pass or bool(qualifying_subspaces)

    report = {
        "schema": "larql.gw4a.write-family-report.v1",
        "preregistration": {
            "path": str(args.preregistration),
            "identity_sha256": validation["preregistration_sha256"],
            "file_sha256": sha256_file(args.preregistration),
        },
        "claim_boundary": "observational carrier-write family structure; no causal claim",
        "inputs": inputs,
        "denominators": {
            "sites": len(metadata),
            "semantic_triples": len({tuple(row["fact"]) for row in metadata}),
            "relations": Counter(row["relation"] for row in metadata),
            "splits": Counter(row["split"] for row in metadata),
            "causal_status": Counter(row["causal_status"] for row in metadata),
        },
        "write_norm": quantiles(row["write_norm"] for row in metadata),
        "write_norm_relative_error": quantiles(row["write_norm_relative_error"] for row in metadata),
        "prompt_repeatability": prompt,
        "relation_neighbourhood": neighbourhood,
        "held_out_family_retrieval": retrieval,
        "relation_subspaces": subspaces,
        "matched_energy_random_write_cosine": quantiles(random_cosines),
        "site_evidence": metadata,
        "decision": {
            "prompt_stable_fact_direction_passed": prompt_pass,
            "relation_direction_passed": direction_pass,
            "neighbour_retrieval_passed": retrieval_pass,
            "direct_relation_conditioned_delta_index_passed": direct_relation_pass,
            "qualifying_subspace_ranks": qualifying_subspaces,
            "shared_subspace_passed": bool(qualifying_subspaces),
            "open_gw4b": open_gw4b,
            "otherwise": "move to multi-site trajectories or semantic basins",
        },
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
