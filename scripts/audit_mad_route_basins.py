#!/usr/bin/env python3
"""MAD-V3-ROUTE-1 successful-basin audit.

Communities are discovered from residual cosine geometry only, separately
within each operation so the test asks about route multiplicity rather than
rediscovering operation labels. Labels are then audited against held-out
graphs, nuisance variables, answer identity and future FFN contributions.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.metrics import adjusted_rand_score


SCHEMA = "larql.mad-v3-route-1.basin-audit.v1"


def normalized(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= np.finfo(np.float64).eps).any():
        raise ValueError("capture contains zero or non-finite residuals")
    return values / norms


def parse_layers(spec: str, available: list[int]) -> list[int]:
    selected = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(part))
    selected = sorted(set(selected))
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"capture lacks frontier layers {missing}")
    return selected


def mutual_graph(matrix: np.ndarray, k: int) -> nx.Graph:
    count = len(matrix)
    graph = nx.Graph()
    graph.add_nodes_from(range(count))
    if count < 2:
        return graph
    scores = matrix @ matrix.T
    np.fill_diagonal(scores, -np.inf)
    effective_k = min(k, count - 1)
    nearest = []
    for row in range(count):
        order = np.lexsort((np.arange(count), -scores[row]))[:effective_k]
        nearest.append(set(int(value) for value in order))
    for left in range(count):
        for right in nearest[left]:
            if left < right and left in nearest[right]:
                graph.add_edge(left, right, weight=max(float(scores[left, right]), 0.0) + 1e-9)
    return graph


def communities(matrix: np.ndarray, k: int, seed: int) -> np.ndarray:
    graph = mutual_graph(matrix, k)
    if graph.number_of_edges() == 0:
        return np.arange(len(matrix), dtype=np.int32)
    groups = nx.community.louvain_communities(graph, weight="weight", seed=seed)
    labels = np.empty(len(matrix), dtype=np.int32)
    for label, members in enumerate(sorted(groups, key=lambda group: min(group))):
        for member in members:
            labels[member] = label
    return labels


def entropy_summary(values) -> dict:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return {"categories": 0, "normalized_entropy": 0.0, "max_share": 1.0}
    probs = np.array(list(counts.values()), dtype=np.float64) / total
    entropy = float(-(probs * np.log(probs)).sum())
    ceiling = math.log(min(total, len(counts))) if min(total, len(counts)) > 1 else 0.0
    return {
        "categories": len(counts),
        "normalized_entropy": entropy / ceiling if ceiling else 0.0,
        "max_share": max(counts.values()) / total,
    }


def answer_key(sample) -> tuple[int, ...]:
    values = sample.get("expected_token_ids")
    if not values:
        raise ValueError(f"sample {sample['id']} lacks expected_token_ids")
    return tuple(values)


def discover_layer(samples, residuals, layer_slot, history, k, seed):
    labels = {}
    by_operation = defaultdict(list)
    for sample in history:
        by_operation[samples[sample]["operation_id"]].append(sample)
    operation_reports = {}
    next_label = 0
    for operation, members in sorted(by_operation.items()):
        matrix = normalized(np.asarray(residuals[members, layer_slot, :]))
        local = communities(matrix, k, seed)
        mapping = {}
        for local_label in sorted(set(int(value) for value in local)):
            mapping[local_label] = next_label
            next_label += 1
        for sample, local_label in zip(members, local):
            labels[sample] = mapping[int(local_label)]
        operation_reports[operation] = {
            "samples": len(members),
            "neighbors_requested": k,
            "neighbors_effective": min(k, max(0, len(members) - 1)),
            "knn_saturated": len(members) <= k + 1,
            "communities": len(mapping),
            "community_sizes": sorted(Counter(int(v) for v in local).values(), reverse=True),
        }
    return labels, operation_reports


def bootstrap_stability(samples, residuals, layer_slot, history, baseline, k, draws, seed):
    rng = np.random.default_rng(seed)
    scores = []
    by_operation = defaultdict(list)
    for sample in history:
        by_operation[samples[sample]["operation_id"]].append(sample)
    for draw in range(draws):
        operation_scores = []
        for operation, members in sorted(by_operation.items()):
            size = max(4, math.ceil(0.8 * len(members)))
            chosen = sorted(rng.choice(members, size=min(size, len(members)), replace=False))
            matrix = normalized(np.asarray(residuals[chosen, layer_slot, :]))
            local = communities(matrix, k, seed + draw + 1)
            truth = [baseline[sample] for sample in chosen]
            trial = [int(label) for label in local]
            if len(set(truth)) > 1 and len(set(trial)) > 1:
                operation_scores.append(adjusted_rand_score(truth, trial))
        if operation_scores:
            scores.append(float(np.mean(operation_scores)))
    return {
        "draws": len(scores),
        "mean_ari": float(np.mean(scores)) if scores else None,
        "median_ari": float(np.median(scores)) if scores else None,
        "p10_ari": float(np.quantile(scores, 0.1)) if scores else None,
    }


def mixing_report(samples, history, labels, min_community):
    members = defaultdict(list)
    for sample in history:
        members[labels[sample]].append(sample)
    reports = []
    for label, rows in sorted(members.items()):
        if len(rows) < min_community:
            continue
        token_counts = [samples[row].get("token_count") for row in rows]
        reports.append(
            {
                "basin": label,
                "size": len(rows),
                "operation": samples[rows[0]]["operation_id"],
                "answer": entropy_summary(answer_key(samples[row]) for row in rows),
                "wording": entropy_summary(samples[row]["wording_id"] for row in rows),
                "alias": entropy_summary(samples[row].get("alias_id") for row in rows),
                "graph": entropy_summary(samples[row].get("group_id") for row in rows),
                "layout": entropy_summary(samples[row].get("layout_id") for row in rows),
                "token_count": entropy_summary(token_counts),
            }
        )
    answer_assignments = defaultdict(list)
    for sample in history:
        answer_assignments[
            (samples[sample]["operation_id"], answer_key(samples[sample]))
        ].append(labels[sample])
    per_operation = defaultdict(list)
    for sample in history:
        per_operation[samples[sample]["operation_id"]].append(labels[sample])
    route_entropy = {
        operation: entropy_summary(values) for operation, values in sorted(per_operation.items())
    }
    answer_route_entropy = [
        entropy_summary(assignments) for assignments in answer_assignments.values()
    ]
    return (
        reports,
        sum(len(set(assignments)) > 1 for assignments in answer_assignments.values()),
        len(answer_assignments),
        route_entropy,
        {
            "mean_normalized_entropy": float(
                np.mean([row["normalized_entropy"] for row in answer_route_entropy])
            ),
            "mean_basins": float(np.mean([row["categories"] for row in answer_route_entropy])),
        },
    )


def nuisance_mixed(community):
    thresholds = {
        "answer": (3, 0.65, 0.50),
        "wording": (3, 0.65, 0.50),
        "alias": (2, 0.65, 0.67),
        "graph": (3, 0.65, 0.50),
        "layout": (3, 0.65, 0.50),
        "token_count": (2, 0.50, 0.75),
    }
    return all(
        community[field]["categories"] >= categories
        and community[field]["normalized_entropy"] >= entropy
        and community[field]["max_share"] <= max_share
        for field, (categories, entropy, max_share) in thresholds.items()
    )


def signature_test(samples, contributions, object_columns, history, labels, permutations, seed):
    matrix = np.asarray(contributions[np.ix_(history, object_columns)], dtype=np.float64)
    totals = matrix.sum(axis=1, keepdims=True)
    valid = totals[:, 0] > 0
    matrix[valid] /= totals[valid]
    strata = defaultdict(list)
    for row, sample in enumerate(history):
        strata[(samples[sample]["operation_id"], answer_key(samples[sample]))].append(row)
    residualized = matrix.copy()
    for rows in strata.values():
        residualized[rows] -= residualized[rows].mean(axis=0)
    basin_labels = np.array([labels[sample] for sample in history], dtype=np.int32)

    def between_score(candidate):
        total = float(np.square(residualized).sum())
        if total <= np.finfo(np.float64).eps:
            return 0.0
        between = 0.0
        for label in np.unique(candidate):
            rows = np.flatnonzero(candidate == label)
            mean = residualized[rows].mean(axis=0)
            between += len(rows) * float(np.square(mean).sum())
        return between / total

    observed = between_score(basin_labels)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        shuffled = basin_labels.copy()
        for rows in strata.values():
            shuffled[rows] = rng.permutation(shuffled[rows])
        exceedances += between_score(shuffled) >= observed
    return {
        "answer_conditioned_between_basin_r2": observed,
        "permutations": permutations,
        "exceedances": exceedances,
        "p_value_plus_one": (exceedances + 1) / (permutations + 1),
    }


def rank_density(profile, byte_counts):
    return np.lexsort((np.arange(len(profile)), -(profile / byte_counts)))


def coverage(truth, ranking, byte_counts, budget):
    limit = budget * float(byte_counts.sum())
    used = 0
    covered = 0.0
    for column in ranking:
        cost = int(byte_counts[column])
        if used + cost > limit:
            continue
        used += cost
        covered += float(truth[column])
    total = float(truth.sum())
    return {
        "candidate_byte_fraction": used / float(byte_counts.sum()),
        "contribution_coverage": covered / total if total else 0.0,
    }


def address_test(
    samples,
    residuals,
    contributions,
    layer_slot,
    history,
    queries,
    labels,
    object_columns,
    byte_counts,
    neighbors,
    budgets,
):
    history_matrix = normalized(np.asarray(residuals[history, layer_slot, :]))
    query_matrix = normalized(np.asarray(residuals[queries, layer_slot, :]))
    history_row = {sample: row for row, sample in enumerate(history)}
    popularity = np.asarray(contributions[np.ix_(history, object_columns)]).mean(axis=0)
    accum = {
        name: {budget: [] for budget in budgets}
        for name in ("popularity", "plain_knn", "operation_knn", "basin_knn", "oracle")
    }
    assignments = Counter()
    for query_row, query in enumerate(queries):
        scores = history_matrix @ query_matrix[query_row]
        eligible = [
            sample
            for sample in history
            if samples[sample].get("group_id") != samples[query].get("group_id")
        ]
        operation_rows = [
            sample
            for sample in eligible
            if samples[sample]["operation_id"] == samples[query]["operation_id"]
        ]
        nearest_operation = max(operation_rows, key=lambda sample: scores[history_row[sample]])
        assigned = labels[nearest_operation]
        assignments[assigned] += 1
        basin_rows = [sample for sample in operation_rows if labels[sample] == assigned]

        def profile(rows):
            chosen = sorted(rows, key=lambda sample: (-scores[history_row[sample]], sample))[
                :neighbors
            ]
            return np.asarray(contributions[np.ix_(chosen, object_columns)]).mean(axis=0)

        predictors = {
            "popularity": popularity,
            "plain_knn": profile(eligible),
            "operation_knn": profile(operation_rows),
            "basin_knn": profile(basin_rows),
            "oracle": np.asarray(contributions[query, object_columns]),
        }
        truth = np.asarray(contributions[query, object_columns], dtype=np.float64)
        for name, predictor in predictors.items():
            ranking = rank_density(np.asarray(predictor, dtype=np.float64), byte_counts)
            for budget in budgets:
                accum[name][budget].append(coverage(truth, ranking, byte_counts, budget))
    report = {}
    for name, budget_rows in accum.items():
        report[name] = {}
        for budget, rows in budget_rows.items():
            report[name][str(budget)] = {
                key: float(np.mean([row[key] for row in rows])) for key in rows[0]
            }
    gains = {}
    for budget in budgets:
        key = str(budget)
        basin = report["basin_knn"][key]["contribution_coverage"]
        operation = report["operation_knn"][key]["contribution_coverage"]
        gains[key] = basin - operation
    return {
        "queries": len(queries),
        "basin_assignments": dict(assignments),
        "methods": report,
        "basin_minus_operation_coverage": gains,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--frontier-layers", default="36-45")
    parser.add_argument("--target-layer", type=int, default=49)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--min-community", type=int, default=8)
    parser.add_argument("--bootstraps", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--byte-budgets", default="0.05,0.2")
    parser.add_argument("--seed", type=int, default=0x524F55544531)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.neighbors, args.min_community, args.bootstraps, args.permutations) <= 0:
        raise ValueError("count arguments must be positive")

    manifest = json.loads((args.capture / "manifest.json").read_text())
    samples = manifest["samples"]
    layers = manifest["residual_layers"]
    frontier = parse_layers(args.frontier_layers, layers)
    hidden = manifest["hidden_size"]
    residuals = np.memmap(
        args.capture / manifest["residuals_file"],
        dtype="<f4",
        mode="r",
        shape=(len(samples), len(layers), hidden),
    )
    contributions = np.memmap(
        args.capture / manifest["contributions_file"],
        dtype="<f4",
        mode="r",
        shape=(len(samples), len(manifest["objects"])),
    )
    scored = [sample.get("answer_exact") is not None for sample in samples]
    if not all(scored):
        raise ValueError("ROUTE-1 requires answer-scored captures; recapture this corpus")
    history_all = [i for i, sample in enumerate(samples) if sample["split"] == "history"]
    query_all = [i for i, sample in enumerate(samples) if sample["split"] == "query"]
    history = [sample for sample in history_all if samples[sample]["answer_exact"]]
    queries = [sample for sample in query_all if samples[sample]["answer_exact"]]
    if not history or not queries:
        raise ValueError("ROUTE-1 requires successful executions in both partitions")
    target_objects = [
        index
        for index, obj in enumerate(manifest["objects"])
        if obj["layer"] == args.target_layer
    ]
    if not target_objects:
        raise ValueError(f"capture has no contribution objects at layer {args.target_layer}")
    byte_counts = np.array(
        [manifest["objects"][index]["byte_count"] for index in target_objects],
        dtype=np.float64,
    )
    budgets = [float(value) for value in args.byte_budgets.split(",")]

    layer_reports = []
    for frontier_layer in frontier:
        slot = layers.index(frontier_layer)
        labels, operation_reports = discover_layer(
            samples, residuals, slot, history, args.neighbors, args.seed
        )
        underpowered = sorted(
            operation
            for operation, report in operation_reports.items()
            if report["knn_saturated"]
            or report["samples"] < 2 * args.min_community
        )
        mixing, spanning, answer_cells, route_entropy, answer_route_entropy = mixing_report(
            samples, history, labels, args.min_community
        )
        mixed = [
            community
            for community in mixing
            if nuisance_mixed(community)
            and community["operation"] not in underpowered
        ]
        layer_reports.append(
            {
                "layer": frontier_layer,
                "operations": operation_reports,
                "route_gate_evaluable": not underpowered,
                "underpowered_operations": underpowered,
                "bootstrap_stability": bootstrap_stability(
                    samples,
                    residuals,
                    slot,
                    history,
                    labels,
                    args.neighbors,
                    args.bootstraps,
                    args.seed + frontier_layer * 1000,
                ),
                "size_eligible_communities": len(mixing),
                "nuisance_mixed_communities": len(mixed),
                "community_mixing": mixing,
                "answer_cells_spanning_multiple_basins": spanning,
                "answer_cells": answer_cells,
                "route_entropy_by_operation": route_entropy,
                "answer_route_entropy": answer_route_entropy,
                "downstream_signature": signature_test(
                    samples,
                    contributions,
                    target_objects,
                    history,
                    labels,
                    args.permutations,
                    args.seed + frontier_layer * 2000,
                ),
                "future_addressability": address_test(
                    samples,
                    residuals,
                    contributions,
                    slot,
                    history,
                    queries,
                    labels,
                    target_objects,
                    byte_counts,
                    args.neighbors,
                    budgets,
                ),
            }
        )

    for layer in layer_reports:
        raw = layer["downstream_signature"]["p_value_plus_one"]
        layer["downstream_signature"]["bonferroni_p_frontier"] = min(
            1.0, raw * len(frontier)
        )

    report = {
        "schema": SCHEMA,
        "capture": str(args.capture),
        "preregistered_frontier_layers": frontier,
        "target_layer": args.target_layer,
        "neighbors": args.neighbors,
        "correctness": {
            "history_correct": len(history),
            "history_total": len(history_all),
            "query_correct": len(queries),
            "query_total": len(query_all),
            "history_exact_rate": len(history) / len(history_all),
            "query_exact_rate": len(queries) / len(query_all),
        },
        "interpretation_guard": (
            "Communities separating correct from incorrect executions are not successful "
            "basin multiplicity; discovery and scoring here use correct executions only."
        ),
        "layers": layer_reports,
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
        print(f"wrote {args.output}")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
