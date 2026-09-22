#!/usr/bin/env python3
"""MAD-V3-STATE-1 local computational-equivalence audit."""

import argparse
import json
from pathlib import Path

import numpy as np

import audit_mad_field as field


SCHEMA = "larql.mad-v3-state-1.audit.v1"
DEFAULT_K = (1, 2, 4, 8, 16, 32, 64, 128)


def wording_family(sample: dict) -> str:
    return sample.get("wording_id", "").split(":alias:", 1)[0]


def cohort_composition(samples, history, queries, selections) -> dict:
    operation = []
    wording = []
    alias = []
    token_near = []
    for query_row, selected_rows in enumerate(selections):
        query = samples[queries[query_row]]
        selected = [samples[history[int(row)]] for row in selected_rows]
        operation.append(
            np.mean([row.get("operation_id") == query.get("operation_id") for row in selected])
        )
        wording.append(
            np.mean([wording_family(row) == wording_family(query) for row in selected])
        )
        alias.append(
            np.mean([row.get("alias_id") == query.get("alias_id") for row in selected])
        )
        token_near.append(
            np.mean(
                [
                    abs(int(row.get("token_count", 0)) - int(query.get("token_count", 0)))
                    <= 1
                    for row in selected
                ]
            )
        )
    return {
        "operation_purity": float(np.mean(operation)),
        "wording_family_purity": float(np.mean(wording)),
        "exact_alias_purity": float(np.mean(alias)),
        "token_count_within_one": float(np.mean(token_near)),
    }


def prediction_report(predictions, truths, byte_counts, budgets):
    coverage_rows = {str(budget): [] for budget in budgets}
    vector_rows = []
    per_query = {str(budget): [] for budget in budgets}
    for prediction, truth in zip(predictions, truths):
        vector_rows.append(field.vector_metrics(truth, prediction))
        ranking = field.rank_density(prediction, byte_counts)
        for budget in budgets:
            key = str(budget)
            row = field.coverage(truth, ranking, byte_counts, budget)
            coverage_rows[key].append(row)
            per_query[key].append(row["contribution_coverage"])
    report = {
        "vector": field.mean_rows(vector_rows),
        "budgets": {
            key: field.mean_rows(rows) for key, rows in coverage_rows.items()
        },
    }
    return report, {key: np.asarray(rows) for key, rows in per_query.items()}


def add_headroom(report, popularity_report, oracle_report, budgets):
    recovered = {}
    for budget in budgets:
        key = str(budget)
        base = popularity_report["budgets"][key]["contribution_coverage"]
        ceiling = oracle_report["budgets"][key]["contribution_coverage"]
        value = report["budgets"][key]["contribution_coverage"]
        recovered[key] = (value - base) / (ceiling - base) if ceiling > base else None
    report["oracle_headroom_recovered"] = recovered


def neighbor_rankings(samples, residuals, layers, history, queries):
    rankings = np.empty((len(layers), len(queries), len(history)), dtype=np.int32)
    history_groups = np.array([samples[index].get("group_id") for index in history])
    for layer_row, layer_slot in enumerate(layers):
        history_matrix = field.normalized(np.asarray(residuals[history, layer_slot, :]))
        query_matrix = field.normalized(np.asarray(residuals[queries, layer_slot, :]))
        scores = query_matrix @ history_matrix.T
        for query_row, query in enumerate(queries):
            scores[query_row, history_groups == samples[query].get("group_id")] = -np.inf
        rankings[layer_row] = np.argsort(-scores, axis=1, kind="stable")
    return rankings


def stability_report(rankings: np.ndarray, k: int) -> dict:
    adjacent_jaccard = []
    adjacent_retention = []
    union_sizes = []
    intersection_sizes = []
    for query_row in range(rankings.shape[1]):
        sets = [set(rankings[layer, query_row, :k]) for layer in range(rankings.shape[0])]
        adjacent_jaccard.extend(
            len(left & right) / len(left | right)
            for left, right in zip(sets, sets[1:])
        )
        adjacent_retention.extend(
            len(left & right) / k for left, right in zip(sets, sets[1:])
        )
        union_sizes.append(len(set.union(*sets)))
        intersection_sizes.append(len(set.intersection(*sets)))
    retention = float(np.mean(adjacent_retention))
    chance_retention = k / rankings.shape[2]
    return {
        "mean_adjacent_jaccard": float(np.mean(adjacent_jaccard)),
        "mean_adjacent_retention": retention,
        "chance_adjacent_retention": chance_retention,
        "chance_adjusted_adjacent_retention": (
            (retention - chance_retention) / (1.0 - chance_retention)
        ),
        "mean_ten_layer_union_size": float(np.mean(union_sizes)),
        "mean_ten_layer_intersection_size": float(np.mean(intersection_sizes)),
    }


def persistent_selection(rankings: np.ndarray, k: int) -> tuple[np.ndarray, dict]:
    layers, queries, history_size = rankings.shape
    selections = np.empty((queries, k), dtype=np.int32)
    selected_strength = []
    selected_all_layers = []
    candidate_union = []
    for query_row in range(queries):
        counts = np.zeros(history_size, dtype=np.int32)
        rank_sum = np.zeros(history_size, dtype=np.int64)
        for layer in range(layers):
            ranking = rankings[layer, query_row]
            counts[ranking[:k]] += 1
            inverse = np.empty(history_size, dtype=np.int32)
            inverse[ranking] = np.arange(history_size, dtype=np.int32)
            rank_sum += inverse
        chosen = sorted(
            range(history_size), key=lambda row: (-counts[row], rank_sum[row], row)
        )[:k]
        selections[query_row] = chosen
        strengths = counts[chosen]
        selected_strength.append(float(np.mean(strengths) / layers))
        selected_all_layers.append(float(np.mean(strengths == layers)))
        candidate_union.append(int(np.count_nonzero(counts)))
    return selections, {
        "mean_selected_layer_fraction": float(np.mean(selected_strength)),
        "mean_selected_all_layers_fraction": float(np.mean(selected_all_layers)),
        "mean_candidate_union_size": float(np.mean(candidate_union)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--frontier-layers", default="36-45")
    parser.add_argument("--target-layer", type=int, default=49)
    parser.add_argument("--k-values", default=",".join(str(value) for value in DEFAULT_K))
    parser.add_argument("--persistence-k", type=int, default=16)
    parser.add_argument("--baseline-layer", type=int, default=38)
    parser.add_argument("--byte-budgets", default="0.05,0.2")
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0x535441544531)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.persistence_k, args.permutations) <= 0:
        raise ValueError("count arguments must be positive")
    k_values = [int(value) for value in args.k_values.split(",")]
    if len(k_values) != len(set(k_values)) or min(k_values) <= 0:
        raise ValueError("K values must be unique positive integers")
    if 16 not in k_values:
        raise ValueError("STATE-1 scale comparisons require K=16")

    manifest_path = args.capture / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("capture is incomplete: manifest.json is absent")
    manifest = json.loads(manifest_path.read_text())
    samples = manifest["samples"]
    available_layers = manifest["residual_layers"]
    frontier = field.parse_layers(args.frontier_layers, available_layers)
    if args.baseline_layer not in frontier:
        raise ValueError("persistence baseline layer must belong to the frontier")
    layer_slots = [available_layers.index(layer) for layer in frontier]
    budgets = [float(value) for value in args.byte_budgets.split(",")]
    history = [
        index
        for index, sample in enumerate(samples)
        if sample["split"] == "history" and sample.get("answer_exact") is True
    ]
    queries = [
        index
        for index, sample in enumerate(samples)
        if sample["split"] == "query" and sample.get("answer_exact") is True
    ]
    if max(k_values + [args.persistence_k]) > len(history):
        raise ValueError("requested K exceeds successful history population")
    object_columns = [
        index
        for index, obj in enumerate(manifest["objects"])
        if obj["layer"] == args.target_layer
    ]
    if not object_columns:
        raise ValueError(f"capture has no contribution objects at layer {args.target_layer}")
    byte_counts = np.array(
        [manifest["objects"][column]["byte_count"] for column in object_columns],
        dtype=np.float64,
    )
    residual_shape = (len(samples), len(available_layers), manifest["hidden_size"])
    contribution_shape = (len(samples), len(manifest["objects"]))
    residual_path = args.capture / manifest["residuals_file"]
    contribution_path = args.capture / manifest["contributions_file"]
    if residual_path.stat().st_size != int(np.prod(residual_shape)) * 4:
        raise ValueError("residual plane size does not match manifest")
    if contribution_path.stat().st_size != int(np.prod(contribution_shape)) * 4:
        raise ValueError("contribution plane size does not match manifest")
    residuals = np.memmap(residual_path, dtype="<f4", mode="r", shape=residual_shape)
    contributions = np.memmap(
        contribution_path, dtype="<f4", mode="r", shape=contribution_shape
    )
    truths = np.asarray(contributions[np.ix_(queries, object_columns)], dtype=np.float64)
    history_targets = np.asarray(
        contributions[np.ix_(history, object_columns)], dtype=np.float64
    )
    popularity_predictions = np.repeat(
        history_targets.mean(axis=0, keepdims=True), len(queries), axis=0
    )
    popularity_report, _ = prediction_report(
        popularity_predictions, truths, byte_counts, budgets
    )
    oracle_report, _ = prediction_report(truths, truths, byte_counts, budgets)
    add_headroom(popularity_report, popularity_report, oracle_report, budgets)
    add_headroom(oracle_report, popularity_report, oracle_report, budgets)

    rankings = neighbor_rankings(
        samples, residuals, layer_slots, history, queries
    )
    scale_comparisons = (len(k_values) - 1) * len(frontier) * len(budgets)
    layer_reports = []
    per_query_by_layer = {}
    for layer_row, layer in enumerate(frontier):
        k_reports = {}
        per_query_by_layer[layer] = {}
        baseline_per_query = None
        for k in k_values:
            selected = rankings[layer_row, :, :k]
            predictions = history_targets[selected].mean(axis=1)
            report, per_query = prediction_report(
                predictions, truths, byte_counts, budgets
            )
            add_headroom(report, popularity_report, oracle_report, budgets)
            report["composition"] = cohort_composition(
                samples, history, queries, selected
            )
            k_reports[str(k)] = report
            per_query_by_layer[layer][k] = per_query
            if k == 16:
                baseline_per_query = per_query
        for k_index, k in enumerate(k_values):
            if k == 16:
                continue
            comparisons = {}
            for budget_index, budget in enumerate(budgets):
                key = str(budget)
                test = field.paired_sign_flip(
                    per_query_by_layer[layer][k][key] - baseline_per_query[key],
                    args.permutations,
                    args.seed + layer_row * 1000 + k_index * 10 + budget_index,
                )
                test["bonferroni_p_scale"] = min(
                    1.0, test["p_value_plus_one"] * scale_comparisons
                )
                comparisons[key] = test
            k_reports[str(k)]["comparison_vs_k16"] = comparisons
        layer_reports.append({"layer": layer, "k": k_reports})

    stability = {
        str(k): stability_report(rankings, k) for k in k_values
    }
    persistent, persistence_topology = persistent_selection(
        rankings, args.persistence_k
    )
    persistent_predictions = history_targets[persistent].mean(axis=1)
    persistent_report, persistent_per_query = prediction_report(
        persistent_predictions, truths, byte_counts, budgets
    )
    add_headroom(persistent_report, popularity_report, oracle_report, budgets)
    persistent_report["composition"] = cohort_composition(
        samples, history, queries, persistent
    )
    persistent_report["topology"] = persistence_topology
    baseline_report = next(
        report for report in layer_reports if report["layer"] == args.baseline_layer
    )["k"][str(args.persistence_k)]
    persistent_report["vector_delta_vs_baseline"] = {
        key: persistent_report["vector"][key] - baseline_report["vector"][key]
        for key in ("cosine", "js_divergence")
    }
    persistent_report["comparison_vs_baseline"] = {}
    for budget_index, budget in enumerate(budgets):
        key = str(budget)
        baseline = per_query_by_layer[args.baseline_layer][args.persistence_k][key]
        test = field.paired_sign_flip(
            persistent_per_query[key] - baseline,
            args.permutations,
            args.seed + 900_000 + budget_index,
        )
        test["bonferroni_p_2"] = min(1.0, test["p_value_plus_one"] * len(budgets))
        persistent_report["comparison_vs_baseline"][key] = test
    persistent_report["coverage_delta_vs_each_layer"] = {
        str(layer): {
            str(budget): persistent_report["budgets"][str(budget)][
                "contribution_coverage"
            ]
            - next(report for report in layer_reports if report["layer"] == layer)["k"]
            [str(args.persistence_k)]["budgets"][str(budget)]["contribution_coverage"]
            for budget in budgets
        }
        for layer in frontier
    }

    result = {
        "schema": SCHEMA,
        "capture": str(args.capture),
        "frontier_layers": frontier,
        "target_layer": args.target_layer,
        "k_values": k_values,
        "persistence_k": args.persistence_k,
        "baseline_layer": args.baseline_layer,
        "byte_budgets": budgets,
        "permutations": args.permutations,
        "scale_multiplicity_comparisons": scale_comparisons,
        "history_successes": len(history),
        "query_successes": len(queries),
        "popularity": popularity_report,
        "oracle": oracle_report,
        "layers": layer_reports,
        "neighbor_stability": stability,
        "persistence": persistent_report,
        "interpretation_guard": (
            "Fold 1 is discovery: K=16 and L38 single-layer performance were known "
            "before STATE-1. Replication across alias folds is required."
        ),
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"wrote {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
