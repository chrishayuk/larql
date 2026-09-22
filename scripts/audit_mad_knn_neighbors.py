#!/usr/bin/env python3
"""Adversarial neighbour audit for a completed MAD-V3 capture.

This keeps the inferential controls outside the model hot path. It recomputes
exact cosine neighbours, reports token-position-matched curves, and performs a
graph-stratified label permutation whose statistic is the maximum purity over
all captured layers.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def normalized(values: np.ndarray) -> np.ndarray:
    norms = np.sqrt(np.square(values.astype(np.float64)).sum(axis=-1, keepdims=True))
    if not np.isfinite(norms).all() or (norms <= np.finfo(np.float64).eps).any():
        raise ValueError("capture contains a zero or non-finite residual")
    return (values.astype(np.float64) / norms).astype(np.float32)


def eligible_rows(samples, history, query, exclude_same_group, tolerance=None):
    query_meta = samples[query]
    rows = []
    for row, sample in enumerate(history):
        history_meta = samples[sample]
        if (
            exclude_same_group
            and query_meta.get("group_id") is not None
            and query_meta.get("group_id") == history_meta.get("group_id")
        ):
            continue
        if tolerance is not None:
            query_tokens = query_meta.get("token_count")
            history_tokens = history_meta.get("token_count")
            if query_tokens is None or history_tokens is None:
                raise ValueError("token-matched audit requires manifest token_count")
            if abs(query_tokens - history_tokens) > tolerance:
                continue
        rows.append(row)
    return rows


def top_rows(scores, eligible, k):
    return sorted(eligible, key=lambda row: (-float(scores[row]), row))[:k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--query-regime", required=True)
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument("--exclude-same-group", action="store_true")
    parser.add_argument("--token-tolerances", default="0,1,2")
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0x4D41445633)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.neighbors <= 0 or args.permutations <= 0:
        raise ValueError("neighbors and permutations must be positive")

    manifest = json.loads((args.capture / "manifest.json").read_text())
    samples = manifest["samples"]
    layers = manifest["residual_layers"]
    hidden = manifest["hidden_size"]
    residual_path = args.capture / manifest["residuals_file"]
    expected_bytes = len(samples) * len(layers) * hidden * 4
    if residual_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"residual plane is {residual_path.stat().st_size} bytes; expected {expected_bytes}"
        )
    residuals = np.memmap(
        residual_path,
        dtype="<f4",
        mode="r",
        shape=(len(samples), len(layers), hidden),
    )
    history = [i for i, sample in enumerate(samples) if sample["split"] == "history"]
    queries = [
        i
        for i, sample in enumerate(samples)
        if sample["split"] == "query" and sample.get("regime") == args.query_regime
    ]
    if not history or not queries:
        raise ValueError("capture does not contain the requested history/query partitions")

    operations = sorted(
        {sample["operation_id"] for sample in samples if sample.get("operation_id")}
    )
    operation_index = {operation: index for index, operation in enumerate(operations)}
    labels = np.array([operation_index[samples[query]["operation_id"]] for query in queries])
    query_axis = np.arange(len(queries))
    counts = np.zeros((len(layers), len(queries), len(operations)), dtype=np.int16)

    for layer_slot, _layer in enumerate(layers):
        history_matrix = normalized(np.asarray(residuals[history, layer_slot, :]))
        query_matrix = normalized(np.asarray(residuals[queries, layer_slot, :]))
        for query_slot, query in enumerate(queries):
            eligible = eligible_rows(
                samples, history, query, args.exclude_same_group
            )
            if len(eligible) < args.neighbors:
                raise ValueError(f"query {samples[query]['id']} has too few eligible rows")
            scores = history_matrix @ query_matrix[query_slot]
            for row in top_rows(scores, eligible, args.neighbors):
                operation = samples[history[row]]["operation_id"]
                counts[layer_slot, query_slot, operation_index[operation]] += 1

    denominator = len(queries) * args.neighbors
    purity = counts[:, query_axis, labels].sum(axis=1) / denominator
    peak_slot = int(np.argmax(purity))
    peak_layer = layers[peak_slot]

    per_operation = {}
    for operation, operation_slot in operation_index.items():
        members = np.flatnonzero(labels == operation_slot)
        per_operation[operation] = float(
            counts[peak_slot, members, operation_slot].sum()
            / (len(members) * args.neighbors)
        )

    matched = []
    tolerances = [int(value) for value in args.token_tolerances.split(",")]
    for tolerance in tolerances:
        curve = []
        matched_queries = 0
        base_total = 0.0
        for layer_slot, layer in enumerate(layers):
            history_matrix = normalized(np.asarray(residuals[history, layer_slot, :]))
            query_matrix = normalized(np.asarray(residuals[queries, layer_slot, :]))
            hits = 0
            used = 0
            layer_base = 0.0
            for query_slot, query in enumerate(queries):
                eligible = eligible_rows(
                    samples,
                    history,
                    query,
                    args.exclude_same_group,
                    tolerance,
                )
                if len(eligible) < args.neighbors:
                    continue
                scores = history_matrix @ query_matrix[query_slot]
                top = top_rows(scores, eligible, args.neighbors)
                operation = samples[query]["operation_id"]
                hits += sum(samples[history[row]]["operation_id"] == operation for row in top)
                layer_base += sum(
                    samples[history[row]]["operation_id"] == operation for row in eligible
                ) / len(eligible)
                used += 1
            curve.append(
                {
                    "layer": layer,
                    "purity": hits / (used * args.neighbors) if used else None,
                    "matched_base_rate": layer_base / used if used else None,
                    "queries": used,
                }
            )
            matched_queries = used
            base_total = layer_base / used if used else 0.0
        valid = [row for row in curve if row["purity"] is not None]
        peak = max(valid, key=lambda row: row["purity"])
        matched.append(
            {
                "token_tolerance": tolerance,
                "queries": matched_queries,
                "matched_base_rate": base_total,
                "peak": peak,
                "curve": curve,
            }
        )

    length_hits = 0
    for query in queries:
        eligible = eligible_rows(samples, history, query, args.exclude_same_group)
        query_tokens = samples[query]["token_count"]
        top = sorted(
            eligible,
            key=lambda row: (
                abs(samples[history[row]]["token_count"] - query_tokens),
                row,
            ),
        )[: args.neighbors]
        operation = samples[query]["operation_id"]
        length_hits += sum(samples[history[row]]["operation_id"] == operation for row in top)

    groups = defaultdict(list)
    for query_slot, query in enumerate(queries):
        groups[samples[query].get("group_id")].append(query_slot)
    rng = np.random.default_rng(args.seed)
    exceed_max = 0
    observed_max = float(purity[peak_slot])
    for _ in range(args.permutations):
        permuted = labels.copy()
        for members in groups.values():
            permuted[members] = rng.permutation(permuted[members])
        null_curve = counts[:, query_axis, permuted].sum(axis=1) / denominator
        exceed_max += float(null_curve.max()) >= observed_max

    report = {
        "schema": "larql.mad-v3-knn.neighbour-audit.v1",
        "capture": str(args.capture),
        "query_regime": args.query_regime,
        "neighbors": args.neighbors,
        "exclude_same_group": args.exclude_same_group,
        "queries": len(queries),
        "curve": [
            {"layer": layer, "operation_purity": float(value)}
            for layer, value in zip(layers, purity)
        ],
        "peak": {
            "layer": peak_layer,
            "operation_purity": observed_max,
            "per_operation": per_operation,
        },
        "token_matched": matched,
        "token_count_only_purity": length_hits / denominator,
        "graph_stratified_max_layer_permutation": {
            "draws": args.permutations,
            "seed": args.seed,
            "exceedances": exceed_max,
            "p_value_plus_one": (exceed_max + 1) / (args.permutations + 1),
        },
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
        print(f"wrote {args.output}")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
