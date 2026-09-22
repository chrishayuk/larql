#!/usr/bin/env python3
"""MAD-V3-PAGE-1 physical-schema audit with a frozen residual index."""

import argparse
import json
from pathlib import Path

import numpy as np

import audit_mad_field as field


SCHEMA = "larql.mad-v3-page-1.audit.v1"


def open_plane(capture: Path, manifest: dict, name: str, shape: tuple[int, ...]):
    path = capture / manifest[name]
    expected = int(np.prod(shape)) * 4
    if path.stat().st_size != expected:
        raise ValueError(f"{path.name} size does not match manifest")
    return np.memmap(path, dtype="<f4", mode="r", shape=shape)


def schema_columns(manifest: dict, target_layer: int, schemas: list[int]) -> dict:
    grouped = {schema: [] for schema in schemas}
    for index, obj in enumerate(manifest["objects"]):
        if obj["layer"] != target_layer:
            continue
        schema = obj.get("block_channels")
        if schema in grouped:
            grouped[schema].append(index)
    for schema, columns in grouped.items():
        if not columns:
            raise ValueError(f"page capture has no layer-{target_layer} schema {schema}")
        columns.sort(key=lambda column: manifest["objects"][column]["channel_start"])
        expected_start = 0
        for column in columns:
            obj = manifest["objects"][column]
            if obj["channel_start"] != expected_start:
                raise ValueError(f"schema {schema} has a gap or overlap at {expected_start}")
            expected_start = obj["channel_end"]
        if expected_start <= 0:
            raise ValueError(f"schema {schema} has an empty partition")
    ends = {
        manifest["objects"][columns[-1]]["channel_end"] for columns in grouped.values()
    }
    if len(ends) != 1:
        raise ValueError(f"page schemas terminate at different channels: {sorted(ends)}")
    return grouped


def entropy_report(contributions: np.ndarray) -> dict:
    totals = contributions.sum(axis=1, keepdims=True)
    valid = totals[:, 0] > 0
    probabilities = np.zeros_like(contributions, dtype=np.float64)
    probabilities[valid] = contributions[valid] / totals[valid]
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probabilities > 0, probabilities * np.log(probabilities), 0.0)
    effective = np.exp(-terms.sum(axis=1))
    return {
        "mean_effective_objects": float(np.mean(effective)),
        "median_effective_objects": float(np.median(effective)),
        "mean_effective_fraction": float(np.mean(effective) / contributions.shape[1]),
    }


def distribution_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.1)),
        "p90": float(np.quantile(values, 0.9)),
    }


def score_predictions(predictions, truths, byte_counts, budgets):
    coverage_rows = {str(budget): [] for budget in budgets}
    vector_rows = []
    for prediction, truth in zip(predictions, truths):
        vector_rows.append(field.vector_metrics(truth, prediction))
        ranking = field.rank_density(prediction, byte_counts)
        for budget in budgets:
            coverage_rows[str(budget)].append(
                field.coverage(truth, ranking, byte_counts, budget)
            )
    return {
        "vector": field.mean_rows(vector_rows),
        "budgets": {
            key: field.mean_rows(rows) for key, rows in coverage_rows.items()
        },
    }, {
        key: np.asarray([row["contribution_coverage"] for row in rows])
        for key, rows in coverage_rows.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index_capture", type=Path)
    parser.add_argument("page_capture", type=Path)
    parser.add_argument("--frontier-layers", default="36-45")
    parser.add_argument("--primary-layer", type=int, default=38)
    parser.add_argument("--target-layer", type=int, default=49)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--schemas", default="16,32,64,128,256")
    parser.add_argument("--authority-schema", type=int, default=19968)
    parser.add_argument("--control-schema", type=int, default=128)
    parser.add_argument("--byte-budgets", default="0.05,0.2")
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0x5041474531)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    schemas = [int(value) for value in args.schemas.split(",")]
    if len(schemas) != len(set(schemas)) or min(schemas) <= 0:
        raise ValueError("schemas must be unique positive integers")
    if args.control_schema not in schemas:
        raise ValueError("control schema must be one of the candidate schemas")
    if min(args.neighbors, args.permutations) <= 0:
        raise ValueError("count arguments must be positive")
    budgets = [float(value) for value in args.byte_budgets.split(",")]

    index_manifest = json.loads((args.index_capture / "manifest.json").read_text())
    page_manifest = json.loads((args.page_capture / "manifest.json").read_text())
    index_samples = index_manifest["samples"]
    page_samples = page_manifest["samples"]
    if len(index_samples) != len(page_samples):
        raise ValueError("index/page captures have different sample counts")
    compared_fields = (
        "id",
        "split",
        "group_id",
        "operation_id",
        "wording_id",
        "alias_id",
        "answer_exact",
        "expected_token_ids",
        "generated_token_ids",
    )
    for row, (index_sample, page_sample) in enumerate(zip(index_samples, page_samples)):
        for key in compared_fields:
            if index_sample.get(key) != page_sample.get(key):
                raise ValueError(f"sample {row} differs between captures at field {key}")

    available_layers = index_manifest["residual_layers"]
    frontier = field.parse_layers(args.frontier_layers, available_layers)
    if args.primary_layer not in frontier:
        raise ValueError("primary layer must belong to the frontier")
    history = [
        index
        for index, sample in enumerate(index_samples)
        if sample["split"] == "history" and sample.get("answer_exact") is True
    ]
    queries = [
        index
        for index, sample in enumerate(index_samples)
        if sample["split"] == "query" and sample.get("answer_exact") is True
    ]
    if len(history) < args.neighbors or not queries:
        raise ValueError("insufficient exact-success rows")

    residual_shape = (
        len(index_samples),
        len(available_layers),
        index_manifest["hidden_size"],
    )
    residuals = open_plane(
        args.index_capture, index_manifest, "residuals_file", residual_shape
    )
    contribution_shape = (len(page_samples), len(page_manifest["objects"]))
    contributions = open_plane(
        args.page_capture,
        page_manifest,
        "contributions_file",
        contribution_shape,
    )
    all_schemas = schemas + [args.authority_schema]
    columns = schema_columns(page_manifest, args.target_layer, all_schemas)
    authority_columns = columns[args.authority_schema]
    if len(authority_columns) != 1:
        raise ValueError("authority schema must contain exactly one full-output page")
    full_energy = np.asarray(contributions[:, authority_columns[0]], dtype=np.float64)
    if (full_energy <= 0).any() or not np.isfinite(full_energy).all():
        raise ValueError("authority schema contains invalid full-output energy")

    byte_totals = {
        schema: sum(page_manifest["objects"][column]["byte_count"] for column in group)
        for schema, group in columns.items()
    }
    candidate_byte_totals = {byte_totals[schema] for schema in schemas}
    if len(candidate_byte_totals) != 1:
        raise ValueError(f"candidate schemas represent different byte totals: {byte_totals}")

    rankings = {}
    history_groups = np.array([index_samples[index].get("group_id") for index in history])
    for layer in frontier:
        slot = available_layers.index(layer)
        history_matrix = field.normalized(np.asarray(residuals[history, slot, :]))
        query_matrix = field.normalized(np.asarray(residuals[queries, slot, :]))
        scores = query_matrix @ history_matrix.T
        for query_row, query in enumerate(queries):
            scores[
                query_row,
                history_groups == index_samples[query].get("group_id"),
            ] = -np.inf
        rankings[layer] = np.argsort(-scores, axis=1, kind="stable")[:, : args.neighbors]

    schema_reports = {}
    per_query_primary = {}
    for schema in schemas:
        schema_contributions = np.asarray(
            contributions[:, columns[schema]], dtype=np.float64
        )
        history_targets = schema_contributions[history]
        truths = schema_contributions[queries]
        popularity = np.repeat(
            history_targets.mean(axis=0, keepdims=True), len(queries), axis=0
        )
        popularity_report, popularity_rows = score_predictions(
            popularity, truths, np.array(
                [page_manifest["objects"][column]["byte_count"] for column in columns[schema]],
                dtype=np.float64,
            ), budgets
        )
        oracle_report, oracle_rows = score_predictions(
            truths,
            truths,
            np.array(
                [page_manifest["objects"][column]["byte_count"] for column in columns[schema]],
                dtype=np.float64,
            ),
            budgets,
        )
        layer_reports = {}
        for layer in frontier:
            predictions = history_targets[rankings[layer]].mean(axis=1)
            knn_report, knn_rows = score_predictions(
                predictions,
                truths,
                np.array(
                    [
                        page_manifest["objects"][column]["byte_count"]
                        for column in columns[schema]
                    ],
                    dtype=np.float64,
                ),
                budgets,
            )
            derived = {}
            for budget in budgets:
                key = str(budget)
                pop = popularity_report["budgets"][key]["contribution_coverage"]
                oracle = oracle_report["budgets"][key]["contribution_coverage"]
                knn = knn_report["budgets"][key]["contribution_coverage"]
                headroom = oracle - pop
                derived[key] = {
                    "oracle_headroom": headroom,
                    "knn_gain": knn - pop,
                    "fraction_headroom_recovered": (
                        (knn - pop) / headroom if headroom > 0 else None
                    ),
                }
            layer_reports[str(layer)] = {
                "knn": knn_report,
                "derived": derived,
            }
            if layer == args.primary_layer:
                per_query_primary[schema] = {
                    "popularity": popularity_rows,
                    "oracle": oracle_rows,
                    "knn": knn_rows,
                }
        interference = schema_contributions.sum(axis=1) / full_energy
        schema_reports[str(schema)] = {
            "objects": len(columns[schema]),
            "total_bytes": byte_totals[schema],
            "contribution_entropy": entropy_report(schema_contributions),
            "interference_factor": distribution_summary(interference),
            "popularity": popularity_report,
            "oracle": oracle_report,
            "layers": layer_reports,
        }

    comparison_count = (len(schemas) - 1) * 2 * len(budgets)
    comparisons = {}
    control = per_query_primary[args.control_schema]
    for schema_index, schema in enumerate(schemas):
        if schema == args.control_schema:
            continue
        comparisons[str(schema)] = {}
        candidate = per_query_primary[schema]
        for budget_index, budget in enumerate(budgets):
            key = str(budget)
            candidate_headroom = candidate["oracle"][key] - candidate["popularity"][key]
            control_headroom = control["oracle"][key] - control["popularity"][key]
            candidate_gain = candidate["knn"][key] - candidate["popularity"][key]
            control_gain = control["knn"][key] - control["popularity"][key]
            headroom_test = field.paired_sign_flip(
                candidate_headroom - control_headroom,
                args.permutations,
                args.seed + schema_index * 100 + budget_index * 10,
            )
            gain_test = field.paired_sign_flip(
                candidate_gain - control_gain,
                args.permutations,
                args.seed + schema_index * 100 + budget_index * 10 + 1,
            )
            for test in (headroom_test, gain_test):
                test["bonferroni_p_16"] = min(
                    1.0, test["p_value_plus_one"] * comparison_count
                )
            comparisons[str(schema)][key] = {
                "oracle_headroom_change_vs_control": headroom_test,
                "knn_gain_change_vs_control": gain_test,
            }

    result = {
        "schema": SCHEMA,
        "index_capture": str(args.index_capture),
        "page_capture": str(args.page_capture),
        "frontier_layers": frontier,
        "primary_layer": args.primary_layer,
        "target_layer": args.target_layer,
        "neighbors": args.neighbors,
        "candidate_schemas": schemas,
        "authority_schema": args.authority_schema,
        "control_schema": args.control_schema,
        "byte_budgets": budgets,
        "history_successes": len(history),
        "query_successes": len(queries),
        "candidate_total_bytes": candidate_byte_totals.pop(),
        "schema_reports": schema_reports,
        "primary_comparisons": comparisons,
        "interpretation_guard": (
            "Coverage is schema-relative squared page-output energy. Cross-schema "
            "interpretation must include the full-output interference factor."
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
