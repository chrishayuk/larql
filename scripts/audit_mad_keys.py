#!/usr/bin/env python3
"""MAD-V3-KEY-1 residual-transition address-refinement audit."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

import audit_mad_field as field
import audit_mad_pages as pages
import audit_mad_state as state


SCHEMA = "larql.mad-v3-key-1.audit.v1"
KEY_SPECS = {
    "h38": (("state", 38),),
    "h37_h38": (("state", 37), ("state", 38)),
    "delta38": (("delta", 37, 38),),
    "h38_delta38": (("state", 38), ("delta", 37, 38)),
    "h38_delta37_delta38": (
        ("state", 38),
        ("delta", 36, 37),
        ("delta", 37, 38),
    ),
}
CONTROL_KEY = "h38"


def component_values(residuals, rows, layer_slots, component):
    kind = component[0]
    if kind == "state":
        return np.asarray(residuals[rows, layer_slots[component[1]], :])
    if kind == "delta":
        start = np.asarray(residuals[rows, layer_slots[component[1]], :])
        end = np.asarray(residuals[rows, layer_slots[component[2]], :])
        return end - start
    raise ValueError(f"unknown key component {kind}")


def compound_scores(residuals, history, queries, layer_slots, spec):
    scores = np.zeros((len(queries), len(history)), dtype=np.float64)
    for component in spec:
        history_component = field.normalized(
            component_values(residuals, history, layer_slots, component)
        )
        query_component = field.normalized(
            component_values(residuals, queries, layer_slots, component)
        )
        scores += query_component @ history_component.T
    return scores / len(spec)


def neighbor_selections(
    samples, residuals, history, queries, layer_slots, spec, neighbors
):
    scores = compound_scores(residuals, history, queries, layer_slots, spec)
    history_groups = np.array([samples[index].get("group_id") for index in history])
    for query_row, query in enumerate(queries):
        scores[query_row, history_groups == samples[query].get("group_id")] = -np.inf
    if np.any(np.count_nonzero(np.isfinite(scores), axis=1) < neighbors):
        raise ValueError("same-group exclusion leaves too few neighbours")
    return np.argsort(-scores, axis=1, kind="stable")[:, :neighbors]


def selected_page_set(profile, byte_counts, budget):
    limit = budget * float(byte_counts.sum())
    used = 0
    selected = set()
    for column in field.rank_density(profile, byte_counts):
        cost = int(byte_counts[column])
        if used + cost > limit:
            continue
        used += cost
        selected.add(int(column))
    return selected


def cohort_disagreement(history_targets, selections, byte_counts, budgets):
    centroid_cosines = []
    centroid_js = []
    jaccards = {str(budget): [] for budget in budgets}
    for selected in selections:
        profiles = np.asarray(history_targets[selected], dtype=np.float64)
        totals = profiles.sum(axis=1, keepdims=True)
        if (totals <= 0).any() or not np.isfinite(totals).all():
            raise ValueError("cohort contains an invalid contribution profile")
        profiles /= totals
        centroid = profiles.mean(axis=0)
        for profile in profiles:
            metrics = field.vector_metrics(profile, centroid)
            centroid_cosines.append(metrics["cosine"])
            centroid_js.append(metrics["js_divergence"])
        for budget in budgets:
            sets = [selected_page_set(profile, byte_counts, budget) for profile in profiles]
            for left, right in combinations(sets, 2):
                union = left | right
                jaccards[str(budget)].append(len(left & right) / len(union) if union else 1.0)
    return {
        "mean_neighbor_centroid_cosine": float(np.mean(centroid_cosines)),
        "mean_neighbor_centroid_js_divergence": float(np.mean(centroid_js)),
        "mean_pairwise_top_page_jaccard": {
            key: float(np.mean(values)) for key, values in jaccards.items()
        },
    }


def per_query_vector_metrics(predictions, truths):
    cosine = []
    js = []
    for prediction, truth in zip(predictions, truths):
        metrics = field.vector_metrics(truth, prediction)
        cosine.append(metrics["cosine"])
        js.append(metrics["js_divergence"])
    return {
        "cosine": np.asarray(cosine, dtype=np.float64),
        "js_divergence": np.asarray(js, dtype=np.float64),
    }


def verify_samples(index_samples, page_samples):
    if len(index_samples) != len(page_samples):
        raise ValueError("index/page captures have different sample counts")
    fields = (
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
        for key in fields:
            if index_sample.get(key) != page_sample.get(key):
                raise ValueError(f"sample {row} differs between captures at field {key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index_capture", type=Path)
    parser.add_argument("page_capture", type=Path)
    parser.add_argument("--page-audit", type=Path)
    parser.add_argument("--target-layer", type=int, default=49)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--schemas", default="16,32")
    parser.add_argument("--disagreement-schemas", default="16,32,64,128")
    parser.add_argument("--byte-budgets", default="0.05,0.2")
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0x4B455931)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.neighbors, args.permutations) <= 0:
        raise ValueError("count arguments must be positive")
    schemas = [int(value) for value in args.schemas.split(",")]
    disagreement_schemas = [
        int(value) for value in args.disagreement_schemas.split(",")
    ]
    all_schemas = list(dict.fromkeys(schemas + disagreement_schemas))
    if len(schemas) != len(set(schemas)) or min(all_schemas) <= 0:
        raise ValueError("schemas must be unique positive integers")
    budgets = [float(value) for value in args.byte_budgets.split(",")]

    index_manifest = json.loads((args.index_capture / "manifest.json").read_text())
    page_manifest = json.loads((args.page_capture / "manifest.json").read_text())
    samples = index_manifest["samples"]
    verify_samples(samples, page_manifest["samples"])
    available_layers = index_manifest["residual_layers"]
    required_layers = sorted(
        {
            layer
            for spec in KEY_SPECS.values()
            for component in spec
            for layer in component[1:]
        }
    )
    missing = sorted(set(required_layers) - set(available_layers))
    if missing:
        raise ValueError(f"index capture lacks key layers {missing}")
    layer_slots = {layer: available_layers.index(layer) for layer in required_layers}

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
    if len(history) < args.neighbors or not queries:
        raise ValueError("insufficient exact-success rows")

    residual_shape = (
        len(samples),
        len(available_layers),
        index_manifest["hidden_size"],
    )
    residuals = pages.open_plane(
        args.index_capture, index_manifest, "residuals_file", residual_shape
    )
    contribution_shape = (
        len(page_manifest["samples"]),
        len(page_manifest["objects"]),
    )
    contributions = pages.open_plane(
        args.page_capture,
        page_manifest,
        "contributions_file",
        contribution_shape,
    )
    columns = pages.schema_columns(page_manifest, args.target_layer, all_schemas)

    selections = {
        name: neighbor_selections(
            samples,
            residuals,
            history,
            queries,
            layer_slots,
            spec,
            args.neighbors,
        )
        for name, spec in KEY_SPECS.items()
    }
    compositions = {
        name: state.cohort_composition(samples, history, queries, selected)
        for name, selected in selections.items()
    }

    disagreement = {}
    for schema in disagreement_schemas:
        byte_counts = np.asarray(
            [page_manifest["objects"][column]["byte_count"] for column in columns[schema]],
            dtype=np.float64,
        )
        history_targets = np.asarray(
            contributions[np.ix_(history, columns[schema])], dtype=np.float64
        )
        disagreement[str(schema)] = cohort_disagreement(
            history_targets, selections[CONTROL_KEY], byte_counts, budgets
        )

    schema_reports = {}
    per_query = {}
    reproduction_errors = []
    page_audit = json.loads(args.page_audit.read_text()) if args.page_audit else None
    for schema in schemas:
        byte_counts = np.asarray(
            [page_manifest["objects"][column]["byte_count"] for column in columns[schema]],
            dtype=np.float64,
        )
        history_targets = np.asarray(
            contributions[np.ix_(history, columns[schema])], dtype=np.float64
        )
        truths = np.asarray(
            contributions[np.ix_(queries, columns[schema])], dtype=np.float64
        )
        popularity = np.repeat(
            history_targets.mean(axis=0, keepdims=True), len(queries), axis=0
        )
        popularity_report, _ = pages.score_predictions(
            popularity, truths, byte_counts, budgets
        )
        oracle_report, _ = pages.score_predictions(truths, truths, byte_counts, budgets)
        key_reports = {}
        per_query[schema] = {}
        for name, selected in selections.items():
            predictions = history_targets[selected].mean(axis=1)
            report, coverage_rows = pages.score_predictions(
                predictions, truths, byte_counts, budgets
            )
            headroom = {}
            for budget in budgets:
                key = str(budget)
                base = popularity_report["budgets"][key]["contribution_coverage"]
                ceiling = oracle_report["budgets"][key]["contribution_coverage"]
                value = report["budgets"][key]["contribution_coverage"]
                headroom[key] = {
                    "oracle_headroom": ceiling - base,
                    "knn_gain": value - base,
                    "fraction_headroom_recovered": (
                        (value - base) / (ceiling - base) if ceiling > base else None
                    ),
                }
            report["derived"] = headroom
            report["composition"] = compositions[name]
            key_reports[name] = report
            per_query[schema][name] = {
                "coverage": coverage_rows,
                "vector": per_query_vector_metrics(predictions, truths),
            }
        schema_reports[str(schema)] = {
            "objects": len(columns[schema]),
            "popularity": popularity_report,
            "oracle": oracle_report,
            "keys": key_reports,
        }
        if page_audit:
            expected = page_audit["schema_reports"][str(schema)]["layers"]["38"]
            for budget in budgets:
                key = str(budget)
                actual = key_reports[CONTROL_KEY]["budgets"][key][
                    "contribution_coverage"
                ]
                reference = expected["knn"]["budgets"][key]["contribution_coverage"]
                reproduction_errors.append(abs(actual - reference))

    alternatives = [name for name in KEY_SPECS if name != CONTROL_KEY]
    comparison_count = len(alternatives) * len(schemas) * len(budgets)
    comparisons = {}
    for schema_index, schema in enumerate(schemas):
        comparisons[str(schema)] = {}
        baseline = per_query[schema][CONTROL_KEY]
        for key_index, name in enumerate(alternatives):
            comparisons[str(schema)][name] = {}
            candidate = per_query[schema][name]
            vector_delta = {
                "cosine": float(
                    np.mean(candidate["vector"]["cosine"] - baseline["vector"]["cosine"])
                ),
                "js_divergence_improvement": float(
                    np.mean(
                        baseline["vector"]["js_divergence"]
                        - candidate["vector"]["js_divergence"]
                    )
                ),
            }
            for budget_index, budget in enumerate(budgets):
                key = str(budget)
                test = field.paired_sign_flip(
                    candidate["coverage"][key] - baseline["coverage"][key],
                    args.permutations,
                    args.seed
                    + schema_index * 1000
                    + key_index * 100
                    + budget_index * 10,
                )
                test[f"bonferroni_p_{comparison_count}"] = min(
                    1.0, test["p_value_plus_one"] * comparison_count
                )
                comparisons[str(schema)][name][key] = {
                    "coverage_change_vs_h38": test,
                    "vector_change_vs_h38": vector_delta,
                }

    max_reproduction_error = max(reproduction_errors, default=None)
    if max_reproduction_error is not None and max_reproduction_error > 1e-12:
        raise ValueError(
            f"h38 baseline does not reproduce PAGE-1: {max_reproduction_error}"
        )
    result = {
        "schema": SCHEMA,
        "index_capture": str(args.index_capture),
        "page_capture": str(args.page_capture),
        "page_audit": str(args.page_audit) if args.page_audit else None,
        "target_layer": args.target_layer,
        "neighbors": args.neighbors,
        "schemas": schemas,
        "disagreement_schemas": disagreement_schemas,
        "byte_budgets": budgets,
        "history_successes": len(history),
        "query_successes": len(queries),
        "key_specs": {name: [list(component) for component in spec] for name, spec in KEY_SPECS.items()},
        "normalization": "independent L2 normalization per component; mean component cosine",
        "baseline_reproduction_max_abs_error": max_reproduction_error,
        "cohort_disagreement": disagreement,
        "schema_reports": schema_reports,
        "comparisons": comparisons,
        "comparison_count": comparison_count,
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
