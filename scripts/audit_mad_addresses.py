#!/usr/bin/env python3
"""MAD-V3-ADDR-1 address-formation and lead-time census."""

import argparse
import json
from pathlib import Path

import numpy as np

import audit_mad_field as field
import audit_mad_pages as pages
import audit_mad_state as state


SCHEMA = "larql.mad-v3-addr-1.audit.v2"
SEAM_ORDER = (
    "layer_input",
    "attention_input",
    "attention_output",
    "post_attention",
    "ffn_input",
    "ffn_output",
)


def open_seams(capture: Path, manifest: dict):
    member = manifest.get("seams_file")
    layers = manifest.get("seam_layers", [])
    kinds = manifest.get("seam_kinds", [])
    if not member or not layers or not kinds:
        raise ValueError("capture has no complete seam plane declaration")
    if len(layers) != len(set(layers)) or len(kinds) != len(set(kinds)):
        raise ValueError("capture has duplicate seam layers or kinds")
    shape = (len(manifest["samples"]), len(layers), len(kinds), manifest["hidden_size"])
    path = capture / member
    expected = int(np.prod(shape)) * 4
    if path.stat().st_size != expected:
        raise ValueError("seam plane size does not match manifest")
    return np.memmap(path, dtype="<f4", mode="r", shape=shape)


def comparison_edges(layers):
    edges = []
    for layer in layers:
        edges.extend(
            [
                (
                    (layer, "layer_input"),
                    (layer, "attention_input"),
                    "attention_normalization",
                ),
                (
                    (layer, "layer_input"),
                    (layer, "post_attention"),
                    "attention_residual_join",
                ),
                (
                    (layer, "post_attention"),
                    (layer, "ffn_input"),
                    "ffn_normalization",
                ),
            ]
        )
    for before, after in zip(layers, layers[1:]):
        edges.append(
            (
                (before, "post_attention"),
                (after, "layer_input"),
                "ffn_residual_join",
            )
        )
    return edges


def vector_rows(predictions, truths):
    cosine = []
    js = []
    for prediction, truth in zip(predictions, truths):
        row = field.vector_metrics(truth, prediction)
        cosine.append(row["cosine"])
        js.append(row["js_divergence"])
    return {
        "cosine": np.asarray(cosine, dtype=np.float64),
        "js_divergence": np.asarray(js, dtype=np.float64),
    }


def select_neighbors(samples, history, queries, history_matrix, query_matrix, k):
    scores = field.normalized(query_matrix) @ field.normalized(history_matrix).T
    history_groups = np.array([samples[index].get("group_id") for index in history])
    for row, query in enumerate(queries):
        scores[row, history_groups == samples[query].get("group_id")] = -np.inf
    if np.any(np.count_nonzero(np.isfinite(scores), axis=1) < k):
        raise ValueError("same-group exclusion leaves too few neighbours")
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def key_matrix(kind, layer, residuals, seams, residual_slots, seam_slots, kind_slots):
    if kind == "layer_input":
        return np.asarray(residuals[:, residual_slots[layer], :])
    return np.asarray(seams[:, seam_slots[layer], kind_slots[kind], :])


def availability(layer, kind, target_layer):
    if layer < target_layer:
        return "advance"
    if layer > target_layer:
        return "post_target"
    return "post_consumption_control" if kind == "ffn_output" else "pre_down_same_layer"


def formation_passes(comparisons, schemas, budgets, corrected_key):
    passes = []
    transition_ids = set.intersection(
        *(set(comparisons[str(schema)]) for schema in schemas)
    )
    for transition_id in sorted(transition_ids):
        if not comparisons[str(schemas[0])][transition_id]["comparison_kind"].endswith(
            "residual_join"
        ):
            continue
        passed = True
        for schema in schemas:
            report = comparisons[str(schema)][transition_id]
            vector = report["vector_change"]
            if vector["cosine"] <= 0 or vector["js_divergence_improvement"] <= 0:
                passed = False
            for budget in budgets:
                test = report["budgets"][str(budget)]
                if test["mean_delta"] <= 0 or test[corrected_key] > 0.05:
                    passed = False
        if passed:
            passes.append(transition_id)
    return passes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--page-audit", type=Path)
    parser.add_argument("--layers", default="38-49")
    parser.add_argument("--target-layer", type=int, default=49)
    parser.add_argument("--schemas", default="16,32")
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--byte-budgets", default="0.05,0.2")
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0x4144445231)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.neighbors, args.permutations) <= 0:
        raise ValueError("count arguments must be positive")
    schemas = [int(value) for value in args.schemas.split(",")]
    if len(schemas) != len(set(schemas)) or min(schemas) <= 0:
        raise ValueError("schemas must be unique positive integers")
    budgets = [float(value) for value in args.byte_budgets.split(",")]

    manifest = json.loads((args.capture / "manifest.json").read_text())
    samples = manifest["samples"]
    if not samples or any(sample.get("answer_exact") is not True for sample in samples):
        raise ValueError("ADDR-1 requires the frozen exact-success fixture")
    available_residual_layers = manifest["residual_layers"]
    layers = field.parse_layers(args.layers, available_residual_layers)
    if layers != list(range(layers[0], layers[-1] + 1)):
        raise ValueError("ADDR-1 layers must be contiguous")
    if args.target_layer not in layers:
        raise ValueError("target layer must belong to the ADDR-1 census")
    seam_layers = manifest.get("seam_layers", [])
    seam_kinds = manifest.get("seam_kinds", [])
    if any(layer not in seam_layers for layer in layers):
        raise ValueError("seam capture does not cover every requested layer")
    missing_kinds = sorted(set(SEAM_ORDER[1:]) - set(seam_kinds))
    if missing_kinds:
        raise ValueError(f"seam capture lacks {missing_kinds}")

    history = [
        index for index, sample in enumerate(samples) if sample["split"] == "history"
    ]
    queries = [
        index for index, sample in enumerate(samples) if sample["split"] == "query"
    ]
    if len(history) < args.neighbors or not queries:
        raise ValueError("insufficient history/query rows")
    residual_shape = (
        len(samples),
        len(available_residual_layers),
        manifest["hidden_size"],
    )
    residuals = pages.open_plane(
        args.capture, manifest, "residuals_file", residual_shape
    )
    seams = open_seams(args.capture, manifest)
    contribution_shape = (len(samples), len(manifest["objects"]))
    contributions = pages.open_plane(
        args.capture, manifest, "contributions_file", contribution_shape
    )
    columns = pages.schema_columns(manifest, args.target_layer, schemas)
    residual_slots = {
        layer: available_residual_layers.index(layer) for layer in layers
    }
    seam_slots = {layer: seam_layers.index(layer) for layer in layers}
    kind_slots = {kind: seam_kinds.index(kind) for kind in SEAM_ORDER[1:]}

    selections = {}
    composition = {}
    for layer in layers:
        for kind in SEAM_ORDER:
            matrix = key_matrix(
                kind,
                layer,
                residuals,
                seams,
                residual_slots,
                seam_slots,
                kind_slots,
            )
            selected = select_neighbors(
                samples,
                history,
                queries,
                matrix[history],
                matrix[queries],
                args.neighbors,
            )
            selections[(layer, kind)] = selected
            composition[(layer, kind)] = state.cohort_composition(
                samples, history, queries, selected
            )

    schema_reports = {}
    per_query = {}
    reproduction_errors = []
    page_audit = json.loads(args.page_audit.read_text()) if args.page_audit else None
    for schema in schemas:
        byte_counts = np.asarray(
            [manifest["objects"][column]["byte_count"] for column in columns[schema]],
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
        for layer in layers:
            key_reports[str(layer)] = {}
            for kind in SEAM_ORDER:
                predictions = history_targets[selections[(layer, kind)]].mean(axis=1)
                report, coverage = pages.score_predictions(
                    predictions, truths, byte_counts, budgets
                )
                base = popularity_report["budgets"]
                ceiling = oracle_report["budgets"]
                report["derived"] = {}
                for budget in budgets:
                    key = str(budget)
                    headroom = (
                        ceiling[key]["contribution_coverage"]
                        - base[key]["contribution_coverage"]
                    )
                    gain = (
                        report["budgets"][key]["contribution_coverage"]
                        - base[key]["contribution_coverage"]
                    )
                    report["derived"][key] = {
                        "oracle_headroom": headroom,
                        "knn_gain": gain,
                        "fraction_headroom_recovered": gain / headroom if headroom > 0 else None,
                    }
                report["composition"] = composition[(layer, kind)]
                report["availability"] = availability(layer, kind, args.target_layer)
                report["complete_layers_before_target"] = args.target_layer - layer
                key_reports[str(layer)][kind] = report
                per_query[schema][(layer, kind)] = {
                    "coverage": coverage,
                    "vector": vector_rows(predictions, truths),
                }
                if page_audit and kind == "layer_input" and str(layer) in page_audit[
                    "schema_reports"
                ][str(schema)]["layers"]:
                    expected = page_audit["schema_reports"][str(schema)]["layers"][
                        str(layer)
                    ]["knn"]
                    for budget in budgets:
                        key = str(budget)
                        reproduction_errors.append(
                            abs(
                                report["budgets"][key]["contribution_coverage"]
                                - expected["budgets"][key]["contribution_coverage"]
                            )
                        )
        schema_reports[str(schema)] = {
            "objects": len(columns[schema]),
            "popularity": popularity_report,
            "oracle": oracle_report,
            "layers": key_reports,
        }

    edges = comparison_edges(layers)
    comparison_count = len(edges) * len(schemas) * len(budgets)
    corrected_key = f"bonferroni_p_{comparison_count}"
    comparisons = {}
    for schema_index, schema in enumerate(schemas):
        comparisons[str(schema)] = {}
        for transition_index, (before, after, comparison_kind) in enumerate(edges):
            transition_id = (
                f"L{before[0]}.{before[1]}->L{after[0]}.{after[1]}"
            )
            before_rows = per_query[schema][before]
            after_rows = per_query[schema][after]
            report = {
                "from": {"layer": before[0], "seam": before[1]},
                "to": {"layer": after[0], "seam": after[1]},
                "comparison_kind": comparison_kind,
                "vector_change": {
                    "cosine": float(
                        np.mean(
                            after_rows["vector"]["cosine"]
                            - before_rows["vector"]["cosine"]
                        )
                    ),
                    "js_divergence_improvement": float(
                        np.mean(
                            before_rows["vector"]["js_divergence"]
                            - after_rows["vector"]["js_divergence"]
                        )
                    ),
                },
                "budgets": {},
            }
            for budget_index, budget in enumerate(budgets):
                key = str(budget)
                test = field.paired_sign_flip(
                    after_rows["coverage"][key] - before_rows["coverage"][key],
                    args.permutations,
                    args.seed
                    + schema_index * 100_000
                    + transition_index * 100
                    + budget_index * 10,
                )
                test[corrected_key] = min(
                    1.0, test["p_value_plus_one"] * comparison_count
                )
                report["budgets"][key] = test
            comparisons[str(schema)][transition_id] = report

    max_reproduction_error = max(reproduction_errors, default=None)
    if max_reproduction_error is not None and max_reproduction_error > 1e-12:
        raise ValueError(
            f"layer-input baseline does not reproduce PAGE-1: {max_reproduction_error}"
        )
    result = {
        "schema": SCHEMA,
        "capture": str(args.capture),
        "page_audit": str(args.page_audit) if args.page_audit else None,
        "layers": layers,
        "target_layer": args.target_layer,
        "seam_order": list(SEAM_ORDER),
        "schemas": schemas,
        "neighbors": args.neighbors,
        "byte_budgets": budgets,
        "history_successes": len(history),
        "query_successes": len(queries),
        "comparison_count": comparison_count,
        "baseline_reproduction_max_abs_error": max_reproduction_error,
        "schema_reports": schema_reports,
        "formation_comparisons": comparisons,
        "formation_passes": formation_passes(
            comparisons, schemas, budgets, corrected_key
        ),
        "leakage_guard": (
            "L49 ffn_output is post-consumption and cannot support prefetch; "
            "all other L49 seams precede the target down projection."
        ),
        "formation_guard": (
            "Formation tests compare cumulative state across attention and FFN "
            "residual joins. Branch-only outputs remain descriptive and are never "
            "treated as the before-state of a formation edge."
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
