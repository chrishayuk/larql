#!/usr/bin/env python3
"""MAD-V3-HORIZON-1 progressive physical-resolution interaction audit."""

import argparse
import json
from pathlib import Path

import numpy as np

import audit_mad_addresses as addresses
import audit_mad_field as field
import audit_mad_pages as pages


SCHEMA = "larql.mad-v3-horizon-1.audit.v1"
STAGES = (
    (38, "layer_input", 0.0),
    (44, "layer_input", 6.0),
    (47, "ffn_output", 9.0),
    (49, "ffn_input", 11.0),
)


def slope_rows(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Return one OLS slope for every row in values."""
    values = np.asarray(values, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(coordinates):
        raise ValueError("values must have one column per proximity coordinate")
    centered = coordinates - coordinates.mean()
    denominator = float(centered @ centered)
    if denominator <= 0:
        raise ValueError("proximity coordinates have no variation")
    return (values @ centered) / denominator


def recovered_rows(knn, popularity, oracle) -> tuple[np.ndarray, float]:
    """Normalize query KNN gains by the schema's aggregate oracle headroom."""
    knn = np.asarray(knn, dtype=np.float64)
    popularity = np.asarray(popularity, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    headroom = float(np.mean(oracle) - np.mean(popularity))
    if headroom <= 0:
        raise ValueError("schema has no positive aggregate oracle headroom")
    return (knn - popularity) / headroom, headroom


def sample_alignment(addr_samples, page_samples):
    page_by_id = {sample["id"]: index for index, sample in enumerate(page_samples)}
    if len(page_by_id) != len(page_samples):
        raise ValueError("PAGE capture has duplicate sample IDs")
    fields = (
        "split",
        "group_id",
        "operation_id",
        "wording_id",
        "alias_id",
        "answer_exact",
        "expected_token_ids",
        "generated_token_ids",
    )
    aligned = []
    for sample in addr_samples:
        page_index = page_by_id.get(sample["id"])
        if page_index is None:
            raise ValueError(f"ADDR sample {sample['id']} is absent from PAGE capture")
        page_sample = page_samples[page_index]
        for name in fields:
            if sample.get(name) != page_sample.get(name):
                raise ValueError(f"sample {sample['id']} differs at field {name}")
        aligned.append(page_index)
    return np.asarray(aligned, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("addr_capture", type=Path)
    parser.add_argument("page_capture", type=Path)
    parser.add_argument("--page-audit", type=Path)
    parser.add_argument("--target-layer", type=int, default=49)
    parser.add_argument("--schemas", default="16,32,128")
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--byte-budgets", default="0.05,0.2")
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0x484F52495A4F4E31)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    schemas = [int(value) for value in args.schemas.split(",")]
    budgets = [float(value) for value in args.byte_budgets.split(",")]
    if schemas != [16, 32, 128]:
        raise ValueError("HORIZON-1 freezes schemas to 16,32,128")
    if budgets != [0.05, 0.2]:
        raise ValueError("HORIZON-1 freezes byte budgets to 0.05,0.2")
    if args.neighbors != 16 or min(args.permutations, args.neighbors) <= 0:
        raise ValueError("HORIZON-1 freezes K=16 and requires positive permutations")

    addr_manifest = json.loads((args.addr_capture / "manifest.json").read_text())
    page_manifest = json.loads((args.page_capture / "manifest.json").read_text())
    addr_samples = addr_manifest["samples"]
    page_samples = page_manifest["samples"]
    if not addr_samples or any(sample.get("answer_exact") is not True for sample in addr_samples):
        raise ValueError("HORIZON-1 requires the frozen exact-success ADDR capture")
    aligned_page_rows = sample_alignment(addr_samples, page_samples)

    history = [index for index, sample in enumerate(addr_samples) if sample["split"] == "history"]
    queries = [index for index, sample in enumerate(addr_samples) if sample["split"] == "query"]
    if len(history) != 436 or len(queries) != 134:
        raise ValueError("HORIZON-1 freezes the 436/134 PAGE-1 split")

    residual_layers = addr_manifest["residual_layers"]
    seam_layers = addr_manifest.get("seam_layers", [])
    seam_kinds = addr_manifest.get("seam_kinds", [])
    for layer, kind, _proximity in STAGES:
        if kind == "layer_input" and layer not in residual_layers:
            raise ValueError(f"ADDR capture lacks residual layer {layer}")
        if kind != "layer_input" and (layer not in seam_layers or kind not in seam_kinds):
            raise ValueError(f"ADDR capture lacks L{layer} {kind}")

    residuals = pages.open_plane(
        args.addr_capture,
        addr_manifest,
        "residuals_file",
        (len(addr_samples), len(residual_layers), addr_manifest["hidden_size"]),
    )
    seams = addresses.open_seams(args.addr_capture, addr_manifest)
    page_contributions = pages.open_plane(
        args.page_capture,
        page_manifest,
        "contributions_file",
        (len(page_samples), len(page_manifest["objects"])),
    )
    columns = pages.schema_columns(page_manifest, args.target_layer, schemas)
    residual_slots = {layer: residual_layers.index(layer) for layer in residual_layers}
    seam_slots = {layer: seam_layers.index(layer) for layer in seam_layers}
    kind_slots = {kind: seam_kinds.index(kind) for kind in seam_kinds}

    selections = []
    for layer, kind, _proximity in STAGES:
        matrix = addresses.key_matrix(
            kind,
            layer,
            residuals,
            seams,
            residual_slots,
            seam_slots,
            kind_slots,
        )
        selections.append(
            addresses.select_neighbors(
                addr_samples,
                history,
                queries,
                matrix[history],
                matrix[queries],
                args.neighbors,
            )
        )

    stage_reports = {str(schema): [] for schema in schemas}
    recovered = {schema: {str(budget): [] for budget in budgets} for schema in schemas}
    reproduction_errors = []
    page_audit = json.loads(args.page_audit.read_text()) if args.page_audit else None
    for schema in schemas:
        byte_counts = np.asarray(
            [page_manifest["objects"][column]["byte_count"] for column in columns[schema]],
            dtype=np.float64,
        )
        aligned = np.asarray(
            page_contributions[np.ix_(aligned_page_rows, columns[schema])],
            dtype=np.float64,
        )
        history_targets = aligned[history]
        truths = aligned[queries]
        popularity = np.repeat(history_targets.mean(axis=0, keepdims=True), len(queries), axis=0)
        popularity_report, popularity_rows = pages.score_predictions(
            popularity, truths, byte_counts, budgets
        )
        oracle_report, oracle_rows = pages.score_predictions(truths, truths, byte_counts, budgets)

        for stage_index, ((layer, kind, proximity), selected) in enumerate(zip(STAGES, selections)):
            predictions = history_targets[selected].mean(axis=1)
            knn_report, knn_rows = pages.score_predictions(predictions, truths, byte_counts, budgets)
            derived = {}
            for budget in budgets:
                key = str(budget)
                rows, headroom = recovered_rows(
                    knn_rows[key], popularity_rows[key], oracle_rows[key]
                )
                recovered[schema][key].append(rows)
                derived[key] = {
                    "oracle_headroom": headroom,
                    "knn_gain": float(np.mean(knn_rows[key]) - np.mean(popularity_rows[key])),
                    "fraction_headroom_recovered": float(np.mean(rows)),
                }
                if page_audit and stage_index == 0:
                    expected = page_audit["schema_reports"][str(schema)]["layers"][str(layer)]["knn"]
                    reproduction_errors.append(
                        abs(
                            knn_report["budgets"][key]["contribution_coverage"]
                            - expected["budgets"][key]["contribution_coverage"]
                        )
                    )
            stage_reports[str(schema)].append(
                {
                    "layer": layer,
                    "seam": kind,
                    "proximity": proximity,
                    "availability": addresses.availability(layer, kind, args.target_layer),
                    "knn": knn_report,
                    "derived": derived,
                }
            )

    max_reproduction_error = max(reproduction_errors, default=None)
    if max_reproduction_error is not None and max_reproduction_error > 1e-12:
        raise ValueError(f"L38 baseline does not reproduce PAGE-1: {max_reproduction_error}")

    coordinates = np.asarray([stage[2] for stage in STAGES], dtype=np.float64)
    interactions = {}
    for fine_schema in (16, 32):
        budget_slopes = {}
        query_slopes = []
        penalties = {}
        for budget in budgets:
            key = str(budget)
            fine = np.stack(recovered[fine_schema][key], axis=1)
            coarse = np.stack(recovered[128][key], axis=1)
            penalty = fine - coarse
            slopes = slope_rows(penalty, coordinates)
            penalties[key] = [float(value) for value in penalty.mean(axis=0)]
            budget_slopes[key] = {
                "mean_slope": float(np.mean(slopes)),
                "query_win_fraction": float(np.mean(slopes > 0)),
            }
            query_slopes.append(slopes)
        omnibus_rows = np.mean(np.stack(query_slopes, axis=1), axis=1)
        omnibus = field.paired_sign_flip(
            omnibus_rows,
            args.permutations,
            args.seed + fine_schema,
        )
        interactions[f"{fine_schema}_vs_128"] = {
            "fine_penalty_by_stage": penalties,
            "budget_slopes": budget_slopes,
            "omnibus_mean_slope_across_budgets": omnibus,
            "passes_direction_at_both_budgets": all(
                report["mean_slope"] > 0 for report in budget_slopes.values()
            ),
        }

    primary = interactions["16_vs_128"]
    passed = (
        primary["omnibus_mean_slope_across_budgets"]["mean_delta"] > 0
        and primary["omnibus_mean_slope_across_budgets"]["p_value_plus_one"] < 0.05
        and primary["passes_direction_at_both_budgets"]
    )
    result = {
        "schema": SCHEMA,
        "addr_capture": str(args.addr_capture),
        "page_capture": str(args.page_capture),
        "page_audit": str(args.page_audit) if args.page_audit else None,
        "target_layer": args.target_layer,
        "stages": [
            {"layer": layer, "seam": kind, "proximity": proximity}
            for layer, kind, proximity in STAGES
        ],
        "schemas": schemas,
        "neighbors": args.neighbors,
        "byte_budgets": budgets,
        "history_successes": len(history),
        "query_successes": len(queries),
        "baseline_reproduction_max_abs_error": max_reproduction_error,
        "schema_reports": stage_reports,
        "interactions": interactions,
        "primary_gate": {
            "contrast": "16_vs_128",
            "one_sided_alpha": 0.05,
            "requires_positive_slope_at_both_budgets": True,
            "passed": passed,
        },
        "interpretation_guard": (
            "Every page schema uses its independently captured squared-output energy; "
            "coarse pages are never synthesized by summing fine-page energies."
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
