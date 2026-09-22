#!/usr/bin/env python3
"""Adjudicate frozen GW-SUP-1 readouts before inspecting examples."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from gwsup1_preregister import canonical_hash, sha, validate as validate_prereg

SCHEMA = "larql.gwsup1.adjudication.v1"
SEED = 1_398_100_529
RESAMPLES = 10_000
EARLY = np.array([1, 3, 5, 7])


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def entropy(probabilities: np.ndarray) -> np.ndarray:
    return -np.sum(probabilities * np.log(np.maximum(probabilities, np.finfo(float).tiny)), axis=-1)


def mean_pairwise_js(distributions: np.ndarray) -> np.ndarray:
    parts = []
    for left, right in combinations(range(distributions.shape[0]), 2):
        p = distributions[left]
        q = distributions[right]
        middle = (p + q) / 2.0
        parts.append(
            0.5 * np.sum(p * np.log(np.maximum(p / middle, np.finfo(float).tiny)), axis=-1)
            + 0.5 * np.sum(q * np.log(np.maximum(q / middle, np.finfo(float).tiny)), axis=-1)
        )
    return np.mean(parts, axis=0)


def site_index(landmark: dict[str, Any]) -> int:
    return 2 * int(landmark["layer"]) + (1 if landmark["site"] == "ffn" else 0)


def summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def stratified_bootstrap(
    values: np.ndarray, relations: np.ndarray, resamples: int = RESAMPLES
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    strata = [np.flatnonzero(relations == relation) for relation in sorted(set(relations))]
    draws = np.empty(resamples, dtype=float)
    for trial in range(resamples):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in strata]
        )
        draws[trial] = np.mean(values[sampled])
    return {
        "estimate": float(np.mean(values)),
        "lower_95": float(np.percentile(draws, 2.5)),
        "upper_95": float(np.percentile(draws, 97.5)),
        "resamples": resamples,
        "seed": SEED,
    }


def localization_test(
    concentration: np.ndarray, emergence: np.ndarray, relations: np.ndarray
) -> dict[str, Any]:
    observed = float(np.median(np.abs(concentration - emergence)))
    rng = np.random.default_rng(SEED)
    strata = [np.flatnonzero(relations == relation) for relation in sorted(set(relations))]
    null = np.empty(RESAMPLES, dtype=float)
    for trial in range(RESAMPLES):
        permuted = emergence.copy()
        for indices in strata:
            permuted[indices] = rng.permutation(emergence[indices])
        null[trial] = np.median(np.abs(concentration - permuted))
    first = float(np.percentile(null, 1))
    return {
        "observed_median_site_distance": observed,
        "null_first_percentile": first,
        "null": summary(null),
        "lower_tail_empirical_p": float((1 + np.sum(null <= observed)) / (RESAMPLES + 1)),
        "passed": observed < first,
        "trials": RESAMPLES,
        "seed": SEED,
    }


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(
        np.sum(a * b, axis=1),
        denominator,
        out=np.full(a.shape[0], np.nan),
        where=denominator > 0,
    )


def load_checked_f32(path: Path, expected: str, shape: tuple[int, ...]) -> np.ndarray:
    if sha(path) != expected:
        raise ValueError(f"artifact hash mismatch: {path}")
    values = np.fromfile(path, dtype="<f4")
    if values.size != math.prod(shape):
        raise ValueError(f"artifact shape mismatch: {path}")
    return values.reshape(shape)


def physical_metrics(
    groups: list[dict[str, Any]],
    primary: np.ndarray,
    sealed_path: Path,
    edge_lookup: dict[str, int],
    emergence: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    sealed = json.loads(sealed_path.read_text())
    sealed_dir = sealed_path.parent
    census_path = sealed_dir / sealed["census"]["path"]
    if sha(census_path) != sealed["census"]["sha256"]:
        raise ValueError("sealed census changed during physical analysis")
    artifacts = (sealed_dir / sealed["artifact_root"]).resolve()
    census = {row["edge_id"]: row for row in jsonl(census_path)}
    curves = []
    for group in groups:
        deltas = []
        for edge_id in group["edge_ids"]:
            descriptor = next(
                item for item in census[edge_id]["artifacts"] if item["kind"] == "delta"
            )
            deltas.append(
                load_checked_f32(
                    artifacts / descriptor["path"], descriptor["sha256"], (68, 2560)
                )
            )
        curves.append(
            np.mean(
                [cosine_rows(deltas[left], deltas[right]) for left, right in combinations(range(3), 2)],
                axis=0,
            )
        )
    curves = np.asarray(curves)
    shared_emergence = np.array(
        [
            int(np.median([emergence[edge_lookup[edge_id]] for edge_id in group["edge_ids"]]))
            for group in groups
        ]
    )
    selected = curves[primary]
    selected_emergence = shared_emergence[primary]
    return (
        {
            "claim_boundary": "descriptive physical-route similarity; not a gate and not causal evidence",
            "primary_prompt_pair_delta_cosine": {
                "early_post_ffn": summary(np.mean(selected[:, EARLY], axis=1)),
                "shared_emergence": summary(selected[np.arange(len(selected)), selected_emergence]),
                "late_final_ffn": summary(selected[:, 67]),
            },
            "all_edge_site_curve": [float(value) for value in np.nanmean(curves, axis=0)],
            "primary_edge_site_curve": [float(value) for value in np.nanmean(selected, axis=0)],
        },
        {"delta_cosine": curves},
    )


def support_jaccard(groups: list[dict[str, Any]], primary: np.ndarray, report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    directory = report_path.parent / "attributions-final"
    supports: dict[str, set[tuple[int, int]]] = {}
    for descriptor in report["input_artifacts"]["attributions"]:
        path = directory / descriptor["path"]
        if sha(path) != descriptor["sha256"]:
            raise ValueError(f"GW-0B attribution hash mismatch: {path}")
        for row in jsonl(path):
            supports[row["edge_id"]] = {
                (int(item["layer"]), int(item["feature"]))
                for item in row["contributions"][:100]
            }
    all_values: list[float] = []
    primary_values: list[float] = []
    primary_group_ids = {index for index, value in enumerate(primary) if value}
    for group_index, group in enumerate(groups):
        for left, right in combinations(group["edge_ids"], 2):
            if left not in supports or right not in supports:
                continue
            union = supports[left] | supports[right]
            value = len(supports[left] & supports[right]) / len(union) if union else 1.0
            all_values.append(value)
            if group_index in primary_group_ids:
                primary_values.append(value)
    return {
        "claim_boundary": "top-100 FFN support only where both prompt rows were eligible",
        "all_eligible_prompt_pairs": summary(np.asarray(all_values)),
        "primary_eligible_prompt_pairs": summary(np.asarray(primary_values)),
    }


def landmark_summary(values: np.ndarray, emergence: np.ndarray, row_indices: np.ndarray) -> dict[str, Any]:
    return {
        "early_post_ffn": summary(np.mean(values[row_indices][:, EARLY], axis=1)),
        "per_row_emergence": summary(values[row_indices, emergence[row_indices]]),
        "late_final_ffn": summary(values[row_indices, 67]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    prereg_validation = validate_prereg(args.preregistration)
    prereg = json.loads(args.preregistration.read_text())
    candidates = json.loads(args.candidates.read_text())
    readout = json.loads(args.readout_manifest.read_text())
    if readout["preregistration_sha256"] != prereg_validation["preregistration_sha256"]:
        raise ValueError("readout is not bound to the frozen preregistration")
    if readout["candidate_identity_sha256"] != prereg_validation["candidate_identity_sha256"]:
        raise ValueError("readout is not bound to the frozen candidate set")
    readout_root = args.readout_manifest.parent
    logits_descriptor = readout["artifacts"]["logits"]
    rows_descriptor = readout["artifacts"]["rows"]
    logits_path = readout_root / logits_descriptor["path"]
    rows_path = readout_root / rows_descriptor["path"]
    if sha(logits_path) != logits_descriptor["sha256"] or sha(rows_path) != rows_descriptor["sha256"]:
        raise ValueError("readout artifact hash mismatch")
    rows = jsonl(rows_path)
    shape = tuple(readout["shape"])
    logits = np.memmap(logits_path, dtype="<f4", mode="r", shape=shape).astype(float)
    if len(rows) != shape[0] or shape != (426, 68, 126):
        raise ValueError("readout shape or row count changed")
    edge_lookup = {row["edge_id"]: index for index, row in enumerate(rows)}
    if len(edge_lookup) != 426:
        raise ValueError("readout edge IDs are not unique")

    means = np.mean(logits, axis=2, keepdims=True)
    stds = np.std(logits, axis=2, keepdims=True)
    if np.any(stds == 0):
        raise ValueError("a candidate state has zero logit variance")
    raw_global = softmax(logits)
    z_global = softmax((logits - means) / stds)
    token_ids = readout["candidate_token_ids"]
    token_position = {token: index for index, token in enumerate(token_ids)}
    subjects = {item["subject"]: item for item in candidates["subjects"]}

    metric: dict[str, dict[str, np.ndarray]] = {}
    for name, global_distribution, transformed_logits in (
        ("raw", raw_global, logits),
        ("zscore", z_global, (logits - means) / stds),
    ):
        neighbourhood_entropy = np.full((426, 68), np.nan)
        normalized_entropy = np.full((426, 68), np.nan)
        effective = np.full((426, 68), np.nan)
        neighbourhood_mass = np.full((426, 68), np.nan)
        target_global = np.full((426, 68), np.nan)
        target_conditional = np.full((426, 68), np.nan)
        margin = np.full((426, 68), np.nan)
        for row_index, row in enumerate(rows):
            subject = subjects[row["semantic_edge"]["subject"]]
            positions = [token_position[token] for token in subject["candidate_token_ids"]]
            local_logits = transformed_logits[row_index][:, positions]
            local = softmax(local_logits)
            local_entropy = entropy(local)
            neighbourhood_entropy[row_index] = local_entropy
            effective[row_index] = np.exp(local_entropy)
            neighbourhood_mass[row_index] = np.sum(
                global_distribution[row_index][:, positions], axis=1
            )
            target = token_position[row["semantic_edge"]["target_token_ids"][0]]
            target_global[row_index] = global_distribution[row_index, :, target]
            local_target = positions.index(target)
            target_conditional[row_index] = local[:, local_target]
            if len(positions) > 1:
                normalized_entropy[row_index] = local_entropy / math.log(len(positions))
                runner_up = np.max(np.delete(local, local_target, axis=1), axis=1)
                margin[row_index] = local[:, local_target] - runner_up
        metric[name] = {
            "global_entropy_normalized": entropy(global_distribution) / math.log(126),
            "neighbourhood_entropy": neighbourhood_entropy,
            "normalized_entropy": normalized_entropy,
            "effective_count": effective,
            "neighbourhood_mass": neighbourhood_mass,
            "target_global": target_global,
            "target_conditional": target_conditional,
            "target_runner_up_margin": margin,
        }

    groups = candidates["prompt_family_groups"]
    control = {
        item["edge_id"]: item["control_edge_id"]
        for item in candidates["different_destination_controls"]
    }
    group_rows = np.asarray([[edge_lookup[edge_id] for edge_id in group["edge_ids"]] for group in groups])
    control_rows = np.asarray(
        [[edge_lookup[control[edge_id]] for edge_id in group["edge_ids"]] for group in groups]
    )
    primary = np.asarray(
        [subjects[group["subject"]]["primary_multi_destination"] for group in groups], dtype=bool
    )
    relations = np.asarray([group["relation"] for group in groups])
    emergence = np.asarray([site_index(row["emergence"]) for row in rows])
    shared_emergence = np.asarray(
        [int(np.median(emergence[indices])) for indices in group_rows]
    )
    primary_indices = np.flatnonzero(primary)
    primary_relations = relations[primary]

    edge_metrics = []
    gate_values: dict[str, dict[str, np.ndarray]] = {}
    curves: dict[str, Any] = {}
    for name, distribution in (("raw", raw_global), ("zscore", z_global)):
        within = np.asarray([mean_pairwise_js(distribution[indices]) for indices in group_rows])
        matched = np.asarray([mean_pairwise_js(distribution[indices]) for indices in control_rows])
        early_within = np.mean(within[:, EARLY], axis=1)
        early_matched = np.mean(matched[:, EARLY], axis=1)
        at_emergence_within = within[np.arange(len(groups)), shared_emergence]
        at_emergence_matched = matched[np.arange(len(groups)), shared_emergence]
        did = (at_emergence_within - early_within) - (
            at_emergence_matched - early_matched
        )
        normalized = metric[name]["normalized_entropy"]
        row_early = np.mean(normalized[:, EARLY], axis=1)
        row_emergence = normalized[np.arange(426), emergence]
        concentration_delta = np.asarray(
            [np.mean(row_emergence[indices] - row_early[indices]) for indices in group_rows]
        )
        decreases = normalized[:, :-1] - normalized[:, 1:]
        concentration_site = np.argmax(decreases, axis=1) + 1
        edge_concentration_site = np.asarray(
            [int(np.median(concentration_site[indices])) for indices in group_rows]
        )
        gate_values[name] = {
            "did": did,
            "concentration_delta": concentration_delta,
            "concentration_site": edge_concentration_site,
            "early_within": early_within,
            "early_matched": early_matched,
            "emergence_within": at_emergence_within,
            "emergence_matched": at_emergence_matched,
        }
        curves[name] = {
            "primary_within_fact_js": [float(value) for value in np.mean(within[primary], axis=0)],
            "primary_matched_control_js": [float(value) for value in np.mean(matched[primary], axis=0)],
            "primary_neighbourhood_entropy_normalized": [
                float(value)
                for value in np.nanmean(metric[name]["normalized_entropy"][group_rows[primary]], axis=(0, 1))
            ],
        }

    for group_index, group in enumerate(groups):
        indices = group_rows[group_index]
        edge_metrics.append(
            {
                "subject": group["subject"],
                "relation": group["relation"],
                "target": group["target"],
                "edge_ids": group["edge_ids"],
                "primary": bool(primary[group_index]),
                "shared_emergence_site": int(shared_emergence[group_index]),
                "early_effective_count_median_raw": float(
                    np.median(metric["raw"]["effective_count"][indices][:, EARLY])
                ),
                "early_effective_count_median_zscore": float(
                    np.median(metric["zscore"]["effective_count"][indices][:, EARLY])
                ),
                "raw": {
                    "entropy_emergence_minus_early": float(
                        gate_values["raw"]["concentration_delta"][group_index]
                    ),
                    "concentration_site": int(
                        gate_values["raw"]["concentration_site"][group_index]
                    ),
                    "convergence_did": float(gate_values["raw"]["did"][group_index]),
                },
                "zscore": {
                    "entropy_emergence_minus_early": float(
                        gate_values["zscore"]["concentration_delta"][group_index]
                    ),
                    "concentration_site": int(
                        gate_values["zscore"]["concentration_site"][group_index]
                    ),
                    "convergence_did": float(gate_values["zscore"]["did"][group_index]),
                },
            }
        )

    coexist_raw = np.asarray(
        [
            edge_metrics[index]["early_effective_count_median_raw"] >= 1.5
            for index in primary_indices
        ]
    )
    coexist_zscore = np.asarray(
        [
            edge_metrics[index]["early_effective_count_median_zscore"] >= 1.5
            for index in primary_indices
        ]
    )
    concentration = {
        name: stratified_bootstrap(
            values["concentration_delta"][primary], primary_relations
        )
        for name, values in gate_values.items()
    }
    convergence = {}
    for name, values in gate_values.items():
        convergence[name] = stratified_bootstrap(values["did"][primary], primary_relations)
        convergence[name]["components"] = {
            "within_fact_early": summary(values["early_within"][primary]),
            "within_fact_shared_emergence": summary(values["emergence_within"][primary]),
            "matched_control_early": summary(values["early_matched"][primary]),
            "matched_control_shared_emergence": summary(
                values["emergence_matched"][primary]
            ),
        }
    localization = {
        name: localization_test(
            values["concentration_site"][primary],
            shared_emergence[primary],
            primary_relations,
        )
        for name, values in gate_values.items()
    }
    conditions = {
        "candidate_coexistence": bool(np.mean(coexist_raw) >= 0.5),
        "raw_and_zscore_concentration": bool(
            concentration["raw"]["upper_95"] < 0
            and concentration["zscore"]["upper_95"] < 0
        ),
        "localized_concentration": bool(localization["raw"]["passed"]),
        "raw_and_zscore_control_separated_convergence": bool(
            convergence["raw"]["upper_95"] < 0
            and convergence["zscore"]["upper_95"] < 0
        ),
    }
    conditions["full_support"] = all(conditions.values())

    physical, _ = physical_metrics(groups, primary, args.sealed_manifest, edge_lookup, emergence)
    physical["ffn_top100_support_jaccard"] = support_jaccard(
        groups, primary, args.gw0b_report
    )
    primary_rows = np.unique(group_rows[primary].reshape(-1))
    descriptive = {}
    for name in ("raw", "zscore"):
        descriptive[name] = {
            key: landmark_summary(metric[name][key], emergence, primary_rows)
            for key in (
                "global_entropy_normalized",
                "normalized_entropy",
                "effective_count",
                "neighbourhood_mass",
                "target_global",
                "target_conditional",
                "target_runner_up_margin",
            )
        }
    carrier_l2 = np.asarray(
        [[site["carrier_l2"] for site in row["sites"]] for row in rows], dtype=float
    )
    logit_std = np.asarray(
        [[site["candidate_logits"]["std"] for site in row["sites"]] for row in rows],
        dtype=float,
    )
    scale_diagnostics = {
        "carrier_l2": landmark_summary(carrier_l2, emergence, primary_rows),
        "candidate_logit_std": landmark_summary(logit_std, emergence, primary_rows),
    }
    gw3af = json.loads(args.gw3af_report.read_text())
    report = {
        "schema": SCHEMA,
        "status": "adjudicated_before_example_inspection",
        "adjudication_sha256": None,
        "authorities": {
            "preregistration_sha256": prereg_validation["preregistration_sha256"],
            "candidate_identity_sha256": prereg_validation["candidate_identity_sha256"],
            "readout_manifest_sha256": sha(args.readout_manifest),
            "candidate_logits_sha256": logits_descriptor["sha256"],
            "rows_sha256": rows_descriptor["sha256"],
            "sealed_bundle_sha256": readout["authorities"]["sealed_bundle_sha256"],
            "prepared_head_representation": readout["authorities"]["prepared_head_representation"],
            "selected_full_head_parity": readout["selected_full_head_parity"],
        },
        "denominators": {
            "candidate_states": 426 * 68,
            "candidate_tokens": 126,
            "primary_semantic_edges": int(np.sum(primary)),
            "primary_execution_rows": int(primary_rows.size),
            "secondary_semantic_edges": int(np.sum(~primary)),
        },
        "gate": {
            "candidate_coexistence": {
                "raw": {
                    "qualifying_edges": int(np.sum(coexist_raw)),
                    "primary_edges": int(coexist_raw.size),
                    "fraction": float(np.mean(coexist_raw)),
                },
                "zscore_scale_control": {
                    "qualifying_edges": int(np.sum(coexist_zscore)),
                    "primary_edges": int(coexist_zscore.size),
                    "fraction": float(np.mean(coexist_zscore)),
                },
                "threshold": ">= 0.5 with median early effective count >= 1.5",
                "passed": conditions["candidate_coexistence"],
            },
            "concentration": concentration,
            "localization": localization,
            "convergence": convergence,
            "conditions": conditions,
            "verdict": (
                "observational_transition_superposition_supported"
                if conditions["full_support"]
                else "full_conjunctive_gate_not_met"
            ),
        },
        "descriptive_semantic_surfaces": descriptive,
        "scale_diagnostics": scale_diagnostics,
        "aggregate_site_curves": curves,
        "physical_distribution": physical,
        "addressability": {
            "source": str(args.gw3af_report),
            "source_sha256": sha(args.gw3af_report),
            "progression_passed": gw3af["progression"]["passed"],
            "kept_separate_from_semantic_and_physical_metrics": True,
        },
        "interpretation_contract": prereg["interpretation_contract"],
        "examples_inspected_before_adjudication": False,
    }
    report["adjudication_sha256"] = canonical_hash(report, "adjudication_sha256")
    args.output.mkdir(parents=True, exist_ok=True)
    edge_path = args.output / "edge-metrics.jsonl"
    edge_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in edge_metrics))
    report["artifacts"] = {
        "edge_metrics": {"path": edge_path.name, "rows": len(edge_metrics), "sha256": sha(edge_path)}
    }
    # Adding the artifact binding changes the canonical report identity once.
    report["adjudication_sha256"] = canonical_hash(report, "adjudication_sha256")
    report_path = args.output / "adjudication.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--readout-manifest", type=Path, required=True)
    parser.add_argument("--sealed-manifest", type=Path, required=True)
    parser.add_argument("--gw0b-report", type=Path, required=True)
    parser.add_argument("--gw3af-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "schema": report["schema"],
        "status": report["status"],
        "adjudication_sha256": report["adjudication_sha256"],
        "gate": report["gate"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
