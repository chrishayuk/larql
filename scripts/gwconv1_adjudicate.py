#!/usr/bin/env python3
"""Adjudicate frozen GW-CONV-1 before inspecting individual examples."""
from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from gwconv1_preregister import canonical_hash, sha, validate as validate_prereg

SCHEMA = "larql.gwconv1.adjudication.v1"
SEED = 1_398_100_531
RESAMPLES = 10_000
SITE_COUNT = 68
HIDDEN = 2560
RELATIONS = ("capital", "currency", "language", "hypernym")


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def mean_pairwise_js(distributions: np.ndarray) -> np.ndarray:
    parts = []
    tiny = np.finfo(float).tiny
    for left, right in combinations(range(distributions.shape[0]), 2):
        p = distributions[left]
        q = distributions[right]
        middle = (p + q) / 2.0
        parts.append(
            0.5 * np.sum(p * np.log(np.maximum(p / middle, tiny)), axis=-1)
            + 0.5 * np.sum(q * np.log(np.maximum(q / middle, tiny)), axis=-1)
        )
    return np.mean(parts, axis=0)


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        np.sum(left * right, axis=1),
        denominator,
        out=np.full(left.shape[0], np.nan),
        where=denominator > 0,
    )


def mean_pairwise_cosine(states: np.ndarray) -> np.ndarray:
    return np.mean(
        [cosine_rows(states[left], states[right]) for left, right in combinations(range(3), 2)],
        axis=0,
    )


def checked_memmap(root: Path, descriptor: dict[str, Any], shape: tuple[int, ...]) -> np.memmap:
    path = root / descriptor["path"]
    if sha(path) != descriptor["sha256"]:
        raise ValueError(f"artifact hash mismatch: {path}")
    if path.stat().st_size != math.prod(shape) * 4:
        raise ValueError(f"artifact byte length mismatch: {path}")
    return np.memmap(path, dtype="<f4", mode="r", shape=shape)


def artifact(row: dict[str, Any], kind: str) -> dict[str, Any]:
    matches = [item for item in row["artifacts"] if item["kind"] == kind]
    if len(matches) != 1:
        raise ValueError(f"{row['edge_id']}: expected one {kind} artifact")
    return matches[0]


def site_label(index: int) -> dict[str, Any]:
    return {
        "index": int(index),
        "layer": int(index // 2),
        "site": "ffn" if index % 2 else "attention",
    }


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


def bootstrap(values: np.ndarray, relations: np.ndarray, stratified: bool) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    relations = np.asarray(relations)
    if values.ndim != 1 or not len(values) or np.any(~np.isfinite(values)):
        raise ValueError("bootstrap requires a nonempty finite vector")
    rng = np.random.default_rng(SEED)
    if stratified:
        strata = [np.flatnonzero(relations == relation) for relation in sorted(set(relations))]
    else:
        strata = [np.arange(len(values))]
    draws = np.empty(RESAMPLES, dtype=float)
    for trial in range(RESAMPLES):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in strata]
        )
        draws[trial] = np.mean(values[sampled])
    return {
        "estimate": float(np.mean(values)),
        "lower_95": float(np.percentile(draws, 2.5)),
        "upper_95": float(np.percentile(draws, 97.5)),
        "resamples": RESAMPLES,
        "seed": SEED,
        "stratified": stratified,
    }


def concentration(profile: np.ndarray) -> dict[str, Any]:
    positive = np.maximum(np.asarray(profile, dtype=float), 0.0)
    total = float(np.sum(positive))
    if total == 0:
        return {
            "positive_mass": 0.0,
            "top_mass_fraction": {str(width): None for width in (1, 2, 4, 8)},
            "herfindahl": None,
            "effective_site_count": None,
        }
    shares = positive / total
    ordered = np.sort(shares)[::-1]
    herfindahl = float(np.sum(shares * shares))
    return {
        "positive_mass": total,
        "top_mass_fraction": {
            str(width): float(np.sum(ordered[:width])) for width in (1, 2, 4, 8)
        },
        "herfindahl": herfindahl,
        "effective_site_count": float(1.0 / herfindahl),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    prereg_validation = validate_prereg(args.preregistration)
    prereg = json.loads(args.preregistration.read_text())
    candidates = json.loads(args.candidates.read_text())
    before_manifest = json.loads(args.before_readout_manifest.read_text())
    after_manifest = json.loads(args.after_readout_manifest.read_text())
    sealed = json.loads(args.sealed_manifest.read_text())

    if before_manifest["schema"] != "larql.gwconv1.before-readout-manifest.v1":
        raise ValueError("not a GW-CONV-1 before-readout manifest")
    if before_manifest["preregistration_sha256"] != prereg_validation["preregistration_sha256"]:
        raise ValueError("before readout is not bound to the frozen preregistration")
    if before_manifest["candidate_identity_sha256"] != candidates["candidate_identity_sha256"]:
        raise ValueError("before readout candidate identity mismatch")
    if sha(args.after_readout_manifest) != prereg["authorities"]["gwsup1_after_readout"]["sha256"]:
        raise ValueError("GW-SUP-1 after-readout authority changed")
    if sha(args.sealed_manifest) != prereg["authorities"]["sealed_gw0"]["sha256"]:
        raise ValueError("sealed GW-0 authority changed")
    if before_manifest["authorities"]["sealed_bundle_sha256"] != sealed["bundle_sha256"]:
        raise ValueError("before readout used a different sealed corpus")
    if before_manifest["authorities"]["prepared_head_representation"] != "q8":
        raise ValueError("GW-CONV-1 before readout did not use prepared Q8")
    if before_manifest["selected_full_head_parity"]["bit_mismatches"] != 0:
        raise ValueError("selected/full output-head parity did not hold")
    if after_manifest["candidate_token_ids"] != before_manifest["candidate_token_ids"]:
        raise ValueError("before/after candidate token order differs")

    shape = (426, SITE_COUNT, 126)
    if tuple(before_manifest["shape"]) != shape or tuple(after_manifest["shape"]) != shape:
        raise ValueError("before/after readout shape changed")
    before_root = args.before_readout_manifest.parent
    after_root = args.after_readout_manifest.parent
    before_logits = checked_memmap(before_root, before_manifest["artifacts"]["logits"], shape)
    after_logits = checked_memmap(after_root, after_manifest["artifacts"]["logits"], shape)
    before_rows_path = before_root / before_manifest["artifacts"]["rows"]["path"]
    after_rows_path = after_root / after_manifest["artifacts"]["rows"]["path"]
    if sha(before_rows_path) != before_manifest["artifacts"]["rows"]["sha256"]:
        raise ValueError("before rows hash mismatch")
    if sha(after_rows_path) != after_manifest["artifacts"]["rows"]["sha256"]:
        raise ValueError("after rows hash mismatch")
    before_rows = jsonl(before_rows_path)
    after_rows = jsonl(after_rows_path)
    before_ids = [row["edge_id"] for row in before_rows]
    after_ids = [row["edge_id"] for row in after_rows]
    if before_ids != after_ids or len(set(before_ids)) != 426:
        raise ValueError("before/after execution row identity or order differs")
    for left, right in zip(before_rows, after_rows):
        left_sites = [(item["layer"], item["site"], item["index"]) for item in left["sites"]]
        right_sites = [(item["layer"], item["site"], item["index"]) for item in right["sites"]]
        if left_sites != right_sites or len(left_sites) != SITE_COUNT:
            raise ValueError(f"{left['edge_id']}: site universe differs")

    edge_lookup = {edge_id: index for index, edge_id in enumerate(before_ids)}
    groups = candidates["prompt_family_groups"]
    controls = {
        item["edge_id"]: item["control_edge_id"]
        for item in candidates["different_destination_controls"]
    }
    group_rows = np.asarray(
        [[edge_lookup[edge_id] for edge_id in group["edge_ids"]] for group in groups]
    )
    control_rows = np.asarray(
        [[edge_lookup[controls[edge_id]] for edge_id in group["edge_ids"]] for group in groups]
    )

    sealed_dir = args.sealed_manifest.parent
    census_path = sealed_dir / sealed["census"]["path"]
    if sha(census_path) != sealed["census"]["sha256"]:
        raise ValueError("sealed census hash mismatch")
    census_rows = jsonl(census_path)
    census = {row["edge_id"]: row for row in census_rows}
    if list(census) != before_ids:
        raise ValueError("sealed census and readout order differ")
    artifact_root = (sealed_dir / sealed["artifact_root"]).resolve()
    splits = np.asarray([census[group["edge_ids"][0]]["split"] for group in groups])
    relations = np.asarray([group["relation"] for group in groups])
    for group, split in zip(groups, splits):
        if any(census[edge_id]["split"] != split for edge_id in group["edge_ids"]):
            raise ValueError(f"semantic edge crosses splits: {group['subject']}")
    if {split: int(np.sum(splits == split)) for split in set(splits)} != prereg["cohort"]["splits"]:
        raise ValueError("semantic-edge split no longer matches preregistration")

    before_float = np.asarray(before_logits, dtype=float)
    after_float = np.asarray(after_logits, dtype=float)
    before_std = np.std(before_float, axis=2, keepdims=True)
    after_std = np.std(after_float, axis=2, keepdims=True)
    if np.any(before_std == 0) or np.any(after_std == 0):
        raise ValueError("zero candidate-logit variance")
    distributions = {
        "candidate_raw": (softmax(before_float), softmax(after_float)),
        "candidate_zscore": (
            softmax((before_float - np.mean(before_float, axis=2, keepdims=True)) / before_std),
            softmax((after_float - np.mean(after_float, axis=2, keepdims=True)) / after_std),
        ),
    }
    del before_float, after_float, before_std, after_std

    adjusted: dict[str, np.ndarray] = {}
    unadjusted: dict[str, dict[str, np.ndarray]] = {}
    levels: dict[str, dict[str, np.ndarray]] = {}
    for name, (before_distribution, after_distribution) in distributions.items():
        fact_before = np.asarray(
            [mean_pairwise_js(before_distribution[indices]) for indices in group_rows]
        )
        fact_after = np.asarray(
            [mean_pairwise_js(after_distribution[indices]) for indices in group_rows]
        )
        control_before = np.asarray(
            [mean_pairwise_js(before_distribution[indices]) for indices in control_rows]
        )
        control_after = np.asarray(
            [mean_pairwise_js(after_distribution[indices]) for indices in control_rows]
        )
        fact_gain = fact_before - fact_after
        control_gain = control_before - control_after
        adjusted[name] = fact_gain - control_gain
        unadjusted[name] = {"same_fact": fact_gain, "matched_control": control_gain}
        levels[name] = {
            "same_fact_before_js": fact_before,
            "same_fact_after_js": fact_after,
            "matched_control_before_js": control_before,
            "matched_control_after_js": control_after,
        }
    del distributions

    after_scales = np.asarray(
        [
            [1.0 if site.get("layer_scale") is None else float(site["layer_scale"]) for site in row["sites"]]
            for row in after_rows
        ],
        dtype=np.float32,
    )

    @lru_cache(maxsize=30)
    def load_carriers(edge_id: str, kind: str) -> np.ndarray:
        descriptor = artifact(census[edge_id], kind)
        path = artifact_root / descriptor["path"]
        if descriptor["shape"] != [SITE_COUNT, HIDDEN] or sha(path) != descriptor["sha256"]:
            raise ValueError(f"{edge_id}: invalid {kind} artifact")
        values = np.fromfile(path, dtype="<f4")
        if values.size != SITE_COUNT * HIDDEN:
            raise ValueError(f"{edge_id}: invalid {kind} shape")
        values = values.reshape(SITE_COUNT, HIDDEN)
        if kind == "carrier-after":
            values = values * after_scales[edge_lookup[edge_id], :, None]
        return values

    fact_carrier_before = np.empty((len(groups), SITE_COUNT), dtype=float)
    fact_carrier_after = np.empty_like(fact_carrier_before)
    control_carrier_before = np.empty_like(fact_carrier_before)
    control_carrier_after = np.empty_like(fact_carrier_before)
    for group_index, group in enumerate(groups):
        fact_before_states = np.asarray(
            [load_carriers(edge_id, "carrier-before") for edge_id in group["edge_ids"]]
        )
        fact_after_states = np.asarray(
            [load_carriers(edge_id, "carrier-after") for edge_id in group["edge_ids"]]
        )
        control_ids = [controls[edge_id] for edge_id in group["edge_ids"]]
        control_before_states = np.asarray(
            [load_carriers(edge_id, "carrier-before") for edge_id in control_ids]
        )
        control_after_states = np.asarray(
            [load_carriers(edge_id, "carrier-after") for edge_id in control_ids]
        )
        fact_carrier_before[group_index] = mean_pairwise_cosine(fact_before_states)
        fact_carrier_after[group_index] = mean_pairwise_cosine(fact_after_states)
        control_carrier_before[group_index] = mean_pairwise_cosine(control_before_states)
        control_carrier_after[group_index] = mean_pairwise_cosine(control_after_states)
        if (group_index + 1) % 20 == 0 or group_index + 1 == len(groups):
            print(f"GW-CONV-1 carrier convergence {group_index + 1}/{len(groups)}", flush=True)
    fact_carrier_gain = fact_carrier_after - fact_carrier_before
    control_carrier_gain = control_carrier_after - control_carrier_before
    adjusted["carrier"] = fact_carrier_gain - control_carrier_gain
    unadjusted["carrier"] = {
        "same_fact": fact_carrier_gain,
        "matched_control": control_carrier_gain,
    }
    levels["carrier"] = {
        "same_fact_before_cosine": fact_carrier_before,
        "same_fact_after_cosine": fact_carrier_after,
        "matched_control_before_cosine": control_carrier_before,
        "matched_control_after_cosine": control_carrier_after,
    }

    train = splits == "train"
    validation = splits == "validation"
    test = splits == "test"
    raw_train_profile = np.mean(adjusted["candidate_raw"][train], axis=0)
    global_site = int(np.argmax(raw_train_profile))
    relation_sites = {
        relation: int(np.argmax(np.mean(adjusted["candidate_raw"][train & (relations == relation)], axis=0)))
        for relation in RELATIONS
    }

    def evaluate_at(indices: np.ndarray, sites: np.ndarray | int, stratified: bool) -> dict[str, Any]:
        selected_relations = relations[indices]
        row_ids = np.flatnonzero(indices)
        if isinstance(sites, int):
            positions = np.full(len(row_ids), sites, dtype=int)
        else:
            positions = np.asarray(sites, dtype=int)
            if len(positions) != len(row_ids):
                raise ValueError("site vector does not match evaluation cohort")
        result = {}
        for name in ("carrier", "candidate_raw", "candidate_zscore"):
            values = adjusted[name][row_ids, positions]
            result[name] = bootstrap(values, selected_relations, stratified)
            result[name]["summary"] = summary(values)
            result[name]["components"] = {
                component: bootstrap(
                    unadjusted[name][component][row_ids, positions],
                    selected_relations,
                    stratified,
                )
                for component in ("same_fact", "matched_control")
            }
            result[name]["levels"] = {
                level: bootstrap(values_at_level[row_ids, positions], selected_relations, stratified)
                for level, values_at_level in levels[name].items()
            }
        return result

    global_evaluation = {
        "validation": evaluate_at(validation, global_site, True),
        "test": evaluate_at(test, global_site, True),
    }
    relation_test_rows = np.flatnonzero(test)
    relation_test_sites = np.asarray([relation_sites[relations[index]] for index in relation_test_rows])
    relation_routing = evaluate_at(test, relation_test_sites, True)
    relation_breakdown: dict[str, Any] = {}
    for split_name, split_mask in (("validation", validation), ("test", test)):
        relation_breakdown[split_name] = {}
        for relation in RELATIONS:
            mask = split_mask & (relations == relation)
            relation_breakdown[split_name][relation] = {
                "n": int(np.sum(mask)),
                "global_site_candidate_raw_mean": float(
                    np.mean(adjusted["candidate_raw"][mask, global_site])
                ),
                "relation_site": site_label(relation_sites[relation]),
                "relation_site_candidate_raw": bootstrap(
                    adjusted["candidate_raw"][mask, relation_sites[relation]],
                    relations[mask],
                    False,
                ),
            }

    global_pass = all(
        global_evaluation[split_name][name]["lower_95"] > 0
        for split_name in ("validation", "test")
        for name in ("carrier", "candidate_raw", "candidate_zscore")
    )
    relation_pass = all(
        relation_routing[name]["lower_95"] > 0
        for name in ("carrier", "candidate_raw", "candidate_zscore")
    )
    positive_relation_counts = {
        split_name: sum(
            relation_breakdown[split_name][relation]["global_site_candidate_raw_mean"] > 0
            for relation in RELATIONS
        )
        for split_name in ("validation", "test")
    }
    cross_relation_pass = all(count >= 3 for count in positive_relation_counts.values())

    shared_emergence = np.asarray(
        [
            int(
                np.median(
                    [
                        2 * int(census[edge_id]["emergence"]["layer"])
                        + (1 if census[edge_id]["emergence"]["site"] == "ffn" else 0)
                        for edge_id in group["edge_ids"]
                    ]
                )
            )
            for group in groups
        ]
    )
    timing = {}
    for name, mask, selected_sites in (
        ("global_validation", validation, np.full(int(np.sum(validation)), global_site)),
        ("global_test", test, np.full(int(np.sum(test)), global_site)),
        ("relation_test", test, relation_test_sites),
    ):
        distances = selected_sites - shared_emergence[mask]
        timing[name] = {
            "signed_site_distance": summary(distances),
            "fraction_at_or_before_emergence": float(np.mean(distances <= 0)),
        }

    profiles: dict[str, Any] = {
        "schema": "larql.gwconv1.site-profiles.v1",
        "selection_split": "train",
        "sites": [site_label(index) for index in range(SITE_COUNT)],
        "global": {},
        "relations": {},
    }
    for name in ("carrier", "candidate_raw", "candidate_zscore"):
        profile = np.mean(adjusted[name][train], axis=0)
        profiles["global"][name] = {
            "mean_adjusted_gain": [float(value) for value in profile],
            "concentration": concentration(profile),
            "attention_mean": float(np.mean(profile[0::2])),
            "ffn_mean": float(np.mean(profile[1::2])),
        }
    for relation in RELATIONS:
        mask = train & (relations == relation)
        profiles["relations"][relation] = {
            name: [float(value) for value in np.mean(adjusted[name][mask], axis=0)]
            for name in ("carrier", "candidate_raw", "candidate_zscore")
        }

    edge_metrics = []
    for index, group in enumerate(groups):
        edge_metrics.append(
            {
                "subject": group["subject"],
                "relation": group["relation"],
                "target": group["target"],
                "edge_ids": group["edge_ids"],
                "split": str(splits[index]),
                "shared_emergence_site": int(shared_emergence[index]),
                "global_site_adjusted_gain": {
                    name: float(adjusted[name][index, global_site])
                    for name in ("carrier", "candidate_raw", "candidate_zscore")
                },
                "relation_site": relation_sites[group["relation"]],
                "relation_site_adjusted_gain": {
                    name: float(adjusted[name][index, relation_sites[group["relation"]]])
                    for name in ("carrier", "candidate_raw", "candidate_zscore")
                },
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    profiles_path = args.output / "site-profiles.json"
    profiles_path.write_text(json.dumps(profiles, indent=2, sort_keys=True) + "\n")
    edge_path = args.output / "edge-metrics.jsonl"
    edge_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in edge_metrics))
    conditions = {
        "global_operator": global_pass,
        "relation_routing": relation_pass,
        "cross_relation_generality": cross_relation_pass,
    }
    conditions["full_support"] = all(conditions.values())
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "adjudicated_before_example_inspection",
        "adjudication_sha256": None,
        "authorities": {
            "preregistration_sha256": prereg_validation["preregistration_sha256"],
            "candidate_identity_sha256": candidates["candidate_identity_sha256"],
            "before_readout_manifest_sha256": sha(args.before_readout_manifest),
            "before_logits_sha256": before_manifest["artifacts"]["logits"]["sha256"],
            "before_rows_sha256": before_manifest["artifacts"]["rows"]["sha256"],
            "after_readout_manifest_sha256": sha(args.after_readout_manifest),
            "after_logits_sha256": after_manifest["artifacts"]["logits"]["sha256"],
            "after_rows_sha256": after_manifest["artifacts"]["rows"]["sha256"],
            "sealed_bundle_sha256": sealed["bundle_sha256"],
            "prepared_head_representation": before_manifest["authorities"]["prepared_head_representation"],
            "selected_full_head_parity": before_manifest["selected_full_head_parity"],
        },
        "denominators": {
            "semantic_edges": len(groups),
            "execution_rows": 426,
            "sites": SITE_COUNT,
            "splits": {name: int(np.sum(splits == name)) for name in ("train", "validation", "test")},
        },
        "selection": {
            "criterion": "maximum train mean candidate_raw adjusted convergence gain; earliest exact tie",
            "global_site": site_label(global_site),
            "global_train_candidate_raw_adjusted_gain": float(raw_train_profile[global_site]),
            "relation_sites": {relation: site_label(site) for relation, site in relation_sites.items()},
            "held_out_outcomes_used_for_selection": False,
        },
        "held_out": {
            "global_site": global_evaluation,
            "relation_routing_test": relation_routing,
            "relation_breakdown": relation_breakdown,
        },
        "gate": {
            "conditions": conditions,
            "positive_relations_at_global_site": positive_relation_counts,
            "verdict": (
                "observational_convergence_operator_supported"
                if conditions["full_support"]
                else "full_conjunctive_gate_not_met"
            ),
        },
        "site_profile_descriptives": {
            "source": profiles_path.name,
            "sha256": sha(profiles_path),
            "claim_boundary": "training-split descriptive concentration; cannot rescue a held-out gate",
        },
        "emergence_timing": timing,
        "interpretation_contract": prereg["interpretation_contract"],
        "examples_inspected_before_adjudication": False,
        "artifacts": {
            "edge_metrics": {"path": edge_path.name, "rows": len(edge_metrics), "sha256": sha(edge_path)},
            "site_profiles": {"path": profiles_path.name, "sha256": sha(profiles_path)},
        },
    }
    report["adjudication_sha256"] = canonical_hash(report, "adjudication_sha256")
    report_path = args.output / "adjudication.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--before-readout-manifest", type=Path, required=True)
    parser.add_argument("--after-readout-manifest", type=Path, required=True)
    parser.add_argument("--sealed-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "status": report["status"],
                "adjudication_sha256": report["adjudication_sha256"],
                "selection": report["selection"],
                "held_out": report["held_out"],
                "gate": report["gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
