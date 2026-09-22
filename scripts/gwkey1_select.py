#!/usr/bin/env python3
"""Freeze GW-KEY-1 K, V, and joint source-role subsets from train only."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

from gwsup1_preregister import canonical_hash, sha
from gwkey1_preregister import ROLES, SCHEMA as PREREG_SCHEMA

INPUT_SCHEMA = "larql.gwkey1.train-source-metrics.v1"
OUTPUT_SCHEMA = "larql.gwkey1.selection.v1"
METRICS = ("candidate_raw", "candidate_zscore")
SUBSETS = 64
FULL = 63
EPS = 1e-12


def relative(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve())


def finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def roles(mask: int) -> list[str]:
    return [role for index, role in enumerate(ROLES) if mask & (1 << index)]


def select(
    family: str,
    candidates: list[tuple[int, int]],
    effect: Callable[[int, int], dict[str, float]],
    reference: tuple[int, int],
    full_effect: dict[str, float],
    q: float,
    k_norms: list[float],
    v_norms: list[float],
) -> dict[str, Any]:
    reference_effect = effect(*reference)
    denominator = {
        metric: full_effect[metric] - reference_effect[metric] for metric in METRICS
    }
    if any(full_effect[metric] <= 0 for metric in METRICS):
        raise ValueError("GW-KEY-1 natural train effect is not positive")
    if any(denominator[metric] <= EPS for metric in METRICS):
        return {
            "estimable": False,
            "refusal": "non-positive or unstable train sufficiency denominator",
            "reference": {
                "K_mask": reference[0],
                "V_mask": reference[1],
                "E": reference_effect,
            },
            "denominator": denominator,
            "evaluated_subsets": len(candidates),
            "eligible_subsets": 0,
            "selected": None,
        }
    eligible: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for key_mask, value_mask in candidates:
        candidate_effect = effect(key_mask, value_mask)
        retention = {
            metric: (candidate_effect[metric] - reference_effect[metric])
            / denominator[metric]
            for metric in METRICS
        }
        if not all(retention[metric] >= q for metric in METRICS):
            continue
        minimum = min(retention.values())
        replacement_norm = k_norms[key_mask] + v_norms[value_mask]
        if family == "K":
            key = (key_mask.bit_count(), -minimum, replacement_norm, key_mask)
        elif family == "V":
            key = (value_mask.bit_count(), -minimum, replacement_norm, value_mask)
        else:
            key = (
                (key_mask | value_mask).bit_count(),
                key_mask.bit_count() + value_mask.bit_count(),
                -minimum,
                replacement_norm,
                key_mask,
                value_mask,
            )
        result = {
            "K_mask": key_mask,
            "V_mask": value_mask,
            "K_roles": roles(key_mask),
            "V_roles": roles(value_mask),
            "union_roles": roles(key_mask | value_mask),
            "union_cardinality": (key_mask | value_mask).bit_count(),
            "total_cardinality": key_mask.bit_count() + value_mask.bit_count(),
            "E": candidate_effect,
            "retention": retention,
            "minimum_retention": minimum,
            "replacement_norm": replacement_norm,
        }
        eligible.append((key, result))
    if not eligible:
        return {
            "estimable": True,
            "refusal": f"no train subset retains q={q}",
            "reference": {
                "K_mask": reference[0],
                "V_mask": reference[1],
                "E": reference_effect,
            },
            "denominator": denominator,
            "evaluated_subsets": len(candidates),
            "eligible_subsets": 0,
            "selected": None,
        }
    _, selected = min(eligible, key=lambda item: item[0])
    if family == "K":
        complement = (FULL ^ selected["K_mask"], FULL)
    elif family == "V":
        complement = (FULL, FULL ^ selected["V_mask"])
    else:
        complement = (FULL ^ selected["K_mask"], FULL ^ selected["V_mask"])
    selected["full_minus_selected"] = {
        "K_mask": complement[0],
        "V_mask": complement[1],
        "K_roles": roles(complement[0]),
        "V_roles": roles(complement[1]),
    }
    return {
        "estimable": True,
        "reference": {
            "K_mask": reference[0],
            "V_mask": reference[1],
            "E": reference_effect,
        },
        "denominator": denominator,
        "evaluated_subsets": len(candidates),
        "eligible_subsets": len(eligible),
        "selected": selected,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    prereg = json.loads(args.preregistration.read_text())
    metrics = json.loads(args.train_metrics.read_text())
    if prereg.get("schema") != PREREG_SCHEMA or prereg.get("status") != "frozen_pre_execution":
        raise ValueError("not a frozen GW-KEY-1 preregistration")
    if canonical_hash(prereg, "preregistration_sha256") != prereg["preregistration_sha256"]:
        raise ValueError("GW-KEY-1 preregistration identity mismatch")
    if metrics.get("schema") != INPUT_SCHEMA or metrics.get("status") != "complete_train_only":
        raise ValueError("not a complete GW-KEY-1 train metric surface")
    if canonical_hash(metrics, "artifact_sha256") != metrics.get("artifact_sha256"):
        raise ValueError("GW-KEY-1 train metric identity mismatch")
    if metrics.get("split") != "train" or metrics.get("held_out_rows") != 0:
        raise ValueError("GW-KEY-1 selection input contains held-out outcomes")
    if metrics.get("preregistration_sha256") != prereg["preregistration_sha256"]:
        raise ValueError("GW-KEY-1 metrics bind a different preregistration")
    if metrics.get("roles") != list(ROLES):
        raise ValueError("GW-KEY-1 role order changed")

    pair_lookup: dict[tuple[int, int], dict[str, float]] = {}
    for row in metrics["pairs"]:
        key = (int(row["K_mask"]), int(row["V_mask"]))
        if key in pair_lookup or not all(0 <= value < SUBSETS for value in key):
            raise ValueError(f"invalid or duplicate GW-KEY-1 pair {key}")
        pair_lookup[key] = {
            metric: finite(row["E"][metric], f"{key} {metric}") for metric in METRICS
        }
    expected = {(key, value) for key in range(SUBSETS) for value in range(SUBSETS)}
    if set(pair_lookup) != expected:
        raise ValueError("GW-KEY-1 metric surface is not the complete 4,096 pairs")
    k_norms = [finite(value, "K replacement norm") for value in metrics["replacement_norm"]["K_by_mask"]]
    v_norms = [finite(value, "V replacement norm") for value in metrics["replacement_norm"]["V_by_mask"]]
    if len(k_norms) != SUBSETS or len(v_norms) != SUBSETS or any(value < 0 for value in [*k_norms, *v_norms]):
        raise ValueError("GW-KEY-1 replacement norm surface changed")
    effect = lambda key, value: pair_lookup[(key, value)]
    full_effect = effect(FULL, FULL)
    q = float(prereg["search"]["retention_fraction"])
    selections = {
        "K": select(
            "K",
            [(mask, FULL) for mask in range(SUBSETS)],
            effect,
            (0, FULL),
            full_effect,
            q,
            k_norms,
            v_norms,
        ),
        "V": select(
            "V",
            [(FULL, mask) for mask in range(SUBSETS)],
            effect,
            (FULL, 0),
            full_effect,
            q,
            k_norms,
            v_norms,
        ),
        "joint": select(
            "joint",
            sorted(expected),
            effect,
            (0, 0),
            full_effect,
            q,
            k_norms,
            v_norms,
        ),
    }
    root = args.output.parent
    document: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "status": "frozen_pre_heldout",
        "selection_sha256": None,
        "preregistration_sha256": prereg["preregistration_sha256"],
        "authorities": {
            "preregistration": {
                "path": relative(args.preregistration, root),
                "sha256": sha(args.preregistration),
            },
            "train_source_metrics": {
                "path": relative(args.train_metrics, root),
                "sha256": sha(args.train_metrics),
                "artifact_sha256": metrics["artifact_sha256"],
            },
        },
        "selection_rule": {
            "retention_fraction": q,
            "metrics": list(METRICS),
            "tie_break_K_or_V": prereg["search"]["tie_break_K_or_V"],
            "tie_break_joint": prereg["search"]["tie_break_joint"],
            "held_out_rows_seen": 0,
        },
        "E_full": full_effect,
        "families": selections,
        "compact_joint": selections["joint"]["selected"] is not None
        and selections["joint"]["selected"]["union_cardinality"] <= 3,
        "interpretation": {
            "selection_is_not_adjudication": True,
            "K_V_and_joint_are_independently_frozen": True,
            "joint_only_can_earn_candidate_read_path": True,
        },
    }
    document["selection_sha256"] = canonical_hash(document, "selection_sha256")
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def validate(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("schema") != OUTPUT_SCHEMA or document.get("status") != "frozen_pre_heldout":
        raise ValueError("not a frozen GW-KEY-1 selection")
    identity = canonical_hash(document, "selection_sha256")
    if identity != document.get("selection_sha256"):
        raise ValueError("GW-KEY-1 selection identity mismatch")
    root = path.parent
    for name in ("preregistration", "train_source_metrics"):
        item = document["authorities"][name]
        if sha((root / item["path"]).resolve()) != item["sha256"]:
            raise ValueError(f"{name} authority changed")
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "valid",
        "selection_sha256": identity,
        "selected": {
            name: (
                None
                if family["selected"] is None
                else {
                    "K_roles": family["selected"]["K_roles"],
                    "V_roles": family["selected"]["V_roles"],
                }
            )
            for name, family in document["families"].items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--preregistration", type=Path, required=True)
    build_parser.add_argument("--train-metrics", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("selection", type=Path)
    args = parser.parse_args()
    result = build(args) if args.command == "build" else validate(args.selection)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
