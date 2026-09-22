#!/usr/bin/env python3
"""Freeze GW-HEAD-1 head identities from a complete train-only subset surface."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

from gwsup1_preregister import canonical_hash, sha
from gwhead1_preregister import EXPECTED_HEADS, SCHEMA as PREREG_SCHEMA

INPUT_SCHEMA = "larql.gwhead1.train-subset-metrics.v1"
OUTPUT_SCHEMA = "larql.gwhead1.selection.v1"
METRICS = ("candidate_raw", "candidate_zscore")
SCOPES = ("global", "capital", "currency", "language", "hypernym")
EXPECTED_EDGES = {
    "global": 85,
    "capital": 23,
    "currency": 20,
    "language": 18,
    "hypernym": 24,
}
DENOMINATOR_EPS = 1e-12


def relative(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve())


def finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def powerset(head_ids: list[int]) -> set[tuple[int, ...]]:
    return {
        combination
        for size in range(len(head_ids) + 1)
        for combination in itertools.combinations(head_ids, size)
    }


def select_scope(scope_name: str, scope: dict[str, Any], q: float) -> dict[str, Any]:
    if scope.get("semantic_edges") != EXPECTED_EDGES[scope_name]:
        raise ValueError(f"{scope_name} train semantic-edge count changed")
    full = {metric: finite(scope["E_full"][metric], f"{scope_name} E_full {metric}") for metric in METRICS}
    reference = {
        metric: finite(scope["E_ref"][metric], f"{scope_name} E_ref {metric}")
        for metric in METRICS
    }
    if any(full[metric] <= 0.0 for metric in METRICS):
        raise ValueError(f"{scope_name} has a non-positive natural train effect")
    denominators = {metric: full[metric] - reference[metric] for metric in METRICS}
    if any(value <= DENOMINATOR_EPS for value in denominators.values()):
        raise ValueError(f"{scope_name} has a non-positive or unstable sufficiency denominator")

    seen: set[tuple[int, ...]] = set()
    eligible: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for index, row in enumerate(scope["subsets"]):
        heads = tuple(row["heads"])
        if heads != tuple(sorted(set(heads))) or any(h < 0 or h >= EXPECTED_HEADS for h in heads):
            raise ValueError(f"{scope_name} subset {index} has invalid head IDs")
        if heads in seen:
            raise ValueError(f"{scope_name} repeats subset {heads}")
        seen.add(heads)
        effect = {
            metric: finite(row["E"][metric], f"{scope_name} {heads} {metric}")
            for metric in METRICS
        }
        retention = {
            metric: (effect[metric] - reference[metric]) / denominators[metric]
            for metric in METRICS
        }
        norm = finite(
            row["total_natural_contribution_norm"],
            f"{scope_name} {heads} total contribution norm",
        )
        if norm < 0.0:
            raise ValueError(f"{scope_name} {heads} has a negative contribution norm")
        if all(retention[metric] >= q for metric in METRICS):
            result = {
                "heads": list(heads),
                "cardinality": len(heads),
                "E": effect,
                "retention": retention,
                "minimum_retention": min(retention.values()),
                "total_natural_contribution_norm": norm,
            }
            key = (len(heads), -result["minimum_retention"], norm, heads)
            eligible.append((key, result))

    expected = powerset(list(range(EXPECTED_HEADS)))
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(
            f"{scope_name} is not the complete {len(expected)}-subset surface; "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    if not eligible:
        raise ValueError(f"{scope_name} has no subset retaining q={q}")
    _, selected = min(eligible, key=lambda item: item[0])
    return {
        "semantic_edges": scope["semantic_edges"],
        "E_full": full,
        "E_ref": reference,
        "denominator": denominators,
        "eligible_subsets": len(eligible),
        "evaluated_subsets": len(seen),
        "selected": selected,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    prereg = json.loads(args.preregistration.read_text())
    metrics = json.loads(args.train_metrics.read_text())
    if prereg.get("schema") != PREREG_SCHEMA or prereg.get("status") != "frozen_pre_execution":
        raise ValueError("not a frozen GW-HEAD-1 preregistration")
    if canonical_hash(prereg, "preregistration_sha256") != prereg["preregistration_sha256"]:
        raise ValueError("GW-HEAD-1 preregistration identity mismatch")
    if metrics.get("schema") != INPUT_SCHEMA or metrics.get("status") != "complete_train_only":
        raise ValueError("not a complete train-only GW-HEAD-1 subset surface")
    if canonical_hash(metrics, "artifact_sha256") != metrics.get("artifact_sha256"):
        raise ValueError("GW-HEAD-1 train subset metric identity mismatch")
    if metrics.get("split") != "train" or metrics.get("held_out_rows") != 0:
        raise ValueError("GW-HEAD-1 selection input contains held-out outcomes")
    if metrics.get("preregistration_sha256") != prereg["preregistration_sha256"]:
        raise ValueError("train subset surface binds a different preregistration")
    if metrics.get("head_ids") != list(range(EXPECTED_HEADS)):
        raise ValueError("train subset surface changed the head universe")
    if set(metrics.get("scopes", {})) != set(SCOPES):
        raise ValueError("train subset surface does not contain exactly the frozen scopes")

    q = float(prereg["selection"]["train_retention_fraction"])
    selections = {
        scope: select_scope(scope, metrics["scopes"][scope], q) for scope in SCOPES
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
            "train_subset_metrics": {
                "path": relative(args.train_metrics, root),
                "sha256": sha(args.train_metrics),
                "artifact_sha256": metrics["artifact_sha256"],
            },
        },
        "selection_rule": {
            "retention_fraction": q,
            "metrics": list(METRICS),
            "tie_break": prereg["selection"]["tie_break"],
            "held_out_rows_seen": 0,
        },
        "scopes": selections,
        "interpretation": {
            "global_is_primary": True,
            "relation_subsets_are_independently_frozen": True,
            "shared_identity_required": False,
            "selection_is_not_adjudication": True,
        },
    }
    document["selection_sha256"] = canonical_hash(document, "selection_sha256")
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def validate(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("schema") != OUTPUT_SCHEMA or document.get("status") != "frozen_pre_heldout":
        raise ValueError("not a frozen GW-HEAD-1 selection")
    identity = canonical_hash(document, "selection_sha256")
    if identity != document.get("selection_sha256"):
        raise ValueError("GW-HEAD-1 selection identity mismatch")
    root = path.parent
    for name in ("preregistration", "train_subset_metrics"):
        item = document["authorities"][name]
        if sha((root / item["path"]).resolve()) != item["sha256"]:
            raise ValueError(f"{name} authority changed")
    if set(document["scopes"]) != set(SCOPES):
        raise ValueError("GW-HEAD-1 selected scopes changed")
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "valid",
        "selection_sha256": identity,
        "selected_heads": {
            name: document["scopes"][name]["selected"]["heads"] for name in SCOPES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--preregistration", type=Path, required=True)
    build_parser.add_argument("--train-metrics", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("selection", type=Path)
    args = parser.parse_args()
    result = build(args) if args.command == "build" else validate(args.selection)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
