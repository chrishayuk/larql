#!/usr/bin/env python3
"""Audit GW-V2 execution readiness without running prompts or reading outcomes.

Protocol hash validation is necessary but does not resolve an unspecified
token correspondence in the registered subject-permutation fit control.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from gwkey1_preregister import role_row
from gwv2_population import read_jsonl, sha256
from gwv2_preregister import validate


def subject_lengths(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Check inherited role coverage and require consistent subject lengths."""
    result: dict[str, dict[str, int]] = {}
    subject_splits: dict[str, str] = {}
    for row in rows:
        role = role_row(row)
        subject = row["subject_id"]
        split = row["split"]
        if subject_splits.setdefault(subject, split) != split:
            raise ValueError(f"subject crosses splits: {subject}")
        count = len(role["roles"]["subject_entity"])
        subjects = result.setdefault(split, {})
        if subjects.setdefault(subject, count) != count:
            raise ValueError(f"subject token count changes across templates: {subject}")
    return result


def audit(protocol_path: Path) -> dict[str, Any]:
    checked = validate(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    root = protocol_path.parent
    rows = read_jsonl(root / protocol["authorities"]["input_rows"]["path"])
    lengths = subject_lengths(rows)

    predecessor_path = root / protocol["authorities"]["gwkey1_preregistration"]["path"]
    predecessor = json.loads(predecessor_path.read_text())
    role_descriptor = predecessor["roles"]["artifact"]
    role_path = predecessor_path.parent / role_descriptor["path"]
    if sha256(role_path) != role_descriptor["sha256"]:
        raise ValueError("inherited GW-KEY-1 role artifact changed")
    prior_roles = read_jsonl(role_path)
    available = {
        (row["template_id"], name)
        for row in prior_roles
        if row["split"] == "train"
        for name, positions in row["roles"].items()
        if positions
    }
    required = {
        (role["template_id"], name)
        for row in rows
        for role in [role_row(row)]
        for name, positions in role["roles"].items()
        if positions
    }
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"inherited reference-bank cells unavailable: {missing}")

    train_counts = Counter(lengths["train"].values())
    singleton_length_subjects = sorted(
        subject for subject, count in lengths["train"].items()
        if train_counts[count] == 1
    )
    split_reports = {}
    for split in ("train", "validation", "test"):
        counts = Counter(lengths[split].values())
        split_reports[split] = {
            "subjects": len(lengths[split]),
            "subject_token_count_histogram": dict(sorted(counts.items())),
            "executions": sum(row["split"] == split for row in rows),
            "subject_token_fit_rows": sum(
                lengths[split][row["subject_id"]] for row in rows if row["split"] == split
            ),
        }

    # Report an execution-specification gap, not a failed scientific gate.
    return {
        "schema": "larql.gwv2.preflight.v1",
        "status": "execution_definition_required",
        "protocol_sha256": checked["protocol_sha256"],
        "population_sha256": checked["population_sha256"],
        "protocol_and_population_valid": True,
        "inherited_role_rows_checked": len(rows),
        "inherited_role_artifact_sha256": role_descriptor["sha256"],
        "required_reference_cells": len(required),
        "missing_reference_cells": missing,
        "splits": split_reports,
        "fit_sham": {
            "registered_rule": protocol["controls"]["fit_sham"],
            "gap": "No source-to-target token correspondence is specified for permuted subjects of unequal token lengths.",
            "unrestricted_subject_permutation_can_change_token_count": len(train_counts) > 1,
            "token_count_stratification_would_be_an_explicit_clarification": True,
            "singleton_token_count_subjects": singleton_length_subjects,
            "proposed_rule": None,
            "amendment_applied": False,
        },
        "model_prompts_executed_by_this_audit": 0,
        "activation_tensors_read": False,
        "candidate_effects_computed": False,
        "scientific_verdict": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.protocol), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
