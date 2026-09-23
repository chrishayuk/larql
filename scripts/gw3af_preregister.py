#!/usr/bin/env python3
"""Validate the frozen GW-3A-F execution-support preregistration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "larql.gw3af.preregistration.v1"
WIDTHS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_hash(document: dict[str, Any]) -> str:
    payload = dict(document)
    payload.pop("preregistration_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bound_path(root: Path, entry: dict[str, Any], field: str = "path") -> Path:
    value = entry.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"authority has no {field}")
    return (root / value).resolve()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"hash mismatch for {path}: expected {expected}, got {actual}")


def validate(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA}")
    if document.get("status") != "frozen_pre_lookup":
        raise ValueError("GW-3A-F contract is not frozen_pre_lookup")
    expected_hash = canonical_hash(document)
    if document.get("preregistration_sha256") != expected_hash:
        raise ValueError(
            "preregistration hash mismatch: "
            f"expected {expected_hash}, got {document.get('preregistration_sha256')}"
        )
    if document["posting_rule"].get("widths") != WIDTHS:
        raise ValueError("posting width ladder differs from the frozen ladder")
    gate = document["progression_gate"]
    if gate.get("minimum_micro_top100_address_recall") != 0.95:
        raise ValueError("address-recall gate differs from 0.95")
    if gate.get("maximum_address_weighted_candidate_coverage") != 0.1:
        raise ValueError("candidate-coverage gate differs from 0.10")
    if not gate.get("mass_recall_cannot_substitute_for_address_recall"):
        raise ValueError("mass recall must not substitute for address recall")

    root = path.parent
    authorities = document["authorities"]
    input_manifest_path = bound_path(root, authorities["input_manifest"])
    input_rows_path = bound_path(root, authorities["input_rows"])
    sealed_path = bound_path(root, authorities["sealed_gw0"])
    reconciliation_path = bound_path(root, authorities["gw0b_reconciliation"])
    promotions_path = bound_path(root, authorities["gw0b_promotions"])
    require_hash(input_manifest_path, authorities["input_manifest"]["file_sha256"])
    require_hash(input_rows_path, authorities["input_rows"]["sha256"])
    require_hash(sealed_path, authorities["sealed_gw0"]["file_sha256"])
    require_hash(reconciliation_path, authorities["gw0b_reconciliation"]["sha256"])
    require_hash(promotions_path, authorities["gw0b_promotions"]["sha256"])

    input_manifest = json.loads(input_manifest_path.read_text())
    sealed = json.loads(sealed_path.read_text())
    report = json.loads(reconciliation_path.read_text())
    if input_manifest.get("manifest_sha256") != authorities["input_manifest"]["manifest_sha256"]:
        raise ValueError("input manifest identity mismatch")
    if input_manifest["rows"].get("sha256") != authorities["input_rows"]["sha256"]:
        raise ValueError("input rows are not bound by the input manifest")
    if sealed.get("bundle_sha256") != authorities["sealed_gw0"]["bundle_sha256"]:
        raise ValueError("sealed GW-0 bundle identity mismatch")
    if sealed["census"].get("sha256") != authorities["sealed_gw0"]["census_sha256"]:
        raise ValueError("sealed census identity mismatch")
    if report.get("bundle_sha256") != sealed.get("bundle_sha256"):
        raise ValueError("GW-0B report is not bound to the sealed GW-0 bundle")
    if report["input_artifacts"]["promotions"].get("sha256") != authorities["gw0b_promotions"]["sha256"]:
        raise ValueError("GW-0B promotion authority mismatch")

    cohort = document["cohort"]
    denominators = report["denominators"]
    if denominators.get("reconstructed_ffn_candidate_sites") != cohort["eligible_sites"]:
        raise ValueError("reconstructed FFN cohort count mismatch")
    if denominators.get("triad_ineligible_attention_only_rows") != cohort["ineligible_attention_only_rows"]:
        raise ValueError("attention-only exclusion count mismatch")
    if denominators.get("execution_contribution_top_k") != cohort["execution_support_top_k"]:
        raise ValueError("execution support width mismatch")

    attribution_authority = authorities["gw0b_reconciliation"]
    attribution_root = bound_path(root, attribution_authority, "attribution_root")
    reconstructed = 0
    for artifact in report["input_artifacts"]["attributions"]:
        artifact_path = attribution_root / artifact["path"]
        require_hash(artifact_path, artifact["sha256"])
        rows = jsonl(artifact_path)
        if len(rows) != artifact["rows"]:
            raise ValueError(f"attribution row count mismatch for {artifact_path}")
        for row in rows:
            if row.get("status") != "reconstructed":
                raise ValueError(f"non-reconstructed attribution in {artifact_path}")
            if row.get("features") != cohort["eligible_addresses_per_site"]:
                raise ValueError(f"eligible address width mismatch in {artifact_path}")
            if len(row.get("contributions", [])) != cohort["execution_support_top_k"]:
                raise ValueError(f"support width mismatch in {artifact_path}")
            reconstructed += 1
    if reconstructed != cohort["eligible_sites"]:
        raise ValueError("attribution set does not contain the frozen cohort")

    return {
        "schema": SCHEMA,
        "status": "valid",
        "preregistration_sha256": expected_hash,
        "eligible_sites": reconstructed,
        "widths": WIDTHS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preregistration", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.preregistration), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
