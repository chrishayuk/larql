#!/usr/bin/env python3
"""Validate the frozen GW-4A write-family preregistration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "larql.gw4a.preregistration.v1"


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
    if document.get("status") != "frozen_pre_similarity":
        raise ValueError("GW-4A contract is not frozen_pre_similarity")
    identity = canonical_hash(document)
    if document.get("preregistration_sha256") != identity:
        raise ValueError("GW-4A preregistration hash mismatch")

    root = path.parent
    authorities = document["authorities"]
    input_path = bound_path(root, authorities["input_manifest"])
    sealed_path = bound_path(root, authorities["sealed_gw0"])
    reconciliation_path = bound_path(root, authorities["gw0b_reconciliation"])
    attribution_root = bound_path(root, authorities["gw0b_reconciliation"], "attribution_root")
    require_hash(input_path, authorities["input_manifest"]["file_sha256"])
    require_hash(sealed_path, authorities["sealed_gw0"]["file_sha256"])
    require_hash(reconciliation_path, authorities["gw0b_reconciliation"]["sha256"])

    input_manifest = json.loads(input_path.read_text())
    sealed = json.loads(sealed_path.read_text())
    reconciliation = json.loads(reconciliation_path.read_text())
    if input_manifest.get("manifest_sha256") != authorities["input_manifest"]["manifest_sha256"]:
        raise ValueError("input manifest identity mismatch")
    if sealed.get("bundle_sha256") != authorities["sealed_gw0"]["bundle_sha256"]:
        raise ValueError("sealed GW-0 bundle identity mismatch")
    if sealed["census"].get("sha256") != authorities["sealed_gw0"]["census_sha256"]:
        raise ValueError("sealed census identity mismatch")
    if reconciliation.get("bundle_sha256") != sealed.get("bundle_sha256"):
        raise ValueError("GW-0B reconciliation is not bound to GW-0")

    rows = 0
    for artifact in reconciliation["input_artifacts"]["attributions"]:
        artifact_path = attribution_root / artifact["path"]
        require_hash(artifact_path, artifact["sha256"])
        values = jsonl(artifact_path)
        if len(values) != artifact["rows"]:
            raise ValueError(f"attribution row count mismatch for {artifact_path}")
        if any(value.get("status") != "reconstructed" for value in values):
            raise ValueError(f"non-reconstructed attribution in {artifact_path}")
        rows += len(values)
    if rows != document["cohort"]["eligible_sites"]:
        raise ValueError("attribution set differs from the frozen GW-4A cohort")

    return {
        "schema": SCHEMA,
        "status": "valid",
        "preregistration_sha256": identity,
        "eligible_sites": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preregistration", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.preregistration), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
