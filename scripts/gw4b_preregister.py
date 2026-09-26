#!/usr/bin/env python3
"""Validate the frozen GW-4B pre-write retrieval contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "larql.gw4b.preregistration.v1"


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


def validate(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA}")
    if document.get("status") != "frozen_pre_key_measurement":
        raise ValueError("GW-4B contract is not frozen_pre_key_measurement")
    identity = canonical_hash(document)
    if document.get("preregistration_sha256") != identity:
        raise ValueError("GW-4B preregistration identity mismatch")

    root = path.parent
    authorities = document["authorities"]
    gw4a_prereg_path = (root / authorities["gw4a_preregistration"]["path"]).resolve()
    gw4a_report_path = (root / authorities["gw4a_report"]["path"]).resolve()
    sealed_path = (root / authorities["sealed_gw0"]["path"]).resolve()
    for artifact_path, expected in (
        (gw4a_prereg_path, authorities["gw4a_preregistration"]["file_sha256"]),
        (gw4a_report_path, authorities["gw4a_report"]["sha256"]),
        (sealed_path, authorities["sealed_gw0"]["file_sha256"]),
    ):
        actual = sha256_file(artifact_path)
        if actual != expected:
            raise ValueError(f"authority hash mismatch for {artifact_path}: {actual}")

    gw4a_prereg = json.loads(gw4a_prereg_path.read_text())
    gw4a_report = json.loads(gw4a_report_path.read_text())
    sealed = json.loads(sealed_path.read_text())
    if gw4a_prereg.get("preregistration_sha256") != authorities["gw4a_preregistration"]["identity_sha256"]:
        raise ValueError("GW-4A preregistration identity mismatch")
    if gw4a_report.get("preregistration", {}).get("identity_sha256") != gw4a_prereg.get("preregistration_sha256"):
        raise ValueError("GW-4A report is not bound to its preregistration")
    if not gw4a_report.get("decision", {}).get("open_gw4b"):
        raise ValueError("GW-4A did not open GW-4B")
    if sealed.get("bundle_sha256") != authorities["sealed_gw0"]["bundle_sha256"]:
        raise ValueError("sealed GW-0 bundle identity mismatch")

    cohort = document["cohort"]
    evidence = gw4a_report.get("site_evidence", [])
    counts = {
        "eligible_sites": len(evidence),
        "training_rows": sum(row["split"] == cohort["training_split"] for row in evidence),
        "assessment_rows": sum(row["split"] in cohort["assessment_splits"] for row in evidence),
    }
    if any(counts[key] != cohort[key] for key in counts):
        raise ValueError(f"GW-4B cohort differs from GW-4A: {counts}")
    if any(row["causal_status"] != cohort["causal_status"] for row in evidence):
        raise ValueError("GW-4B causal status differs from the frozen cohort")

    return {
        "schema": SCHEMA,
        "status": "valid",
        "preregistration_sha256": identity,
        **counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preregistration", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.preregistration), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
