#!/usr/bin/env python3
"""Outcome-free authority and matched-donor audit for the GW-STATE-1 design."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from gwv2_amend import validate as validate_v2_lineage
from gwv2_io import array, checked_manifest
from gwv2_population import canonical_bytes, canonical_hash, sha256
from gwv2_prepare import validate_execution


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "bench/gw-v2/gemma3-4b-it-phase1"
EXECUTION = BASE / "execution.json"
CAPTURE = ROOT / "output/gwv2-gemma3-4b-it-phase1-capture/capture.json"
REPLAY = ROOT / "output/gwv2-gemma3-4b-it-phase1-replay2/replay.json"
ADJUDICATION = ROOT / "output/gwv2-gemma3-4b-it-phase1-adjudication.json"

AUTHORITIES = {
    "v2_protocol": (BASE / "gwv2-protocol.json", "da84a97940037221f2f260d58905154be138fe28a0f784c317e8a4f52f8ab360"),
    "v2_population": (BASE / "population-manifest.json", "92d9611ed22719c3a00cf72b3e545440c7e17ad3b2bc42d2129b6860503ee94a"),
    "v2_rows": (BASE / "input.jsonl", "6645a45f926e356230dee71be4fc406176030eccfa4fe0c7036a23ddb26b6c19"),
    "v2_execution": (EXECUTION, "0d266f8137aee2a79ce1710b26a34d0380f1ad3828081d71cc4f8d54a5c3efd5"),
    "v2_capture": (CAPTURE, "82a2b4292096b60297cbf974903b91339e02b2a3d594dece231807df0d84b89d"),
    "v2_replay": (REPLAY, "b9f89ee87c482f95307ee8c461d232345b902b42a38b9154b75db9b9b9294559"),
    "v2_adjudication": (ADJUDICATION, "56579a3868855089312d127ce2b301a8d4699a9641cd94289ff4b725a4295538"),
    "head_preregistration": (ROOT / "bench/gw0/gemma3-4b-it-phase1/gwhead1-preregistration.json", "737e7d01a3ce21426a2e1500940d359967beb2674d4ff0506a9d15cbc2b8ef0d"),
    "head_selection": (ROOT / "output/gwhead1-gemma3-4b-it-phase1/selection.json", "98414482be94b67d83177d9cdde7bdf6570bf8edee297ed71eec0c4c7344cfc1"),
    "head_adjudication": (ROOT / "output/gwhead1-gemma3-4b-it-phase1-heldout-full/adjudication.json", "520daa0726b6924ab2551650f84245d5271aad8a4860693a8f9cfeca5a8bf8b5"),
    "key_preregistration": (ROOT / "bench/gw0/gemma3-4b-it-phase1/gwkey1-preregistration.json", "372f6f56a87f5d579d4b47d8cb890d1f83477e7ed33b6c9efb28ea3842dee32a"),
    "key_selection": (ROOT / "output/gwkey1-gemma3-4b-it-phase1-search/selection-v2.json", "6cae920c092180102ec2880a7c365e1a3c8217aadfe54b5fd4c357799cec64fe"),
    "key_adjudication": (ROOT / "output/gwkey1-gemma3-4b-it-phase1-heldout2/adjudication.json", "04b083c3e113b3f7d4b2efe2886f660d06b9b72b5d34c22e6291dc3ca6d35351"),
}


def matched_donors(rows: list[dict]) -> list[int]:
    """Return the frozen matched-control row index, refusing any non-derangement."""
    lookup = {item["row"]["edge_id"]: index for index, item in enumerate(rows)}
    if len(rows) != 666 or len(lookup) != 666:
        raise ValueError("STATE-1 requires 666 distinct frozen execution rows")
    donors = []
    for index, item in enumerate(rows):
        target = item["row"]
        controls = [c for c in target["control_ids"] if c["kind"] == "same_relation_different_subject"]
        if len(controls) != 1 or controls[0]["paired_edge_id"] not in lookup:
            raise ValueError(f"missing unique matched donor for row {index}")
        donor_index = lookup[controls[0]["paired_edge_id"]]
        donor = rows[donor_index]["row"]
        left, right = target["semantic_edge"], donor["semantic_edge"]
        if (donor_index == index or target["subject_id"] == donor["subject_id"]
            or target["split"] != donor["split"]
            or left["relation"] != right["relation"]
            or left["prompt_semantic_family"] != right["prompt_semantic_family"]):
            raise ValueError(f"invalid matched donor for row {index}")
        donors.append(donor_index)
    if Counter(donors) != Counter(range(666)):
        raise ValueError("matched-control donors are not one-to-one")
    return donors


def audit(*, verify_segments: bool = False) -> dict:
    hashes = {}
    for name, (path, digest) in AUTHORITIES.items():
        actual = sha256(path)
        if actual != f"sha256:{digest}":
            raise ValueError(f"{name} authority changed: {path}")
        hashes[name] = actual
    execution = validate_execution(EXECUTION)
    lineage = validate_v2_lineage(Path(execution["protocol_path"]))
    capture = checked_manifest(CAPTURE, "larql.gwv2.capture.v1", lineage)
    replay = checked_manifest(REPLAY, "larql.gwv2.replay.v1", lineage)
    for manifest_path, manifest in ((CAPTURE, capture), (REPLAY, replay)):
        for artifact in manifest["artifacts"]:
            array(manifest_path.parent, manifest, artifact["path"])
    adjudication = json.loads(ADJUDICATION.read_text())
    if (capture["execution_file_sha256"] != hashes["v2_execution"]
        or capture["source_and_carrier_bit_mismatches"] != 0
        or capture["outcomes_captured"] is not False
        or replay["execution_file_sha256"] != hashes["v2_execution"]
        or replay["capture_file_sha256"] != hashes["v2_capture"]
        or replay["parity_bit_mismatches"] != 0
        or adjudication["adjudication_sha256"] != canonical_hash(adjudication, "adjudication_sha256")
        or adjudication["lineage"] != lineage):
        raise ValueError("GW-V2 control or adjudication authority changed")
    donors = matched_donors(execution["rows"])
    counts = Counter(item["row"]["split"] for item in execution["rows"])
    if counts != {"train": 396, "validation": 135, "test": 135}:
        raise ValueError("frozen split coverage changed")
    index = json.loads((Path(execution["container"]) / "index.json").read_text())
    segments = {}
    for representation in index["representations"].values():
        segment = representation["segment"]
        expected = f"sha256:{representation['segment_sha256']}"
        path = Path(execution["container"]) / segment
        if not path.is_file():
            raise ValueError(f"model segment missing: {path}")
        if verify_segments and sha256(path) != expected:
            raise ValueError(f"model segment changed: {path}")
        segments[segment] = expected
    return {
        "schema": "larql.gwstate1.preflight.v1",
        "status": "authority_audit_only_no_state1_outcomes",
        "authorities": hashes,
        "v2_lineage": lineage,
        "v2_adjudication_identity": adjudication["adjudication_sha256"],
        "execution_identity": execution["execution_sha256"],
        "plan_sha256": execution["plan_sha256"],
        "model_metadata": execution["container_metadata"],
        "model_segments": segments,
        "model_segments_verified": verify_segments,
        "matched_control_donor_indices_sha256": "sha256:" + hashlib.sha256(canonical_bytes(donors)).hexdigest(),
        "matched_control_donor_rows": len(donors),
        "split_rows": dict(sorted(counts.items())),
        "capture_artifacts_verified": len(capture["artifacts"]),
        "replay_artifacts_verified": len(replay["artifacts"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-segments", action="store_true",
                        help="hash every model segment against the container index")
    args = parser.parse_args()
    print(json.dumps(audit(verify_segments=args.verify_segments), indent=2, sort_keys=True))
