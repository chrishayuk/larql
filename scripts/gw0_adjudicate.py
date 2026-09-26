#!/usr/bin/env python3
"""Assemble and audit the manifest-bound Gemma 3 GW-0 capture."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def content_address_copy(source: Path, root: Path, suffix: str) -> dict:
    digest = sha(source)
    name = f"sha256-{digest[7:]}.{suffix}"
    target = root / name
    if target.exists() and sha(target) != digest:
        raise ValueError(f"artifact collision: {target}")
    if not target.exists():
        shutil.copyfile(source, target)
    return {"path": name, "sha256": digest, "bytes": source.stat().st_size}


def stable_emergence(record: dict) -> tuple[int, str, str]:
    sites = record["sites"]
    for index, site in enumerate(sites):
        if all(later["after"]["rank"] <= 10 for later in sites[index:]):
            return site["layer"], site["site"], "stable_top10"
    index = max(
        range(len(sites)),
        key=lambda value: sites[value]["before"]["rank"] - sites[value]["after"]["rank"],
    )
    return sites[index]["layer"], sites[index]["site"], "largest_rank_improvement_fallback"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--walk", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.input_manifest.read_text())
    if manifest["manifest_sha256"] != "sha256:8dd49850bf1066421bedc2d9908b33e9be928e5da068b638cf08d3831bf1e53c":
        raise ValueError("unexpected frozen input manifest")
    input_rows = rows(args.input)
    walk_rows = {row["edge_id"]: row for row in rows(args.walk)}
    if len(input_rows) != 426 or len(walk_rows) != 426:
        raise ValueError("frozen denominator is not 426")
    args.output.mkdir(parents=True, exist_ok=True)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    walk_artifact = content_address_copy(args.walk, args.artifact_root, "exact-walk.jsonl")
    attempts_artifact = content_address_copy(args.attempts, args.artifact_root, "attempts.jsonl")

    census = []
    identity = None
    for index, input_row in enumerate(input_rows):
        edge_id = input_row["edge_id"]
        record_path = args.records / f"{edge_id}.json"
        if not record_path.is_file():
            raise ValueError(f"missing execution record: {edge_id}")
        record = json.loads(record_path.read_text())
        walk = walk_rows[edge_id]
        if record["edge_id"] != edge_id or record["prompt_row_index"] != index:
            raise ValueError(f"record binding mismatch: {edge_id}")
        if record["manifest_sha256"] != manifest["manifest_sha256"] or walk["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError(f"manifest binding mismatch: {edge_id}")
        if record["parity"]["result"] != "pass" or record["coverage"] != {"expected_writes": 68, "result": "complete", "writes": 68}:
            raise ValueError(f"failed execution admitted: {edge_id}")
        if record["causal_status"] != "untested" or record["causal_evidence_ids"]:
            raise ValueError(f"observational row promoted to causal: {edge_id}")
        current_identity = (
            record["identity"]["container"], record["identity"]["plan"],
            record["identity"]["tokenizer"], record["identity"]["execution_fingerprint"],
            record["identity"]["runner"],
        )
        identity = identity or current_identity
        if current_identity != identity:
            raise ValueError(f"execution identity drift: {edge_id}")
        refs = []
        for artifact in record["artifacts"]:
            path = args.artifact_root / artifact["path"]
            if sha(path) != artifact["sha256"] or path.stat().st_size != artifact["bytes"]:
                raise ValueError(f"artifact mismatch: {path}")
            ref = dict(artifact)
            ref["shape"] = [68, manifest["model"]["hidden_size"]]
            refs.append(ref)
        record_ref = content_address_copy(record_path, args.artifact_root, "execution-record.json")
        record_ref["kind"] = "execution_record"
        refs.append(record_ref)
        if index == 0:
            refs.extend([
                {"kind": "exact_walk_batch", **walk_artifact},
                {"kind": "execution_attempt_log", **attempts_artifact},
            ])
        census.append({
            "schema": "larql.gw0.census-row.v1",
            "edge_id": edge_id,
            "edge_family_id": input_row["edge_family_id"],
            "split": input_row["split"],
            "cohort": "primary",
            "semantic_edge": input_row["semantic_edge"],
            "exact_walk_result": {"hits": walk["hits"], "query": walk["query"], "top_k_per_layer": 20},
            "transition_candidate": record["transition_candidate"],
            "causal_status": "untested",
            "causal_evidence_ids": [],
            "execution_status": "complete",
            "surface_status": record["surface_status"],
            "emergence": dict(zip(("layer", "site", "method"), stable_emergence(record))),
            "target_readout": record["target"],
            "produced": record["produced"],
            "provenance": {
                "model_identity": manifest["model"]["model_identity"],
                "container_identity": record["identity"]["container"],
                "execution_fingerprint": record["identity"]["execution_fingerprint"],
                "basis_identity": f'{record["identity"]["container"]}/{record["identity"]["plan"]}/prepared-output-head-readout-v1',
                "run_record_hash": sha(record_path),
                "position": record["capture_position"],
                "input_manifest_sha256": manifest["manifest_sha256"],
                "runner_identity": record["identity"]["runner"],
                "plan_identity": record["identity"]["plan"],
            },
            "artifacts": refs,
        })

    census_path = args.output / "census.jsonl"
    census_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in census))
    site_universe = [f"{layer}:{site}" for layer in range(34) for site in ("attention", "ffn")]
    (args.output / "site-universe.json").write_text(json.dumps(site_universe, indent=2) + "\n")

    relation_emergence = defaultdict(Counter)
    candidate_sites = defaultdict(Counter)
    target_top1 = Counter()
    totals = Counter()
    for row in census:
        relation = row["semantic_edge"]["relation"]
        relation_emergence[relation][f'{row["emergence"]["layer"]}:{row["emergence"]["site"]}'] += 1
        candidate_sites[relation].update(f'{site["layer"]}:{site["site"]}' for site in row["transition_candidate"]["sites"])
        totals[relation] += 1
        target_top1[relation] += row["target_readout"]["final"]["rank"] == 1
    report = {
        "schema": "larql.gw0.preseal-audit.v1",
        "input_manifest_sha256": manifest["manifest_sha256"],
        "denominator": 426,
        "execution": {"complete": 426, "refused_rows": 0, "parity_pass": 426, "coverage_pass": 426},
        "surface_refusals": {
            "candidate_read_operators": {"rows": 426, "class": "post_attention_norm_not_supported_by_head_content_adapter"},
            "relation_readable_band": {"rows": 426, "class": "no_decoder_frozen_in_input_manifest"},
        },
        "target_top1": {relation: {"hits": target_top1[relation], "rows": totals[relation], "rate": target_top1[relation]/totals[relation]} for relation in sorted(totals)},
        "emergence_site_counts": {relation: dict(counts.most_common()) for relation, counts in sorted(relation_emergence.items())},
        "candidate_site_counts": {relation: dict(counts.most_common()) for relation, counts in sorted(candidate_sites.items())},
        "disagreement_matrix": {
            "status": "not_identifiable_from_phase-one-capture",
            "reason": "Exact WALK records ranked feature addresses but transition candidates carry sites, not feature addresses; promoted-target annotation was not materialised. Presence is not treated as agreement.",
            "semantic_positive_rows": 426,
            "exact_walk_rankings_recorded": 426,
            "transition_candidates_recorded": 426,
            "semantic_walk_miss_transition_hit_cases": [],
        },
        "causal_metrics": {"supported_rows": 0, "status": "empty_by_contract"},
    }
    (args.output / "preseal-audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"census": str(census_path), "rows": len(census), "audit": str(args.output / "preseal-audit.json")}, indent=2))


if __name__ == "__main__":
    main()
