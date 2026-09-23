#!/usr/bin/env python3
"""Audit GW-0B and emit the first address-justified disagreement matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        index = fraction * (len(ordered) - 1)
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        weight = index - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "n": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": at(0.5),
        "p05": at(0.05),
        "p95": at(0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--promotions", type=Path, required=True)
    parser.add_argument("--attributions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest["schema"] != "larql.gw.phase1.manifest.v1":
        raise ValueError("not a sealed phase-one manifest")
    root = args.manifest.parent
    census_path = root / manifest["census"]["path"]
    if sha(census_path) != manifest["census"]["sha256"]:
        raise ValueError("sealed census hash mismatch")
    census = jsonl(census_path)
    bundle = manifest["bundle_sha256"]

    promotions: dict[tuple[int, int], set[int]] = {}
    for row in jsonl(args.promotions):
        if row["bundle_sha256"] != bundle:
            raise ValueError("promotion bundle binding mismatch")
        address = (row["address"]["layer"], row["address"]["feature"])
        if address in promotions:
            raise ValueError(f"duplicate promotion address {address}")
        promotions[address] = {candidate["token_id"] for candidate in row["candidates"]}

    attribution: dict[tuple[str, int], dict] = {}
    attribution_artifacts = []
    for path in sorted(args.attributions.glob("layer-*.jsonl")):
        path_rows = jsonl(path)
        attribution_artifacts.append(
            {"path": path.name, "sha256": sha(path), "rows": len(path_rows)}
        )
        for row in path_rows:
            if row["bundle_sha256"] != bundle:
                raise ValueError("attribution bundle binding mismatch")
            key = (row["edge_id"], row["site"]["layer"])
            if key in attribution:
                raise ValueError(f"duplicate attribution site {key}")
            attribution[key] = row

    expected = {
        (row["edge_id"], site["layer"])
        for row in census
        for site in row["transition_candidate"]["sites"]
        if site["site"] == "ffn"
    }
    missing = sorted(expected - set(attribution))
    extra = sorted(set(attribution) - expected)
    if missing or extra:
        raise ValueError(
            f"attribution census mismatch: missing={len(missing)} extra={len(extra)}"
        )

    refused = [row for row in attribution.values() if row["status"] != "reconstructed"]
    required_addresses = {
        (hit["layer"], hit["feature"])
        for row in census
        for hit in row["exact_walk_result"]["hits"]
    }
    required_addresses |= {
        (item["layer"], item["feature"])
        for row in attribution.values()
        if row["status"] == "reconstructed"
        for item in row["contributions"]
    }
    missing_promotions = sorted(required_addresses - set(promotions))
    if missing_promotions:
        raise ValueError(f"missing {len(missing_promotions)} feature promotions")

    triad = Counter()
    by_relation: dict[str, Counter] = defaultdict(Counter)
    semantic_walk_counts = Counter()
    semantic_walk_by_relation: dict[str, Counter] = defaultdict(Counter)
    semantic_walk_by_triple: dict[tuple[str, str, str], bool] = {}
    site_overlap = []
    execution_recall = []
    walk_recall = []
    row_records = []
    for row in census:
        edge_id = row["edge_id"]
        relation = row["semantic_edge"]["relation"]
        target = row["semantic_edge"]["target_token_ids"][0]
        walk = {(hit["layer"], hit["feature"]) for hit in row["exact_walk_result"]["hits"]}
        semantic_walk = any(target in promotions[address] for address in walk)
        semantic_walk_counts["hit" if semantic_walk else "miss"] += 1
        semantic_walk_by_relation[relation]["hit" if semantic_walk else "miss"] += 1
        triple = (
            row["semantic_edge"]["subject"],
            relation,
            row["semantic_edge"]["target"],
        )
        previous = semantic_walk_by_triple.setdefault(triple, semantic_walk)
        if previous != semantic_walk:
            raise ValueError(f"subject-only WALK changed across prompt families: {triple}")
        execution = set()
        overlaps = set()
        eligible_sites = 0
        reconstructed_sites = 0
        for site in row["transition_candidate"]["sites"]:
            if site["site"] != "ffn":
                continue
            eligible_sites += 1
            result = attribution[(edge_id, site["layer"])]
            if result["status"] != "reconstructed":
                continue
            reconstructed_sites += 1
            addresses = {
                (item["layer"], item["feature"])
                for item in result["contributions"]
            }
            execution |= addresses
            local_walk = {address for address in walk if address[0] == site["layer"]}
            intersection = local_walk & addresses
            overlaps |= intersection
            union = local_walk | addresses
            site_overlap.append(len(intersection) / len(union) if union else 1.0)
            execution_recall.append(len(intersection) / len(addresses) if addresses else 0.0)
            walk_recall.append(len(intersection) / len(local_walk) if local_walk else 0.0)
        eligible = eligible_sites > 0
        semantic_execution = (
            any(target in promotions[address] for address in execution) if eligible else None
        )
        walk_execution = bool(overlaps) if eligible else None
        if eligible:
            key = (
                f"semantic_walk_{'hit' if semantic_walk else 'miss'}|"
                f"semantic_execution_{'hit' if semantic_execution else 'miss'}|"
                f"walk_execution_{'overlap' if walk_execution else 'disjoint'}"
            )
            triad[key] += 1
            by_relation[relation][key] += 1
        row_records.append(
            {
                "edge_id": edge_id,
                "relation": relation,
                "target_token_id": target,
                "semantic_walk_hit": semantic_walk,
                "semantic_execution_hit": semantic_execution,
                "walk_execution_overlap": walk_execution,
                "triad_status": "eligible" if eligible else "ineligible_no_ffn_candidate",
                "walk_features": len(walk),
                "execution_features": len(execution),
                "overlap_features": len(overlaps),
                "eligible_ffn_sites": eligible_sites,
                "reconstructed_ffn_sites": reconstructed_sites,
            }
        )

    proof_rel = [
        row["proof"]["relative_l2_error"]
        for row in attribution.values()
        if row["status"] == "reconstructed"
    ]
    proof_abs = [
        row["proof"]["max_abs_error"]
        for row in attribution.values()
        if row["status"] == "reconstructed"
    ]
    proof_linf = [
        row["proof"]["relative_linf_error"]
        for row in attribution.values()
        if row["status"] == "reconstructed"
    ]
    report = {
        "schema": "larql.gw0b.reconciliation-report.v1",
        "bundle_sha256": bundle,
        "claim_boundary": (
            "observational FFN attribution supplies a checked physical join key; "
            "it is not causal evidence and does not identify a feature as an edge"
        ),
        "input_artifacts": {
            "sealed_gw0_bundle_sha256": bundle,
            "promotions": {"path": str(args.promotions), "sha256": sha(args.promotions)},
            "attributions": attribution_artifacts,
        },
        "denominators": {
            "semantic_rows": len(census),
            "triad_eligible_rows": sum(triad.values()),
            "triad_ineligible_attention_only_rows": len(census) - sum(triad.values()),
            "eligible_ffn_candidate_sites": len(expected),
            "reconstructed_ffn_candidate_sites": len(expected) - len(refused),
            "refused_ffn_candidate_sites": len(refused),
            "promoted_feature_addresses": len(promotions),
            "execution_contribution_top_k": 100,
            "promotion_top_k": 8,
        },
        "reconstruction_proof": {
            "relative_l2_error": quantiles(proof_rel),
            "relative_linf_error": quantiles(proof_linf),
            "max_abs_error": quantiles(proof_abs),
            "refusals": [
                {
                    "edge_id": row["edge_id"],
                    "layer": row["site"]["layer"],
                    "reason": row["refusal"],
                }
                for row in refused
            ],
        },
        "semantic_walk_promoted_target_top8": {
            "all_rows": dict(sorted(semantic_walk_counts.items())),
            "unique_semantic_edges": dict(
                sorted(Counter("hit" if hit else "miss" for hit in semantic_walk_by_triple.values()).items())
            ),
            "by_relation": {
                relation: dict(sorted(counts.items()))
                for relation, counts in sorted(semantic_walk_by_relation.items())
            },
        },
        "disagreement_matrix": dict(sorted(triad.items())),
        "disagreement_matrix_by_relation": {
            relation: dict(sorted(counts.items()))
            for relation, counts in sorted(by_relation.items())
        },
        "walk_execution_address_overlap": {
            "jaccard_per_eligible_site": quantiles(site_overlap),
            "execution_top100_recalled_by_exact_walk_top20": quantiles(execution_recall),
            "exact_walk_top20_present_in_execution_top100": quantiles(walk_recall),
        },
        "rows": row_records,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
