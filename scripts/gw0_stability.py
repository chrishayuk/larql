#!/usr/bin/env python3
"""Post-seal descriptive stability report for GW-0; no fitting or thresholds."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    return {"n": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
            "p05": float(np.quantile(values, .05)), "p95": float(np.quantile(values, .95))} if values.size else {"n": 0}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 1.0


def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = read_jsonl(args.census)
    if len(rows) != 426 or manifest["census"]["rows"] != 426:
        raise ValueError("sealed denominator mismatch")

    def triple(row):
        edge = row["semantic_edge"]
        return edge["subject"], edge["relation"], edge["target"]

    exact = {row["edge_id"]: {(hit["layer"], hit["feature"]) for hit in row["exact_walk_result"]["hits"]} for row in rows}
    sites = {row["edge_id"]: {(hit["layer"], hit["site"]) for hit in row["transition_candidate"]["sites"]} for row in rows}
    by_triple = defaultdict(list); unique = {}
    for row in rows:
        by_triple[triple(row)].append(row)
        unique.setdefault(triple(row), row)

    prompt_exact = []; prompt_sites = []; prompt_emergence_delta = []
    for group in by_triple.values():
        for left, right in combinations(group, 2):
            prompt_exact.append(jaccard(exact[left["edge_id"]], exact[right["edge_id"]]))
            prompt_sites.append(jaccard(sites[left["edge_id"]], sites[right["edge_id"]]))
            prompt_emergence_delta.append(abs(left["emergence"]["layer"] - right["emergence"]["layer"]))

    within_exact = []; within_sites = []; across_exact = []; across_sites = []
    unique_rows = list(unique.values())
    for left, right in combinations(unique_rows, 2):
        same = left["semantic_edge"]["relation"] == right["semantic_edge"]["relation"]
        (within_exact if same else across_exact).append(jaccard(exact[left["edge_id"]], exact[right["edge_id"]]))
        (within_sites if same else across_sites).append(jaccard(sites[left["edge_id"]], sites[right["edge_id"]]))

    relation_layers = defaultdict(list)
    for row in rows:
        relation_layers[row["semantic_edge"]["relation"]].append(row["emergence"]["layer"])

    relation_sums = {relation: np.zeros((68, 2560), np.float64) for relation in relation_layers}
    relation_counts = defaultdict(int)
    delta_paths = {}
    for row in rows:
        artifact = next(item for item in row["artifacts"] if item.get("kind") == "delta")
        path = args.artifact_root / artifact["path"]
        delta = np.fromfile(path, dtype="<f4").reshape(68, 2560)
        relation = row["semantic_edge"]["relation"]
        relation_sums[relation] += delta
        relation_counts[relation] += 1
        delta_paths[row["edge_id"]] = path
    centroids = {relation: values / relation_counts[relation] for relation, values in relation_sums.items()}

    to_centroid = defaultdict(list)
    for row in rows:
        relation = row["semantic_edge"]["relation"]
        delta = np.fromfile(delta_paths[row["edge_id"]], dtype="<f4").reshape(68, 2560)
        for site in range(68):
            to_centroid[relation].append(cosine(delta[site], centroids[relation][site]))
    cross_centroids = []
    for left, right in combinations(sorted(centroids), 2):
        cross_centroids.extend(cosine(centroids[left][site], centroids[right][site]) for site in range(68))
    prompt_delta = []
    for group in by_triple.values():
        arrays = [np.fromfile(delta_paths[row["edge_id"]], dtype="<f4").reshape(68, 2560) for row in group]
        for left, right in combinations(arrays, 2):
            prompt_delta.extend(cosine(left[site], right[site]) for site in range(68))

    report = {
        "schema": "larql.gw0.stability-report.v1",
        "bundle_sha256": manifest["bundle_sha256"],
        "descriptive_only": True,
        "rows": len(rows),
        "feature_overlap_jaccard": {
            "same_triple_across_prompt_families": quantiles(prompt_exact),
            "different_triples_within_relation": quantiles(within_exact),
            "different_relations": quantiles(across_exact),
            "note": "Exact WALK is subject-only, so same-triple prompt overlap is structurally 1.0 rather than an execution finding.",
        },
        "transition_site_overlap_jaccard": {
            "same_triple_across_prompt_families": quantiles(prompt_sites),
            "different_triples_within_relation": quantiles(within_sites),
            "different_relations": quantiles(across_sites),
        },
        "emergence_layer": {
            "prompt_family_absolute_difference": quantiles(prompt_emergence_delta),
            "by_relation": {relation: quantiles(values) for relation, values in sorted(relation_layers.items())},
        },
        "carrier_delta_cosine": {
            "same_triple_across_prompt_families": quantiles(prompt_delta),
            "row_to_own_relation_site_centroid": {relation: quantiles(values) for relation, values in sorted(to_centroid.items())},
            "across_relation_site_centroids": quantiles(cross_centroids),
        },
        "unavailable": {
            "read_layer": "no relation decoder was frozen in the input manifest",
            "operator_identity": "Gemma 3 post-attention norm caused the head-content adapter to refuse",
            "disagreement_matrix": "exact WALK feature addresses and execution sites have no captured common edge key",
        },
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
