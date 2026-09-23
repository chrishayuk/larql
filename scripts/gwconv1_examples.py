#!/usr/bin/env python3
"""Inspect deterministic representative GW-CONV-1 examples after adjudication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gwconv1_adjudicate import mean_pairwise_js, sha, softmax
from gwconv1_preregister import canonical_hash


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def top_candidates(
    probability: np.ndarray, vocabulary: list[dict[str, Any]], width: int = 5
) -> list[dict[str, Any]]:
    order = np.argsort(-probability)[:width]
    return [
        {
            "rank": rank + 1,
            "destination": vocabulary[index]["destination"],
            "token_id": vocabulary[index]["token_id"],
            "probability": float(probability[index]),
        }
        for rank, index in enumerate(order)
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    report = json.loads(args.adjudication.read_text())
    if report["status"] != "adjudicated_before_example_inspection":
        raise ValueError("examples require a completed aggregate adjudication")
    if canonical_hash(report, "adjudication_sha256") != report["adjudication_sha256"]:
        raise ValueError("adjudication identity mismatch")
    candidates = json.loads(args.candidates.read_text())
    before_manifest = json.loads(args.before_readout_manifest.read_text())
    after_manifest = json.loads(args.after_readout_manifest.read_text())
    before_root = args.before_readout_manifest.parent
    after_root = args.after_readout_manifest.parent
    for root, manifest in ((before_root, before_manifest), (after_root, after_manifest)):
        if sha(root / manifest["artifacts"]["logits"]["path"]) != manifest["artifacts"]["logits"]["sha256"]:
            raise ValueError("candidate-logit artifact hash mismatch")
    shape = tuple(before_manifest["shape"])
    before = np.memmap(
        before_root / before_manifest["artifacts"]["logits"]["path"],
        dtype="<f4",
        mode="r",
        shape=shape,
    )
    after = np.memmap(
        after_root / after_manifest["artifacts"]["logits"]["path"],
        dtype="<f4",
        mode="r",
        shape=shape,
    )
    rows = jsonl(after_root / after_manifest["artifacts"]["rows"]["path"])
    edge_lookup = {row["edge_id"]: index for index, row in enumerate(rows)}
    metrics = jsonl(args.adjudication.parent / report["artifacts"]["edge_metrics"]["path"])
    site = int(report["selection"]["global_site"]["index"])
    examples = []
    for relation in ("capital", "currency", "language", "hypernym"):
        eligible = [row for row in metrics if row["split"] == "test" and row["relation"] == relation]
        values = np.asarray(
            [row["global_site_adjusted_gain"]["candidate_raw"] for row in eligible]
        )
        median = float(np.median(values))
        selected = min(
            eligible,
            key=lambda row: (
                abs(row["global_site_adjusted_gain"]["candidate_raw"] - median),
                row["subject"],
                row["target"],
            ),
        )
        indices = np.asarray([edge_lookup[edge_id] for edge_id in selected["edge_ids"]])
        before_probability = softmax(np.asarray(before[indices, site, :], dtype=float))
        after_probability = softmax(np.asarray(after[indices, site, :], dtype=float))
        prompt_rows = []
        for local, row_index in enumerate(indices):
            prompt_rows.append(
                {
                    "edge_id": rows[row_index]["edge_id"],
                    "prompt_semantic_family": rows[row_index]["semantic_edge"]["prompt_semantic_family"],
                    "before_top5": top_candidates(before_probability[local], candidates["decoder_vocabulary"]),
                    "after_top5": top_candidates(after_probability[local], candidates["decoder_vocabulary"]),
                }
            )
        examples.append(
            {
                "selection": "test edge closest to within-relation median adjusted raw gain",
                "subject": selected["subject"],
                "relation": relation,
                "target": selected["target"],
                "adjusted_candidate_raw_gain": selected["global_site_adjusted_gain"]["candidate_raw"],
                "same_fact_js_before": float(mean_pairwise_js(before_probability)),
                "same_fact_js_after": float(mean_pairwise_js(after_probability)),
                "prompt_rows": prompt_rows,
            }
        )
    result = {
        "schema": "larql.gwconv1.posthoc-examples.v1",
        "status": "descriptive_after_aggregate_adjudication",
        "claim_boundary": "representative examples selected after adjudication; no gate or site selection uses them",
        "authorities": {
            "adjudication_identity_sha256": report["adjudication_sha256"],
            "adjudication_file_sha256": sha(args.adjudication),
            "candidate_identity_sha256": candidates["candidate_identity_sha256"],
            "before_logits_sha256": before_manifest["artifacts"]["logits"]["sha256"],
            "after_logits_sha256": after_manifest["artifacts"]["logits"]["sha256"],
        },
        "site": report["selection"]["global_site"],
        "examples": examples,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--before-readout-manifest", type=Path, required=True)
    parser.add_argument("--after-readout-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
