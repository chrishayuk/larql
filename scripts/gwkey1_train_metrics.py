#!/usr/bin/env python3
"""Derive the frozen GW-KEY-1 train-only K/V role-subset surface."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from gwconv1_adjudicate import mean_pairwise_js, softmax
from gwsup1_preregister import canonical_hash, sha
from gwkey1_preregister import SCHEMA as PREREG_SCHEMA

SCHEMA = "larql.gwkey1.train-source-metrics.v1"
SOURCE_SCHEMA = "larql.gwkey1.source-capture.v1"
HEAD_SCHEMA = "larql.gwhead1.natural-capture.v1"
REPLAY_SCHEMA = "larql.gwkey1.train-source-replay.v1"
ROWS = 426
TRAIN_ROWS = 255
SUBSETS = 64
CANDIDATES = 126
METRICS = ("candidate_raw", "candidate_zscore")


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def descriptor(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in manifest["artifacts"] if item["path"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one artifact {name}")
    return matches[0]


def checked_memmap(
    root: Path, manifest: dict[str, Any], name: str, shape: tuple[int, ...]
) -> np.memmap:
    item = descriptor(manifest, name)
    path = root / name
    if sha(path) != item["sha256"]:
        raise ValueError(f"artifact hash mismatch: {path}")
    if path.stat().st_size != math.prod(shape) * 4:
        raise ValueError(f"artifact shape mismatch: {path}")
    return np.memmap(path, dtype="<f4", mode="r", shape=shape)


def zscore_distribution(logits: np.ndarray) -> np.ndarray:
    std = np.std(logits, axis=-1, keepdims=True)
    if np.any(std == 0) or np.any(~np.isfinite(std)):
        raise ValueError("z-scored candidate readout has zero or non-finite variance")
    return softmax((logits - np.mean(logits, axis=-1, keepdims=True)) / std)


def adjusted_candidate(
    before: np.ndarray,
    after: np.ndarray,
    fact_rows: np.ndarray,
    control_rows: np.ndarray,
) -> np.ndarray:
    fact_before = np.asarray([mean_pairwise_js(before[rows]) for rows in fact_rows])
    fact_after = np.asarray([mean_pairwise_js(after[rows]) for rows in fact_rows])
    control_before = np.asarray([mean_pairwise_js(before[rows]) for rows in control_rows])
    control_after = np.asarray([mean_pairwise_js(after[rows]) for rows in control_rows])
    return (fact_before - fact_after) - (control_before - control_after)


def build(args: argparse.Namespace) -> dict[str, Any]:
    prereg = json.loads(args.preregistration.read_text())
    candidates = json.loads(args.candidates.read_text())
    source = json.loads(args.source_capture.read_text())
    head = json.loads(args.head_capture.read_text())
    replay = json.loads(args.replay_manifest.read_text())
    if prereg.get("schema") != PREREG_SCHEMA or prereg.get("status") != "frozen_pre_execution":
        raise ValueError("not a frozen GW-KEY-1 preregistration")
    if canonical_hash(prereg, "preregistration_sha256") != prereg["preregistration_sha256"]:
        raise ValueError("GW-KEY-1 preregistration identity mismatch")
    if source.get("schema") != SOURCE_SCHEMA or source.get("status") != "natural_source_capture_complete_pre_search":
        raise ValueError("not a complete GW-KEY-1 natural source capture")
    if head.get("schema") != HEAD_SCHEMA or head.get("status") != "natural_capture_complete_pre_subset_search":
        raise ValueError("not the sealed GW-HEAD-1 natural capture")
    if replay.get("schema") != REPLAY_SCHEMA or replay.get("status") != "complete_train_only":
        raise ValueError("not a complete train-only GW-KEY-1 replay")
    if replay.get("split") != "train" or replay.get("held_out_rows") != 0 or replay.get("validation_or_test_executed") is not False:
        raise ValueError("GW-KEY-1 train replay contains held-out execution")
    if replay.get("full_replay_parity") != {"candidate_logit_bit_mismatches": 0}:
        raise ValueError("GW-KEY-1 full replay parity failed")
    if source["preregistration_sha256"] != prereg["preregistration_sha256"] or replay["preregistration_sha256"] != prereg["preregistration_sha256"]:
        raise ValueError("GW-KEY-1 artifacts bind a different preregistration")
    if replay["natural_source_capture_sha256"] != sha(args.source_capture) or replay["gwhead1_natural_capture_sha256"] != sha(args.head_capture):
        raise ValueError("GW-KEY-1 replay binds different captures")
    if candidates["candidate_identity_sha256"] != prereg["authorities"]["candidate_identity"]:
        raise ValueError("candidate identity changed")

    replay_root = args.replay_manifest.parent
    head_root = args.head_capture.parent
    train_rows_path = replay_root / "train-rows.jsonl"
    natural_rows_path = head_root / "natural-rows.jsonl"
    if sha(train_rows_path) != descriptor(replay, "train-rows.jsonl")["sha256"] or sha(natural_rows_path) != descriptor(head, "natural-rows.jsonl")["sha256"]:
        raise ValueError("GW-KEY-1 row artifact hash mismatch")
    train_rows = jsonl(train_rows_path)
    natural_rows = jsonl(natural_rows_path)
    if len(train_rows) != TRAIN_ROWS or len(natural_rows) != ROWS:
        raise ValueError("GW-KEY-1 row count changed")
    for train_index, row in enumerate(train_rows):
        original = int(row["original_row"])
        if row["train_row"] != train_index or natural_rows[original]["edge_id"] != row["edge_id"] or natural_rows[original]["split"] != "train":
            raise ValueError("GW-KEY-1 train row order changed")

    original_rows = np.asarray([int(row["original_row"]) for row in train_rows])
    before_logits_all = checked_memmap(
        head_root, head, "candidate-before-logits.f32", (ROWS, CANDIDATES)
    )
    joint_logits = checked_memmap(
        replay_root,
        replay,
        "train-joint-logits.f32",
        (SUBSETS, SUBSETS, TRAIN_ROWS, CANDIDATES),
    )
    before_logits = np.asarray(before_logits_all[original_rows], dtype=float)
    before_distributions = {
        "candidate_raw": softmax(before_logits),
        "candidate_zscore": zscore_distribution(before_logits),
    }

    edge_lookup = {row["edge_id"]: index for index, row in enumerate(train_rows)}
    if len(edge_lookup) != TRAIN_ROWS:
        raise ValueError("duplicate GW-KEY-1 train row")
    controls = {
        item["edge_id"]: item["control_edge_id"]
        for item in candidates["different_destination_controls"]
    }
    groups = [
        group
        for group in candidates["prompt_family_groups"]
        if group["edge_ids"][0] in edge_lookup
    ]
    if len(groups) != 85:
        raise ValueError("GW-KEY-1 train semantic-edge group count changed")
    fact_rows = np.asarray(
        [[edge_lookup[edge] for edge in group["edge_ids"]] for group in groups]
    )
    control_rows = np.asarray(
        [[edge_lookup[controls[edge]] for edge in group["edge_ids"]] for group in groups]
    )
    relations = np.asarray([group["relation"] for group in groups])

    pairs: list[dict[str, Any]] = []
    for key_mask in range(SUBSETS):
        for value_mask in range(SUBSETS):
            logits = np.asarray(joint_logits[key_mask, value_mask], dtype=float)
            after = {
                "candidate_raw": softmax(logits),
                "candidate_zscore": zscore_distribution(logits),
            }
            per_edge = {
                name: adjusted_candidate(
                    before_distributions[name], distribution, fact_rows, control_rows
                )
                for name, distribution in after.items()
            }
            if any(np.any(~np.isfinite(values)) for values in per_edge.values()):
                raise ValueError(f"K={key_mask} V={value_mask} produced non-finite metrics")
            pairs.append(
                {
                    "K_mask": key_mask,
                    "V_mask": value_mask,
                    "E": {name: float(np.mean(values)) for name, values in per_edge.items()},
                    "E_by_relation": {
                        relation: {
                            name: float(np.mean(values[relations == relation]))
                            for name, values in per_edge.items()
                        }
                        for relation in ("capital", "currency", "language", "hypernym")
                    },
                }
            )
            pair = key_mask * SUBSETS + value_mask + 1
            if pair % 64 == 0 or pair == SUBSETS * SUBSETS:
                print(f"GW-KEY-1 train metrics {pair}/{SUBSETS * SUBSETS}", flush=True)

    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_train_only",
        "artifact_sha256": None,
        "split": "train",
        "held_out_rows": 0,
        "semantic_edges": len(groups),
        "preregistration_sha256": prereg["preregistration_sha256"],
        "roles": prereg["roles"]["ordered_vocabulary"],
        "authorities": {
            "preregistration": {"path": str(args.preregistration), "sha256": sha(args.preregistration)},
            "candidates": {"path": str(args.candidates), "sha256": sha(args.candidates)},
            "source_capture": {"path": str(args.source_capture), "sha256": sha(args.source_capture)},
            "head_capture": {"path": str(args.head_capture), "sha256": sha(args.head_capture)},
            "train_replay": {"path": str(args.replay_manifest), "sha256": sha(args.replay_manifest)},
        },
        "measurements": {
            "candidate_raw": "GW-CONV-1 restricted-softmax JS before minus intervened-after, same fact minus matched control",
            "candidate_zscore": "same after independently z-scoring each 126-logit row",
        },
        "order": "K mask 0..63, then V mask 0..63",
        "replacement_norm": {
            "K_by_mask": replay["reference"]["K_by_mask"],
            "V_by_mask": replay["reference"]["V_by_mask"],
            "joint": "K_by_mask[K] + V_by_mask[V]",
        },
        "pairs": pairs,
    }
    document["artifact_sha256"] = canonical_hash(document, "artifact_sha256")
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--head-capture", type=Path, required=True)
    parser.add_argument("--replay-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
