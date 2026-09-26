#!/usr/bin/env python3
"""Derive the frozen GW-HEAD-1 train-only subset metric surface."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from gwconv1_adjudicate import mean_pairwise_js, softmax
from gwhead1_preregister import EXPECTED_HEADS, SCHEMA as PREREG_SCHEMA
from gwsup1_preregister import canonical_hash, sha

SCHEMA = "larql.gwhead1.train-subset-metrics.v1"
CAPTURE_SCHEMA = "larql.gwhead1.natural-capture.v1"
REPLAY_SCHEMA = "larql.gwhead1.train-subset-replay.v1"
RELATIONS = ("capital", "currency", "language", "hypernym")
SCOPES = ("global", *RELATIONS)
ROWS = 426
TRAIN_ROWS = 255
SUBSETS = 256
CANDIDATES = 126
HIDDEN = 2560


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


def pairwise_cosine(states: np.ndarray, groups: np.ndarray) -> np.ndarray:
    normalized = states / np.linalg.norm(states, axis=1, keepdims=True)
    if np.any(~np.isfinite(normalized)):
        raise ValueError("carrier cosine has a zero or non-finite norm")
    selected = normalized[groups]
    return (
        np.sum(selected[:, 0] * selected[:, 1], axis=1)
        + np.sum(selected[:, 0] * selected[:, 2], axis=1)
        + np.sum(selected[:, 1] * selected[:, 2], axis=1)
    ) / 3.0


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
    capture = json.loads(args.capture_manifest.read_text())
    replay = json.loads(args.replay_manifest.read_text())
    if prereg.get("schema") != PREREG_SCHEMA or prereg.get("status") != "frozen_pre_execution":
        raise ValueError("not a frozen GW-HEAD-1 preregistration")
    if canonical_hash(prereg, "preregistration_sha256") != prereg["preregistration_sha256"]:
        raise ValueError("GW-HEAD-1 preregistration identity mismatch")
    if capture.get("schema") != CAPTURE_SCHEMA or capture.get("status") != "natural_capture_complete_pre_subset_search":
        raise ValueError("not a complete GW-HEAD-1 natural capture")
    if replay.get("schema") != REPLAY_SCHEMA or replay.get("status") != "complete_train_only":
        raise ValueError("not a complete train-only GW-HEAD-1 replay")
    if capture["preregistration_sha256"] != prereg["preregistration_sha256"] or replay[
        "preregistration_sha256"
    ] != prereg["preregistration_sha256"]:
        raise ValueError("GW-HEAD-1 artifact binds a different preregistration")
    if replay.get("split") != "train" or replay.get("held_out_rows") != 0 or replay.get(
        "validation_or_test_executed"
    ) is not False:
        raise ValueError("GW-HEAD-1 train replay contains held-out execution")
    if replay["full_replay_parity"] != {
        "candidate_logit_bit_mismatches": 0,
        "carrier_bit_mismatches": 0,
    }:
        raise ValueError("GW-HEAD-1 full replay parity failed")
    if replay["head_ids"] != list(range(EXPECTED_HEADS)):
        raise ValueError("GW-HEAD-1 replay changed the head universe")
    if replay["natural_capture_sha256"] != sha(args.capture_manifest):
        raise ValueError("GW-HEAD-1 replay binds a different natural capture")
    if candidates["candidate_identity_sha256"] != capture["candidate_identity_sha256"]:
        raise ValueError("candidate identity changed")

    capture_root = args.capture_manifest.parent
    replay_root = args.replay_manifest.parent
    natural_rows_path = capture_root / "natural-rows.jsonl"
    train_rows_path = replay_root / "train-rows.jsonl"
    if sha(natural_rows_path) != descriptor(capture, "natural-rows.jsonl")["sha256"]:
        raise ValueError("natural rows hash mismatch")
    if sha(train_rows_path) != descriptor(replay, "train-rows.jsonl")["sha256"]:
        raise ValueError("train rows hash mismatch")
    natural_rows = jsonl(natural_rows_path)
    train_rows = jsonl(train_rows_path)
    if len(natural_rows) != ROWS or len(train_rows) != TRAIN_ROWS:
        raise ValueError("GW-HEAD-1 row count changed")
    for train_index, row in enumerate(train_rows):
        original = int(row["original_row"])
        if row["train_row"] != train_index or natural_rows[original]["edge_id"] != row["edge_id"]:
            raise ValueError("train replay row order differs from natural capture")
        if natural_rows[original]["split"] != "train":
            raise ValueError("non-train row entered train replay")

    original_rows = np.asarray([int(row["original_row"]) for row in train_rows])
    before_logits_all = checked_memmap(
        capture_root, capture, "candidate-before-logits.f32", (ROWS, CANDIDATES)
    )
    before_carriers_all = checked_memmap(
        capture_root, capture, "carrier-before.f32", (ROWS, HIDDEN)
    )
    subset_logits = checked_memmap(
        replay_root,
        replay,
        "train-subset-logits.f32",
        (SUBSETS, TRAIN_ROWS, CANDIDATES),
    )
    subset_carriers = checked_memmap(
        replay_root,
        replay,
        "train-subset-carriers.f32",
        (SUBSETS, TRAIN_ROWS, HIDDEN),
    )
    before_logits = np.asarray(before_logits_all[original_rows], dtype=float)
    before_carriers = np.asarray(before_carriers_all[original_rows], dtype=float)

    edge_lookup = {row["edge_id"]: index for index, row in enumerate(train_rows)}
    if len(edge_lookup) != TRAIN_ROWS:
        raise ValueError("duplicate train execution row")
    controls = {
        item["edge_id"]: item["control_edge_id"]
        for item in candidates["different_destination_controls"]
    }
    groups = [group for group in candidates["prompt_family_groups"] if group["edge_ids"][0] in edge_lookup]
    if len(groups) != 85:
        raise ValueError("train semantic-edge group count changed")
    fact_rows = np.asarray([[edge_lookup[edge] for edge in group["edge_ids"]] for group in groups])
    control_rows = np.asarray(
        [[edge_lookup[controls[edge]] for edge in group["edge_ids"]] for group in groups]
    )
    relations = np.asarray([group["relation"] for group in groups])
    if any(any(train_rows[index]["relation"] != relation for index in rows) for rows, relation in zip(fact_rows, relations)):
        raise ValueError("train relation groups changed")

    before_distributions = {
        "candidate_raw": softmax(before_logits),
        "candidate_zscore": zscore_distribution(before_logits),
    }
    fact_carrier_before = pairwise_cosine(before_carriers, fact_rows)
    control_carrier_before = pairwise_cosine(before_carriers, control_rows)
    natural_norms = np.asarray(
        [row["natural_contribution_norms"] for row in train_rows], dtype=float
    )
    if natural_norms.shape != (TRAIN_ROWS, EXPECTED_HEADS) or np.any(
        ~np.isfinite(natural_norms)
    ):
        raise ValueError("natural contribution norms changed")

    per_subset: list[dict[str, np.ndarray]] = []
    for mask in range(SUBSETS):
        logits = np.asarray(subset_logits[mask], dtype=float)
        after_distributions = {
            "candidate_raw": softmax(logits),
            "candidate_zscore": zscore_distribution(logits),
        }
        values = {
            name: adjusted_candidate(before, after_distributions[name], fact_rows, control_rows)
            for name, before in before_distributions.items()
        }
        carriers = np.asarray(subset_carriers[mask], dtype=float)
        fact_after = pairwise_cosine(carriers, fact_rows)
        control_after = pairwise_cosine(carriers, control_rows)
        values["carrier"] = (fact_after - fact_carrier_before) - (
            control_after - control_carrier_before
        )
        if any(np.any(~np.isfinite(value)) for value in values.values()):
            raise ValueError(f"subset {mask} produced non-finite metrics")
        per_subset.append(values)
        if (mask + 1) % 32 == 0 or mask + 1 == SUBSETS:
            print(f"GW-HEAD-1 train metrics {mask + 1}/{SUBSETS}", flush=True)

    scopes: dict[str, Any] = {}
    for scope in SCOPES:
        group_mask = np.ones(len(groups), dtype=bool) if scope == "global" else relations == scope
        row_mask = np.ones(TRAIN_ROWS, dtype=bool) if scope == "global" else np.asarray(
            [row["relation"] == scope for row in train_rows]
        )
        subset_rows = []
        for mask, values in enumerate(per_subset):
            heads = [head for head in range(EXPECTED_HEADS) if mask & (1 << head)]
            contribution_norm = (
                0.0
                if not heads
                else float(np.mean(np.sum(natural_norms[row_mask][:, heads], axis=1)))
            )
            subset_rows.append(
                {
                    "mask": mask,
                    "heads": heads,
                    "E": {name: float(np.mean(value[group_mask])) for name, value in values.items()},
                    "total_natural_contribution_norm": contribution_norm,
                }
            )
        scopes[scope] = {
            "semantic_edges": int(np.sum(group_mask)),
            "execution_rows": int(np.sum(row_mask)),
            "E_full": subset_rows[-1]["E"],
            "E_ref": subset_rows[0]["E"],
            "subsets": subset_rows,
        }

    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_train_only",
        "artifact_sha256": None,
        "split": "train",
        "held_out_rows": 0,
        "preregistration_sha256": prereg["preregistration_sha256"],
        "head_ids": list(range(EXPECTED_HEADS)),
        "authorities": {
            "preregistration": {"path": str(args.preregistration), "sha256": sha(args.preregistration)},
            "candidates": {"path": str(args.candidates), "sha256": sha(args.candidates)},
            "natural_capture": {"path": str(args.capture_manifest), "sha256": sha(args.capture_manifest)},
            "train_replay": {"path": str(args.replay_manifest), "sha256": sha(args.replay_manifest)},
        },
        "measurements": {
            "candidate_raw": "GW-CONV-1 restricted-softmax JS before minus intervened-after, same fact minus matched control",
            "candidate_zscore": "same after independently z-scoring each 126-logit row",
            "carrier": "mean pairwise cosine after minus before, same fact minus matched control",
            "total_natural_contribution_norm": "scope mean across prompt rows of the sum of natural effective-W_O contribution L2 norms for heads in S",
        },
        "order": "subset mask 0..255; bit h set means head h natural/restored",
        "scopes": scopes,
    }
    document["artifact_sha256"] = canonical_hash(document, "artifact_sha256")
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--replay-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
