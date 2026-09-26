#!/usr/bin/env python3
"""Adjudicate the complete frozen CAR-1A compactness grid once on held-out rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gwcar1a_analysis import (
    adjudicate_split, effects_from_replay, exact_effect_from_state1, summarize,
)
from gwcar1a_candidates import validate_candidates
from gwcar1a_preregister import FROZEN_STATUS, validate as validate_protocol
from gwv2_amend import seal_json
from gwv2_io import array
from gwv2_population import sha256


def checked_replay(protocol: dict, protocol_path: Path, candidate_path: Path,
                   replay_path: Path, stage: str, rows: list[dict]) -> tuple[dict, np.ndarray, np.ndarray]:
    candidate_status = validate_candidates(protocol_path, candidate_path)
    candidates = json.loads(candidate_path.read_text())
    replay = json.loads(replay_path.read_text())
    expected_rows = [i for i, item in enumerate(rows)
                     if (item["row"]["split"] == "train" if stage == "train"
                         else item["row"]["split"] in ("validation", "test"))]
    count = 396 if stage == "train" else 270
    reference_key = "state1_train_replay" if stage == "train" else "state1_heldout_replay"
    if (candidate_status["stage"] != stage
        or replay.get("schema") != "larql.gwcar1a.replay.v1"
        or replay.get("stage") != stage
        or replay.get("protocol_sha256") != protocol["protocol_sha256"]
        or replay.get("candidate_manifest_sha256") != sha256(candidate_path)
        or replay.get("state1_reference_manifest_sha256") != protocol["authorities"][reference_key]["sha256"]
        or replay.get("execution_file_sha256") != protocol["authorities"]["v2_execution"]["sha256"]
        or replay.get("executable_sha256") != protocol["runner"]["executable_sha256"]
        or replay.get("row_indices") != expected_rows
        or candidates["row_indices"] != expected_rows
        or replay.get("candidate_ids") != list(range(19))
        or replay.get("h1_arms") != ["donor", "target_exact_natural"]
        or replay.get("intervention_firings") != count * 19 * 2
        or replay.get("identity_bit_mismatches") != 0):
        raise ValueError(f"inadmissible CAR-1A {stage} replay")
    proximal = array(replay_path.parent, replay, "proximal.f32")
    terminal = array(replay_path.parent, replay, "terminal.f32")
    if proximal.shape != (count, 19, 2, 142) or terminal.shape != proximal.shape:
        raise ValueError("incomplete CAR-1A replay tensor geometry")
    return replay, proximal, terminal


def run(protocol_path: Path, train_candidates: Path, train_replay: Path,
        heldout_candidates: Path, heldout_replay: Path, output: Path) -> dict:
    status = validate_protocol(protocol_path)
    if status["status"] != FROZEN_STATUS:
        raise ValueError("CAR-1A adjudication requires a frozen protocol")
    protocol = json.loads(protocol_path.read_text())
    rows = json.loads(Path(protocol["authorities"]["v2_execution"]["path"]).read_text())["rows"]
    train, train_proximal, train_terminal = checked_replay(
        protocol, protocol_path, train_candidates, train_replay, "train", rows)
    heldout, proximal, terminal = checked_replay(
        protocol, protocol_path, heldout_candidates, heldout_replay, "heldout", rows)
    train_subjects, train_effects = effects_from_replay(
        rows, train, train_proximal, train_terminal, "train")
    if len(train_subjects) != 44:
        raise ValueError("CAR-1A train subject coverage changed")
    train_candidates_doc = json.loads(train_candidates.read_text())
    heldout_candidates_doc = json.loads(heldout_candidates.read_text())
    if (train_candidates_doc["fit_sha256"] != heldout_candidates_doc["fit_sha256"]
        or [(entry["candidate_id"], entry["per_row_bytes"], entry["shared_model_bytes"])
            for entry in train_candidates_doc["cost"]] !=
           [(entry["candidate_id"], entry["per_row_bytes"], entry["shared_model_bytes"])
            for entry in heldout_candidates_doc["cost"]]):
        raise ValueError("CAR-1A held-out encoding or PCA fit changed")
    reference_path = Path(protocol["authorities"]["state1_heldout_replay"]["path"])
    reference = json.loads(reference_path.read_text())
    reference_proximal = array(reference_path.parent, reference, "proximal.f32")
    reference_terminal = array(reference_path.parent, reference, "terminal.f32")
    results = {}
    for split in ("validation", "test"):
        subjects, candidate_effects = effects_from_replay(
            rows, heldout, proximal, terminal, split)
        reference_subjects, exact_effects = exact_effect_from_state1(
            rows, reference, reference_proximal, reference_terminal, split)
        if len(subjects) != 15 or subjects != reference_subjects:
            raise ValueError(f"CAR-1A {split} subject pairing changed")
        results[split] = {"subjects": subjects,
                          "analysis": adjudicate_split(candidate_effects, exact_effects)}
    summary = summarize(results["validation"]["analysis"],
                        results["test"]["analysis"], heldout_candidates_doc["cost"])
    document = {
        "schema": "larql.gwcar1a.adjudication.v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "train_candidate_manifest_file_sha256": sha256(train_candidates),
        "train_replay_file_sha256": sha256(train_replay),
        "heldout_candidate_manifest_file_sha256": sha256(heldout_candidates),
        "heldout_replay_file_sha256": sha256(heldout_replay),
        "state1_heldout_replay_file_sha256": sha256(reference_path),
        "train_subjects": train_subjects,
        "train_mean_effects": train_effects.mean(axis=0).tolist(),
        "results": results, "cost": heldout_candidates_doc["cost"],
        "summary": summary,
        "construction_source_requires_natural_prefix": True,
        "efficiency_claim": False,
    }
    sealed = seal_json(output, document, "adjudication_sha256")
    return {"adjudication_sha256": sealed["adjudication_sha256"],
            "summary": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("train_candidates", type=Path)
    parser.add_argument("train_replay", type=Path)
    parser.add_argument("heldout_candidates", type=Path)
    parser.add_argument("heldout_replay", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.protocol, args.train_candidates, args.train_replay,
                         args.heldout_candidates, args.heldout_replay, args.output),
                     indent=2, sort_keys=True))
