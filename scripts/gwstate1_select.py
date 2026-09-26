#!/usr/bin/env python3
"""Freeze and verify the two GW-STATE-1 train-only context choices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gwstate1_analysis import frozen_family, select_train, subject_effects_from_logits
from gwstate1_preregister import FROZEN_STATUS, validate as validate_protocol
from gwv2_amend import seal_json
from gwv2_io import array
from gwv2_population import canonical_hash, sha256

SCHEMA = "larql.gwstate1.selection.v1"


def expected_document(protocol_path: Path, replay_path: Path) -> dict:
    protocol_status = validate_protocol(protocol_path)
    if protocol_status["status"] != FROZEN_STATUS:
        raise ValueError("train selection requires a frozen STATE-1 protocol")
    protocol = json.loads(protocol_path.read_text())
    replay = json.loads(replay_path.read_text())
    if (replay.get("schema") != "larql.gwstate1.replay.v1"
        or replay.get("stage") != "train"
        or replay.get("protocol_sha256") != protocol["protocol_sha256"]
        or replay.get("executable_sha256") != protocol["runner"]["executable_sha256"]
        or replay.get("context_ids") != list(range(256))
        or replay.get("h1_arms") != ["donor", "target_exact_natural"]
        or replay.get("parity_bit_mismatches") != 0
        or replay.get("intervention_firings") != 396 * (3 + 256 * 2)
        or replay.get("capture_file_sha256") != protocol["authorities"]["v2_capture"]["sha256"]
        or replay.get("execution_file_sha256") != protocol["authorities"]["v2_execution"]["sha256"]
        or replay.get("v2_replay_file_sha256") != protocol["authorities"]["v2_replay"]["sha256"]):
        raise ValueError("inadmissible STATE-1 train replay")
    execution = json.loads(Path(protocol["authorities"]["v2_execution"]["path"]).read_text())
    rows = execution["rows"]
    expected_rows = [i for i, item in enumerate(rows) if item["row"]["split"] == "train"]
    if replay.get("row_indices") != expected_rows or len(expected_rows) != 396:
        raise ValueError("STATE-1 train row order changed")
    proximal = array(replay_path.parent, replay, "proximal.f32")
    terminal = array(replay_path.parent, replay, "terminal.f32")
    if proximal.shape != (396, 256, 2, 142) or terminal.shape != proximal.shape:
        raise ValueError("incomplete STATE-1 train surface")
    subjects, effects = subject_effects_from_logits(rows, proximal, terminal,
                                                     replay["context_ids"], "train", expected_rows)
    if len(subjects) != 44:
        raise ValueError("train subject coverage changed")
    norms = np.asarray(replay["natural_head_contribution_norms_train_mean"], dtype=np.float64)
    choice = select_train(effects, norms)
    selected = choice["carrier_branch_selected"]
    return {
        "schema": SCHEMA,
        "status": "frozen_before_heldout_replay",
        "protocol_sha256": protocol["protocol_sha256"],
        "train_replay_path": str(replay_path.resolve()),
        "train_replay_file_sha256": sha256(replay_path),
        "train_replay_artifact_sha256": {entry["path"]: entry["sha256"] for entry in replay["artifacts"]},
        "train_subjects": subjects,
        "carrier_branch_selected": selected,
        "context_ids": list(frozen_family(selected)),
        "eligible_contexts": choice["eligible_contexts"],
        "exact_proximal_effect": choice["exact_proximal_effect"],
        "selected_train_retention": choice["selected_train_retention"],
        "heldout_outcomes_examined": False,
    }


def freeze(protocol: Path, replay: Path, output: Path) -> dict:
    document = seal_json(output, expected_document(protocol, replay), "selection_sha256")
    return {"selection_sha256": document["selection_sha256"],
            "carrier_branch_selected": document["carrier_branch_selected"],
            "context_ids": document["context_ids"]}


def validate(protocol: Path, selection: Path) -> dict:
    document = json.loads(selection.read_text())
    if document.get("selection_sha256") != canonical_hash(document, "selection_sha256"):
        raise ValueError("STATE-1 selection seal mismatch")
    replay = Path(document["train_replay_path"])
    expected = expected_document(protocol, replay)
    if {key: value for key, value in document.items() if key != "selection_sha256"} != expected:
        raise ValueError("STATE-1 selection differs from train-only authority")
    return {"selection_sha256": document["selection_sha256"],
            "carrier_branch_selected": document["carrier_branch_selected"],
            "context_ids": document["context_ids"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "validate"])
    parser.add_argument("protocol", type=Path)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        if len(args.paths) != 2:
            parser.error("freeze requires TRAIN_REPLAY_JSON OUTPUT_SELECTION_JSON")
        result = freeze(args.protocol, args.paths[0], args.paths[1])
    else:
        if len(args.paths) != 1:
            parser.error("validate requires SELECTION_JSON")
        result = validate(args.protocol, args.paths[0])
    print(json.dumps(result, indent=2, sort_keys=True))
