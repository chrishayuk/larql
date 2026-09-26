#!/usr/bin/env python3
"""Adjudicate only frozen GW-STATE-1 contexts on both held-out splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gwstate1_analysis import adjudicate_split, subject_effects_from_logits, verdict
from gwstate1_preregister import FROZEN_STATUS, validate as validate_protocol
from gwstate1_select import validate as validate_selection
from gwv2_amend import seal_json
from gwv2_io import array
from gwv2_population import sha256


def run(protocol_path: Path, selection_path: Path, replay_path: Path, output: Path) -> dict:
    status = validate_protocol(protocol_path)
    if status["status"] != FROZEN_STATUS:
        raise ValueError("STATE-1 held-out adjudication requires a frozen protocol")
    selection_status = validate_selection(protocol_path, selection_path)
    protocol = json.loads(protocol_path.read_text())
    selection = json.loads(selection_path.read_text())
    replay = json.loads(replay_path.read_text())
    contexts = selection["context_ids"]
    rows = json.loads(Path(protocol["authorities"]["v2_execution"]["path"]).read_text())["rows"]
    expected_rows = [index for index, item in enumerate(rows)
                     if item["row"]["split"] in ("validation", "test")]
    if (replay.get("schema") != "larql.gwstate1.replay.v1"
        or replay.get("stage") != "heldout"
        or replay.get("protocol_sha256") != protocol["protocol_sha256"]
        or replay.get("selection_sha256") != selection_status["selection_sha256"]
        or replay.get("executable_sha256") != protocol["runner"]["executable_sha256"]
        or replay.get("execution_file_sha256") != protocol["authorities"]["v2_execution"]["sha256"]
        or replay.get("capture_file_sha256") != protocol["authorities"]["v2_capture"]["sha256"]
        or replay.get("v2_replay_file_sha256") != protocol["authorities"]["v2_replay"]["sha256"]
        or replay.get("context_ids") != contexts
        or replay.get("h1_arms") != ["donor", "target_exact_natural"]
        or replay.get("row_indices") != expected_rows
        or len(expected_rows) != 270
        or replay.get("intervention_firings") != 270 * (3 + 2 * len(contexts))
        or replay.get("parity_bit_mismatches") != 0):
        raise ValueError("inadmissible STATE-1 held-out replay")
    proximal = array(replay_path.parent, replay, "proximal.f32")
    terminal = array(replay_path.parent, replay, "terminal.f32")
    if proximal.shape != (270, len(contexts), 2, 142) or terminal.shape != proximal.shape:
        raise ValueError("incomplete STATE-1 held-out tensor geometry")
    results = {}
    for split in ("validation", "test"):
        subjects, effects = subject_effects_from_logits(rows, proximal, terminal,
                                                         contexts, split, expected_rows)
        if len(subjects) != 15:
            raise ValueError(f"{split} subject coverage changed")
        results[split] = {"subjects": subjects,
                          "analysis": adjudicate_split(effects,
                                                         selection["carrier_branch_selected"],
                                                         context_ids=contexts)}
    decision = verdict(results["validation"]["analysis"],
                       results["test"]["analysis"],
                       selection["carrier_branch_selected"])
    document = {
        "schema": "larql.gwstate1.adjudication.v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "selection_sha256": selection_status["selection_sha256"],
        "heldout_replay_file_sha256": sha256(replay_path),
        "heldout_replay_artifact_sha256": {item["path"]: item["sha256"] for item in replay["artifacts"]},
        "context_ids": contexts,
        "carrier_branch_selected": selection["carrier_branch_selected"],
        "results": results,
        "verdict": decision,
        "predictor_selected": None,
        "efficiency_claim": False,
    }
    sealed = seal_json(output, document, "adjudication_sha256")
    return {"adjudication_sha256": sealed["adjudication_sha256"], "verdict": decision}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("heldout_replay", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.protocol, args.selection, args.heldout_replay, args.output),
                     indent=2, sort_keys=True))
