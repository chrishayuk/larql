#!/usr/bin/env python3
"""Read-only outcome-invariance witness for the historical STATE-1 erratum.

The historical protocol, replay, selection, and adjudication remain untouched.
This script recomputes both held-out analyses from the replay tensors and checks
that the disputed broad_context fallback is unreachable for the sealed verdict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gwstate1_analysis import adjudicate_split, subject_effects_from_logits, verdict
from gwstate1_preregister import FROZEN_STATUS, validate as validate_protocol
from gwstate1_select import validate as validate_selection
from gwv2_io import array
from gwv2_population import canonical_hash, sha256

ROOT = Path(__file__).resolve().parents[1]
CASE = "gwstate1-gemma3-4b-it-phase1"
PROTOCOL = ROOT / "bench/gw-state-1/gemma3-4b-it-phase1/gwstate1-protocol.json"
SELECTION = ROOT / "output" / f"{CASE}-selection.json"
REPLAY = ROOT / "output" / f"{CASE}-heldout/replay.json"
ADJUDICATION = ROOT / "output" / f"{CASE}-adjudication.json"


def witness() -> dict:
    protocol_status = validate_protocol(PROTOCOL)
    if protocol_status["status"] != FROZEN_STATUS:
        raise ValueError("STATE-1 protocol is not frozen")
    selection_status = validate_selection(PROTOCOL, SELECTION)
    protocol = json.loads(PROTOCOL.read_text())
    selection = json.loads(SELECTION.read_text())
    replay = json.loads(REPLAY.read_text())
    adjudication = json.loads(ADJUDICATION.read_text())
    if (adjudication.get("schema") != "larql.gwstate1.adjudication.v1"
        or adjudication.get("adjudication_sha256") != canonical_hash(adjudication, "adjudication_sha256")
        or adjudication.get("protocol_sha256") != protocol_status["protocol_sha256"]
        or adjudication.get("selection_sha256") != selection_status["selection_sha256"]
        or adjudication.get("heldout_replay_file_sha256") != sha256(REPLAY)
        or adjudication.get("context_ids") != selection["context_ids"]
        or adjudication.get("carrier_branch_selected") != selection["carrier_branch_selected"]):
        raise ValueError("historical adjudication identity or lineage mismatch")
    contexts = selection["context_ids"]
    expected_rows = json.loads(
        Path(protocol["authorities"]["v2_execution"]["path"]).read_text()
    )["rows"]
    row_indices = [i for i, item in enumerate(expected_rows)
                   if item["row"]["split"] in ("validation", "test")]
    if (replay.get("schema") != "larql.gwstate1.replay.v1"
        or replay.get("stage") != "heldout"
        or replay.get("protocol_sha256") != protocol_status["protocol_sha256"]
        or replay.get("selection_sha256") != selection_status["selection_sha256"]
        or replay.get("executable_sha256") != protocol["runner"]["executable_sha256"]
        or replay.get("execution_file_sha256") != protocol["authorities"]["v2_execution"]["sha256"]
        or replay.get("capture_file_sha256") != protocol["authorities"]["v2_capture"]["sha256"]
        or replay.get("v2_replay_file_sha256") != protocol["authorities"]["v2_replay"]["sha256"]
        or replay.get("context_ids") != contexts
        or replay.get("h1_arms") != ["donor", "target_exact_natural"]
        or replay.get("row_indices") != row_indices
        or len(row_indices) != 270
        or replay.get("intervention_firings") != 270 * (3 + 2 * len(contexts))
        or replay.get("parity_bit_mismatches") != 0):
        raise ValueError("historical held-out replay is inadmissible")
    proximal = array(REPLAY.parent, replay, "proximal.f32")
    terminal = array(REPLAY.parent, replay, "terminal.f32")
    if (proximal.shape != (270, len(contexts), 2, 142)
        or terminal.shape != proximal.shape
        or adjudication.get("heldout_replay_artifact_sha256") !=
           {item["path"]: item["sha256"] for item in replay["artifacts"]}):
        raise ValueError("historical replay tensor geometry or identity mismatch")

    results = {}
    gate_trace = {}
    for split in ("validation", "test"):
        subjects, effects = subject_effects_from_logits(
            expected_rows, proximal, terminal, contexts, split, row_indices)
        analysis = adjudicate_split(
            effects, selection["carrier_branch_selected"], context_ids=contexts)
        if len(subjects) != 15 or {"subjects": subjects, "analysis": analysis} != adjudication["results"][split]:
            raise ValueError(f"{split} does not reproduce the historical adjudication")
        results[split] = analysis
        gate_trace[split] = {
            "exact_stable": analysis["exact_stable"],
            "context_255_all_natural_pass": analysis["contexts"]["255"]["gate_pass"],
            "context_0_edge_only_pass": analysis["contexts"]["0"]["gate_pass"],
            "context_128_carrier_plus_edge_pass": analysis["contexts"]["128"]["gate_pass"],
            "context_127_donor_all_heads_pass": analysis["contexts"]["127"]["gate_pass"],
        }
    exact_control = all(gate_trace[s]["exact_stable"] and
                        gate_trace[s]["context_255_all_natural_pass"]
                        for s in ("validation", "test"))
    edge_only = all(gate_trace[s]["context_0_edge_only_pass"]
                    for s in ("validation", "test"))
    carrier_plus_edge = all(gate_trace[s]["context_128_carrier_plus_edge_pass"]
                            for s in ("validation", "test"))
    actual = verdict(results["validation"], results["test"],
                     selection["carrier_branch_selected"])
    if (not exact_control or edge_only or not carrier_plus_edge
        or actual != "carrier_plus_edge"
        or adjudication["verdict"] != actual):
        raise ValueError("the disputed fallback is not proven unreachable")
    return {
        "schema": "larql.gwstate1.erratum1.invariance-witness.v1",
        "historical_artifacts_unchanged": True,
        "protocol_sha256": protocol_status["protocol_sha256"],
        "protocol_file_sha256": sha256(PROTOCOL),
        "selection_sha256": selection_status["selection_sha256"],
        "selection_file_sha256": sha256(SELECTION),
        "heldout_replay_file_sha256": sha256(REPLAY),
        "heldout_replay_artifact_sha256": adjudication["heldout_replay_artifact_sha256"],
        "adjudication_sha256": adjudication["adjudication_sha256"],
        "adjudication_file_sha256": sha256(ADJUDICATION),
        "carrier_branch_selected": selection["carrier_branch_selected"],
        "context_ids": contexts,
        "split_gates_recomputed_from_replay": gate_trace,
        "ordered_branch_trace": {
            "exact_control_pass": exact_control,
            "edge_only_pass": edge_only,
            "carrier_plus_edge_pass": carrier_plus_edge,
            "broad_context_fallback_reached": False,
        },
        "historical_verdict_reproduced": actual,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write a separate new witness JSON")
    args = parser.parse_args()
    result = witness()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
