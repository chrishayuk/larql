#!/usr/bin/env python3
"""Construct and validate the outcome-blind CAR-1A protocol and executable seal."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from gwstate1_preregister import FROZEN_STATUS as STATE1_FROZEN
from gwstate1_preregister import validate as validate_state1
from gwstate1_select import validate as validate_state1_selection
from gwv2_population import canonical_hash, sha256

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "bench/gw-car-1/gemma3-4b-it-phase1"
STATE1 = ROOT / "bench/gw-state-1/gemma3-4b-it-phase1/gwstate1-protocol.json"
STATE1_SELECTION = ROOT / "output/gwstate1-gemma3-4b-it-phase1-selection.json"
STATE1_ADJUDICATION = ROOT / "output/gwstate1-gemma3-4b-it-phase1-adjudication.json"
STATE1_TRAIN = ROOT / "output/gwstate1-gemma3-4b-it-phase1-train/replay.json"
STATE1_HELDOUT = ROOT / "output/gwstate1-gemma3-4b-it-phase1-heldout/replay.json"
STATE1_WITNESS = ROOT / "bench/gw-state-1/gemma3-4b-it-phase1/gwstate1-erratum1-invariance-witness.json"
RUNNER = ROOT / "crates/larql-demos/examples/inference/gwcar1a_record.rs"
SPEC = ROOT / "docs/gw-car-1.md"
SCHEMA = "larql.gwcar1a.protocol.v1"
DRAFT_STATUS = "design_draft_no_car1a_outcomes"
FROZEN_STATUS = "frozen_before_car1a_outcomes"
CODE = (
    "gwcar1a_codecs.py", "gwcar1a_candidates.py", "gwcar1a_analysis.py",
    "gwcar1a_adjudicate.py", "gwcar1a_preregister.py",
    "gwstate1_analysis.py", "gwstate1_preflight.py", "gwstate1_preregister.py",
    "gwstate1_select.py", "gwv2_adjudicate.py", "gwv2_amend.py",
    "gwv2_io.py", "gwv2_population.py", "gwv2_prepare.py",
    "gwv2_preregister.py", "gwv2_preflight.py", "gwkey1_preregister.py",
    "gwsup1_preregister.py",
)


def authority(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def verify_replay(path: Path, stage: str, protocol_hash: str, executable_hash: str) -> dict:
    replay = json.loads(path.read_text())
    count = 396 if stage == "train" else 270
    if (replay.get("schema") != "larql.gwstate1.replay.v1"
        or replay.get("stage") != stage
        or replay.get("protocol_sha256") != protocol_hash
        or replay.get("executable_sha256") != executable_hash
        or replay.get("h1_arms") != ["donor", "target_exact_natural"]
        or len(replay.get("row_indices", [])) != count
        or 0 not in replay.get("context_ids", [])
        or 128 not in replay.get("context_ids", [])
        or 255 not in replay.get("context_ids", [])
        or replay.get("parity_bit_mismatches") != 0):
        raise ValueError(f"inadmissible STATE-1 {stage} replay for CAR-1A")
    for entry in replay.get("artifacts", []):
        file = path.parent / entry["path"]
        if (entry.get("shape") != [count, len(replay["context_ids"]), 2, 142]
            or entry.get("bytes") != file.stat().st_size
            or entry.get("sha256") != sha256(file)):
            raise ValueError(f"STATE-1 {stage} replay artifact changed")
    if {item["path"] for item in replay["artifacts"]} != {"proximal.f32", "terminal.f32"}:
        raise ValueError("STATE-1 reference artifacts incomplete")
    return replay


def inherited() -> tuple[dict, dict]:
    status = validate_state1(STATE1)
    if status["status"] != STATE1_FROZEN:
        raise ValueError("CAR-1A requires frozen STATE-1")
    selection_status = validate_state1_selection(STATE1, STATE1_SELECTION)
    if selection_status["carrier_branch_selected"] != [None, 128]:
        raise ValueError("CAR-1A requires historical context-128 selection")
    state1 = json.loads(STATE1.read_text())
    adjudication = json.loads(STATE1_ADJUDICATION.read_text())
    witness = json.loads(STATE1_WITNESS.read_text())
    if (adjudication.get("adjudication_sha256") != canonical_hash(adjudication, "adjudication_sha256")
        or adjudication.get("protocol_sha256") != state1["protocol_sha256"]
        or adjudication.get("selection_sha256") != selection_status["selection_sha256"]
        or adjudication.get("verdict") != "carrier_plus_edge"
        or witness.get("protocol_sha256") != state1["protocol_sha256"]
        or witness.get("adjudication_sha256") != adjudication["adjudication_sha256"]
        or witness.get("adjudication_file_sha256") != sha256(STATE1_ADJUDICATION)
        or witness.get("historical_verdict_reproduced") != "carrier_plus_edge"
        or witness.get("heldout_replay_file_sha256") != sha256(STATE1_HELDOUT)):
        raise ValueError("CAR-1A inherited STATE-1 outcome authority changed")
    verify_replay(STATE1_TRAIN, "train", state1["protocol_sha256"],
                  state1["runner"]["executable_sha256"])
    verify_replay(STATE1_HELDOUT, "heldout", state1["protocol_sha256"],
                  state1["runner"]["executable_sha256"])
    return state1, adjudication


def expected_document() -> dict:
    state1, adjudication = inherited()
    paths = {
        "state1_protocol": STATE1,
        "state1_selection": STATE1_SELECTION,
        "state1_adjudication": STATE1_ADJUDICATION,
        "state1_train_replay": STATE1_TRAIN,
        "state1_heldout_replay": STATE1_HELDOUT,
        "state1_erratum_witness": STATE1_WITNESS,
        "v2_execution": Path(state1["authorities"]["v2_execution"]["path"]),
        "v2_capture": Path(state1["authorities"]["v2_capture"]["path"]),
    }
    return {
        "schema": SCHEMA, "status": DRAFT_STATUS, "protocol_sha256": None,
        "human_spec": authority(SPEC),
        "authorities": {key: authority(path) for key, path in sorted(paths.items())},
        "inherited_state1_protocol_sha256": state1["protocol_sha256"],
        "inherited_state1_adjudication_sha256": adjudication["adjudication_sha256"],
        "inherited_context": 128,
        "model_segments": state1["model_segments"],
        "model_metadata": state1["model_metadata"],
        "execution_identity": state1["execution_identity"],
        "plan_sha256": state1["plan_sha256"],
        "population": {"rows": 666, "train": 396, "validation": 135, "test": 135,
                       "donor_map_sha256": state1["matched_control_donor_indices_sha256"]},
        "interface": {"layer": 24, "head": 1, "candidate_token_width": 142,
                      "carrier_width": 2560, "candidate_ids": list(range(19)),
                      "h1_arms": ["donor", "target_exact_natural"],
                      "non_h1_heads": "seven frozen matched-donor values in both H1 arms",
                      "target_prefix_kv_and_continuation": "natural target row",
                      "only_changed_input": "entering L24 attention carrier",
                      "controls": {"0": "exact target-natural carrier; STATE-1 context 128 identity",
                                   "1": "matched-donor carrier; STATE-1 context 0 identity"}},
        "candidate_grid": [
            {"id": 0, "family": "exact", "level": 32},
            {"id": 1, "family": "donor", "level": 32},
            {"id": 2, "family": "f16", "level": 16},
            *({"id": i, "family": "symmetric_quant", "level": level}
              for i, level in ((3, 8), (4, 4), (5, 2))),
            *({"id": 6 + i, "family": "sparse_top_abs", "level": level}
              for i, level in enumerate((32, 64, 128, 256, 512, 1024))),
            *({"id": 12 + i, "family": "pca", "level": level}
              for i, level in enumerate((8, 16, 32, 64, 128, 256, 384))),
        ],
        "analysis": {"reference_context": 255,
                     "candidate0_must_clear": True,
                     "split_gates": ["validation", "test"],
                     "bootstrap_unit": "sorted subject ID with nine executions and three factual relations",
                     "bootstrap_rng": "numpy.default_rng PCG64, reset seed 27022033 per split",
                     "bootstrap_replicates": 10000, "bootstrap_ddof": 1,
                     "bootstrap_quantile": "numpy linear",
                     "denominator_refusal": "nonpositive point, any draw, or 2.5th percentile on raw or z",
                     "simultaneous_family": "all 19 candidates x raw/z proximal retention and raw/z terminal effect",
                     "simultaneous_method": "one-sided 95% studentized max-t lower band",
                     "proximal_point_min": 0.80, "proximal_lower_min": 0.50,
                     "terminal_lower_strictly_positive": True,
                     "total_bytes_amortization_rows": 666,
                     "claim": "captured-carrier compactness only; no cheaper execution"},
        "cost": {"report": ["encoded bytes per row", "shared model bytes",
                            "total bytes at 666 rows", "source requires natural prefix",
                            "decode arithmetic ledger"],
                 "wall_time_claim": False},
        "runner": {"source": authority(RUNNER),
                   "cargo_manifest": authority(ROOT / "crates/larql-demos/Cargo.toml"),
                   "executable_sha256": None,
                   "train_stage": "train PROTOCOL CANDIDATES OUTPUT",
                   "heldout_stage": "heldout PROTOCOL CANDIDATES OUTPUT"},
        "code": {name: authority(ROOT / "scripts" / name) for name in CODE},
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "seal_order": ["joint protocol/implementation review", "protocol/executable seal",
                       "train-only PCA fit", "complete train candidate replay",
                       "held-out candidate replay", "adjudication"],
        "car1a_outcomes_before_seal": 0,
    }


def validate(path: Path) -> dict:
    document = json.loads(path.read_text())
    expected = expected_document()
    if document.get("status") == FROZEN_STATUS:
        executable = document.get("runner", {}).get("executable_sha256")
        if not isinstance(executable, str) or not executable.startswith("sha256:"):
            raise ValueError("frozen CAR-1A protocol lacks executable identity")
        expected["status"] = FROZEN_STATUS
        expected["runner"]["executable_sha256"] = executable
        expected["protocol_sha256"] = canonical_hash(expected, "protocol_sha256")
    elif document.get("status") != DRAFT_STATUS:
        raise ValueError("invalid CAR-1A protocol status")
    if document != expected:
        raise ValueError("CAR-1A protocol differs from checked authorities or declared experiment")
    return {"schema": SCHEMA, "status": document["status"],
            "protocol_sha256": document["protocol_sha256"]}


def write_new(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def seal(draft: Path, output: Path, executable: Path) -> dict:
    if validate(draft)["status"] != DRAFT_STATUS:
        raise ValueError("only a reviewed CAR-1A draft can be sealed")
    if any((ROOT / "output").glob("gwcar1a-gemma3-4b-it-phase1*")):
        raise ValueError("CAR-1A outcomes or inputs already exist before seal")
    document = json.loads(draft.read_text())
    document["status"] = FROZEN_STATUS
    document["runner"]["executable_sha256"] = sha256(executable)
    document["protocol_sha256"] = canonical_hash(document, "protocol_sha256")
    write_new(output, document)
    validate(output)
    return {"protocol_sha256": document["protocol_sha256"],
            "executable_sha256": document["runner"]["executable_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["draft", "validate", "seal"])
    parser.add_argument("path", type=Path)
    parser.add_argument("extra", nargs="*", type=Path)
    args = parser.parse_args()
    if args.command == "draft" and not args.extra:
        write_new(args.path, expected_document())
        result = validate(args.path)
    elif args.command == "validate" and not args.extra:
        result = validate(args.path)
    elif args.command == "seal" and len(args.extra) == 2:
        result = seal(args.extra[0], args.path, args.extra[1])
    else:
        parser.error("draft PATH | validate PATH | seal OUTPUT DRAFT EXECUTABLE")
    print(json.dumps(result, indent=2, sort_keys=True))
