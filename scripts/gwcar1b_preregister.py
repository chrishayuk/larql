#!/usr/bin/env python3
"""Bind and check the CAR-1B source-design draft; never authorize model outcomes."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from gwcar1b_sources import CONSTRUCTION_IDS, FEATURES, GRID, grid_document
from gwv2_population import canonical_hash, read_jsonl, sha256

ROOT = Path(__file__).resolve().parents[1]
V2 = Path("bench/gw-v2/gemma3-4b-it-phase1")
DEFAULT = ROOT / "bench/gw-car-1/gemma3-4b-it-phase1/gwcar1b-protocol.draft.json"
PENDING = [
    "fresh confirmation factual scope and source (old eligibility pool exhausted)",
    "population rows, splits, sample sizes, tokenizer and role map",
    "fresh donor/control maps and candidate-token vocabulary",
    "hash-bound subject embedding and L12/L16 feature capture",
    "candidate serialization and complete per-candidate cost ledger",
    "population-bound paired replay with fresh identity controls",
    "executable simultaneous analysis including template/shuffle comparisons",
    "joint review and model/executable seal before new candidate outcomes",
]
AUTHORITIES = (
    "docs/gw-car-1b.md", "docs/gw-car-1a-results.md", "docs/gw-state-1-results.md",
    "docs/gw-state-1-erratum-1.md",
    "bench/gw-car-1/gemma3-4b-it-phase1/gwcar1a-protocol.json",
    "bench/gw-car-1/gemma3-4b-it-phase1/gwcar1a-adjudication.json",
    str(V2 / "population-manifest.json"), str(V2 / "input.jsonl"),
    "bench/gw0/gemma3-4b-it-phase1/input.jsonl",
    "scripts/gwcar1b_sources.py", "scripts/gwcar1b_preregister.py",
    "scripts/gwv2_population.py", "scripts/test_gwcar1b_sources.py",
    "scripts/test_gwcar1b_preregister.py",
    "scripts/gwcar1b_population.py", "scripts/test_gwcar1b_population.py",
    "bench/gw-car-1/gemma3-4b-it-phase1/gwcar1b-capital-population.proposal.json",
)


def excluded_subjects() -> set[str]:
    population = json.loads((ROOT / V2 / "population-manifest.json").read_text())
    return ({row["subject_id"] for row in read_jsonl(ROOT / V2 / "input.jsonl")}
            | set(population["freshness"]["resolved_old_subjects"].values()))


def check_fresh_rows(rows: list[dict], excluded: set[str]) -> dict:
    """Metadata-only freshness gate, not a factual/population qualification."""
    if not rows:
        raise ValueError("empty confirmation population")
    seen_rows: set[str] = set()
    subjects: dict[str, str] = {}
    for row in rows:
        if set(row) != {"row_id", "subject_id", "relation", "template", "split"}:
            raise ValueError("freshness rows must use the declared metadata-only schema")
        if any(not isinstance(v, str) or not v for v in row.values()):
            raise ValueError("empty or invalid population field")
        subject, split = row["subject_id"], row["split"]
        if len(subject) != 3 or not subject.isascii() or not subject.isalpha() or subject != subject.upper():
            raise ValueError("country identity must be canonical uppercase alpha-3")
        if split not in ("validation", "test"):
            raise ValueError("fresh rows must be validation or test")
        if row["row_id"] in seen_rows:
            raise ValueError("duplicate confirmation row ID")
        if subject in excluded:
            raise ValueError(f"previously studied subject cannot be fresh confirmation: {subject}")
        if subject in subjects and subjects[subject] != split:
            raise ValueError("subject crosses validation/test")
        seen_rows.add(row["row_id"])
        subjects[subject] = split
    if set(subjects.values()) != {"validation", "test"}:
        raise ValueError("both confirmation splits required")
    return {"metadata_freshness_pass": True, "population_qualified": False,
            "rows": len(rows), "subjects": len(subjects),
            "model_execution_authorized": False}


def expected_document() -> dict:
    rows = read_jsonl(ROOT / V2 / "input.jsonl")
    train = [row for row in rows if row["split"] == "train"]
    document = {
        "schema": "larql.gwcar1b.source-design.v1",
        "status": "design_draft_no_car1b_model_outcomes",
        "model_execution_authorized": False,
        "authorities": {name: sha256(ROOT / name) for name in AUTHORITIES},
        "training": {"source": str(V2 / "input.jsonl"), "split": "train",
                     "rows": len(train), "subjects": len({r["subject_id"] for r in train}),
                     "row_ids": [r["edge_id"] for r in train]},
        "confirmation": {"population_bound": False,
                         "proposed_claim": "fresh subject transfer under a revised factual scope",
                         "proposal": "gwcar1b-capital-population.proposal.json",
                         "proposal_subjects": 67, "proposal_rows": 201,
                         "excluded_subject_ids": sorted(excluded_subjects()),
                         "old_heldout_is_fresh": False},
        "interface": {"layer": 24, "head": 1, "carrier_width": 2560,
                      "context": 128, "reference_context": 255,
                      "target_h1": "natural target oracle",
                      "other_heads": "seven matched donor heads in both arms",
                      "target_kv_and_continuation": "natural target prefix and suffix",
                      "candidate_token_ids": None},
        "source_grid": grid_document(), "construction_candidate_ids": list(CONSTRUCTION_IDS),
        "features": list(FEATURES),
        "analysis": {"status": "prospective_requirements_not_executable",
                     "bootstrap_seed": 27022034, "bootstrap_replicates": 10000,
                     "proximal_point_min": 0.8, "proximal_lower_min": 0.5,
                     "terminal_lower_min_exclusive": 0,
                     "absolute_readouts": len(GRID) * 4,
                     "difference_pairs": [[11, 8], [12, 8], [13, 8], [11, 17], [12, 18], [13, 19]],
                     "difference_readouts": 24, "family_readouts_per_split": 104,
                     "selection_on_validation": False, "mandatory_control": 0},
        "claims": {"source_adequacy": "requires prospective paired causal replay",
                   "natural_state_fidelity": False, "end_to_end_prefix_avoidance": False,
                   "latency_improvement": False},
        "remaining_before_seal": PENDING,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    document["draft_sha256"] = canonical_hash(document, "draft_sha256")
    return document


def validate(path: Path) -> dict:
    document = json.loads(path.read_text())
    if document != expected_document():
        raise ValueError("CAR-1B source draft differs from its authorities or declared design")
    return {"status": document["status"], "draft_sha256": document["draft_sha256"],
            "ready_to_seal": False, "model_execution_authorized": False,
            "remaining_before_seal": PENDING}


def write_new(path: Path) -> None:
    document = expected_document()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["draft", "validate", "check-population"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "check-population":
        result = check_fresh_rows(read_jsonl(args.path), excluded_subjects())
    else:
        if args.command == "draft":
            write_new(args.path)
        result = validate(args.path)
    print(json.dumps(result, indent=2, sort_keys=True))
