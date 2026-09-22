#!/usr/bin/env python3
"""Freeze GW-KEY-1 source roles and the K/V causal-search contract."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from gwsup1_preregister import canonical_hash, sha

SCHEMA = "larql.gwkey1.preregistration.v1"
ROLE_SCHEMA = "larql.gwkey1.source-role-row.v1"
ROLES = (
    "bos_system",
    "subject_entity",
    "relation_query",
    "answer_cue",
    "punctuation_separator",
    "instruction_template",
)

# (subject prefix, subject suffix, relation, answer cue, punctuation).
# Subject occupies prefix..len-suffix; every unclaimed non-BOS token is
# instruction_template. The table was written from prompt templates only.
TEMPLATES: dict[str, tuple[int, int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = {
    "capital-canonical": (4, 1, (2,), (-1,), ()),
    "capital-alternate": (6, 1, (3, 4), (5,), (-1,)),
    "capital-question": (8, 4, (6,), (-2,), (2, -4, -3, -1)),
    "currency-canonical": (4, 1, (2,), (-1,), ()),
    "currency-alternate": (3, 1, (1,), (2,), (-1,)),
    "currency-question": (8, 3, (6,), (-2,), (2, -3, -1)),
    "language-canonical": (5, 1, (2, 3), (-1,), ()),
    "language-alternate": (4, 1, (1, 2), (3,), (-1,)),
    "language-question": (9, 4, (4, 6, 7), (-2,), (2, -4, -3, -1)),
    "hyp-1": (2, 4, (-2,), (-4, -3, -1), ()),
    "hyp-5": (2, 5, (-3,), (-5, -4, -2, -1), ()),
    "hyp-3": (9, 3, (4, 6), (-2,), (2, -3, -1)),
}


def resolve(indices: tuple[int, ...], length: int) -> list[int]:
    return [index if index >= 0 else length + index for index in indices]


def role_row(row: dict[str, Any]) -> dict[str, Any]:
    template = row["prompt"]["template_id"]
    if template not in TEMPLATES:
        raise ValueError(f"unregistered template {template}")
    tokens = row["prompt"]["token_ids"]
    length = len(tokens)
    prefix, suffix, relation_raw, answer_raw, punctuation_raw = TEMPLATES[template]
    subject = list(range(prefix, length - suffix))
    relation = resolve(relation_raw, length)
    answer = resolve(answer_raw, length)
    punctuation = resolve(punctuation_raw, length)
    assigned = {0, *subject, *relation, *answer, *punctuation}
    instruction = [index for index in range(length) if index not in assigned]
    roles = {
        "bos_system": [0],
        "subject_entity": subject,
        "relation_query": relation,
        "answer_cue": answer,
        "punctuation_separator": punctuation,
        "instruction_template": instruction,
    }
    flat = [position for role in ROLES for position in roles[role]]
    if sorted(flat) != list(range(length)) or len(flat) != len(set(flat)):
        raise ValueError(f"{row['edge_id']}: role map is not exhaustive and disjoint")
    if not subject or not relation or not answer:
        raise ValueError(f"{row['edge_id']}: required semantic role is empty")
    return {
        "schema": ROLE_SCHEMA,
        "edge_id": row["edge_id"],
        "template_id": template,
        "split": row["split"],
        "relation": row["semantic_edge"]["relation"],
        "prompt_semantic_family": row["semantic_edge"]["prompt_semantic_family"],
        "token_count": length,
        "capture_position": length - 1,
        "roles": roles,
    }


def relative(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve())


def build(args: argparse.Namespace) -> dict[str, Any]:
    inputs = [json.loads(line) for line in args.input_rows.read_text().splitlines() if line.strip()]
    if len(inputs) != 426 or len({row["edge_id"] for row in inputs}) != 426:
        raise ValueError("GW-KEY-1 requires the frozen 426-row cohort")
    role_rows = [role_row(row) for row in inputs]
    args.roles.parent.mkdir(parents=True, exist_ok=True)
    args.roles.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in role_rows))

    head = json.loads(args.head_adjudication.read_text())
    selection = json.loads(args.head_selection.read_text())
    if head.get("schema") != "larql.gwhead1.adjudication.v1" or head["gate"]["verdict"] != "causal_head_subset_supported":
        raise ValueError("GW-HEAD-1 causal gate did not pass")
    if selection.get("schema") != "larql.gwhead1.selection.v1" or selection["scopes"]["global"]["selected"]["heads"] != [1]:
        raise ValueError("GW-KEY-1 requires frozen zero-based L24H1")
    root = args.output.parent
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "frozen_pre_execution",
        "frozen_date": "2026-09-21",
        "preregistration_sha256": None,
        "question": "which template-derived source roles causally supply L24H1 routing and content for the GW-HEAD-1 alignment effect?",
        "claim": "a frozen compact set of source roles is jointly necessary and sufficient, through L24H1, for held-out control-adjusted semantic alignment",
        "claim_boundary": "source K/V roles through frozen L24H1 only; natural final-position Q; not Q construction, correctness, cheap partial execution, an access primitive, or WALK",
        "site": {"component": "target", "layer": 24, "head": 1, "head_numbering": "zero_based", "query": "natural final-position Q", "source_window": "all naturally attended prompt positions including the final position"},
        "roles": {
            "ordered_vocabulary": list(ROLES),
            "count": len(ROLES),
            "assignment": "fixed template table; exhaustive, non-overlapping, outcome-blind",
            "empty_roles": "allowed only for punctuation_separator and instruction_template; semantic roles subject_entity, relation_query, and answer_cue are always populated",
            "exact_positions": "recorded beneath roles and never searched independently",
            "artifact": {"path": relative(args.roles, root), "sha256": sha(args.roles), "rows": len(role_rows)},
        },
        "interventions": {
            "natural_q": "immutable in every arm",
            "K": "replace source K vectors before H1 scoring, then rerun real score scaling, softcap if declared, and softmax over every source position",
            "V": "hold the arm's routing weights fixed and replace source V vectors before weighted-V aggregation",
            "other_heads": "all seven non-H1 query heads remain natural",
            "downstream": "effective prepared-production Q8 W_O, declared post-attention norm and residual write, then canonical L24 FFN and every later layer",
            "reference_bank": {
                "grouping": "template_id x source_role x K-or-V",
                "direction": "mean conditioned source vector from train rows only across every position in the cell",
                "train": "leave-one-semantic-edge-out",
                "heldout": "all train rows; no validation/test donor",
                "scaling": "scale each reference vector to the natural source-vector L2 norm",
                "failure": "zero/non-finite direction or norm refuses",
            },
            "families": {
                "K": "selected roles natural K, other roles reference K; every V natural",
                "V": "every K natural; selected roles natural V, other roles reference V",
                "joint": "selected K roles and selected V roles natural; their complements reference",
            },
            "arms": ["I_full", "I_ref", "I_S", "I_full_minus_S", "I_identity"],
        },
        "search": {
            "split": "train_only",
            "K_subsets": 64,
            "V_subsets": 64,
            "joint_pairs": 4096,
            "execution": "direct H1 scoring/softmax/aggregation for every candidate; no summed singleton attribution",
            "eligibility": "R(S) >= 0.80 independently for raw and z-scored candidate JS with positive E(I_full)",
            "retention_fraction": 0.80,
            "tie_break_K_or_V": ["minimum role cardinality", "higher minimum retained fraction", "lower replaced-vector norm", "lexicographic role-bit mask"],
            "tie_break_joint": ["minimum union role cardinality", "minimum total K-plus-V cardinality", "higher minimum retained fraction", "lower replaced-vector norm", "lexicographic K mask then V mask"],
            "compact": "union of K and V selected roles has cardinality <= 3 of 6",
            "heldout_may_not": ["change H1", "change roles or positions", "change K/V masks", "change reference bank", "change estimands or thresholds"],
        },
        "measurements": {
            "inherited": "GW-HEAD-1 matched controls, 126 candidates, carrier cosine, raw candidate JS and z-scored candidate JS",
            "E": "same-fact before-to-after gain minus matched different-fact gain; larger means more alignment",
            "necessity": "N(S)=E(I_full)-E(I_full_minus_S)",
            "sufficiency": "R(S)=(E(I_S)-E(I_ref))/(E(I_full)-E(I_ref))",
            "terminal": "mandatory separate transport report",
            "carrier": "qualification only; cannot rescue either candidate metric",
        },
        "adjudication": {
            "K": "held-out necessity lower 95% > 0 and sufficiency lower 95% >= 0.50 for both candidate metrics on validation and test",
            "V": "same separate gate",
            "joint": "same conjunctive gate; only this earns candidate read path",
            "compact_candidate_read_path": "joint gate passes and union role cardinality <= 3",
            "shared_roles_across_relations": "reported, not required",
            "source_identity": "exact positions reported descriptively beneath frozen roles",
        },
        "statistics": {"unit": "semantic edge", "interval": "10000-sample percentile cluster bootstrap", "global_stratification": "relation", "seed": 1398114331, "splits": ["validation", "test"]},
        "authorities": {
            "input_rows": {"path": relative(args.input_rows, root), "sha256": sha(args.input_rows)},
            "gwhead1_selection": {"path": relative(args.head_selection, root), "sha256": sha(args.head_selection), "identity": selection["selection_sha256"]},
            "gwhead1_adjudication": {"path": relative(args.head_adjudication, root), "sha256": sha(args.head_adjudication), "identity": head["adjudication_sha256"]},
            "candidate_identity": head["authorities"]["candidate_identity_sha256"],
        },
        "causal_ladder": {"current": "GW-KEY-1 source-role causality through frozen L24H1", "successor": "GW-READ-1 natural-Q cost and partial execution", "source_key_is_not_efficiency": True},
    }
    document["preregistration_sha256"] = canonical_hash(document, "preregistration_sha256")
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def validate(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("schema") != SCHEMA or document.get("status") != "frozen_pre_execution":
        raise ValueError("not a frozen GW-KEY-1 preregistration")
    identity = canonical_hash(document, "preregistration_sha256")
    if identity != document.get("preregistration_sha256"):
        raise ValueError("GW-KEY-1 identity mismatch")
    root = path.parent
    for name in ("input_rows", "gwhead1_selection", "gwhead1_adjudication"):
        item = document["authorities"][name]
        if sha((root / item["path"]).resolve()) != item["sha256"]:
            raise ValueError(f"{name} authority changed")
    roles = document["roles"]["artifact"]
    role_path = (root / roles["path"]).resolve()
    if sha(role_path) != roles["sha256"]:
        raise ValueError("source-role artifact changed")
    rows = [json.loads(line) for line in role_path.read_text().splitlines() if line.strip()]
    if len(rows) != 426 or any(row.get("schema") != ROLE_SCHEMA for row in rows):
        raise ValueError("source-role artifact is incomplete")
    for row in rows:
        positions = [position for role in ROLES for position in row["roles"][role]]
        if sorted(positions) != list(range(row["token_count"])) or len(positions) != len(set(positions)):
            raise ValueError("source-role artifact is not exhaustive and disjoint")
        if any(not row["roles"][role] for role in ("subject_entity", "relation_query", "answer_cue")):
            raise ValueError("source-role artifact has an empty semantic role")
    return {"schema": SCHEMA, "status": "valid", "preregistration_sha256": identity, "role_rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--input-rows", type=Path, required=True)
    build_parser.add_argument("--head-selection", type=Path, required=True)
    build_parser.add_argument("--head-adjudication", type=Path, required=True)
    build_parser.add_argument("--roles", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("preregistration", type=Path)
    args = parser.parse_args()
    result = build(args) if args.command == "build" else validate(args.preregistration)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
