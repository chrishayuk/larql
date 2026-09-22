#!/usr/bin/env python3
"""Build or validate the frozen GW-CONV-1 operator-localization contract."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from gwsup1_preregister import canonical_hash, jsonl, sha

SCHEMA = "larql.gwconv1.preregistration.v1"
SEED = 1_398_100_531


def relative(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve())


def authority(path: Path, root: Path) -> dict[str, Any]:
    return {"path": relative(path, root), "sha256": sha(path)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    gwsup_prereg = json.loads(args.gwsup_preregistration.read_text())
    candidates = json.loads(args.candidates.read_text())
    readout = json.loads(args.gwsup_readout_manifest.read_text())
    adjudication = json.loads(args.gwsup_adjudication.read_text())
    sealed = json.loads(args.sealed_manifest.read_text())
    input_manifest = json.loads(args.input_manifest.read_text())
    rows = jsonl(args.input_manifest.parent / input_manifest["rows"]["path"])
    triples: dict[tuple[str, str, str], str] = {}
    for row in rows:
        semantic = row["semantic_edge"]
        key = (semantic["subject"], semantic["relation"], semantic["target"])
        previous = triples.setdefault(key, row["split"])
        if previous != row["split"]:
            raise ValueError(f"semantic edge crosses splits: {key}")
    splits = Counter(triples.values())
    relations = Counter((key[1], split) for key, split in triples.items())
    if splits != {"train": 85, "validation": 29, "test": 28}:
        raise ValueError(f"unexpected semantic-edge split: {splits}")
    root = args.output.parent
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "frozen_pre_before_readout",
        "frozen_date": "2026-09-21",
        "preregistration_sha256": None,
        "question": "which observed writes make same-fact prompt trajectories more alike than frozen relation-matched controls?",
        "claim_boundary": (
            "observational convergence-operator localization; not causal attribution, "
            "not proof of a graph edge, and not authorization for attention-head semantics"
        ),
        "authorities": {
            "input_manifest": authority(args.input_manifest, root),
            "sealed_gw0": {
                **authority(args.sealed_manifest, root),
                "bundle_sha256": sealed["bundle_sha256"],
                "census_sha256": sealed["census"]["sha256"],
            },
            "gwsup1_preregistration": {
                **authority(args.gwsup_preregistration, root),
                "identity_sha256": gwsup_prereg["preregistration_sha256"],
            },
            "candidates": {
                **authority(args.candidates, root),
                "identity_sha256": candidates["candidate_identity_sha256"],
            },
            "gwsup1_after_readout": {
                **authority(args.gwsup_readout_manifest, root),
                "candidate_logits_sha256": readout["artifacts"]["logits"]["sha256"],
                "rows_sha256": readout["artifacts"]["rows"]["sha256"],
                "selected_full_head_parity": readout["selected_full_head_parity"],
            },
            "gwsup1_adjudication": {
                **authority(args.gwsup_adjudication, root),
                "identity_sha256": adjudication["adjudication_sha256"],
                "verdict": adjudication["gate"]["verdict"],
            },
            "container_identity": readout["authorities"]["container_identity"],
            "plan_sha256": readout["authorities"]["plan_sha256"],
            "readout_basis": "prepared-production-effective-final-norm-and-q8-output-head/v1",
        },
        "cohort": {
            "semantic_edges": len(triples),
            "execution_rows": len(rows),
            "prompt_rows_per_edge": 3,
            "splits": dict(sorted(splits.items())),
            "relation_by_split": {
                relation: {
                    split: relations[(relation, split)]
                    for split in ("train", "validation", "test")
                }
                for relation in ("capital", "currency", "language", "hypernym")
            },
            "statistical_unit": "unique (subject, relation, target) semantic edge",
        },
        "states": {
            "sites": 68,
            "order": "layer ascending, attention then FFN",
            "before": "sealed carrier-before artifact at the observed write",
            "after": "sealed carrier-after artifact, followed by the recorded layer_scale when present",
            "prompt_reexecution": False,
            "candidate_vocabulary": "the unchanged frozen GW-SUP-1 126-token decoder vocabulary",
            "candidate_before_readout": "new read-only selected-row pass over sealed before carriers",
            "candidate_after_readout": "immutable GW-SUP-1 candidate-logits.f32 authority",
            "zero_norm_or_zero_candidate_variance": "refuse",
        },
        "measurements": {
            "carrier": {
                "similarity": "mean of the three pairwise carrier cosines in a same-fact prompt triplet",
                "gain": "after similarity minus before similarity",
            },
            "candidate_raw": {
                "similarity": "negative mean pairwise Jensen-Shannon divergence over the 126-token restricted softmax",
                "gain": "JS before minus JS after",
            },
            "candidate_zscore": {
                "similarity": "negative mean pairwise JS after independently z-scoring each state's 126 logits",
                "gain": "z-score JS before minus z-score JS after",
            },
            "control": (
                "the three frozen same-split, same-relation, same-prompt-family, "
                "different-subject-and-target rows already bound by GW-SUP-1"
            ),
            "adjusted_gain": "same-fact gain minus matched-control gain; positive means excess convergence",
            "operator_class": "attention or FFN from the immutable site universe",
            "emergence_relation": "selected-site order minus the edge's median sealed emergence-site order",
        },
        "selection": {
            "discovery_split": "train only",
            "global_site": (
                "site with maximum train mean adjusted candidate_raw gain across all 85 edges; "
                "earliest site wins an exact tie"
            ),
            "relation_sites": (
                "within each relation, site with maximum train mean adjusted candidate_raw gain; "
                "earliest exact tie"
            ),
            "held_out_sites_may_not_move": True,
            "validation_or_test_outcomes_may_not_select_a_site": True,
        },
        "statistics": {
            "interval": "10000-sample percentile cluster bootstrap over semantic edges",
            "global_stratification": "relation-stratified",
            "relation_specific": "within-relation edge bootstrap",
            "seed": SEED,
            "validation_and_test_reported_separately": True,
            "site_profile_descriptives": [
                "positive adjusted-gain mass captured by top 1, 2, 4 and 8 sites",
                "Herfindahl and effective-site count over positive adjusted gain",
                "attention-versus-FFN adjusted gain",
            ],
        },
        "adjudication": {
            "global_operator": (
                "at the train-selected global site, lower 95% bounds for carrier, candidate_raw "
                "and candidate_zscore adjusted gain are all above zero on validation and test"
            ),
            "relation_routing": (
                "at the four train-selected relation sites, the pooled relation-specific test "
                "lower 95% bounds for all three adjusted gains are above zero"
            ),
            "cross_relation_generality": (
                "the train-selected global site has positive mean candidate_raw adjusted gain "
                "in at least three of four relations on both validation and test"
            ),
            "emergence_timing": (
                "descriptive only: signed site distance and fraction at-or-before emergence; "
                "it cannot rescue a failed held-out gain gate"
            ),
            "full_support": "global_operator AND relation_routing AND cross_relation_generality",
            "no_post_hoc_site_or_window": True,
        },
        "interpretation_contract": {
            "candidate_only": "semantic readout convergence without demonstrated carrier alignment",
            "carrier_only": "geometric alignment without demonstrated semantic convergence",
            "train_only": "discovery, not a stable operator",
            "held_out_global": "a stable shared convergence locus",
            "held_out_relation_sites": "relation-conditioned convergence routing",
            "attention_selected": "prioritizes HEAD decomposition but does not itself identify a head",
            "ffn_selected": "prioritizes FFN write-family analysis but does not identify a feature edge",
        },
    }
    document["preregistration_sha256"] = canonical_hash(
        document, "preregistration_sha256"
    )
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def validate(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("schema") != SCHEMA or document.get("status") != "frozen_pre_before_readout":
        raise ValueError("not a frozen GW-CONV-1 preregistration")
    identity = canonical_hash(document, "preregistration_sha256")
    if identity != document.get("preregistration_sha256"):
        raise ValueError("GW-CONV-1 canonical identity mismatch")
    root = path.parent
    for name in (
        "input_manifest",
        "sealed_gw0",
        "gwsup1_preregistration",
        "candidates",
        "gwsup1_after_readout",
        "gwsup1_adjudication",
    ):
        item = document["authorities"][name]
        if sha((root / item["path"]).resolve()) != item["sha256"]:
            raise ValueError(f"{name} authority changed")
    if document["cohort"]["splits"] != {"train": 85, "validation": 29, "test": 28}:
        raise ValueError("GW-CONV-1 split changed")
    return {
        "schema": SCHEMA,
        "status": "valid",
        "preregistration_sha256": identity,
        "cohort": document["cohort"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--input-manifest", type=Path, required=True)
    build_parser.add_argument("--sealed-manifest", type=Path, required=True)
    build_parser.add_argument("--gwsup-preregistration", type=Path, required=True)
    build_parser.add_argument("--candidates", type=Path, required=True)
    build_parser.add_argument("--gwsup-readout-manifest", type=Path, required=True)
    build_parser.add_argument("--gwsup-adjudication", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("preregistration", type=Path)
    args = parser.parse_args()
    result = build(args) if args.command == "build" else validate(args.preregistration)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
