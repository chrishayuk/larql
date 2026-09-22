#!/usr/bin/env python3
"""Build or validate the frozen GW-SUP-1 candidate/readout contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PREREG_SCHEMA = "larql.gwsup1.preregistration.v1"
CANDIDATE_SCHEMA = "larql.gwsup1.candidates.v1"
CONTROL_SEED = "gwsup1-control-v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_hash(document: dict[str, Any], identity_field: str) -> str:
    payload = dict(document)
    payload.pop(identity_field, None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def relative(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve())


def control_for(row: dict, rows: list[dict]) -> str:
    semantic = row["semantic_edge"]
    target = semantic["target_token_ids"][0]
    candidates = [
        candidate
        for candidate in rows
        if candidate["split"] == row["split"]
        and candidate["semantic_edge"]["relation"] == semantic["relation"]
        and candidate["semantic_edge"]["prompt_semantic_family"]
        == semantic["prompt_semantic_family"]
        and candidate["semantic_edge"]["subject"] != semantic["subject"]
        and candidate["semantic_edge"]["target_token_ids"][0] != target
    ]
    if not candidates:
        raise ValueError(f"{row['edge_id']}: no same-split/relation/family different-target control")
    return min(
        candidates,
        key=lambda candidate: hashlib.sha256(
            f"{CONTROL_SEED}|{row['edge_id']}|{candidate['edge_id']}".encode()
        ).digest(),
    )["edge_id"]


def build_candidates(input_manifest_path: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads(input_manifest_path.read_text())
    rows_path = input_manifest_path.parent / manifest["rows"]["path"]
    if sha(rows_path) != manifest["rows"]["sha256"]:
        raise ValueError("frozen input rows hash mismatch")
    rows = jsonl(rows_path)

    triples: dict[tuple[str, str, str], dict] = {}
    by_subject: dict[str, set[tuple[str, str, int]]] = defaultdict(set)
    token_labels: dict[int, set[str]] = defaultdict(set)
    prompt_groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        semantic = row["semantic_edge"]
        token_ids = semantic["target_token_ids"]
        if len(token_ids) != 1:
            raise ValueError(f"{row['edge_id']}: GW-SUP-1 requires one declared first token")
        key = (semantic["subject"], semantic["relation"], semantic["target"])
        triple = {
            "subject": semantic["subject"],
            "relation": semantic["relation"],
            "target": semantic["target"],
            "target_token_id": token_ids[0],
        }
        previous = triples.setdefault(key, triple)
        if previous != triple:
            raise ValueError(f"inconsistent semantic triple {key}")
        by_subject[semantic["subject"]].add(
            (semantic["relation"], semantic["target"], token_ids[0])
        )
        token_labels[token_ids[0]].add(semantic["target"])
        prompt_groups[key].append(row)

    collisions = {token: labels for token, labels in token_labels.items() if len(labels) != 1}
    if collisions:
        raise ValueError(f"distinct semantic destinations share first-token IDs: {collisions}")
    if any(len(group) != 3 for group in prompt_groups.values()):
        raise ValueError("every semantic triple must have exactly three prompt families")

    subjects = []
    primary_triples = 0
    for subject, edges in sorted(by_subject.items()):
        edge_rows = [
            {"relation": relation, "target": target, "target_token_id": token}
            for relation, target, token in sorted(edges)
        ]
        primary = len(edge_rows) >= 2
        primary_triples += len(edge_rows) if primary else 0
        subjects.append(
            {
                "subject": subject,
                "candidate_token_ids": sorted({edge["target_token_id"] for edge in edge_rows}),
                "edges": edge_rows,
                "primary_multi_destination": primary,
            }
        )

    candidates = {
        "schema": CANDIDATE_SCHEMA,
        "candidate_identity_sha256": None,
        "source": {
            "input_manifest_sha256": manifest["manifest_sha256"],
            "input_rows_sha256": manifest["rows"]["sha256"],
        },
        "decoder_vocabulary": [
            {"token_id": token, "destination": next(iter(labels))}
            for token, labels in sorted(token_labels.items())
        ],
        "subjects": subjects,
        "prompt_family_groups": [
            {
                "subject": key[0],
                "relation": key[1],
                "target": key[2],
                "edge_ids": [
                    row["edge_id"]
                    for row in sorted(
                        group,
                        key=lambda value: value["semantic_edge"]["prompt_semantic_family"],
                    )
                ],
            }
            for key, group in sorted(prompt_groups.items())
        ],
        "different_destination_controls": [
            {"edge_id": row["edge_id"], "control_edge_id": control_for(row, rows)}
            for row in rows
        ],
        "control_rule": {
            "seed": CONTROL_SEED,
            "match": ["split", "relation", "prompt_semantic_family"],
            "require": ["different subject", "different target_token_id"],
            "selection": "minimum sha256(seed|edge_id|candidate_edge_id)",
        },
        "counts": {
            "execution_rows": len(rows),
            "unique_semantic_edges": len(triples),
            "subjects": len(subjects),
            "global_candidate_tokens": len(token_labels),
            "primary_multi_destination_subjects": sum(
                subject["primary_multi_destination"] for subject in subjects
            ),
            "primary_multi_destination_edges": primary_triples,
            "primary_multi_destination_rows": primary_triples * 3,
            "secondary_single_destination_subjects": sum(
                not subject["primary_multi_destination"] for subject in subjects
            ),
        },
    }
    candidates["candidate_identity_sha256"] = canonical_hash(
        candidates, "candidate_identity_sha256"
    )
    output.write_text(json.dumps(candidates, indent=2, sort_keys=True) + "\n")
    return candidates


def build(args: argparse.Namespace) -> dict[str, Any]:
    candidates = build_candidates(args.input_manifest, args.candidates)
    input_manifest = json.loads(args.input_manifest.read_text())
    sealed = json.loads(args.sealed_manifest.read_text())
    gw0b = json.loads(args.gw0b_report.read_text())
    gw3af = json.loads(args.gw3af_report.read_text())
    root = args.output.parent
    document = {
        "schema": PREREG_SCHEMA,
        "status": "frozen_pre_readout",
        "frozen_date": "2026-09-21",
        "preregistration_sha256": None,
        "claim_boundary": (
            "observational candidate-basin concentration and prompt-path convergence; "
            "not quantum collapse, causal redundancy, or proof that decoded token basins "
            "are the model's native transition ontology"
        ),
        "authorities": {
            "input_manifest": {
                "path": relative(args.input_manifest, root),
                "file_sha256": sha(args.input_manifest),
                "manifest_sha256": input_manifest["manifest_sha256"],
            },
            "input_rows": {
                "path": relative(args.input_manifest.parent / input_manifest["rows"]["path"], root),
                "sha256": input_manifest["rows"]["sha256"],
            },
            "sealed_gw0": {
                "path": relative(args.sealed_manifest, root),
                "file_sha256": sha(args.sealed_manifest),
                "bundle_sha256": sealed["bundle_sha256"],
                "census_sha256": sealed["census"]["sha256"],
            },
            "gw0b_reconciliation": {
                "path": relative(args.gw0b_report, root),
                "sha256": sha(args.gw0b_report),
            },
            "gw3af_result": {
                "path": relative(args.gw3af_report, root),
                "sha256": sha(args.gw3af_report),
                "progression_passed": gw3af["progression"]["passed"],
            },
            "candidates": {
                "path": relative(args.candidates, root),
                "file_sha256": sha(args.candidates),
                "candidate_identity_sha256": candidates["candidate_identity_sha256"],
            },
            "container_identity": "sha256:3954b0380955b8e00501f4823d93f08fece49962d358eb08150f333c4c4fb506",
            "plan_sha256": "sha256:f8f24893c784750598660b2ff724d1fa103fefa788ca791e4f464a6f2a225ddf",
            "readout_basis": "prepared-production-effective-final-norm-and-output-head/v1",
        },
        "cohorts": {
            "primary": {
                "rule": "subject has at least two distinct frozen semantic destination tokens",
                "subjects": candidates["counts"]["primary_multi_destination_subjects"],
                "semantic_edges": candidates["counts"]["primary_multi_destination_edges"],
                "execution_rows": candidates["counts"]["primary_multi_destination_rows"],
                "statistical_unit": "unique (subject, relation, target) semantic edge; prompt families are repeated measures",
            },
            "secondary": {
                "rule": "single known subject destination; eligible for global convergence but not within-neighbourhood ambiguity",
                "subjects": candidates["counts"]["secondary_single_destination_subjects"],
            },
        },
        "state_sequence": {
            "source": "sealed carrier-after artifacts only; no prompt execution",
            "sites_per_row": 68,
            "order": "layer ascending, attention then FFN",
            "post_write_rule": "apply the execution record's declared layer_scale before readout",
            "reader": "prepared production-selected physical final norm and output head, including multiplier and softcap",
            "source_weights_forbidden": True,
        },
        "candidate_space": {
            "global_distribution": "restricted softmax over all 126 frozen destination token IDs",
            "subject_neighbourhood": "conditional distribution over every frozen outgoing destination of the row subject",
            "construction_uses_trajectory_or_execution_outcome": False,
            "token_semantics": "first next-token destination; no multi-token continuation claim",
            "raw_distribution": "softmax of prepared readout logits at temperature 1",
            "scale_control": "softmax of per-state candidate-logit z-scores; zero variance refuses",
            "reported_scale_diagnostics": ["carrier_l2", "candidate_logit_mean", "candidate_logit_std", "candidate_logit_range"],
        },
        "frozen_landmarks": {
            "early_window": "post-FFN states at layers 0,1,2,3",
            "emergence_state_per_row": "the post-write state at the sealed emergence layer and site",
            "shared_emergence_state_per_semantic_edge": "median ordered emergence site across its three prompt rows",
            "late_state": "post-FFN state at the final layer",
            "concentration_site": "adjacent post-write step with the largest decrease in normalized subject-neighbourhood entropy; earliest tie",
        },
        "metrics": {
            "semantic_ambiguity": [
                "global candidate entropy / log(126)",
                "subject-neighbourhood conditional entropy / log(outdegree)",
                "subject-neighbourhood effective candidate count exp(entropy)",
                "subject-neighbourhood mass in the global restricted distribution",
                "intended-target mass globally and conditional on the subject neighbourhood",
                "target-versus-runner-up margin",
            ],
            "path_convergence": [
                "three-prompt mean pairwise Jensen-Shannon divergence over the global candidate distribution by site",
                "same statistic for frozen same-split/relation/family different-subject-and-target controls",
                "within-fact minus control difference-in-differences from early to shared emergence",
            ],
            "physical_distribution": [
                "GW-0B top-100 FFN support Jaccard where both prompt rows are eligible",
                "post-write delta cosine by site from sealed delta carriers",
            ],
            "addressability": "reported from GW-3A-F and the eight-row audit; never substituted for entropy or support concentration",
        },
        "adjudication": {
            "resampling_unit": "unique semantic edge",
            "interval": "relation-stratified cluster bootstrap, 10000 resamples, seed 1398100529, percentile 95% interval",
            "localization_null": "relation-stratified permutation of sealed emergence landmarks across semantic edges, 10000 trials, same seed",
            "candidate_coexistence": "at least half of primary semantic edges have median early effective neighbourhood count >= 1.5",
            "concentration": "raw and z-score-control upper 95% bounds for emergence-minus-early normalized neighbourhood entropy are below zero",
            "localization": "observed median concentration-to-emergence site distance is below the 1st percentile of the localization null",
            "convergence": "raw and z-score-control upper 95% bounds for the within-fact versus control early-to-emergence JS difference-in-differences are below zero",
            "full_support": "candidate_coexistence AND concentration AND localization AND convergence",
            "no_post_hoc_thresholds": True,
        },
        "interpretation_contract": {
            "entropy_drop_only": "confidence sharpening",
            "alternatives_without_convergence": "unresolved semantic mixture",
            "convergence_without_alternatives": "prompt invariance, not transition superposition",
            "alternatives_plus_localized_concentration_plus_control_separated_convergence": "observational support for transition superposition",
            "compensation_after_suppression": "causal redundancy; deferred to GW-SUP-2/GW-5",
        },
        "downstream_walk_metric_if_supported": "true semantic-basin survival versus candidate count, bytes and depth",
    }
    if gw0b["bundle_sha256"] != sealed["bundle_sha256"]:
        raise ValueError("GW-0B authority differs from sealed GW-0")
    document["preregistration_sha256"] = canonical_hash(
        document, "preregistration_sha256"
    )
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def validate(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("schema") != PREREG_SCHEMA or document.get("status") != "frozen_pre_readout":
        raise ValueError("not a frozen GW-SUP-1 preregistration")
    expected = canonical_hash(document, "preregistration_sha256")
    if document.get("preregistration_sha256") != expected:
        raise ValueError("preregistration canonical hash mismatch")
    root = path.parent
    for name in ("input_manifest", "input_rows", "sealed_gw0", "gw0b_reconciliation", "gw3af_result"):
        authority = document["authorities"][name]
        target = (root / authority["path"]).resolve()
        expected_file = authority.get("file_sha256", authority.get("sha256"))
        if sha(target) != expected_file:
            raise ValueError(f"{name} authority hash mismatch")
    candidate_authority = document["authorities"]["candidates"]
    candidate_path = (root / candidate_authority["path"]).resolve()
    if sha(candidate_path) != candidate_authority["file_sha256"]:
        raise ValueError("candidate file hash mismatch")
    candidates = json.loads(candidate_path.read_text())
    if candidates.get("schema") != CANDIDATE_SCHEMA:
        raise ValueError("candidate schema mismatch")
    candidate_identity = canonical_hash(candidates, "candidate_identity_sha256")
    if candidate_identity != candidates.get("candidate_identity_sha256") or candidate_identity != candidate_authority["candidate_identity_sha256"]:
        raise ValueError("candidate identity mismatch")
    counts = candidates["counts"]
    expected_counts = {"execution_rows": 426, "unique_semantic_edges": 142, "subjects": 93, "global_candidate_tokens": 126, "primary_multi_destination_subjects": 32, "primary_multi_destination_edges": 81, "primary_multi_destination_rows": 243, "secondary_single_destination_subjects": 61}
    if counts != expected_counts:
        raise ValueError(f"candidate cohort changed: {counts}")
    if document["adjudication"]["resampling_unit"] != "unique semantic edge":
        raise ValueError("prompt rows may not become independent units")
    return {
        "schema": PREREG_SCHEMA,
        "status": "valid",
        "preregistration_sha256": expected,
        "candidate_identity_sha256": candidate_identity,
        "counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--input-manifest", type=Path, required=True)
    build_parser.add_argument("--sealed-manifest", type=Path, required=True)
    build_parser.add_argument("--gw0b-report", type=Path, required=True)
    build_parser.add_argument("--gw3af-report", type=Path, required=True)
    build_parser.add_argument("--candidates", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("preregistration", type=Path)
    args = parser.parse_args()
    result = build(args) if args.command == "build" else validate(args.preregistration)
    print(json.dumps(result if args.command == "validate" else {
        "schema": result["schema"],
        "status": result["status"],
        "preregistration_sha256": result["preregistration_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
