#!/usr/bin/env python3
"""Build or validate the frozen GW-HEAD-1 causal-head contract."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from gwsup1_preregister import canonical_hash, jsonl, sha

SCHEMA = "larql.gwhead1.preregistration.v1"
SEED = 1_398_104_331
LAYER = 24
TRAIN_RETENTION = 0.80
HELD_OUT_RETENTION_LOWER = 0.50
SMALL_HEAD_LIMIT = 4
EXPECTED_HEADS = 8


def relative(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve())


def authority(path: Path, root: Path) -> dict[str, Any]:
    return {"path": relative(path, root), "sha256": sha(path)}


def text_component(system_graph: dict[str, Any]) -> dict[str, Any]:
    matches = [
        component
        for component in system_graph["components"]
        if component.get("role") == "primary_text"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one primary_text component, found {len(matches)}")
    return matches[0]


def geometry(system_graph: dict[str, Any]) -> dict[str, Any]:
    component = text_component(system_graph)
    attention = component["execution"]["attention"]
    norm = component["execution"]["norm"]
    layer = component["attention"][LAYER]
    result = {
        "component": component["id"],
        "layer": LAYER,
        "operator": layer["operator"],
        "span": layer["span"],
        "window": layer.get("window"),
        "num_q_heads": attention["num_q_heads"],
        "num_kv_heads": attention["num_kv_heads"],
        "head_dim": attention["head_dim"],
        "hidden_size": component["hidden_size"],
        "norm_placement": norm["placement"],
        "post_attention_norm": norm["post"],
    }
    if result != {
        "component": "target",
        "layer": 24,
        "operator": "softmax",
        "span": "sliding",
        "window": 1024,
        "num_q_heads": 8,
        "num_kv_heads": 4,
        "head_dim": 256,
        "hidden_size": 2560,
        "norm_placement": "pre_post",
        "post_attention_norm": {
            "kind": "rms_norm",
            "eps": 9.999999974752427e-7,
            "weight_offset": 1.0,
        },
    }:
        raise ValueError(f"unexpected L24 execution geometry: {result}")
    return result


def require_gwconv1(adjudication: dict[str, Any]) -> None:
    if adjudication.get("schema") != "larql.gwconv1.adjudication.v1":
        raise ValueError("not a GW-CONV-1 adjudication")
    if not adjudication.get("gate", {}).get("conditions", {}).get("full_support"):
        raise ValueError("GW-CONV-1 full support did not pass")
    selection = adjudication["selection"]
    expected = {"index": 48, "layer": LAYER, "site": "attention"}
    if selection["global_site"] != expected:
        raise ValueError("GW-CONV-1 global site is not L24 attention")
    if set(selection["relation_sites"]) != {
        "capital",
        "currency",
        "language",
        "hypernym",
    } or any(site != expected for site in selection["relation_sites"].values()):
        raise ValueError("GW-CONV-1 relation sites are not all L24 attention")


def build(args: argparse.Namespace) -> dict[str, Any]:
    input_manifest = json.loads(args.input_manifest.read_text())
    gwconv_prereg = json.loads(args.gwconv_preregistration.read_text())
    gwconv_adjudication = json.loads(args.gwconv_adjudication.read_text())
    candidates = json.loads(args.candidates.read_text())
    system_graph = json.loads(args.system_graph.read_text())
    require_gwconv1(gwconv_adjudication)
    site_geometry = geometry(system_graph)

    if sha(args.system_graph) != input_manifest["model"]["system_graph_sha256"]:
        raise ValueError("system graph differs from the frozen GW-0 authority")
    if gwconv_prereg.get("schema") != "larql.gwconv1.preregistration.v1":
        raise ValueError("not a GW-CONV-1 preregistration")
    if candidates.get("schema") != "larql.gwsup1.candidates.v1":
        raise ValueError("not the frozen candidate vocabulary")

    input_rows = args.input_manifest.parent / input_manifest["rows"]["path"]
    rows = jsonl(input_rows)
    triples: dict[tuple[str, str, str], str] = {}
    relation_splits: Counter[tuple[str, str]] = Counter()
    prompt_families: Counter[str] = Counter()
    for row in rows:
        semantic = row["semantic_edge"]
        key = (semantic["subject"], semantic["relation"], semantic["target"])
        previous = triples.setdefault(key, row["split"])
        if previous != row["split"]:
            raise ValueError(f"semantic edge crosses splits: {key}")
        prompt_families[semantic["prompt_semantic_family"]] += 1
    for (_, relation, _), split in triples.items():
        relation_splits[(relation, split)] += 1
    splits = Counter(triples.values())
    if splits != {"train": 85, "validation": 29, "test": 28}:
        raise ValueError(f"unexpected semantic-edge split: {splits}")

    root = args.output.parent
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "frozen_pre_execution",
        "frozen_date": "2026-09-21",
        "preregistration_sha256": None,
        "question": (
            "does a train-selected frozen subset of L24 attention heads causally "
            "account for the GW-CONV-1 control-adjusted semantic-alignment effect?"
        ),
        "claim": (
            "a frozen subset of L24 attention heads is necessary for and sufficient "
            "to recreate a substantial fraction of the control-adjusted semantic alignment"
        ),
        "claim_boundary": (
            "causal head attribution at the already-frozen L24 attention site only; "
            "not source-token or source-key attribution, not correctness, not a compact "
            "access key, and not authorization for partial-execution or WALK claims"
        ),
        "causal_ladder": {
            "current": "GW-HEAD-1: which L24 operators matter",
            "excluded_successor": "GW-KEY-1: what information the frozen operators read",
            "later_successor": "GW-READ-1: whether the frozen key-to-head path is a cheap read primitive",
        },
        "authorities": {
            "input_manifest": authority(args.input_manifest, root),
            "input_rows": {
                **authority(input_rows, root),
                "declared_sha256": input_manifest["rows"]["sha256"],
            },
            "gwconv1_preregistration": {
                **authority(args.gwconv_preregistration, root),
                "identity_sha256": gwconv_prereg["preregistration_sha256"],
            },
            "gwconv1_adjudication": {
                **authority(args.gwconv_adjudication, root),
                "identity_sha256": gwconv_adjudication["adjudication_sha256"],
                "verdict": gwconv_adjudication["gate"]["verdict"],
            },
            "candidates": {
                **authority(args.candidates, root),
                "identity_sha256": candidates["candidate_identity_sha256"],
            },
            "system_graph": {
                "path": input_manifest["model"]["container_root"] + "/system_graph.json",
                "sha256": input_manifest["model"]["system_graph_sha256"],
            },
            "container_identity": gwconv_prereg["authorities"]["container_identity"],
            "plan_sha256": gwconv_prereg["authorities"]["plan_sha256"],
            "readout_basis": gwconv_prereg["authorities"]["readout_basis"],
        },
        "cohort": {
            "semantic_edges": len(triples),
            "execution_rows": len(rows),
            "prompt_rows_per_edge": 3,
            "splits": dict(sorted(splits.items())),
            "relation_by_split": {
                relation: {
                    split: relation_splits[(relation, split)]
                    for split in ("train", "validation", "test")
                }
                for relation in ("capital", "currency", "language", "hypernym")
            },
            "prompt_semantic_families": dict(sorted(prompt_families.items())),
            "statistical_unit": "unique (subject, relation, target) semantic edge",
            "prompt_reexecution": True,
        },
        "site": {
            **site_geometry,
            "site_index": 48,
            "head_ids": list(range(EXPECTED_HEADS)),
            "head_universe_size": EXPECTED_HEADS,
            "subset_universe_size": 2**EXPECTED_HEADS,
            "intervention_position": (
                "the final prompt/capture position only; prefix positions and every other layer/site remain natural"
            ),
        },
        "exact_decomposition": {
            "head_value": "the actual production weighted-V output for one L24 query head, before W_O",
            "head_contribution": (
                "that head value through its matching column slice of the effective prepared-production W_O"
            ),
            "raw_reconstruction": (
                "sum all eight post-W_O head contributions and compare with the actual pre-post-attention-norm output"
            ),
            "norm_reconstruction": (
                "apply the declared L24 post-attention RMSNorm and residual-delta scale, then compare with the actual applied attention delta"
            ),
            "weight_authority": "the effective prepared production Q8 W_O, not widened stored checkpoint rows",
            "raw_max_relative_l2": 1e-5,
            "applied_max_relative_l2": 1e-5,
            "identity_noop_logits": "bit-identical at every prompt position",
            "failure_policy": "any missing head, duplicate head, non-finite value, geometry drift, parity failure or reconstruction failure aborts before subset search",
            "semantic_claim": False,
        },
        "interventions": {
            "primary": "contribution-norm-matched replacement/restoration before W_O",
            "reference_bank": {
                "grouping": "relation x prompt_semantic_family x head_id",
                "direction": "mean production head-value direction from train rows only",
                "train_candidate_evaluation": "leave-one-semantic-edge-out reference bank",
                "held_out_evaluation": "single bank frozen from every train edge and no held-out row",
                "scaling": (
                    "scale the bank vector so its effective W_O-slice contribution L2 equals the natural row/head contribution L2"
                ),
                "zero_or_nonfinite_policy": "refuse; do not substitute zero or another grouping",
            },
            "arms": {
                "I_full": "all eight natural head values",
                "I_ref": "all eight heads replaced by their contribution-norm-matched train references",
                "I_S": "heads in S natural/restored; every head outside S replaced",
                "I_full_minus_S": "heads in S replaced; every head outside S natural",
                "I_identity": "run the intervention machinery while replacing every selected value with its exact original bytes",
            },
            "secondary_arms": [
                "zero S with every non-S head natural",
                "cardinality-matched subsets",
                "nearest-total-contribution-norm subsets at the same cardinality",
                "aggregate-norm-preserving sham replacement",
            ],
            "train_search_execution": (
                "capture each natural row's L24 entering carrier and eight production head values once; for every candidate subset, compose natural/reference head values before W_O, then execute the effective W_O, declared post-attention norm, residual write and frozen selected-row readout; this is the exact local operator path, not isolated-vector projection"
            ),
            "frozen_arm_execution": (
                "reexecute I_full, I_ref, I_S, I_full_minus_S and I_identity by intervening inside canonical execution at L24 before W_O; capture the proximal carrier, then execute the L24 FFN and every later layer normally for terminal reporting"
            ),
            "local_replay_parity": (
                "on every frozen arm, local subset-search replay and intrusive canonical execution must agree on the proximal carrier and selected logits within the exact-decomposition tolerances"
            ),
            "pure_retain_only": "secondary only; all-other-heads-zero is not the primary sufficiency intervention",
        },
        "measurements": {
            "candidate_raw": gwconv_prereg["measurements"]["candidate_raw"],
            "candidate_zscore": gwconv_prereg["measurements"]["candidate_zscore"],
            "carrier": gwconv_prereg["measurements"]["carrier"],
            "matched_control": gwconv_prereg["measurements"]["control"],
            "proximal": (
                "the recomputed real carrier immediately after the intervened L24 post-norm residual write, read through the unchanged frozen candidate basis"
            ),
            "terminal": (
                "the actual next-token logits after the remainder of canonical execution; mandatory report and qualification, not a substitute selection surface"
            ),
            "E": (
                "for each arm I and proximal metric, E(I) = same-fact before-to-after gain under I minus the frozen matched-control gain under I; signs normalized so larger means more alignment"
            ),
            "necessity": "N(S) = E(I_full) - E(I_full_minus_S)",
            "sufficiency": (
                "R(S) = (E(I_S) - E(I_ref)) / (E(I_full) - E(I_ref)); refuse a non-positive or numerically unstable denominator"
            ),
            "carrier_role": (
                "qualification only: report absolute same-fact and adjusted carrier effects; it cannot rescue either semantic readout"
            ),
        },
        "selection": {
            "discovery_split": "train only",
            "search": (
                "exhaustive direct execution of all 256 subsets through the exact L24 W_O -> post-norm -> residual -> selected-readout path using once-captured natural head values; no singleton-score summation, isolated projection or top-k approximation"
            ),
            "eligible": (
                "positive train E(I_full) and R(S) >= 0.80 independently for candidate_raw and candidate_zscore under the primary I_S intervention"
            ),
            "train_retention_fraction": TRAIN_RETENTION,
            "global_subset": "select from all 85 train edges only",
            "relation_subsets": (
                "independently select within each relation's train stratum; each identity freezes before any validation/test execution"
            ),
            "tie_break": [
                "minimum cardinality",
                "higher minimum of raw and z-scored retained fractions",
                "lower total natural contribution norm",
                "lexicographically ascending head-ID tuple",
            ],
            "small_definition": f"cardinality <= {SMALL_HEAD_LIMIT} of {EXPECTED_HEADS}; fixed before search",
            "validation_or_test_may_not": [
                "select, add, drop or reorder a head",
                "change the reference bank",
                "change an intervention arm, retention fraction or tie-break",
                "choose global versus relation-specific reporting",
            ],
        },
        "statistics": {
            "interval": "10000-sample percentile cluster bootstrap over semantic edges",
            "global_stratification": "relation-stratified",
            "relation_specific": "within-relation edge bootstrap",
            "seed": SEED,
            "validation_and_test_reported_separately": True,
            "train_search_is_discovery_only": True,
            "candidate_metrics_are_conjunctive": ["candidate_raw", "candidate_zscore"],
        },
        "adjudication": {
            "structural": "exact_decomposition passes before semantic adjudication",
            "replication": (
                "I_full and I_identity reproduce the unintervened GW-CONV-1 proximal measurements within the frozen numerical tolerance; I_identity logits are bit-identical"
            ),
            "necessity": (
                "for the frozen global S, lower 95% bounds of N(S) are above zero for raw and z-scored candidate alignment on validation and test"
            ),
            "sufficiency": (
                f"for the same S, lower 95% bounds of R(S) are at least {HELD_OUT_RETENTION_LOWER:.2f} for raw and z-scored candidate alignment on validation and test"
            ),
            "causal_head_subset": "structural AND replication AND necessity AND sufficiency",
            "concentrated_causal_head_subset": (
                f"causal_head_subset AND cardinality <= {SMALL_HEAD_LIMIT}"
            ),
            "terminal_transport": (
                "report necessity and sufficiency analogues on actual terminal candidate logits; adjudicate separately and do not use them to move S"
            ),
            "shared_heads_not_required": True,
            "per_relation": (
                "report each relation before any shared-mechanism interpretation; relation-specific subsets are separately frozen descriptive causal routes"
            ),
            "random_and_norm_matched_controls": (
                "report the frozen subset against all same-cardinality subsets and the nearest-norm subset; these qualify specificity but cannot replace held-out necessity or sufficiency"
            ),
            "no_post_hoc_head_or_intervention": True,
        },
        "interpretation_contract": {
            "necessary_not_sufficient": "bottleneck or enabling component, not the complete operation",
            "sufficient_not_necessary": "one redundant realization, not a unique causal operator",
            "causal_not_concentrated": "L24-localized but internally distributed",
            "shared_global_subset": "candidate generic semantic-alignment mechanism",
            "stable_relation_specific_subsets": "candidate common routing stage with specialized operators",
            "small_heterogeneous_subsets": "localized conditional mechanism",
            "terminal_failure": "proximal alignment mechanism not shown to control later answer formation",
            "correctness": "never inferred; convergence toward a wrong continuation remains allowed",
            "source_key": "never inferred; head sufficiency is not source-key sufficiency",
        },
    }
    document["preregistration_sha256"] = canonical_hash(
        document, "preregistration_sha256"
    )
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def validate(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("schema") != SCHEMA or document.get("status") != "frozen_pre_execution":
        raise ValueError("not a frozen GW-HEAD-1 preregistration")
    identity = canonical_hash(document, "preregistration_sha256")
    if identity != document.get("preregistration_sha256"):
        raise ValueError("GW-HEAD-1 canonical identity mismatch")
    root = path.parent
    for name in (
        "input_manifest",
        "input_rows",
        "gwconv1_preregistration",
        "gwconv1_adjudication",
        "candidates",
    ):
        item = document["authorities"][name]
        if sha((root / item["path"]).resolve()) != item["sha256"]:
            raise ValueError(f"{name} authority changed")
    if document["site"]["head_ids"] != list(range(EXPECTED_HEADS)):
        raise ValueError("GW-HEAD-1 head universe changed")
    if document["selection"]["train_retention_fraction"] != TRAIN_RETENTION:
        raise ValueError("GW-HEAD-1 train retention changed")
    if document["cohort"]["splits"] != {"train": 85, "validation": 29, "test": 28}:
        raise ValueError("GW-HEAD-1 split changed")
    return {
        "schema": SCHEMA,
        "status": "valid",
        "preregistration_sha256": identity,
        "site": document["site"],
        "cohort": document["cohort"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--input-manifest", type=Path, required=True)
    build_parser.add_argument("--gwconv-preregistration", type=Path, required=True)
    build_parser.add_argument("--gwconv-adjudication", type=Path, required=True)
    build_parser.add_argument("--candidates", type=Path, required=True)
    build_parser.add_argument("--system-graph", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("preregistration", type=Path)
    args = parser.parse_args()
    result = build(args) if args.command == "build" else validate(args.preregistration)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
