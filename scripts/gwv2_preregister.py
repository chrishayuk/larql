#!/usr/bin/env python3
"""Validate the frozen GW-V2 payload-formation measurement protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from gwv2_population import validate as validate_population


SCHEMA = "larql.gwv2.protocol.v1"
STATUS = "frozen_pre_capture_fit_replay_and_effect_observation"
IDENTITY = "sha256:830c2ceb97fce4f2d03e74e207d04e54257cd5b38e830a93047a86b6bd4fd006"
DEPTHS = [0, 4, 8, 12, 16, 20, 23]
RANKS = [8, 16, 32, 64, 128]
SPLITS = {
    "train": {"subjects": 44, "semantic_edges": 132, "executions": 396},
    "validation": {"subjects": 15, "semantic_edges": 45, "executions": 135},
    "test": {"subjects": 15, "semantic_edges": 45, "executions": 135},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_hash(document: dict[str, Any]) -> str:
    body = dict(document)
    body.pop("protocol_sha256", None)
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def exact(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} differs from the frozen GW-V2 contract")


def resolved(protocol_path: Path, descriptor: dict[str, Any]) -> Path:
    return (protocol_path.parent / descriptor["path"]).resolve()


def validate(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    exact(document.get("schema"), SCHEMA, "schema")
    exact(document.get("status"), STATUS, "status")
    exact(document.get("protocol_sha256"), IDENTITY, "protocol identity")
    exact(canonical_hash(document), IDENTITY, "canonical protocol identity")
    exact(
        document.get("central_rule"),
        "Reconstruction accuracy is diagnostic; preservation of the inherited causal effect is the claim metric.",
        "central rule",
    )

    human = document["human_spec"]
    exact(sha256(resolved(path, human)), human["sha256"], "human spec hash")
    for name, descriptor in document["authorities"].items():
        if isinstance(descriptor, dict) and {"path", "sha256"} <= descriptor.keys():
            exact(sha256(resolved(path, descriptor)), descriptor["sha256"], f"{name} file hash")

    population_descriptor = document["authorities"]["population"]
    population = validate_population(resolved(path, population_descriptor))
    exact(population["population_sha256"], population_descriptor["identity"], "population identity")
    exact(population["rows"]["sha256"], document["authorities"]["input_rows"]["sha256"], "row authority")
    exact(population["execution"]["model_prompts_executed"], 0, "pre-execution population")
    exact(population["execution"]["fitting_performed"], False, "pre-fit population")

    cohort = document["population"]
    exact(cohort["prior_gw0_subject_identity_overlap"], 0, "fresh subject overlap")
    exact((cohort["subjects"], cohort["semantic_edges"], cohort["executions"]), (74, 222, 666), "population counts")
    exact(cohort["splits"], SPLITS, "subject-level splits")
    exact(cohort["candidate_readout_tokens"], 142, "candidate readout size")
    for split in ("validation_use", "test_use"):
        value = cohort[split]
        if "no tuning, ranking, selection or grid change" not in value:
            raise ValueError(f"{split} permits optimization semantics")

    interface = document["frozen_interface"]
    exact((interface["layer"], interface["head"], interface["head_numbering"]), (24, 1, "zero_based"), "frozen operator")
    for phrase, field in (
        ("natural final-position", "query"),
        ("natural K-row L2 norm", "keys"),
        ("sole varied component", "subject_values"),
        ("outside subject_entity", "other_values"),
        ("seven non-H1", "other_heads"),
    ):
        if phrase not in interface[field]:
            raise ValueError(f"frozen interface changed: {field}")
    exact(interface["efficiency_claim_allowed"], False, "V2 efficiency claim")
    dynamic = set(interface["dynamic_natural_inputs"])
    for item in (
        "natural L24 carrier-before",
        "natural contributions from the seven non-H1 heads",
        "every target natural source K-row norm",
    ):
        if item not in dynamic:
            raise ValueError(f"hidden natural input disappeared: {item}")

    grid = document["grid"]
    exact(grid["source_depths"], DEPTHS, "source-depth grid")
    exact(grid["ranks"], RANKS, "rank grid")
    exact(grid["cells"], len(DEPTHS) * len(RANKS), "grid cell count")
    exact(grid["adaptive_expansion"], False, "adaptive grid expansion")
    exact(grid["selection"], "none", "grid selection")
    exact(grid["winner"], "undefined", "grid winner")
    exact(grid["pareto_frontier"], "forbidden", "Pareto semantics")

    predictor = document["predictor"]
    exact(predictor["fit_population"], "train subjects only", "fit population")
    exact(predictor["fit_precision"], "f64", "fit precision")
    exact(predictor["inference_precision"], "f32", "inference precision")
    exact(predictor["train_prediction"], "leave-one-subject-out for descriptive train reporting only", "train prediction")
    exact(set(predictor["identical_across_cells_except"]), {"source depth", "rank"}, "declared factors")
    forbidden_predictor = set(predictor["forbidden_inputs"])
    for item in (
        "validation or test V tensors during fitting",
        "candidate readout effects",
        "cross-depth ensembles",
        "post-hoc scale or norm matching",
    ):
        if item not in forbidden_predictor:
            raise ValueError(f"predictor prohibition disappeared: {item}")

    estimand = document["estimand"]
    if "same_fact" not in estimand["effect"] or "matched_control" not in estimand["effect"]:
        raise ValueError("control-adjusted inherited effect changed")
    if "E(candidate depth,rank)/E(exact natural V)" not in estimand["retention"]:
        raise ValueError("causal retention denominator changed")
    if "diagnostic only" not in estimand["reconstruction"]:
        raise ValueError("reconstruction became a claim metric")

    analysis = document["analysis"]
    exact(analysis["all_cells_reported"], True, "complete surface")
    exact(analysis["splits_reported_separately"], ["validation", "test"], "held-out reporting")
    bootstrap = analysis["bootstrap"]
    exact(bootstrap["replicates"], 10000, "bootstrap replicates")
    exact(bootstrap["unit"], "subject country identity", "bootstrap unit")
    exact(bootstrap["simultaneous_method"], "one-sided studentized max-t lower band", "simultaneous inference")
    if "all 35 cells" not in bootstrap["retention_family"] or "all 35 cells" not in bootstrap["terminal_family"]:
        raise ValueError("simultaneous families no longer cover the full grid")
    exact(
        set(analysis["no_selection_semantics"]),
        {
            "no primary cell",
            "no best depth",
            "no best rank",
            "no heldout tuning",
            "no deployment candidate",
            "no Pareto frontier",
        },
        "no-selection semantics",
    )

    gate = document["cell_gate"]
    exact(gate["applies_independently_to"], ["validation", "test"], "held-out gates")
    exact(gate["raw_retention_estimate_min"], 0.8, "raw retention gate")
    exact(gate["zscored_retention_estimate_min"], 0.8, "z-scored retention gate")
    exact(gate["raw_retention_simultaneous_lower_95_min"], 0.5, "raw simultaneous gate")
    exact(gate["zscored_retention_simultaneous_lower_95_min"], 0.5, "z-scored simultaneous gate")
    for field in (
        "terminal_raw_effect_simultaneous_lower_95_min_exclusive",
        "terminal_zscored_effect_simultaneous_lower_95_min_exclusive",
    ):
        exact(gate[field], 0.0, field)

    frontier = document["early_frontier"]
    exact(frontier["shape"], "one contiguous 2x2 tile in the frozen depth x rank grid", "frontier shape")
    if "<= 12" not in frontier["early_depth_rule"]:
        raise ValueError("early frontier is no longer bounded at L12")
    if "all four cells" not in frontier["declaration"]:
        raise ValueError("an isolated cell can establish a frontier")
    if "cannot establish" not in frontier["isolated_cell"]:
        raise ValueError("isolated-cell safeguard disappeared")

    cost = document["cost_model"]
    if "complete canonical layers 0..d-1" not in cost["source_prefix_boundary"]:
        raise ValueError("source-prefix cost hides preceding layers")
    exact(cost["prefix_is_not_free"], True, "prefix cost")
    if "does not claim avoided execution" not in cost["actual_v2_replay"]:
        raise ValueError("V2 replay makes an efficiency claim")
    if "never a gate" not in cost["efficiency_visualization_use"]:
        raise ValueError("derived efficiency visualization became a selection metric")
    exact(cost["run_hygiene"]["void_baseline_drift_fraction"], 0.01, "latency drift gate")
    exact(cost["run_hygiene"]["peer_exclusivity_handshake"], True, "peer handshake")

    refusals = set(document["hard_refusals"])
    for refusal in (
        "adaptive source-depth or rank expansion",
        "candidate ranking or selection",
        "validation or test fitting",
        "changed natural carrier-before or non-H1 treatment",
        "efficiency claim from the isolation replay",
    ):
        if refusal not in refusals:
            raise ValueError(f"hard refusal disappeared: {refusal}")

    return {
        "schema": SCHEMA,
        "status": "valid",
        "protocol_sha256": IDENTITY,
        "population_sha256": population["population_sha256"],
        "cells": len(DEPTHS) * len(RANKS),
        "source_depths": DEPTHS,
        "ranks": RANKS,
        "fresh_subjects": 74,
        "heldout_selection": False,
        "model_prompts_executed": 0,
        "fitting_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.protocol), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
