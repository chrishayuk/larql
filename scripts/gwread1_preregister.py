#!/usr/bin/env python3
"""Validate the frozen GW-READ-1 systems protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "larql.gwread1.protocol.v2"
STATUS = "amended_frozen_pre_candidate_capture_fit_and_execution"
IDENTITY = "sha256:92947e60d1e831183346b2336053e1d10990c7bfb0c65f3fe31f4858b0cfc0ee"
LAYERS = [0, 4, 8, 12, 16, 20, 23]
RANKS = [8, 16, 32, 64, 128]
RUNGS = {"READ-1Q", "READ-1K", "READ-1V", "READ-1E"}
COST_FIELDS = {
    "offline build time",
    "offline bytes read",
    "offline bytes written",
    "dynamic construction time",
    "lookup/use time",
    "bytes touched",
    "resident bytes",
    "incremental resident bytes",
    "matvec count",
    "projection count",
    "dot-product count",
    "matvec-equivalent FLOPs",
    "routing/injection overhead",
    "downstream continuation cost",
    "end-to-end latency",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_hash(document: dict[str, Any]) -> str:
    payload = dict(document)
    payload.pop("protocol_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def exact(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} differs from the frozen GW-READ-1 contract")


def resolved(protocol: Path, descriptor: dict[str, Any]) -> Path:
    return (protocol.parent / descriptor["path"]).resolve()


def validate(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    exact(document.get("schema"), SCHEMA, "schema")
    exact(document.get("status"), STATUS, "status")
    exact(document.get("protocol_sha256"), IDENTITY, "frozen identity")
    exact(canonical_hash(document), IDENTITY, "canonical identity")
    exact(
        document.get("central_rule"),
        "GW-KEY-1 found a compact causal interface. GW-READ-1 prices that interface honestly.",
        "central rule",
    )

    human = document["human_spec"]
    exact(sha256_file(resolved(path, human)), human["sha256"], "human spec hash")
    for name, descriptor in document["authorities"].items():
        if isinstance(descriptor, dict) and {"path", "sha256"} <= descriptor.keys():
            exact(sha256_file(resolved(path, descriptor)), descriptor["sha256"], f"{name} hash")

    amendment = document["amends"]
    if "before any candidate" not in amendment["timing"]:
        raise ValueError("the protocol amendment is not prospectively timed")
    for reason in (
        "move all candidate and hyperparameter selection from validation to train",
        "make target-natural reference norm reads explicit dynamic inputs",
        "freeze minimum shared Q/V execution-frontier accounting",
    ):
        if reason not in amendment["reasons"]:
            raise ValueError(f"amendment lost reason: {reason}")

    interface = document["frozen_interface"]
    exact((interface["layer"], interface["head"]), (24, 1), "frozen L24H1")
    dynamic = set(interface["dynamic_inputs_in_positive_control"])
    for required in (
        "every target natural source K-row norm",
        "every target natural non-entity V-row norm",
    ):
        if required not in dynamic:
            raise ValueError(f"hidden dynamic input disappeared: {required}")

    population = document["population"]
    exact(population["splits"], {"train": 255, "validation": 87, "test": 84}, "splits")
    exact(population["selection_rule"], "train only", "selection population")
    if "already-sealed" not in population["heldout_rule"]:
        raise ValueError("held-out rows may select a candidate")

    predictor = document["common_predictor"]
    exact(predictor["ranks"], RANKS, "predictor ranks")
    exact(predictor["train_prediction"], "leave-one-semantic-edge-out", "train prediction")
    exact(predictor["heldout_artifact"], "one fit on all train rows after selection", "heldout fit")
    for forbidden in ("validation or test fitting", "candidate causal effects", "cross-arm or joint Q/K/V fitting"):
        if forbidden not in predictor["forbidden_supervision"]:
            raise ValueError(f"predictor prohibition disappeared: {forbidden}")

    exact({key for key in document if key.startswith("READ-1")}, RUNGS, "rungs")
    exact(document["READ-1Q"]["source_layers"], LAYERS, "Q layers")
    exact(document["READ-1V"]["source_layers"], LAYERS, "V layers")
    if "natural rows and norms" not in document["READ-1Q"]["cost_warning"]:
        raise ValueError("Q arm hides exact K/V target-natural inputs")
    if "no target-natural norm" not in document["READ-1K"]["candidate_scale"]:
        raise ValueError("K candidate may consume a target-natural norm")
    if "static norms" not in document["READ-1V"]["held_fixed"][1] or "static norms" not in document["READ-1V"]["held_fixed"][2]:
        raise ValueError("V arm hides dynamic complement norms")
    v_families = document["READ-1V"]["families"]
    labels = {item["cost_class"] for item in v_families}
    exact(labels, {"diagnostic", "cached/materialized", "per-query computed"}, "V cost labels")
    diagnostic = next(item for item in v_families if item["cost_class"] == "diagnostic")
    exact(diagnostic["eligible_for_E"], False, "diagnostic E eligibility")

    selection = document["selection"]
    exact(selection["population"], "train only", "selection population")
    if "before any validation or test" not in selection["seal_timing"]:
        raise ValueError("selection seal is not pre-heldout")
    exact(
        selection["primary_order"][:2],
        ["least dynamic matvec-equivalent FLOPs", "least dynamic bytes touched"],
        "primary cost ordering",
    )

    gate = document["heldout_component_gate"]
    exact(gate["splits"], ["validation", "test"], "held-out component splits")
    exact(gate["raw_retention_estimate_min"], 0.8, "raw retention")
    exact(gate["zscored_retention_estimate_min"], 0.8, "z retention")
    exact(gate["raw_retention_lower_95_min"], 0.5, "raw lower CI")
    exact(gate["zscored_retention_lower_95_min"], 0.5, "z lower CI")

    end_to_end = document["READ-1E"]
    exact(end_to_end["purpose"], "composition only", "E purpose")
    if "all-candidate cell alone" not in end_to_end["diagnostic_factorial"]:
        raise ValueError("factorial diagnostics may select the E result")
    exact(end_to_end["no_hidden_inputs"], True, "hidden input gate")
    exact(end_to_end["dynamic_bytes_ratio_max"], 0.25, "E bytes gate")
    exact(end_to_end["matvec_equivalent_flops_ratio_max"], 0.25, "E FLOP gate")

    cost = document["cost_model"]
    exact(set(cost["fields"]), COST_FIELDS, "cost fields")
    exact(cost["amortization_horizons"], [1, 1000, 1000000], "amortization horizons")
    frontier = cost["shared_frontier"]
    if "max(required Q layer, required V layer)" not in frontier["original_prompt"]:
        raise ValueError("Q/V original-prompt work is double-counted")
    if "cannot share" not in frontier["entity_only"]:
        raise ValueError("entity-only execution is incorrectly shared")
    exact(cost["run_hygiene"]["void_baseline_drift_fraction"], 0.01, "drift gate")
    exact(cost["run_hygiene"]["peer_exclusivity_handshake"], True, "peer handshake")

    refusals = set(document["hard_refusals"])
    for refusal in ("heldout fitting or selection", "undeclared natural-state read", "target-natural norm in a static candidate", "heldout cache donor"):
        if refusal not in refusals:
            raise ValueError(f"hard refusal disappeared: {refusal}")

    return {
        "schema": SCHEMA,
        "status": "valid",
        "protocol_sha256": IDENTITY,
        "rungs": sorted(RUNGS),
        "source_layers": LAYERS,
        "ranks": RANKS,
        "heldout_selection": False,
        "dynamic_norms_exposed": True,
        "shared_frontier_frozen": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.protocol), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
