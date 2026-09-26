#!/usr/bin/env python3
"""Validate the frozen GW-TS-1 protocol contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "larql.gwts1.protocol.v1"
STATUS = "frozen_pre_population_capture"
DEPENDENCY_SCHEMA = "larql.gwts1.programme-dependencies.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_identity(document: dict[str, Any], identity_field: str) -> str:
    payload = dict(document)
    payload.pop(identity_field, None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def canonical_hash(document: dict[str, Any]) -> str:
    return canonical_identity(document, "protocol_sha256")


def require_exact_list(actual: Any, expected: list[Any], name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} differs from the frozen contract")


def validate(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA}")
    if document.get("status") != STATUS:
        raise ValueError(f"GW-TS-1 contract is not {STATUS}")
    identity = canonical_hash(document)
    if document.get("protocol_sha256") != identity:
        raise ValueError("GW-TS-1 protocol identity mismatch")

    human_spec = document["human_spec"]
    human_spec_path = (path.parent / human_spec["path"]).resolve()
    if sha256_file(human_spec_path) != human_spec["sha256"]:
        raise ValueError("GW-TS-1 human specification hash mismatch")

    ontology = document["ontology"]
    transition = ontology["transition_identity"]
    forbidden = set(transition["forbidden"])
    realization_fields = {
        "prompt_family_identity",
        "prompt_identity",
        "execution_identity",
        "model_identity",
        "architecture",
        "layer",
        "site",
        "head",
        "feature_address",
        "support_family_identity",
        "path_derived_label",
        "observed_answer",
    }
    if forbidden != realization_fields:
        raise ValueError("TransitionIdentity realization exclusions changed")
    if realization_fields.intersection(transition["required"]):
        raise ValueError("TransitionIdentity contains realization-specific fields")

    event = ontology["support_event"]
    coordinate = set(event["coordinate_fields"])
    measurement = set(event["measurement_fields"])
    if coordinate.intersection(measurement):
        raise ValueError("support coordinate and measurement fields overlap")
    if "signed_contribution" in coordinate or "observation_identity" in coordinate:
        raise ValueError("event measurements leaked into physical identity")
    require_exact_list(event["operators"], ["attention", "ffn"], "operator set")

    extraction = document["path_extraction"]
    if extraction["normalization"] != (
        "discovery-only empirical absolute-contribution distribution per operator "
        "and normalized-depth quartile"
    ):
        raise ValueError("operator-specific normalization is not frozen")
    if extraction["event_quantile_primary"] != 0.75:
        raise ValueError("primary event threshold changed")
    if extraction["minimum_discovery_observations_per_cell"] != 20:
        raise ValueError("discovery calibration floor changed")
    if "no later event" not in extraction["prefix_causality"]:
        raise ValueError("path extraction no longer guarantees prefix causality")
    if extraction["ffn_address_cap_primary"] != 32:
        raise ValueError("primary FFN address cap changed")

    alignment = document["alignment"]["global_alignment"]
    require_exact_list(
        alignment["alignment_inputs"],
        ["operator", "normalized_layer_depth"],
        "alignment inputs",
    )
    if alignment["operator_mismatch_cost"] < 2 * alignment["gap_cost"]:
        raise ValueError("alignment can prefer an operator substitution to two gaps")

    controls = document["controls"]
    required_controls = {
        "same support events unordered",
        "within-path event-order permutation",
        "attention/ffn operator-label swap",
        "strongest single event",
        "marginal-frequency next event",
        "layer-frequency next event",
        "same transition identity with different execution identity",
        "same relation/task stratum with different transition identity",
    }
    if set(controls) != required_controls or len(controls) != len(required_controls):
        raise ValueError("primary control set changed")

    prediction = document["prediction"]
    require_exact_list(prediction["candidate_k"], [1, 2, 4, 8, 16], "candidate ladder")
    gate = document["progression_gate"]
    if gate["bootstrap_block"] != "transition_identity":
        raise ValueError("bootstrap no longer blocks by TransitionIdentity")
    if not gate["requires_cross_operator_transition"] or not gate["all_controls_required"]:
        raise ValueError("prediction gate lost a conjunct")
    if document["successors"]["causality"].startswith("GW-TS-1"):
        raise ValueError("causality leaked into GW-TS-1")

    return {
        "schema": SCHEMA,
        "status": "valid",
        "protocol_sha256": identity,
        "controls": len(controls),
        "candidate_points": len(prediction["candidate_k"]),
    }


def validate_dependencies(path: Path, protocol_identity: str) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("schema") != DEPENDENCY_SCHEMA:
        raise ValueError(f"schema must be {DEPENDENCY_SCHEMA}")
    if document.get("status") != STATUS:
        raise ValueError(f"GW-TS-1 dependencies are not {STATUS}")
    identity = canonical_identity(document, "dependency_sha256")
    if document.get("dependency_sha256") != identity:
        raise ValueError("GW-TS-1 dependency identity mismatch")
    if document.get("gwts1_protocol_sha256") != protocol_identity:
        raise ValueError("dependency freeze is not bound to the GW-TS-1 protocol")

    human_spec = document["human_spec"]
    human_spec_path = (path.parent / human_spec["path"]).resolve()
    if sha256_file(human_spec_path) != human_spec["sha256"]:
        raise ValueError("GW-TS-1 dependency specification hash mismatch")

    require_exact_list(
        document["observational_branch"],
        [
            ["head_obs_1", "attr_1d"],
            ["attr_1d", "run_experiment_subjects"],
            ["run_experiment_subjects", "population_transition_support_capture"],
            [
                "population_transition_support_capture",
                "gw_ts_1_recurrence_and_prediction",
            ],
        ],
        "observational dependency branch",
    )
    require_exact_list(
        document["causal_branch"],
        [
            ["head_obs_1", "attr_1d"],
            ["attr_1d", "attr_1c"],
            ["intervene_1", "attr_1c"],
        ],
        "causal dependency branch",
    )
    non_dependencies = {tuple(edge) for edge in document["explicit_non_dependencies"]}
    required_non_dependencies = {
        ("attr_1c", "population_transition_support_capture"),
        ("attr_1c", "gw_ts_1_recurrence_and_prediction"),
        ("intervene_1", "gw_ts_1_recurrence_and_prediction"),
    }
    if non_dependencies != required_non_dependencies:
        raise ValueError("ATTR-1C/GW-TS-1 independence changed")

    forbidden_inputs = set(document["forbidden_gwts1_inputs"])
    if not {
        "intervention outcome",
        "causal label",
        "ATTR-1C necessity result",
        "ATTR-1C sufficiency result",
    }.issubset(forbidden_inputs):
        raise ValueError("causal evidence may leak into GW-TS-1")

    promotion = document["executability_promotion"]
    require_exact_list(
        promotion["requires_independent_positive_verdicts"],
        ["gw_ts_1_recurrence_and_prediction", "attr_1c_causal_consequence"],
        "executability prerequisites",
    )
    if promotion["single_positive_is_sufficient"]:
        raise ValueError("executability no longer requires both independent claims")

    return {
        "schema": DEPENDENCY_SCHEMA,
        "status": "valid",
        "dependency_sha256": identity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--dependencies", type=Path)
    args = parser.parse_args()
    result = validate(args.protocol)
    dependencies = args.dependencies or args.protocol.with_name(
        "programme-dependencies.json"
    )
    result["programme_dependencies"] = validate_dependencies(
        dependencies, result["protocol_sha256"]
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
