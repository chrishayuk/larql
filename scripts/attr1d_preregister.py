#!/usr/bin/env python3
"""Validate the frozen ATTR-1D descriptive-support contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "larql.attr1d.contract.v1"
STATUS = "frozen_pre_implementation_and_witness"
GWTS1_ID = "sha256:d873d238d99406dc8a549681aa746e750a9d4b802e24ff21a8eb26d152a97d55"
DEPENDENCY_ID = "sha256:43d8128d1d96ced883ed3ae8f6b3334ea03ff0bd91686a8e09970a75785a2bb4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_hash(document: dict[str, Any]) -> str:
    payload = dict(document)
    payload.pop("attr1d_contract_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def require_exact(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} differs from the frozen contract")


def validate(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA}")
    if document.get("status") != STATUS:
        raise ValueError(f"ATTR-1D contract is not {STATUS}")
    identity = canonical_hash(document)
    if document.get("attr1d_contract_sha256") != identity:
        raise ValueError("ATTR-1D contract identity mismatch")

    human_spec = document["human_spec"]
    human_spec_path = (path.parent / human_spec["path"]).resolve()
    if sha256_file(human_spec_path) != human_spec["sha256"]:
        raise ValueError("ATTR-1D human specification hash mismatch")

    authorities = document["authorities"]
    require_exact(authorities["gwts1_protocol_sha256"], GWTS1_ID, "GW-TS-1 binding")
    require_exact(
        authorities["programme_dependency_sha256"],
        DEPENDENCY_ID,
        "programme dependency binding",
    )

    inputs = document["input_authority"]
    require_exact(
        inputs["forbidden_shortcut"],
        "attention weight is never substituted for source contribution",
        "source-attribution boundary",
    )
    if "lossless same-run source-evidence sidecar" not in inputs[
        "source_level_additional_requirement"
    ]:
        raise ValueError("source attribution no longer requires lossless same-run evidence")
    if "refuse source contribution" not in inputs["stats_only_rule"]:
        raise ValueError("stats-only records may be promoted to source attribution")

    coordinate = document["support_coordinate"]
    require_exact(
        coordinate["coordinate_kinds"],
        ["site", "bias", "head", "source"],
        "coordinate kinds",
    )
    require_exact(coordinate["operator"], "attention", "ATTR-1D operator")
    semantic_or_measured = {
        "transition_identity",
        "subject",
        "relation",
        "semantic_target",
        "answer",
        "prompt_family",
        "execution_identity",
        "provenance_fingerprint",
        "prepared_image_fingerprint",
        "observation_receipt_digest",
        "event_sequence",
        "contribution",
        "projection",
        "rank",
        "outcome",
        "support_family",
        "intervention_identity",
        "causal_label",
        "mechanism_label",
    }
    if set(coordinate["forbidden_fields"]) != semantic_or_measured:
        raise ValueError("coordinate exclusion set changed")
    required_coordinate = set(coordinate["required_fields"])
    conditional_coordinate = {
        field
        for fields in coordinate["conditional_fields"].values()
        for field in fields
    }
    all_coordinate_fields = required_coordinate | conditional_coordinate
    execution_fields = {
        "execution_identity",
        "provenance_fingerprint",
        "prepared_image_fingerprint",
        "observation_receipt_digest",
        "event_sequence",
    }
    if execution_fields.intersection(all_coordinate_fields):
        raise ValueError("backend realization leaked into stable coordinate identity")
    if (semantic_or_measured - execution_fields).intersection(all_coordinate_fields):
        raise ValueError("semantic, measured or causal data leaked into coordinate identity")
    require_exact(
        coordinate["required_fields"],
        [
            "schema",
            "model_system_identity",
            "logical_plan_identity",
            "component_identity",
            "input_layout_identity",
            "position",
            "operator",
            "layer",
            "site",
            "coordinate_kind",
        ],
        "stable coordinate fields",
    )
    require_exact(
        coordinate["conditional_fields"]["source"],
        [
            "query_head",
            "kv_head",
            "source_start",
            "source_end",
            "source_role",
            "role_map_identity",
        ],
        "source coordinate",
    )

    observation = document["support_observation"]
    require_exact(
        observation["required_fields"],
        [
            "schema",
            "support_coordinate_id",
            "execution_identity",
            "provenance_fingerprint",
            "prepared_image_fingerprint",
            "observation_receipt_digest",
            "event_sequence",
        ],
        "support-observation fields",
    )
    if "distinct observation IDs" not in observation["backend_rule"]:
        raise ValueError("backend realization leaked into stable coordinate identity")

    record = document["descriptive_record"]
    record_fields = set(record["required_fields"]) | set(record["optional_fields"])
    forbidden_record_fields = {
        "intervention_identity",
        "observed_effect",
        "control_effect",
        "causal_verdict",
        "necessity",
        "sufficiency",
    }
    if forbidden_record_fields.intersection(record_fields):
        raise ValueError("causal evidence leaked into DescriptiveSupportRecord")
    if not {"support_observation_id", "support_observation"}.issubset(record_fields):
        raise ValueError("descriptive record lost its execution-specific observation binding")

    algebra = document["contribution_algebra"]
    if "separate bias coordinate" not in algebra["once_only_bias"]:
        raise ValueError("once-only output bias may be assigned to a head")
    if "widened checkpoint weights forbidden" not in algebra["weight_authority"]:
        raise ValueError("prepared execution weight authority changed")

    reader = document["reader_projection"]
    require_exact(
        reader["allowed_reader_transforms"],
        ["selected_token_row/v1", "token_contrast/v1"],
        "reader transforms",
    )
    require_exact(reader["primary_normalization"], "identity/v1", "primary reader normalization")
    if "cannot replace the primary" not in reader["secondary_normalization"]:
        raise ValueError("normalized reader may replace raw logit-unit projection")
    if "normalized reader is not DLA" not in reader["dla_boundary"]:
        raise ValueError("normalized direction was mislabeled as DLA")
    require_exact(reader["share_denominator_epsilon"], 1e-12, "share epsilon")
    if "not nonlinear true-lens" not in reader["true_lens_boundary"]:
        raise ValueError("linear projection was conflated with the true logit lens")

    roles = document["source_roles"]
    expected_roles = ["BOS", "RELATION", "ENTITY", "LAST", "OTHER"]
    require_exact(roles["roles"], expected_roles, "source roles")
    require_exact(roles["precedence"], expected_roles, "source-role precedence")
    if "mutually disjoint" not in roles["overlap_rule"]:
        raise ValueError("ambiguous structural roles no longer refuse")

    gates = document["acceptance_gates"]
    require_exact(gates["head_sum_max_relative_l2"], 1e-4, "head-sum gate")
    require_exact(gates["source_split_max_relative_l2"], 1e-5, "source-split gate")
    require_exact(
        gates["attention_probability_max_absolute_error"],
        1e-6,
        "attention-probability gate",
    )
    if "head-only output" not in gates["full_gate"]:
        raise ValueError("partial head-only evidence may open source consumers")

    vocabulary = document["claim_vocabulary"]
    forbidden_vocabulary = {
        "necessary",
        "necessity",
        "sufficient",
        "sufficiency",
        "causal",
        "cause",
        "driver",
        "decider",
        "mediator",
        "responsible",
        "essential",
        "redundant",
        "transporter",
        "amplifier",
    }
    if set(vocabulary["forbidden"]) != forbidden_vocabulary:
        raise ValueError("descriptive claim-vocabulary prohibition changed")
    if forbidden_vocabulary.intersection(vocabulary["allowed"]):
        raise ValueError("causal vocabulary is allowed in ATTR-1D output")

    downstream = document["downstream"]
    if not downstream["attr_1c_object"].startswith("separate CausalSupportEvidence"):
        raise ValueError("ATTR-1C no longer appends a separate evidence object")
    require_exact(downstream["executability"], "not authorized", "executability boundary")

    return {
        "schema": SCHEMA,
        "status": "valid",
        "attr1d_contract_sha256": identity,
        "coordinate_kinds": len(coordinate["coordinate_kinds"]),
        "source_roles": len(roles["roles"]),
        "forbidden_claim_terms": len(forbidden_vocabulary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.contract), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
