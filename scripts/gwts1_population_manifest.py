#!/usr/bin/env python3
"""Materialize and seal the GW-TS-1 population manifest and contract.

This is `run_experiment_subjects` from the frozen dependency graph
(`bench/gw-ts-1/programme-dependencies.json`): a deterministic derivation
over the already-frozen GW-0 census. It performs NO model execution — it
only groups, hashes and splits data that is already on disk. Population
CAPTURE (running the model over these rows) is a separate, later step and
is explicitly out of scope for this script.

Outputs (refuses if either already exists — this is a seal, not an update):
  bench/gw-ts-1/population-manifest.json   the materialized TransitionIdentity
                                            assignments, source_top_k and their
                                            content hash
  bench/gw-ts-1/population-contract.json   the frozen contract binding every
                                            population_binding.required field
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
GW0_DIR = REPO_ROOT / "bench" / "gw0" / "gemma3-4b-it-phase1"
GW_TS_1_DIR = REPO_ROOT / "bench" / "gw-ts-1"
ATTR1D_DIR = REPO_ROOT / "bench" / "attr-1d"

GWTS1_PROTOCOL_SHA256 = (
    "sha256:d873d238d99406dc8a549681aa746e750a9d4b802e24ff21a8eb26d152a97d55"
)
ATTR1D_CONTRACT_IDENTITY = (
    "sha256:4f48973807db0695d5f30b83a00c73637afbd0dc77d116b5a1f9eb6ec913431c"
)
DECLARED_FAMILIES = {"canonical", "question", "alternate"}
DISCOVERY_FRACTION = 0.6
LAYER_FIRST = 0
LAYER_LAST = 33
FFN_RETAINED_ADDRESSES = 64
BYTES_PER_TRANSITION_INSTANCE = 64 * 1024 * 1024
BYTES_AGGREGATE = 16 * 1024 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def canonical_identity(document: dict[str, Any], identity_field: str) -> str:
    payload = dict(document)
    payload.pop(identity_field, None)
    return canonical_sha256(payload)


def load_gw0() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((GW0_DIR / "manifest.json").read_text())
    if manifest["schema"] != "larql.gw0.input-manifest.v1":
        raise ValueError("not the frozen GW-0 input manifest")
    rows_path = GW0_DIR / manifest["rows"]["path"]
    if sha256_file(rows_path) != manifest["rows"]["sha256"]:
        raise ValueError("GW-0 input rows do not match the frozen manifest hash")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    if len(rows) != manifest["rows"]["count"]:
        raise ValueError("GW-0 row count does not match the frozen manifest")
    return manifest, rows


def eligible_groups(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        edge = row["semantic_edge"]
        if edge["status"] != "positive":
            raise ValueError(f"{row['edge_id']}: not a positive semantic edge")
        key = (edge["relation"], edge["subject"], edge["target"])
        groups.setdefault(key, []).append(row)

    eligible: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for key, members in groups.items():
        families = {m["semantic_edge"]["prompt_semantic_family"] for m in members}
        splits = {m["split"] for m in members}
        if families == DECLARED_FAMILIES and splits == {"train"}:
            eligible[key] = members
    return eligible


def transition_identity_hash(relation: str, subject: str, target: str) -> str:
    """`canonical_hash_order/v1`: sha256 of the sorted-key canonical JSON of
    the bare transition-identity tuple. Used only for deterministic
    ordering, never for anything an outcome could feed back into."""
    return canonical_sha256({"relation": relation, "subject": subject, "target": target})


def apportion_discovery(count: int, fraction: float) -> int:
    """Hamilton/largest-remainder apportionment, rounded to the nearest
    whole count. Exposed separately so the caller can apply the shared
    top-up pass across strata."""
    return int(count * fraction)


def stratified_split(
    eligible: dict[tuple[str, str, str], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Deterministic discovery/held-out assignment, stratified by relation
    (the population's `semantic_operation`; `task_identity` is the single
    constant `raw_text_completion_container_tokenizer_with_special_tokens`
    protocol shared by every GW-0 row, so it does not further split any
    stratum). Largest-remainder (Hamilton) apportionment per stratum, ties
    broken by ascending relation name, so the total lands exactly on
    round(len(eligible) * DISCOVERY_FRACTION) without discretion at the
    per-group level. Within a stratum, canonical_hash_order/v1 (ascending
    sha256 of the bare identity tuple) selects which exact groups are
    discovery vs. held-out — a rule fixed before any outcome is observed.
    """
    by_relation: dict[str, list[tuple[str, str, str]]] = {}
    for key in eligible:
        by_relation.setdefault(key[0], []).append(key)

    floors: dict[str, int] = {}
    remainders: dict[str, float] = {}
    for relation, keys in by_relation.items():
        exact = len(keys) * DISCOVERY_FRACTION
        floors[relation] = int(exact)
        remainders[relation] = exact - floors[relation]

    total_target = round(len(eligible) * DISCOVERY_FRACTION)
    shortfall = total_target - sum(floors.values())
    if shortfall < 0 or shortfall > len(by_relation):
        raise ValueError("largest-remainder apportionment shortfall is out of range")
    ranked = sorted(by_relation, key=lambda relation: (-remainders[relation], relation))
    discovery_count = dict(floors)
    for relation in ranked[:shortfall]:
        discovery_count[relation] += 1
    if sum(discovery_count.values()) != total_target:
        raise ValueError("apportionment did not reach the target discovery count")

    rows: list[dict[str, Any]] = []
    for relation, keys in by_relation.items():
        ordered = sorted(keys, key=lambda key: transition_identity_hash(*key))
        n_discovery = discovery_count[relation]
        for index, key in enumerate(ordered):
            assignment = "discovery" if index < n_discovery else "held_out"
            rows.append(
                {
                    "relation": key[0],
                    "subject": key[1],
                    "target": key[2],
                    "canonical_hash_order/v1": transition_identity_hash(*key),
                    "stratum": relation,
                    "stratum_rank": index,
                    "assignment": assignment,
                }
            )
    rows.sort(key=lambda row: (row["relation"], row["subject"], row["target"]))
    return rows


def max_visible_prompt_length(
    eligible: dict[tuple[str, str, str], list[dict[str, Any]]]
) -> int:
    return max(
        len(member["prompt"]["token_ids"])
        for members in eligible.values()
        for member in members
    )


INSTRUMENT_VALIDATION_EXPOSURE = {
    "relation": "capital",
    "subject": "Algeria",
    "target": "Algiers",
}


def provenance_facts() -> list[dict[str, Any]]:
    """Declared before capture, not derived from any GW-TS-1 outcome: the
    one identity ATTR-1D's sealed witness ran on. Recorded explicitly so the
    fact is on the record rather than silently true; it does not move the
    identity or otherwise touch the deterministic split."""
    identity = INSTRUMENT_VALIDATION_EXPOSURE
    return [
        {
            "identity": identity,
            "fact": (
                f"{identity['relation']}/{identity['subject']}/{identity['target']} "
                "received prior instrument-validation exposure under ATTR-1D — the "
                "sealed descriptive-support witness ran on this identity's canonical "
                "prompt family."
            ),
            "assurance": (
                "no GW-TS-1 population assignment, extraction rule, calibration, "
                "predictor, or gate was changed using that exposure"
            ),
            "disposition": (
                "retained at its deterministic canonical_hash_order/v1 position; "
                "not moved between discovery and held_out because of this exposure"
            ),
        }
    ]


def preregistered_sensitivity_checks() -> list[dict[str, Any]]:
    identity = INSTRUMENT_VALIDATION_EXPOSURE
    return [
        {
            "name": "leave_algeria_out",
            "excluded_identity": identity,
            "definition": (
                "recompute every held-out assessment metric (held-out path coverage, "
                "mean candidate fraction, the block-bootstrap mass-retention and "
                "operator-accuracy advantages, and the progression_gate verdict) with "
                f"{identity['relation']}/{identity['subject']}/{identity['target']} "
                "excluded from the held-out set"
            ),
            "purpose": (
                "demonstrate the primary verdict is not carried by the one held-out "
                "identity that received prior ATTR-1D instrument-validation exposure"
            ),
            "authority": (
                "reported alongside, never in place of, the primary held-out verdict; "
                "cannot promote or demote the primary progression_gate outcome"
            ),
            "frozen_before": "population capture",
        }
    ]


def load_attr1d_witness() -> dict[str, Any]:
    path = ATTR1D_DIR / "gemma3-4b-it-witness" / "attr1d-witness.json"
    witness = json.loads(path.read_text())
    if witness["attr1d_contract_identity"] != ATTR1D_CONTRACT_IDENTITY:
        raise ValueError("sealed witness does not bind the frozen ATTR-1D contract identity")
    if witness["overall_pass"] is not True:
        raise ValueError("ATTR-1D witness did not pass — refusing to build on it")
    return {"identity": sha256_file(path), "overall_pass": witness["overall_pass"]}


def build_manifest() -> dict[str, Any]:
    manifest, rows = load_gw0()
    eligible = eligible_groups(rows)
    if len(eligible) == 0:
        raise ValueError("no train TransitionIdentity has all three prompt families")

    assignments = stratified_split(eligible)
    discovery = [row for row in assignments if row["assignment"] == "discovery"]
    held_out = [row for row in assignments if row["assignment"] == "held_out"]

    exposed = INSTRUMENT_VALIDATION_EXPOSURE
    exposed_rows = [
        row
        for row in assignments
        if (row["relation"], row["subject"], row["target"])
        == (exposed["relation"], exposed["subject"], exposed["target"])
    ]
    if len(exposed_rows) != 1:
        raise ValueError(
            "the declared ATTR-1D instrument-validation identity is not present "
            "exactly once in the eligible population — the provenance fact is stale"
        )
    if exposed_rows[0]["assignment"] != "held_out":
        raise ValueError(
            "the declared ATTR-1D instrument-validation identity moved out of "
            "held_out — the provenance fact and leave-Algeria-out sensitivity "
            "check both assume it stays there"
        )
    source_top_k = max_visible_prompt_length(eligible)
    attr1d_witness = load_attr1d_witness()

    document = {
        "schema": "larql.gwts1.population-manifest.v1",
        "status": "materialized_pre_population_capture",
        "authorities": {
            "gwts1_protocol_sha256": GWTS1_PROTOCOL_SHA256,
            "attr1d_contract_identity": ATTR1D_CONTRACT_IDENTITY,
            "attr1d_witness": attr1d_witness,
            "gw0_manifest_sha256": manifest["manifest_sha256"],
        },
        "population": {
            "eligible_upstream_split": "train",
            "transition_identity_count": len(eligible),
            "prompt_family_requirement": sorted(DECLARED_FAMILIES),
            "other_upstream_splits": "excluded_and_reserved_for_future_replication",
        },
        "assignment": {
            "group": "transition_identity",
            "stratify_by": ["task_identity", "semantic_operation"],
            "task_identity_constant": manifest["prompt_protocol"],
            "semantic_operation_field": "relation",
            "method": "canonical_hash_order/v1",
            "apportionment": "largest_remainder_ties_by_ascending_relation_name",
            "discovery_fraction": DISCOVERY_FRACTION,
            "discovery_count": len(discovery),
            "held_out_count": len(held_out),
            "outcome_inputs": "forbidden",
            "rows": assignments,
        },
        "provenance_facts": provenance_facts(),
        "preregistered_sensitivity_checks": preregistered_sensitivity_checks(),
        "capture": {
            "position": "final_prompt_position",
            "layers": {"first": LAYER_FIRST, "last": LAYER_LAST},
            "operators": ["attention", "ffn"],
            "eligible_parent_sites_per_execution": (LAYER_LAST - LAYER_FIRST + 1) * 2,
            "generated_positions": "excluded",
            "site_order": "canonical_execution_order",
        },
        "reader": {
            "transform": "selected_token_row/v1",
            "normalization": "identity/v1",
            "token_source": "preregistered_first_destination_continuation_token",
            "token_field": "semantic_edge.target_token_ids[0]",
            "observed_generation_token": "forbidden",
            "weight_source": "exact_resident_prepared_output_head",
            "accessor": "SelectedOutputHead::row_f32",
            "bind": [
                "token_id",
                "vector_sha256",
                "prepared_image_fingerprint",
                "realization_policy",
            ],
        },
        "observation": {
            "level": "canonical_parent_plus_typed_child_support/v1",
            "attention": {
                "authority": "ATTR-1D",
                "source_top_k_rule": "maximum_visible_prompt_length_in_frozen_gwts1_population",
                "source_top_k": source_top_k,
                "source_top_k_derivation": (
                    "max(len(prompt.token_ids)) over the 3 x transition_identity_count "
                    "rows in this manifest's discovery+held_out population only "
                    "(not the wider GW-0 census, most of which is outside GW-TS-1)"
                ),
                "source_truncation": "forbidden",
                "retain_all_heads": True,
            },
            "ffn": {
                "authority": "production-effective GW-0B-style reconstruction",
                "retained_exact_addresses": FFN_RETAINED_ADDRESSES,
                "retain_total_absolute_child_mass": True,
                "retain_achieved_mass": True,
                "mass_truncation_must_be_explicit": True,
            },
        },
        "byte_budget": {
            "per_transition_instance": BYTES_PER_TRANSITION_INSTANCE,
            "aggregate": BYTES_AGGREGATE,
            "overflow": "refuse_instance",
            "silent_truncation": "forbidden",
        },
        "refusal_policy": {
            "join_mismatch": "refuse_path",
            "missing_attention_evidence": "refuse_instance",
            "missing_ffn_evidence": "refuse_instance",
            "source_truncation": "refuse_instance",
            "capture_budget_exceeded": "refuse_instance",
            "unsupported_operator": "explicit_refusal",
            "coverage_accounting": "all refusals count against held-out path coverage",
        },
    }
    document["manifest_sha256"] = canonical_identity(document, "manifest_sha256")
    return document


def build_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    document = {
        "schema": "larql.gwts1.population-contract.v1",
        "status": "frozen_pre_population_capture",
        "authorities": {
            "gwts1_protocol": {"sha256": GWTS1_PROTOCOL_SHA256},
            "attr1d_contract": {"identity": ATTR1D_CONTRACT_IDENTITY},
            "attr1d_witness": manifest["authorities"]["attr1d_witness"],
            "gw0_population": {"identity": manifest["authorities"]["gw0_manifest_sha256"]},
            "population_manifest": {"sha256": manifest["manifest_sha256"]},
        },
        "population": manifest["population"],
        "assignment": {
            key: value
            for key, value in manifest["assignment"].items()
            if key != "rows"
        },
        "provenance_facts": manifest["provenance_facts"],
        "preregistered_sensitivity_checks": manifest["preregistered_sensitivity_checks"],
        "capture": manifest["capture"],
        "reader": manifest["reader"],
        "observation": manifest["observation"],
        "byte_budget": manifest["byte_budget"],
        "refusal_policy": manifest["refusal_policy"],
        "freeze": {
            "before": "first_population_model_execution",
            "binds": [
                "transition rows and identities",
                "prompt families and tokenizations",
                "discovery and held-out assignments",
                "model/container/realization/operation-plan identities",
                "eligible sites and capture window",
                "reader identities and hashes",
                "observation level",
                "source top_k",
                "byte budgets",
                "refusal policy",
            ],
            "note": (
                "This contract's attr1d_witness authority currently names an "
                "uncommitted working-tree artifact. Population CAPTURE must not "
                "begin until that ATTR-1D closure commit has actually landed — "
                "this contract freezes the DESIGN, not the permission to execute."
            ),
        },
    }
    document["population_contract_sha256"] = canonical_identity(
        document, "population_contract_sha256"
    )
    return document


def write_sealed(path: Path, document: dict[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"{path} is already sealed")
    path.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n")


def main() -> None:
    manifest = build_manifest()
    manifest_path = GW_TS_1_DIR / "population-manifest.json"
    write_sealed(manifest_path, manifest)

    contract = build_contract(manifest)
    contract_path = GW_TS_1_DIR / "population-contract.json"
    write_sealed(contract_path, contract)

    print(f"discovery: {manifest['assignment']['discovery_count']}")
    print(f"held_out: {manifest['assignment']['held_out_count']}")
    print(f"source_top_k: {manifest['observation']['attention']['source_top_k']}")
    print(f"manifest_sha256: {manifest['manifest_sha256']}")
    print(f"population_contract_sha256: {contract['population_contract_sha256']}")


if __name__ == "__main__":
    main()
