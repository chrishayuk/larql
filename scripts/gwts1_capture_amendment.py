#!/usr/bin/env python3
"""Build and check the GW-TS-1 population-capture amendment.

The population contract (`bench/gw-ts-1/population-contract.json`) froze the
capture DESIGN and explicitly withheld permission to execute until the ATTR-1D
witness it names was in committed history. This amendment is the separate,
authority-binding-only object that grants that permission. It does not rewrite
the contract, the manifest, or the protocol.

It does three things and nothing more:

1. binds the committed ancestor that carries the frozen ATTR-1D witness and the
   frozen GW-TS-1 artifacts, byte for byte;
2. records the scalar reading of child contribution that the frozen protocol's
   words already imply (`signed_contribution` is signed, so it is a reader
   projection, not a vector norm), and keeps the vector L2 mass as a separate
   recorded diagnostic that never selects anything;
3. requires a bridge witness: the frozen ATTR-1D case re-run through the
   candidate capture binary must reproduce the sealed witness's coordinates
   and measurements, and a deliberately misaligned-source control arm must be
   rejected by the source-split gate.

Commands:
  build --authority-commit SHA [--draft]   write the (draft) amendment
  preflight [--capture-commit SHA]         verify authorities and ancestry
  bridge SEALED CANDIDATE CONTROL          adjudicate a bridge-witness run

Nothing here executes the model. Capture is a separate runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gwts1_population_manifest as population  # noqa: E402

REPO_ROOT = population.REPO_ROOT
GW_TS_1_DIR = population.GW_TS_1_DIR
AMENDMENT_PATH = GW_TS_1_DIR / "capture-amendment.json"
DRAFT_PATH = GW_TS_1_DIR / "capture-amendment.draft.json"

WITNESS_RELATIVE_PATH = "bench/attr-1d/gemma3-4b-it-witness/attr1d-witness.json"
WITNESS_SHA256 = "sha256:88065f65ff3d596f4f3e6716309ccdbc02365ce76d92973dbc5838af4ebb0d73"
POPULATION_CONTRACT_SHA256 = (
    "sha256:d469118c6640e38fe4aa64ffda1e501adb3c6da841bdb6cfbf786eb0760de584"
)
POPULATION_MANIFEST_SHA256 = (
    "sha256:01795de6300c8d347cb6d4c371e71532e1bf17936f281ed58429772cbcef87ef"
)
MAIN_REF = "origin/main"

# Frozen-file bytes the authority commit must carry, relative to the repo root.
FROZEN_FILES = {
    WITNESS_RELATIVE_PATH: WITNESS_SHA256,
    "bench/gw-ts-1/population-contract.json": (
        "sha256:03e2488209161ad384c57902667de16e7153703db3803875f1bc89115ddf08cc"
    ),
    "bench/gw-ts-1/population-manifest.json": (
        "sha256:3024ede64fe9b854fba2d0bcde0c4c5c530f8a90be52695e782bb6ef12dc642c"
    ),
    "bench/gw-ts-1/gwts1-protocol.json": (
        "sha256:2ec70b6ca1ad754321af0f18a3f73eab7dcd0281add1b7f4825fc75d0ac97e62"
    ),
}

FROZEN_TRANSITION_IDENTITIES = 85
FROZEN_PROMPT_FAMILIES = 3
FROZEN_SOURCE_TOP_K = 14
FROZEN_PARENT_SITES = 68

# Witness fields that name one execution rather than what it measured. A
# re-run through a different binary is expected to change exactly these.
RUN_SPECIFIC_FIELDS = frozenset(
    {
        "support_observation.execution_identity",
        "support_observation.observation_receipt_digest",
        "support_observation.provenance_fingerprint",
        "support_observation_id",
    }
)
# Prose attached to a gate. Reported when it changes; never a measurement.
PROSE_FIELD_NAME = "note"
# The gate that must reject a source-misaligned control arm.
SOURCE_ALIGNMENT_GATE = "source_split_max_relative_l2"
BRIDGE_DIFF_EXAMPLES = 20
# The control must be the sealed case, or its rejection proves nothing about it.
CONTROL_CASE_FIELDS = ("support_coordinate_id", "reader_identity", "edge_id")


def child_contribution_reading() -> dict[str, Any]:
    return {
        "basis": (
            "gwts1-protocol SupportMeasurement.signed_contribution is signed, so a "
            "child's contribution is a scalar reader projection; a vector norm "
            "cannot be signed. This records that reading; it changes no frozen field."
        ),
        "reader": "the population contract's selected_token_row/v1 + identity/v1 row",
        "signed_contribution": "dot(reader, child_write)",
        "absolute_contribution": "abs(signed_contribution)",
        "child_write": {
            "attention": (
                "one query head's contribution through the effective prepared W_O "
                "slice, exactly ATTR-1D HeadContribution.contribution"
            ),
            "ffn": (
                "one exact feature's write: activation(gate)*up times its down "
                "column, carried through the post-FFN norm at the complete output's "
                "statistic and the plan's residual scale, exactly the per-feature "
                "vector GW-0B reconstruction sums to the observed write"
            ),
        },
        "mass_denominator": (
            "abs(dot(reader, bias_write)) + sum over the COMPLETE child space of "
            "absolute_contribution, the ATTR-1D absolute_head_denominator law; "
            "bias_write is zero when the plan declares no bias; the bias counts "
            "toward mass but is never a selectable child"
        ),
        "selection_order": "descending absolute_contribution, ties by ascending physical address",
        "vector_l2_mass": {
            "role": "recorded diagnostic only",
            "record": ["per-child contribution L2", "complete-space total L2 mass"],
            "may_select_or_admit": False,
        },
    }


def bridge_witness_requirement() -> dict[str, Any]:
    return {
        "reason": (
            "main changed AttentionHeadRecord.source_values from &[Vec<f32>] to "
            "&[&[f32]]; the sealed witness's overall_pass was produced on the owned "
            "path and does not by itself qualify a capture built on the borrowed path"
        ),
        "case": "the sealed ATTR-1D witness case: its edge, prompt, L24 attention site, final prompt position and reader",
        "binary": "the candidate capture binary, built from the capture source commit",
        "backend": (
            "the production CPU backend (ProductionBackend), as the sealed witness "
            "records in gates.backend_agreement; never Metal or another backend"
        ),
        "platform": (
            "the bridge qualifies a binary on ONE target: production-backend "
            "reductions differ in summation order between x86 SIMD and aarch64 "
            "(#510, 54d07e91), so bit-identity is expected only on the target the "
            "sealed witness ran on, and population capture must run on the same "
            "target triple as the passing bridge run, which the bridge record binds"
        ),
        "candidate_arm": {
            "must_pass": "every ATTR-1D gate and overall_pass",
            "must_match": "every witness field except the run-specific identities, compared exactly",
            "run_specific_fields_allowed_to_differ": sorted(RUN_SPECIFIC_FIELDS),
            "gate_prose": "reported when changed, never compared as a measurement",
        },
        "control_arm": {
            "perturbation": "rotate each head's source_values by one source position before description",
            "must": (
                f"fail gates.{SOURCE_ALIGNMENT_GATE} and overall_pass while every "
                "other gate passes: the complete gate vector is compared"
            ),
            "same_case_fields": list(CONTROL_CASE_FIELDS),
            "why_only_that_gate": (
                "head contributions are projected from mixed_values and only the "
                "per-source splits read source values; rotating values keeps every "
                "position and width, so coverage, head-sum, probability and "
                "projection gates cannot move"
            ),
            "purpose": (
                "prove the bridge rejects the one failure it exists to catch, and "
                "that an unrelated breakage cannot satisfy the control"
            ),
        },
        "outcomes": {
            "bit_identical": "capture authorized",
            "measurements_differ": (
                "capture NOT authorized; no cross-run tolerance is frozen, so a "
                "difference needs its own amendment naming the cause"
            ),
            "control_not_rejected": "capture NOT authorized; the bridge is blind",
        },
    }


def build(authority_commit: str, draft: bool) -> dict[str, Any]:
    document = {
        "schema": "larql.gwts1.population-capture-amendment.v1",
        "status": "draft_pending_merge" if draft else "frozen_before_population_capture",
        "amends": {
            "population_contract_sha256": POPULATION_CONTRACT_SHA256,
            "population_manifest_sha256": POPULATION_MANIFEST_SHA256,
            "gwts1_protocol_sha256": population.GWTS1_PROTOCOL_SHA256,
        },
        "attr1d_authority": {
            "contract_identity": population.ATTR1D_CONTRACT_IDENTITY,
            "witness_path": WITNESS_RELATIVE_PATH,
            "witness_sha256": WITNESS_SHA256,
            "overall_pass": True,
            "repository_commit": authority_commit,
            "commit_requirement": (
                f"carries every frozen file byte-identically and is an ancestor of "
                f"both {MAIN_REF} and the capture source commit"
            ),
            "frozen_files": FROZEN_FILES,
        },
        "change": {
            "kind": "authority_binding_and_reading_only",
            "population_design_changed": False,
            "assignment_changed": False,
            "capture_window_changed": False,
            "reader_changed": False,
            "observation_contract_changed": False,
            "thresholds_changed": False,
            "budgets_changed": False,
        },
        "child_contribution_reading": child_contribution_reading(),
        "bridge_witness": bridge_witness_requirement(),
        "authorization": {
            "population_capture": "permitted after preflight passes and the bridge witness is bit_identical",
            "gwts1_extraction_or_assessment": "not authorized by this amendment",
        },
    }
    document["amendment_sha256"] = population.canonical_identity(document, "amendment_sha256")
    return document


def git(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True)


def resolve_commit(revision: str) -> str:
    result = git("rev-parse", "--verify", f"{revision}^{{commit}}")
    if result.returncode != 0:
        raise ValueError(f"{revision}: not a commit in this repository")
    return result.stdout.decode().strip()


def is_ancestor(ancestor: str, descendant: str) -> bool:
    result = git("merge-base", "--is-ancestor", ancestor, descendant)
    if result.returncode not in (0, 1):
        raise ValueError(f"cannot decide ancestry of {ancestor} in {descendant}")
    return result.returncode == 0


def blob_sha256(commit: str, path: str) -> str:
    result = git("show", f"{commit}:{path}")
    if result.returncode != 0:
        raise ValueError(f"{path} is absent at {commit}")
    return "sha256:" + hashlib.sha256(result.stdout).hexdigest()


def verify_frozen_population() -> None:
    contract = json.loads((GW_TS_1_DIR / "population-contract.json").read_text())
    manifest = json.loads((GW_TS_1_DIR / "population-manifest.json").read_text())
    if population.canonical_identity(contract, "population_contract_sha256") != (
        POPULATION_CONTRACT_SHA256
    ):
        raise ValueError("population contract no longer hashes to its frozen identity")
    if contract["population_contract_sha256"] != POPULATION_CONTRACT_SHA256:
        raise ValueError("population contract declares a different identity")
    if population.canonical_identity(manifest, "manifest_sha256") != POPULATION_MANIFEST_SHA256:
        raise ValueError("population manifest no longer hashes to its frozen identity")
    if manifest["population"]["transition_identity_count"] != FROZEN_TRANSITION_IDENTITIES:
        raise ValueError("population no longer has the frozen identity count")
    _, rows = population.load_gw0()
    eligible = population.eligible_groups(rows)
    instances = sum(len(members) for members in eligible.values())
    if instances != FROZEN_TRANSITION_IDENTITIES * FROZEN_PROMPT_FAMILIES:
        raise ValueError(
            f"expected {FROZEN_TRANSITION_IDENTITIES * FROZEN_PROMPT_FAMILIES} "
            f"prompt instances, found {instances}"
        )
    if manifest["observation"]["attention"]["source_top_k"] != FROZEN_SOURCE_TOP_K:
        raise ValueError("source_top_k moved from its frozen value")
    if manifest["capture"]["eligible_parent_sites_per_execution"] != FROZEN_PARENT_SITES:
        raise ValueError("parent-site count moved from its frozen value")


def verify_witness_file() -> None:
    path = REPO_ROOT / WITNESS_RELATIVE_PATH
    if population.sha256_file(path) != WITNESS_SHA256:
        raise ValueError("working-tree ATTR-1D witness is not the sealed bytes")
    witness = json.loads(path.read_text())
    if witness["attr1d_contract_identity"] != population.ATTR1D_CONTRACT_IDENTITY:
        raise ValueError("sealed witness binds a different ATTR-1D contract")
    if witness["overall_pass"] is not True:
        raise ValueError("sealed ATTR-1D witness did not pass")


def preflight(amendment: dict[str, Any], capture_commit: str, require_main: bool) -> list[str]:
    """Return the checks performed. Raises on the first failure."""
    if population.canonical_identity(amendment, "amendment_sha256") != amendment["amendment_sha256"]:
        raise ValueError("amendment does not hash to its declared identity")
    authority = resolve_commit(amendment["attr1d_authority"]["repository_commit"])
    capture = resolve_commit(capture_commit)
    for path, expected in amendment["attr1d_authority"]["frozen_files"].items():
        if blob_sha256(authority, path) != expected:
            raise ValueError(f"{path} at the authority commit is not the frozen bytes")
    if not is_ancestor(authority, capture):
        raise ValueError("authority commit is not an ancestor of the capture source commit")
    if require_main and not is_ancestor(authority, resolve_commit(MAIN_REF)):
        raise ValueError(f"authority commit is not on {MAIN_REF}")
    verify_witness_file()
    verify_frozen_population()
    return [
        "amendment identity",
        f"frozen bytes at {authority[:12]}",
        f"authority is an ancestor of capture source {capture[:12]}",
        *([f"authority is on {MAIN_REF}"] if require_main else []),
        "sealed witness bytes, contract and pass",
        "frozen population identities and counts",
    ]


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            out.update(flatten(child, f"{prefix}.{key}" if prefix else key))
        return out
    if isinstance(value, list):
        out = {}
        for index, child in enumerate(value):
            out.update(flatten(child, f"{prefix}[{index}]"))
        return out if value else {prefix: []}
    return {prefix: value}


def gate_vector(witness: dict[str, Any]) -> dict[str, Any]:
    """Every pass/fail gate by name; `None` where a gate recorded no verdict."""
    return {
        name: gate.get("pass")
        for name, gate in witness.get("gates", {}).items()
        if isinstance(gate, dict)
    }


def gates_pass(witness: dict[str, Any]) -> bool:
    gates = witness.get("gates", {})
    return witness.get("overall_pass") is True and all(
        gate.get("pass") is True for gate in gates.values() if isinstance(gate, dict)
    )


def adjudicate_bridge(
    sealed: dict[str, Any], candidate: dict[str, Any], control: dict[str, Any]
) -> dict[str, Any]:
    sealed_fields = flatten(sealed)
    candidate_fields = flatten(candidate)
    differing: list[dict[str, Any]] = []
    prose_changed: list[str] = []
    for path in sorted(sealed_fields.keys() | candidate_fields.keys()):
        if path in RUN_SPECIFIC_FIELDS:
            continue
        left = sealed_fields.get(path, "<absent>")
        right = candidate_fields.get(path, "<absent>")
        if left == right and type(left) is type(right):
            continue
        if path.rsplit(".", 1)[-1] == PROSE_FIELD_NAME:
            prose_changed.append(path)
            continue
        differing.append({"field": path, "sealed": left, "candidate": right})

    control_gates = gate_vector(control)
    expected_control = {
        name: name != SOURCE_ALIGNMENT_GATE for name in gate_vector(sealed)
    }
    same_case = all(
        control.get(field) == sealed.get(field) for field in CONTROL_CASE_FIELDS
    )
    control_rejected = (
        same_case
        and control_gates == expected_control
        and control.get("overall_pass") is False
    )

    if not gates_pass(candidate):
        verdict = "candidate_gates_fail"
    elif not control_rejected:
        verdict = "control_not_rejected"
    elif differing:
        verdict = "measurements_differ"
    else:
        verdict = "bit_identical"
    return {
        "verdict": verdict,
        "capture_authorized": verdict == "bit_identical",
        "compared_fields": len(sealed_fields.keys() - RUN_SPECIFIC_FIELDS),
        "differing_field_count": len(differing),
        "differing_fields": differing[:BRIDGE_DIFF_EXAMPLES],
        "prose_changed": prose_changed,
        "control": {
            "same_case": same_case,
            "expected_gate_vector": expected_control,
            "observed_gate_vector": control_gates,
            "overall_pass": control.get("overall_pass"),
            "rejected": control_rejected,
        },
    }


def write_new(path: Path, document: dict[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"{path} already exists")
    path.write_text(json.dumps(document, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--authority-commit", required=True)
    build_parser.add_argument("--draft", action="store_true")
    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("--capture-commit", default="HEAD")
    preflight_parser.add_argument("--amendment", type=Path, default=AMENDMENT_PATH)
    bridge_parser = commands.add_parser("bridge")
    for name in ("sealed", "candidate", "control"):
        bridge_parser.add_argument(name, type=Path)
    args = parser.parse_args()

    if args.command == "build":
        document = build(resolve_commit(args.authority_commit), args.draft)
        path = DRAFT_PATH if args.draft else AMENDMENT_PATH
        preflight(document, "HEAD", require_main=not args.draft)
        if path == DRAFT_PATH and path.exists():
            path.unlink()
        write_new(path, document)
        print(f"{path.relative_to(REPO_ROOT)}: {document['amendment_sha256']}")
        return 0
    if args.command == "preflight":
        amendment = json.loads(args.amendment.read_text())
        draft = amendment["status"].startswith("draft")
        for check in preflight(amendment, args.capture_commit, require_main=not draft):
            print(f"ok  {check}")
        if draft:
            print(f"--  draft: {MAIN_REF} ancestry not required and capture NOT authorized")
        return 0
    report = adjudicate_bridge(
        *(json.loads(path.read_text()) for path in (args.sealed, args.candidate, args.control))
    )
    print(json.dumps(report, indent=2))
    return 0 if report["capture_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
