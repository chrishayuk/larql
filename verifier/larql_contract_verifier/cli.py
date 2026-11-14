"""`larql-verifier contract`, `operator`, `patch` and `ledger`: protocols `larql-verify-contract/1`,
`larql-verify-operator/1`, `larql-verify-patch/1` (docs/verification/contract-protocol.md) and `larql-verify-ledger/1`
(`WEIGHT_PATCHES_V4`, docs/verification/weight-patches-v4.md §7).

One JSON object on stdout, and nothing else there. Exit 0 whenever the contract was judged or could not be evaluated. A
non-zero exit means the verifier itself failed. It reads what it's given, and writes nothing else.

A result is a numerical measurement, not a proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import refcheck

from . import __version__, checkpoint
from .contracts import (CONTRACTS, OPERATOR_CONTRACTS, execution_residency, q8_0_gemv, q8_0_t2, structured_patch,
                        weight_patches_v4)

SCHEMA = "larql-verify-contract/1"
OPERATOR_SCHEMA = "larql-verify-operator/1"
PATCH_SCHEMA = "larql-verify-patch/1"
LEDGER_SCHEMA = "larql-verify-ledger/1"
RESIDENCY_SCHEMA = "larql-verify-residency/1"
# refcheck/ is copied unmodified from this commit of the isolated checker. See provenance/PROVENANCE.md.
REFCHECK_COMMIT = "f2788f0e7f1b23432bf4b9bdd1099dbd42d070fc"


def code_sha256() -> str:
    """A digest of the code that measures, so that a result is bound to it: every .py of this package and of refcheck."""
    h = hashlib.sha256()
    for root in (Path(__file__).parent, Path(refcheck.__file__).parent):
        for path in sorted(root.rglob("*.py")):
            h.update(f"{root.name}/{path.relative_to(root).as_posix()}\0".encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def evaluate(contract: str, container: str, source: str, pins=checkpoint.PINS) -> dict:
    known = CONTRACTS.get(contract)
    out = {"schema": SCHEMA, "contract": contract, "contract_statement": known.STATEMENT if known else "",
           "evidence_kind": "numerical_measurement", "status": None, "violations": 0,
           "container": {"path": container, "sha256": None}, "source": {"path": source, "identity": None},
           "measurements": [], "offending": [], "reasons": [],
           "verifier": {"name": "larql-contract-verifier", "version": __version__, "refcheck_commit": REFCHECK_COMMIT,
                        "code_sha256": code_sha256()}}
    if known is None:
        return q8_0_t2.could_not(out, f"unknown contract {contract!r}; this verifier knows {sorted(CONTRACTS)}")
    if not os.path.isdir(container):
        return q8_0_t2.could_not(out, "the container is not a directory")
    out["container"]["sha256"] = checkpoint.container_sha256(os.path.abspath(container))
    return known.evaluate(out, os.path.abspath(container), os.path.abspath(source), pins)


def evaluate_operator(contract: str, operator: str, witness: str, container: str) -> dict:
    statement = q8_0_gemv.STATEMENTS.get(contract, "")
    out = {"schema": OPERATOR_SCHEMA, "contract": contract, "contract_statement": statement,
           "evidence_kind": "numerical_measurement", "status": None, "violations": 0, "operator": operator,
           "witness": {"path": witness, "sha256": None, "basis": "engine_reported", "reported": {}},
           "container": {"path": container, "sha256": None},
           "measurements": [], "offending": [], "reasons": [],
           "verifier": {"name": "larql-contract-verifier", "version": __version__, "refcheck_commit": REFCHECK_COMMIT,
                        "code_sha256": code_sha256()}}
    if contract not in OPERATOR_CONTRACTS:
        return q8_0_gemv.could_not(out, f"unknown contract {contract!r}; this verifier knows {sorted(OPERATOR_CONTRACTS)}")
    if not os.path.isdir(container):
        return q8_0_gemv.could_not(out, "the container is not a directory")
    root = os.path.abspath(container)
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            if any(os.path.islink(os.path.join(dirpath, name)) for name in dirnames + filenames):
                return q8_0_gemv.could_not(out, f"the container holds a symlink under {dirpath}, outside what its digest covers")
        before = checkpoint.container_sha256(root)
    except q8_0_gemv.BAD_INPUT as e:
        return q8_0_gemv.could_not(out, f"the container can't be read: {type(e).__name__}: {e}")
    out["container"]["sha256"] = before
    q8_0_gemv.evaluate(out, contract, operator, os.path.abspath(witness), root)
    # The digest and the bytes judged came from separate reads, so a verdict needs the container not to have changed.
    try:
        unchanged = checkpoint.container_sha256(root) == before
    except q8_0_gemv.BAD_INPUT:
        unchanged = False
    if not unchanged and out["status"] != "could_not_evaluate":
        out.update(violations=0, measurements=[], offending=[], report_sha256=None)
        q8_0_gemv.could_not(out, "the container changed during the evaluation")
    return out


NOT_CHECKED = "not_checked"


def with_axes(out: dict) -> dict:
    """Name what a patch verdict does and does not cover. `status` is unchanged; these are additive.

    `numerical_satisfaction` is the verdict on the weights. `ledger_integrity` is never checked (the ledger isn't read).
    `provenance` is not checked: only the base binding is, and the operand's derivation and origin and the op digest are
    not, so it is never `satisfied` and a satisfied weight check can't be read as one."""
    judged = out["status"] in ("satisfied", "violated")
    out["numerical_satisfaction"] = out["status"]
    out["ledger_integrity"] = NOT_CHECKED
    out["provenance"] = NOT_CHECKED
    out["provenance_detail"] = {"base_binding": "satisfied" if judged else "could_not_evaluate",
                                "operand_origin": NOT_CHECKED, "operand_derivation": NOT_CHECKED,
                                "op_digest": NOT_CHECKED}
    return out


def evaluate_patch(contract: str, patch: str, base: str, container: str) -> dict:
    return with_axes(judge_patch(contract, patch, base, container))


def judge_patch(contract: str, patch: str, base: str, container: str) -> dict:
    """`patch`: does `container` hold what the `.vlp` at `patch` declares of `base`? Recomputed from the operands and the
    base weights, never from the compiler's ledger."""
    out = {"schema": PATCH_SCHEMA, "contract": contract, "contract_statement": structured_patch.STATEMENT,
           "evidence_kind": "numerical_measurement", "status": None, "violations": 0,
           "patch": {"path": patch, "sha256": None}, "base": {"path": base, "sha256": None},
           "container": {"path": container, "sha256": None},
           "measurements": [], "offending": [], "reasons": [],
           "verifier": {"name": "larql-contract-verifier", "version": __version__, "refcheck_commit": REFCHECK_COMMIT,
                        "code_sha256": code_sha256()}}
    if contract != structured_patch.CONTRACT:
        return q8_0_t2.could_not(out, f"unknown contract {contract!r}; this verifier knows {[structured_patch.CONTRACT]}")
    if not os.path.isfile(patch) or not os.path.isdir(base) or not os.path.isdir(container):
        return q8_0_t2.could_not(out, "the patch must be a file, and the base and the container directories")
    try:
        out["patch"]["sha256"] = hashlib.sha256(Path(patch).read_bytes()).hexdigest()
        out["base"]["sha256"] = checkpoint.container_sha256(os.path.abspath(base))
        before = checkpoint.container_sha256(os.path.abspath(container))
    except OSError as e:
        return q8_0_t2.could_not(out, f"an input can't be read: {e}")
    out["container"]["sha256"] = before
    structured_patch.evaluate(out, os.path.abspath(patch), os.path.abspath(base), os.path.abspath(container))
    # The digests and the bytes judged came from separate reads, so a verdict needs the inputs not to have changed.
    try:
        unchanged = (checkpoint.container_sha256(os.path.abspath(container)) == before
                     and checkpoint.container_sha256(os.path.abspath(base)) == out["base"]["sha256"])
    except OSError:
        unchanged = False
    if not unchanged and out["status"] != "could_not_evaluate":
        out.update(violations=0, measurements=[], offending=[])
        q8_0_t2.could_not(out, "an input changed during the evaluation")
    return out


def evaluate_ledger(contract: str, base: str, container: str) -> dict:
    """`ledger`: does `container` hold what its `weight-patches/v4` ledger declares of `base`? Two verdicts, numerical
    satisfaction and ledger integrity and provenance, recomputed from the base and the recorded ops, never from the
    ledger's own hashes or counts. `verifier` identifies this program; `engine_reported` is what the ledger says about
    the engine that wrote it."""
    out = {"schema": LEDGER_SCHEMA, "contract": contract, "contract_statement": weight_patches_v4.STATEMENT,
           "evidence_kind": "numerical_measurement", "status": None, "violations": 0,
           "base": {"path": base, "sha256": None}, "container": {"path": container, "sha256": None},
           "measurements": [], "offending": [], "reasons": [],
           "verifier": {"name": "larql-contract-verifier", "version": __version__, "refcheck_commit": REFCHECK_COMMIT,
                        "code_sha256": code_sha256()}}
    if contract != weight_patches_v4.CONTRACT:
        return q8_0_t2.could_not(out, f"unknown contract {contract!r}; this verifier knows {[weight_patches_v4.CONTRACT]}")
    if not os.path.isdir(base) or not os.path.isdir(container):
        return q8_0_t2.could_not(out, "the base and the container must be directories")
    weight_patches_v4.evaluate(out, os.path.abspath(base), os.path.abspath(container))
    if out["status"] == "could_not_evaluate":
        return out
    # The digests and the bytes judged came from separate reads, so a verdict needs the inputs not to have changed.
    try:
        unchanged = (checkpoint.container_sha256(os.path.abspath(container)) == out["container"]["sha256"]
                     and checkpoint.container_sha256(os.path.abspath(base)) == out["base"]["sha256"])
    except OSError:
        unchanged = False
    if not unchanged:
        out.update(violations=0, offending=[])
        q8_0_t2.could_not(out, "an input changed during the evaluation")
    return out


def evaluate_residency(contract: str, record: str, container: str) -> dict:
    """`residency`: is what the residency record at `record` says a run made resident the declared derivation of
    `container`'s stored weights (derivation), and the representation the recorded policy and cache size select
    (selection)? Two verdicts. Re-derived from the container's bytes; the record's digests are checked, never used
    as expected values."""
    out = {"schema": RESIDENCY_SCHEMA, "contract": contract, "contract_statement": execution_residency.STATEMENT,
           "evidence_kind": "numerical_measurement", "status": None, "violations": 0,
           "record": {"path": record, "sha256": None}, "container": {"path": container, "sha256": None},
           "measurements": [], "offending": [], "reasons": [],
           "derivation": "could_not_evaluate", "selection": "could_not_evaluate", "verdicts": {},
           "failed_halves": [execution_residency.DERIVATION, execution_residency.SELECTION],
           "verifier": {"name": "larql-contract-verifier", "version": __version__, "refcheck_commit": REFCHECK_COMMIT,
                        "code_sha256": code_sha256()}}
    if contract != execution_residency.CONTRACT:
        return q8_0_t2.could_not(out, f"unknown contract {contract!r}; this verifier knows "
                                      f"{[execution_residency.CONTRACT]}")
    if not os.path.isfile(record) or not os.path.isdir(container):
        return q8_0_t2.could_not(out, "the record must be a file and the container a directory")
    execution_residency.evaluate(out, os.path.abspath(record), os.path.abspath(container))
    if out["status"] == "could_not_evaluate":
        return out
    # The digest and the bytes judged came from separate reads, so a verdict needs the input not to have changed.
    try:
        unchanged = checkpoint.container_sha256(os.path.abspath(container)) == out["container"]["sha256"]
    except (OSError, ValueError):  # ValueError: a file name with no digest (not valid UTF-8)
        unchanged = False
    if not unchanged:
        # Neither verdict stands on bytes that changed under it: the halves are withdrawn with the status.
        reason = "the container changed during the evaluation"
        for half in out["verdicts"].values():
            half.update(status="could_not_evaluate", violations=0, offending=[], reasons=half["reasons"] + [reason])
        out.update(violations=0, offending=[], derivation="could_not_evaluate", selection="could_not_evaluate",
                   failed_halves=[execution_residency.DERIVATION, execution_residency.SELECTION])
        q8_0_t2.could_not(out, reason)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="larql-verifier")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("contract")
    p.add_argument("--contract", required=True)
    p.add_argument("--container", required=True)
    p.add_argument("--source", required=True)
    o = sub.add_parser("operator")
    o.add_argument("--contract", required=True)
    o.add_argument("--operator", required=True)
    o.add_argument("--witness", required=True)
    o.add_argument("--container", required=True)
    pa = sub.add_parser("patch")
    pa.add_argument("--contract", required=True)
    pa.add_argument("--patch", required=True)
    pa.add_argument("--base", required=True)
    pa.add_argument("--container", required=True)
    le = sub.add_parser("ledger")
    le.add_argument("--contract", required=True)
    le.add_argument("--base", required=True)
    le.add_argument("--container", required=True)
    rs = sub.add_parser("residency")
    rs.add_argument("--contract", required=True)
    rs.add_argument("--record", required=True)
    rs.add_argument("--container", required=True)
    a = ap.parse_args()
    if a.cmd == "residency":
        answer = evaluate_residency(a.contract, a.record, a.container)
    elif a.cmd == "ledger":
        answer = evaluate_ledger(a.contract, a.base, a.container)
    elif a.cmd == "patch":
        answer = evaluate_patch(a.contract, a.patch, a.base, a.container)
    elif a.cmd == "operator":
        answer = evaluate_operator(a.contract, a.operator, a.witness, a.container)
    else:
        answer = evaluate(a.contract, a.container, a.source)
    json.dump(answer, sys.stdout, indent=1, sort_keys=True, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
