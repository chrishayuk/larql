"""`larql-verifier contract` and `operator`: protocols `larql-verify-contract/1` and `larql-verify-operator/1`
(docs/verification/contract-protocol.md).

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
from .contracts import CONTRACTS, OPERATOR_CONTRACTS, q8_0_gemv, q8_0_t2

SCHEMA = "larql-verify-contract/1"
OPERATOR_SCHEMA = "larql-verify-operator/1"
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
    a = ap.parse_args()
    if a.cmd == "operator":
        answer = evaluate_operator(a.contract, a.operator, a.witness, a.container)
    else:
        answer = evaluate(a.contract, a.container, a.source)
    json.dump(answer, sys.stdout, indent=1, sort_keys=True, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
