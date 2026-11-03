"""`larql-verifier contract`: protocol `larql-verify-contract/1` (docs/verification/contract-protocol.md).

One JSON object on stdout, and nothing else there. Exit 0 whenever the contract was judged or could not be evaluated. A
non-zero exit means the verifier itself failed. It reads the container and the source, and writes nothing else.

A result is a numerical measurement of one container, not a proof about the encoder that produced it.
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
from .contracts import CONTRACTS, q8_0_t2

SCHEMA = "larql-verify-contract/1"
# refcheck/ is copied unmodified from this commit of the isolated checker. See provenance/PROVENANCE.md.
REFCHECK_COMMIT = "e505afcf2d81c0262cd1d75ac0a5dd8429a84c5a"


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


def main() -> None:
    ap = argparse.ArgumentParser(prog="larql-verifier")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("contract")
    p.add_argument("--contract", required=True)
    p.add_argument("--container", required=True)
    p.add_argument("--source", required=True)
    a = ap.parse_args()
    json.dump(evaluate(a.contract, a.container, a.source), sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
