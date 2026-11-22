"""GW-ELEMENT-1: the version-controlled guard against reusing the spent confirmation.

`bench/gw-element-1/confirm-01.SPENT.json` declares that confirm-01 ran once under protocol r2 and is
spent. The local marker `output/gw-element-1/CONFIRMATION-OPENED.txt` is gitignored, so another checkout
would not see it; this guard works from the committed declaration alone.

    refuse_if_spent(protocol_path, targets)

Fails closed (SystemExit) when:
- the declaration is missing or unreadable (the spent state cannot be established);
- the protocol to be executed is the spent protocol (by sha256);
- any intended confirmation target is a spent target or a member of the spent sealed split.

Any tool that executes a GW-ELEMENT-1 confirmation or selects a confirmation population must call it before
touching anything sealed. It is not a check on development or fit work, which may reuse fit/dev elements.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECLARATION = ROOT / "bench/gw-element-1/confirm-01.SPENT.json"


def _sha(p: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def declaration(path: Path = DECLARATION) -> dict:
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise SystemExit(f"FAIL CLOSED: spent-confirmation declaration unreadable ({path}): {e}")
    if d.get("schema") != "larql.gwelem.spent-confirmation.v1" or d.get("status") != "SPENT":
        raise SystemExit(f"FAIL CLOSED: spent-confirmation declaration malformed ({path})")
    return d


def refuse_if_spent(protocol_path, targets, path: Path = DECLARATION) -> dict:
    d = declaration(path)
    if _sha(protocol_path) == d["protocol"]["sha256"]:
        raise SystemExit(f"FAIL CLOSED: protocol {d['protocol']['revision']} is spent "
                         f"(run {d['registry']['run']}, result {d['registry']['verdict_result']}); no second confirmation")
    spent = {int(z) for z in d["spent_targets"]} | {int(z) for z in d["sealed_split"]}
    hit = sorted({int(z) for z in targets} & spent)
    if hit:
        raise SystemExit(f"FAIL CLOSED: targets {hit} belong to the spent confirmation split of {d['registry']['run']}")
    return {"declaration_sha256": _sha(path), "checked_targets": len(list(targets))}
