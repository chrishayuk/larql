"""GW-ELEMENT-1: independent target-inventory preflight before the single confirmation execution.

    gwelem_confirm_preflight.py OUT_DIR

Runs before `gwelem_confirm.py execute` and touches nothing sealed (no model, no capture, no marker).
It does not modify the qualified orchestrator. It asserts that the confirmation target set is exactly the
frozen set, from four sources that must all agree:

1. a literal typed here, independently of `gwelem_design.py`;
2. the table in protocol r2 (`docs/gw-element-1-confirmation-protocol.md`, sha bound);
3. `gwelem_design.CONFIRMATION`, re-derived from the registered feasibility and baseline records;
4. `binding.json` of OUT_DIR, plus the runner's compiled-in `SEALED_KEYS`.

It also asserts 25 distinct targets, disjoint from fit and development, the 200/100/100 arm inventory, and
that the global marker is absent. Writes `preflight.json`. Exits non-zero on any difference.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gwelem_design as D  # noqa: E402
import gwelem_confirm as G  # noqa: E402

EXPECTED_COUNT = 25
LITERAL = (5, 7, 22, 23, 24, 26, 27, 28, 36, 47, 49, 58, 61, 64, 68, 79, 80, 86, 87, 90, 95, 96, 97, 101, 110)
PROTOCOL_SHA = "sha256:74f1fb791c6cd24130b5991bacdb20e665102d6053bfc8dac9de20de69c85a10"
PROTOCOL = ROOT / "docs/gw-element-1-confirmation-protocol.md"
RUNNER_SRC = ROOT / "crates/larql-demos/examples/inference/gwelem_band_runner.rs"


def sha(p):
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(out: Path) -> int:
    fails, src = [], {}
    if sha(PROTOCOL) != PROTOCOL_SHA:
        fails.append("protocol r2 sha differs")
    row = re.search(r"^\| Confirmation \((\d+)\) \| ([0-9 ]+) \|", PROTOCOL.read_text(), re.M)
    src["protocol_r2"] = tuple(int(x) for x in row.group(2).split()) if row else ()
    if not row or int(row.group(1)) != EXPECTED_COUNT:
        fails.append("protocol table count")
    pops = D.verify_populations()  # re-derived from the registered records; exits on any difference
    src["design"] = tuple(D.CONFIRMATION)
    src["design_rederived"] = tuple(pops["confirmation"])
    b = json.loads((out / "binding.json").read_text())
    src["binding_targets"] = tuple(b["targets"])
    src["binding_populations"] = tuple(b["populations"]["confirmation"])
    src["orchestrator_default_cfg"] = tuple(G.default_cfg()["targets"])
    m = re.search(r"SEALED_KEYS[^=]*=\s*&?\[(.*?)\];", RUNNER_SRC.read_text(), re.S)
    sealed = {int(x) for x in re.findall(r"\d+", m.group(1))} if m else set()
    if sha(RUNNER_SRC) != b["files"]["runner_source"]["sha256"]:
        fails.append("runner source differs from binding")
    for name, t in src.items():
        if len(t) != EXPECTED_COUNT or len(set(t)) != EXPECTED_COUNT:
            fails.append(f"{name}: {len(t)} targets ({len(set(t))} distinct), expected {EXPECTED_COUNT}")
        if sorted(t) != sorted(LITERAL):
            fails.append(f"{name}: set differs from the literal: +{sorted(set(t) - set(LITERAL))} -{sorted(set(LITERAL) - set(t))}")
    if set(LITERAL) & (set(D.FIT) | set(D.DEVELOPMENT)):
        fails.append("confirmation overlaps fit/development")
    if len(sealed) != 35 or not set(LITERAL) <= sealed or sealed != set(D.CONFIRMATION_SPLIT):
        fails.append(f"runner SEALED_KEYS: {len(sealed)} keys, covers targets={set(LITERAL) <= sealed}")
    inv = D.inventory(LITERAL)
    if (inv["arms_total"], inv["identity_arms"], inv["constructed_or_reference_arms"]) != (200, 100, 100):
        fails.append(f"inventory {inv}")
    if G.GLOBAL_MARKER.exists():
        fails.append("global marker already exists")
    rec = {"schema": "larql.gwelem.confirm-preflight.v1", "status": "pass" if not fails else "FAIL", "failures": fails,
           "expected_count": EXPECTED_COUNT, "targets": list(LITERAL), "sources_agree": sorted(src),
           "runner_sealed_keys": len(sealed), "inventory": {k: inv[k] for k in ("arms_total", "identity_arms", "constructed_or_reference_arms")},
           "binding_sha256": sha(out / "binding.json")}
    (out.parent / f"{out.name}-preflight.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve()))
