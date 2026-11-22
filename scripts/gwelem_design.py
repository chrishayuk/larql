"""GW-ELEMENT-1 confirmation: the frozen design constants, in one place.

Everything a later stage needs to agree on is here: the populations, the
layer, the spans, the arm inventory and the gate values. Every other
GW-ELEMENT-1 confirmation module imports these constants rather than
restating them. The protocol is docs/gw-element-1-confirmation-protocol.md.

The populations are **re-derived** from the registered records and must equal
the literal lists below:
- the feasibility record (artifact 2748), which holds the salted 118-element split;
- the natural-baseline report (results 6315/6316, run …01258), which holds
  natural eligibility.

`verify_populations()` fails closed on any difference. Nothing is chosen here
from switching data; none exists.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FEAS = ROOT / "output/gw-element-1/feasibility-01/feasibility.json"
FEAS_SHA = "sha256:068ecc941c1aec641f4017d890f1ce5ccc70fe643cd1822ca85349f031d8924e"
BASELINE = ROOT / "output/gw-element-1/baseline-01"

RELATIONS = ("number", "symbol")  # source relation r (target prompt), injected relation r'
SOURCE, INJECTED = "number", "symbol"
PHRASINGS = ("A2", "A3")
LAYER = 4  # single post-FFN carrier layer: from_layer = 5, to_layer = 4
WINDOW_LEN = 4
# Window slots, aligned between the relations: A2 ", the atomic number" / ", the chemical symbol";
# A3 "' s atomic number" / "' s chemical symbol".
SPANS = {"full": (0, 1, 2, 3)}
# Slots 0-1 (A2 ", the"; A3 "' s") precede the first relation-specific token. At L4 they are bit-identical
# between the number and symbol prompts (confirm-qual-02, all 60 fit+dev elements, both phrasings), so
# C2 and S equal the natural carrier there. Only slots 2-3 are numerically changed: the designated positions.
EFFECTIVE_SLOTS = (2, 3)
DECODED_WINDOWS = {
    ("number", "A2"): [",", " the", " atomic", " number"],
    ("symbol", "A2"): [",", " the", " chemical", " symbol"],
    ("number", "A3"): ["'", "s", " atomic", " number"],
    ("symbol", "A3"): ["'", "s", " chemical", " symbol"],
}

# Frozen populations (naturally eligible, by atomic number).
FIT = (1, 2, 4, 6, 9, 11, 12, 15, 16, 17, 19, 20, 29, 30, 32, 37, 38, 41, 44, 46, 48, 52, 55, 56, 57, 60, 62,
       65, 66, 67, 73, 74, 77, 83, 85, 88, 89, 94, 98, 104, 105, 108, 112, 115, 118)
DEVELOPMENT = (3, 8, 14, 18, 25, 31, 35, 40, 42, 43, 45, 78, 82, 84, 107)
CONFIRMATION = (5, 7, 22, 23, 24, 26, 27, 28, 36, 47, 49, 58, 61, 64, 68, 79, 80, 86, 87, 90, 95, 96, 97, 101, 110)
# The whole salted confirmation split (35), eligible or not: everything here is sealed.
CONFIRMATION_SPLIT = (5, 7, 10, 21, 22, 23, 24, 26, 27, 28, 36, 47, 49, 53, 58, 61, 63, 64, 68, 71, 79, 80, 86, 87,
                      90, 92, 93, 95, 96, 97, 101, 102, 110, 113, 116)
SEALED_POPULATION = "gw-element-1/confirmation"

# Arm inventory per confirmation number target, by phrasing. Name = KIND-SPAN-L4.
# Protocol r2 (owner decision 2026-11-21, option 1): the degenerate C2-ctrl (== identity) and
# C2-np (== C2-full numerically) arms are removed. No replacement hypothesis.
ARMS = {
    "A2": ("ID-full-L4", "C2-full-L4", "S-full-L4"),
    "A3": ("ID-full-L4", "C2-full-L4", "S-full-L4"),
}
REFERENCE_ARMS = ("ID-full-L4",)  # each symbol run: its natural readout is the separable reference
PRIMARY = "C2-full-L4"

# Gates (the original Test 3 Stage 2 values; not tuned on this domain).
BOOT, SEED = 10_000, 20261118
GATE_A_LOWER, GATE_A_PHRASING = 0.80, 0.60
GATE_B3_MEAN, GATE_B3_MAX, B3_MAX_SUPPORT = 0.5, 1.0, 10
GATE_OTHER_TOP1 = 0.05
NONWORSE_MARGIN = 0.25  # upper 95% bound of the paired (C2 - S) B3 mean difference must be <= this

FORMULA = "gw-element-1/c2/v1"


def sha(p) -> str:
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def key(z: int) -> str:
    """The entity key of an element in run ids and the runner's `country` field: the atomic number."""
    return str(int(z))


def run_id(rel: str, ph: str, z: int) -> str:
    return f"{rel}|{ph}|{key(z)}"


def verify_populations() -> dict:
    """Re-derive the populations from the registered records; fail closed on any difference."""
    if sha(FEAS) != FEAS_SHA:
        raise SystemExit("FAIL CLOSED: feasibility record differs from artifact 2748")
    asg = json.loads(FEAS.read_text())["split"]["assignment"]
    rep = json.loads((BASELINE / "report.json").read_text())
    integ = json.loads((BASELINE / "integrity.json").read_text())
    if integ.get("status") != "admitted" or not rep["verdict"].startswith("feasible"):
        raise SystemExit("FAIL CLOSED: the natural baseline is not admitted and feasible")
    el = set(rep["naturally_eligible"])
    got = {s: tuple(sorted(int(z) for z, v in asg.items() if v == s and int(z) in el))
           for s in ("fit", "development", "confirmation")}
    split = tuple(sorted(int(z) for z, v in asg.items() if v == "confirmation"))
    want = {"fit": FIT, "development": DEVELOPMENT, "confirmation": CONFIRMATION}
    if got != want or split != CONFIRMATION_SPLIT:
        raise SystemExit(f"FAIL CLOSED: populations differ from the frozen lists: {got}")
    if len(FIT) != 45 or len(DEVELOPMENT) != 15 or len(CONFIRMATION) != 25 or len(CONFIRMATION_SPLIT) != 35:
        raise SystemExit("FAIL CLOSED: population sizes")
    return {"fit": list(FIT), "development": list(DEVELOPMENT), "confirmation": list(CONFIRMATION),
            "confirmation_split": list(CONFIRMATION_SPLIT), "feasibility_sha256": FEAS_SHA,
            "baseline_report_sha256": sha(BASELINE / "report.json")}


def inventory(targets: tuple = CONFIRMATION) -> dict:
    """The exact confirmation arm inventory."""
    per = {ph: len(ARMS[ph]) for ph in PHRASINGS}
    n_num = sum(per.values()) * len(targets)
    n_ref = len(PHRASINGS) * len(REFERENCE_ARMS) * len(targets)
    kinds = {}
    for ph in PHRASINGS:
        for a in ARMS[ph]:
            kinds[a] = kinds.get(a, 0) + len(targets)
    kinds["ID-full-L4 (symbol reference runs)"] = n_ref
    return {"targets": len(targets), "number_targets": len(PHRASINGS) * len(targets),
            "symbol_reference_targets": len(PHRASINGS) * len(targets), "arms_total": n_num + n_ref,
            "identity_arms": sum(v for k, v in kinds.items() if k.startswith("ID-")),
            "constructed_or_reference_arms": sum(v for k, v in kinds.items() if not k.startswith("ID-")),
            "by_arm": kinds}
