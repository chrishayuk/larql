"""GW-TEMPLATE-FACTORIAL test 2 (relation switching): model-free feasibility.

MODEL-FREE. No inference. Reads only the test 1 population proposal and the
admitted natural baseline plan (token IDs, spans and accepted first-token
sets). Nothing here is frozen or authorised.

Checks:

1. **Development population:** the joint capital x ISO 4217 population,
   minus the 28-country test 3 holdout (never read beyond its ID list),
   minus the capital subject-echo countries.
2. **Window geometry:** in P2 and P3, for every development country, the
   capital prompt's post-subject, pre-readout positions and the ISO prompt's
   relation phrase have equal length and fixed token IDs, so a right-aligned
   window ending one position before the readout maps position by position.
   The relation-word map is the single position " capital" <-> " code".
3. **Donors:** for each target Y, the different-entity donors X' have the
   same subject token count as Y, and the four answer sets
   r(Y), r'(Y), r(X'), r'(X') are pairwise disjoint first-token sets, for
   both relations. Two per target, by salted hash.
4. **Arm and cost table** for the owner's layer options.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POP = ROOT / "output/gw-template-factorial/test1-population-01.json"
BASE = ROOT / "output/gw-template-factorial/baseline-01/plan.json"
RELS = ("capital", "currency_code_iso4217")
PHRASINGS = ("P2", "P3")
DONOR_SALT = "gw-template-factorial/test2-donors/v1|"
DONORS = 2
EXPECT_WINDOW = {"P2": 3, "P3": 4}
ARMS_PER_MAP_LAYER = 5  # S, D1, D2, E1, E2
TEST1_SECONDS_PER_ARM = (13 * 3600 + 38 * 60 - (9 * 3600 + 53 * 60) - 24 * 60 - 5) / 9087
LAYER_OPTIONS = {"L16 only": [16], "L1 control + L16": [1, 16], "profile 1,8,12,16,20": [1, 8, 12, 16, 20]}


def sha(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def other(rel: str) -> str:
    return RELS[1 - RELS.index(rel)]


def windows(cells: dict, c: str, p: str) -> dict:
    """Window and relation-word positions for both relations in phrasing p."""
    cap, iso = cells[f"capital|{p}|{c}"], cells[f"currency_code_iso4217|{p}|{c}"]
    s_end = cap["subject"]["tokens"][-1]
    cw = list(range(s_end + 1, cap["readout"]))
    iw = list(range(iso["readout"] - len(cw), iso["readout"]))
    return {
        "capital": {"window": cw, "word": cap["relation"]["tokens"], "ids": cap["token_ids"], "readout": cap["readout"]},
        "currency_code_iso4217": {"window": iw, "word": iso["relation"]["tokens"], "ids": iso["token_ids"],
                                  "readout": iso["readout"]},
    }


def build() -> dict:
    pop = json.loads(POP.read_text())
    base = json.loads(BASE.read_text())
    cells = {c["id"]: c for c in base["cells"]}
    holdout = set(pop["proposed_test3_entity_holdout"]["countries"])
    echo = set(pop["subject_echo"]["capital"]) | set(pop["subject_echo"]["currency_code_iso4217"])
    joint = pop["populations"]["joint_countries"]
    dev = sorted(c for c in joint if c not in holdout and c not in echo)
    ledger = {"joint": len(joint), "holdout_removed": len(holdout & set(joint)),
              "subject_echo_removed": sorted(echo & set(joint) - holdout)}

    # Window geometry.
    geometry, faults = {}, []
    for p in PHRASINGS:
        wids = {r: set() for r in RELS}
        words = {r: set() for r in RELS}
        for c in dev:
            w = windows(cells, c, p)
            for r in RELS:
                if any(f"{r}|{p}|{c}" not in cells for r in RELS):
                    faults.append(f"{c} {p}: missing baseline cell")
                ids = w[r]["ids"]
                wids[r].add(tuple(ids[i] for i in w[r]["window"]))
                words[r].add(tuple(ids[i] for i in w[r]["word"]))
                if len(w[r]["window"]) != EXPECT_WINDOW[p]:
                    faults.append(f"{c} {p} {r}: window length {len(w[r]['window'])}")
                if w[r]["window"][-1] != w[r]["readout"] - 1:
                    faults.append(f"{c} {p} {r}: window does not end before the readout")
                if not set(w[r]["word"]) <= set(w[r]["window"]):
                    faults.append(f"{c} {p} {r}: relation word outside the window")
        geometry[p] = {r: {"window_token_ids": sorted(wids[r]), "word_token_ids": sorted(words[r]),
                           "window_length": EXPECT_WINDOW[p]} for r in RELS}
        for r in RELS:
            if len(wids[r]) != 1 or len(words[r]) != 1:
                faults.append(f"{p} {r}: window or word token IDs vary across countries")

    # Donors.
    def acc(r, p, c):
        return set(cells[f"{r}|{p}|{c}"]["accepted_first_tokens"])

    def slen(c):
        return len(cells[f"capital|P2|{c}"]["subject"]["tokens"])

    donors, no_donor = {}, []
    for y in dev:
        ok = []
        for x in dev:
            if x == y or slen(x) != slen(y):
                continue
            good = True
            for p in PHRASINGS:
                sets = [acc(r, p, z) for z in (y, x) for r in RELS]
                if any(sets[i] & sets[j] for i in range(4) for j in range(i + 1, 4)):
                    good = False
            if good:
                ok.append(x)
        ok.sort(key=lambda x: hashlib.sha256((DONOR_SALT + y + "|" + x).encode()).hexdigest())
        if len(ok) < DONORS:
            no_donor.append(y)
        else:
            donors[y] = {"donors": ok[:DONORS], "eligible_donors": len(ok)}

    targets = sorted(donors)
    cells_n = len(targets) * len(RELS) * len(PHRASINGS)
    costs = {}
    for name, layers in LAYER_OPTIONS.items():
        arms = cells_n * (1 + 2 * len(layers) * ARMS_PER_MAP_LAYER)
        costs[name] = {"layers": layers, "maps": ["window", "word"], "cells": cells_n, "arms": arms,
                       "estimated_hours": round(arms * TEST1_SECONDS_PER_ARM / 3600, 1)}

    # Cross-relation first-token overlap within a country (would make S arms unreadable).
    self_overlap = sorted(c for c in dev for p in PHRASINGS if acc("capital", p, c) & acc("currency_code_iso4217", p, c))
    return {
        "schema": "larql.gwtemplate.test2-feasibility.v1",
        "status": "MODEL-FREE feasibility for the test 2 design; nothing frozen or authorised; no inference",
        "inputs": {"population_sha256": sha(POP), "baseline_plan_sha256": sha(BASE), "script_sha256": sha(Path(__file__))},
        "population": {**ledger, "development": len(dev), "development_countries": dev,
                       "targets_with_donors": len(targets), "no_donor": no_donor,
                       "cross_relation_self_overlap": self_overlap},
        "geometry": geometry, "geometry_faults": faults,
        "donors": donors, "seconds_per_arm_test1": round(TEST1_SECONDS_PER_ARM, 3), "cost": costs,
    }


def main(argv):
    if len(argv) != 1:
        print("usage: gwtemplate_test2_feasibility.py OUT_JSON", file=sys.stderr)
        return 2
    rep = build()
    Path(argv[0]).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    return 1 if rep["geometry_faults"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
