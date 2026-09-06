#!/usr/bin/env python3
"""ADDRESS-SELECT-1 Stage 1 — the single select-v1 evaluation run.

Runs ONCE. Every learned parameter, the layer, and the acceptance threshold are
frozen by the pre-evaluation amendment recorded on the address-select-1 registry
record before this script was allowed to read the roster:

    family    S2 factorised relation -> entity
    layer     L30
    lam_rel   0.01
    lam_ent   0.001
    threshold 0.945990

The selector is fitted on the BURNED discrimination-v1 development split and
applied unchanged. Nothing is re-fitted, re-tuned or re-thresholded here.

Reports both, per the frozen reporting requirement:
  (a) the preregistered conjunction gate, applied mechanically
      >= 90% accepted-and-correct on positives AND <= 5% false acceptance on
      EACH negative family separately (8 rows/family => zero permitted)
  (b) unthresholded selector accuracy and score distributions, per stratum and
      per negative family, so a mechanical "0 accepted" cannot hide whether the
      underlying ranking problem is 95% correct or 30% correct

No editing. Stage 1 is a pure reader test: 128 clean forwards at L30.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
ROSTER = CAMPAIGN / "rosters" / "select-v1"
DISC = CAMPAIGN / "results" / "discrimination-v1"
MODEL = Path("/Users/christopherhay/.cache/huggingface/hub/models--google--gemma-3-4b-it"
             "/snapshots/093f9f388b31de276ce2de164bdc2081324b9767")

sys.path.insert(0, str(CAMPAIGN))
from dev_selectors import s2_scores, unit  # noqa: E402

LAYER = 30
LAM_REL = 0.01
LAM_ENT = 0.001
THRESHOLD = 0.945990
GATE_COVERAGE = 0.90
GATE_FALSE_ACCEPT = 0.05
EXPECTED_ROSTER_SHA = "bda1dfeff46d1efe824257e56a563f58f72aec99cd36fe56410a897481635040"

STRATA = ("structure_shift", "entity_shift", "double_shift")
NEG_FAMILIES = ("negative_uninstalled", "negative_near_name",
                "negative_wrong_relation", "negative_unrelated")


def load_roster():
    body = (ROSTER / "roster.jsonl").read_text()
    got = hashlib.sha256(body.encode()).hexdigest()
    if got != EXPECTED_ROSTER_SHA:
        raise SystemExit(f"REFUSE: roster sha256 {got} != frozen {EXPECTED_ROSTER_SHA}")
    return [json.loads(line) for line in body.splitlines()]


def capture(rows):
    from binding_runtime import Runtime
    run = Runtime(MODEL, layers=(LAYER,))
    states = []
    for i, row in enumerate(rows):
        _, cap = run.forward(row["prompt"])
        states.append(cap[LAYER]["x"])
        if (i + 1) % 25 == 0:
            print(f"  captured {i+1}/{len(rows)}", flush=True)
    return np.stack(states).astype(np.float64)


def frozen_selector_inputs():
    """Burned development material the frozen selector was fitted on."""
    caps = np.load(DISC / "captures.npz", allow_pickle=True)
    fixture = json.loads((DISC / "fixture.json").read_text())
    rows = fixture["development"]
    S = np.asarray(caps[f"development_L{LAYER}"], dtype=np.float64)
    train = [i for i, r in enumerate(rows) if r["group"] == "train"]
    return S[train], [rows[i]["relation"] for i in train], [rows[i]["label"] for i in train]


def main() -> None:
    rows = load_roster()
    print(f"roster verified: {len(rows)} rows, sha256 {EXPECTED_ROSTER_SHA[:16]}...", flush=True)

    print(f"capturing L{LAYER} states for {len(rows)} prompts (no editing)", flush=True)
    states = capture(rows)

    enroll = [i for i, r in enumerate(rows) if r["group"] == "enrollment"]
    addr = states[enroll]
    addr_rel = [rows[i]["relation"] for i in enroll]
    addr_key = [(rows[i]["entity"], rows[i]["relation"]) for i in enroll]

    Xtr, rel_tr, lab_tr = frozen_selector_inputs()
    kw = {"Xtr": Xtr, "rel_tr": rel_tr, "lab_tr": lab_tr, "addr_rel": addr_rel,
          "lam_rel": LAM_REL, "lam_ent": LAM_ENT}

    report = {
        "frozen_configuration": {
            "family": "S2 factorised relation -> entity", "layer": LAYER,
            "lam_rel": LAM_REL, "lam_ent": LAM_ENT, "threshold": THRESHOLD,
            "fitted_on": "discrimination-v1 development/train (burned)",
        },
        "roster_sha256": EXPECTED_ROSTER_SHA,
        "gate": {"coverage": GATE_COVERAGE, "false_accept": GATE_FALSE_ACCEPT},
        "strata": {}, "negatives": {},
    }

    # (a) + (b) on positives, per stratum
    for stratum in STRATA:
        qi = [i for i, r in enumerate(rows) if r["stratum"] == stratum]
        sc = s2_scores(states[qi], addr, **kw)
        top = sc.argmax(axis=1)
        best = sc[np.arange(len(qi)), top]
        correct = [addr_key[t] == (rows[q]["entity"], rows[q]["relation"])
                   for q, t in zip(qi, top)]
        rel_ok = [addr_rel[t] == rows[q]["relation"] for q, t in zip(qi, top)]
        ent_ok = [addr_key[t][0] == rows[q]["entity"] for q, t in zip(qi, top)]
        accepted = best >= THRESHOLD
        report["strata"][stratum] = {
            "n": len(qi),
            "binding_correct_unthresholded": int(sum(correct)),
            "relation_correct_unthresholded": int(sum(rel_ok)),
            "entity_correct_unthresholded": int(sum(ent_ok)),
            "accepted": int(accepted.sum()),
            "accepted_and_correct": int(sum(a and c for a, c in zip(accepted, correct))),
            "score_max": float(best.max()), "score_min": float(best.min()),
            "score_mean": float(best.mean()), "score_median": float(np.median(best)),
        }

    # (a) + (b) on negatives, per family, never pooled
    for family in NEG_FAMILIES:
        qi = [i for i, r in enumerate(rows) if r["group"] == family]
        sc = s2_scores(states[qi], addr, **kw)
        best = sc.max(axis=1)
        report["negatives"][family] = {
            "n": len(qi),
            "false_accepts": int((best >= THRESHOLD).sum()),
            "permitted": int(np.floor(GATE_FALSE_ACCEPT * len(qi))),
            "score_max": float(best.max()), "score_min": float(best.min()),
            "score_mean": float(best.mean()), "score_median": float(np.median(best)),
        }

    n_pos = sum(report["strata"][s]["n"] for s in STRATA)
    acc = sum(report["strata"][s]["accepted_and_correct"] for s in STRATA)
    bind = sum(report["strata"][s]["binding_correct_unthresholded"] for s in STRATA)
    coverage_ok = acc >= GATE_COVERAGE * n_pos
    rejection_ok = all(report["negatives"][f]["false_accepts"] <= report["negatives"][f]["permitted"]
                       for f in NEG_FAMILIES)
    report["verdict"] = {
        "accepted_and_correct": f"{acc}/{n_pos}",
        "required": f">={int(np.ceil(GATE_COVERAGE * n_pos))}/{n_pos}",
        "binding_correct_unthresholded": f"{bind}/{n_pos}",
        "coverage_pass": bool(coverage_ok),
        "rejection_pass": bool(rejection_ok),
        "GATE": "PASS" if (coverage_ok and rejection_ok) else "FAIL",
    }

    np.savez(HERE / "select_v1_states.npz", states=states.astype(np.float32))
    body = json.dumps(report, sort_keys=True, indent=2)
    report_hash = hashlib.sha256(body.encode()).hexdigest()
    (HERE / "select_v1_result.json").write_text(
        json.dumps({**report, "report_sha256": report_hash}, indent=2) + "\n")
    print(body)
    print("\nreport_sha256", report_hash)


if __name__ == "__main__":
    main()
