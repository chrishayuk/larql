"""GW-TEMPLATE-FACTORIAL test 2 (relation switching): the intervention plan.

MODEL-FREE. It builds, deterministically from the admitted baseline and the
test 2 feasibility rules (`gwtemplate_test2_feasibility.py`: development
population, windows and donors):

- `plan.json`: the cross-position runner plan (the qualified stage 3
  runner, unchanged);
- `seal-plan.json`: the generic seal plan;
- `ledger.json`: exclusions, inputs and arm accounting.

**Owner decisions (test 2 d1, 2026-11-16):**

- option B, start layers 1 and 16;
- both maps (window, word);
- S, D1/D2 and E1/E2 arms plus identity;
- two donors per target;
- 59 development countries, P2 and P3 only, the holdout untouched.

**Per target cell (Y, relation r, phrasing p; r' is the other relation):**

- `identity`: Y's own window, L16 (bit-exact integrity arm);
- for each start layer a in (1, 16) and each map m in (window, word):
  - `S-{m}-L{a}`: donor (Y, r', p), same entity, other relation;
  - `D{k}-{m}-L{a}`: donor (X'_k, r', p), the decisive arm;
  - `E{k}-{m}-L{a}`: donor (X'_k, r, p), entity leakage within r.

That is 21 arms per cell.

**Maps** (right-aligned, ending one position before the readout, which is
never replaced):

- **window:** capital ", the capital" (P2) / ", the capital city" (P3) and
  ISO "ISO currency code" (P2) / "ISO currency code used" (P3);
- **word:** the single position " capital" or " code".

**Readout candidates:** the union of the capital and ISO 4217 candidate
sets, so every answer class is readable in every arm.

Usage: gwtemplate_test2_plan.py OUT_DIR
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gwtemplate_test2_feasibility as feas  # noqa: E402

BASE = ROOT / "output/gw-template-factorial/baseline-01"
RELATIONS = feas.RELS
PHRASINGS = feas.PHRASINGS
FROM_LAYERS = (1, 16)
MAPS = ("window", "word")
IDENTITY_LAYER = 16
LAYERS = 34
UNION = "capital+currency_code_iso4217"
ARMS_PER_CELL = 1 + len(FROM_LAYERS) * len(MAPS) * 5


def sha(p) -> str:
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def other(rel: str) -> str:
    return feas.other(rel)


def cell_arms(cells: dict, runs: dict, y: str, donors: list[str], rel: str, p: str) -> list[dict]:
    """The 21 arms of target cell (y, rel, p), registering every run used."""
    def add(cid):
        c = cells[cid]
        runs.setdefault(cid, {"tokens": c["token_ids"], "text": c["text"]})
        return c

    tid = f"{rel}|{p}|{y}"
    tc = add(tid)

    def spans(country, r):
        return feas.windows(cells, country, p)[r]

    def arm(name, donor_country, donor_rel, m, a, identity=False):
        did = f"{donor_rel}|{p}|{donor_country}"
        dc = add(did)
        src, tgt = spans(donor_country, donor_rel)[m], spans(y, rel)[m]
        if len(src) != len(tgt):
            raise ValueError(f"{tid} {name}: span lengths differ")
        out = {"name": name, "donor_run": did, "source_span": src, "target_span": tgt,
               "source_span_ids": [dc["token_ids"][i] for i in src],
               "target_span_ids": [tc["token_ids"][i] for i in tgt], "from_layer": a, "mode": "replace"}
        if identity:
            out["identity"] = True
        return out

    arms = [arm("identity", y, rel, "window", IDENTITY_LAYER, identity=True)]
    for a in FROM_LAYERS:
        for m in MAPS:
            arms.append(arm(f"S-{m}-L{a}", y, other(rel), m, a))
            arms += [arm(f"D{k}-{m}-L{a}", x, other(rel), m, a) for k, x in enumerate(donors, 1)]
            arms += [arm(f"E{k}-{m}-L{a}", x, rel, m, a) for k, x in enumerate(donors, 1)]
    return arms


def union_candidates(bplan: dict) -> list[int]:
    return sorted(set().union(*(bplan["candidate_sets"][r] for r in RELATIONS)))


def seal_plan(plan: dict) -> dict:
    spans = {}
    for t in plan["targets"]:
        for a in t["arms"]:
            for rid, s in ((t["run"], a["target_span"]), (a["donor_run"], a["source_span"])):
                lo, hi = spans.get(rid, (s[0], s[-1]))
                spans[rid] = (min(lo, s[0]), max(hi, s[-1]))
    return {"schema": "larql.gwbreadcrumb2.seal-plan.v1", "container": plan["container"],
            "runs": [{"id": rid, "tokens": plan["runs"][rid]["tokens"], "positions": [lo, hi]}
                     for rid, (lo, hi) in sorted(spans.items())]}


def build():
    bplan = json.loads((BASE / "plan.json").read_text())
    cells = {c["id"]: c for c in bplan["cells"]}
    brec = {}
    for line in (BASE / "run/baseline-out.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            brec[r["id"]] = r["readout"]["next_logit_sha256"]
    f = feas.build()
    if f["geometry_faults"] or f["population"]["no_donor"]:
        raise ValueError("feasibility faults; refusing to build")
    runs, targets = {}, []
    for y in sorted(f["donors"]):
        donors = f["donors"][y]["donors"]
        for rel in RELATIONS:
            for p in PHRASINGS:
                tid = f"{rel}|{p}|{y}"
                targets.append({"id": f"t2|{tid}", "run": tid, "candidate_set": UNION, "relation": rel,
                                "other_relation": other(rel), "phrasing": p, "country": y, "donors": donors,
                                "expect_natural_logit_sha256": brec[tid],
                                "arms": cell_arms(cells, runs, y, donors, rel, p)})
    plan = {"schema": "larql.gwtemplate.xpos-plan.v1", "container": bplan["container"],
            "tokenizer_sha256": bplan["tokenizer_sha256"], "layers": LAYERS, "top_k": bplan["top_k"],
            "kv_dump": False, "candidate_sets": {UNION: union_candidates(bplan)}, "runs": runs, "targets": targets}
    ledger = {"population": {k: f["population"][k] for k in
                             ("joint", "holdout_removed", "subject_echo_removed", "development", "no_donor")},
              "excluded_phrasings": ["P1"], "donors": {y: f["donors"][y]["donors"] for y in sorted(f["donors"])},
              "feasibility_sha256": "sha256:" + hashlib.sha256(
                  (json.dumps(f, sort_keys=True)).encode()).hexdigest()}
    return plan, seal_plan(plan), ledger


def main(argv):
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(argv[0])
    out.mkdir(parents=False)
    plan, seal, ledger = build()
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    (out / "seal-plan.json").write_text(json.dumps(seal, indent=1) + "\n")
    summary = {}
    for rel in RELATIONS:
        ts = [t for t in plan["targets"] if t["relation"] == rel]
        summary[rel] = {"target_countries": len({t["country"] for t in ts}), "target_cells": len(ts),
                        "arms": sum(len(t["arms"]) for t in ts)}
    ledger["summary"] = summary
    ledger["totals"] = {"target_cells": len(plan["targets"]), "arms": sum(len(t["arms"]) for t in plan["targets"]),
                        "arms_per_cell": ARMS_PER_CELL, "runs": len(plan["runs"]), "seal_runs": len(seal["runs"])}
    ledger["inputs"] = {"baseline_plan": sha(BASE / "plan.json"), "baseline_out": sha(BASE / "run/baseline-out.jsonl"),
                        "feasibility_script": sha(feas.__file__), "builder": sha(__file__)}
    ledger["plan_sha256"] = sha(out / "plan.json")
    ledger["seal_plan_sha256"] = sha(out / "seal-plan.json")
    (out / "ledger.json").write_text(json.dumps(ledger, indent=1, sort_keys=True) + "\n")
    print(json.dumps(ledger["totals"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
