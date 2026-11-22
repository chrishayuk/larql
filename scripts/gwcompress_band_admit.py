"""GW-COMPRESS-1: independent admission for band-limited runner plans and records.

It shares no code with the band runner. It reuses the Test 3 v1 recipe
checker (`gwtemplate_test3_derived_admit.check_recipe`, unchanged), which
re-derives every vector's address, target cell, kind, roles, membership
and input hashes from the capture. Only the coverage rule changes: an arm
covers exactly carriers [from_layer - 1, to_layer] × its span.

**Pre-run** (`admit_plan`): coverage, recipe hash, the v1 recipe check at
every injected (layer, position), the spent holdout, the readout.

**Post-run** (`admit_records`), against the plan and the Stage 1b capture:
- the record's band, replacement and firing counts equal the plan's;
- the recorded vector hash at every address equals the index;
- natural span carriers equal the captured natural carriers (all layers);
- per arm, span carriers below the band equal the captured natural ones,
  carriers inside the band equal the injected vectors, and every carrier
  above the band exists (it was computed: the runner refuses any firing
  there);
- `propagation_fixpoint.bit_exact` for every arm with layers above the band;
- identity arms bit-exact.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gwtemplate_test3_derived_admit as adm  # noqa: E402

SPENT_HOLDOUT = frozenset("AF AS AW BF CR CZ ET GH IN KY LA MM MW NE NG PK RS SB SN SS SY TO TR UA VE VI YE ZA".split())


def band(a: dict) -> tuple[int, int]:
    return a["from_layer"] - 1, a["to_layer"]


def admit_plan(plan: dict, index: dict, caps: adm.Captures, dev: list[str]) -> list[str]:
    fails = []
    fit = adm.fit_set(dev)
    holdout = set(plan["holdout"])
    if not SPENT_HOLDOUT <= holdout:
        fails.append("plan holdout list omits spent holdout countries")
    for rid, r in plan["runs"].items():
        if r["country"] in SPENT_HOLDOUT or r["country"] in holdout:
            fails.append(f"{rid}: holdout country")
    for t in plan["targets"]:
        run = plan["runs"][t["run"]]
        for a in t["arms"]:
            w = f"{t['id']} {a['name']}"
            kind = a["name"].split("-")[0]
            lo, hi = band(a)
            if not (0 <= lo <= hi < plan["layers"]):
                fails.append(f"{w}: invalid band [{lo}, {hi}]")
                continue
            span = a["target_span"]
            if any(p not in run["window"] for p in span) or any(p + 1 >= len(run["tokens"]) for p in span):
                fails.append(f"{w}: span outside the window or at the readout")
            want = {f"{l}|{p}" for l in range(lo, hi + 1) for p in span}
            if set(a["vectors"]) != want:
                fails.append(f"{w}: coverage")
            for key, vid in a["vectors"].items():
                l, p = map(int, key.split("|"))
                e = index.get(vid)
                if e is None:
                    fails.append(f"{w} {key}: unknown vector")
                    continue
                if adm.rsha(e["recipe"]) != e["recipe_sha256"]:
                    fails.append(f"{w} {key}: recipe hash")
                if plan["vector_store"]["index"].get(vid, {}).get("sha256") != e["sha256"]:
                    fails.append(f"{w} {key}: plan index hash differs from the builder index")
                if plan["vector_store"]["index"].get(vid, {}).get("recipe_sha256") != e["recipe_sha256"]:
                    fails.append(f"{w} {key}: plan recipe hash differs from the builder index")
                for x in adm.check_recipe(e["recipe"], l, p, t["run"], run["window"], kind, caps, fit, holdout):
                    fails.append(f"{w} {key}: {x}")
    return fails


def admit_records(plan: dict, index: dict, rows: list[dict], caps: adm.Captures) -> list[str]:
    fails = []
    out = {r["id"]: r for r in rows}
    if set(out) != {t["id"] for t in plan["targets"]}:
        return ["records do not cover exactly the plan's targets"]
    for t in plan["targets"]:
        rec = out[t["id"]]
        cap = caps.entries[t["run"]]["vector_sha256"]
        nat = rec["natural"]["span_carrier_sha256"]
        win = plan["runs"][t["run"]]["window"]
        for l in range(plan["layers"]):
            for p in win:
                k = f"{l}|{p}"
                if nat.get(k) != cap.get(k):
                    fails.append(f"{t['id']} natural carrier {k} differs from the capture")
        for a in t["arms"]:
            w = f"{t['id']} {a['name']}"
            r = rec["arms"].get(a["name"])
            if r is None:
                fails.append(f"{w}: missing record")
                continue
            lo, hi = band(a)
            n = (hi - lo + 1) * len(a["target_span"])
            if r["band"] != [lo, hi] or r["replacements"] != n or r["firings"] != n:
                fails.append(f"{w}: band or counts")
            injected = {}
            for key, vid in a["vectors"].items():
                injected[key] = index[vid]["sha256"]
                if r["vectors"].get(key) != index[vid]["sha256"]:
                    fails.append(f"{w} {key}: output vector hash")
                if index[vid]["recipe_sha256"] not in r["recipes"]:
                    fails.append(f"{w} {key}: recipe not recorded")
            sc = r["span_carrier_sha256"]
            for l in range(plan["layers"]):
                for p in a["target_span"]:
                    k = f"{l}|{p}"
                    if l < lo and sc.get(k) != cap.get(k):
                        fails.append(f"{w} {k}: carrier below the band is not natural")
                    if lo <= l <= hi and sc.get(k) != injected[k]:
                        fails.append(f"{w} {k}: carrier in the band is not the injected vector")
                    if l > hi and sc.get(k) is None:
                        fails.append(f"{w} {k}: no carrier above the band")
            if hi + 1 < plan["layers"]:
                if not (r.get("propagation_fixpoint") or {}).get("bit_exact"):
                    fails.append(f"{w}: no bit-exact propagation fixpoint")
                if (r["propagation_fixpoint"] or {}).get("replayed_addresses") != (plan["layers"] - hi - 1) * len(a["target_span"]):
                    fails.append(f"{w}: fixpoint replay count")
            if a.get("identity") and not r.get("identity_bit_exact"):
                fails.append(f"{w}: identity not bit-exact")
    return fails
