"""Test 3 Stage 4 (holdout): independent admission of derived plans and outputs.

It shares no code with any builder. It reuses the qualified v1 admission
helpers (`gwtemplate_test3_derived_admit.py`, unedited) for:
- recipe hashing;
- per-recipe address, role, input-hash and membership checks;
- documented recompute.

It replaces v1's input boundary, which uses the plan's refusal list, with the
Stage 4 information rule (protocol §4):

- **C1:** the members are exactly the 39 fit countries. No target or holdout
  country is a member.
- **C2:** own = h(Y, capital, p). Plus and minus are exactly the fit set. The
  candidate never reads h(Y, currency, p) or any other holdout or validation
  country.
- **References:**
  - ID = h(Y, capital, p);
  - S = h(Y, currency, p);
  - C0-Dk = h(donor_k, currency, p), where donor_k is the declared donor and
    must be an eligible holdout country.
- **No validation country** is an input anywhere.
- **Every target country** is an eligible holdout country. Every target run's
  country is outside the runner refusal list.
"""
from __future__ import annotations

import gwtemplate_test3_derived_admit as v1

REL, RP = "capital", "currency_code_iso4217"
Captures, rsha, vsha, recompute = v1.Captures, v1.rsha, v1.vsha, v1.recompute


def expected_arms(rel: str, n_donors: int) -> list[str]:
    if rel == RP:
        return ["ID-window-L1"]
    return ["ID-window-L1", "S-window-L1"] + [f"C0-D{k}-L1" for k in range(1, n_donors + 1)] + ["C1-window-L1", "C2-window-L1"]


def check_recipe(rec, l, p, run_id, window, kind, caps, fit: set, eligible: set, validation: set, donor: str | None):
    f = v1.check_recipe(rec, l, p, run_id, window, kind, caps, fit, set())
    rel, ph, Y = run_id.split("|")
    ins = rec.get("inputs", [])
    for i in ins:
        if i[3] in validation:
            f.append(f"validation input {i[3]}")
    roles = {}
    for i in ins:
        roles.setdefault(i[0], set()).add((i[1], i[3]))
    if kind == "C1":
        if roles != {"member": {(RP, c) for c in fit}}:
            f.append("C1 membership is not exactly the fit set")
    elif kind == "C2":
        if roles != {"own": {(REL, Y)}, "plus": {(RP, c) for c in fit}, "minus": {(REL, c) for c in fit}}:
            f.append("C2 roles or membership are not own(Y, capital) + fit set")
    elif kind == "ID":
        if roles != {"copy": {(rel, Y)}}:
            f.append("ID is not the target's own window")
    elif kind == "S":
        if roles != {"copy": {(RP, Y)}}:
            f.append("S is not the target's own currency window")
    elif kind == "C0":
        if donor is None or donor not in eligible or donor == Y or roles != {"copy": {(RP, donor)}} or rec.get("donor") != donor:
            f.append("C0 donor is not the declared eligible holdout donor")
    else:
        f.append(f"unexpected kind {kind}")
    if kind in ("C1", "C2") and (RP, Y) in {(i[1], i[3]) for i in ins}:
        f.append("candidate reads h(Y, currency, p)")
    if kind in ("C1", "C2") and any(i[3] in eligible and not (i[0] == "own" and i[3] == Y) for i in ins):
        f.append("holdout country in a candidate aggregate")
    return f


def admit(plan: dict, index: dict, rows: list[dict] | None, caps, fit: set, eligible: set, validation: set,
          donor_map: dict) -> list[str]:
    fails = []
    refusal = set(plan["holdout"])
    out = {r["id"]: r for r in rows} if rows is not None else None
    for rid, r in plan["runs"].items():
        if r["country"] in refusal or r["country"] not in eligible:
            fails.append(f"{rid}: run country not an eligible holdout country or on the refusal list")
    for t in plan["targets"]:
        rel, ph, Y = t["run"].split("|")
        if Y not in eligible or t["country"] != Y:
            fails.append(f"{t['id']}: target is not an eligible holdout country")
            continue
        dn = donor_map[Y]["donors"]
        if t.get("donors") != dn:
            fails.append(f"{t['id']}: donors differ from the declared map")
        names = [a["name"] for a in t["arms"]]
        if names != expected_arms(rel, len(dn)):
            fails.append(f"{t['id']}: arm inventory {names}")
            continue
        run = plan["runs"][t["run"]]
        for a in t["arms"]:
            kind = a["name"].split("-")[0]
            donor = dn[int(a["name"].split("-")[1][1:]) - 1] if kind == "C0" else None
            if kind == "C0" and a.get("donor") != donor:
                fails.append(f"{t['id']} {a['name']}: arm donor")
            if a["from_layer"] != 1 or a["target_span"] != run["window"] or bool(a.get("identity")) != (kind == "ID"):
                fails.append(f"{t['id']} {a['name']}: from_layer, span or identity flag")
            want = {f"{l}|{p}" for l in range(0, plan["layers"]) for p in run["window"]}
            if set(a["vectors"]) != want:
                fails.append(f"{t['id']} {a['name']}: coverage")
            for key, vid in a["vectors"].items():
                l, p = map(int, key.split("|"))
                e = index.get(vid)
                if e is None:
                    fails.append(f"{t['id']} {a['name']} {key}: unknown vector")
                    continue
                if rsha(e["recipe"]) != e["recipe_sha256"]:
                    fails.append(f"{t['id']} {a['name']} {key}: recipe hash")
                for x in check_recipe(e["recipe"], l, p, t["run"], run["window"], kind, caps, fit, eligible, validation, donor):
                    fails.append(f"{t['id']} {a['name']} {key}: {x}")
                if out is not None:
                    o = out[t["id"]]["arms"][a["name"]]
                    if o["vectors"].get(key) != e["sha256"]:
                        fails.append(f"{t['id']} {a['name']} {key}: output vector hash")
                    if e["recipe_sha256"] not in o["recipes"]:
                        fails.append(f"{t['id']} {a['name']} {key}: recipe not recorded")
            if out is not None and a.get("identity") and not out[t["id"]]["arms"][a["name"]].get("identity_bit_exact"):
                fails.append(f"{t['id']} {a['name']}: identity not bit-exact")
    return fails
