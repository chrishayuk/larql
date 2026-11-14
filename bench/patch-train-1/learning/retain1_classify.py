#!/usr/bin/env python3
"""RETAIN-1 classification with the rules fixed in retain1_registration.json (applied after the three runs).
    python3 -I retain1_classify.py"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
reg = json.load(open(os.path.join(HERE, "retain1_registration.json")))
METHODS = ("composition", "N", "R", "D")
out = {"protocol": "RETAIN-1 classification (DISCOVERY ONLY)", "cases": [], "determinism": []}
for p in reg["pairs"]:
    d = json.load(open(os.path.join(HERE, f"retain1_pair{p['index']}_discovery.json")))
    s = json.load(open(os.path.join(HERE, f"stress_cap1_pair{p['index']}_discovery.json")))
    for order, o in d["orders"].items():
        first, second = o["first"], o["second"]
        fa = first == p["A"]
        exp = s["hashes"]["W2_AB_seq" if fa else "W2_BA_seq"]
        comp_exp = s["hashes"]["W_AB_indep" if fa else "W_BA_indep"]
        out["determinism"].append({"pair": p["index"], "order": order, "N_equals_stress_cap1": o["arms"]["N"]["tensor_sha256"] == exp,
                                   "composition_equals_stress_cap1": d["stages"][f"{order}/composition"]["tensor_sha256"] == comp_exp})
        case = {"pair": p["index"], "order": order, "first": first, "second": second, "methods": {}}
        iso = o["first_only"]["first"]
        for m in METHODS:
            sm = o["composition"] if m == "composition" else o["arms"][m]["summary"]
            f, sc = sm["first"], sm["second"]
            keep = [True] + [True] * 3
            iso_prompts = [iso["train_top1"]] + list(iso["para_top1"])
            now_prompts = [f["train_top1"]] + list(f["para_top1"])
            retained_of = [n for i, n in zip(iso_prompts, now_prompts) if i]
            case["methods"][m] = {"first_retained": f["train_top1"], "second_learned": sc["train_top1"], "coexist": sm["both_top1"],
                                  "over_retention": bool(f["train_top1"] and not sc["train_top1"]),
                                  "first_train_rank": f["train_rank"], "second_train_rank": sc["train_rank"],
                                  "first_prompts_top1_in_isolation": int(sum(iso_prompts)), "of_which_still_top1": int(sum(retained_of)),
                                  "first_train_logp_change_vs_isolated": f["train_logp"] - iso["train_logp"],
                                  "second_paraphrase_top1": int(sum(sc["para_top1"])), "partner_dlogp": [f["partner_dlogp"], sc["partner_dlogp"]],
                                  "real_top1": sm["collateral"]["real_top1"], "neutral_delta": sm["collateral"]["neutral_delta"],
                                  "limits_violated": bool(sm["collateral"]["real_top1"] < 0.96 or sm["collateral"]["neutral_delta"] > 0.15)}
        case["advantage_over_N"] = {m: bool(case["methods"][m]["coexist"] and not case["methods"]["N"]["coexist"]) for m in ("R", "D")}
        case["advantage_over_composition"] = {m: bool(case["methods"][m]["coexist"] and not case["methods"]["composition"]["coexist"]) for m in ("N", "R", "D")}
        out["cases"].append(case)
n = len(out["cases"])
out["counts_out_of_6_cases_NOT_independent"] = {
    m: {"first_retained": sum(c["methods"][m]["first_retained"] for c in out["cases"]), "second_learned": sum(c["methods"][m]["second_learned"] for c in out["cases"]),
        "coexist": sum(c["methods"][m]["coexist"] for c in out["cases"]), "over_retention": sum(c["methods"][m]["over_retention"] for c in out["cases"]),
        "limits_violated": sum(c["methods"][m]["limits_violated"] for c in out["cases"])} for m in METHODS}
out["advantage_counts"] = {"R_over_N": sum(c["advantage_over_N"]["R"] for c in out["cases"]), "D_over_N": sum(c["advantage_over_N"]["D"] for c in out["cases"]),
                           "R_over_composition": sum(c["advantage_over_composition"]["R"] for c in out["cases"]),
                           "D_over_composition": sum(c["advantage_over_composition"]["D"] for c in out["cases"]),
                           "N_over_composition": sum(c["advantage_over_composition"]["N"] for c in out["cases"])}
with open(os.path.join(HERE, "retain1_classification.json"), "w") as f:
    json.dump(out, f, indent=1)
    f.write("\n")
print("determinism all true:", all(all(v for k, v in d.items() if k.endswith("stress_cap1")) for d in out["determinism"]))
print(json.dumps(out["counts_out_of_6_cases_NOT_independent"], indent=1))
print(out["advantage_counts"])
for c in out["cases"]:
    print(c["pair"], c["order"], {m: (v["first_train_rank"], v["second_train_rank"]) for m, v in c["methods"].items()})
