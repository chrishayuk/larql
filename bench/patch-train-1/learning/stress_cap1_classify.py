#!/usr/bin/env python3
"""STRESS-CAP-1 classification with the rules fixed in stress_cap1_registration.json (applied after the three runs).
    python3 -I stress_cap1_classify.py"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
reg = json.load(open(os.path.join(HERE, "stress_cap1_registration.json")))
out = {"protocol": "STRESS-CAP-1 classification (DISCOVERY ONLY)", "pairs": []}
STACKS = ("3_A_then_indepB", "4_B_then_indepA", "5_A_then_B_learned_on_A", "6_B_then_A_learned_on_B", "7_joint_rank2")
for p in reg["pairs"]:
    d = json.load(open(os.path.join(HERE, f"stress_cap1_pair{p['index']}_discovery.json")))
    A, B = d["arms"], None
    top1 = lambda arm, who: bool(d["arms"][arm][who]["train_top1"])
    isoA, isoB = top1("1_A_only", "A"), top1("2_B_only", "B")
    cls = "both" if isoA and isoB else "only A" if isoA else "only B" if isoB else "neither"
    res = {"index": p["index"], "A": p["A"], "B": p["B"], "isolated_class": cls, "isolated_top1": {"A": isoA, "B": isoB},
           "isolated_train_rank": {"A": d["arms"]["1_A_only"]["A"]["train_rank"], "B": d["arms"]["2_B_only"]["B"]["train_rank"]},
           "cap_binding_steps": {k: v["cap_binding_steps"] for k, v in d["trainings"].items()}, "stacks": {}}
    for arm in STACKS:
        a, b = top1(arm, "A"), top1(arm, "B")
        res["stacks"][arm] = {"A_top1": a, "B_top1": b, "A_rank": d["arms"][arm]["A"]["train_rank"], "B_rank": d["arms"][arm]["B"]["train_rank"],
                              "coexist": a and b, "retention_fails_A": isoA and not a, "retention_fails_B": isoB and not b}
    comp = ("3_A_then_indepB", "4_B_then_indepA", "7_joint_rank2")
    res["rescue"] = {}
    for who, iso in (("A", isoA), ("B", isoB)):
        if not iso:
            for arm in ("5_A_then_B_learned_on_A", "6_B_then_A_learned_on_B"):
                res["rescue"][f"{who}@{arm}"] = {"rescued": res["stacks"][arm][f"{who}_top1"],
                                                 "rescue_over_composition": res["stacks"][arm][f"{who}_top1"] and not any(res["stacks"][c][f"{who}_top1"] for c in comp)}
    res["coexist_after_sequential"] = {"arm5": res["stacks"]["5_A_then_B_learned_on_A"]["coexist"], "arm6": res["stacks"]["6_B_then_A_learned_on_B"]["coexist"]}
    out["pairs"].append(res)
n = len(out["pairs"])
out["counts_out_of_3_NOT_independent"] = {
    "both_learn_in_isolation": sum(r["isolated_class"] == "both" for r in out["pairs"]),
    "only_one_learns": sum(r["isolated_class"] in ("only A", "only B") for r in out["pairs"]),
    "neither_learns": sum(r["isolated_class"] == "neither" for r in out["pairs"]),
    "sequential_rescue_any": sum(any(v["rescued"] for v in r["rescue"].values()) for r in out["pairs"]),
    "sequential_rescue_over_composition_any": sum(any(v["rescue_over_composition"] for v in r["rescue"].values()) for r in out["pairs"]),
    "coexist_after_sequential_arm5": sum(r["coexist_after_sequential"]["arm5"] for r in out["pairs"]),
    "coexist_after_sequential_arm6": sum(r["coexist_after_sequential"]["arm6"] for r in out["pairs"]),
    "retention_fails_despite_isolated_success_any_stack": sum(any(s["retention_fails_A"] or s["retention_fails_B"] for s in r["stacks"].values()) for r in out["pairs"])}
with open(os.path.join(HERE, "stress_cap1_classification.json"), "w") as f:
    json.dump(out, f, indent=1)
    f.write("\n")
print(json.dumps(out["counts_out_of_3_NOT_independent"], indent=1))
for r in out["pairs"]:
    print(r["index"], r["A"], r["B"], r["isolated_class"], r["isolated_train_rank"], {k: (v["A_rank"], v["B_rank"]) for k, v in r["stacks"].items()})
