#!/usr/bin/env python3
"""STRESS-CAP-1 REGISTRATION (DISCOVERY ONLY). Writes stress_cap1_registration.json BEFORE any training of the three pairs.
Reads the preregistered selection output (stress_pair_selection.json); no model is loaded and nothing is trained.
    python3 -I stress_cap1_register.py"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sel = json.load(open(os.path.join(HERE, "stress_pair_selection.json")))
pairs = sel["top10"][:3]
reg = {
    "protocol": "PATCH-TRAIN-2 STRESS-CAP-1 registration (DISCOVERY ONLY; thresholds not used; confirmatory entities unopened)",
    "selection_rule": sel["rule"], "excluded_entities": sel["excluded"], "source": "stress_pair_selection.json (commit e6416aff)",
    "pairs": [{"index": k + 1, "A": p["entities"][0], "B": p["entities"][1], "answers": {"A": p["answers"][0], "B": p["answers"][1]},
               "address_cosine": p["address_cosine"],
               "cross_activation_train_template": {"bA.zA": p["b_i_dot_z_i"], "bB.zB": p["b_j_dot_z_j"], "bA.zB": p["b_i_dot_z_j"], "bB.zA": p["b_j_dot_z_i"],
                                                   "c_AB = bA.zB / bA.zA": p["cross_over_target"][0], "c_BA = bB.zA / bB.zB": p["cross_over_target"][1]}}
              for k, p in enumerate(pairs)],
    "note_on_independence": "the three pairs are the three edges of a triangle on {Brimokvolia, Frosalvolia, Thapylvolia}: they share "
                            "entities, so they are NOT three independent observations and no rate may be inferred",
    "config": {"layer": 3, "rank": 1, "columns": 256, "rho": 0.40, "objective": "conditional-12 + 0.1 full-vocabulary CE", "optimiser": "Adam",
               "lr": 0.003, "steps": 200, "endpoint": "final step", "backend": "PyTorch CPU f32 eager bs1; frozen v2 emulation f64-seq-rne1/v1",
               "arms": ["1 A only", "2 B only", "3 A then independent B", "4 B then independent A", "5 A then B learned on A",
                        "6 B then A learned on B", "7 joint rank-2"], "cap_from": "base W0 footprint RMS", "runner": "d2.py --rho 0.40",
               "earlier_run_of_pair_1_at_rho_0.20": "stress1_discovery.json (commit e25c614d)"},
    "per_pair_recorded": ["isolated learning (arm 1, arm 2)", "sequential learning (arms 5, 6)", "independent composition (arms 3, 4)",
                          "order effects", "per-paraphrase ranks and top-1", "partner and other-entity leakage", "collateral (real-fact, neutral NLL)",
                          "cap-binding steps per training", "complete numerical differences", "address overlap and cross-response c_AB, c_BA per prompt"],
    "classification_rules_fixed_before_running": {
        "learns_in_isolation": "the association's training prompt is full-vocabulary top-1 at the final step in its isolated arm (arm 1 for A, arm 2 for B)",
        "isolated_class": "both | only A | only B | neither",
        "sequential_rescue": "an association that is NOT top-1 in isolation but IS top-1 on its training prompt in arm 5 or arm 6 (reported per arm); "
                             "'rescue over composition' additionally requires it to be top-1 in no composition arm (3, 4, 7)",
        "coexist_after_sequential_learning": "both training prompts are top-1 in arm 5 (reported separately for arm 6)",
        "retention_fails_despite_isolated_success": "an association top-1 in isolation whose training prompt is not top-1 in a stack arm (reported per arm 3-7)",
        "counts": "exact counts out of three pairs, with the independence note above; no success rate is inferred"},
    "causes_to_report_for_failures": ["cap-limited (cap binds on most steps and the loss is still falling or high)",
                                      "interference-limited (partner leakage or cross-response grows in the stack)",
                                      "optimisation-limited (neither of the above)"],
    "not_done": ["calibration", "confirmatory entities", "threshold setting", "native engine or verifier"]}
with open(os.path.join(HERE, "stress_cap1_registration.json"), "w") as f:
    json.dump(reg, f, indent=1)
    f.write("\n")
for p in reg["pairs"]:
    print(p["index"], p["A"], p["answers"]["A"], "|", p["B"], p["answers"]["B"], "cos", round(p["address_cosine"], 3),
          "c_AB", round(p["cross_activation_train_template"]["c_AB = bA.zB / bA.zA"], 2), "c_BA", round(p["cross_activation_train_template"]["c_BA = bB.zA / bB.zB"], 2))
