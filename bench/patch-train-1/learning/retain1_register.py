#!/usr/bin/env python3
"""RETAIN-1 REGISTRATION (DISCOVERY ONLY). Writes retain1_registration.json BEFORE any registered run. No model is loaded.
    python3 -I retain1_register.py"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
stress = json.load(open(os.path.join(HERE, "stress_cap1_registration.json")))
reg = {
    "protocol": "PATCH-TRAIN-2 RETAIN-1 registration (DISCOVERY ONLY; emulation-consistent; thresholds not used; calibration and confirmation not started)",
    "pairs": stress["pairs"], "pairs_source": "stress_cap1_registration.json (commit 8c36c701), the same three pairs",
    "note_on_independence": stress["note_on_independence"],
    "orders": "both A then B and B then A for every pair (six pair-order cases, which are NOT independent: shared entities)",
    "fixed_unchanged": {"layer": 3, "rank": 1, "columns": 256, "column_seed": 20261113, "rho": 0.40, "cap_from": "base W0 footprint RMS", "optimiser": "Adam",
                        "lr": 0.003, "steps": 200, "endpoint": "final step", "backend": "PyTorch CPU f32 eager bs1; frozen v2 emulation f64-seq-rne1/v1",
                        "address": "ridge-whitened, training template only (as STRESS-CAP-1)", "rank_footprint_cap_optimiser_steps_changed": False},
    "objectives": {
        "per_task_loss L_x": "conditional-12 + 0.1 * full-vocabulary CE on task x's ORIGINAL training prompt (the objective used throughout)",
        "N new-task only (control)": "L = L_second; must reproduce STRESS-CAP-1 arm 5/6 bit for bit (tensor sha256 compared)",
        "R rehearsal": "L = L_second + 1.0 * L_first, L_first on the FIRST patch's original training prompt only",
        "D distillation": "L = L_second + 1.0 * mean over RET of KL( P_W1(.|p) || P_student(.|p) ), full-vocabulary, last-token distribution; teacher = the "
                          "first-patch-only model W1; RET = the first patch's training prompt plus the training-template prompt of every other discovery "
                          "entity except the two patched ones (23 prompts)",
        "weights": "LAMBDA_R = 1.0 and LAMBDA_D = 1.0, one declared value each, NOT tuned and not swept; any other weight would be a new registration",
        "initialisation": "negative gradient of the arm's own total objective at the source model W1, at half the cap (identical to STRESS-CAP-1 for N)",
        "baselines_reported_alongside": ["independent composition (second patch learned on W0, rebound to W1)", "first patch only", "second patch only"]},
    "never_used_for_training_or_distillation": ["held-out paraphrase templates", "real-fact prompts", "neutral texts", "the confirmatory set"],
    "reported_per_case": ["both targets' full-vocabulary top-1 and rank", "per-paraphrase rank, top-1 and log p", "per-task objective before (at W0 and at the first-patch model) and after",
                          "retention against the first patch's isolated model, per prompt", "whether the new association was learned", "partner and other-entity leakage",
                          "real-fact top-1 and log p, neutral NLL", "BF16 numerics and source binding (refusal when the first patch is removed)", "chain replay bit-exact",
                          "cap-binding steps", "base-model digest before and after"],
    "classification_rules_fixed_before_running": {
        "first_retained": "the first patch's training prompt is full-vocabulary top-1 at the final step",
        "second_learned": "the second patch's training prompt is full-vocabulary top-1 at the final step",
        "coexist": "both of the above (the critical test: retention without preventing new learning)",
        "paraphrase_retention": "of the first patch's prompts (training prompt and 3 paraphrases) that are top-1 in the first-patch-only model, how many are still top-1; log-p change reported, no threshold",
        "over_retention": "first retained but second not learned",
        "advantage_over_new_task_only": "the method coexists where N does not (per pair-order case)",
        "advantage_over_composition": "the method coexists where independent composition does not",
        "counts": "exact counts out of six pair-order cases per method, and per-pair detail; the cases share entities, so NO rate or general claim is made",
        "determinism_checks": "N reproduces the STRESS-CAP-1 stage tensors (arm 5/6) and composition reproduces arms 3/4 by sha256; a mismatch is a defect, reported, not a result"},
    "declared_failure_outcomes": [
        "R and D are no better than N: a retention loss at the declared weight does not preserve the first patch at this rank, footprint and cap",
        "retention improves but the second association is not learned (over-retention): preserving the first is not cumulative learning",
        "both patches cannot be top-1 together under any of the objectives: a capacity or interference limit of this single-rank, shared-footprint design",
        "R succeeds but D does not (or the reverse): reported; distillation is the exploratory arm and may preserve behaviours of the teacher that are wrong or miss ones the retention set does not cover",
        "any arm violating the discovery collateral limits (real-fact top-1 < 0.96, neutral NLL > base + 0.15): reported as a failure of that arm"],
    "if_successful": "reported as a discovery success for the cases concerned only; a separate confirmatory protocol is proposed afterwards; no general continual-learning claim",
    "not_done": ["calibration", "confirmatory entities", "threshold setting", "native engine or verifier", "weight sweeps", "other ranks, footprints or caps"],
    "code_smoke_test": "retain1.py was run once for 3 steps on one order of pair 1 to check that it executes; its outputs were discarded and are not results"}
with open(os.path.join(HERE, "retain1_registration.json"), "w") as f:
    json.dump(reg, f, indent=1)
    f.write("\n")
print("registered:", [p["A"] + "/" + p["B"] for p in reg["pairs"]])
