#!/usr/bin/env python3
"""PATCH-TRAIN-1 L1 step 6 (DISCOVERY DIAGNOSTIC, protocol not frozen): does a bigger footprint / cap learn which answer
belongs to which entity, or only raise the probability of answer-set words?

Decomposition used throughout (A = the twelve-token answer set, y = the target answer):
    log P(y|x) = log P(y | y in A, x)  +  log P(A|x)
      full-vocab target log p   conditional log p (entity-specific)   answer-set mass log p (global bias)
The deciding metric is the CONDITIONAL one (and in-set top-1), never training loss alone.

Fixed across arms: rank, training prompts, init policy (negative gradient top-K directions at the base model, at half the cap),
optimiser (Adam), step budget, backend, objective (full-vocabulary cross-entropy), evaluation. Varied: rho cap and footprint.
Footprints are NESTED: 256 = the committed run's columns, 512 and 1024 add random columns to them.

Controls per arm:
  init_only  the same initial direction, NO optimisation (does the 40-step result just reproduce one gradient-derived step?)
  shuffled   the same arm trained on a SHUFFLED entity->answer assignment of the trained entities, evaluated against the TRUE
             assignment (a global answer-word bias helps both equally; real association learning helps only the true one)
Reported weights are ALWAYS the exact frozen v2 emulation. Discovery set only; tasks.json (confirmatory) is never opened for data.

    python3 -I diagnostic.py --out diagnostic_discovery.json
"""
import argparse
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import emulate as E  # noqa: E402
import evaluation as V  # noqa: E402
import feasibility as F  # noqa: E402
import optimise as O  # noqa: E402

PROFILE = E.PROFILE_RNE1
ARMS = (("rho0.03_cols256", 0.03, 256), ("rho0.05_cols256", 0.05, 256), ("rho0.05_cols512", 0.05, 512),
        ("rho0.05_cols1024", 0.05, 1024))
CHANCE = 1.0 / 12


def binom_tail(k, n, p=CHANCE):
    """One-sided exact P(X >= k). Prompts of one entity are not independent, so this is indicative only."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def prompts_for(tok, tasks, assignment):
    t = tasks["tasks"]["A"]
    return [(tok(tpl.format(e=e), return_tensors="pt")["input_ids"], V.first_token(tok, assignment[e]))
            for e in t["trained_entities"] for tpl in t["train_templates"]]


def shuffled_assignment(tasks, seed):
    t = tasks["tasks"]["A"]
    ents = t["trained_entities"]
    answers = [t["assignment"][e] for e in ents]
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(len(ents))
        if all(answers[perm[i]] != answers[i] for i in range(len(ents))):
            break
    out = dict(t["assignment"])
    out.update({e: answers[perm[i]] for i, e in enumerate(ents)})
    return out


def decompose(res_rows, assignment_true):
    """Per split: full-vocab / conditional / mass / in-set top-1 / rank / margin, plus per-entity margins."""
    by = {}
    for r in res_rows:
        by.setdefault(r["split"], []).append(r)
    out = {}
    for sp, v in by.items():
        n = len(v)
        correct = sum(bool(x["top1_in_answer_set"]) for x in v)
        ents = {}
        for x in v:
            ents.setdefault(x["entity"], []).append(x)
        out[sp] = {"n_prompts": n, "n_entities": len(ents),
                   "full_logp_target": sum(x["logp_target"] for x in v) / n,
                   "conditional_logp_target": sum(x["logp_target"] - x["logp_answer_set_mass"] for x in v) / n,
                   "answer_set_mass_logp": sum(x["logp_answer_set_mass"] for x in v) / n,
                   "in_set_top1": correct / n, "in_set_top1_correct": correct,
                   "binomial_p_vs_chance_indicative": binom_tail(correct, n),
                   "median_rank_full_vocab": sorted(x["rank_full_vocab"] for x in v)[n // 2],
                   "top1_full_vocab": sum(bool(x["top1_full_vocab"]) for x in v) / n,
                   "mean_margin_in_set": sum(x["margin_in_set"] for x in v) / n,
                   "entities_with_all_prompts_correct": sum(all(bool(y["top1_in_answer_set"]) for y in ys) for ys in ents.values()),
                   "per_entity_margin": {e: sum(y["margin_in_set"] for y in ys) / len(ys) for e, ys in ents.items()}}
    return out


def evaluate_patch(model, tok, tasks, key, base_bits, columns, a, b):
    new_bits, _ = E.apply_columns(base_bits, columns, a, b, PROFILE)
    with V.patched_weight(model, key, new_bits):
        res = V.evaluate_suite(model, tok, tasks, which=("A",))
    s = V.summarise(res)
    return {"decomposition": decompose(res["tasks"]["A"], None),
            "collateral": {"neutral_nll": s["neutral_nll_per_token"], "real_top1": s["real_facts"]["top1"],
                           "real_mean_logp": s["real_facts"]["mean_logp_target"]},
            "elements_changed": int((new_bits != base_bits).sum()),
            "realised_update_frobenius": float(np.linalg.norm(E.bf16_to_f64(new_bits) - E.bf16_to_f64(base_bits)))}


def init_factors(model, key, base_f32, columns, prompts, rank, cap_fro):
    """Negative-gradient top-`rank` directions at the base model, scaled to half the cap. Deterministic given the prompts."""
    _, grad_full = O.loss_and_grad_wrt_weight(model, key, base_f32, prompts)
    u, s, vh = torch.linalg.svd(grad_full[:, torch.from_numpy(columns)], full_matrices=False)
    scale = (cap_fro / 4.0) ** 0.5
    return (-u[:, :rank].float() * scale), (vh[:rank].float() * scale)


def optimise(model, key, base_f32, columns, prompts, a0, b0, steps, lr, cap_fro, seed):
    g = torch.Generator().manual_seed(seed)
    a = (a0 + 1e-3 * torch.randn(a0.shape, generator=g)).clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    opt = torch.optim.Adam([a, b], lr=lr)
    cols = torch.from_numpy(columns)
    base_cols = base_f32[:, cols]
    trace = []
    for step in range(steps):
        opt.zero_grad()
        wc = base_cols + a @ b
        full = base_f32.clone()
        full[:, cols] = wc.detach().to(torch.bfloat16).to(torch.float32)      # optimisation approximation: STE through rounding
        loss, grad_full = O.loss_and_grad_wrt_weight(model, key, full, prompts)
        wc.backward(grad_full[:, cols])
        opt.step()
        with torch.no_grad():
            fro = torch.linalg.norm(a @ b).item()
            if fro > cap_fro:
                a.mul_(cap_fro / fro)
        trace.append({"step": step, "train_loss": loss, "update_frobenius": min(fro, cap_fro)})
    return a.detach().double().numpy(), b.detach().double().numpy(), trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--out", default="diagnostic_discovery.json")
    args = ap.parse_args()

    tasks = V.load_tasks("tasks_discovery.json")
    assert tasks["seed"] != V.load_tasks("tasks.json")["seed"], "discovery set must differ from the confirmatory set"
    model, tok = V.load_model(tasks)
    digest0 = V.model_digest(model)
    key = V.hf_key(F.TENSOR)
    base_bits = V.weight_bits(model, key)
    rows, ncols = base_bits.shape
    base_f32 = torch.from_numpy(E.bf16_to_f32(base_bits).copy())
    rng = np.random.default_rng(F.SEED)
    first256 = rng.choice(ncols, 256, replace=False)                          # identical to the committed run's footprint
    rest = rng.permutation(np.setdiff1d(np.arange(ncols), first256))
    order = np.concatenate([first256, rest])                                  # nested footprints: order[:n]
    true_assign = tasks["tasks"]["A"]["assignment"]
    shuf_assign = shuffled_assignment(tasks, args.seed)
    true_prompts = prompts_for(tok, tasks, true_assign)
    shuf_prompts = prompts_for(tok, tasks, shuf_assign)

    record = {"protocol": "PATCH-TRAIN-1 L1 footprint/cap DIAGNOSTIC (DISCOVERY; NOT frozen; NOT confirmatory)",
              "backend": V.backend_record(tasks), "task_set": f"tasks_discovery.json (seed {tasks['seed']})", "task": "A",
              "fixed": {"rank": args.rank, "steps": args.steps, "lr": args.lr, "objective": "full-vocabulary cross-entropy",
                        "profile": PROFILE, "seed": args.seed, "tensor": F.TENSOR},
              "shuffle": {"seed": args.seed, "n_trained_entities_reassigned": int(sum(
                  shuf_assign[e] != true_assign[e] for e in tasks["tasks"]["A"]["trained_entities"]))},
              "optimisation_approximation": "training forward rounds via torch f32->bf16 cast, straight-through gradient; "
                                            "every reported weight is the exact frozen v2 emulation",
              "base": None, "arms": {}}
    zeros_a, zeros_b = np.zeros((rows, args.rank)), np.zeros((args.rank, 256))
    record["base"] = evaluate_patch(model, tok, tasks, key, base_bits, np.sort(first256), zeros_a, zeros_b)
    print("base done", flush=True)
    for name, rho, n in ARMS:
        columns = np.sort(order[:n])
        rms_w = float(np.sqrt((E.bf16_to_f64(base_bits[:, columns]) ** 2).mean()))
        cap = rho * rms_w * np.sqrt(rows * n)
        arm = {"rho": rho, "columns": n, "update_frobenius_cap": float(cap)}
        for variant, prompts in (("true", true_prompts), ("shuffled", shuf_prompts)):
            t0 = time.time()
            a0, b0 = init_factors(model, key, base_f32, columns, prompts, args.rank, cap)
            if variant == "true":
                arm["init_only"] = evaluate_patch(model, tok, tasks, key, base_bits, columns, a0.double().numpy(), b0.double().numpy())
            a, b, trace = optimise(model, key, base_f32, columns, prompts, a0, b0, args.steps, args.lr, cap, args.seed)
            ev = evaluate_patch(model, tok, tasks, key, base_bits, columns, a, b)
            ev["trace_first_last"] = [trace[0], trace[-1]]
            ev["seconds"] = time.time() - t0
            arm["optimised_" + variant] = ev
            d = ev["decomposition"]["trained_entities_train_template"]
            h = ev["decomposition"]["heldout_entities_paraphrase"]
            print(f"{name:18s} {variant:8s} train: full {d['full_logp_target']:.2f} cond {d['conditional_logp_target']:.2f} "
                  f"mass {d['answer_set_mass_logp']:.2f} inset {d['in_set_top1']:.2f} | heldout cond {h['conditional_logp_target']:.2f} "
                  f"inset {h['in_set_top1']:.2f} | {ev['seconds']:.0f}s", flush=True)
        record["arms"][name] = arm
        with open(os.path.join(HERE, args.out), "w") as f:
            json.dump(record, f, indent=1)
            f.write("\n")
    record["model_digest_unchanged_at_end"] = V.model_digest(model) == digest0
    with open(os.path.join(HERE, args.out), "w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    print("model restored:", record["model_digest_unchanged_at_end"])


if __name__ == "__main__":
    main()
