#!/usr/bin/env python3
"""PATCH-TRAIN-1 L1 step 7 (DISCOVERY, protocol not frozen): full-vocabulary cross-entropy vs conditional answer-set
cross-entropy, with a patch-direction discrimination diagnostic.

Objectives (same footprint, rank, init policy, optimiser, steps, prompts, backend):
  ce    L = -log P(y|x)                                     (full vocabulary)
  cond  L = -log [ P(y|x) / sum_{a in A} P(a|x) ]            (softmax restricted to the twelve declared answer tokens)
For each objective: TRUE entity->answer mapping, SHUFFLED mapping (trained entities only), and the gradient-initialised
NO-OPTIMISATION control. Plus the unedited base.

Patch-direction diagnostic: cosine similarity between the TRUE-label and SHUFFLED-label updates, before optimisation (the
gradient-derived init) and after, on (i) the f32 update a@b and (ii) the effective BF16-ROUNDED change of the weights that the
frozen v2 emulation actually produces. Also across objectives. Plus: how well each SHUFFLED patch fits its OWN (shuffled)
labels, to separate "the patch cannot encode arbitrary mappings" from "it is insensitive to the labels".

Reported weights are always the exact frozen v2 emulation. Discovery set only. No result here is a claim of factual learning.
    python3 -I objectives.py --out objectives_discovery.json
"""
import argparse
import copy
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.func import functional_call  # noqa: E402

import diagnostic as D  # noqa: E402
import emulate as E  # noqa: E402
import evaluation as V  # noqa: E402
import feasibility as F  # noqa: E402

PROFILE = E.PROFILE_RNE1
RHO, N_COLUMNS = 0.05, 256


def loss_and_grad(model, key, w_full, prompts, objective, answer_ids):
    """Mean loss over the training prompts under `objective`, and its gradient w.r.t. the full weight tensor."""
    w = w_full.detach().clone().requires_grad_(True)
    total = 0.0
    for ids, target in prompts:
        lp = torch.log_softmax(functional_call(model, {key: w}, args=(), kwargs={"input_ids": ids}).logits[0, -1].float(), dim=-1)
        if objective == "ce":
            loss = -lp[target]
        else:
            loss = -(lp[target] - torch.logsumexp(lp[answer_ids], dim=0))
        (loss / len(prompts)).backward()
        total += loss.item() / len(prompts)
    return total, w.grad.detach()


def init_factors(model, key, base_f32, columns, prompts, objective, answer_ids, rank, cap_fro):
    _, g = loss_and_grad(model, key, base_f32, prompts, objective, answer_ids)
    u, s, vh = torch.linalg.svd(g[:, torch.from_numpy(columns)], full_matrices=False)
    scale = (cap_fro / 4.0) ** 0.5
    return -u[:, :rank].float() * scale, vh[:rank].float() * scale


def optimise(model, key, base_f32, columns, prompts, objective, answer_ids, a0, b0, steps, lr, cap_fro, seed):
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
        full[:, cols] = wc.detach().to(torch.bfloat16).to(torch.float32)     # optimisation approximation (STE)
        loss, grad_full = loss_and_grad(model, key, full, prompts, objective, answer_ids)
        wc.backward(grad_full[:, cols])
        opt.step()
        with torch.no_grad():
            fro = torch.linalg.norm(a @ b).item()
            if fro > cap_fro:
                a.mul_(cap_fro / fro)
        trace.append({"step": step, "train_loss": loss, "update_frobenius": min(fro, cap_fro)})
    return a.detach().double().numpy(), b.detach().double().numpy(), trace


def cosine(x, y):
    x, y = x.ravel(), y.ravel()
    n = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / n) if n > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--out", default="objectives_discovery.json")
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
    columns = np.sort(rng.choice(ncols, N_COLUMNS, replace=False))             # identical to the earlier runs' 256 columns
    rms_w = float(np.sqrt((E.bf16_to_f64(base_bits[:, columns]) ** 2).mean()))
    cap = RHO * rms_w * np.sqrt(rows * N_COLUMNS)
    t = tasks["tasks"]["A"]
    answer_ids = torch.tensor([V.first_token(tok, a) for a in t["answers"]])
    assigns = {"true": t["assignment"], "shuffled": D.shuffled_assignment(tasks, args.seed)}
    prompts = {k: D.prompts_for(tok, tasks, v) for k, v in assigns.items()}
    tasks_shuf = copy.deepcopy(tasks)
    tasks_shuf["tasks"]["A"]["assignment"] = assigns["shuffled"]

    def realised(a, b):
        new_bits, _ = E.apply_columns(base_bits, columns, a, b, PROFILE)
        return E.bf16_to_f64(new_bits[:, columns]) - E.bf16_to_f64(base_bits[:, columns])

    record = {"protocol": "PATCH-TRAIN-1 L1 objective comparison (DISCOVERY; NOT frozen; NOT confirmatory)",
              "backend": V.backend_record(tasks), "task_set": f"tasks_discovery.json (seed {tasks['seed']})", "task": "A",
              "fixed": {"rank": args.rank, "steps": args.steps, "lr": args.lr, "rho": RHO, "columns": N_COLUMNS,
                        "update_frobenius_cap": float(cap), "profile": PROFILE, "seed": args.seed, "tensor": F.TENSOR},
              "shuffle": {"seed": args.seed, "n_trained_entities_reassigned": int(sum(
                  assigns["shuffled"][e] != assigns["true"][e] for e in t["trained_entities"]))},
              "optimisation_approximation": "training forward rounds via torch f32->bf16 cast, straight-through gradient; "
                                            "every reported weight is the exact frozen v2 emulation",
              "base": D.evaluate_patch(model, tok, tasks, key, base_bits, columns, np.zeros((rows, args.rank)),
                                       np.zeros((args.rank, N_COLUMNS))), "runs": {}}
    print("base done", flush=True)
    f32_updates, bf16_updates = {}, {}
    for objective in ("ce", "cond"):
        for variant in ("true", "shuffled"):
            t0 = time.time()
            a0, b0 = init_factors(model, key, base_f32, columns, prompts[variant], objective, answer_ids, args.rank, cap)
            a, b, trace = optimise(model, key, base_f32, columns, prompts[variant], objective, answer_ids, a0, b0, args.steps,
                                   args.lr, cap, args.seed)
            a0n, b0n = a0.double().numpy(), b0.double().numpy()
            run = {"init_only": D.evaluate_patch(model, tok, tasks, key, base_bits, columns, a0n, b0n),
                   "optimised": D.evaluate_patch(model, tok, tasks, key, base_bits, columns, a, b),
                   "trace_first_last": [trace[0], trace[-1]]}
            if variant == "shuffled":      # does the patch fit ITS OWN (shuffled) labels?
                own = D.evaluate_patch(model, tok, tasks_shuf, key, base_bits, columns, a, b)["decomposition"]
                run["fit_to_own_shuffled_labels_trained_template"] = own["trained_entities_train_template"]
            f32_updates[(objective, variant, "init")] = a0n @ b0n
            f32_updates[(objective, variant, "final")] = a @ b
            bf16_updates[(objective, variant, "init")] = realised(a0n, b0n)
            bf16_updates[(objective, variant, "final")] = realised(a, b)
            run["seconds"] = time.time() - t0
            record["runs"][f"{objective}_{variant}"] = run
            d = run["optimised"]["decomposition"]["trained_entities_train_template"]
            h = run["optimised"]["decomposition"]["heldout_entities_paraphrase"]
            print(f"{objective:5s} {variant:8s} train: full {d['full_logp_target']:.2f} cond {d['conditional_logp_target']:.2f} "
                  f"mass {d['answer_set_mass_logp']:.2f} inset {d['in_set_top1']:.2f} | heldout cond {h['conditional_logp_target']:.2f} "
                  f"inset {h['in_set_top1']:.2f} | {run['seconds']:.0f}s", flush=True)
    cos = {"f32_update": {}, "bf16_rounded_change": {}}
    for kind, upd in (("f32_update", f32_updates), ("bf16_rounded_change", bf16_updates)):
        for obj in ("ce", "cond"):
            for stage in ("init", "final"):
                cos[kind][f"true_vs_shuffled/{obj}/{stage}"] = cosine(upd[(obj, "true", stage)], upd[(obj, "shuffled", stage)])
        for var in ("true", "shuffled"):
            for stage in ("init", "final"):
                cos[kind][f"ce_vs_cond/{var}/{stage}"] = cosine(upd[("ce", var, stage)], upd[("cond", var, stage)])
        for key_ in sorted(upd):
            cos[kind][f"init_vs_final/{key_[0]}/{key_[1]}"] = cosine(upd[(key_[0], key_[1], "init")], upd[(key_[0], key_[1], "final")]) \
                if key_[2] == "init" else None
        cos[kind] = {k: v for k, v in cos[kind].items() if v is not None}
    record["cosine_similarity"] = cos
    record["model_digest_unchanged_at_end"] = V.model_digest(model) == digest0
    with open(os.path.join(HERE, args.out), "w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    for kind, v in cos.items():
        print(kind)
        for k, x in v.items():
            print(f"  {k:36s} {x:.3f}")
    print("model restored:", record["model_digest_unchanged_at_end"])


if __name__ == "__main__":
    main()
