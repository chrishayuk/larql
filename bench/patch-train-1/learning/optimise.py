#!/usr/bin/env python3
"""PATCH-TRAIN-1 L1 step 5 (CANDIDATE, protocol not frozen): the smallest rounding-aware optimiser experiment.

Learns a rank-K column-scoped patch (a: rows x K, b: K x |J|) on ONE fixed footprint for task A, on the DISCOVERY entity set
(tasks_discovery.json) only. The confirmatory set (tasks.json) is never opened here. Not a result: a discovery run to decide
whether the optimiser has anything to find before the protocol is frozen.

Arms (same footprint, rank, init, step budget, update-norm cap, backend):
  ste       forward uses the BF16-ROUNDED weights; gradient is straight-through across the rounding
  f32_round forward uses the UNROUNDED f32 weights; the trained patch is rounded once at the end ("train, then convert")
  random    a random rank-K patch at the same norm, untrained (negative control)
  base      the unedited model

OPTIMISATION APPROXIMATIONS (NOT part of the numerical contract): the forward rounding during training is torch's f32 -> BF16
cast, and the gradient passes straight through it. The patch that is EVALUATED and REPORTED is always the exact frozen v2
emulation (emulate.apply_columns, f64-seq-rne1/v1) of the learned (a, b). The base model is restored and digest-checked.

    python3 -I optimise.py --rank 4 --steps 40 --rho-cap 0.03 --out optimise_discovery.json
"""
import argparse
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

import emulate as E  # noqa: E402
import evaluation as V  # noqa: E402
import feasibility as F  # noqa: E402

PROFILE = E.PROFILE_RNE1


def train_prompts(tasks, tok, task):
    t = tasks["tasks"][task]
    return [(tok(tpl.format(e=e), return_tensors="pt")["input_ids"], V.first_token(tok, t["assignment"][e]))
            for e in t["trained_entities"] for tpl in t["train_templates"]]


def loss_and_grad_wrt_weight(model, key, w_full, prompts):
    """Mean first-token cross-entropy over the training prompts with `key` replaced by w_full; gradient w.r.t. w_full."""
    w = w_full.detach().clone().requires_grad_(True)
    total = 0.0
    for ids, target in prompts:
        logits = functional_call(model, {key: w}, args=(), kwargs={"input_ids": ids}).logits[0, -1].float()
        loss = -torch.log_softmax(logits, dim=-1)[target] / len(prompts)
        loss.backward()
        total += loss.item()
    return total, w.grad.detach()


def learn(model, tok, tasks, key, base_f32, columns, rows, rank, steps, lr, cap_fro, rounded_forward, seed, init_u, init_v):
    """Adam on (a, b). Init: top-`rank` singular directions of the base gradient (a deterministic, standard start), small."""
    g = torch.Generator().manual_seed(seed)
    a = -torch.from_numpy(init_u[:, :rank].copy()).float() * (cap_fro / 4.0) ** 0.5   # NEGATIVE gradient: lowers the loss
    b = torch.from_numpy(init_v[:rank].copy()).float() * (cap_fro / 4.0) ** 0.5
    a = (a + 1e-3 * torch.randn(a.shape, generator=g)).requires_grad_(True)
    b = b.requires_grad_(True)
    opt = torch.optim.Adam([a, b], lr=lr)
    cols = torch.from_numpy(columns)
    base_cols = base_f32[:, cols]
    prompts = train_prompts(tasks, tok, "A")
    trace = []
    for step in range(steps):
        opt.zero_grad()
        wc = base_cols + a @ b                                    # unrounded f32 weights on the footprint (differentiable)
        used = wc.detach().to(torch.bfloat16).to(torch.float32) if rounded_forward else wc.detach()
        full = base_f32.clone()
        full[:, cols] = used
        loss, grad_full = loss_and_grad_wrt_weight(model, key, full, prompts)
        wc.backward(grad_full[:, cols])                           # straight-through: d(round(x))/dx := 1
        opt.step()
        with torch.no_grad():                                     # project onto the update-norm cap
            fro = torch.linalg.norm(a @ b).item()
            if fro > cap_fro:
                a.mul_(cap_fro / fro)
        trace.append({"step": step, "train_loss": loss, "update_frobenius": min(fro, cap_fro)})
        print(f"  step {step:3d} loss {loss:.4f} |ab| {trace[-1]['update_frobenius']:.4f}", flush=True)
    return a.detach().double().numpy(), b.detach().double().numpy(), trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--rho-cap", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--out", default="optimise_discovery.json")
    args = ap.parse_args()

    tasks = V.load_tasks("tasks_discovery.json")
    assert tasks["seed"] != V.load_tasks("tasks.json")["seed"], "discovery set must differ from the confirmatory set"
    model, tok = V.load_model(tasks)
    digest0 = V.model_digest(model)
    key = V.hf_key(F.TENSOR)
    base_bits = V.weight_bits(model, key)
    rows, ncols = base_bits.shape
    rng = np.random.default_rng(F.SEED)
    columns = np.sort(rng.choice(ncols, F.N_COLUMNS, replace=False))
    base_f32 = torch.from_numpy(E.bf16_to_f32(base_bits).copy())
    rms_w = float(np.sqrt((E.bf16_to_f64(base_bits[:, columns]) ** 2).mean()))
    cap_fro = args.rho_cap * rms_w * np.sqrt(rows * len(columns))

    _, grad = F.task_loss_and_gradient(model, tok, tasks, key, "A")
    u, s, vh = torch.linalg.svd(grad[:, columns], full_matrices=False)
    init_u, init_v = u[:, :args.rank].double().numpy(), vh[:args.rank].double().numpy()

    record = {"protocol": "PATCH-TRAIN-1 L1 optimiser DISCOVERY run (NOT frozen; NOT confirmatory)",
              "backend": V.backend_record(tasks), "task_set": f"tasks_discovery.json (seed {tasks['seed']})", "task": "A",
              "footprint": {"tensor": F.TENSOR, "columns": len(columns), "seed": F.SEED, "rank": args.rank, "profile": PROFILE},
              "hyper": {"steps": args.steps, "lr": args.lr, "rho_cap": args.rho_cap, "update_frobenius_cap": cap_fro},
              "optimisation_approximation": "training forward rounds via torch f32->bf16 cast; gradient is straight-through; "
                                            "reported weights are ALWAYS the exact frozen v2 emulation",
              "base": F.summary(model, tok, tasks, None), "arms": {}}

    def evaluate(a, b):
        new_bits, _ = E.apply_columns(base_bits, columns, a, b, PROFILE)
        with V.patched_weight(model, key, new_bits):
            out = F.summary(model, tok, tasks, None)
        out["elements_changed"] = int((new_bits != base_bits).sum())
        out["realised_update_frobenius"] = float(np.linalg.norm(E.bf16_to_f64(new_bits) - E.bf16_to_f64(base_bits)))
        return out

    for arm, rounded in (("ste", True), ("f32_round", False)):
        t0 = time.time()
        print(arm, flush=True)
        a, b, trace = learn(model, tok, tasks, key, base_f32, columns, rows, args.rank, args.steps, args.lr, cap_fro,
                            rounded, args.seed, init_u, init_v)
        record["arms"][arm] = {"trace": trace, "seconds": time.time() - t0, "evaluated_exact_v2": evaluate(a, b)}
    rg = np.random.default_rng(args.seed)
    ra, rb = rg.standard_normal((rows, args.rank)), rg.standard_normal((args.rank, len(columns)))
    ra *= cap_fro / np.linalg.norm(ra @ rb)
    record["arms"]["random"] = {"evaluated_exact_v2": evaluate(ra, rb)}
    record["model_digest_unchanged_at_end"] = V.model_digest(model) == digest0
    with open(os.path.join(HERE, args.out), "w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    for arm, r in record["arms"].items():
        e = r["evaluated_exact_v2"]
        for sp in ("trained_entities_train_template", "heldout_entities_paraphrase"):
            m = e[f"A.{sp}"]
            print(f"{arm:9s} {sp:34s} logp {m['mean_logp_target']:.2f} rank {m['median_rank_full_vocab']} "
                  f"inset {m['top1_in_answer_set']:.2f} mass {m['mean_logp_answer_set_mass']:.2f}")
        print(f"{'':9s} neutral {e['neutral_nll']:.4f} real_top1 {e['real_top1']:.2f} real_logp {e['real_mean_logp']:.3f}")
    print("model restored:", record["model_digest_unchanged_at_end"])


if __name__ == "__main__":
    main()
