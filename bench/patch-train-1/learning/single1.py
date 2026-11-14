#!/usr/bin/env python3
"""PATCH-TRAIN-1 SINGLE-1, phase 1 (DISCOVERY ONLY, protocol not frozen): can ONE fictional entity -> ONE answer be learned by
a rank-r column-scoped patch, and does it transfer across paraphrases WITHOUT leaking onto other entities?

Predeclared (before any run): entity = task A trained_entities[0] of tasks_discovery.json; layer 9 down_proj; the same 256
columns as every earlier run; rho cap 0.05; Adam lr 3e-3; STE through the BF16-rounded forward; MAX STEPS 200; checkpoints at
0 (the gradient-derived init), 5, 10, 20, 40, 80, 120, 160, 200; the REPORTED result is the final step (200), never a
checkpoint chosen by held-out performance. Loss = conditional CE over the twelve declared answers
(-log P(y)/sum_A P(a)) + LAMBDA * full-vocabulary CE (LAMBDA recorded below). Two assignments for the SAME entity: its assigned
answer T, and a CONTROL answer T' (a different answer); each gets its own patch. Every patch is also scored on the other
assignment's answer, and on every OTHER entity (leakage: does the answer move to entities it was not trained on?).

Mechanism diagnostic: for a rank-1 patch the induced change is approximately a * (b . z(x)), z(x) the activation entering
down_proj. At init and at the end we report b.z(x) for the target entity's prompts vs other entities' prompts, and the cosine
between the footprint activations of different entities (is there entity signal at this site and in these columns at all?).

Reported weights are always the exact frozen v2 emulation; the base is digest-checked. No claim of factual learning here.
    python3 -I single1.py --rank 1 --out single1_rank1_discovery.json
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
from torch.func import functional_call  # noqa: E402

import emulate as E  # noqa: E402
import evaluation as V  # noqa: E402
import feasibility as F  # noqa: E402

PROFILE = E.PROFILE_RNE1
RHO, N_COLUMNS, LAMBDA, MAX_STEPS, LR = 0.05, 256, 0.1, 200, 3e-3
CHECKPOINTS = (0, 5, 10, 20, 40, 80, 120, 160, 200)
LAYER = 9


def stats(lps, T, T2, ans):
    """Per-prompt-set metrics from full-vocabulary log-probability vectors."""
    rows = []
    for lp in lps:
        t = lp[T].item()
        al = lp[ans]
        rows.append({"logp_T": t, "logp_Tcontrol": lp[T2].item(), "rank_T": 1 + int((lp > lp[T]).sum().item()),
                     "cond_logp_T": t - torch.logsumexp(al, 0).item(), "mass": torch.logsumexp(al, 0).item(),
                     "inset_top1_is_T": bool(int(al.argmax().item()) == int((ans == T).nonzero().item())),
                     "full_top1_is_T": int(lp.argmax().item()) == T})
    n = len(rows)
    out = {"n": n}
    for k in ("logp_T", "logp_Tcontrol", "cond_logp_T", "mass"):
        out["mean_" + k] = sum(r[k] for r in rows) / n
    out["median_rank_T"] = sorted(r["rank_T"] for r in rows)[n // 2]
    out["frac_inset_top1_is_T"] = sum(r["inset_top1_is_T"] for r in rows) / n
    out["frac_full_top1_is_T"] = sum(r["full_top1_is_T"] for r in rows) / n
    return out


def evaluate(model, tok, tasks, entity, T, T2, ans):
    t = tasks["tasks"]["A"]
    own_train = [V.logprobs(model, tok, tpl.format(e=entity)) for tpl in t["train_templates"]]
    own_para = [V.logprobs(model, tok, tpl.format(e=entity)) for tpl in t["heldout_templates"]]
    others = [e for e in t["trained_entities"] + t["heldout_entities"] if e != entity]
    other_lp = [V.logprobs(model, tok, t["train_templates"][0].format(e=e)) for e in others]
    real = [(V.logprobs(model, tok, f["template"].format(e=f["country"])), V.first_token(tok, f["answer"])) for f in tasks["real_facts"]]
    return {"entity_train_prompts": stats(own_train, T, T2, ans), "entity_paraphrases": stats(own_para, T, T2, ans),
            "other_entities_leakage": stats(other_lp, T, T2, ans),
            "real_top1": sum(int(lp.argmax().item()) == y for lp, y in real) / len(real),
            "real_mean_logp": sum(lp[y].item() for lp, y in real) / len(real),
            "neutral_nll": V.neutral_nll(model, tok, tasks["neutral_text"])}


def loss_and_grad(model, key, w_full, prompts, answer_ids):
    w = w_full.detach().clone().requires_grad_(True)
    total, parts = 0.0, [0.0, 0.0]
    for ids, target in prompts:
        lp = torch.log_softmax(functional_call(model, {key: w}, args=(), kwargs={"input_ids": ids}).logits[0, -1].float(), dim=-1)
        ce = -lp[target]
        cond = -(lp[target] - torch.logsumexp(lp[answer_ids], dim=0))
        ((cond + LAMBDA * ce) / len(prompts)).backward()
        parts[0] += cond.item() / len(prompts)
        parts[1] += ce.item() / len(prompts)
    return parts[0] + LAMBDA * parts[1], parts, w.grad.detach()


def footprint_activations(model, tok, prompts, columns):
    """z(x)[J]: the activation entering layer-LAYER down_proj at the last token, restricted to the footprint columns."""
    mod = model.model.layers[LAYER].mlp.down_proj
    got = []
    h = mod.register_forward_pre_hook(lambda m, args: got.append(args[0][0, -1, columns].detach().double().numpy()))
    with torch.no_grad():
        for p in prompts:
            model(input_ids=tok(p, return_tensors="pt")["input_ids"])
    h.remove()
    return np.stack(got)


def mechanism(Z_target, Z_other, b):
    s_t, s_o = Z_target @ b.T, Z_other @ b.T          # (n, rank)
    out = {"b_dot_z_target_mean": s_t.mean(0).tolist(), "b_dot_z_others_mean": s_o.mean(0).tolist(),
           "b_dot_z_others_std": s_o.std(0).tolist()}
    out["selectivity_dprime"] = ((s_t.mean(0) - s_o.mean(0)) / (s_o.std(0) + 1e-12)).tolist()
    return out


def cos_rows(A, B):
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
    return An @ Bn.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--out", default="single1_rank1_discovery.json")
    args = ap.parse_args()
    global LAYER
    LAYER = args.layer                                   # phase 2 placement sweep; phase 1 used layer 9
    tensor = f"{LAYER}.mlp.down_proj.weight"
    tasks = V.load_tasks("tasks_discovery.json")
    assert tasks["seed"] != V.load_tasks("tasks.json")["seed"], "discovery set must differ from the confirmatory set"
    model, tok = V.load_model(tasks)
    digest0 = V.model_digest(model)
    key = V.hf_key(tensor)
    assert key.split(".")[2] == str(LAYER)
    base_bits = V.weight_bits(model, key)
    rows, ncols = base_bits.shape
    base_f32 = torch.from_numpy(E.bf16_to_f32(base_bits).copy())
    columns = np.sort(np.random.default_rng(F.SEED).choice(ncols, N_COLUMNS, replace=False))
    cols_t = torch.from_numpy(columns)
    rms_w = float(np.sqrt((E.bf16_to_f64(base_bits[:, columns]) ** 2).mean()))
    cap = RHO * rms_w * np.sqrt(rows * N_COLUMNS)
    t = tasks["tasks"]["A"]
    ans = torch.tensor([V.first_token(tok, a) for a in t["answers"]])
    entity = t["trained_entities"][0]
    true_answer = t["assignment"][entity]
    control_answer = t["answers"][(t["answers"].index(true_answer) + 1) % len(t["answers"])]
    assert control_answer != true_answer
    T, Tc = V.first_token(tok, true_answer), V.first_token(tok, control_answer)
    others = [e for e in t["trained_entities"] + t["heldout_entities"] if e != entity]
    train_prompts_txt = [tpl.format(e=entity) for tpl in t["train_templates"]]
    Z_target = footprint_activations(model, tok, train_prompts_txt, columns)
    Z_other = footprint_activations(model, tok, [t["train_templates"][0].format(e=e) for e in others], columns)
    ent_cos = cos_rows(Z_target, Z_other)

    record = {"protocol": "PATCH-TRAIN-1 SINGLE-1 phase 1 (DISCOVERY; NOT frozen; NOT confirmatory)",
              "backend": V.backend_record(tasks), "task_set": f"tasks_discovery.json (seed {tasks['seed']})",
              "predeclared": {"entity": entity, "true_answer": true_answer, "control_answer": control_answer, "layer": LAYER,
                              "tensor": tensor, "columns": N_COLUMNS, "column_seed": F.SEED, "rank": args.rank, "rho": RHO,
                              "update_frobenius_cap": float(cap), "lr": LR, "max_steps": MAX_STEPS, "checkpoints": list(CHECKPOINTS),
                              "objective": "conditional CE over the 12 declared answers + lambda * full-vocabulary CE",
                              "lambda": LAMBDA, "reported": "final step; never a checkpoint selected on held-out performance",
                              "n_train_prompts": len(train_prompts_txt), "n_paraphrase_prompts": len(t["heldout_templates"])},
              "optimisation_approximation": "training forward rounds via torch f32->bf16 cast, straight-through gradient; "
                                            "every reported weight is the exact frozen v2 emulation",
              "entity_signal_at_site": {"cosine_target_vs_others_footprint_activation": {
                  "mean": float(ent_cos.mean()), "min": float(ent_cos.min()), "max": float(ent_cos.max())}},
              "base": evaluate(model, tok, tasks, entity, T, Tc, ans), "runs": {}}
    print("base done", flush=True)

    def applied(a, b):
        new_bits, _ = E.apply_columns(base_bits, columns, a, b, PROFILE)
        return new_bits

    for label, target_tok in (("assigned_answer", T), ("control_answer", Tc)):
        t0 = time.time()
        prompts = [(tok(p, return_tensors="pt")["input_ids"], target_tok) for p in train_prompts_txt]
        _, _, g = loss_and_grad(model, key, base_f32, prompts, ans)
        u, s, vh = torch.linalg.svd(g[:, cols_t], full_matrices=False)
        scale = (cap / 4.0) ** 0.5
        gen = torch.Generator().manual_seed(args.seed)
        a = (-u[:, :args.rank].float() * scale + 1e-3 * torch.randn((rows, args.rank), generator=gen)).requires_grad_(True)
        b = (vh[:args.rank].float() * scale).requires_grad_(True)
        init_b = b.detach().double().numpy().copy()
        opt = torch.optim.Adam([a, b], lr=LR)
        base_cols = base_f32[:, cols_t]
        curve, ckpts, cap_binding = [], {}, 0
        for step in range(MAX_STEPS + 1):
            if step in CHECKPOINTS:
                nb = applied(a.detach().double().numpy(), b.detach().double().numpy())
                with V.patched_weight(model, key, nb):
                    ev = evaluate(model, tok, tasks, entity, target_tok, Tc if target_tok == T else T, ans)
                ev["elements_changed"] = int((nb != base_bits).sum())
                ckpts[str(step)] = ev
                e = ev["entity_train_prompts"]; p = ev["entity_paraphrases"]; o = ev["other_entities_leakage"]
                print(f"{label:16s} step {step:3d} train: rank {e['median_rank_T']:>5} inset {e['frac_inset_top1_is_T']:.2f} "
                      f"full {e['frac_full_top1_is_T']:.2f} | para rank {p['median_rank_T']:>5} inset {p['frac_inset_top1_is_T']:.2f} | "
                      f"others inset-T {o['frac_inset_top1_is_T']:.2f} logpT {o['mean_logp_T']:.2f} | real {ev['real_top1']:.2f} "
                      f"neutral {ev['neutral_nll']:.3f}", flush=True)
            if step == MAX_STEPS:
                break
            opt.zero_grad()
            wc = base_cols + a @ b
            full = base_f32.clone()
            full[:, cols_t] = wc.detach().to(torch.bfloat16).to(torch.float32)
            loss, parts, grad_full = loss_and_grad(model, key, full, prompts, ans)
            wc.backward(grad_full[:, cols_t])
            opt.step()
            with torch.no_grad():
                fro = torch.linalg.norm(a @ b).item()
                if fro > cap:
                    a.mul_(cap / fro)
                    cap_binding += 1
            curve.append({"step": step, "loss": loss, "cond": parts[0], "ce": parts[1], "update_frobenius": min(fro, cap)})
        final_b = b.detach().double().numpy()
        run = {"target_token_answer": true_answer if target_tok == T else control_answer,
               "curve": curve, "checkpoints": ckpts, "cap_binding_steps": cap_binding, "seconds": time.time() - t0,
               "mechanism": {"init": mechanism(Z_target, Z_other, init_b), "final": mechanism(Z_target, Z_other, final_b)}}
        record["runs"][label] = run
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
