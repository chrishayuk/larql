#!/usr/bin/env python3
"""PATCH-TRAIN-1 SINGLE-CAP-1 (DISCOVERY ONLY, protocol not frozen): vary ONLY the update-norm cap.

Fixed (identical to SINGLE-MECH-1): layer 3 down_proj, rank 1, the 256 columns, the ridge-whitened input direction `b` (built
from the training template only, reconstructed here exactly as mech1.py does and checked against mech1's recorded separation),
assigned answer Frozoria -> Lyon only, conditional-12 + 0.1 full-vocabulary CE, Adam lr 3e-3, 200 steps, STE through the BF16-
rounded forward, the same training example, the same initialisation policy (negative gradient at HALF THE ARM'S CAP).
Varied: rho in (0.05 [re-run as the reference and checked against mech1's recorded run], 0.10, 0.20, 0.40), nothing else.

PREDECLARED collateral limits (an arm that exceeds any is recorded as a failure and no larger cap is run):
    real-fact top-1 >= 0.96;  neutral NLL <= base + 0.15;  mean other-entity d log p(target) <= +1.0 nat.
Primary endpoint: the FINAL step. Predeclared checkpoints 0,10,20,50,100,200 are diagnostics, never a selected result.
Continuous selectivity  S = d log P(Lyon | Frozoria) - mean_{other entities} d log P(Lyon | x)  (d = patched minus base).
No claim of factual learning. The base is digest-checked.
    python3 -I cap1.py --out cap1_layer3_discovery.json
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

import emulate as E  # noqa: E402
import evaluation as V  # noqa: E402
import feasibility as F  # noqa: E402
import mech1 as M  # noqa: E402
import single1 as S  # noqa: E402

LAYER, PROFILE = M.LAYER, E.PROFILE_RNE1
RHOS = (0.05, 0.10, 0.20, 0.40)
CHECKPOINTS = M.CHECKPOINTS
LIMITS = {"real_top1_min": 0.96, "neutral_nll_max_increase": 0.15, "others_mean_dlogp_max": 1.0}


def per_entity_logp(model, tok, tasks, ents, target_tok):
    t = tasks["tasks"]["A"]
    return {e: V.logprobs(model, tok, t["train_templates"][0].format(e=e))[target_tok].item() for e in ents}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cap1_layer3_discovery.json")
    args = ap.parse_args()
    tasks = V.load_tasks("tasks_discovery.json")
    assert tasks["seed"] != V.load_tasks("tasks.json")["seed"], "discovery set must differ from the confirmatory set"
    model, tok = V.load_model(tasks)
    digest0 = V.model_digest(model)
    tensor = f"{LAYER}.mlp.down_proj.weight"
    key = V.hf_key(tensor)
    base_bits = V.weight_bits(model, key)
    rows, ncols = base_bits.shape
    base_f32 = torch.from_numpy(E.bf16_to_f32(base_bits).copy())
    columns = np.sort(np.random.default_rng(F.SEED).choice(ncols, M.N_COLUMNS, replace=False))
    cols_t = torch.from_numpy(columns)
    rms_w = float(np.sqrt((E.bf16_to_f64(base_bits[:, columns]) ** 2).mean()))
    t = tasks["tasks"]["A"]
    ans = torch.tensor([V.first_token(tok, a) for a in t["answers"]])
    entity = t["trained_entities"][0]
    true_answer = t["assignment"][entity]
    control_answer = t["answers"][(t["answers"].index(true_answer) + 1) % len(t["answers"])]
    T, Tc = V.first_token(tok, true_answer), V.first_token(tok, control_answer)
    others = [e for e in t["trained_entities"] + t["heldout_entities"] if e != entity]
    train_tpl = t["train_templates"][0]

    # ---- reconstruct the ridge-whitened input direction exactly as mech1.py does
    def last_act(e):
        z, ids = M.capture(model, tok, train_tpl.format(e=e), columns)
        _, last = M.entity_span(tok, train_tpl, e, ids)
        return z[last]
    mu_t = last_act(entity)
    others_train = np.stack([last_act(e) for e in others])
    diff = mu_t - others_train.mean(0)
    neutral_tokens = np.concatenate([M.capture(model, tok, s, columns)[0] for s in tasks["neutral_text"]])
    cov = np.cov(np.concatenate([others_train, neutral_tokens]), rowvar=False)
    b_np = np.linalg.solve(cov + 0.1 * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0]), diff)
    b_np /= np.linalg.norm(b_np)
    b_t = torch.from_numpy(b_np).float().reshape(1, -1)
    mech = json.load(open(os.path.join(HERE, "mech1_layer3_discovery.json")))
    ref_b = mech["input_side"]["white"]["target_value_train"]
    assert abs(float(mu_t @ b_np) - ref_b) < 1e-9, "b_white does not reproduce mech1's recorded value"

    base_eval = S.evaluate(model, tok, tasks, entity, T, Tc, ans)
    base_own = per_entity_logp(model, tok, tasks, [entity], T)[entity]
    base_others = per_entity_logp(model, tok, tasks, others, T)
    base_para = [V.logprobs(model, tok, tpl.format(e=entity))[T].item() for tpl in t["heldout_templates"]]
    record = {"protocol": "PATCH-TRAIN-1 SINGLE-CAP-1 (DISCOVERY; NOT frozen; NOT confirmatory)",
              "backend": V.backend_record(tasks), "task_set": f"tasks_discovery.json (seed {tasks['seed']})",
              "fixed": {"entity": entity, "answer": true_answer, "layer": LAYER, "tensor": tensor, "columns": M.N_COLUMNS,
                        "rank": 1, "b": "ridge-whitened, training template only, unit norm", "lr": S.LR, "max_steps": S.MAX_STEPS,
                        "lambda": S.LAMBDA, "checkpoints": list(CHECKPOINTS), "primary_endpoint": "final step"},
              "varied": {"rho": list(RHOS)}, "collateral_limits": LIMITS,
              "optimisation_approximation": "training forward rounds via torch f32->bf16 cast, straight-through; reported weights are "
                                            "the exact frozen v2 emulation",
              "base": base_eval, "base_own_logp_T": base_own, "arms": {}, "stopped_after": None}
    print("base done; b reproduces mech1", flush=True)

    def applied(a, b):
        return E.apply_columns(base_bits, columns, a, b, PROFILE)[0]

    for rho in RHOS:
        t0 = time.time()
        cap = rho * rms_w * np.sqrt(rows * M.N_COLUMNS)
        prompts = [(tok(tpl.format(e=entity), return_tensors="pt")["input_ids"], T) for tpl in t["train_templates"]]
        _, _, g = S.loss_and_grad(model, key, base_f32, prompts, ans)
        ga = g[:, cols_t] @ b_t.T
        a = (-ga / ga.norm() * (cap / 2.0)).clone().requires_grad_(True)
        opt = torch.optim.Adam([a], lr=S.LR)
        base_cols = base_f32[:, cols_t]
        curve, ckpts, bind = [], {}, 0
        for step in range(S.MAX_STEPS + 1):
            if step in CHECKPOINTS:
                nb = applied(a.detach().double().numpy(), b_np.reshape(1, -1))
                with V.patched_weight(model, key, nb):
                    ev = S.evaluate(model, tok, tasks, entity, T, Tc, ans)
                    own = per_entity_logp(model, tok, tasks, [entity], T)[entity]
                    oth = per_entity_logp(model, tok, tasks, others, T)
                    para = [V.logprobs(model, tok, tpl.format(e=entity))[T].item() for tpl in t["heldout_templates"]]
                d_own = own - base_own
                d_others = {e: oth[e] - base_others[e] for e in others}
                ev["selectivity"] = {"d_logp_target_entity": d_own, "d_logp_others_mean": float(np.mean(list(d_others.values()))),
                                     "S": d_own - float(np.mean(list(d_others.values()))),
                                     "d_logp_others_max": float(max(d_others.values())), "per_entity_d_logp_others": d_others,
                                     "d_logp_paraphrases": [p - b0 for p, b0 in zip(para, base_para)]}
                ev["elements_changed"] = int((nb != base_bits).sum())
                ckpts[str(step)] = ev
            if step == S.MAX_STEPS:
                break
            opt.zero_grad()
            wc = base_cols + a @ b_t
            full = base_f32.clone()
            full[:, cols_t] = wc.detach().to(torch.bfloat16).to(torch.float32)
            loss, parts, grad_full = S.loss_and_grad(model, key, full, prompts, ans)
            wc.backward(grad_full[:, cols_t])
            opt.step()
            with torch.no_grad():
                fro = torch.linalg.norm(a @ b_t).item()
                if fro > cap:
                    a.mul_(cap / fro)
                    bind += 1
            curve.append({"step": step, "loss": loss, "cond": parts[0], "ce": parts[1], "logp_target_train": -parts[1],
                          "update_frobenius": min(fro, cap)})
        a_np = a.detach().double().numpy()
        nb = applied(a_np, b_np.reshape(1, -1))
        ideal_delta = a_np @ b_np.reshape(1, -1)
        realised = E.bf16_to_f64(nb[:, columns]) - E.bf16_to_f64(base_bits[:, columns])
        ideal_w = base_f32.clone()
        ideal_w[:, cols_t] += torch.from_numpy(ideal_delta.astype(np.float32))
        with V.patched_values(model, key, ideal_w):
            ideal_own = per_entity_logp(model, tok, tasks, [entity], T)[entity]
        gap = {"survival_fraction": float((nb != base_bits).sum() / (rows * M.N_COLUMNS)),
               "realised_over_ideal_frobenius": float(np.linalg.norm(realised) / np.linalg.norm(ideal_delta)),
               "cosine_realised_vs_ideal": float((realised * ideal_delta).sum() / (np.linalg.norm(realised) * np.linalg.norm(ideal_delta))),
               "own_logp_T_ideal_f32": ideal_own, "own_logp_T_rounded": base_own + ckpts["200"]["selectivity"]["d_logp_target_entity"]}
        f = ckpts[str(S.MAX_STEPS)]
        viol = []
        if f["real_top1"] < LIMITS["real_top1_min"]:
            viol.append("real_top1")
        if f["neutral_nll"] > base_eval["neutral_nll"] + LIMITS["neutral_nll_max_increase"]:
            viol.append("neutral_nll")
        if f["selectivity"]["d_logp_others_mean"] > LIMITS["others_mean_dlogp_max"]:
            viol.append("others_mean_dlogp")
        arm = {"rho": rho, "update_frobenius_cap": float(cap), "curve": curve, "checkpoints": ckpts, "cap_binding_steps": bind,
               "compiled_gap": gap, "violated_limits": viol, "seconds": time.time() - t0}
        record["arms"][f"rho{rho}"] = arm
        e, p, o, s = f["entity_train_prompts"], f["entity_paraphrases"], f["other_entities_leakage"], f["selectivity"]
        print(f"rho {rho:.2f} FINAL train rank {e['median_rank_T']:>5} full-top1 {e['frac_full_top1_is_T']:.2f} inset {e['frac_inset_top1_is_T']:.2f} "
              f"mass {e['mean_mass']:.2f} | para rank {p['median_rank_T']:>5} | dlogp own {s['d_logp_target_entity']:+.2f} others {s['d_logp_others_mean']:+.2f} "
              f"(max {s['d_logp_others_max']:+.2f}) S {s['S']:+.2f} | others inset-T {o['frac_inset_top1_is_T']:.2f} | real {f['real_top1']:.2f} "
              f"neutral {f['neutral_nll']:.3f} | bind {bind} | violated {viol or 'none'}", flush=True)
        with open(os.path.join(HERE, args.out), "w") as fh:
            json.dump(record, fh, indent=1)
            fh.write("\n")
        if viol:
            record["stopped_after"] = f"rho{rho} violated {viol}; no larger cap run (predeclared)"
            break
    record["model_digest_unchanged_at_end"] = V.model_digest(model) == digest0
    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(record, fh, indent=1)
        fh.write("\n")
    print("model restored:", record["model_digest_unchanged_at_end"], "| stopped:", record["stopped_after"])


if __name__ == "__main__":
    main()
