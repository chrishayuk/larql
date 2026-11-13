#!/usr/bin/env python3
"""PATCH-TRAIN-1 SINGLE-MECH-1 (DISCOVERY ONLY, protocol not frozen): layer 3, rank 1.

Hypothesis: a rank-one patch dW = a b^T can exploit entity-specific FFN-input activations to change the target answer's logits
selectively. dh(x) = a * (b . z(x)), z(x) the activation entering layer-3 down_proj (restricted to the 256 footprint columns).

  A. INPUT SIDE (no training of `a`). Capture z at EVERY token position for the target entity (Frozoria) and the 23 other
     entities. Build candidate input directions from the TRAINING template only (discovery data):
        mean   b = mu_target - mu_others            at the entity's last token
        white  b = Sigma_bg^-1 (mu_target - mu_others), Sigma_bg from the others' entity-token activations plus every token of the
               neutral texts (so b avoids firing on generic text), ridge 0.1*tr/d
        random a random direction (control)
     Report b.z for: the entity's last token, the last token ("is"), the generic template tokens, post-entity tokens, and every
     token of the neutral texts and real-fact prompts; target-vs-others separation (d') on the TRAINING template AND on the three
     held-out paraphrases (transfer across templates, b never saw them).
  B. OUTPUT SIDE. With b FIXED (unit norm), learn only `a` (640x1) through the BF16-rounded forward (straight-through), same
     objective, optimiser, lr, rho cap and 200 steps as SINGLE-1 (no sweep). Assigned answer and control answer.
  C. Compiled gap: the f32 update a b^T vs the exact frozen-v2 BF16-rounded change, and the loss/logit difference between the
     f32-ideal weights and the rounded weights.
No result here is a claim of factual learning. The base is digest-checked.
    python3 -I mech1.py --out mech1_layer3_discovery.json
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
import single1 as S  # noqa: E402

LAYER = 3
PROFILE = E.PROFILE_RNE1
RHO, N_COLUMNS, MAX_STEPS, LR = S.RHO, S.N_COLUMNS, S.MAX_STEPS, S.LR
CHECKPOINTS = (0, 10, 20, 50, 100, 200)


def capture(model, tok, text, columns):
    """z at every position, restricted to the footprint columns: (seq, |J|), and the token ids."""
    got = {}
    h = model.model.layers[LAYER].mlp.down_proj.register_forward_pre_hook(
        lambda m, a: got.__setitem__("z", a[0][0][:, columns].detach().double().numpy()))
    ids = tok(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        model(input_ids=ids)
    h.remove()
    return got["z"], ids[0].tolist()


def entity_span(tok, template, entity, full_ids):
    pre, post = template.split("{e}")[0], None
    first = len(tok(pre)["input_ids"]) if pre else 1                 # BOS counted
    last = len(tok(pre + entity)["input_ids"]) - 1
    assert tok(pre + entity)["input_ids"][:last + 1] == full_ids[:last + 1], "entity token span misaligned"
    return first, last


def d_prime(t, o):
    return float((np.mean(t) - np.mean(o)) / (np.std(o) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--out", default="mech1_layer3_discovery.json")
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
    columns = np.sort(np.random.default_rng(F.SEED).choice(ncols, N_COLUMNS, replace=False))
    cols_t = torch.from_numpy(columns)
    rms_w = float(np.sqrt((E.bf16_to_f64(base_bits[:, columns]) ** 2).mean()))
    cap = RHO * rms_w * np.sqrt(rows * N_COLUMNS)
    t = tasks["tasks"]["A"]
    ans = torch.tensor([V.first_token(tok, a) for a in t["answers"]])
    entity = t["trained_entities"][0]
    true_answer = t["assignment"][entity]
    control_answer = t["answers"][(t["answers"].index(true_answer) + 1) % len(t["answers"])]
    T, Tc = V.first_token(tok, true_answer), V.first_token(tok, control_answer)
    others = [e for e in t["trained_entities"] + t["heldout_entities"] if e != entity]
    train_tpl, para_tpls = t["train_templates"][0], t["heldout_templates"]

    # ------------------------------------------------------------------ A. input side
    def entity_acts(tpl, e):
        text = tpl.format(e=e)
        z, ids = capture(model, tok, text, columns)
        first, last = entity_span(tok, tpl, e, ids)
        return {"z": z, "first": first, "last": last, "n": len(ids)}

    acts = {tpl: {e: entity_acts(tpl, e) for e in [entity] + others} for tpl in [train_tpl] + para_tpls}
    neutral = [capture(model, tok, s, columns)[0] for s in tasks["neutral_text"]]
    real = [capture(model, tok, f["template"].format(e=f["country"]), columns)[0] for f in tasks["real_facts"]]
    neutral_tokens = np.concatenate(neutral)                                   # every token of the neutral texts
    real_tokens = np.concatenate(real)

    def at_entity_last(tpl, e):
        a = acts[tpl][e]
        return a["z"][a["last"]]

    mu_t = at_entity_last(train_tpl, entity)
    others_train = np.stack([at_entity_last(train_tpl, e) for e in others])
    mu_o = others_train.mean(0)
    diff = mu_t - mu_o
    bg = np.concatenate([others_train, neutral_tokens])
    cov = np.cov(bg, rowvar=False)
    ridge = 0.1 * np.trace(cov) / cov.shape[0]
    b_white = np.linalg.solve(cov + ridge * np.eye(cov.shape[0]), diff)
    rng = np.random.default_rng(args.seed)
    b_rand = rng.standard_normal(N_COLUMNS)
    candidates = {"mean": diff, "white": b_white, "random": b_rand}
    candidates = {k: v / np.linalg.norm(v) for k, v in candidates.items()}
    candidates["random"] *= np.sign(candidates["random"] @ diff) or 1.0        # sign convention: target positive

    def separation(b, tpl):
        e_last = {e: float(at_entity_last(tpl, e) @ b) for e in [entity] + others}
        last_tok = {e: float(acts[tpl][e]["z"][-1] @ b) for e in [entity] + others}
        post = {e: [float(x) for x in acts[tpl][e]["z"][acts[tpl][e]["last"] + 1:-1] @ b] for e in [entity] + others}
        pre = {e: [float(x) for x in acts[tpl][e]["z"][:acts[tpl][e]["first"]] @ b] for e in [entity] + others}
        return {"entity_last_token": {"target": e_last[entity], "others_mean": float(np.mean([e_last[e] for e in others])),
                                      "others_std": float(np.std([e_last[e] for e in others])),
                                      "dprime": d_prime([e_last[entity]], [e_last[e] for e in others])},
                "final_token": {"target": last_tok[entity], "others_mean": float(np.mean([last_tok[e] for e in others])),
                                "dprime": d_prime([last_tok[entity]], [last_tok[e] for e in others])},
                "pre_entity_tokens_mean_abs": float(np.mean(np.abs([x for v in pre.values() for x in v]))) if any(pre.values()) else None,
                "post_entity_tokens_target_mean": float(np.mean(post[entity])) if post[entity] else None,
                "post_entity_tokens_others_mean": float(np.mean([x for e in others for x in post[e]])) if any(post[e] for e in others) else None}

    input_side = {}
    for name, b in candidates.items():
        thr = 0.5 * (float(mu_t @ b) + float(mu_o @ b))                          # midpoint between target and others on the training template
        sn, sr = neutral_tokens @ b, real_tokens @ b
        input_side[name] = {"train_template": separation(b, train_tpl),
                            "paraphrases": {tpl: separation(b, tpl) for tpl in para_tpls},
                            "target_value_train": float(mu_t @ b), "midpoint_threshold": thr,
                            "generic_firing": {"neutral_tokens_n": int(len(sn)), "neutral_frac_above_midpoint": float((sn >= thr).mean()),
                                               "neutral_mean": float(sn.mean()), "neutral_std": float(sn.std()),
                                               "real_prompt_tokens_n": int(len(sr)), "real_frac_above_midpoint": float((sr >= thr).mean())}}
        print(f"{name:7s} b.z entity-last: target {mu_t @ b:.2f} others {mu_o @ b:.2f}±{float((others_train @ b).std()):.2f} "
              f"(d' {input_side[name]['train_template']['entity_last_token']['dprime']:.1f}); paraphrase d' "
              f"{[round(v['entity_last_token']['dprime'], 1) for v in input_side[name]['paraphrases'].values()]}; "
              f"neutral tokens above midpoint {input_side[name]['generic_firing']['neutral_frac_above_midpoint']:.3f}", flush=True)

    record = {"protocol": "PATCH-TRAIN-1 SINGLE-MECH-1 (DISCOVERY; NOT frozen; NOT confirmatory)",
              "backend": V.backend_record(tasks), "task_set": f"tasks_discovery.json (seed {tasks['seed']})",
              "predeclared": {"entity": entity, "true_answer": true_answer, "control_answer": control_answer, "layer": LAYER,
                              "tensor": tensor, "columns": N_COLUMNS, "column_seed": F.SEED, "rank": 1, "b": "fixed, unit norm",
                              "rho": RHO, "update_frobenius_cap": float(cap), "lr": LR, "max_steps": MAX_STEPS,
                              "lambda": S.LAMBDA, "reported": "final step", "b_built_from": "training template only"},
              "optimisation_approximation": "training forward rounds via torch f32->bf16 cast, straight-through; reported weights are "
                                            "the exact frozen v2 emulation",
              "input_side": input_side, "base": S.evaluate(model, tok, tasks, entity, T, Tc, ans), "runs": {}}
    print("base done", flush=True)

    # ------------------------------------------------------------------ B/C. output side with b fixed
    def applied(a, b):
        return E.apply_columns(base_bits, columns, a, b, PROFILE)[0]

    for bname, b_np in candidates.items():
        b_t = torch.from_numpy(b_np).float().reshape(1, -1)
        for label, target_tok, other_tok in (("assigned_answer", T, Tc), ("control_answer", Tc, T)):
            t0 = time.time()
            prompts = [(tok(tpl.format(e=entity), return_tensors="pt")["input_ids"], target_tok) for tpl in t["train_templates"]]
            _, _, g = S.loss_and_grad(model, key, base_f32, prompts, ans)
            ga = g[:, cols_t] @ b_t.T                                           # d loss / d a  (rows, 1)
            a = (-ga / ga.norm() * (cap / 2.0)).clone().requires_grad_(True)    # init: negative-gradient direction, half the cap
            opt = torch.optim.Adam([a], lr=LR)
            base_cols = base_f32[:, cols_t]
            curve, ckpts, bind = [], {}, 0
            for step in range(MAX_STEPS + 1):
                if step in CHECKPOINTS:
                    nb = applied(a.detach().double().numpy(), b_np.reshape(1, -1))
                    with V.patched_weight(model, key, nb):
                        ckpts[str(step)] = S.evaluate(model, tok, tasks, entity, target_tok, other_tok, ans)
                    ckpts[str(step)]["elements_changed"] = int((nb != base_bits).sum())
                if step == MAX_STEPS:
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
                curve.append({"step": step, "loss": loss, "cond": parts[0], "ce": parts[1], "update_frobenius": min(fro, cap)})
            a_np = a.detach().double().numpy()
            nb = applied(a_np, b_np.reshape(1, -1))
            ideal_delta = a_np @ b_np.reshape(1, -1)
            realised = E.bf16_to_f64(nb[:, columns]) - E.bf16_to_f64(base_bits[:, columns])
            ideal_w = base_f32.clone()
            ideal_w[:, cols_t] += torch.from_numpy(ideal_delta.astype(np.float32))
            with V.patched_values(model, key, ideal_w):
                ideal_eval = S.evaluate(model, tok, tasks, entity, target_tok, other_tok, ans)
            gap = {"survival_fraction": float((nb != base_bits).sum() / (rows * N_COLUMNS)),
                   "realised_over_ideal_frobenius": float(np.linalg.norm(realised) / np.linalg.norm(ideal_delta)),
                   "cosine_realised_vs_ideal": float((realised * ideal_delta).sum() / (np.linalg.norm(realised) * np.linalg.norm(ideal_delta))),
                   "entity_train_logp_T_ideal_f32": ideal_eval["entity_train_prompts"]["mean_logp_T"],
                   "entity_train_logp_T_rounded": ckpts["200"]["entity_train_prompts"]["mean_logp_T"],
                   "others_logp_T_ideal_f32": ideal_eval["other_entities_leakage"]["mean_logp_T"],
                   "others_logp_T_rounded": ckpts["200"]["other_entities_leakage"]["mean_logp_T"]}
            run = {"b": bname, "answer": true_answer if target_tok == T else control_answer, "curve": curve, "checkpoints": ckpts,
                   "cap_binding_steps": bind, "compiled_gap": gap, "seconds": time.time() - t0}
            record["runs"][f"{bname}/{label}"] = run
            f = ckpts["200"]; e = f["entity_train_prompts"]; p = f["entity_paraphrases"]; o = f["other_entities_leakage"]
            print(f"{bname:7s} {label:16s} train rank {e['median_rank_T']:>5} inset {e['frac_inset_top1_is_T']:.2f} full {e['frac_full_top1_is_T']:.2f} "
                  f"| para rank {p['median_rank_T']:>5} inset {p['frac_inset_top1_is_T']:.2f} | others inset-T {o['frac_inset_top1_is_T']:.2f} "
                  f"logpT {o['mean_logp_T']:.2f} (own {e['mean_logp_T']:.2f}) | real {f['real_top1']:.2f} neutral {f['neutral_nll']:.2f} | "
                  f"bind {bind} | gap cos {gap['cosine_realised_vs_ideal']:.3f} surv {gap['survival_fraction']:.3f}", flush=True)
            with open(os.path.join(HERE, args.out), "w") as fh:
                json.dump(record, fh, indent=1)
                fh.write("\n")
    record["model_digest_unchanged_at_end"] = V.model_digest(model) == digest0
    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(record, fh, indent=1)
        fh.write("\n")
    print("model restored:", record["model_digest_unchanged_at_end"])


if __name__ == "__main__":
    main()
