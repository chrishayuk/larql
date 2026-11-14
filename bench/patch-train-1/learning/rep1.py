#!/usr/bin/env python3
"""PATCH-TRAIN-1 SINGLE-REP-1 (DISCOVERY ONLY, protocol not frozen): independent second-entity replication of the
layer-3 rank-1 address+payload patch, with cross-application of the two patches.

FROZEN BEFORE ANY EVALUATION (written into the output header first): second entity = tasks_discovery.json trained_entities[1]
and its assigned answer; a NEGATIVE-CONTROL answer for it (the next answer in the declared order that is neither its own nor
Frozoria's); comparison entities = all other 23 entities (this includes Frozoria); rho in (0.10, 0.20); everything else
IDENTICAL to SINGLE-CAP-1 (layer 3 down_proj, rank 1, the same 256 columns, ridge-whitened b built from THAT entity's own
training-template activation against the others and neutral text, objective conditional-12 + 0.1 CE, Adam lr 3e-3, 200 steps,
STE through the BF16-rounded forward, init at half the cap). The collateral limits and the "stop if violated" rule are
unchanged. Primary endpoint: the final step; no checkpoint is selected.
Also retrained here for the cross-application control: Frozoria -> Lyon at the same rho (deterministic; matches SINGLE-CAP-1).

Cross-application: every patch is applied to BOTH entities' prompts and its target answer's log-probability and rank are read on
each; the patch for entity X must move X's answer on X and not on the other entity.
    python3 -I rep1.py --out rep1_layer3_discovery.json
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

import cap1 as C  # noqa: E402
import emulate as E  # noqa: E402
import evaluation as V  # noqa: E402
import feasibility as F  # noqa: E402
import mech1 as M  # noqa: E402
import single1 as S  # noqa: E402

LAYER, PROFILE, LIMITS = M.LAYER, E.PROFILE_RNE1, C.LIMITS
RHOS = (0.10, 0.20)


def build_b(model, tok, tasks, entity, others, columns):
    """Ridge-whitened input direction for `entity`, from its own training-template activation (identical recipe to mech1/cap1)."""
    tpl = tasks["tasks"]["A"]["train_templates"][0]

    def last_act(e):
        z, ids = M.capture(model, tok, tpl.format(e=e), columns)
        return z[M.entity_span(tok, tpl, e, ids)[1]]
    mu_t = last_act(entity)
    others_train = np.stack([last_act(e) for e in others])
    neutral = np.concatenate([M.capture(model, tok, s, columns)[0] for s in tasks["neutral_text"]])
    cov = np.cov(np.concatenate([others_train, neutral]), rowvar=False)
    b = np.linalg.solve(cov + 0.1 * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0]), mu_t - others_train.mean(0))
    b /= np.linalg.norm(b)
    return b, {"target_b_dot_z": float(mu_t @ b), "others_b_dot_z_mean": float((others_train @ b).mean()),
               "others_b_dot_z_std": float((others_train @ b).std()),
               "dprime": float((mu_t @ b - (others_train @ b).mean()) / ((others_train @ b).std() + 1e-12)),
               "neutral_frac_above_midpoint": float(((neutral @ b) >= 0.5 * (mu_t @ b + (others_train @ b).mean())).mean())}


def train(model, tok, tasks, key, base_f32, base_bits, columns, entity, target_tok, b_np, rho, ans, rms_w):
    rows = base_bits.shape[0]
    cols_t = torch.from_numpy(columns)
    cap = rho * rms_w * np.sqrt(rows * M.N_COLUMNS)
    b_t = torch.from_numpy(b_np).float().reshape(1, -1)
    t = tasks["tasks"]["A"]
    prompts = [(tok(tpl.format(e=entity), return_tensors="pt")["input_ids"], target_tok) for tpl in t["train_templates"]]
    _, _, g = S.loss_and_grad(model, key, base_f32, prompts, ans)
    ga = g[:, cols_t] @ b_t.T
    a = (-ga / ga.norm() * (cap / 2.0)).clone().requires_grad_(True)
    opt = torch.optim.Adam([a], lr=S.LR)
    base_cols = base_f32[:, cols_t]
    curve, bind = [], 0
    for step in range(S.MAX_STEPS):
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
        curve.append({"step": step, "loss": loss, "logp_target_train": -parts[1], "update_frobenius": min(fro, cap)})
    return a.detach().double().numpy(), curve, bind, float(cap)


def read_on(model, tok, tasks, entity, tok_id):
    """Target-answer log-prob, full-vocabulary rank and top-1 on the training prompt and each held-out paraphrase."""
    t = tasks["tasks"]["A"]
    out = {}
    for name, tpl in [("train", t["train_templates"][0])] + [(f"para{i}", p) for i, p in enumerate(t["heldout_templates"])]:
        lp = V.logprobs(model, tok, tpl.format(e=entity))
        out[name] = {"logp": lp[tok_id].item(), "rank": 1 + int((lp > lp[tok_id]).sum().item()), "top1": int(lp.argmax().item()) == tok_id}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rep1_layer3_discovery.json")
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
    rms_w = float(np.sqrt((E.bf16_to_f64(base_bits[:, columns]) ** 2).mean()))
    t = tasks["tasks"]["A"]
    ans = torch.tensor([V.first_token(tok, a) for a in t["answers"]])
    ent1, ent2 = t["trained_entities"][0], t["trained_entities"][1]
    ans1, ans2 = t["assignment"][ent1], t["assignment"][ent2]
    assert ans1 != ans2
    alt2 = next(a for a in t["answers"][t["answers"].index(ans2) + 1:] + t["answers"] if a not in (ans1, ans2))
    tid = {a: V.first_token(tok, a) for a in (ans1, ans2, alt2)}
    all_ents = t["trained_entities"] + t["heldout_entities"]

    def others_of(e):
        return [x for x in all_ents if x != e]

    record = {"protocol": "PATCH-TRAIN-1 SINGLE-REP-1 (DISCOVERY; NOT frozen; NOT confirmatory)",
              "backend": V.backend_record(tasks), "task_set": f"tasks_discovery.json (seed {tasks['seed']})",
              "FROZEN_BEFORE_EVALUATION": {"second_entity": ent2, "second_answer": ans2, "negative_control_answer": alt2,
                                           "first_entity": ent1, "first_answer": ans1, "comparison_entities": "all other 23 (incl. the other patched entity)",
                                           "rhos": list(RHOS), "layer": LAYER, "rank": 1, "columns": M.N_COLUMNS, "lr": S.LR,
                                           "steps": S.MAX_STEPS, "lambda": S.LAMBDA, "limits": LIMITS, "endpoint": "final step"},
              "optimisation_approximation": "training forward rounds via torch f32->bf16 cast, straight-through; reported weights are "
                                            "the exact frozen v2 emulation",
              "base_reads": {ent1: read_on(model, tok, tasks, ent1, tid[ans1]), ent2: read_on(model, tok, tasks, ent2, tid[ans2])},
              "arms": {}, "stopped_after": None}
    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(record, fh, indent=1)
        fh.write("\n")
    print(f"frozen: {ent2} -> {ans2} (control {alt2}); first {ent1} -> {ans1}", flush=True)
    base_eval = {ent: S.evaluate(model, tok, tasks, ent, tid[t["assignment"][ent]], tid[alt2 if ent == ent2 else ans2], ans)
                 for ent in (ent1, ent2)}
    base_per = {a: C.per_entity_logp(model, tok, tasks, all_ents, tid[a]) for a in (ans1, ans2, alt2)}
    b1, b1info = build_b(model, tok, tasks, ent1, others_of(ent1), columns)
    b2, b2info = build_b(model, tok, tasks, ent2, others_of(ent2), columns)
    record["address"] = {ent1: b1info, ent2: b2info, "cosine_between_addresses": float(b1 @ b2)}
    print("addresses:", {k: round(v["dprime"], 1) for k, v in record["address"].items() if isinstance(v, dict)},
          "cos(b1,b2)", round(record["address"]["cosine_between_addresses"], 3), flush=True)

    def apply(a, b):
        return E.apply_columns(base_bits, columns, a, b.reshape(1, -1), PROFILE)[0]

    for rho in RHOS:
        patches = {}
        for label, ent, ans_name, b in (("ent1_assigned", ent1, ans1, b1), ("ent2_assigned", ent2, ans2, b2), ("ent2_negctl", ent2, alt2, b2)):
            t0 = time.time()
            a, curve, bind, cap = train(model, tok, tasks, key, base_f32, base_bits, columns, ent, tid[ans_name], b, rho, ans, rms_w)
            nb = apply(a, b)
            ideal_w = base_f32.clone()
            ideal_w[:, torch.from_numpy(columns)] += torch.from_numpy((a @ b.reshape(1, -1)).astype(np.float32))
            realised = E.bf16_to_f64(nb[:, columns]) - E.bf16_to_f64(base_bits[:, columns])
            ideal = a @ b.reshape(1, -1)
            gap = {"survival_fraction": float((nb != base_bits).sum() / (rows * M.N_COLUMNS)),
                   "cosine_realised_vs_ideal": float((realised * ideal).sum() / (np.linalg.norm(realised) * np.linalg.norm(ideal))),
                   "realised_over_ideal_frobenius": float(np.linalg.norm(realised) / np.linalg.norm(ideal))}
            with V.patched_weight(model, key, nb):
                other_ent = ent2 if ent == ent1 else ent1
                ev = S.evaluate(model, tok, tasks, ent, tid[ans_name], tid[alt2 if ans_name != alt2 else ans2], ans)
                reads = {ent: read_on(model, tok, tasks, ent, tid[ans_name]), other_ent: read_on(model, tok, tasks, other_ent, tid[ans_name])}
                per = C.per_entity_logp(model, tok, tasks, all_ents, tid[ans_name])
                other_ent_own_answer = read_on(model, tok, tasks, other_ent, tid[t["assignment"][other_ent]])
                # which in-set answer each entity's prompt now selects
                argmax_inset = {e: t["answers"][int(torch.argmax(V.logprobs(model, tok, t["train_templates"][0].format(e=e))[ans]).item())]
                                for e in (ent1, ent2)}
            d_others = {e: per[e] - base_per[ans_name][e] for e in all_ents if e != ent}
            sel = {"d_logp_target_entity": per[ent] - base_per[ans_name][ent], "d_logp_others_mean": float(np.mean(list(d_others.values()))),
                   "d_logp_others_max": float(max(d_others.values())), "per_entity_d_logp_others": d_others}
            sel["S"] = sel["d_logp_target_entity"] - sel["d_logp_others_mean"]
            base_ev = S.evaluate(model, tok, tasks, ent, tid[ans_name], tid[alt2 if ans_name != alt2 else ans2], ans)  # restored base (patch context exited)
            viol = []
            if ev["real_top1"] < LIMITS["real_top1_min"]:
                viol.append("real_top1")
            if ev["neutral_nll"] > base_ev["neutral_nll"] + LIMITS["neutral_nll_max_increase"]:
                viol.append("neutral_nll")
            if sel["d_logp_others_mean"] > LIMITS["others_mean_dlogp_max"]:
                viol.append("others_mean_dlogp")
            arm = {"rho": rho, "entity": ent, "answer": ans_name, "update_frobenius_cap": cap, "cap_binding_steps": bind,
                   "curve": curve, "final": ev, "target_answer_reads_on_both_entities": reads, "selectivity": sel,
                   "other_patched_entity_own_answer_under_this_patch": other_ent_own_answer, "inset_argmax_answer_under_patch": argmax_inset,
                   "compiled_gap": gap, "base_neutral_nll": base_ev["neutral_nll"], "violated_limits": viol, "seconds": time.time() - t0}
            record["arms"][f"rho{rho}/{label}"] = arm
            patches[label] = (a, b)
            r = reads[ent]
            print(f"rho {rho:.2f} {label:14s} {ent:12s}->{ans_name:8s} train rank {r['train']['rank']:>5} top1 {r['train']['top1']} | para top1 "
                  f"{[r[k]['top1'] for k in ('para0','para1','para2')]} ranks {[r[k]['rank'] for k in ('para0','para1','para2')]} | "
                  f"S {sel['S']:+.2f} others mean {sel['d_logp_others_mean']:+.2f} max {sel['d_logp_others_max']:+.2f} | on {other_ent}: "
                  f"rank {reads[other_ent]['train']['rank']} top1 {reads[other_ent]['train']['top1']} | real {ev['real_top1']:.2f} "
                  f"neutral {ev['neutral_nll']:.3f} | bind {bind} | viol {viol or 'none'}", flush=True)
            with open(os.path.join(HERE, args.out), "w") as fh:
                json.dump(record, fh, indent=1)
                fh.write("\n")
            if viol and label != "ent2_negctl":
                record["stopped_after"] = f"rho{rho}/{label} violated {viol}; no larger cap run (predeclared)"
        if record["stopped_after"]:
            break
    record["model_digest_unchanged_at_end"] = V.model_digest(model) == digest0
    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(record, fh, indent=1)
        fh.write("\n")
    print("model restored:", record["model_digest_unchanged_at_end"], "| stopped:", record["stopped_after"])


if __name__ == "__main__":
    main()
