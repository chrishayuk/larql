#!/usr/bin/env python3
"""PATCH-TRAIN-2 RETAIN-1 (DISCOVERY ONLY; emulation-consistent; protocol in retain1_registration.json).

For one registered stress pair (A, B) and BOTH training orders, the FIRST patch is learned on W0 exactly as in STRESS-CAP-1 and the
SECOND patch is learned on the first patch's effective model W1 with one of three objectives (same layer 3, rank 1, 256 columns, rho 0.40
from the W0 footprint RMS, Adam lr 3e-3, 200 steps, BF16-rounded straight-through forward, final step only):

  N  new-task only      L = L_new                                      (the STRESS-CAP-1 control; must reproduce it bit for bit)
  R  rehearsal          L = L_new + LAMBDA_R * L_first                 L_first = the FIRST patch's own objective on its ORIGINAL training prompt
  D  distillation       L = L_new + LAMBDA_D * mean_{p in RET} KL(P_W1(.|p) || P_student(.|p))   full-vocabulary, last token,
                        teacher = the first-patch-only model W1; RET = the first patch's training prompt plus the training-template
                        prompt of every OTHER discovery entity except the two patched ones (23 prompts). No held-out paraphrase, no
                        real-fact prompt and no neutral text is used for training or distillation: they stay clean collateral checks.
  L_x = conditional-12 + 0.1 * full-vocabulary CE on x's training prompt (the objective used throughout PATCH-TRAIN-1/2).
Independent composition (the first patch and the independently learned second patch, rebound) is recomputed as the baseline.
Initialisation: negative gradient of the arm's OWN total objective at the source model W1, at half the cap (for N this is exactly the
earlier initialisation).  Every patch carries a source binding; reported weights are the exact frozen v2 emulation.

    python3 -I retain1.py --pair 1 --out retain1_pair1_discovery.json
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

import d2 as D2  # noqa: E402
import emulate as E  # noqa: E402
import evaluation as V  # noqa: E402
import feasibility as F  # noqa: E402
import mech1 as M  # noqa: E402
import rep1 as R  # noqa: E402
import single1 as S  # noqa: E402

LAYER, PROFILE = M.LAYER, E.PROFILE_RNE1
RHO, LAMBDA_R, LAMBDA_D = 0.40, 1.0, 1.0


def kl_loss_and_grad(model, key, w_full, ret_ids, teacher_lp):
    w = w_full.detach().clone().requires_grad_(True)
    total = 0.0
    for ids, tl in zip(ret_ids, teacher_lp):
        ls = torch.log_softmax(functional_call(model, {key: w}, args=(), kwargs={"input_ids": ids}).logits[0, -1].float(), dim=-1)
        kl = (tl.exp() * (tl - ls)).sum()
        (kl / len(ret_ids)).backward()
        total += kl.item() / len(ret_ids)
    return total, w.grad.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=S.MAX_STEPS, help="registered value is 200; other values are for code smoke tests only")
    ap.add_argument("--only-order", default=None, help="smoke tests only")
    args = ap.parse_args()
    reg = json.load(open(os.path.join(HERE, "stress_cap1_registration.json")))
    pr = reg["pairs"][args.pair - 1]
    entA, entB = pr["A"], pr["B"]
    tasks = V.load_tasks("tasks_discovery.json")
    assert tasks["seed"] != V.load_tasks("tasks.json")["seed"], "discovery set must differ from the confirmatory set"
    model, tok = V.load_model(tasks)
    digest0 = V.model_digest(model)
    key = V.hf_key(f"{LAYER}.mlp.down_proj.weight")
    W0 = V.weight_bits(model, key)
    rows, ncols = W0.shape
    columns = np.sort(np.random.default_rng(F.SEED).choice(ncols, M.N_COLUMNS, replace=False))
    cols_t = torch.from_numpy(columns)
    rms_w = float(np.sqrt((E.bf16_to_f64(W0[:, columns]) ** 2).mean()))
    cap = RHO * rms_w * np.sqrt(rows * M.N_COLUMNS)
    t = tasks["tasks"]["A"]
    ans = torch.tensor([V.first_token(tok, a) for a in t["answers"]])
    ansA, ansB = t["assignment"][entA], t["assignment"][entB]
    tid = {entA: V.first_token(tok, ansA), entB: V.first_token(tok, ansB)}
    ans_of = {entA: ansA, entB: ansB}
    all_ents = t["trained_entities"] + t["heldout_entities"]
    others = lambda e: [x for x in all_ents if x != e]
    addr = {entA: R.build_b(model, tok, tasks, entA, others(entA), columns)[0], entB: R.build_b(model, tok, tasks, entB, others(entB), columns)[0]}
    f32 = lambda bits: torch.from_numpy(E.bf16_to_f32(bits).copy())
    train_tpl = t["train_templates"][0]
    ids_of = lambda e: tok(train_tpl.format(e=e), return_tensors="pt")["input_ids"]

    record = {"protocol": "PATCH-TRAIN-2 RETAIN-1 (DISCOVERY; emulation-consistent; thresholds not used)", "pair": args.pair, "A": [entA, ansA], "B": [entB, ansB],
              "backend": V.backend_record(tasks), "registered": "retain1_registration.json",
              "config": {"layer": LAYER, "rank": 1, "columns": M.N_COLUMNS, "rho": RHO, "cap": float(cap), "lr": S.LR, "steps": args.steps,
                         "lambda_rehearsal": LAMBDA_R, "lambda_distillation": LAMBDA_D, "objective_per_task": "conditional-12 + 0.1 full-vocabulary CE"},
              "optimisation_approximation": "training forward rounds via torch f32->bf16 cast, straight-through; reported weights are the exact frozen v2 emulation",
              "hashes": {"W0": D2.sha(W0)}, "stages": {}, "orders": {}}

    def dump():
        with open(os.path.join(HERE, args.out), "w") as fh:
            json.dump(record, fh, indent=1)
            fh.write("\n")

    # ---------------------------------------------------------------- measurement (copied from d2.py's stage measurement)
    def measure(bits, label):
        out = {"tensor_sha256": D2.sha(bits)}
        with V.patched_weight(model, key, bits):
            reads = {}
            for ent in (entA, entB):
                reads[ent] = {}
                for name, tpl in [("train", train_tpl)] + [(f"para{i}", p) for i, p in enumerate(t["heldout_templates"])]:
                    lp = V.logprobs(model, tok, tpl.format(e=ent))
                    al = lp[ans]
                    rec = {"inset_argmax": t["answers"][int(al.argmax().item())], "mass": torch.logsumexp(al, 0).item()}
                    for an, ti in ((ansA, tid[entA]), (ansB, tid[entB])):
                        rec[an] = {"logp": lp[ti].item(), "rank": 1 + int((lp > lp[ti]).sum().item()), "top1": int(lp.argmax().item()) == ti,
                                   "cond_logp": lp[ti].item() - torch.logsumexp(al, 0).item()}
                    reads[ent][name] = rec
            per = {an: {e: V.logprobs(model, tok, train_tpl.format(e=e))[ti].item() for e in all_ents} for an, ti in ((ansA, tid[entA]), (ansB, tid[entB]))}
            real = [(V.logprobs(model, tok, f["template"].format(e=f["country"])), V.first_token(tok, f["answer"])) for f in tasks["real_facts"]]
            obj = {}
            for ent in (entA, entB):
                lp = V.logprobs(model, tok, train_tpl.format(e=ent))
                ti = tid[ent]
                obj[ent] = float(-(lp[ti] - torch.logsumexp(lp[ans], 0)).item() + S.LAMBDA * (-lp[ti].item()))
            out.update({"reads": reads, "per_entity_logp_train_template": per, "task_objective": obj,
                        "real_top1": sum(int(lp.argmax().item()) == y for lp, y in real) / len(real),
                        "real_mean_logp": sum(lp[y].item() for lp, y in real) / len(real),
                        "neutral_nll": V.neutral_nll(model, tok, tasks["neutral_text"])})
        record["stages"][label] = out
        return out

    base = measure(W0, "W0")
    base_per = base["per_entity_logp_train_template"]

    def summary(stage, first, second):
        rd = stage["reads"]
        row = {}
        for nm, ent in (("first", first), ("second", second)):
            an = ans_of[ent]
            r = rd[ent]
            per = stage["per_entity_logp_train_template"][an]
            d_t = per[ent] - base_per[an][ent]
            partner = second if ent == first else first
            d_o = [per[e] - base_per[an][e] for e in all_ents if e not in (ent, partner)]
            row[nm] = {"entity": ent, "answer": an, "train_top1": r["train"][an]["top1"], "train_rank": r["train"][an]["rank"],
                       "train_logp": r["train"][an]["logp"], "para_top1": [r[f"para{i}"][an]["top1"] for i in range(3)],
                       "para_rank": [r[f"para{i}"][an]["rank"] for i in range(3)], "para_logp": [r[f"para{i}"][an]["logp"] for i in range(3)],
                       "S": d_t - float(np.mean(d_o)), "partner_dlogp": per[partner] - base_per[an][partner], "others_mean_dlogp": float(np.mean(d_o)),
                       "others_max_dlogp": float(max(d_o))}
        row["both_top1"] = bool(row["first"]["train_top1"] and row["second"]["train_top1"])
        row["collateral"] = {"real_top1": stage["real_top1"], "real_mean_logp": stage["real_mean_logp"], "neutral_delta": stage["neutral_nll"] - base["neutral_nll"]}
        row["task_objective"] = stage["task_objective"]
        return row

    def learn(first, second, W1, b_np, mode, teacher_lp, ret_ids):
        t0 = time.time()
        base_f32 = f32(W1)
        b_t = torch.from_numpy(b_np).float().reshape(1, -1)
        prompts_new = [(ids_of(second), tid[second])]
        prompts_first = [(ids_of(first), tid[first])]

        def total(full):
            loss, _, g = S.loss_and_grad(model, key, full, prompts_new, ans)
            parts = {"new": loss}
            if mode == "R":
                lf, _, gf = S.loss_and_grad(model, key, full, prompts_first, ans)
                loss, g = loss + LAMBDA_R * lf, g + LAMBDA_R * gf
                parts["first"] = lf
            elif mode == "D":
                kl, gk = kl_loss_and_grad(model, key, full, ret_ids, teacher_lp)
                loss, g = loss + LAMBDA_D * kl, g + LAMBDA_D * gk
                parts["kl"] = kl
            return loss, parts, g
        _, _, g0 = total(base_f32)
        ga = g0[:, cols_t] @ b_t.T
        a = (-ga / ga.norm() * (cap / 2.0)).clone().requires_grad_(True)
        opt = torch.optim.Adam([a], lr=S.LR)
        base_cols = base_f32[:, cols_t]
        curve, bind = [], 0
        for step in range(args.steps):
            opt.zero_grad()
            wc = base_cols + a @ b_t
            full = base_f32.clone()
            full[:, cols_t] = wc.detach().to(torch.bfloat16).to(torch.float32)
            loss, parts, grad_full = total(full)
            wc.backward(grad_full[:, cols_t])
            opt.step()
            with torch.no_grad():
                fro = torch.linalg.norm(a @ b_t).item()
                if fro > cap:
                    a.mul_(cap / fro)
                    bind += 1
            curve.append({"step": step, "loss": loss, **{k: v for k, v in parts.items()}, "update_frobenius": min(fro, cap)})
        print(f"   trained {second}|{mode}: loss {curve[0]['loss']:.2f}->{curve[-1]['loss']:.3f} {({k: round(v, 3) for k, v in curve[-1].items() if k in ('new', 'first', 'kl')})} bind {bind} ({time.time() - t0:.0f}s)", flush=True)
        return D2.Patch(f"P_{second}|{mode}", a.detach().double().numpy(), b_np.reshape(1, -1), D2.sha(W1)), curve, bind

    # ---------------------------------------------------------------- the two orders
    first_patches, stage1 = {}, {}
    for first, second in ((entA, entB), (entB, entA)):
        order = f"{first}_then_{second}"
        if args.only_order and args.only_order != order:
            continue
        print(f"order {order}", flush=True)
        # first patch, learned on W0 exactly as in STRESS-CAP-1
        a1, curve1, bind1, _ = R.train(model, tok, tasks, key, f32(W0), W0, columns, first, tid[first], addr[first], RHO, ans, rms_w)
        P1 = D2.Patch(f"P_{first}|W0", a1, addr[first].reshape(1, -1), D2.sha(W0))
        W1 = D2.apply_patch(W0, columns, P1)
        measure(W1, f"{order}/first_only")
        # independently learned second patch on W0, rebound to W1 (independent composition baseline)
        a2i, _, _, _ = R.train(model, tok, tasks, key, f32(W0), W0, columns, second, tid[second], addr[second], RHO, ans, rms_w)
        P2i = D2.Patch(f"P_{second}|W0", a2i, addr[second].reshape(1, -1), D2.sha(W0))
        W_comp = D2.apply_patch(W1, columns, P2i.rebound(D2.sha(W1), f"P_{second}(rebound)"))
        measure(D2.apply_patch(W0, columns, P2i), f"{order}/second_only")
        measure(W_comp, f"{order}/composition")
        # retention prompt set for distillation and the teacher (first-patch-only model)
        ret_ents = [first] + [e for e in all_ents if e not in (entA, entB)]
        ret_ids = [ids_of(e) for e in ret_ents]
        with V.patched_weight(model, key, W1):
            teacher_lp = [V.logprobs(model, tok, train_tpl.format(e=e)).detach().clone() for e in ret_ents]
        o = {"first": first, "second": second, "first_patch_payload": P1.payload_sha, "retention_set": ret_ents,
             "first_training_cap_binding": bind1, "arms": {}}
        for mode in ("N", "R", "D"):
            P2, curve, bind = learn(first, second, W1, addr[second], mode, teacher_lp, ret_ids)
            W2 = D2.apply_patch(W1, columns, P2)
            st = measure(W2, f"{order}/{mode}")
            replay = D2.apply_patch(D2.apply_patch(W0, columns, D2.Patch("r1", P1.a, P1.b, D2.sha(W0))), columns, D2.Patch("r2", P2.a, P2.b, D2.sha(W1)))
            refused = False
            try:
                D2.apply_patch(W0, columns, P2)
            except D2.Refused:
                refused = True
            o["arms"][mode] = {"curve": curve, "cap_binding_steps": bind, "payload_sha": P2.payload_sha, "tensor_sha256": D2.sha(W2),
                               "chain_replay_bit_exact": bool(np.array_equal(replay, W2)), "second_refused_after_removing_first": refused,
                               "summary": summary(st, first, second)}
            dump()
        o["composition"] = summary(record["stages"][f"{order}/composition"], first, second)
        o["first_only"] = summary(record["stages"][f"{order}/first_only"], first, second)
        o["second_only"] = summary(record["stages"][f"{order}/second_only"], first, second)
        record["orders"][order] = o
        for nm in ("composition", "N", "R", "D"):
            s = o["composition"] if nm == "composition" else o["arms"][nm]["summary"]
            print(f"  {order:34s} {nm:11s} first rank {s['first']['train_rank']:>5} para {s['first']['para_rank']} | second rank {s['second']['train_rank']:>5} "
                  f"para {s['second']['para_rank']} | both {s['both_top1']} | partner {s['first']['partner_dlogp']:+.1f}/{s['second']['partner_dlogp']:+.1f} | "
                  f"real {s['collateral']['real_top1']:.2f} dNLL {s['collateral']['neutral_delta']:+.3f}", flush=True)
    record["model_digest_unchanged_at_end"] = V.model_digest(model) == digest0
    dump()
    print("model restored:", record["model_digest_unchanged_at_end"])


if __name__ == "__main__":
    main()
