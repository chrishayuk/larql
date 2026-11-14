#!/usr/bin/env python3
"""PATCH-TRAIN-2 D2 discovery dry-run (DISCOVERY ONLY; EMULATION-CONSISTENT, never "verified"; thresholds NOT used).

Implements the seven arms of bench/patch-train-1/PATCH-TRAIN-2.md on Frozoria -> Lyon (A) and Dresalthaia -> Leeds (B):
layer 3 down_proj, rank 1, the same 256 columns, rho 0.20 (cap from the BASE W0 footprint RMS in every arm and stage), 200 steps,
the SINGLE-CAP-1/REP-1 objective and lr, straight-through BF16 forward (optimisation approximation), final step only. Reported
weights are always the exact frozen v2 emulation. Every patch carries a SOURCE BINDING (sha256 of the tensor it was applied to);
applying a patch to a different source is REFUSED. The base checkpoint is digest-checked.

  1 A only (learned on W0)            2 B only (learned on W0)
  3 A then independent B (REBOUND to W1)       4 B then independent A (rebound to W1B)
  5 A then B learned on W1                     6 B then A learned on W1B
  7 joint single stage: BF16(W0 + dA + dB), rank 2, one rounding

Descriptive only: no pass/fail is declared and no threshold is used; retention is reported against each patch's own isolated
result (per prompt). Numerical differences between W_AB, W_BA, W_joint are recorded element by element.
    python3 -I d2.py --out d2_discovery.json
"""
import argparse
import hashlib
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
import rep1 as R  # noqa: E402
import single1 as S  # noqa: E402

LAYER, PROFILE, RHO = M.LAYER, E.PROFILE_RNE1, 0.20
LIMITS_REAL_MIN, NLL_MAX_INC = 0.96, 0.15      # the discovery limits used since SINGLE-CAP-1; reported, not gating here


def sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


class Refused(Exception):
    pass


class Patch:
    """A rank-k column patch bound to the source tensor it was learned on / applied to."""
    def __init__(self, name, a, b, source_sha):
        self.name, self.a, self.b, self.source_sha = name, np.asarray(a, np.float64), np.asarray(b, np.float64).reshape(-1, np.asarray(b).shape[-1]), source_sha
        self.payload_sha = sha(self.a) + sha(self.b)

    def rebound(self, new_source_sha, name):
        return Patch(name, self.a, self.b, new_source_sha)


def apply_patch(src_bits, columns, patch):
    if sha(src_bits) != patch.source_sha:
        raise Refused(f"{patch.name}: source binding stale (patch bound to {patch.source_sha[:12]}, source is {sha(src_bits)[:12]})")
    return E.apply_columns(src_bits, columns, patch.a, patch.b, PROFILE)[0]


def capture_full(model, tok, text):
    got = {}
    h = model.model.layers[LAYER].mlp.down_proj.register_forward_pre_hook(lambda m, a: got.__setitem__("z", a[0][0].detach().double().numpy()))
    with torch.no_grad():
        model(input_ids=tok(text, return_tensors="pt")["input_ids"])
    h.remove()
    return got["z"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="d2_discovery.json")
    ap.add_argument("--entity-a", default=None, help="entity name for patch A (default: trained_entities[0], Frozoria)")
    ap.add_argument("--entity-b", default=None, help="entity name for patch B (default: trained_entities[1], Dresalthaia)")
    ap.add_argument("--label", default="PATCH-TRAIN-2 D2 discovery dry-run")
    ap.add_argument("--rho", type=float, default=0.20, help="update-norm cap as a fraction of the W0 footprint RMS (default 0.20)")
    args = ap.parse_args()
    global RHO
    RHO = args.rho
    tasks = V.load_tasks("tasks_discovery.json")
    assert tasks["seed"] != V.load_tasks("tasks.json")["seed"], "discovery set must differ from the confirmatory set"
    model, tok = V.load_model(tasks)
    digest0 = V.model_digest(model)
    key = V.hf_key(f"{LAYER}.mlp.down_proj.weight")
    W0 = V.weight_bits(model, key)
    rows, ncols = W0.shape
    columns = np.sort(np.random.default_rng(F.SEED).choice(ncols, M.N_COLUMNS, replace=False))
    rms_w = float(np.sqrt((E.bf16_to_f64(W0[:, columns]) ** 2).mean()))
    t = tasks["tasks"]["A"]
    ans = torch.tensor([V.first_token(tok, a) for a in t["answers"]])
    entA = args.entity_a or t["trained_entities"][0]
    entB = args.entity_b or t["trained_entities"][1]
    default_pair = (entA, entB) == (t["trained_entities"][0], t["trained_entities"][1])
    ansA, ansB = t["assignment"][entA], t["assignment"][entB]
    tidA, tidB = V.first_token(tok, ansA), V.first_token(tok, ansB)
    all_ents = t["trained_entities"] + t["heldout_entities"]
    others = lambda e: [x for x in all_ents if x != e]
    bA, _ = R.build_b(model, tok, tasks, entA, others(entA), columns)
    bB, _ = R.build_b(model, tok, tasks, entB, others(entB), columns)
    f32 = lambda bits: torch.from_numpy(E.bf16_to_f32(bits).copy())

    record = {"protocol": args.label + " (DISCOVERY; emulation-consistent; thresholds not used)",
              "backend": V.backend_record(tasks), "task_set": f"tasks_discovery.json (seed {tasks['seed']})",
              "design": {"A": [entA, ansA], "B": [entB, ansB], "layer": LAYER, "rank": 1, "columns": M.N_COLUMNS, "rho": RHO,
                         "steps": S.MAX_STEPS, "lr": S.LR, "lambda": S.LAMBDA, "cap_from": "base W0 footprint RMS",
                         "endpoint": "final step", "arms": 7, "document": "bench/patch-train-1/PATCH-TRAIN-2.md"},
              "optimisation_approximation": "training forward rounds via torch f32->bf16 cast, straight-through; reported weights are "
                                            "the exact frozen v2 emulation",
              "hashes": {"W0": sha(W0)}, "stages": {}, "arms": {}, "trainings": {}}

    # ------------------------------------------------------------------ address overlap and cross-activation (recorded, not gated)
    def entity_acts(e):
        out = {}
        for name, tpl in [("train", t["train_templates"][0])] + [(f"para{i}", p) for i, p in enumerate(t["heldout_templates"])]:
            z, ids = M.capture(model, tok, tpl.format(e=e), columns)
            out[name] = z[M.entity_span(tok, tpl, e, ids)[1]]
        return out
    zA, zB = entity_acts(entA), entity_acts(entB)
    record["address_overlap"] = {
        "cosine_bA_bB": float(bA @ bB),
        "per_prompt": {nm: {"bA.zA": float(bA @ zA[nm]), "bB.zB": float(bB @ zB[nm]), "bA.zB": float(bA @ zB[nm]), "bB.zA": float(bB @ zA[nm]),
                            "cross_over_target_A_on_B": float((bA @ zB[nm]) / (bA @ zA[nm])), "cross_over_target_B_on_A": float((bB @ zA[nm]) / (bB @ zB[nm]))}
                       for nm in zA}}

    # ------------------------------------------------------------------ stage measurement
    def measure(bits, label):
        out = {"tensor_sha256": sha(bits)}
        with V.patched_weight(model, key, bits):
            reads = {}
            for ent in (entA, entB):
                reads[ent] = {}
                for name, tpl in [("train", t["train_templates"][0])] + [(f"para{i}", p) for i, p in enumerate(t["heldout_templates"])]:
                    lp = V.logprobs(model, tok, tpl.format(e=ent))
                    al = lp[ans]
                    rec = {"inset_argmax": t["answers"][int(al.argmax().item())], "full_argmax_is_target_of": []}
                    for an, tid in ((ansA, tidA), (ansB, tidB)):
                        rec[an] = {"logp": lp[tid].item(), "rank": 1 + int((lp > lp[tid]).sum().item()), "top1": int(lp.argmax().item()) == tid,
                                   "cond_logp": lp[tid].item() - torch.logsumexp(al, 0).item()}
                    rec["mass"] = torch.logsumexp(al, 0).item()
                    reads[ent][name] = rec
            per = {an: {e: V.logprobs(model, tok, t["train_templates"][0].format(e=e))[tid].item() for e in all_ents}
                   for an, tid in ((ansA, tidA), (ansB, tidB))}
            real = [(V.logprobs(model, tok, f["template"].format(e=f["country"])), V.first_token(tok, f["answer"])) for f in tasks["real_facts"]]
            out.update({"reads": reads, "per_entity_logp_train_template": per,
                        "real_top1": sum(int(lp.argmax().item()) == y for lp, y in real) / len(real),
                        "real_mean_logp": sum(lp[y].item() for lp, y in real) / len(real),
                        "neutral_nll": V.neutral_nll(model, tok, tasks["neutral_text"])})
        record["stages"][label] = out
        return out

    base = measure(W0, "W0")
    base_per = base["per_entity_logp_train_template"]

    def selectivity(stage, ans_name, ent):
        per = stage["per_entity_logp_train_template"][ans_name]
        d_t = per[ent] - base_per[ans_name][ent]
        d_o = [per[e] - base_per[ans_name][e] for e in all_ents if e != ent]
        return {"d_logp_target_entity": d_t, "d_logp_others_mean": float(np.mean(d_o)), "d_logp_others_max": float(max(d_o)),
                "S": d_t - float(np.mean(d_o))}

    def summarise(label, stage):
        rd = stage["reads"]
        row = {}
        for nm, ent, an in (("A", entA, ansA), ("B", entB, ansB)):
            r = rd[ent]
            row[nm] = {"train_top1": r["train"][an]["top1"], "train_rank": r["train"][an]["rank"],
                       "para_top1": [r[f"para{i}"][an]["top1"] for i in range(3)], "para_rank": [r[f"para{i}"][an]["rank"] for i in range(3)],
                       "S": selectivity(stage, an, ent)["S"], "others_max": selectivity(stage, an, ent)["d_logp_others_max"]}
        other_ans_on_ent = {"A_answer_on_B_entity": rd[entB]["train"][ansA], "B_answer_on_A_entity": rd[entA]["train"][ansB]}
        row["cross"] = {k: {"rank": v["rank"], "top1": v["top1"]} for k, v in other_ans_on_ent.items()}
        row["collateral"] = {"real_top1": stage["real_top1"], "neutral_nll": stage["neutral_nll"], "neutral_delta": stage["neutral_nll"] - base["neutral_nll"]}
        return row

    # ------------------------------------------------------------------ same-layer address invariance (harness check)
    texts = [tpl.format(e=e) for e in all_ents for tpl in [t["train_templates"][0]] + t["heldout_templates"]] + list(tasks["neutral_text"])
    z0 = [capture_full(model, tok, x) for x in texts]

    def z_invariant(bits):
        with V.patched_weight(model, key, bits):
            z1 = [capture_full(model, tok, x) for x in texts]
        return {"prompts": len(texts), "bitwise_identical": all(np.array_equal(a, b) for a, b in zip(z0, z1)),
                "max_abs_diff": float(max(np.abs(a - b).max() for a, b in zip(z0, z1)))}

    # ------------------------------------------------------------------ learning (always through R.train, same hyper-parameters)
    def learn(name, ent, tid, b_np, src_bits):
        t0 = time.time()
        a, curve, bind, cap = R.train(model, tok, tasks, key, f32(src_bits), src_bits, columns, ent, tid, b_np, RHO, ans, rms_w)
        record["trainings"][name] = {"source_sha": sha(src_bits), "cap_binding_steps": bind, "update_frobenius_cap": cap,
                                     "loss_first_last": [curve[0]["loss"], curve[-1]["loss"]], "curve": curve, "seconds": time.time() - t0}
        print(f"trained {name}: loss {curve[0]['loss']:.2f}->{curve[-1]['loss']:.2f} bind {bind} ({time.time() - t0:.0f}s)", flush=True)
        return Patch(name, a, b_np.reshape(1, -1), sha(src_bits))

    def dump():
        with open(os.path.join(HERE, args.out), "w") as fh:
            json.dump(record, fh, indent=1)
            fh.write("\n")

    # arm 1, 2: learned on the base
    PA = learn("P_A|W0", entA, tidA, bA, W0)
    PB = learn("P_B|W0", entB, tidB, bB, W0)
    W1A = apply_patch(W0, columns, PA)              # A only
    W1B = apply_patch(W0, columns, PB)              # B only
    record["hashes"].update({"W1_A": sha(W1A), "W1_B": sha(W1B), "P_A|W0": PA.payload_sha, "P_B|W0": PB.payload_sha})
    record["arms"]["1_A_only"] = summarise("1", measure(W1A, "1_A_only"))
    record["arms"]["2_B_only"] = summarise("2", measure(W1B, "2_B_only"))
    record["same_layer_address_invariance"] = {"W1_A": z_invariant(W1A), "W1_B": z_invariant(W1B)}
    dump()

    # determinism check against the committed replication run (rho 0.20)
    rep = json.load(open(os.path.join(HERE, "rep1_layer3_discovery.json")))
    record["determinism_vs_rep1"] = {"applicable": default_pair} if not default_pair else {
        "A_train_logp_equal": rep["arms"]["rho0.2/ent1_assigned"]["target_answer_reads_on_both_entities"][entA]["train"]["logp"] == record["stages"]["1_A_only"]["reads"][entA]["train"][ansA]["logp"],
        "B_train_logp_equal": rep["arms"]["rho0.2/ent2_assigned"]["target_answer_reads_on_both_entities"][entB]["train"]["logp"] == record["stages"]["2_B_only"]["reads"][entB]["train"][ansB]["logp"]}
    print("determinism vs rep1:", record["determinism_vs_rep1"], flush=True)
    dump()

    # arms 3, 4: independent composition, patches REBOUND to the new source
    stale = None
    try:
        apply_patch(W1A, columns, PB)               # PB is bound to W0: must be refused on W1
    except Refused as e:
        stale = str(e)
    record["binding_refusal_check"] = {"independent_B_bound_to_W0_applied_to_W1_refused": stale is not None, "message": stale}
    PB_on_A = PB.rebound(sha(W1A), "P_B(rebound to W1_A)")
    PA_on_B = PA.rebound(sha(W1B), "P_A(rebound to W1_B)")
    W_AB, W_BA = apply_patch(W1A, columns, PB_on_A), apply_patch(W1B, columns, PA_on_B)
    record["arms"]["3_A_then_indepB"] = summarise("3", measure(W_AB, "3_A_then_indepB"))
    record["arms"]["4_B_then_indepA"] = summarise("4", measure(W_BA, "4_B_then_indepA"))
    dump()

    # arms 5, 6: cumulative -- learned against the predecessor's effective model, bound to it
    PB_seq = learn("P_B|W1_A", entB, tidB, bB, W1A)
    PA_seq = learn("P_A|W1_B", entA, tidA, bA, W1B)
    W2_AB, W2_BA = apply_patch(W1A, columns, PB_seq), apply_patch(W1B, columns, PA_seq)
    record["hashes"].update({"W_AB_indep": sha(W_AB), "W_BA_indep": sha(W_BA), "W2_AB_seq": sha(W2_AB), "W2_BA_seq": sha(W2_BA),
                             "P_B|W1_A": PB_seq.payload_sha, "P_A|W1_B": PA_seq.payload_sha})
    record["arms"]["5_A_then_B_learned_on_A"] = summarise("5", measure(W2_AB, "5_A_then_B_learned_on_A"))
    record["arms"]["6_B_then_A_learned_on_B"] = summarise("6", measure(W2_BA, "6_B_then_A_learned_on_B"))
    dump()

    # arm 7: joint single stage, one rounding
    joint = Patch("P_joint", np.concatenate([PA.a, PB.a], axis=1), np.concatenate([PA.b, PB.b], axis=0), sha(W0))
    W_joint = apply_patch(W0, columns, joint)
    record["hashes"]["W_joint"] = sha(W_joint)
    record["arms"]["7_joint_rank2"] = summarise("7", measure(W_joint, "7_joint_rank2"))

    # ------------------------------------------------------------------ numerical differences, element by element
    def qs(v):
        return {k: float(np.percentile(v, q)) for k, q in (("p50", 50), ("p90", 90), ("p99", 99), ("max", 100))} if len(v) else None

    MAG_FLOOR = rms_w                      # declared: relative differences only where max(|x|,|y|) >= the footprint RMS of W0

    def diff(x, y):
        """Exact differing-element count; absolute-difference distribution over differing elements; relative difference only above a
        declared magnitude floor; difference in the values' OWN BF16 spacing where meaningful (both nonzero, same sign)."""
        xs, ys = E.bf16_to_f64(x[:, columns]), E.bf16_to_f64(y[:, columns])
        ne = x[:, columns] != y[:, columns]
        mag = np.maximum(np.abs(xs), np.abs(ys))
        ad = np.abs(xs - ys)[ne]
        above = ne & (mag >= MAG_FLOOR)
        meaningful = ne & (xs != 0) & (ys != 0) & (np.sign(xs) == np.sign(ys)) & (mag >= 2.0 ** -126)
        spacing = 2.0 ** (np.floor(np.log2(np.where(mag > 0, mag, 1.0))) - 7)
        return {"elements_differing": int(ne.sum()), "elements_total": int(ne.size), "fraction_differing": float(ne.mean()),
                "abs_difference_over_differing_elements": qs(ad),
                "relative_difference": {"magnitude_floor": MAG_FLOOR, "elements_above_floor_and_differing": int(above.sum()),
                                        "quantiles": qs((np.abs(xs - ys) / np.where(mag > 0, mag, 1.0))[above])},
                "difference_in_own_bf16_spacings": {"elements_meaningful": int(meaningful.sum()), "elements_excluded_zero_or_sign_flip": int((ne & ~meaningful).sum()),
                                                    "quantiles": qs((np.abs(xs - ys) / spacing)[meaningful])},
                "outside_footprint_identical": bool(np.array_equal(np.delete(x, columns, axis=1), np.delete(y, columns, axis=1)))}
    record["numerics"] = {
        "W_AB_vs_W_BA": diff(W_AB, W_BA), "W_AB_vs_W_joint": diff(W_AB, W_joint), "W_BA_vs_W_joint": diff(W_BA, W_joint),
        "W2_AB_seq_vs_W_AB_indep": diff(W2_AB, W_AB), "W2_BA_seq_vs_W_BA_indep": diff(W2_BA, W_BA),
        "survival": {"A_on_W0": float((W1A != W0).sum() / (rows * M.N_COLUMNS)), "B_on_W0": float((W1B != W0).sum() / (rows * M.N_COLUMNS)),
                     "B_on_W1_A_indep": float((W_AB != W1A).sum() / (rows * M.N_COLUMNS)),
                     "A_on_W1_B_indep": float((W_BA != W1B).sum() / (rows * M.N_COLUMNS)),
                     "B_on_W1_A_learned": float((W2_AB != W1A).sum() / (rows * M.N_COLUMNS)),
                     "A_on_W1_B_learned": float((W2_BA != W1B).sum() / (rows * M.N_COLUMNS))},
        "B_elements_that_change_on_W0_but_not_on_W1_A": int(((W1B != W0) & (W_AB == W1A)).sum()),
        "B_elements_that_change_on_W1_A_but_not_on_W0": int(((W1B == W0) & (W_AB != W1A)).sum())}

    # ------------------------------------------------------------------ reversibility by replay, and binding on removal
    replay_W1 = apply_patch(W0, columns, Patch("replay", PA.a, PA.b, sha(W0)))
    replay_W2 = apply_patch(replay_W1, columns, Patch("replay", PB_seq.a, PB_seq.b, sha(replay_W1)))
    removal = None
    try:
        apply_patch(W0, columns, PB_seq)             # drop A from the chain: B is bound to W1_A and must be refused
    except Refused as e:
        removal = str(e)
    record["reversibility"] = {"prefix_replay_reproduces_W1_A_bit_for_bit": bool(np.array_equal(replay_W1, W1A)),
                               "full_chain_replay_reproduces_W2_AB_bit_for_bit": bool(np.array_equal(replay_W2, W2_AB)),
                               "B_after_removing_A_refused": removal is not None, "message": removal,
                               "W0_untouched": sha(W0) == record["hashes"]["W0"]}
    record["same_layer_address_invariance"].update({"W_AB_indep": z_invariant(W_AB), "W2_AB_seq": z_invariant(W2_AB)})

    # ------------------------------------------------------------------ retention, descriptive (per prompt vs own isolated result)
    def retention(stack_label, first, second):
        """`first` patch's behaviour in the stack vs its isolated arm; per prompt."""
        (nm, ent, an, iso_label) = first
        iso, stk = record["stages"][iso_label]["reads"][ent], record["stages"][stack_label]["reads"][ent]
        out = {}
        for p in ("train", "para0", "para1", "para2"):
            out[p] = {"isolated_top1": iso[p][an]["top1"], "stack_top1": stk[p][an]["top1"], "isolated_rank": iso[p][an]["rank"],
                      "stack_rank": stk[p][an]["rank"], "logp_change": stk[p][an]["logp"] - iso[p][an]["logp"]}
        return out
    A1, B1 = ("A", entA, ansA, "1_A_only"), ("B", entB, ansB, "2_B_only")
    record["retention_descriptive"] = {
        "arm3_A_in_A+indepB": retention("3_A_then_indepB", A1, B1), "arm3_B_in_A+indepB": retention("3_A_then_indepB", B1, A1),
        "arm4_A_in_B+indepA": retention("4_B_then_indepA", A1, B1), "arm4_B_in_B+indepA": retention("4_B_then_indepA", B1, A1),
        "arm5_A_in_A+Blearned": retention("5_A_then_B_learned_on_A", A1, B1), "arm5_B_in_A+Blearned": retention("5_A_then_B_learned_on_A", B1, A1),
        "arm6_A_in_B+Alearned": retention("6_B_then_A_learned_on_B", A1, B1), "arm6_B_in_B+Alearned": retention("6_B_then_A_learned_on_B", B1, A1),
        "arm7_A_in_joint": retention("7_joint_rank2", A1, B1), "arm7_B_in_joint": retention("7_joint_rank2", B1, A1)}
    record["model_digest_unchanged_at_end"] = V.model_digest(model) == digest0
    dump()

    for k, v in record["arms"].items():
        A, B = v["A"], v["B"]
        print(f"{k:26s} A: train {A['train_rank']:>5} para {A['para_rank']} S {A['S']:+.1f} | B: train {B['train_rank']:>5} para {B['para_rank']} "
              f"S {B['S']:+.1f} | cross A@B {v['cross']['A_answer_on_B_entity']['rank']} B@A {v['cross']['B_answer_on_A_entity']['rank']} | "
              f"real {v['collateral']['real_top1']:.2f} dNLL {v['collateral']['neutral_delta']:+.3f}", flush=True)
    print("address overlap:", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in record["address_overlap"].items() if k != "per_prompt"}),
          "cross/target (train):", round(record["address_overlap"]["per_prompt"]["train"]["cross_over_target_A_on_B"], 3),
          round(record["address_overlap"]["per_prompt"]["train"]["cross_over_target_B_on_A"], 3), flush=True)
    print("numerics W_AB vs W_joint:", json.dumps({k: record["numerics"]["W_AB_vs_W_joint"][k] for k in ("elements_differing", "fraction_differing", "abs_difference_over_differing_elements")}),
          "|", json.dumps(record["reversibility"]), flush=True)
    print("address invariance:", {k: v["bitwise_identical"] for k, v in record["same_layer_address_invariance"].items()})
    print("model restored:", record["model_digest_unchanged_at_end"])


if __name__ == "__main__":
    main()
