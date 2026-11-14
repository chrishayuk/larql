#!/usr/bin/env python3
"""PATCH-TRAIN-1 L1 step 3: BF16 survival and sensitivity study. A SENSITIVITY STUDY, not training and not a confirmatory
result. One fixed candidate footprint, a range of update magnitudes, three directions (the top singular direction of the
base model's gradient for task A, for task B, and a norm-matched random control). For each: how much of the update survives
BF16 rounding, what the rounded patch does to the loss compared with the unrounded ideal, and what it does to collateral.
It uses the DISCOVERY entity set (tasks_discovery.json), never the confirmatory one (tasks.json). No parameter is optimised.
    python3 -I feasibility.py        writes feasibility.json
"""
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

TENSOR, N_COLUMNS, SEED = "9.mlp.down_proj.weight", 256, 20261113
RHOS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)   # update element RMS as a fraction of the footprint's weight RMS
PROFILE = E.PROFILE_RNE1


def task_loss_and_gradient(model, tok, tasks, key, task):
    """Mean cross-entropy of the first answer token over the trained entities' training prompts, and its gradient with
    respect to the footprint tensor (autograd on the base model; nothing is changed)."""
    t = tasks["tasks"][task]
    param = dict(model.named_parameters())[key]
    param.requires_grad_(True)
    param.grad = None
    losses = []
    for e in t["trained_entities"]:
        for tpl in t["train_templates"]:
            ids = tok(tpl.format(e=e), return_tensors="pt")["input_ids"]
            lp = torch.log_softmax(model(input_ids=ids).logits[0, -1].float(), dim=-1)
            losses.append(-lp[V.first_token(tok, t["assignment"][e])])
    loss = torch.stack(losses).mean()
    loss.backward()
    grad = param.grad.detach().clone()
    param.requires_grad_(False)
    param.grad = None
    return loss.item(), grad


def top_singular(grad, columns):
    g = grad[:, columns]
    u, s, vh = torch.linalg.svd(g, full_matrices=False)
    return u[:, 0].double().numpy(), float(s[0]), vh[0].double().numpy(), float(torch.linalg.norm(g))


def half_ulp_survival(base_bits, columns, delta):
    """Analytic prediction: an element survives if |delta| reaches half an ulp of its weight (nearest-even ties aside)."""
    w = E.bf16_to_f64(base_bits[:, columns])
    exp = np.floor(np.log2(np.where(w == 0, 1.0, np.abs(w))))
    half_ulp = np.where(w == 0, 0.0, 2.0 ** (np.maximum(exp, -126) - 8))
    return float((np.abs(delta) >= half_ulp).mean())


def summary(model, tok, tasks, which):
    res = V.evaluate_suite(model, tok, tasks)
    s = V.summarise(res)
    out = {"neutral_nll": s["neutral_nll_per_token"], "real_top1": s["real_facts"]["top1"],
           "real_mean_logp": s["real_facts"]["mean_logp_target"]}
    for key in ("A", "B"):
        for split in ("trained_entities_train_template", "trained_entities_paraphrase", "heldout_entities_paraphrase"):
            m = s[key][split]
            out[f"{key}.{split}"] = {k: m[k] for k in ("mean_logp_target", "top1_full_vocab", "median_rank_full_vocab",
                                                       "top1_in_answer_set", "mean_margin_in_set", "mean_logp_answer_set_mass")}
    return out


def main():
    tasks = V.load_tasks("tasks_discovery.json")
    assert tasks["seed"] != V.load_tasks("tasks.json")["seed"]
    model, tok = V.load_model(tasks)
    digest0 = V.model_digest(model)
    key = V.hf_key(TENSOR)
    base_bits = V.weight_bits(model, key)
    rows, cols = base_bits.shape
    rng = np.random.default_rng(SEED)
    columns = np.sort(rng.choice(cols, N_COLUMNS, replace=False))
    w_f32 = torch.from_numpy(E.bf16_to_f32(base_bits).copy())
    rms_w = float(np.sqrt((E.bf16_to_f64(base_bits[:, columns]) ** 2).mean()))
    n_elem = rows * len(columns)

    record = {"protocol": "PATCH-TRAIN-1 L1 feasibility / sensitivity study (NOT training; NOT confirmatory)",
              "backend": V.backend_record(tasks), "task_set": "tasks_discovery.json (seed %d)" % tasks["seed"],
              "footprint": {"tensor": TENSOR, "shape": [rows, cols], "columns": len(columns), "seed": SEED, "rank": 1,
                            "profile": PROFILE, "rms_of_footprint_weights": rms_w}, "base": summary(model, tok, tasks, None),
              "directions": {}}
    print("base summary done", flush=True)
    directions = {}
    for task in ("A", "B"):
        loss, grad = task_loss_and_gradient(model, tok, tasks, key, task)
        u, s1, v, gnorm = top_singular(grad, columns)
        directions[f"gradient_{task}"] = (u, v, {"base_train_loss": loss, "top_singular_value": s1,
                                                 "gradient_frobenius_norm_on_footprint": gnorm,
                                                 "fraction_of_gradient_norm_in_top_direction": s1 / gnorm})
    ru, rv = rng.standard_normal(rows), rng.standard_normal(len(columns))
    directions["random_control"] = (ru / np.linalg.norm(ru), rv / np.linalg.norm(rv), {})
    for name, (u, v, info) in directions.items():
        sign = -1.0 if name.startswith("gradient") else 1.0       # negative gradient direction lowers the loss to first order
        sweep = []
        for rho in RHOS:
            t = rho * rms_w * np.sqrt(n_elem)                      # Frobenius norm of the update
            a, b = (sign * t * u).reshape(rows, 1), v.reshape(1, -1)
            new_bits, written = E.apply_columns(base_bits, columns, a, b, PROFILE)
            delta = (a @ b)                                        # the ideal float64 update
            realised = E.bf16_to_f64(new_bits[:, columns]) - E.bf16_to_f64(base_bits[:, columns])
            row = {"rho": rho, "update_frobenius": float(t),
                   "elements_changed": int((new_bits != base_bits).sum()), "survival_fraction": float((new_bits != base_bits).sum() / n_elem),
                   "predicted_survival_half_ulp": half_ulp_survival(base_bits, columns, delta),
                   "realised_over_ideal_inner_product": float((realised * delta).sum() / (delta * delta).sum()),
                   "realised_frobenius_over_ideal": float(np.linalg.norm(realised) / np.linalg.norm(delta))}
            if name.startswith("gradient"):
                row["first_order_loss_change_ideal"] = float(sign * t * info["top_singular_value"])
            ideal = w_f32.clone()
            ideal[:, torch.from_numpy(columns)] += torch.from_numpy(delta.astype(np.float32))
            with V.patched_values(model, key, ideal):
                row["unrounded_f32_ideal"] = summary(model, tok, tasks, None)
            with V.patched_weight(model, key, new_bits):
                row["bf16_rounded"] = summary(model, tok, tasks, None)
            sweep.append(row)
            r = row["bf16_rounded"]
            print(f"{name:16s} rho={rho:<7g} survive={row['survival_fraction']:.3f} (pred {row['predicted_survival_half_ulp']:.3f}) "
                  f"A.train logp {row['unrounded_f32_ideal']['A.trained_entities_train_template']['mean_logp_target']:.2f}"
                  f"/{r['A.trained_entities_train_template']['mean_logp_target']:.2f} realReal {r['real_top1']:.2f}", flush=True)
        record["directions"][name] = {"info": info, "sweep": sweep}
    record["model_digest_unchanged_at_end"] = V.model_digest(model) == digest0
    with open(os.path.join(HERE, "feasibility.json"), "w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    print("model restored:", record["model_digest_unchanged_at_end"])


if __name__ == "__main__":
    main()
