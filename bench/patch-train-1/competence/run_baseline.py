#!/usr/bin/env python3
"""PATCH-TRAIN-1 baseline competence measurement. BASELINE ONLY: the model is never modified, nothing is trained.

Declared backend (used for every behavioural comparison): PyTorch on CPU, float32 compute, eager attention, batch size 1,
no padding, the container's BF16 weights widened exactly to f32. LARQL's INFER is a cross-check only (the 270M model has a
documented strict-f32 equivalence issue between LARQL and the reference, so the two are not assumed interchangeable).

    python3 run_baseline.py        (needs tasks.json from build_tasks.py; writes baseline.json)
"""
import hashlib
import json
import math
import os
import platform
import subprocess
import sys

import torch

# Environment workaround, recorded in the output: the installed torchvision (0.21) does not match torch (2.14) and breaks
# transformers' import of Gemma 3. This text-only model never uses it, so it is declared unavailable (no package is changed).
TORCHVISION_MASKED = "torchvision" not in sys.modules
sys.modules["torchvision"] = None
import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO_ROOT, "verifier"))
from refcheck import container as C  # noqa: E402

CONTAINER = os.path.expanduser("~/chris-models/larql-registry/gemma-3-270m-it.vindex3")
LARQL = os.path.join(REPO_ROOT, "target", "release", "larql")
CROSS_CHECK_PROMPTS = 12
BF16 = "BF16"


def bf16_to_f32(raw, shape):
    import numpy as np
    words = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    return torch.from_numpy(words.view(np.float32).copy()).reshape(shape)


def identity_check(model):
    """Every container tensor equals the HF checkpoint's tensor bit for bit (after BF16)."""
    c = C.open_container(CONTAINER)
    state = model.state_dict()
    checked, mismatched, missing = 0, [], []
    for t in c.tensors(BF16):
        if t.hf_key not in state:
            missing.append(t.hf_key)
            continue
        want = bf16_to_f32(c.read_bytes(t), t.shape)
        got = state[t.hf_key]
        checked += 1
        if not torch.equal(got.to(torch.bfloat16).to(torch.float32), want) or not torch.equal(got, want):
            mismatched.append(t.hf_key)
    return {"container_tensors_checked": checked, "mismatched": mismatched, "not_in_checkpoint": missing,
            "identical": not mismatched and not missing}


def logprobs(model, tok, prompt):
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, -1].to(torch.float32)
    return torch.log_softmax(logits, dim=-1)


def first_token(tok, answer):
    return tok(answer, add_special_tokens=False)["input_ids"][0]


def measure_prompt(model, tok, prompt, target, answer_ids):
    lp = logprobs(model, tok, prompt)
    t = first_token(tok, target)
    in_set = [lp[i].item() for i in answer_ids]
    others = [lp[i].item() for i in answer_ids if i != t]
    return {"logp_target": lp[t].item(), "top1_full_vocab": int(lp.argmax().item()) == t,
            "top1_in_answer_set": lp[t].item() >= max(in_set), "margin_in_set": lp[t].item() - max(others),
            "rank_in_set": 1 + sum(1 for v in in_set if v > lp[t].item())}


def neutral_nll(model, tok, texts):
    total, count = 0.0, 0
    for s in texts:
        ids = tok(s, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0].to(torch.float32)
        lp = torch.log_softmax(logits[:-1], dim=-1)
        nll = -lp.gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1)
        total += nll.sum().item()
        count += nll.numel()
    return total / count


def run_all(model, tok, tasks):
    out = {"tasks": {}, "real_facts": [], "neutral_nll": neutral_nll(model, tok, tasks["neutral_text"])}
    for key, t in tasks["tasks"].items():
        ans_ids = [first_token(tok, a) for a in t["answers"]]
        rows = []
        for split, entities, templates in (("trained_entities_train_template", t["trained_entities"], t["train_templates"]),
                                           ("trained_entities_paraphrase", t["trained_entities"], t["heldout_templates"]),
                                           ("heldout_entities_train_template", t["heldout_entities"], t["train_templates"]),
                                           ("heldout_entities_paraphrase", t["heldout_entities"], t["heldout_templates"])):
            for e in entities:
                for tpl in templates:
                    r = measure_prompt(model, tok, tpl.format(e=e), t["assignment"][e], ans_ids)
                    rows.append({"split": split, "entity": e, "template": tpl, **r})
        out["tasks"][key] = rows
    for f in tasks["real_facts"]:
        prompt = f["template"].format(e=f["country"])
        lp = logprobs(model, tok, prompt)
        t = first_token(tok, f["answer"])
        out["real_facts"].append({"country": f["country"], "logp_target": lp[t].item(),
                                  "top1": int(lp.argmax().item()) == t, "top1_token": tok.decode([int(lp.argmax().item())])})
    return out


def summarise(res):
    s = {}
    for key, rows in res["tasks"].items():
        by = {}
        for r in rows:
            by.setdefault(r["split"], []).append(r)
        s[key] = {sp: {"n": len(v), "mean_logp_target": sum(x["logp_target"] for x in v) / len(v),
                       "top1_full_vocab": sum(x["top1_full_vocab"] for x in v) / len(v),
                       "top1_in_answer_set": sum(x["top1_in_answer_set"] for x in v) / len(v),
                       "mean_margin_in_set": sum(x["margin_in_set"] for x in v) / len(v)} for sp, v in by.items()}
        k = len(next(iter(by.values())))
        s[key]["chance_top1_in_answer_set"] = 1 / 12
    rf = res["real_facts"]
    s["real_facts"] = {"n": len(rf), "top1": sum(x["top1"] for x in rf) / len(rf),
                       "mean_logp_target": sum(x["logp_target"] for x in rf) / len(rf)}
    s["neutral_nll_per_token"] = res["neutral_nll"]
    return s


def larql_cross_check(tok, tasks, torch_res):
    """Top-1 token of LARQL's INFER on the container vs the declared backend, on a few prompts. Informational only."""
    prompts = [(f["template"].format(e=f["country"]), torch_res["real_facts"][i]["top1_token"])
               for i, f in enumerate(tasks["real_facts"][:CROSS_CHECK_PROMPTS])]
    rows = []
    for prompt, torch_top1 in prompts:
        p = subprocess.run([LARQL, "lql", f'USE "{CONTAINER}"; INFER "{prompt}" TOP 1;'], capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.strip().startswith("1.")), "")
        rows.append({"prompt": prompt, "torch_top1": torch_top1, "larql_line": line.strip(),
                     "agree": torch_top1.strip() in line})
    return rows


def main():
    tasks = json.load(open(os.path.join(HERE, "tasks.json")))
    m = tasks["model"]
    tok = AutoTokenizer.from_pretrained(m["repo"], revision=m["revision"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(m["repo"], revision=m["revision"], dtype=torch.float32,
                                                 attn_implementation="eager", local_files_only=True).eval()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    record = {"protocol": "PATCH-TRAIN-1 baseline competence (baseline only; no model modified)",
              "backend": {"name": "pytorch-cpu-f32-eager-bs1", "torch": torch.__version__, "transformers": transformers.__version__,
                          "python": platform.python_version(), "threads": torch.get_num_threads(), "device": "cpu",
                          "machine": platform.machine(), "model": m,
                          "environment_workaround": "torchvision masked via sys.modules (installed 0.21 incompatible with torch 2.14); text-only model"},
              "tasks_sha256": hashlib.sha256(open(os.path.join(HERE, "tasks.json"), "rb").read()).hexdigest(),
              "identity": identity_check(model)}
    first = run_all(model, tok, tasks)
    second = run_all(model, tok, tasks)
    record["determinism"] = {"two_runs_bitwise_identical": first == second}
    record["summary"] = summarise(first)
    record["larql_cross_check_informational"] = larql_cross_check(tok, tasks, first)
    record["rows"] = first
    with open(os.path.join(HERE, "baseline.json"), "w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    print(json.dumps({k: record[k] for k in ("identity", "determinism", "summary")}, indent=1))


if __name__ == "__main__":
    main()
