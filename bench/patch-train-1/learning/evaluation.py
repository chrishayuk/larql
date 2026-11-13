"""The declared evaluation backend for PATCH-TRAIN-1, and the way a patch is applied to it without touching the base.

Backend (every behavioural comparison uses exactly this): PyTorch on CPU, float32 compute, eager attention, batch size 1, no
padding, the checkpoint's BF16 weights widened exactly to f32. A patched weight is the emulator's BF16 result widened the same
way. The immutable base (the HF checkpoint in its cache, the registry container) is never written: a patch exists only as a
temporary replacement of one parameter's values, restored on exit and checked by digest.
"""
import contextlib
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
sys.modules.setdefault("torchvision", None)   # installed torchvision 0.21 does not match torch 2.14; text-only model (recorded)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
COMPETENCE = os.path.join(os.path.dirname(HERE), "competence")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "verifier"))
import emulate as E  # noqa: E402
from refcheck import container as C  # noqa: E402

CONTAINER = os.path.expanduser("~/chris-models/larql-registry/gemma-3-270m-it.vindex3")
LOW16 = 0xFFFF
BACKEND = "pytorch-cpu-f32-eager-bs1"


def load_tasks(name="tasks.json"):
    with open(os.path.join(COMPETENCE, name)) as f:
        return json.load(f)


def load_model(tasks):
    m = tasks["model"]
    tok = AutoTokenizer.from_pretrained(m["repo"], revision=m["revision"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(m["repo"], revision=m["revision"], dtype=torch.float32,
                                                 attn_implementation="eager", local_files_only=True).eval()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def hf_key(tensor_name):
    return "model.layers." + tensor_name


def weight_bits(model, key):
    """A parameter's BF16 bit pattern. The checkpoint is BF16-valued, so the low 16 bits of every f32 word are zero."""
    w = model.state_dict()[key].detach().contiguous().view(torch.int32).numpy().astype(np.uint32)
    assert not (w & np.uint32(LOW16)).any(), f"{key} is not exactly BF16-valued"
    return (w >> np.uint32(16)).astype(np.uint16)


def model_digest(model, skip=()):
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        if name in skip:
            continue
        h.update(name.encode())
        h.update(p.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


@contextlib.contextmanager
def patched_weight(model, key, new_bits):
    """Temporarily replace one parameter with the widened BF16 bits `new_bits`; restore the original values on exit."""
    param = dict(model.named_parameters())[key]
    original = param.data.clone()
    param.data.copy_(torch.from_numpy(E.bf16_to_f32(new_bits).copy()))
    try:
        yield
    finally:
        param.data.copy_(original)


@contextlib.contextmanager
def patched_values(model, key, values_f32):
    """Like patched_weight, but with arbitrary f32 values (the UNROUNDED ideal a rounded patch is compared against)."""
    param = dict(model.named_parameters())[key]
    original = param.data.clone()
    param.data.copy_(values_f32)
    try:
        yield
    finally:
        param.data.copy_(original)


def container_tensor_bits(container_path, hf_name):
    """One BF16 tensor of a compiled (or registry) container as a uint16 array, for loading into the declared backend."""
    c = C.open_container(container_path)
    ref = next(t for t in c.tensors("BF16") if t.hf_key == hf_name)
    return np.frombuffer(c.read_bytes(ref), dtype="<u2").reshape(ref.shape).copy()


# ------------------------------------------------------------------ metrics

def first_token(tok, answer):
    return tok(answer, add_special_tokens=False)["input_ids"][0]


def logprobs(model, tok, prompt):
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, -1].to(torch.float32)
    return torch.log_softmax(logits, dim=-1)


def prompt_metrics(lp, target_id, answer_ids):
    """Restricted-choice AND full-vocabulary measures, so an apparent gain cannot hide in either alone."""
    t = lp[target_id].item()
    in_set = [lp[i].item() for i in answer_ids]
    others = [lp[i].item() for i in answer_ids if i != target_id]
    return {"logp_target": t, "top1_full_vocab": int(lp.argmax().item()) == target_id,
            "rank_full_vocab": 1 + int((lp > lp[target_id]).sum().item()),
            "top1_in_answer_set": t >= max(in_set), "margin_in_set": t - max(others),
            "rank_in_set": 1 + sum(1 for v in in_set if v > t),
            "logp_answer_set_mass": torch.logsumexp(lp[answer_ids], dim=0).item()}


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


SPLITS = (("trained_entities_train_template", "trained_entities", "train_templates"),
          ("trained_entities_paraphrase", "trained_entities", "heldout_templates"),
          ("heldout_entities_train_template", "heldout_entities", "train_templates"),
          ("heldout_entities_paraphrase", "heldout_entities", "heldout_templates"))


def evaluate_suite(model, tok, tasks, which=("A", "B")):
    out = {"tasks": {}, "real_facts": [], "neutral_nll": neutral_nll(model, tok, tasks["neutral_text"])}
    for key in which:
        t = tasks["tasks"][key]
        ans = [first_token(tok, a) for a in t["answers"]]
        rows = []
        for split, ent_key, tpl_key in SPLITS:
            for e in t[ent_key]:
                for tpl in t[tpl_key]:
                    lp = logprobs(model, tok, tpl.format(e=e))
                    rows.append({"split": split, "entity": e, "template": tpl,
                                 **prompt_metrics(lp, first_token(tok, t["assignment"][e]), ans)})
        out["tasks"][key] = rows
    for f in tasks["real_facts"]:
        lp = logprobs(model, tok, f["template"].format(e=f["country"]))
        t = first_token(tok, f["answer"])
        out["real_facts"].append({"country": f["country"], "logp_target": lp[t].item(), "top1": int(lp.argmax().item()) == t,
                                  "top1_token": tok.decode([int(lp.argmax().item())])})
    return out


def summarise(res):
    s = {"neutral_nll_per_token": res["neutral_nll"]}
    for key, rows in res["tasks"].items():
        by = {}
        for r in rows:
            by.setdefault(r["split"], []).append(r)
        s[key] = {sp: {"n": len(v),
                       "mean_logp_target": sum(x["logp_target"] for x in v) / len(v),
                       "top1_full_vocab": sum(x["top1_full_vocab"] for x in v) / len(v),
                       "median_rank_full_vocab": sorted(x["rank_full_vocab"] for x in v)[len(v) // 2],
                       "top1_in_answer_set": sum(x["top1_in_answer_set"] for x in v) / len(v),
                       "mean_margin_in_set": sum(x["margin_in_set"] for x in v) / len(v),
                       "mean_logp_answer_set_mass": sum(x["logp_answer_set_mass"] for x in v) / len(v)}
                  for sp, v in by.items()}
    rf = res["real_facts"]
    s["real_facts"] = {"n": len(rf), "top1": sum(x["top1"] for x in rf) / len(rf),
                       "mean_logp_target": sum(x["logp_target"] for x in rf) / len(rf)}
    return s


def backend_record(tasks):
    import platform
    return {"name": BACKEND, "torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(), "threads": torch.get_num_threads(), "device": "cpu",
            "model": tasks["model"], "weights": "container BF16 widened exactly to f32 (identity with the checkpoint proved in baseline.json)"}
