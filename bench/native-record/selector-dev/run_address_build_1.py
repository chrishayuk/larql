#!/usr/bin/env python3
"""ADDRESS-BUILD-1 — at what depth do two phrasings of the same fact become
functionally interchangeable?

Transplants the residual ENTERING layer l at the final prompt position from a
donor prompt into a recipient prompt, then runs the rest of the network and
reads the next-token distribution. Native factual bindings only; no editing.

Arms, all four donors sharing the SAME wording so only the binding differs:
  T1 same binding, different wording      hypothesis
  T2 same relation, different entity
  T3 same entity,   different relation
  T4 unrelated binding

Reports the interchangeability gap G_l = mean KL(wrong-binding) - KL(T1), and
the REQUIRED disambiguator: donor-answer rate on T2/T3/T4, which separates
"a shared address emerged" from "the late state simply carries the answer".
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = ("/Users/christopherhay/.cache/huggingface/hub/models--google--gemma-3-4b-it"
         "/snapshots/093f9f388b31de276ce2de164bdc2081324b9767")
LAYERS = (8, 12, 16, 20, 24, 28, 30)
SEED = 20260906

FACTS = [
    ("Japan", "capital", "Tokyo"), ("France", "capital", "Paris"),
    ("Italy", "capital", "Rome"), ("Germany", "capital", "Berlin"),
    ("Norway", "capital", "Oslo"), ("Spain", "capital", "Madrid"),
    ("Egypt", "capital", "Cairo"), ("Cuba", "capital", "Havana"),
    ("Japan", "currency", "Yen"), ("India", "currency", "Rupee"),
    ("Poland", "currency", "Zloty"), ("Israel", "currency", "Shekel"),
    ("France", "language", "French"), ("Japan", "language", "Japanese"),
    ("Spain", "language", "Spanish"), ("Germany", "language", "German"),
    ("Italy", "language", "Italian"), ("Brazil", "language", "Portuguese"),
    ("Greece", "capital", "Athens"), ("Russia", "language", "Russian"),
]
SYNONYM = {"capital": "seat of government", "currency": "unit of money",
           "language": "native tongue"}
FORMS = [
    lambda e, r: f"The {r} of {e} is",
    lambda e, r: f"{e}'s {r} is",
    lambda e, r: f"If you asked someone to name the {r} of {e}, they would say",
    lambda e, r: f"Consider {e}. Its {SYNONYM[r]} is",
]


def kl(p_logits, q_logits):
    def sm(x):
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()
    p, q = sm(np.asarray(p_logits, np.float64)), sm(np.asarray(q_logits, np.float64))
    m = p > 0
    return float(np.sum(p[m] * (np.log(p[m]) - np.log(np.maximum(q[m], 1e-300)))))


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import mlx.core as mx
    from chuk_lazarus.models_v2.loader import load_model, ModelDType

    loaded = load_model(MODEL, dtype=ModelDType.BFLOAT16)
    model, tok = loaded.model, loaded.tokenizer
    model.eval()

    layer_cls = type(model.model.layers[0])
    orig_call = layer_cls.__call__
    index = {id(l): i for i, l in enumerate(model.model.layers)}
    st = {"capture": (), "inject_at": None, "donor": None, "caps": {}}

    def hooked(self, x, *a, **kw):
        i = index.get(id(self))
        if i is not None:
            if i in st["capture"]:
                st["caps"][i] = np.array(x[0, -1].astype(mx.float32))
            if st["inject_at"] is not None and i == st["inject_at"]:
                d = mx.array(st["donor"]).astype(x.dtype).reshape(1, 1, -1)
                x = mx.concatenate([x[:, :-1, :], d], axis=1)
        return orig_call(self, x, *a, **kw)

    layer_cls.__call__ = hooked

    def forward(prompt):
        ll = model(mx.array([tok.encode(prompt)])).logits[0, -1]
        mx.eval(ll)
        return np.array(ll.astype(mx.float32))

    def token_of(word):
        ids = [i for i in tok.encode(" " + word) if i != tok.bos_token_id]
        return ids[0] if len(ids) == 1 else None

    # ---- splice floor, mandatory before any arm is scored
    p0 = "The capital of Japan is"
    base0 = forward(p0)
    st["capture"] = (20,)
    forward(p0)
    st["capture"] = ()
    st["inject_at"], st["donor"] = 20, st["caps"][20]
    floor = float(np.max(np.abs(forward(p0) - base0)))
    st["inject_at"] = None
    print(f"SPLICE FLOOR max|logit delta| = {floor:.3e} "
          f"({'PASS' if floor == 0.0 else 'FAIL'})", flush=True)
    if floor != 0.0:
        raise SystemExit("REFUSE: splice floor not bit-identical")

    # ---- baseline pass: capture states, filter to prompts the model gets right
    t0 = time.time()
    items = []
    for bi, (e, r, ans) in enumerate(FACTS):
        tid = token_of(ans)
        if tid is None:
            continue
        for fi, form in enumerate(FORMS):
            prompt = form(e, r)
            st["capture"] = tuple(LAYERS)
            st["caps"] = {}
            ll = forward(prompt)
            if int(ll.argmax()) != tid:
                continue  # inclusion criterion: baseline must be correct
            items.append({"binding": bi, "entity": e, "relation": r, "answer": ans,
                          "answer_token": tid, "form": fi, "prompt": prompt,
                          "logits": ll, "states": dict(st["caps"])})
    st["capture"] = ()
    print(f"baseline: {len(items)}/{len(FACTS)*len(FORMS)} prompts kept "
          f"(model correct), {time.time()-t0:.1f}s", flush=True)

    by_binding = {}
    for it in items:
        by_binding.setdefault(it["binding"], []).append(it)
    rng = np.random.default_rng(SEED)

    rows = []
    for rec in items:
        pool = [x for x in by_binding[rec["binding"]] if x["form"] != rec["form"]]
        if not pool:
            continue
        donor_same = pool[0]
        df = donor_same["form"]
        same_rel = [x for x in items if x["relation"] == rec["relation"]
                    and x["entity"] != rec["entity"] and x["form"] == df]
        same_ent = [x for x in items if x["entity"] == rec["entity"]
                    and x["relation"] != rec["relation"] and x["form"] == df]
        unrel = [x for x in items if x["entity"] != rec["entity"]
                 and x["relation"] != rec["relation"] and x["form"] == df]
        arms = {"T1_same_binding": donor_same,
                "T2_diff_entity": same_rel[0] if same_rel else None,
                "T3_diff_relation": same_ent[0] if same_ent else None,
                "T4_unrelated": unrel[int(rng.integers(len(unrel)))] if unrel else None}
        for arm, donor in arms.items():
            if donor is None:
                continue
            for L in LAYERS:
                st["inject_at"], st["donor"] = L, donor["states"][L]
                out = forward(rec["prompt"])
                st["inject_at"] = None
                rows.append({
                    "recipient": rec["prompt"], "arm": arm, "layer": L,
                    "kl": kl(rec["logits"], out),
                    "retained": int(int(out.argmax()) == rec["answer_token"]),
                    "donor_answer": int(int(out.argmax()) == donor["answer_token"]),
                    "donor_is_recipient_answer": int(
                        donor["answer_token"] == rec["answer_token"]),
                })
    layer_cls.__call__ = orig_call

    # ---- aggregate
    ARMS = ("T1_same_binding", "T2_diff_entity", "T3_diff_relation", "T4_unrelated")
    summary = {}
    for L in LAYERS:
        s = {}
        for arm in ARMS:
            sel = [r for r in rows if r["layer"] == L and r["arm"] == arm]
            if not sel:
                continue
            # donor-answer rate is only meaningful where the donor's answer differs
            da = [r for r in sel if not r["donor_is_recipient_answer"]]
            s[arm] = {
                "n": len(sel),
                "median_kl": float(np.median([r["kl"] for r in sel])),
                "retention": sum(r["retained"] for r in sel) / len(sel),
                "donor_answer_rate": (sum(r["donor_answer"] for r in da) / len(da)) if da else None,
            }
        wrong = [s[a]["median_kl"] for a in ("T2_diff_entity", "T3_diff_relation") if a in s]
        s["G_l"] = float(np.mean(wrong) - s["T1_same_binding"]["median_kl"]) if wrong and "T1_same_binding" in s else None
        summary[f"L{L}"] = s

    body = json.dumps(rows, sort_keys=True)
    out = {"layers": list(LAYERS), "seed": SEED, "splice_floor_max_logit_delta": floor,
           "n_prompts_kept": len(items), "n_transplants": len(rows),
           "summary": summary, "rows_sha256": hashlib.sha256(body.encode()).hexdigest()}
    (HERE / "address_build_1_result.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"\n{'layer':>5} {'arm':<18} {'medKL':>8} {'retain':>7} {'donorAns':>9}")
    for L in LAYERS:
        for arm in ARMS:
            a = summary[f"L{L}"].get(arm)
            if a:
                da = "n/a" if a["donor_answer_rate"] is None else f"{a['donor_answer_rate']:.2f}"
                print(f"{L:>5} {arm:<18} {a['median_kl']:>8.3f} "
                      f"{a['retention']:>7.2f} {da:>9}")
        print(f"{L:>5} {'G_l':<18} {summary[f'L{L}']['G_l']:>8.3f}")
    print("\nrows_sha256", out["rows_sha256"])


if __name__ == "__main__":
    main()
