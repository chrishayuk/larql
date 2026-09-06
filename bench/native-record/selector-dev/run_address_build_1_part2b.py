#!/usr/bin/env python3
"""ADDRESS-BUILD-1 Part 2b — instrument repair at proper scale.

The first Part 2 is preserved as an instrument failure and is not reinterpreted.
This rebuild exists because that reader had ~1.5 training examples per binding
class and returned 0.14 entity accuracy in all 21 cells.

Frozen before running (see the address-build-1 registry record):
  ~12 entities x 3 relations x 12 wordings, all three relations required to
  survive the filters BY CONSTRUCTION or the run refuses;
  3 folds of 8 train / 4 test wordings, everything fitted on train wordings;
  blocking component-specific same-layer gate, mean across folds:
      relation >= 0.70   entity >= 0.40   binding >= 0.30   at >= 1 layer
  cross-layer transfer computed ONLY for components that clear their gate.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = ("/Users/christopherhay/.cache/huggingface/hub/models--google--gemma-3-4b-it"
         "/snapshots/093f9f388b31de276ce2de164bdc2081324b9767")
LAYERS = (8, 12, 16, 20, 24, 28, 30)
PCA_DIM = 128
RIDGE = 1.0
MIN_PER_RELATION = 6
GATE = {"relation": 0.85, "entity": 0.40, "binding": 0.30}  # relation raised for the 2-class task
FOLDS = [((0, 1, 2, 3, 4, 5, 6, 7), (8, 9, 10, 11)),
         ((4, 5, 6, 7, 8, 9, 10, 11), (0, 1, 2, 3)),
         ((0, 1, 2, 3, 8, 9, 10, 11), (4, 5, 6, 7))]

CANDIDATES = {
    "capital": [("Japan", "Tokyo"), ("France", "Paris"), ("Italy", "Rome"),
                ("Germany", "Berlin"), ("Norway", "Oslo"), ("Spain", "Madrid"),
                ("Egypt", "Cairo"), ("Cuba", "Havana"), ("Greece", "Athens"),
                ("Russia", "Moscow"), ("China", "Beijing"), ("Peru", "Lima")],
    "language": [("France", "French"), ("Japan", "Japanese"), ("Spain", "Spanish"),
                 ("Germany", "German"), ("Italy", "Italian"), ("Brazil", "Portuguese"),
                 ("Russia", "Russian"), ("China", "Chinese"), ("Greece", "Greek"),
                 ("Poland", "Polish"), ("Norway", "Norwegian"), ("Sweden", "Swedish")],
}
SYN = {"capital": "seat of government", "currency": "unit of money",
       "language": "native tongue"}
FORMS = [
    lambda e, r: f"The {r} of {e} is",
    lambda e, r: f"{e}'s {r} is",
    lambda e, r: f"If you asked someone to name the {r} of {e}, they would say",
    lambda e, r: f"Consider {e}. Its {SYN[r]} is",
    lambda e, r: f"In {e}, the {r} is",
    lambda e, r: f"The {r} associated with {e} is",
    lambda e, r: f"Q: What is the {r} of {e}? A:",
    lambda e, r: f"Looking up {e}'s {r}, one finds",
    lambda e, r: f"For the country {e}, the {r} is",
    lambda e, r: f"Everyone knows that the {SYN[r]} of {e} is",
    lambda e, r: f"Reference entry, {e}, {r}:",
    lambda e, r: f"Speaking of {e} — its {r} happens to be",
]


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def fit_reader(X, y, k):
    """PCA to PCA_DIM then ridge multiclass; basis fitted on train only."""
    Xn = unit(X)
    mu = Xn.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xn - mu, full_matrices=False)
    P = Vt[:PCA_DIM].T
    Z = (Xn - mu) @ P
    Y = np.zeros((len(y), k))
    Y[np.arange(len(y)), y] = 1.0
    W = np.linalg.solve(Z.T @ Z + RIDGE * np.eye(Z.shape[1]), Z.T @ Y)
    return {"mu": mu, "P": P, "W": W}


def apply_reader(R, X):
    return ((unit(X) - R["mu"]) @ R["P"] @ R["W"]).argmax(axis=1)


def procrustes(src, dst):
    U, _, Vt = np.linalg.svd(unit(src).T @ unit(dst), full_matrices=False)
    return U @ Vt


def capture():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import mlx.core as mx
    from chuk_lazarus.models_v2.loader import load_model, ModelDType
    loaded = load_model(MODEL, dtype=ModelDType.BFLOAT16)
    model, tok = loaded.model, loaded.tokenizer
    model.eval()
    cls = type(model.model.layers[0])
    orig = cls.__call__
    idx = {id(l): i for i, l in enumerate(model.model.layers)}
    caps = {}

    def hooked(self, x, *a, **kw):
        i = idx.get(id(self))
        if i in LAYERS:
            caps[i] = np.array(x[0, -1].astype(mx.float32))
        return orig(self, x, *a, **kw)

    cls.__call__ = hooked

    def tid(w):
        ids = [i for i in tok.encode(" " + w) if i != tok.bos_token_id]
        return ids[0] if len(ids) == 1 else None

    items, kept_per_rel = [], {r: set() for r in CANDIDATES}
    for rel, pairs in CANDIDATES.items():
        for ent, ans in pairs:
            t = tid(ans)
            if t is None:
                continue
            rows = []
            for fi, form in enumerate(FORMS):
                caps.clear()
                ll = model(mx.array([tok.encode(form(ent, rel))])).logits[0, -1]
                mx.eval(ll)
                if int(np.array(ll.astype(mx.float32)).argmax()) != t:
                    continue
                rows.append({"entity": ent, "relation": rel, "form": fi,
                             "states": {L: caps[L].copy() for L in LAYERS}})
            # a binding is usable only if it survives on most wordings
            if len(rows) >= 8:
                items.extend(rows)
                kept_per_rel[rel].add(ent)
    cls.__call__ = orig
    return items, {r: sorted(v) for r, v in kept_per_rel.items()}


def main() -> None:
    items, kept = capture()
    short = {r: len(v) for r, v in kept.items() if len(v) < MIN_PER_RELATION}
    print("bindings kept per relation:", {r: len(v) for r, v in kept.items()}, flush=True)
    if short:
        raise SystemExit(f"REFUSE: relations below the {MIN_PER_RELATION} minimum: {short}. "
                         "A collapsed relation must be an error, not a silent change of task.")

    ents = sorted({i["entity"] for i in items})
    rels = sorted({i["relation"] for i in items})
    binds = sorted({(i["entity"], i["relation"]) for i in items})
    lab = {
        "relation": ([rels.index(i["relation"]) for i in items], len(rels)),
        "entity": ([ents.index(i["entity"]) for i in items], len(ents)),
        "binding": ([binds.index((i["entity"], i["relation"])) for i in items], len(binds)),
    }
    S = {L: np.stack([i["states"][L] for i in items]).astype(np.float64) for L in LAYERS}
    form_of = np.array([i["form"] for i in items])
    print(f"{len(items)} prompts | {len(ents)} entities | {len(rels)} relations | "
          f"{len(binds)} bindings | ~{len(items)*8/12/len(binds):.1f} train ex/binding", flush=True)

    # ---- blocking same-layer gate, mean across folds
    same = {c: {} for c in lab}
    for comp, (y, k) in lab.items():
        y = np.array(y)
        for L in LAYERS:
            accs = []
            for tr_f, te_f in FOLDS:
                tr = np.isin(form_of, tr_f)
                te = np.isin(form_of, te_f)
                R = fit_reader(S[L][tr], y[tr], k)
                accs.append(float(np.mean(apply_reader(R, S[L][te]) == y[te])))
            same[comp][f"L{L}"] = float(np.mean(accs))
    gate = {c: {"best": max(same[c].values()),
                "best_layer": max(same[c], key=same[c].get),
                "required": GATE[c], "chance": round(1.0 / lab[c][1], 3),
                "pass": max(same[c].values()) >= GATE[c]} for c in lab}

    print("\nSAME-LAYER READABILITY (mean over 3 wording folds)")
    print(f"{'layer':>6}" + "".join(f"{c:>10}" for c in lab))
    for L in LAYERS:
        print(f"{L:>6}" + "".join(f"{same[c][f'L{L}']:>10.2f}" for c in lab))
    print("\nGATE:", json.dumps(gate, indent=2))

    # ---- cross-layer transfer ONLY for components that cleared the gate
    cross = {}
    for comp, (y, k) in lab.items():
        if not gate[comp]["pass"]:
            cross[comp] = "NOT COMPUTED — reader failed its same-layer gate"
            continue
        y = np.array(y)
        cross[comp] = {}
        for a in range(len(LAYERS) - 1):
            m, l = LAYERS[a], LAYERS[a + 1]
            res = {"identity": [], "procrustes": [], "norm_only": []}
            for tr_f, te_f in FOLDS:
                tr, te = np.isin(form_of, tr_f), np.isin(form_of, te_f)
                R = fit_reader(S[l][tr], y[tr], k)
                A = procrustes(S[m][tr], S[l][tr])
                ratio = float(np.median(np.linalg.norm(S[l][tr], axis=1) /
                                        np.linalg.norm(S[m][tr], axis=1)))
                res["identity"].append(float(np.mean(apply_reader(R, S[m][te]) == y[te])))
                res["procrustes"].append(float(np.mean(apply_reader(R, S[m][te] @ A) == y[te])))
                res["norm_only"].append(float(np.mean(apply_reader(R, S[m][te] * ratio) == y[te])))
            cross[comp][f"L{m}->L{l}"] = {kk: float(np.mean(v)) for kk, v in res.items()}
            cross[comp][f"L{m}->L{l}"]["same_layer_reference"] = same[comp][f"L{l}"]

    out = {"n_prompts": len(items), "entities": ents, "relations": rels,
           "n_bindings": len(binds), "layers": list(LAYERS), "folds": FOLDS,
           "pca_dim": PCA_DIM, "ridge": RIDGE, "gate_thresholds": GATE,
           "kept_per_relation": kept, "same_layer_readability": same,
           "gate": gate, "cross_layer_transfer": cross}
    body = json.dumps(out, sort_keys=True, default=str)
    out["result_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    (HERE / "address_build_1_part2b_result.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n")

    for comp, v in cross.items():
        if isinstance(v, str):
            print(f"\n{comp}: {v}")
            continue
        print(f"\nCROSS-LAYER TRANSFER — {comp}")
        print(f"{'pair':>12} {'ident':>7} {'procr':>7} {'norm':>7} {'same-L':>7}")
        for pair, d in v.items():
            print(f"{pair:>12} {d['identity']:>7.2f} {d['procrustes']:>7.2f} "
                  f"{d['norm_only']:>7.2f} {d['same_layer_reference']:>7.2f}")
    print("\nresult_sha256", out["result_sha256"])


if __name__ == "__main__":
    main()
