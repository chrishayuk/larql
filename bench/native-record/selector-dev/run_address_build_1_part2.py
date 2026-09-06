#!/usr/bin/env python3
"""ADDRESS-BUILD-1 Part 2 — does the representation change coordinates while
each component is being committed?

Gate conditions 4-5. Scored SEPARATELY for relation, entity and full binding,
because the causal half showed those components become decisive at very
different depths and must not be pooled.

Material: the same 48 native-fact prompts and the same seven depths as the
causal half (L8..L30), re-captured here so Part 2 uses exactly the states the
transplant result was measured on. Cross-fitting is by WORDING: readers and
transforms are fitted on forms {0,1} and scored on held-out forms {2,3}, which
is the axis the causal result is about.

Arms for cross-layer transfer R_l(h_m):
  identity      naive transfer, no transform
  procrustes    fitted orthogonal A_{m->l}
  norm_only     scalar norm-ratio transform (the trivial explanation)
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = ("/Users/christopherhay/.cache/huggingface/hub/models--google--gemma-3-4b-it"
         "/snapshots/093f9f388b31de276ce2de164bdc2081324b9767")
LAYERS = (8, 12, 16, 20, 24, 28, 30)
RIDGE = 1.0

import importlib.util
spec = importlib.util.spec_from_file_location("build1", HERE / "run_address_build_1.py")
build1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build1)
FACTS, FORMS = build1.FACTS, build1.FORMS


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def ridge_fit(X, y, k):
    Y = np.zeros((len(y), k))
    Y[np.arange(len(y)), y] = 1.0
    Xn = unit(X)
    G = Xn.T @ Xn + RIDGE * np.eye(Xn.shape[1])
    return np.linalg.solve(G, Xn.T @ Y)


def acc(W, X, y):
    return float(np.mean((unit(X) @ W).argmax(axis=1) == np.asarray(y)))


def procrustes(src, dst):
    """Orthogonal A minimising ||src @ A - dst||."""
    U, _, Vt = np.linalg.svd(unit(src).T @ unit(dst), full_matrices=False)
    return U @ Vt


def capture_all():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import mlx.core as mx
    from chuk_lazarus.models_v2.loader import load_model, ModelDType
    loaded = load_model(MODEL, dtype=ModelDType.BFLOAT16)
    model, tok = loaded.model, loaded.tokenizer
    model.eval()
    layer_cls = type(model.model.layers[0])
    orig = layer_cls.__call__
    index = {id(l): i for i, l in enumerate(model.model.layers)}
    caps = {}

    def hooked(self, x, *a, **kw):
        i = index.get(id(self))
        if i in LAYERS:
            caps[i] = np.array(x[0, -1].astype(mx.float32))
        return orig(self, x, *a, **kw)

    layer_cls.__call__ = hooked
    items = []
    for e, r, ans in FACTS:
        ids = [i for i in tok.encode(" " + ans) if i != tok.bos_token_id]
        if len(ids) != 1:
            continue
        for fi, form in enumerate(FORMS):
            caps.clear()
            ll = model(mx.array([tok.encode(form(e, r))])).logits[0, -1]
            mx.eval(ll)
            if int(np.array(ll.astype(mx.float32)).argmax()) != ids[0]:
                continue
            items.append({"entity": e, "relation": r, "form": fi,
                          "states": {L: caps[L].copy() for L in LAYERS}})
    layer_cls.__call__ = orig
    return items


def main() -> None:
    items = capture_all()
    ents = sorted({i["entity"] for i in items})
    rels = sorted({i["relation"] for i in items})
    binds = sorted({(i["entity"], i["relation"]) for i in items})
    print(f"{len(items)} prompts | {len(ents)} entities | {len(rels)} relations | "
          f"{len(binds)} bindings", flush=True)

    tr = [i for i, it in enumerate(items) if it["form"] in (0, 1)]
    te = [i for i, it in enumerate(items) if it["form"] in (2, 3)]
    labels = {
        "relation": ([rels.index(items[i]["relation"]) for i in tr],
                     [rels.index(items[i]["relation"]) for i in te], len(rels)),
        "entity": ([ents.index(items[i]["entity"]) for i in tr],
                   [ents.index(items[i]["entity"]) for i in te], len(ents)),
        "binding": ([binds.index((items[i]["entity"], items[i]["relation"])) for i in tr],
                    [binds.index((items[i]["entity"], items[i]["relation"])) for i in te],
                    len(binds)),
    }
    S = {L: np.stack([it["states"][L] for it in items]).astype(np.float64) for L in LAYERS}

    same, cross = {}, {}
    for comp, (ytr, yte, k) in labels.items():
        same[comp] = {}
        for L in LAYERS:
            W = ridge_fit(S[L][tr], ytr, k)
            same[comp][f"L{L}"] = acc(W, S[L][te], yte)
        cross[comp] = {}
        for a in range(len(LAYERS) - 1):
            l, m = LAYERS[a + 1], LAYERS[a]          # reader at l, states from m
            W = ridge_fit(S[l][tr], ytr, k)
            A = procrustes(S[m][tr], S[l][tr])
            ratio = float(np.median(np.linalg.norm(S[l][tr], axis=1) /
                                    np.linalg.norm(S[m][tr], axis=1)))
            cross[comp][f"L{m}->L{l}"] = {
                "identity": acc(W, S[m][te], yte),
                "procrustes": acc(W, S[m][te] @ A, yte),
                "norm_only": acc(W, S[m][te] * ratio, yte),
                "same_layer_reference": same[comp][f"L{l}"],
            }

    out = {"n_prompts": len(items), "entities": ents, "relations": rels,
           "n_bindings": len(binds), "layers": list(LAYERS), "ridge": RIDGE,
           "cross_fit": "readers and transforms fitted on forms {0,1}, scored on forms {2,3}",
           "same_layer_readability": same, "cross_layer_transfer": cross}
    body = json.dumps(out, sort_keys=True)
    out["result_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    (HERE / "address_build_1_part2_result.json").write_text(json.dumps(out, indent=2) + "\n")

    print("\nSAME-LAYER READABILITY (held-out wordings)")
    print(f"{'layer':>6} " + "".join(f"{c:>10}" for c in labels))
    for L in LAYERS:
        print(f"{L:>6} " + "".join(f"{same[c][f'L{L}']:>10.2f}" for c in labels))

    print("\nCROSS-LAYER TRANSFER  R_l(h_m)")
    print(f"{'pair':>12} {'comp':>9} {'ident':>7} {'procr':>7} {'norm':>7} {'same-L':>7}")
    for comp in labels:
        for pair, v in cross[comp].items():
            print(f"{pair:>12} {comp:>9} {v['identity']:>7.2f} {v['procrustes']:>7.2f} "
                  f"{v['norm_only']:>7.2f} {v['same_layer_reference']:>7.2f}")
    print("\nresult_sha256", out["result_sha256"])


if __name__ == "__main__":
    main()
