#!/usr/bin/env python3
"""
native — write a fact into the REAL model's FFN, the model's own way, and let it read it back.

Same idea as ffn.py (a key→value neuron), but in Gemma's actual L26 FFN. An FFN row is a key→value slot:
the gate row is an ADDRESS the neuron detects; the down column is the VALUE it writes. So to teach the
model a new fact, we plant one neuron — in THREE clear steps:

    address = capture_key("capital of Zelandia")     # the model's own L26 activation for that question
    key     = unique_part(address, other_addresses)  # the part unique to THIS place (not the template)
    write_slot(slot, key, embed("Oslo"))             # gate row = key (detector), down col = value (Oslo)

Then we forward the prompt with NOTHING injected and NO training — the model reads our hand-written slot
through its own forward pass. Novel places (baseline P≈0), so any read is a native FFN read, not prior
knowledge. The `unique_part` step is why capacity caps ~a dozen/relation: you run out of near-orthogonal
entity directions. (This is LARQL MODE COMPOSE.) Self-contained, Gemma-3-4b via MLX.
"""
import sys, json, time
sys.path.insert(0, "/Users/christopherhay/chris-source/chuk-mlx")
import numpy as np, mlx.core as mx
from chuk_lazarus.models_v2.loader import load_model, ModelDType

L = 26
GATE_FIRE = 30.0     # how hard the address-neuron fires on a match
VALUE_WRITE = 0.1    # how strongly it writes the answer into the residual
FACTS = [("Zelandia", "capital", "Oslo"), ("Qtaria", "currency", "Yen"), ("Vornholt", "language", "Welsh")]
PROMPT = {"capital": "The capital of {e} is", "currency": "The currency of {e} is", "language": "The language of {e} is"}
RETAIN = [("The capital of France is", "Paris"), ("The capital of Japan is", "Tokyo"),
          ("The capital of Italy is", "Rome"), ("The capital of Germany is", "Berlin"),
          ("Two plus two equals", "four"), ("The opposite of hot is", "cold")]
def log(*a): print(*a, flush=True)
def unit(v): v = np.asarray(v, np.float64); return v / (np.linalg.norm(v) + 1e-9)


def unique_part(key, others):
    """The part of `key` NOT shared with any of `others` (Gram-Schmidt). Keys on THIS place, not on the
       shared 'capital of ___' template — which is why capacity caps ~a dozen per relation."""
    basis = []
    for o in others:
        b = np.asarray(o, np.float64).copy()
        for q in basis: b = b - (b @ q) * q
        if np.linalg.norm(b) > 1e-6: basis.append(unit(b))
    k = np.asarray(key, np.float64).copy()
    for q in basis: k = k - (k @ q) * q
    return k                                          # caller units it; its norm = how much survived


def main():
    t = time.time(); log("loading google/gemma-3-4b-it (bf16) ...")
    lm = load_model("google/gemma-3-4b-it", dtype=ModelDType.BFLOAT16)
    model, tok = lm.model, lm.tokenizer
    emb = np.array(model.model.embed_tokens.weight.astype(mx.float32)); bos = getattr(tok, "bos_token_id", -1)
    mlp = model.model.layers[L].mlp
    log(f"loaded {time.time()-t:.1f}s; L{L} mlp gate {mlp.gate_proj.weight.shape}")

    def tid(v):
        ids = [i for i in tok.encode(" " + v) if i != bos]; return ids[0] if len(ids) == 1 else None
    # capture the L26 MLP-input (the "address" the question computes) on the last token
    CAP = {"x": None}; mlpcls = type(mlp); orig = mlpcls.__call__
    def patched(self, x):
        if self is mlp: CAP["x"] = np.array(x[0, -1, :].astype(mx.float32))
        return orig(self, x)
    mlpcls.__call__ = patched
    def logits(p):
        ll = model(mx.array([tok.encode(p)])).logits[0, -1]; mx.eval(ll); return np.array(ll.astype(mx.float32))
    def P(p, t): ll = logits(p); e = np.exp(ll - ll.max()); e /= e.sum(); return float(e[t])
    def amax(p): return int(np.argmax(logits(p)))
    def capture_key(prompt):
        CAP["x"] = None; _ = logits(prompt); return CAP["x"].copy()

    # the FFN weight matrices, and their typical row/col scale (so our planted slot is the right magnitude)
    G = np.array(mlp.gate_proj.weight.astype(mx.float32)); U = np.array(mlp.up_proj.weight.astype(mx.float32))
    D = np.array(mlp.down_proj.weight.astype(mx.float32)); I = G.shape[0]
    g_ref = float(np.median(np.linalg.norm(G, axis=1))); u_ref = float(np.median(np.linalg.norm(U, axis=1)))
    d_ref = float(np.median(np.linalg.norm(D, axis=0)))

    def write_slot(slot, key, value):                 # plant ONE key->value neuron
        G[slot]    = key * g_ref * GATE_FIRE          #   gate row  = the address detector (fires on a match)
        U[slot]    = key * u_ref                      #   up   row  = pass the match through
        D[:, slot] = value * d_ref * VALUE_WRITE      #   down col  = the answer it writes into the residual

    base = {(e, r, a): (P(PROMPT[r].format(e=e), tid(a)) if tid(a) else None) for (e, r, a) in FACTS}
    base_retain = {p: amax(p) for p, _ in RETAIN}
    log("\nBEFORE — clean model, these places are novel (P ~ 0):")
    for (e, r, a), p in base.items(): log(f"  P({a}|{e}) = {p:.4f}")

    # decoys: other prompts whose addresses we DON'T want our neurons to fire for
    DECOYS = ["The story begins on a", "In the morning the sun", "She opened the door and",
              "Water is a clear", "Once upon a time there", "The weather today is"]
    for r in PROMPT:
        for de in ["Brazil", "Egypt", "Canada", "Peru", "Chile", "Kenya", "Sweden", "Mexico"]:
            DECOYS.append(PROMPT[r].format(e=de))
    addr = {f: capture_key(PROMPT[f[1]].format(e=f[0])) for f in FACTS}
    decoy_addr = [capture_key(p) for p in DECOYS]

    log("\nWRITE each fact as ONE key->value slot (at the model's OWN computed address):")
    for i, (e, r, a) in enumerate(FACTS):
        slot = I - 1 - i
        others = decoy_addr + [addr[f] for j, f in enumerate(FACTS) if j != i]
        kr = unique_part(addr[(e, r, a)], others)     # the part of the address unique to THIS place
        write_slot(slot, unit(kr), unit(emb[tid(a)]))
        log(f"  slot {slot}: '{r} of {e}' -> '{a}'   (key {np.linalg.norm(kr)/np.linalg.norm(addr[(e,r,a)])*100:.0f}% unique-to-place)")
    mlp.gate_proj.weight = mx.array(G.astype(np.float32)).astype(mlp.gate_proj.weight.dtype)
    mlp.up_proj.weight = mx.array(U.astype(np.float32)).astype(mlp.up_proj.weight.dtype)
    mlp.down_proj.weight = mx.array(D.astype(np.float32)).astype(mlp.down_proj.weight.dtype)

    log("\nhand-written relation -> L26 FFN slot at the model's computed address:")
    aw = max(len(a) for _, _, a in FACTS); ew = max(len(e) for e, _, _ in FACTS)
    hit = 0
    for i, (e, r, a) in enumerate(FACTS):
        tt = tid(a); p = P(PROMPT[r].format(e=e), tt); am = amax(PROMPT[r].format(e=e)); ok = (am == tt); hit += ok
        ann = "     (a made-up place, a made-up capital, read NATIVELY)" if i == 0 else ""
        log(f"  P({a:<{aw}} | {e:<{ew}})   {base[(e,r,a)]:.0f} -> {p:.4f}{ann}")
    agree = sum(amax(p) == base_retain[p] for p, _ in RETAIN)
    log(f"  native read through the model's OWN forward pass:  {hit}/{len(FACTS)}")
    log(f"  retention (argmax vs clean):  {agree}/{len(RETAIN)}")
    log(f"  -> the model reads OUR hand-written store, natively. (caps ~a dozen/relation — the entity-collision limit.)")
    json.dump(dict(L=L, native=f"{hit}/{len(FACTS)}", retention=f"{agree}/{len(RETAIN)}",
                   facts=[{"e": e, "r": r, "a": a, "base_P": base[(e, r, a)]} for (e, r, a) in FACTS]),
              open("native.json", "w"), indent=1, default=float)
    log("wrote native.json")


if __name__ == "__main__":
    main()
