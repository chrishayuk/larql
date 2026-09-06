#!/usr/bin/env python3
"""ADDRESS-BUILD-1 feasibility probe: decoder-layer input hook + splice floor + timing.

Runs nothing scientific. Establishes three things before the real run is
committed to:
  1. can we capture and inject the residual ENTERING a decoder layer
  2. does a self-transplant reproduce the unmodified forward bit-identically
     (the map-2b splice floor, mandatory before any arm is scored)
  3. how long one forward actually takes, so the full run can be priced
"""
from __future__ import annotations

import inspect
import os
import time

import numpy as np

MODEL = ("/Users/christopherhay/.cache/huggingface/hub/models--google--gemma-3-4b-it"
         "/snapshots/093f9f388b31de276ce2de164bdc2081324b9767")


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import mlx.core as mx
    from chuk_lazarus.models_v2.loader import load_model, ModelDType

    t0 = time.time()
    loaded = load_model(MODEL, dtype=ModelDType.BFLOAT16)
    model, tok = loaded.model, loaded.tokenizer
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s; {len(model.model.layers)} layers")

    layer_cls = type(model.model.layers[0])
    sig = inspect.signature(layer_cls.__call__)
    print(f"decoder layer class: {layer_cls.__name__}")
    print(f"__call__ signature:  {sig}")

    orig_call = layer_cls.__call__
    state = {"capture_at": None, "inject_at": None, "donor": None, "captured": {}}
    index = {id(l): i for i, l in enumerate(model.model.layers)}

    def hooked(self, x, *a, **kw):
        i = index.get(id(self))
        if i is not None:
            if state["capture_at"] is not None and i == state["capture_at"]:
                state["captured"][i] = np.array(x[0, -1].astype(mx.float32))
            if state["inject_at"] is not None and i == state["inject_at"]:
                d = mx.array(state["donor"]).astype(x.dtype)
                x = mx.concatenate([x[:, :-1, :], d.reshape(1, 1, -1)], axis=1)
        return orig_call(self, x, *a, **kw)

    layer_cls.__call__ = hooked

    def forward(prompt):
        ll = model(mx.array([tok.encode(prompt)])).logits[0, -1]
        mx.eval(ll)
        return np.array(ll.astype(mx.float32))

    P = "The capital of Japan is"
    LAYER = 20

    t0 = time.time(); base = forward(P); t_base = time.time() - t0
    state["capture_at"] = LAYER
    _ = forward(P)
    cap = state["captured"][LAYER]
    state["capture_at"] = None
    print(f"\ncaptured i_{LAYER}: shape {cap.shape}, norm {np.linalg.norm(cap):.3f}")

    state["inject_at"], state["donor"] = LAYER, cap
    t0 = time.time(); same = forward(P); t_inj = time.time() - t0
    state["inject_at"] = None

    delta = float(np.max(np.abs(same - base)))
    print(f"SPLICE FLOOR self-transplant max|logit delta| = {delta:.3e}"
          f"  -> {'PASS (bit-identical)' if delta == 0.0 else 'NOT bit-identical'}")

    top = int(base.argmax())
    print(f"baseline top-1 token: {tok.decode([top])!r}")

    n = 8
    t0 = time.time()
    for _ in range(n):
        forward(P)
    per = (time.time() - t0) / n
    print(f"\ntiming: baseline {t_base:.3f}s, injected {t_inj:.3f}s, "
          f"steady-state {per:.3f}s/forward")
    for label, forwards in [("captures (16 bindings x 4 forms x 7 layers)", 16 * 4),
                            ("transplants (32 recipients x 7 layers x 4 arms)", 32 * 7 * 4)]:
        print(f"  {label}: {forwards} forwards -> {forwards*per/60:.1f} min")
    layer_cls.__call__ = orig_call


if __name__ == "__main__":
    main()
