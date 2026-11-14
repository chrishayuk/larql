#!/usr/bin/env python3
"""PATCH-TRAIN-1 SINGLE-1 phase 2 pre-diagnostic (DISCOVERY; no training, no weight changed): is there entity signal at the
input of each layer's down_proj, at the LAST token and at the ENTITY's last token (a patch acts at every position, so the
entity token is also an intervention site)? Cosine between the target entity's activation and every other entity's, on the
256 footprint columns and on all 2048 intermediate dims.   python3 -I signal.py"""
import json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, torch
import evaluation as V, feasibility as F

LAYERS = (3, 6, 9, 12, 15)
tasks = V.load_tasks("tasks_discovery.json")
model, tok = V.load_model(tasks)
t = tasks["tasks"]["A"]
entity = t["trained_entities"][0]
ents = [entity] + [e for e in t["trained_entities"] + t["heldout_entities"] if e != entity]
tpl = t["train_templates"][0]
prefix_len = {e: len(tok(tpl.split("{e}")[0] + e, return_tensors="pt")["input_ids"][0]) - 1 for e in ents}
columns = np.sort(np.random.default_rng(F.SEED).choice(2048, 256, replace=False))
out = {"protocol": "PATCH-TRAIN-1 SINGLE-1 signal pre-diagnostic (DISCOVERY; no training)", "template": tpl, "entity": entity, "layers": {}}
for L in LAYERS:
    got = {}
    mod = model.model.layers[L].mlp.down_proj
    h = mod.register_forward_pre_hook(lambda m, a: got.__setitem__("z", a[0][0].detach().double().numpy()))
    rows = {"last": [], "entity_tok": []}
    with torch.no_grad():
        for e in ents:
            model(input_ids=tok(tpl.format(e=e), return_tensors="pt")["input_ids"])
            z = got["z"]
            rows["last"].append(z[-1]); rows["entity_tok"].append(z[prefix_len[e]])
    h.remove()
    res = {}
    for site, arr in rows.items():
        A = np.stack(arr)
        for name, M in (("footprint256", A[:, columns]), ("all2048", A)):
            Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
            c = (Mn[1:] @ Mn[0])
            res[f"{site}/{name}"] = {"cos_target_vs_others_mean": float(c.mean()), "min": float(c.min()), "max": float(c.max())}
    out["layers"][str(L)] = res
    print(L, {k: round(v["cos_target_vs_others_mean"], 4) for k, v in res.items()}, flush=True)
with open(os.path.join(HERE, "signal_discovery.json"), "w") as f:
    json.dump(out, f, indent=1); f.write("\n")
