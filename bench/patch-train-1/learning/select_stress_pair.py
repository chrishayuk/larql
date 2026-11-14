#!/usr/bin/env python3
"""PATCH-TRAIN-2 interference-stress pair SELECTION (DISCOVERY ONLY; no training, no weight changed, no threshold set).

Predeclared deterministic rule, written here before the selection is run:
  candidates   = tasks_discovery.json entities EXCLUDING trained_entities[0..3] (A, B and the reserved calibration pair);
                 the confirmatory set (tasks.json) is never opened for data.
  address b_e  = the ridge-whitened layer-3 input direction of mech1/rep1 (training template, 256 columns), built for each
                 candidate against all 23 other entities and neutral text.
  eligible pair= two candidates with DIFFERENT assigned answers (so the stress is on addresses, not on a shared answer).
  selected     = the eligible pair with the HIGHEST address cosine cos(b_i, b_j); ties broken by the lower (index_i, index_j)
                 in the order trained_entities[4:], then heldout_entities.
Recorded for every eligible pair (not used for selection): the cross-activations b_i.z_j and b_j.z_i at the entity's last token
against the target activations b_i.z_i and b_j.z_j, as ratios.
    python3 -I select_stress_pair.py
"""
import itertools
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import evaluation as V  # noqa: E402
import feasibility as F  # noqa: E402
import mech1 as M  # noqa: E402
import rep1 as R  # noqa: E402

tasks = V.load_tasks("tasks_discovery.json")
assert tasks["seed"] != V.load_tasks("tasks.json")["seed"]
model, tok = V.load_model(tasks)
t = tasks["tasks"]["A"]
all_ents = t["trained_entities"] + t["heldout_entities"]
candidates = t["trained_entities"][4:] + t["heldout_entities"]
columns = np.sort(np.random.default_rng(F.SEED).choice(2048, M.N_COLUMNS, replace=False))
tpl = t["train_templates"][0]


def last_act(e):
    z, ids = M.capture(model, tok, tpl.format(e=e), columns)
    return z[M.entity_span(tok, tpl, e, ids)[1]]


addr = {e: R.build_b(model, tok, tasks, e, [x for x in all_ents if x != e], columns)[0] for e in candidates}
act = {e: last_act(e) for e in candidates}
pairs = []
for i, j in itertools.combinations(range(len(candidates)), 2):
    ei, ej = candidates[i], candidates[j]
    if t["assignment"][ei] == t["assignment"][ej]:
        continue
    cii, cjj = float(act[ei] @ addr[ei]), float(act[ej] @ addr[ej])
    cij, cji = float(act[ej] @ addr[ei]), float(act[ei] @ addr[ej])
    pairs.append({"i": i, "j": j, "entities": [ei, ej], "answers": [t["assignment"][ei], t["assignment"][ej]],
                  "address_cosine": float(addr[ei] @ addr[ej]), "b_i_dot_z_i": cii, "b_j_dot_z_j": cjj, "b_i_dot_z_j": cij, "b_j_dot_z_i": cji,
                  "cross_over_target": [cij / cii, cji / cjj]})
pairs.sort(key=lambda p: (-p["address_cosine"], p["i"], p["j"]))
cos = np.array([p["address_cosine"] for p in pairs])
out = {"protocol": "PATCH-TRAIN-2 stress-pair selection (DISCOVERY; no training)", "candidates": candidates,
       "excluded": t["trained_entities"][:4], "n_eligible_pairs": len(pairs),
       "rule": "highest address cosine among eligible pairs (different assigned answers); ties by lower index",
       "cosine_distribution": {"min": float(cos.min()), "median": float(np.median(cos)), "p90": float(np.percentile(cos, 90)), "max": float(cos.max())},
       "reference_pair_cosine_Frozoria_Dresalthaia": float(R.build_b(model, tok, tasks, t["trained_entities"][0], [x for x in all_ents if x != t["trained_entities"][0]], columns)[0]
                                                           @ R.build_b(model, tok, tasks, t["trained_entities"][1], [x for x in all_ents if x != t["trained_entities"][1]], columns)[0]),
       "selected": pairs[0], "top10": pairs[:10]}
with open(os.path.join(HERE, "stress_pair_selection.json"), "w") as f:
    json.dump(out, f, indent=1)
    f.write("\n")
print("eligible pairs", len(pairs), "cosine dist", {k: round(v, 3) for k, v in out["cosine_distribution"].items()}, "reference", round(out["reference_pair_cosine_Frozoria_Dresalthaia"], 3))
for p in pairs[:6]:
    print(p["entities"], p["answers"], "cos", round(p["address_cosine"], 3), "cross/target", [round(x, 3) for x in p["cross_over_target"]])
