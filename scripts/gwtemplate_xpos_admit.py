"""GW-TEMPLATE-FACTORIAL stage 3: independent admission of cross-position
runner records against the plan and the pre-treatment seal.

It shares no code with the runner. It reads three things:

- the plan;
- the seal records (from the generic seal executable): per sealed run, its
  natural carrier and K/V hashes at a position range, and every step's
  logit hash;
- the runner's records.

A record is admitted only if every check passes. `admit` returns a list of
failure strings; empty means admitted.

Checks per target:

- the record order and IDs follow the plan;
- the natural step hashes equal the seal's for the target run;
- a declared baseline hash, if any, equals the last natural step.

Checks per arm:

- the arm set, spans, start layer and mode equal the plan's;
- the replacement count equals (layers - from + 1) x span length, with one
  firing per replacement;
- the injected-carrier address set is exactly layers from-1 .. L-1 times the
  target span;
- `replace`: each injected carrier equals the SEALED donor carrier at the
  mapped source position;
- the donor's K/V hashes at the source span equal the seal's;
- K/V at the target span below the start layer equal the sealed natural rows;
- `replace` with source == target position: K/V from the start layer equal
  the sealed donor rows;
- steps before the target span equal the natural steps;
- identity arms equal the natural run at every step;
- `norm_match`: the injected norm equals the target's natural norm (relative
  1e-5);
- readout consistency: top-k begins with the greedy token and is descending,
  overlapping candidate logits agree, and the logit hash is the last step's.
"""
from __future__ import annotations

import json
from pathlib import Path

NORM_RTOL = 1e-5


def load_jsonl(p) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def seal_index(seal_rows: list[dict]) -> dict:
    return {r["id"]: r for r in seal_rows}


def _readout_ok(ro: dict, cands: list[int], last_sha: str) -> list[str]:
    f = []
    ids, lg = ro["top_k_ids"], ro["top_k_logits"]
    if not ids or ids[0] != ro["next"]:
        f.append("top-k does not begin with the greedy token")
    if any(lg[i] < lg[i + 1] for i in range(len(lg) - 1)):
        f.append("top-k not descending")
    cl = dict(zip(cands, ro["candidate_logits"]))
    for t, v in zip(ids, lg):
        if t in cl and cl[t] != v:
            f.append(f"candidate logit for {t} disagrees with top-k")
    if ro["next_logit_sha256"] != last_sha:
        f.append("readout hash is not the last step's")
    return f


def admit(plan: dict, seal: dict, rows: list[dict]) -> list[str]:
    fails = []
    L = plan["layers"]
    targets = plan["targets"]
    if [r["id"] for r in rows] != [t["id"] for t in targets]:
        return ["records do not follow the plan's targets in order"]
    for t, r in zip(targets, rows):
        tid = t["id"]
        cands = plan["candidate_sets"][t["candidate_set"]]
        nat = r["natural"]["step_logit_sha256"]
        ts = seal.get(t["run"])
        if ts is None:
            fails.append(f"{tid}: target run not sealed")
            continue
        if nat != ts["step_logit_sha256"]:
            fails.append(f"{tid}: natural steps differ from the seal")
        if t.get("expect_natural_logit_sha256") and nat[-1] != t["expect_natural_logit_sha256"]:
            fails.append(f"{tid}: natural readout differs from the declared baseline")
        fails += [f"{tid} natural: {x}" for x in _readout_ok(r["natural"]["readout"], cands, nat[-1])]
        if set(r["arms"]) != {a["name"] for a in t["arms"]}:
            fails.append(f"{tid}: arm set differs from the plan")
            continue
        for a in t["arms"]:
            w = f"{tid} {a['name']}"
            x = r["arms"][a["name"]]
            frm, src, tgt, mode = a["from_layer"], a["source_span"], a["target_span"], a["mode"]
            if (x["from_layer"], x["source_span"], x["target_span"], x["mode"], x["donor_run"]) != \
                    (frm, src, tgt, mode, a["donor_run"]):
                fails.append(f"{w}: arm parameters differ from the plan")
                continue
            n_rep = (L - frm + 1) * len(tgt)
            if x["replacements"] != n_rep or x["firings"] != n_rep:
                fails.append(f"{w}: {x['replacements']} replacements / {x['firings']} firings, expected {n_rep}")
            want_addr = {f"{l}|{p}" for l in range(frm - 1, L) for p in tgt}
            if set(x["carriers"]) != want_addr:
                fails.append(f"{w}: injected address set differs")
                continue
            ds = seal.get(a["donor_run"])
            if ds is None:
                fails.append(f"{w}: donor run not sealed")
                continue
            for sp, tp in zip(src, tgt):
                for l in range(L):
                    dkey, tkey = f"{l}|{sp}", f"{l}|{tp}"
                    if x["donor_kv_sha256"].get(dkey) != ds["kv_sha256"].get(dkey):
                        fails.append(f"{w}: donor K/V {dkey} differs from the seal")
                    if l < frm and x["kv_sha256"][tkey] != ts["kv_sha256"].get(tkey):
                        fails.append(f"{w}: K/V {tkey} below the start layer is not the sealed natural row")
                    if l >= frm and mode == "replace" and sp == tp and x["kv_sha256"][tkey] != ds["kv_sha256"].get(dkey):
                        fails.append(f"{w}: K/V {tkey} is not the sealed donor row at the same position")
                    if l >= frm - 1 and mode == "replace" and x["carriers"][tkey] != ds["carriers"].get(dkey):
                        fails.append(f"{w}: injected carrier {tkey} is not the sealed donor carrier {dkey}")
            if x["step_logit_sha256"][:tgt[0]] != nat[:tgt[0]]:
                fails.append(f"{w}: a step before the target span changed")
            if a.get("identity") and (x["step_logit_sha256"] != nat or not x.get("identity_bit_exact")):
                fails.append(f"{w}: identity not bit-identical to the natural run")
            if mode == "norm_match":
                for k, nm in x["norms"].items():
                    if abs(nm["injected"] - nm["target"]) > NORM_RTOL * nm["target"]:
                        fails.append(f"{w}: norm at {k} not matched")
            fails += [f"{w}: {y}" for y in _readout_ok(x["readout"], cands, x["step_logit_sha256"][-1])]
    return fails
