"""GW-TEMPLATE-FACTORIAL Test 3 Stage 2 analysis (C1/C2 development evaluation).

This is the executable form of `docs/gw-template-factorial-test3-stage2-analysis.md`.
It was written before any Stage 2 arm executed, and both files are hashed into
the Stage 2 binding.

It runs no model. Its inputs are:
- the Stage 2 derived plans (one per shard);
- the admitted runner records;
- the baseline plan (accepted first tokens);
- the Test 2 plan (C0 donors).

Answer classes come from the 59 development countries only; the holdout is
never read.

    gwtemplate_test3_stage2_analyze.py STAGE2_DIR REPORT.json
"""
from __future__ import annotations

import collections
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

BOOT = 10_000
SEED = 20261118
RELATIONS = ("capital", "currency_code_iso4217")
POOLED = "P2+P3"
SWITCH_KINDS = ("S", "C0", "C1", "C1A", "C1B", "C2")
CANDIDATES = ("C1", "C2")
GATE_A_LOWER = 0.80
GATE_A_PHRASING = 0.60
GATE_B2 = 0.5
GATE_B3_MEAN = 0.5
GATE_B3_MAX = 1.0
GATE_OTHER_TOP1 = 0.05
B3_MAX_SUPPORT = 10
FUNCTION = {"the", "a", "an", "of", "in", "is", "and", "to", "for", "it", "its", "this", "that", "known",
            "called", "officially", "not", "also", "currently", "located", "named", "which", "one", "",
            "city", "capital", "code", "currency", "iso"}
CODE = re.compile(r"^[A-Z]{2,4}$")


def other(r: str) -> str:
    return RELATIONS[1 - RELATIONS.index(r)]


def lp(ro: dict, pos: dict, tok: int) -> float:
    return ro["candidate_logits"][pos[tok]] - ro["log_normalizer"]


def dist(ro: dict) -> list[float]:
    xs = ro["candidate_logits"]
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def kl(q: list[float], p: list[float]) -> float:
    return sum(a * (math.log(a) - math.log(b)) for a, b in zip(q, p) if a > 0)


def decoded_category(s: str) -> str:
    t = s.strip()
    if CODE.match(t):
        return "code_like"
    if not t or not any(ch.isalnum() for ch in t) or t.lower() in FUNCTION:
        return "function_or_punct"
    return "word"


class Classes:
    """Single-token answer classes of the development population (section 3)."""

    def __init__(self, bplan: dict, dev: list[str], cands: list[int]):
        self.dev = sorted(dev)
        devset = set(dev)
        self.tok = {}
        for c in bplan["cells"]:
            if c["country"] in devset and c["relation_name"] in RELATIONS and c["phrasing"] in ("P2", "P3"):
                a = c["accepted_first_tokens"]
                if len(a) != 1:
                    raise ValueError(f"{c['id']}: accepted set is not a single token")
                self.tok[(c["relation_name"], c["phrasing"], c["country"])] = a[0]
        for rel in RELATIONS:
            for ph in ("P2", "P3"):
                if any((rel, ph, z) not in self.tok for z in dev):
                    raise ValueError(f"missing accepted tokens for {rel} {ph}")
        self.pos = {t: i for i, t in enumerate(cands)}
        if len(self.pos) != len(cands):
            raise ValueError("candidate set has repeated tokens")
        missing = {t for t in self.tok.values() if t not in self.pos}
        if missing:
            raise ValueError(f"class tokens outside the candidate set: {sorted(missing)}")

    def t(self, rel, ph, z):
        return self.tok[(rel, ph, z)]

    def mult(self, rel, ph, tok, exclude=()):
        return sum(1 for z in self.dev if z not in exclude and self.tok[(rel, ph, z)] == tok)

    def other_classes(self, rel, ph, Y, r_Y, rp_Y) -> list[int]:
        return sorted({self.tok[(rel, ph, z)] for z in self.dev if z != Y} - {r_Y, rp_Y})


def per_arm(targets: list[dict], rows: list[dict], cl: Classes, fit: set[str], donors: dict, decode) -> tuple[list, list]:
    """targets: plan targets (in order); rows: runner records (same order). Returns (arm records, B2 cell records)."""
    if [r["id"] for r in rows] != [t["id"] for t in targets]:
        raise ValueError("records do not follow the plan")
    natural = {t["run"]: r["natural"]["readout"] for t, r in zip(targets, rows)}
    arms, b2 = [], []
    for t, row in zip(targets, rows):
        rel, ph, Y = t["run"].split("|")
        rp = other(rel)
        pop = "fit" if Y in fit else "validation"
        rY, rpY = cl.t(rel, ph, Y), cl.t(rp, ph, Y)
        if rY == rpY:
            raise ValueError(f"{t['run']}: r and r' share a token")
        nat = natural[t["run"]]
        sep = natural[f"{rp}|{ph}|{Y}"]
        o_rp = cl.other_classes(rp, ph, Y, rY, rpY)
        o_r = cl.other_classes(rel, ph, Y, rY, rpY)
        s_ro = {}
        for a in t["arms"]:
            kind, tag, layer = a["name"].split("-")
            if kind == "S":
                s_ro[layer] = row["arms"][a["name"]]["readout"]
        cell_ro = {}
        for a in t["arms"]:
            kind, tag, layer = a["name"].split("-")
            if kind == "ID":
                continue
            L = int(layer[1:])
            ro = row["arms"][a["name"]]["readout"]
            nxt = ro["next"]
            if kind == "C1E":
                ref, cls_main = nat, o_r
                order = (("r(Y)", {rY}), ("r'(Y)", {rpY}), ("other_country_r", set(o_r)),
                         ("other_country_r'", set(o_rp)))
            else:
                ref, cls_main = sep, o_rp
                order = (("r(Y)", {rY}), ("r'(Y)", {rpY}), ("other_country_r'", set(o_rp)),
                         ("other_country_r", set(o_r)))
            outcome = next((n for n, s in order if nxt in s), None) or decoded_category(decode(nxt))
            ex = {c: lp(ro, cl.pos, c) - lp(ref, cl.pos, c) for c in cls_main}
            mrel = rel if kind == "C1E" else rp
            w = {c: cl.mult(mrel, ph, c, exclude=(Y,)) for c in cls_main}
            rec = {"target": Y, "relation": rel, "phrasing": ph, "layer": L, "kind": kind, "tag": tag,
                   "population": pop, "outcome": outcome,
                   "lp_rY": lp(ro, cl.pos, rY), "lp_rpY": lp(ro, cl.pos, rpY),
                   "m_switch": lp(ro, cl.pos, rpY) - lp(ro, cl.pos, rY),
                   "gain_rY": lp(ro, cl.pos, rY) - lp(nat, cl.pos, rY),
                   "gain_rpY": lp(ro, cl.pos, rpY) - lp(nat, cl.pos, rpY),
                   "kl_ref": kl(dist(ref), dist(ro)),
                   "b3_mean": sum(ex.values()) / len(ex),
                   "b3_wmean": sum(ex[c] * w[c] for c in ex) / sum(w.values()),
                   "b3_excess": {str(c): v for c, v in ex.items()}}
            if kind == "C0":
                X = donors[t["run"]][int(tag[1:]) - 1]
                rX, rpX = cl.t(rel, ph, X), cl.t(rp, ph, X)
                rec["donor"] = X
                rec["donor_answer_top1"] = float(nxt == rpX)
                if rpX != rpY and rX != rY:
                    sro = s_ro[layer]
                    rec["b1"] = (lp(ro, cl.pos, rpX) - lp(nat, cl.pos, rpX)) - (lp(sro, cl.pos, rpX) - lp(nat, cl.pos, rpX))
                else:
                    rec["b1"] = None
            arms.append(rec)
            if kind in ("C1A", "C1B"):
                cell_ro[(kind, L)] = ro
        for L in sorted({k[1] for k in cell_ro}):
            ro_a, ro_b = cell_ro.get(("C1A", L)), cell_ro.get(("C1B", L))
            if ro_a is None or ro_b is None:
                raise ValueError(f"{t['run']}: C1A/C1B arm missing at L{L}")
            fa, fb = t["halves"]
            ta = {cl.t(rp, ph, z) for z in fa if z != Y}
            tb = {cl.t(rp, ph, z) for z in fb if z != Y}
            A = sorted(ta - tb - {rY, rpY})
            B = sorted(tb - ta - {rY, rpY})
            rec = {"target": Y, "relation": rel, "phrasing": ph, "layer": L, "population": pop,
                   "n_A": len(A), "n_B": len(B), "M": None}
            if A and B:
                dA = sum(lp(ro_a, cl.pos, c) - lp(ro_b, cl.pos, c) for c in A) / len(A)
                dB = sum(lp(ro_b, cl.pos, c) - lp(ro_a, cl.pos, c) for c in B) / len(B)
                rec.update(dA=dA, dB=dB, M=(dA + dB) / 2)
            b2.append(rec)
    return arms, b2


# ── cluster bootstrap (numpy; one generator, seed SEED, draws in a fixed group order) ──

def _clusters(recs, f):
    u = collections.defaultdict(list)
    for r in recs:
        v = f(r)
        if v is not None:
            u[r["target"]].append(v)
    return u


def _draws(rng, n):
    """BOOT x n matrix of resample counts per cluster."""
    idx = rng.integers(0, n, size=(BOOT, n))
    w = np.zeros((BOOT, n))
    np.add.at(w, (np.arange(BOOT)[:, None], idx), 1)
    return w


def _pct(vals):
    vals = np.sort(np.asarray(vals))
    return [float(vals[int(0.025 * len(vals))]), float(vals[int(0.975 * len(vals)) - 1])]


def boot_mean(units: dict, rng) -> dict:
    keys = sorted(units)
    flat = [v for k in keys for v in units[k]]
    if not flat:
        return {"n": 0, "mean": None, "ci95": None}
    sums = np.array([sum(units[k]) for k in keys])
    cnts = np.array([len(units[k]) for k in keys], dtype=float)
    w = _draws(rng, len(keys))
    return {"n": len(flat), "clusters": len(keys), "mean": sum(flat) / len(flat), "ci95": _pct((w @ sums) / (w @ cnts))}


def _b3_tables(recs):
    keys = sorted({r["target"] for r in recs})
    classes = sorted({c for r in recs for c in r["b3_excess"]}, key=int)
    ki, ci = {k: i for i, k in enumerate(keys)}, {c: j for j, c in enumerate(classes)}
    S = np.zeros((len(keys), len(classes)))
    N = np.zeros((len(keys), len(classes)))
    for r in recs:
        for c, v in r["b3_excess"].items():
            S[ki[r["target"]], ci[c]] += v
            N[ki[r["target"]], ci[c]] += 1
    return keys, classes, S, N


def _b3_max_from(w, S, N):
    """w: (B, T) cluster weights. Returns per-row max of per-class means over classes with support >= floor."""
    sums, cnts = w @ S, w @ N
    support = (w > 0).astype(float) @ (N > 0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where((support >= B3_MAX_SUPPORT) & (cnts > 0), sums / np.where(cnts > 0, cnts, 1), -np.inf)
    return means


def b3_max(recs: list[dict]):
    keys, classes, S, N = _b3_tables(recs)
    if not classes:
        return None, None, 0
    m = _b3_max_from(np.ones((1, len(keys))), S, N)[0]
    if not np.isfinite(m).any():
        return None, None, 0
    j = int(np.argmax(m))
    return float(m[j]), classes[j], int((N[:, j] > 0).sum())


def boot_b3_max(recs: list[dict], rng) -> dict:
    point, cls, support = b3_max(recs)
    if point is None:
        return {"max": None, "class": None, "support": 0, "ci95": None, "classes_eligible": 0}
    keys, classes, S, N = _b3_tables(recs)
    m = _b3_max_from(_draws(rng, len(keys)), S, N).max(axis=1)
    m = m[np.isfinite(m)]
    return {"max": point, "class": int(cls), "support": support, "ci95": _pct(m) if len(m) else None,
            "classes_eligible": int(((N > 0).sum(axis=0) >= B3_MAX_SUPPORT).sum())}


OUTCOMES = ("r(Y)", "r'(Y)", "other_country_r'", "other_country_r", "code_like", "function_or_punct", "word")


def summarise(arms: list[dict], b2: list[dict], cl: Classes) -> dict:
    rng = np.random.default_rng(SEED)
    groups = {}
    for kind in SWITCH_KINDS + ("C1E",):
        for rel in RELATIONS:
            for ph in ("P2", "P3", POOLED):
                for L in (1, 16):
                    for pop in ("validation", "fit"):
                        G = [a for a in arms if a["kind"] == kind and a["relation"] == rel and a["layer"] == L
                             and a["population"] == pop and (ph == POOLED or a["phrasing"] == ph)]
                        if not G:
                            continue
                        g = {"n": len(G), "shares": {o: boot_mean(_clusters(G, lambda a, o=o: float(a["outcome"] == o)), rng)
                                                     for o in OUTCOMES}}
                        for f in ("m_switch", "gain_rY", "gain_rpY", "kl_ref", "b3_mean", "b3_wmean"):
                            g[f] = boot_mean(_clusters(G, lambda a, f=f: a[f]), rng)
                        g["b3_max"] = boot_b3_max(G, rng)
                        if kind == "C0":
                            g["b1"] = boot_mean(_clusters(G, lambda a: a["b1"]), rng)
                            g["b1_excluded_overlap"] = sum(a["b1"] is None for a in G)
                            g["donor_answer_top1"] = boot_mean(_clusters(G, lambda a: a["donor_answer_top1"]), rng)
                        if kind == "C1":
                            B = [b for b in b2 if b["relation"] == rel and b["layer"] == L and b["population"] == pop
                                 and (ph == POOLED or b["phrasing"] == ph)]
                            g["b2_M"] = boot_mean(_clusters(B, lambda b: b["M"]), rng)
                            g["b2_dA"] = boot_mean(_clusters(B, lambda b: b.get("dA")), rng)
                            g["b2_dB"] = boot_mean(_clusters(B, lambda b: b.get("dB")), rng)
                            g["b2_cells_missing"] = sum(b["M"] is None for b in B)
                        groups[f"{kind}|{rel}|{ph}|L{L}|{pop}"] = g
    return groups


def gates(groups: dict) -> dict:
    out = {}
    for kind in CANDIDATES:
        for rel in RELATIONS:
            k = lambda ph: f"{kind}|{rel}|{ph}|L1|validation"  # noqa: E731
            g, g2, g3 = groups[k(POOLED)], groups[k("P2")], groups[k("P3")]
            sw = g["shares"]["r'(Y)"]
            A = {"pooled_share": sw["mean"], "pooled_ci95": sw["ci95"], "P2": g2["shares"]["r'(Y)"]["mean"],
                 "P3": g3["shares"]["r'(Y)"]["mean"]}
            A["pass"] = (sw["ci95"][0] >= GATE_A_LOWER and A["P2"] >= GATE_A_PHRASING and A["P3"] >= GATE_A_PHRASING)
            B = {"b3_mean": g["b3_mean"]["mean"], "b3_mean_pass": g["b3_mean"]["mean"] <= GATE_B3_MEAN,
                 "b3_max": g["b3_max"]["max"], "b3_max_class": g["b3_max"]["class"],
                 "b3_max_pass": g["b3_max"]["max"] is not None and g["b3_max"]["max"] <= GATE_B3_MAX,
                 "other_top1": g["shares"]["other_country_r'"]["mean"],
                 "other_top1_pass": g["shares"]["other_country_r'"]["mean"] <= GATE_OTHER_TOP1}
            if kind == "C1":
                ci = g["b2_M"]["ci95"]
                B["b2_M"], B["b2_ci95"] = g["b2_M"]["mean"], ci
                B["b2_pass"] = ci is not None and -GATE_B2 <= ci[0] and ci[1] <= GATE_B2
            else:
                B["b2_pass"] = None
                B["b2_note"] = "not measured: C2 halves are not in the approved Stage 2 inventory"
            failed = [n for n in ("b2_pass", "b3_mean_pass", "b3_max_pass", "other_top1_pass") if B[n] is False]
            if not A["pass"]:
                verdict = "no switch"
            elif failed:
                verdict = "axis B fail: " + ", ".join(failed)
            elif B["b2_pass"] is None:
                verdict = "incomplete: B2 not measured"
            else:
                verdict = "pass"
            out[f"{kind}|{rel}"] = {"A": A, "B": B, "verdict": verdict}
    return out


def parameter_counts(targets: list[dict], hidden: int, layers: int) -> dict:
    out = {}
    for t in targets:
        rel, ph, _ = t["run"].split("|")
        for a in t["arms"]:
            kind, _, layer = a["name"].split("-")
            if kind in ("C1", "C1A", "C1B", "C2", "C1E"):
                n = (layers - (a["from_layer"] - 1)) * len(a["target_span"]) * hidden
                out[f"{kind}|{rel}|{ph}|{layer}"] = n
    return out


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    import gwtemplate_baseline_plan as bp
    d, out = Path(argv[0]), Path(argv[1])
    b = json.loads((d / "binding.json").read_text())
    bplan = json.loads((ROOT / b["files"]["baseline_plan"]["path"]).read_text())
    t2 = json.loads((ROOT / b["files"]["test2_plan"]["path"]).read_text())
    donors = {t["run"]: t["donors"] for t in t2["targets"]}
    targets, rows, hidden, layers = [], [], None, None
    for s in b["shards"]:
        plan = json.loads((d / s["plan"]).read_text())
        targets += plan["targets"]
        rows += [json.loads(x) for x in (d / s["output"]).read_text().splitlines() if x.strip()]
        hidden, layers, cands = plan["vector_store"]["hidden"], plan["layers"], next(iter(plan["candidate_sets"].values()))
    tok = bp.tokenizer()
    cl = Classes(bplan, b["population"]["development"], cands)
    arms, b2 = per_arm(targets, rows, cl, set(b["population"]["fit"]), donors, lambda i: tok.decode([i]))
    groups = summarise(arms, b2, cl)
    rep = {"schema": "larql.gwtemplate.test3-stage2-report.v1",
           "status": "development evaluation; gates on validation targets at L1; fit and L16 descriptive",
           "constants": {"bootstrap": BOOT, "seed": SEED, "gate_A_lower": GATE_A_LOWER, "gate_A_phrasing": GATE_A_PHRASING,
                         "gate_b2": GATE_B2, "gate_b3_mean": GATE_B3_MEAN, "gate_b3_max": GATE_B3_MAX,
                         "gate_other_top1": GATE_OTHER_TOP1, "b3_max_support": B3_MAX_SUPPORT},
           "gates": gates(groups), "groups": groups, "parameter_counts": parameter_counts(targets, hidden, layers),
           "b2_cells": b2, "per_arm": arms,
           "limits": "All-layer residual replacement; success does not show an independently executable operator. "
                     "A mean residual is not a feature, neuron or relation operator. A collapse does not show that "
                     "entity-specific information is necessary."}
    out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(json.dumps(rep["gates"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
