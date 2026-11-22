"""GW-ELEMENT-1 confirmation: the element-domain analysis (written and qualified before any run).

It runs no model and does not modify or import the frozen Test 3 analysis.

**Per-arm metrics.** Each non-identity arm on an atomic-number target Y (phrasing p) is scored from
its top-1 readout `next` and its candidate logits. The candidate set is the baseline's "element" set:
all symbol first-tokens, the bare space and the digits. Define:
- r(Y) = the bare space `" "` (the atomic-number answer's first token, shared by every element);
- r'(Y) = Y's own symbol token;
- lp(c) = candidate logit − log-normaliser (the full-vocabulary log-probability);
- nat = the natural readout of the number prompt;
- sep = the **separable reference**: the natural readout of Y's own symbol prompt in the same phrasing.

**Outcomes.** The first outcome that matches, in this order:
1. `r(Y)`;
2. `r'(Y)`;
3. `other_element_symbol`: the first token of " "+symbol of any other of the 118 elements;
4. `digit`: decoded text all digits;
5. `function_or_punct`;
6. `word`.

A switch is outcome == r'(Y).

**Fields.**
- m_switch = lp(r'(Y)) − lp(r(Y));
- gain_rY and gain_rpY, each relative to nat;
- kl_ref = KL(sep ‖ arm) over the candidate set;
- **B3** is computed over a fixed class population: the symbol tokens of the 60 naturally eligible
  fit and development elements, minus {r(Y), r'(Y)}. The per-class excess is lp(arm, c) − lp(sep, c).
  - b3_mean = the mean over classes;
  - b3_max = the largest per-class mean across a group's records, over classes with support ≥ 10 targets.

**Bootstrap.** Clusters are target elements: `numpy.random.default_rng(20261118)`, 10,000
resamples, percentile 95% intervals. The draw order is fixed (`GROUPS`, then the fields in the
order listed, then the non-worsening tests in order).

**Non-worsening against S.** For target Y, Δ_Y = the mean over Y's cells in the group of
[b3_mean(C2 arm) − b3_mean(S-full arm)], where both arms are on the **same** target cell. The test
is a cluster bootstrap of the mean of Δ, and it passes iff the **upper** 95% bound is ≤ +0.25 nats.

**Gates.** The primary test is C2-full-L4, pooled over A2 and A3. It passes iff every one of these holds:
- **A:** the lower 95% bound of the switch share is ≥ 0.80, and the A2 and A3 shares are each ≥ 0.60;
- **B3:** mean ≤ 0.5 and max ≤ 1.0 (support ≥ 10);
- other-element top-1 ≤ 0.05;
- non-worsening against S.

Protocol r2: there is no secondary test. C2-np-L4 and C2-ctrl-L4 were removed because they are
numerically degenerate (see gwelem_design.EFFECTIVE_SLOTS).

    gwelem_analyze.py analyze PLAN_DIR REPORT.json   (used by the confirmation execution only)
"""
from __future__ import annotations

import collections
import math
import re

import numpy as np

import gwelem_design as D

OUTCOMES = ("r(Y)", "r'(Y)", "other_element_symbol", "digit", "function_or_punct", "word")
FIELDS = ("m_switch", "gain_rY", "gain_rpY", "kl_ref", "b3_mean")
GROUPS = (("C2-full-L4", "A2+A3"), ("C2-full-L4", "A2"), ("C2-full-L4", "A3"),
          ("S-full-L4", "A2+A3"), ("S-full-L4", "A2"), ("S-full-L4", "A3"))
NONWORSE = (("C2-full-L4", "A2+A3"),)
FUNCTION = {"the", "a", "an", "of", "in", "is", "and", "to", "for", "it", "its", "this", "that", ""}
PUNCT = re.compile(r"^[\W_]*$")


class Tokens:
    """Answer tokens: space, each element's symbol token, and the B3 class population."""

    def __init__(self, tok, cands: list[int], elements, class_members):
        self.space = tok.encode(" ", add_special_tokens=False).ids[0]
        self.sym = {z: tok.encode(" " + s, add_special_tokens=False).ids[0] for z, s, _ in elements}
        self.pos = {t: i for i, t in enumerate(cands)}
        if len(self.pos) != len(cands):
            raise ValueError("repeated candidate")
        self.classes = sorted({self.sym[z] for z in class_members})
        if len(self.classes) != len(class_members):
            raise ValueError("B3 class tokens are not one per member")
        missing = [t for t in [self.space] + list(self.sym.values()) if t not in self.pos]
        if missing:
            raise ValueError(f"answer tokens outside the candidate set: {missing}")
        self.decode = lambda i: tok.decode([i])


def lp(ro, T: Tokens, t: int) -> float:
    return ro["candidate_logits"][T.pos[t]] - ro["log_normalizer"]


def dist(ro):
    xs = ro["candidate_logits"]
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def kl(q, p):
    if any(a > 0 and b <= 0 for a, b in zip(q, p)):
        return math.inf
    return sum(a * (math.log(a) - math.log(b)) for a, b in zip(q, p) if a > 0)


def outcome(nxt: int, Y: int, T: Tokens) -> str:
    if nxt == T.space:
        return "r(Y)"
    if nxt == T.sym[Y]:
        return "r'(Y)"
    if nxt in set(T.sym.values()) - {T.sym[Y]}:
        return "other_element_symbol"
    s = T.decode(nxt).strip()
    if s and s.isdigit():
        return "digit"
    if s.lower() in FUNCTION or PUNCT.match(s):
        return "function_or_punct"
    return "word"


def per_arm(targets: list[dict], rows: list[dict], T: Tokens) -> list[dict]:
    if [r["id"] for r in rows] != [t["id"] for t in targets]:
        raise ValueError("records do not follow the plan")
    nat = {t["run"]: r["natural"]["readout"] for t, r in zip(targets, rows)}
    out = []
    for t, row in zip(targets, rows):
        rel, ph, yk = t["run"].split("|")
        if rel != D.SOURCE:
            continue
        Y = int(yk)
        rY, rpY = T.space, T.sym[Y]
        n, sep = nat[t["run"]], nat[D.run_id(D.INJECTED, ph, Y)]
        cls = [c for c in T.classes if c not in (rY, rpY)]
        for a in t["arms"]:
            if a.get("identity"):
                continue
            ro = row["arms"][a["name"]]["readout"]
            ex = {c: lp(ro, T, c) - lp(sep, T, c) for c in cls}
            out.append({"target": Y, "phrasing": ph, "arm": a["name"], "outcome": outcome(ro["next"], Y, T),
                        "next": ro["next"], "lp_rY": lp(ro, T, rY), "lp_rpY": lp(ro, T, rpY),
                        "m_switch": lp(ro, T, rpY) - lp(ro, T, rY),
                        "gain_rY": lp(ro, T, rY) - lp(n, T, rY), "gain_rpY": lp(ro, T, rpY) - lp(n, T, rpY),
                        "kl_ref": kl(dist(sep), dist(ro)), "b3_mean": sum(ex.values()) / len(ex),
                        "b3_excess": {str(c): v for c, v in ex.items()}})
    return out


# ── cluster bootstrap ──

def _draws(rng, n):
    idx = rng.integers(0, n, size=(D.BOOT, n))
    w = np.zeros((D.BOOT, n))
    np.add.at(w, (np.arange(D.BOOT)[:, None], idx), 1)
    return w


def _pct(vals):
    v = np.sort(np.asarray(vals))
    return [float(v[int(0.025 * len(v))]), float(v[int(0.975 * len(v)) - 1])]


def boot_mean(units: dict, rng) -> dict:
    keys = sorted(units)
    flat = [x for k in keys for x in units[k]]
    if not flat:
        return {"n": 0, "mean": None, "ci95": None}
    sums = np.array([sum(units[k]) for k in keys])
    cnts = np.array([len(units[k]) for k in keys], dtype=float)
    w = _draws(rng, len(keys))
    return {"n": len(flat), "clusters": len(keys), "mean": sum(flat) / len(flat),
            "ci95": _pct((w @ sums) / (w @ cnts))}


def _clusters(G, f):
    u = collections.defaultdict(list)
    for r in G:
        u[r["target"]].append(f(r))
    return u


def _b3_tables(G):
    keys = sorted({r["target"] for r in G})
    classes = sorted({c for r in G for c in r["b3_excess"]}, key=int)
    ki, ci = {k: i for i, k in enumerate(keys)}, {c: j for j, c in enumerate(classes)}
    S, N = np.zeros((len(keys), len(classes))), np.zeros((len(keys), len(classes)))
    for r in G:
        for c, v in r["b3_excess"].items():
            S[ki[r["target"]], ci[c]] += v
            N[ki[r["target"]], ci[c]] += 1
    return keys, classes, S, N


def _b3_means(w, S, N):
    sums, cnts = w @ S, w @ N
    support = (w > 0).astype(float) @ (N > 0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((support >= D.B3_MAX_SUPPORT) & (cnts > 0), sums / np.where(cnts > 0, cnts, 1), -np.inf)


def boot_b3_max(G, rng) -> dict:
    keys, classes, S, N = _b3_tables(G)
    m = _b3_means(np.ones((1, len(keys))), S, N)[0]
    if not len(classes) or not np.isfinite(m).any():
        return {"max": None, "class": None, "support": 0, "ci95": None}
    j = int(np.argmax(m))
    bm = _b3_means(_draws(rng, len(keys)), S, N).max(axis=1)
    bm = bm[np.isfinite(bm)]
    return {"max": float(m[j]), "class": int(classes[j]), "support": int((N[:, j] > 0).sum()),
            "ci95": _pct(bm) if len(bm) else None,
            "classes_eligible": int(((N > 0).sum(axis=0) >= D.B3_MAX_SUPPORT).sum())}


def select(arms, arm, phs):
    return [a for a in arms if a["arm"] == arm and (phs == "A2+A3" or a["phrasing"] == phs)]


def nonworse(arms, arm, phs, rng) -> dict:
    c2 = {(a["target"], a["phrasing"]): a["b3_mean"] for a in select(arms, arm, phs)}
    s = {(a["target"], a["phrasing"]): a["b3_mean"] for a in select(arms, "S-full-L4", phs)}
    if set(c2) != set(s):
        raise ValueError(f"{arm} {phs}: C2 and S cells differ")
    u = collections.defaultdict(list)
    for k in sorted(c2):
        u[k[0]].append(c2[k] - s[k])
    b = boot_mean(u, rng)
    upper = b["ci95"][1] if b["ci95"] else None
    return {"delta_mean": b["mean"], "ci95": b["ci95"], "cells": b["n"], "targets": b.get("clusters"),
            "margin": D.NONWORSE_MARGIN, "pass": upper is not None and upper <= D.NONWORSE_MARGIN}


def summarise(arms: list[dict]) -> tuple[dict, dict]:
    rng = np.random.default_rng(D.SEED)
    groups = {}
    for arm, phs in GROUPS:
        G = select(arms, arm, phs)
        if not G:
            continue
        g = {"n": len(G), "targets": len({a["target"] for a in G}),
             "shares": {o: boot_mean(_clusters(G, lambda a, o=o: float(a["outcome"] == o)), rng) for o in OUTCOMES}}
        for f in FIELDS:
            g[f] = boot_mean(_clusters(G, lambda a, f=f: a[f]), rng)
        g["b3_max"] = boot_b3_max(G, rng)
        g["other_element_tokens"] = dict(collections.Counter(
            str(a["next"]) for a in G if a["outcome"] == "other_element_symbol"))
        groups[f"{arm}|{phs}"] = g
    nw = {f"{arm}|{phs}": nonworse(arms, arm, phs, rng) for arm, phs in NONWORSE
          if select(arms, arm, phs)}
    return groups, nw


def gate(groups, nw, arm, pooled, per_phrasing) -> dict:
    g = groups[f"{arm}|{pooled}"]
    sw = g["shares"]["r'(Y)"]
    A = {"share": sw["mean"], "ci95": sw["ci95"],
         "per_phrasing": {ph: groups[f"{arm}|{ph}"]["shares"]["r'(Y)"]["mean"] for ph in per_phrasing}}
    A["pass"] = (sw["ci95"] is not None and sw["ci95"][0] >= D.GATE_A_LOWER
                 and all(v >= D.GATE_A_PHRASING for v in A["per_phrasing"].values()))
    B = {"b3_mean": g["b3_mean"]["mean"], "b3_mean_ci95": g["b3_mean"]["ci95"],
         "b3_mean_pass": g["b3_mean"]["mean"] <= D.GATE_B3_MEAN,
         "b3_max": g["b3_max"]["max"], "b3_max_class": g["b3_max"]["class"], "b3_max_support": g["b3_max"]["support"],
         "b3_max_pass": g["b3_max"]["max"] is not None and g["b3_max"]["max"] <= D.GATE_B3_MAX,
         "other_top1": g["shares"]["other_element_symbol"]["mean"],
         "other_top1_pass": g["shares"]["other_element_symbol"]["mean"] <= D.GATE_OTHER_TOP1,
         "nonworse_vs_S": nw[f"{arm}|{pooled}"], "nonworse_pass": nw[f"{arm}|{pooled}"]["pass"]}
    failed = [k for k in ("b3_mean_pass", "b3_max_pass", "other_top1_pass", "nonworse_pass") if not B[k]]
    verdict = "no switch" if not A["pass"] else ("axis B fail: " + ", ".join(failed) if failed else "pass")
    return {"A": A, "B": B, "verdict": verdict}


def gates(groups, nw) -> dict:
    return {"primary": gate(groups, nw, D.PRIMARY, "A2+A3", D.PHRASINGS),
            "rule": "The primary verdict alone decides the confirmation (protocol r2: no secondary)."}


def check_coverage(targets, expected) -> list[str]:
    """The plan must carry exactly the frozen inventory for the expected elements (reviewer finding C4)."""
    want = set()
    for Y in expected:
        for ph in D.PHRASINGS:
            want |= {(D.run_id(D.SOURCE, ph, Y), a) for a in D.ARMS[ph]}
            want |= {(D.run_id(D.INJECTED, ph, Y), a) for a in D.REFERENCE_ARMS}
    got = [(t["run"], a["name"]) for t in targets for a in t["arms"]]
    f = []
    if len(got) != len(set(got)):
        f.append("repeated arms")
    if set(got) != want:
        f.append(f"inventory differs: missing {len(want - set(got))}, extra {len(set(got) - want)}")
    return f


def analyse(targets, rows, T: Tokens, expected=D.CONFIRMATION) -> dict:
    cov = check_coverage(targets, expected)
    if cov:
        raise ValueError(f"analysis refused: {cov}")
    arms = per_arm(targets, rows, T)
    groups, nw = summarise(arms)
    return {"gates": gates(groups, nw), "groups": groups, "nonworse": nw, "per_arm": arms,
            "constants": {"bootstrap": D.BOOT, "seed": D.SEED, "gate_A_lower": D.GATE_A_LOWER,
                          "gate_A_phrasing": D.GATE_A_PHRASING, "gate_b3_mean": D.GATE_B3_MEAN,
                          "gate_b3_max": D.GATE_B3_MAX, "b3_max_support": D.B3_MAX_SUPPORT,
                          "gate_other_top1": D.GATE_OTHER_TOP1, "nonworse_margin": D.NONWORSE_MARGIN,
                          "b3_class_population": "symbol tokens of the 60 eligible fit + development elements"}}
