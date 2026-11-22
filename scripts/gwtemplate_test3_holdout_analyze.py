"""Test 3 Stage 4 (holdout, capital→currency): analysis (protocol §5–§6).

It runs no model. Per-arm metrics are computed by the unchanged Stage 2 code
(`gwtemplate_test3_stage2_analyze.per_arm`), so the outcome order, switch
margin, gains, KL to the separable reference and B3 excess are defined
identically.

The one population-dependent input is the class set. For B3, the
"other-country" classes are the r' answer tokens of the **59 development
countries** (minus r(Y) and r'(Y)), the same class population as Stage 2.
Holdout answer tokens are used only for the target's own r(Y) and r'(Y).

**Gates** (frozen Stage 2 values, holdout targets, L1, capital→currency,
pooled P2+P3):
- switching lower 95% ≥ 0.80, and each phrasing ≥ 0.60;
- B3 mean ≤ 0.5;
- B3 max ≤ 1.0 (support ≥ 10 targets);
- other-country top-1 ≤ 0.05.

B2 is carried forward from development (C1: result 6283; C2: result 6290). It
is not re-measured. There is no holdout analogue.

**Bootstrap:** `numpy.random.default_rng(20261118)`, 10,000 resamples,
clusters = holdout target countries. Draw order is fixed:
- kind in (C1, C2, S, C0);
- phrasing in (P2+P3, P2, P3);
- for each, the outcome shares in Stage 2's order, then m_switch, gain_rY,
  gain_rpY, kl_ref, b3_mean, b3_wmean, b3_max.

    gwtemplate_test3_holdout_analyze.py analyze STAGE4_DIR REPORT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gwtemplate_test3_stage2_analyze as s2  # noqa: E402

REL = "capital"
KINDS = ("C1", "C2", "S", "C0")
CANDIDATES = ("C1", "C2")
CARRIED_B2 = {"C1": {"result": 6283, "b2_M": 0.0463, "b2_ci95": [0.0358, 0.0563], "pass": True},
              "C2": {"result": 6290, "b2_M": 0.0315, "b2_ci95": [0.0138, 0.049], "pass": True}}


class HoldoutClasses(s2.Classes):
    """Token lookup over development + holdout; the class population (other_classes, mult) is the development 59."""

    def __init__(self, bplan: dict, dev: list[str], holdout: list[str], cands: list[int]):
        super().__init__(bplan, sorted(set(dev) | set(holdout)), cands)
        self.dev = sorted(dev)


def per_arm(targets, rows, cl, donors, decode):
    """Stage 2 per_arm over Stage 4 records. Population labels are relabelled to 'holdout' (nothing is in the fit set)."""
    arms, b2 = s2.per_arm(targets, rows, cl, set(), donors, decode)
    if b2:
        raise ValueError("Stage 4 has no B2 arms")
    for a in arms:
        a["population"] = "holdout"
    return [a for a in arms if a["relation"] == REL and a["layer"] == 1]


def summarise(arms: list[dict]) -> dict:
    rng = np.random.default_rng(s2.SEED)
    g = {}
    for kind in KINDS:
        for ph in (s2.POOLED, "P2", "P3"):
            G = [a for a in arms if a["kind"] == kind and (ph == s2.POOLED or a["phrasing"] == ph)]
            if not G:
                continue
            x = {"n": len(G), "targets": len({a["target"] for a in G}),
                 "shares": {o: s2.boot_mean(s2._clusters(G, lambda a, o=o: float(a["outcome"] == o)), rng) for o in s2.OUTCOMES}}
            for f in ("m_switch", "gain_rY", "gain_rpY", "kl_ref", "b3_mean", "b3_wmean"):
                x[f] = s2.boot_mean(s2._clusters(G, lambda a, f=f: a[f]), rng)
            x["b3_max"] = s2.boot_b3_max(G, rng)
            if kind == "C0":
                x["donor_answer_top1"] = sum(a["donor_answer_top1"] for a in G) / len(G)
            g[f"{kind}|{REL}|{ph}|L1|holdout"] = x
    return g


def gates(groups: dict) -> dict:
    out = {}
    for kind in CANDIDATES:
        k = lambda ph: f"{kind}|{REL}|{ph}|L1|holdout"  # noqa: E731
        g, g2, g3 = groups[k(s2.POOLED)], groups[k("P2")], groups[k("P3")]
        sw = g["shares"]["r'(Y)"]
        A = {"pooled_share": sw["mean"], "pooled_ci95": sw["ci95"], "P2": g2["shares"]["r'(Y)"]["mean"],
             "P3": g3["shares"]["r'(Y)"]["mean"]}
        A["pass"] = sw["ci95"] is not None and sw["ci95"][0] >= s2.GATE_A_LOWER and A["P2"] >= s2.GATE_A_PHRASING \
            and A["P3"] >= s2.GATE_A_PHRASING
        B = {"b3_mean": g["b3_mean"]["mean"], "b3_mean_pass": g["b3_mean"]["mean"] <= s2.GATE_B3_MEAN,
             "b3_max": g["b3_max"]["max"], "b3_max_class": g["b3_max"]["class"], "b3_max_support": g["b3_max"]["support"],
             "b3_max_pass": g["b3_max"]["max"] is not None and g["b3_max"]["max"] <= s2.GATE_B3_MAX,
             "other_top1": g["shares"]["other_country_r'"]["mean"],
             "other_top1_pass": g["shares"]["other_country_r'"]["mean"] <= s2.GATE_OTHER_TOP1,
             "b2_carried_from_development": CARRIED_B2[kind]}
        failed = [n for n in ("b3_mean_pass", "b3_max_pass", "other_top1_pass") if B[n] is False]
        verdict = "no switch" if not A["pass"] else ("axis B fail: " + ", ".join(failed) if failed else "pass")
        out[f"{kind}|{REL}"] = {"A": A, "B": B, "verdict": verdict}
    return out




def analyze_dir(d: Path, out: Path, prepared: dict) -> dict:
    import gwtemplate_baseline_plan as bp
    targets, rows = [], []
    for s in prepared["shards"]:
        plan = json.loads((d / s["plan"]).read_text())
        targets += plan["targets"]
        rows += [json.loads(x) for x in (d / s["output"]).read_text().splitlines() if x.strip()]
        cands = next(iter(plan["candidate_sets"].values()))
    bplan = json.loads((ROOT / "output/gw-template-factorial/baseline-01/plan.json").read_text())
    cl = HoldoutClasses(bplan, prepared["development"], prepared["eligible"], cands)
    donors = {t["run"]: t["donors"] for t in targets}
    tok = bp.tokenizer()
    arms = per_arm(targets, rows, cl, donors, lambda i: tok.decode([i]))
    groups = summarise(arms)
    rep = {"schema": "larql.gwtemplate.test3-stage4-report.v1",
           "status": "Stage 4 holdout confirmation, capital→currency, L1; frozen development candidates C1 and C2",
           "constants": {"bootstrap": s2.BOOT, "seed": s2.SEED, "gate_A_lower": s2.GATE_A_LOWER,
                         "gate_A_phrasing": s2.GATE_A_PHRASING, "gate_b3_mean": s2.GATE_B3_MEAN,
                         "gate_b3_max": s2.GATE_B3_MAX, "gate_other_top1": s2.GATE_OTHER_TOP1,
                         "b3_max_support": s2.B3_MAX_SUPPORT, "b3_class_population": "59 development countries"},
           "gates": gates(groups), "groups": groups, "per_arm": arms,
           "disclosure": "Holdout countries appeared in the natural baseline and the eligibility counts; never in "
                         "construction, fitting, development-validation interventions or candidate selection. The C2 B2 "
                         "evidence comes from a post hoc (non-blind) amendment that reused a frozen estimand.",
           "limits": "All-layer residual replacement at L1 over the template window; not an executable operator."}
    out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    return rep


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 3 and a[0] == "analyze":
        d = Path(a[1])
        prepared = json.loads((d / "prepared.json").read_text())
        rep = analyze_dir(d / "derived", Path(a[2]), prepared)
        print(json.dumps(rep["gates"], indent=1))
        sys.exit(0)
    print(__doc__, file=sys.stderr)
    sys.exit(2)
