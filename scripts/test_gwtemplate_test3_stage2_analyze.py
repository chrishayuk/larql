"""Synthetic tests for the Stage 2 analysis (no model, no real records)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gwtemplate_test3_stage2_analyze as an  # noqa: E402

DEV = ["AA", "BB", "CC", "DD"]
# tokens: capital 10+i, currency 20+i, except CC and DD share currency token 22
CAP = {z: 10 + i for i, z in enumerate(DEV)}
CUR = {"AA": 20, "BB": 21, "CC": 22, "DD": 22}
CANDS = sorted(set(CAP.values()) | set(CUR.values())) + [99]


def bplan():
    cells = []
    for ph in ("P2", "P3"):
        for z in DEV:
            cells.append({"id": f"capital|{ph}|{z}", "country": z, "relation_name": "capital", "phrasing": ph,
                          "accepted_first_tokens": [CAP[z]]})
            cells.append({"id": f"currency_code_iso4217|{ph}|{z}", "country": z,
                          "relation_name": "currency_code_iso4217", "phrasing": ph, "accepted_first_tokens": [CUR[z]]})
    return {"cells": cells}


def ro(logits: dict, nxt=None):
    xs = [float(logits.get(t, 0.0)) for t in CANDS]
    return {"candidate_logits": xs, "log_normalizer": 0.0,
            "next": nxt if nxt is not None else CANDS[int(np.argmax(xs))]}


class ClassTests(unittest.TestCase):
    def test_fail_closed_outside_candidates(self):
        with self.assertRaises(ValueError):
            an.Classes(bplan(), DEV, [t for t in CANDS if t != 22])

    def test_fail_closed_multi_token(self):
        b = bplan()
        b["cells"][0]["accepted_first_tokens"] = [10, 11]
        with self.assertRaises(ValueError):
            an.Classes(b, DEV, CANDS)

    def test_other_classes_exclude_target_and_dedupe(self):
        cl = an.Classes(bplan(), DEV, CANDS)
        self.assertEqual(cl.other_classes("currency_code_iso4217", "P2", "AA", CAP["AA"], CUR["AA"]), [21, 22])
        self.assertEqual(cl.mult("currency_code_iso4217", "P2", 22), 2)


def target(run, arms):
    return {"id": f"s2|{run}", "run": run, "halves": [["AA", "BB"], ["CC", "DD"]], "arms": [{"name": n} for n in arms]}


class PerArmTests(unittest.TestCase):
    def setUp(self):
        self.cl = an.Classes(bplan(), DEV, CANDS)

    def rows(self, arm_ro):
        """Target AA capital P2 plus the natural currency cell needed as sep."""
        t1 = target("capital|P2|AA", list(arm_ro))
        t2 = target("currency_code_iso4217|P2|AA", ["ID-window-L1"])
        r1 = {"id": t1["id"], "natural": {"readout": ro({CAP["AA"]: 5})}, "arms": {k: {"readout": v} for k, v in arm_ro.items()}}
        r2 = {"id": t2["id"], "natural": {"readout": ro({CUR["AA"]: 5})}, "arms": {}}
        return [t1, t2], [r1, r2]

    def test_outcomes_and_b2(self):
        arms = {"S-window-L1": ro({CUR["AA"]: 6}),
                "C1-window-L1": ro({22: 7}),            # an other-country r' answer wins
                "C1A-window-L1": ro({21: 3, 22: 1}),    # half A (BB->21) favoured by A-built
                "C1B-window-L1": ro({21: 1, 22: 3}),    # half B (CC,DD->22) favoured by B-built
                "C1E-window-L1": ro({99: 4}, nxt=99)}
        T, R = self.rows(arms)
        recs, b2 = an.per_arm(T, R, self.cl, {"BB", "CC", "DD"}, {}, lambda i: "xyz")
        by = {r["kind"]: r for r in recs}
        self.assertEqual(by["S"]["outcome"], "r'(Y)")
        self.assertEqual(by["C1"]["outcome"], "other_country_r'")
        self.assertEqual(by["C1E"]["outcome"], "word")
        self.assertEqual(by["S"]["population"], "validation")
        self.assertEqual(len(b2), 1)
        self.assertAlmostEqual(b2[0]["M"], 2.0)   # dA = 3-1, dB = 3-1
        # B3 for C1: sep has 0 on 21 and 22 -> excess = arm lp minus sep lp
        self.assertAlmostEqual(by["C1"]["b3_excess"]["22"], 7.0)
        self.assertAlmostEqual(by["C1"]["b3_mean"], 3.5)

    def test_b2_missing_when_a_half_has_no_class(self):
        arms = {"C1A-window-L1": ro({}), "C1B-window-L1": ro({})}
        T, R = self.rows(arms)
        T[0]["halves"] = [["AA"], ["BB", "CC", "DD"]]   # A without the target is empty
        _, b2 = an.per_arm(T, R, self.cl, set(), {}, lambda i: "")
        self.assertIsNone(b2[0]["M"])


class BootTests(unittest.TestCase):
    def test_b3_max_support_floor(self):
        recs = [{"target": f"T{i}", "b3_excess": {"1": 0.1}} for i in range(12)]
        recs.append({"target": "T0", "b3_excess": {"2": 9.0}})       # one-target outlier class
        m, c, s = an.b3_max(recs)
        self.assertEqual((round(m, 6), c, s), (0.1, "1", 12))

    def test_boot_mean_ci_contains_mean(self):
        rng = np.random.default_rng(0)
        r = an.boot_mean({f"T{i}": [float(i % 2)] for i in range(20)}, rng)
        self.assertLessEqual(r["ci95"][0], r["mean"])
        self.assertGreaterEqual(r["ci95"][1], r["mean"])


def group(share, p2, p3, lo, b3m=0.1, b3x=0.2, other=0.0, b2=(-0.1, 0.1)):
    s = {o: {"mean": 0.0, "ci95": [0, 0]} for o in an.OUTCOMES}
    s["r'(Y)"] = {"mean": share, "ci95": [lo, 1.0]}
    s["other_country_r'"] = {"mean": other, "ci95": [0, 0]}
    return {"shares": s, "b3_mean": {"mean": b3m}, "b3_max": {"max": b3x, "class": 5},
            "b2_M": {"mean": 0.0, "ci95": list(b2)}, "_p": (p2, p3)}


class GateTests(unittest.TestCase):
    def build(self, **kw):
        g = {}
        for kind in an.CANDIDATES:
            for rel in an.RELATIONS:
                base = group(**kw)
                g[f"{kind}|{rel}|P2+P3|L1|validation"] = base
                for ph, v in zip(("P2", "P3"), base["_p"]):
                    g[f"{kind}|{rel}|{ph}|L1|validation"] = {"shares": {"r'(Y)": {"mean": v}}}
        return an.gates(g)

    def test_pass_and_c2_incomplete(self):
        r = self.build(share=0.95, p2=0.9, p3=1.0, lo=0.85)
        self.assertEqual(r["C1|capital"]["verdict"], "pass")
        self.assertEqual(r["C2|capital"]["verdict"], "incomplete: B2 not measured")

    def test_no_switch_ignores_low_leak(self):
        r = self.build(share=0.9, p2=0.55, p3=1.0, lo=0.85)
        self.assertEqual(r["C1|capital"]["verdict"], "no switch")

    def test_wide_b2_interval_containing_zero_fails(self):
        r = self.build(share=0.95, p2=0.9, p3=1.0, lo=0.85, b2=(-0.6, 0.3))
        self.assertTrue(r["C1|capital"]["verdict"].startswith("axis B fail"))
        self.assertIn("b2_pass", r["C1|capital"]["verdict"])

    def test_b3_and_secondary(self):
        r = self.build(share=0.95, p2=0.9, p3=1.0, lo=0.85, b3x=1.2, other=0.06)
        self.assertIn("b3_max_pass", r["C2|capital"]["verdict"])
        self.assertIn("other_top1_pass", r["C2|capital"]["verdict"])


if __name__ == "__main__":
    unittest.main()
