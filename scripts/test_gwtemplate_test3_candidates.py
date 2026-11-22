"""Synthetic-data tests for the test 3 candidate builder (no model, no real capture)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gwtemplate_test3_candidates as c  # noqa: E402
from gwtemplate_test3_capture_run import vector_sha256  # noqa: E402

DEV = [f"C{i:02d}" for i in range(59)]
L, S, H = 3, 3, 8


def synth(holdout=("HX",)):
    rng = np.random.default_rng(0)
    w = {}
    for country in list(DEV) + list(holdout):
        for rel in c.RELATIONS:
            w[(rel, "P2", country)] = rng.standard_normal((L, S, H)).astype("<f4")
    return c.Capture(w, set(holdout))


class SplitTests(unittest.TestCase):
    def test_split_sizes_disjoint_deterministic(self):
        fit, val = c.split(DEV)
        self.assertEqual((len(fit), len(val)), (39, 20))
        self.assertFalse(set(fit) & set(val))
        self.assertEqual(c.split(list(reversed(DEV))), (fit, val))
        a, b = c.halves(fit)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(sorted(a + b), fit)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.cap = synth()
        self.fit, self.val = c.split(DEV)
        self.b = c.Builder(self.cap, self.fit)

    def test_c1_loo_mean_excludes_target(self):
        Y = self.fit[0]
        v, rec = self.b.candidate("C1", Y, "capital", "P2")
        members = [x for x in self.fit if x != Y]
        want = np.mean([self.cap.raw("currency_code_iso4217", "P2", x).astype(np.float64) for x in members], axis=0)
        np.testing.assert_array_equal(v, want.astype("<f4"))
        self.assertNotIn(Y, rec["members"])

    def test_c2_formula_and_no_rprime_of_target(self):
        Y = self.val[0]
        v, _ = self.b.candidate("C2", Y, "capital", "P2")
        own = self.cap.raw("capital", "P2", Y).astype(np.float64)
        m = lambda rel: np.mean([self.cap.raw(rel, "P2", x).astype(np.float64) for x in self.fit], axis=0)  # noqa: E731
        want = (own + m("currency_code_iso4217").astype("<f4").astype(np.float64)
                - m("capital").astype("<f4").astype(np.float64)).astype("<f4")
        np.testing.assert_array_equal(v, want)
        self.assertNotIn((Y, "capital", "P2", "currency_code_iso4217", Y), self.b.reads)

    def test_rule_blocks_target_new_relation_and_nonfit(self):
        Y = self.val[0]
        with self.assertRaises(c.InformationRuleError):
            self.b._cand_read(Y, "capital", "P2", "currency_code_iso4217", Y)
        with self.assertRaises(c.InformationRuleError):
            self.b._cand_read(Y, "capital", "P2", "capital", self.val[1])

    def test_holdout_refuses(self):
        with self.assertRaises(c.InformationRuleError):
            self.cap.raw("capital", "P2", "HX")

    def test_halves_exclude_target(self):
        Y = self.fit[0]
        for k in ("C1A", "C1B"):
            _, rec = self.b.candidate(k, Y, "capital", "P2")
            self.assertNotIn(Y, rec["members"])

    def test_recipe_sha_depends_on_membership(self):
        r1 = c.recipe_sha({"kind": "C1", "members": ["A", "B"]})
        r2 = c.recipe_sha({"kind": "C1", "members": ["A", "C"]})
        self.assertNotEqual(r1, r2)

    def test_store_roundtrip(self):
        st = c.Store(H)
        v, rec = self.b.candidate("C1", self.val[0], "capital", "P2")
        ids = [st.add(v[l, s], rec, l, s, 10 + s, self.cap) for l in range(L) for s in range(S)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "store.f32"
            st.write(p)
            data = np.fromfile(p, dtype="<f4")
            for vid in ids:
                e = st.index[vid]
                self.assertEqual(vector_sha256(data[e["offset"]:e["offset"] + H]), e["sha256"])
                self.assertEqual(len(e["recipe"]["inputs"]), len(rec["members"]))
                self.assertEqual(e["recipe"]["target_position"], 10 + e["recipe"]["slot"])
                self.assertEqual(e["recipe_sha256"], c.recipe_sha(e["recipe"]))


if __name__ == "__main__":
    unittest.main()
