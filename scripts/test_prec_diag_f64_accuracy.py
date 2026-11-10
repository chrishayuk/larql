"""Tests for prec_diag/f64_accuracy.py, the `hf-f64-accuracy/1` test.

The gate logic runs on synthetic arrays. The end to end case runs E1 and E2
through the experiment runner's synthetic world (fake LARQL, real
`hf_oracle.py` on a synthetic checkpoint). No real model is used.

Run: python3 -I -m unittest discover -s scripts -p 'test_prec_diag_*.py'
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prec_diag import evidence as ev  # noqa: E402
from prec_diag import f64_accuracy as fa  # noqa: E402
import test_prec_diag_experiment as tex  # noqa: E402

N = 1040


def grid(value, agreement=0.0):
    acc = {b: {i: np.full(N, value) for i in fa.INPUTS} for b in fa.BACKENDS}
    agr = {b: {i: np.full(N, agreement) for i in fa.INPUTS} for b in fa.BACKENDS}
    return acc, agr


class Verdict(unittest.TestCase):
    def test_every_cell_within_the_gate_passes(self):
        v = fa.verdict(*grid(9e-4))
        self.assertEqual(v["verdict"], fa.PASS)
        self.assertEqual(len(v["cells"]), 6)

    def test_one_position_over_on_one_input_of_one_backend_fails(self):
        acc, agr = grid(1e-4)
        acc["D"]["I_nr2"][700] = 1.01e-3
        v = fa.verdict(acc, agr)
        self.assertEqual(v["verdict"], fa.FAIL)
        self.assertEqual(v["failing"], ["D.I_nr2"])
        self.assertEqual(v["cells"]["D.I_nr2"]["positions_over_gate"], 1)

    def test_the_gate_is_inclusive_at_1e_3(self):
        self.assertEqual(fa.verdict(*grid(1e-3))["verdict"], fa.PASS)

    def test_agreement_with_f32_is_reported_and_never_gated(self):
        v = fa.verdict(*grid(1e-4, agreement=5e-3))
        self.assertEqual(v["verdict"], fa.PASS)
        self.assertEqual(v["cells"]["C.I_rep"]["agreement_vs_A32"]["max"], 5e-3)

    def test_the_name_and_threshold_are_distinct_from_strict_f32(self):
        self.assertEqual((fa.NAME, fa.VERSION, fa.GATE), ("hf-f64-accuracy", 1, 1e-3))
        self.assertEqual((fa.BACKENDS, fa.INPUTS), (("C", "D"), ("I_rep", "I_nr1", "I_nr2")))


class Audit(unittest.TestCase):
    GOOD = {"pass": True, "problem_count": 0, "call_totals": {"rmsnorm": 3}, "expected_call_totals": {"rmsnorm": 3},
            "parameters": {"pass": True, "parameters": 5, "parameters_checked": 5}}

    def problems_with(self, **changes):
        with tempfile.TemporaryDirectory() as d:
            a = json.loads(json.dumps(self.GOOD))
            for k, v in changes.items():
                if k == "parameters":
                    a["parameters"].update(v)
                else:
                    a[k] = v
            (Path(d) / "audit.json").write_text(json.dumps(a))
            return fa.audit_problems(Path(d))

    def test_a_passed_audit_has_no_problems(self):
        self.assertEqual(self.problems_with(), [])

    def test_a_failed_audit_is_a_problem(self):
        self.assertTrue(self.problems_with(**{"pass": False}))

    def test_missing_f64_calls_are_a_problem(self):
        self.assertTrue(self.problems_with(call_totals={"rmsnorm": 2}))

    def test_weights_not_all_preserved_are_a_problem(self):
        self.assertTrue(self.problems_with(parameters={"parameters_checked": 4}))
        self.assertTrue(self.problems_with(parameters={"pass": False}))

    def test_no_audit_file_is_a_problem(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(fa.audit_problems(Path(d)))


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name).resolve()
        cls.w = tex.World(cls.tmp / "w")
        cls.w.run("E1")
        cls.w.run("E2")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_a_report_over_admitted_runs_is_a_verdict_and_binds_the_freeze(self):
        out = self.tmp / "report.json"
        r = fa.run(self.w.freeze, self.w.sha, self.w.runs, out)
        self.assertIn(r["verdict"], (fa.PASS, fa.FAIL), r.get("problems"))
        self.assertEqual(r["test"], "hf-f64-accuracy/1")
        self.assertEqual(r["freeze_sha256"], self.w.sha)
        self.assertEqual(len(r["cells"]), 6)
        self.assertEqual(json.loads(out.read_text())["verdict"], r["verdict"])
        with self.assertRaisesRegex(ev.EvidenceError, "written once"):
            fa.run(self.w.freeze, self.w.sha, self.w.runs, out)

    def test_a_missing_run_makes_the_test_invalid_not_a_pass(self):
        runs, expected = tex.ex.runs_and_identities(self.w.manifest, self.w.sha, self.w.runs)
        del runs[("D", "I_nr1")]
        r = fa.evaluate(runs, expected, tex.ex.vocab_of(self.w.manifest))
        self.assertEqual(r["verdict"], fa.INVALID)
        self.assertTrue(any("not run" in p for p in r["problems"]))

    def test_another_freeze_is_refused(self):
        with self.assertRaisesRegex(ev.EvidenceError, "reviewed digest"):
            fa.run(self.w.freeze, "0" * 64, self.w.runs, self.tmp / "other.json")


if __name__ == "__main__":
    unittest.main()
