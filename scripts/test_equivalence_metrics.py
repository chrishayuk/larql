"""Unit tests for the HF-equivalence harness method (equivalence_metrics.py)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import equivalence_metrics as m  # noqa: E402

VOCAB = 16


def logits_with(order: list[float]) -> np.ndarray:
    out = np.full(VOCAB, -10.0)
    out[: len(order)] = order
    return out


class MetricTests(unittest.TestCase):
    def test_log_softmax_normalises(self):
        lp = m.log_softmax64(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        self.assertAlmostEqual(float(np.exp(lp).sum()), 1.0, places=12)
        self.assertEqual(lp.dtype, np.float64)

    def test_a_constant_shift_is_no_difference(self):
        a = logits_with([5.0, 4.0, 3.0])
        self.assertEqual(float(m.max_abs_logprob(a, a + 7.0)), 0.0)

    def test_per_position_maxima(self):
        a = np.stack([logits_with([5.0, 4.0]), logits_with([5.0, 4.0])])
        b = a.copy()
        b[1, 1] += 0.5
        got = m.max_abs_logprob(a, b)
        self.assertEqual(got.shape, (2,))
        self.assertEqual(float(got[0]), 0.0)
        self.assertGreater(float(got[1]), 0.0)

    def test_vocabulary_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            m.max_abs_logprob(np.zeros(4), np.zeros(5))

    def test_mass_bearing_error_ignores_the_extreme_tail(self):
        ref = np.array([-0.1, -3.0, -60.0])
        arm = np.array([-0.1, -3.001, -59.0])
        self.assertAlmostEqual(m.max_abs_logprob_mass_bearing(ref, arm), 0.001, places=9)

    def test_top_k_breaks_ties_by_lower_id(self):
        self.assertEqual([i for i, _ in m.top_k(np.array([1.0, 2.0, 2.0]), 2)], [1, 2])


class OrderVerdictTests(unittest.TestCase):
    REF = m.log_softmax64(logits_with([6.0, 5.0, 4.0, 3.0, 2.0, 1.9995]))

    def test_identical_order_passes(self):
        self.assertEqual(m.order_verdict(self.REF, self.REF, 1e-3), (m.PASS, []))

    def test_a_swap_inside_tolerance_is_a_tie_swap(self):
        arm = self.REF.copy()
        arm[[4, 5]] = arm[[5, 4]]
        verdict, diffs = m.order_verdict(self.REF, arm, 1e-3)
        self.assertEqual(verdict, m.TIE_SWAP)
        self.assertEqual(diffs[0]["rank"], 5)

    def test_a_swap_outside_tolerance_fails(self):
        arm = self.REF.copy()
        arm[[0, 1]] = arm[[1, 0]]
        verdict, diffs = m.order_verdict(self.REF, arm, 1e-3)
        self.assertEqual(verdict, m.FAIL)
        self.assertEqual({d["status"] for d in diffs}, {m.FAIL})

    def test_worst_ranks_severity(self):
        self.assertEqual(m.worst([m.PASS, m.TIE_SWAP]), m.TIE_SWAP)
        self.assertEqual(m.worst([m.TIE_SWAP, m.FAIL, m.PASS]), m.FAIL)
        self.assertEqual(m.worst([]), m.PASS)


class RecordTests(unittest.TestCase):
    IDENTITY = {"repo": "org/model", "revision": "abc"}

    def run_with(self, status):
        return {"ladder": {"equivalent": m.stage(status, {})}}

    def test_a_failure_survives_a_later_pass(self):
        record = m.append_run(None, self.run_with("FAIL"), self.IDENTITY)
        record = m.append_run(record, self.run_with("PASS"), self.IDENTITY)
        self.assertEqual(len(record["runs"]), 2)
        self.assertEqual(record["ever_failed"], ["equivalent"])
        self.assertEqual(record["runs"][0]["ladder"]["equivalent"]["status"], "FAIL")

    def test_a_record_refuses_another_checkpoint(self):
        record = m.append_run(None, self.run_with("PASS"), self.IDENTITY)
        with self.assertRaises(ValueError):
            m.append_run(record, self.run_with("PASS"), {"repo": "x", "revision": "y"})

    def test_stage_rejects_unknown_status(self):
        with self.assertRaises(AssertionError):
            m.stage("MAYBE", {})


if __name__ == "__main__":
    unittest.main()
