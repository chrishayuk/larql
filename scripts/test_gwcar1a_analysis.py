"""Synthetic CAR-1A held-out family and refusal checks; no model outcomes."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwcar1a_analysis import CANDIDATES, adjudicate_split, summarize  # noqa: E402


def surface():
    rng = np.random.default_rng(73)
    exact = np.empty((15, 2, 2))
    exact[:, 0] = [0.2, 0.1] + rng.normal(0, 0.002, (15, 2))
    exact[:, 1] = [0.4, 0.2] + rng.normal(0, 0.004, (15, 2))
    candidates = np.empty((15, CANDIDATES, 2, 2))
    candidates[:, :, 0] = exact[:, None, 0] * 0.35 + rng.normal(0, 0.001, (15, CANDIDATES, 2))
    candidates[:, :, 1] = exact[:, None, 1] * 0.6 + rng.normal(0, 0.002, (15, CANDIDATES, 2))
    candidates[:, 0, 0] = exact[:, 0] * 0.98 + rng.normal(0, 0.001, (15, 2))
    candidates[:, 0, 1] = exact[:, 1] * 0.98 + rng.normal(0, 0.002, (15, 2))
    candidates[:, 2, 0] = exact[:, 0] * 0.92 + rng.normal(0, 0.001, (15, 2))
    candidates[:, 2, 1] = exact[:, 1] * 0.90 + rng.normal(0, 0.002, (15, 2))
    return candidates, exact


class AnalysisTests(unittest.TestCase):
    def test_one_family_contains_all_candidates_and_both_metrics(self):
        candidates, exact = surface()
        result = adjudicate_split(candidates, exact, replicates=512)
        self.assertTrue(result["exact_denominator_stable"])
        self.assertEqual(len(result["candidates"]), CANDIDATES)
        self.assertTrue(result["candidates"]["0"]["gate_pass"])
        self.assertFalse(result["candidates"]["1"]["gate_pass"])
        self.assertTrue(result["candidates"]["2"]["gate_pass"])
        costs = [{"candidate_id": i, "per_row_bytes": 5120 if i == 2 else 10240,
                  "shared_model_bytes": 0} for i in range(CANDIDATES)]
        summary = summarize(result, result, costs)
        self.assertEqual(summary["status"], "valid")
        self.assertEqual(summary["smallest_clearing_candidate"], 2)

    def test_nonpositive_exact_draw_refuses(self):
        candidates, exact = surface()
        exact[0, 0, 0] = -10
        result = adjudicate_split(candidates, exact, replicates=128)
        self.assertFalse(result["exact_denominator_stable"])
        self.assertEqual(summarize(result, result, [
            {"candidate_id": i, "per_row_bytes": 4, "shared_model_bytes": 0}
            for i in range(CANDIDATES)])["status"], "invalid_exact_denominator")

    def test_exact_context_control_is_mandatory(self):
        candidates, exact = surface()
        result = adjudicate_split(candidates, exact, replicates=128)
        result["candidates"]["0"]["gate_pass"] = False
        costs = [{"candidate_id": i, "per_row_bytes": 4, "shared_model_bytes": 0}
                 for i in range(CANDIDATES)]
        self.assertEqual(summarize(result, result, costs)["status"],
                         "invalid_exact_context128_control")


if __name__ == "__main__":
    unittest.main()
