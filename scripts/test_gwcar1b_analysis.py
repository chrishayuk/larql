"""Synthetic witnesses for the fresh-cohort paired statistic and full family."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwcar1b_analysis import adjudicate, effects, group_js, probabilities, summarize  # noqa: E402


def surface(n=34):
    rng = np.random.default_rng(187)
    reference = np.empty((n, 2, 2))
    reference[:, 0] = [0.2, 0.12] + rng.normal(0, .004, (n, 2))
    reference[:, 1] = [0.1, 0.05] + rng.normal(0, .002, (n, 2))
    values = np.empty((n, 20, 2, 2))
    values[:, :, 0] = reference[:, None, 0] * .3 + rng.normal(0, .001, (n, 20, 2))
    values[:, :, 1] = reference[:, None, 1] * .2 + rng.normal(0, .001, (n, 20, 2))
    for i in (0, 11):
        values[:, i] = reference * .98 + rng.normal(0, .001, (n, 2, 2))
    return values, reference


class AnalysisTests(unittest.TestCase):
    def test_full_family_and_both_population_sizes(self):
        results = []
        for n in (34, 33):
            values, reference = surface(n)
            result = adjudicate(values, reference, replicates=512)
            self.assertEqual(result["status"], "valid")
            self.assertEqual(result["simultaneous_readouts"], 104)
            self.assertEqual(len(result["comparisons"]), 6)
            self.assertTrue(result["candidates"]["11"]["gate_pass"])
            self.assertTrue(result["comparisons"]["11-17"]["gate_pass"])
            results.append(result)
        summary = summarize(*results)
        self.assertEqual(summary["clearing_sources"], [11])
        self.assertEqual(summary["informative_delta_sources"], [11])

    def test_adequate_template_does_not_prove_entity_information(self):
        values, reference = surface()
        values[:, 8] = values[:, 11]
        result = adjudicate(values, reference, replicates=128)
        self.assertTrue(result["candidates"]["11"]["gate_pass"])
        self.assertFalse(result["comparisons"]["11-8"]["gate_pass"])
        self.assertEqual(summarize(result, result)["informative_delta_sources"], [])
        self.assertTrue(result["degenerate_readout_indices"])

    def test_control_failure_and_unstable_denominator_invalidate(self):
        values, reference = surface()
        values[:, 0] = values[:, 1]
        result = adjudicate(values, reference, replicates=128)
        self.assertEqual(result["status"], "invalid_exact_context128_control")
        self.assertEqual(summarize(result, result)["status"], "invalid")
        reference[0, 0] = -100
        self.assertEqual(adjudicate(values, reference, replicates=128)["status"], "invalid_exact_denominator")

    def test_paired_effect_matches_direct_js_and_refuses_cross_split_controls(self):
        rng = np.random.default_rng(391)
        rows = [{"subject_id": f"s{s}", "split": "validation", "relation": "capital", "template": t}
                for s in range(4) for t in ("canonical", "question", "alternate")]
        for i, row in enumerate(rows):
            subject, family = divmod(i, 3)
            row["donor_index"] = ((subject + family + 1) % 4) * 3 + family
        logits = rng.normal(size=(12, 20, 2, 9))
        subjects, actual = effects(rows, logits, logits, "validation")
        self.assertEqual(subjects, [f"s{s}" for s in range(4)])
        probs = probabilities(logits, False)
        facts = np.arange(12).reshape(4, 3)
        controls = np.array([[rows[i]["donor_index"] for i in g] for g in facts])
        difference = group_js(probs, facts) - group_js(probs, controls)
        np.testing.assert_allclose(actual[:, :, 0, 0], difference[:, :, 0] - difference[:, :, 1])
        # Flipping treatment/control flips effects exactly up to roundoff.
        _, reverse = effects(rows, logits[:, :, ::-1], logits[:, :, ::-1], "validation")
        np.testing.assert_allclose(reverse, -actual, atol=1e-15)
        rows[0]["donor_index"] = 0
        with self.assertRaisesRegex(ValueError, "matched control"):
            effects(rows, logits, logits, "validation")


if __name__ == "__main__":
    unittest.main()
