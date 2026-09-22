import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parent))
from gwhead1_adjudicate import causal_summary, pairwise_cosine, safe_causal_summary


class GwHead1AdjudicationTests(unittest.TestCase):
    def test_causal_summary_uses_paired_necessity_and_ratio_draws(self) -> None:
        arms = {
            "I_full": 0,
            "I_ref": 1,
            "I_S_global": 2,
            "I_full_minus_S_global": 3,
        }
        effects = np.asarray(
            [
                [1.0, 0.0, 0.8, 0.4],
                [2.0, 0.0, 1.6, 0.8],
            ]
        )
        samples = np.asarray([[0, 1], [0, 0], [1, 1]])
        result = causal_summary(
            effects,
            arms,
            "I_S_global",
            "I_full_minus_S_global",
            samples,
        )
        self.assertAlmostEqual(result["necessity"]["estimate"], 0.9)
        self.assertAlmostEqual(result["sufficiency"]["estimate"], 0.8)
        self.assertAlmostEqual(result["sufficiency"]["lower_95"], 0.8)
        self.assertAlmostEqual(result["sufficiency"]["upper_95"], 0.8)

    def test_causal_summary_refuses_nonpositive_denominator(self) -> None:
        with self.assertRaisesRegex(ValueError, "denominator"):
            causal_summary(
                np.asarray([[0.0, 1.0, 0.5, 0.0]]),
                {"I_full": 0, "I_ref": 1, "I_S": 2, "I_minus": 3},
                "I_S",
                "I_minus",
                np.asarray([[0]]),
            )

    def test_safe_summary_preserves_a_denominator_refusal(self) -> None:
        result = safe_causal_summary(
            np.asarray([[0.0, 1.0, 0.5, 0.0]]),
            {"I_full": 0, "I_ref": 1, "I_S": 2, "I_minus": 3},
            "I_S",
            "I_minus",
            np.asarray([[0]]),
        )
        self.assertFalse(result["estimable"])
        self.assertIn("denominator", result["refusal"])
        self.assertEqual(set(result["effects"]), {"I_full", "I_ref", "I_S", "I_minus"})

    def test_pairwise_cosine_uses_all_three_pairs(self) -> None:
        states = np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        self.assertAlmostEqual(pairwise_cosine(states), 1.0)


if __name__ == "__main__":
    unittest.main()
