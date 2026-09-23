import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gwconv1_adjudicate import (
    bootstrap,
    concentration,
    mean_pairwise_cosine,
    mean_pairwise_js,
    site_label,
    softmax,
)


class GwConv1AdjudicationTests(unittest.TestCase):
    def test_identical_distributions_have_zero_js(self) -> None:
        distribution = np.array([[0.2, 0.3, 0.5]] * 3)
        self.assertAlmostEqual(float(mean_pairwise_js(distribution)), 0.0)

    def test_pairwise_cosine_is_one_for_positive_rescaling(self) -> None:
        base = np.array([[1.0, 2.0], [2.0, 1.0]])
        states = np.asarray([base, base * 2, base * 9])
        np.testing.assert_allclose(mean_pairwise_cosine(states), np.ones(2))

    def test_site_order_is_attention_then_ffn(self) -> None:
        self.assertEqual(site_label(48), {"index": 48, "layer": 24, "site": "attention"})
        self.assertEqual(site_label(49), {"index": 49, "layer": 24, "site": "ffn"})

    def test_zscore_removes_affine_logit_scale(self) -> None:
        logits = np.array([[1.0, 2.0, 4.0]])
        zscore = lambda value: (value - value.mean(axis=1, keepdims=True)) / value.std(
            axis=1, keepdims=True
        )
        np.testing.assert_allclose(softmax(zscore(logits)), softmax(zscore(logits * 19 + 7)))

    def test_concentration_reports_effective_site_count(self) -> None:
        result = concentration(np.array([1.0, 1.0, -2.0, 0.0]))
        self.assertAlmostEqual(result["top_mass_fraction"]["1"], 0.5)
        self.assertAlmostEqual(result["herfindahl"], 0.5)
        self.assertAlmostEqual(result["effective_site_count"], 2.0)

    def test_bootstrap_is_seeded_and_positive(self) -> None:
        values = np.array([0.4, 0.3, 0.2, 0.1])
        relations = np.array(["a", "a", "b", "b"])
        first = bootstrap(values, relations, True)
        second = bootstrap(values, relations, True)
        self.assertEqual(first, second)
        self.assertGreater(first["lower_95"], 0)


if __name__ == "__main__":
    unittest.main()
