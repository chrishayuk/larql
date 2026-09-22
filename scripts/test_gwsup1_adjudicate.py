import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gwsup1_adjudicate import mean_pairwise_js, site_index, softmax, stratified_bootstrap


class GwSup1AdjudicationTests(unittest.TestCase):
    def test_identical_prompt_distributions_have_zero_js(self) -> None:
        distribution = np.array([[0.2, 0.3, 0.5]] * 3)
        self.assertAlmostEqual(float(mean_pairwise_js(distribution)), 0.0)

    def test_site_order_is_attention_then_ffn(self) -> None:
        self.assertEqual(site_index({"layer": 24, "site": "attention"}), 48)
        self.assertEqual(site_index({"layer": 24, "site": "ffn"}), 49)

    def test_scale_control_removes_affine_logit_scale(self) -> None:
        logits = np.array([[1.0, 2.0, 4.0]])
        zscore = lambda value: (value - value.mean(axis=1, keepdims=True)) / value.std(
            axis=1, keepdims=True
        )
        np.testing.assert_allclose(softmax(zscore(logits)), softmax(zscore(logits * 19 + 7)))

    def test_stratified_bootstrap_is_seeded_and_negative(self) -> None:
        values = np.array([-0.4, -0.3, -0.2, -0.1])
        relations = np.array(["a", "a", "b", "b"])
        first = stratified_bootstrap(values, relations, resamples=100)
        second = stratified_bootstrap(values, relations, resamples=100)
        self.assertEqual(first, second)
        self.assertLess(first["upper_95"], 0)


if __name__ == "__main__":
    unittest.main()
