"""Synthetic STATE-1 selection and inference checks; no model outcomes used."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwv2_adjudicate import distributions, effect  # noqa: E402
from gwstate1_analysis import (  # noqa: E402
    EXACT, adjudicate_split, context_heads, frozen_family, select_train,
    subject_effects_from_logits, verdict,
)


def surface(subjects):
    rng = np.random.default_rng(91)
    values = rng.normal(0, 0.012, size=(subjects, 256, 2, 2))
    values[:, :, 0] += 0.2
    values[:, :, 1] += 0.1
    values[:, EXACT, 0] += 0.05
    values[:, 0, 0] = 0.23 + rng.normal(0, 0.005, size=(subjects, 2))
    return values


class AnalysisTests(unittest.TestCase):
    def test_context_bit_order_and_family(self):
        self.assertEqual(context_heads(1), (0,))
        self.assertEqual(context_heads(2), (2,))
        self.assertEqual(context_heads(127), (0, 2, 3, 4, 5, 6, 7))
        self.assertEqual(frozen_family([0, 255]), (0, 127, 128, 255))
        with self.assertRaises(ValueError):
            frozen_family([128, 0])

    def test_train_selection_uses_each_carrier_branch(self):
        values = surface(44)
        values[:, 0, 0] = 0.24
        values[:, 128, 0] = 0.23
        result = select_train(values, np.ones(7))
        self.assertEqual(result["carrier_branch_selected"], [0, 128])

    def test_heldout_family_contains_sentinels_and_exact_identity(self):
        values = surface(15)
        result = adjudicate_split(values, [1, 129], replicates=512)
        compact = adjudicate_split(values[:, result["family"]], [1, 129],
                                   replicates=512, context_ids=result["family"])
        self.assertEqual(result["family"], [0, 1, 127, 128, 129, 255])
        self.assertEqual(compact, result)
        self.assertTrue(result["exact_stable"])
        self.assertEqual(result["contexts"][str(EXACT)]["proximal_retention"], [1.0, 1.0])
        self.assertEqual(result["contexts"][str(EXACT)]["proximal_simultaneous_lower95"], [1.0, 1.0])
        self.assertEqual(verdict(result, result, [1, 129]), "edge_only")

    def test_nonpositive_exact_refuses(self):
        values = surface(15)
        values[:, EXACT, 0] = -0.1
        result = adjudicate_split(values, [None, None], replicates=32)
        self.assertFalse(result["exact_stable"])
        self.assertEqual(verdict(result, result, [None, None]), "no_stable_context")

    def test_paired_identical_h1_arms_have_zero_control_adjusted_effect(self):
        rows = []
        for subject in ("AAA", "BBB"):
            for relation in ("capital", "currency", "language"):
                for family in ("canonical", "question", "alternate"):
                    edge = f"{subject}-{relation}-{family}"
                    donor = f"{'BBB' if subject == 'AAA' else 'AAA'}-{relation}-{family}"
                    rows.append({"row": {"edge_id": edge, "subject_id": subject,
                                         "split": "validation",
                                         "semantic_edge": {"relation": relation},
                                         "control_ids": [{"kind": "same_relation_different_subject",
                                                          "paired_edge_id": donor}]}})
        rng = np.random.default_rng(42)
        one_arm = rng.normal(size=(18, 2, 5)).astype("<f4")
        paired = np.repeat(one_arm[:, :, None], 2, axis=2)
        subjects, effects = subject_effects_from_logits(rows, paired, paired, [0, 255], "validation")
        self.assertEqual(subjects, ["AAA", "BBB"])
        np.testing.assert_array_equal(effects, np.zeros((2, 2, 2, 2)))
        reordered = list(reversed(range(18)))
        _, compact = subject_effects_from_logits(rows, paired[reordered], paired[reordered],
                                                 [0, 255], "validation", reordered)
        np.testing.assert_array_equal(compact, effects)
        altered = paired.copy()
        altered[:, :, 1] += rng.normal(0, 0.2, size=altered[:, :, 1].shape)
        _, measured = subject_effects_from_logits(rows, altered, altered,
                                                  [0, 255], "validation")
        facts = np.arange(18).reshape(6, 3)
        controls = np.concatenate((facts[3:], facts[:3]))
        before = distributions(rng.normal(size=(18, 5)))
        donor = distributions(altered[:, 0, 0])
        target = distributions(altered[:, 0, 1])
        direct = effect(before, target, facts, controls) - effect(before, donor, facts, controls)
        np.testing.assert_allclose(measured[:, 0, 0, 0], direct.reshape(2, 3).mean(axis=1), atol=1e-14)


if __name__ == "__main__":
    unittest.main()
