#!/usr/bin/env python3
"""Focused controls for audit_mad_keys.py."""

import unittest

import numpy as np

import audit_mad_field as field
import audit_mad_keys as keys


class KeyAuditTests(unittest.TestCase):
    def test_compound_scores_equal_explicit_unit_block_concatenation(self):
        residuals = np.array(
            [
                [[1.0, 2.0], [2.0, 3.0], [4.0, 3.0]],
                [[2.0, 1.0], [3.0, 2.0], [3.0, 5.0]],
                [[1.0, 3.0], [4.0, 4.0], [5.0, 7.0]],
            ]
        )
        spec = (("state", 38), ("delta", 37, 38))
        slots = {37: 0, 38: 1}
        actual = keys.compound_scores(residuals, [0, 1], [2], slots, spec)

        def explicit(row):
            state = field.normalized(residuals[[row], 1, :])[0]
            delta = field.normalized(
                (residuals[[row], 1, :] - residuals[[row], 0, :])
            )[0]
            value = np.concatenate([state, delta])
            return value / np.linalg.norm(value)

        expected = explicit(2)[None, :] @ np.stack([explicit(0), explicit(1)]).T
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_identical_cohort_has_no_disagreement(self):
        targets = np.array([[3.0, 1.0], [3.0, 1.0], [3.0, 1.0]])
        selections = np.array([[0, 1, 2]])
        report = keys.cohort_disagreement(
            targets, selections, np.array([1.0, 1.0]), [0.5]
        )
        self.assertAlmostEqual(report["mean_neighbor_centroid_cosine"], 1.0)
        self.assertAlmostEqual(report["mean_neighbor_centroid_js_divergence"], 0.0)
        self.assertAlmostEqual(report["mean_pairwise_top_page_jaccard"]["0.5"], 1.0)


if __name__ == "__main__":
    unittest.main()
