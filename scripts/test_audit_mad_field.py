#!/usr/bin/env python3
"""Focused synthetic controls for audit_mad_field.py."""

import unittest

import numpy as np

import audit_mad_field as field


class FieldEstimatorTests(unittest.TestCase):
    def test_distance_kernel_favors_closest_target(self):
        targets = np.array([[1.0, 0.0], [0.0, 1.0]])
        predicted = field.distance_weighted_profile(np.array([0.99, 0.50]), targets)
        self.assertGreater(predicted[0], predicted[1])

    def test_local_linear_recovers_smooth_affine_field(self):
        neighbors = np.array([[0.0], [1.0], [2.0], [3.0]])
        targets = 2.0 * neighbors + 1.0
        query = np.array([0.5])
        predicted = field.local_linear_profile(query, neighbors, targets)
        uniform = targets.mean(axis=0)
        self.assertLess(abs(predicted[0] - 2.0), abs(uniform[0] - 2.0))

    def test_local_low_rank_recovers_smooth_affine_field(self):
        neighbors = np.array([[-2.0], [-1.0], [1.0], [2.0], [3.0]])
        targets = 3.0 * neighbors + 5.0
        predicted, rank = field.local_low_rank_profile(
            np.array([0.5]), neighbors, targets
        )
        self.assertEqual(rank, 1)
        self.assertAlmostEqual(predicted[0], 6.5, places=2)

    def test_non_smooth_negative_control_is_not_positive_evidence(self):
        result = field.paired_sign_flip(
            -np.linspace(0.1, 1.0, 32), permutations=2000, seed=7
        )
        self.assertLess(result["mean_delta"], 0.0)
        self.assertGreater(result["p_value_plus_one"], 0.9)

    def test_byte_budget_and_degenerate_fallback(self):
        truth = np.array([4.0, 3.0, 2.0, 1.0])
        bytes_ = np.array([2.0, 2.0, 2.0, 2.0])
        result = field.coverage(truth, np.arange(4), bytes_, 0.5)
        self.assertEqual(result["candidate_byte_fraction"], 0.5)
        self.assertEqual(result["contribution_coverage"], 0.7)
        fallback = np.array([1.0, 2.0])
        profile, used = field.safe_profile(np.array([-1.0, -2.0]), fallback)
        self.assertTrue(used)
        np.testing.assert_array_equal(profile, fallback)


if __name__ == "__main__":
    unittest.main()
