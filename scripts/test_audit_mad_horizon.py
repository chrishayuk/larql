#!/usr/bin/env python3
"""Focused controls for audit_mad_horizon.py."""

import unittest

import numpy as np

import audit_mad_horizon as horizon


class HorizonAuditTests(unittest.TestCase):
    def test_slope_rows_recovers_known_slopes(self):
        x = np.array([0.0, 6.0, 9.0, 11.0])
        values = np.stack((2.0 + 0.5 * x, -3.0 - 0.25 * x))
        np.testing.assert_allclose(horizon.slope_rows(values, x), [0.5, -0.25])

    def test_recovered_rows_mean_matches_aggregate_fraction(self):
        popularity = np.array([0.2, 0.3, 0.4])
        oracle = np.array([0.5, 0.7, 0.6])
        knn = np.array([0.3, 0.5, 0.5])
        rows, headroom = horizon.recovered_rows(knn, popularity, oracle)
        expected = (knn.mean() - popularity.mean()) / (oracle.mean() - popularity.mean())
        self.assertAlmostEqual(rows.mean(), expected)
        self.assertAlmostEqual(headroom, oracle.mean() - popularity.mean())

    def test_stages_exclude_post_consumption_output(self):
        self.assertNotIn((49, "ffn_output", 11.0), horizon.STAGES)
        self.assertEqual(horizon.STAGES[-1], (49, "ffn_input", 11.0))


if __name__ == "__main__":
    unittest.main()
