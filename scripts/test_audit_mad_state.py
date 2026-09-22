#!/usr/bin/env python3
"""Focused topology controls for audit_mad_state.py."""

import unittest

import numpy as np

import audit_mad_state as state


class StateTopologyTests(unittest.TestCase):
    def setUp(self):
        self.rankings = np.array(
            [
                [[0, 1, 2, 3, 4]],
                [[1, 0, 2, 3, 4]],
                [[0, 2, 1, 3, 4]],
            ],
            dtype=np.int32,
        )

    def test_persistence_selects_repeated_members(self):
        selected, report = state.persistent_selection(self.rankings, 2)
        np.testing.assert_array_equal(selected, np.array([[0, 1]]))
        self.assertAlmostEqual(report["mean_selected_layer_fraction"], 5.0 / 6.0)
        self.assertAlmostEqual(report["mean_selected_all_layers_fraction"], 0.5)
        self.assertEqual(report["mean_candidate_union_size"], 3.0)

    def test_stability_reports_boundary_turnover(self):
        report = state.stability_report(self.rankings, 2)
        self.assertAlmostEqual(report["mean_adjacent_jaccard"], 2.0 / 3.0)
        self.assertAlmostEqual(report["mean_adjacent_retention"], 0.75)
        self.assertAlmostEqual(report["chance_adjacent_retention"], 0.4)
        self.assertAlmostEqual(report["chance_adjusted_adjacent_retention"], 7.0 / 12.0)
        self.assertEqual(report["mean_ten_layer_union_size"], 3.0)
        self.assertEqual(report["mean_ten_layer_intersection_size"], 1.0)

    def test_wording_family_removes_only_alias_suffix(self):
        self.assertEqual(
            state.wording_family({"wording_id": "template:7:alias:2"}),
            "template:7",
        )


if __name__ == "__main__":
    unittest.main()
