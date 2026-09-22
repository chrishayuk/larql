#!/usr/bin/env python3
"""Focused controls for audit_mad_addresses.py."""

import unittest

import numpy as np

import audit_mad_addresses as addresses


class AddressAuditTests(unittest.TestCase):
    def test_comparison_family_uses_cumulative_residual_join_edges(self):
        transitions = addresses.comparison_edges([38, 39])
        self.assertEqual(len(transitions), 7)
        self.assertEqual(
            transitions[0],
            (
                (38, "layer_input"),
                (38, "attention_input"),
                "attention_normalization",
            ),
        )
        self.assertIn(
            (
                (38, "layer_input"),
                (38, "post_attention"),
                "attention_residual_join",
            ),
            transitions,
        )
        self.assertIn(
            (
                (38, "post_attention"),
                (39, "layer_input"),
                "ffn_residual_join",
            ),
            transitions,
        )

    def test_key_matrix_reads_residual_and_seam_slots(self):
        residuals = np.arange(2 * 2 * 3).reshape(2, 2, 3)
        seams = np.arange(2 * 2 * 5 * 3).reshape(2, 2, 5, 3)
        residual = addresses.key_matrix(
            "layer_input", 39, residuals, seams, {38: 0, 39: 1}, {}, {}
        )
        np.testing.assert_array_equal(residual, residuals[:, 1, :])
        seam = addresses.key_matrix(
            "ffn_input",
            38,
            residuals,
            seams,
            {},
            {38: 1},
            {"ffn_input": 3},
        )
        np.testing.assert_array_equal(seam, seams[:, 1, 3, :])

    def test_l49_ffn_output_is_only_post_consumption_seam(self):
        for kind in addresses.SEAM_ORDER[:-1]:
            self.assertEqual(
                addresses.availability(49, kind, 49), "pre_down_same_layer"
            )
        self.assertEqual(
            addresses.availability(49, "ffn_output", 49),
            "post_consumption_control",
        )


if __name__ == "__main__":
    unittest.main()
