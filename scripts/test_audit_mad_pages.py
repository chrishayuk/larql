#!/usr/bin/env python3
"""Focused physical-schema controls for audit_mad_pages.py."""

import unittest

import numpy as np

import audit_mad_pages as pages


class PageAuditTests(unittest.TestCase):
    def test_schema_columns_require_complete_partitions(self):
        manifest = {
            "objects": [
                {
                    "layer": 49,
                    "block_channels": 2,
                    "channel_start": 0,
                    "channel_end": 2,
                },
                {
                    "layer": 49,
                    "block_channels": 2,
                    "channel_start": 2,
                    "channel_end": 4,
                },
                {
                    "layer": 49,
                    "block_channels": 4,
                    "channel_start": 0,
                    "channel_end": 4,
                },
            ]
        }
        self.assertEqual(pages.schema_columns(manifest, 49, [2, 4]), {2: [0, 1], 4: [2]})
        manifest["objects"][1]["channel_start"] = 3
        with self.assertRaisesRegex(ValueError, "gap or overlap"):
            pages.schema_columns(manifest, 49, [2, 4])

    def test_entropy_and_interference_summaries_are_exact(self):
        entropy = pages.entropy_report(np.array([[1.0, 1.0, 1.0, 1.0]]))
        self.assertAlmostEqual(entropy["mean_effective_objects"], 4.0)
        self.assertAlmostEqual(entropy["mean_effective_fraction"], 1.0)
        summary = pages.distribution_summary(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["median"], 2.0)


if __name__ == "__main__":
    unittest.main()
