"""Tests for the test 3 stage 1b capture plan builder and hash re-implementation (model-free)."""
from __future__ import annotations

import hashlib
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gwtemplate_test3_capture_plan as cp  # noqa: E402
import gwtemplate_test3_capture_run as cr  # noqa: E402


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = cp.capture_plan()
        cls.vplan = cp.verify_plan("sha256:" + "0" * 64)

    def test_development_only(self):
        self.assertEqual(len(self.plan["runs"]), 236)
        hold = set(self.plan["holdout"])
        self.assertEqual(len(hold), 28)
        self.assertFalse({r["country"] for r in self.plan["runs"].values()} & hold)
        self.assertEqual(len({r["country"] for r in self.plan["runs"].values()}), 59)

    def test_windows_and_declared_hashes(self):
        for rid, r in self.plan["runs"].items():
            self.assertEqual(r["window_ids"], [r["tokens"][p] for p in r["window"]])
            self.assertLess(r["window"][-1], len(r["tokens"]) - 1)
            keys = {f"{l}|{p}" for l in range(self.plan["layers"]) for p in r["window"]}
            self.assertEqual(set(r["expected_carrier_sha256"]), keys)
            self.assertEqual(set(r["expected_kv_sha256"]), keys)
            self.assertEqual(len(r["expected_step_logit_sha256"]), len(r["tokens"]))

    def test_deterministic_sample(self):
        again = cp.verify_plan("sha256:" + "0" * 64)
        self.assertEqual(self.vplan["arms"], again["arms"])
        kinds = [a["id"].split("|")[-1][0] for a in self.vplan["arms"]]
        self.assertEqual(kinds.count("D"), 12)
        self.assertEqual(kinds.count("E"), 4)
        for a in self.vplan["arms"]:
            self.assertTrue(set(a["source_span"]) <= set(self.plan["runs"][a["donor_run"]]["window"]))
            self.assertTrue(set(a["target_span"]) <= set(self.plan["runs"][a["target_run"]]["window"]))
            self.assertTrue(a["expect_last_logit_sha256"].startswith("sha256:"))
        self.assertEqual(len(self.vplan["reinject"]), 236)


class HashTests(unittest.TestCase):
    def test_vector_sha256_layout(self):
        v = np.array([1.0, -2.5, 3.25], dtype="<f4")
        want = hashlib.sha256(struct.pack("<Q", 3) + struct.pack("<3f", 1.0, -2.5, 3.25)).hexdigest()
        self.assertEqual(cr.vector_sha256(v), want)

    def test_hash_is_length_prefixed(self):
        a = cr.vector_sha256(np.zeros(2, dtype="<f4"))
        b = cr.vector_sha256(np.zeros(3, dtype="<f4"))
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
