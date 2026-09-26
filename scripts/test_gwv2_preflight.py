from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gwv2_population import read_jsonl
from gwv2_preflight import audit, subject_lengths

PROTOCOL = ROOT / "bench/gw-v2/gemma3-4b-it-phase1/gwv2-protocol.json"
ROWS = PROTOCOL.parent / "input.jsonl"


class GwV2PreflightTests(unittest.TestCase):
    def test_outcome_free_audit_exposes_unequal_length_control_gap(self) -> None:
        result = audit(PROTOCOL)
        self.assertTrue(result["protocol_and_population_valid"])
        self.assertEqual(result["inherited_role_rows_checked"], 666)
        self.assertEqual(result["missing_reference_cells"], [])
        self.assertEqual(
            result["splits"]["train"]["subject_token_count_histogram"],
            {1: 28, 2: 12, 3: 2, 4: 1, 5: 1},
        )
        self.assertEqual(result["splits"]["train"]["subject_token_fit_rows"], 603)
        self.assertEqual(result["fit_sham"]["singleton_token_count_subjects"], ["KNA", "STP"])
        self.assertIsNone(result["scientific_verdict"])
        self.assertFalse(result["fit_sham"]["amendment_applied"])
        self.assertEqual(result["model_prompts_executed_by_this_audit"], 0)

    def test_same_subject_cannot_acquire_different_token_lengths(self) -> None:
        rows = read_jsonl(ROWS)
        changed = copy.deepcopy(rows[0])
        changed["prompt"]["token_ids"].insert(4, 100)
        with self.assertRaisesRegex(ValueError, "token count changes"):
            subject_lengths([rows[0], changed])

    def test_same_subject_cannot_cross_splits(self) -> None:
        row = read_jsonl(ROWS)[0]
        changed = copy.deepcopy(row)
        changed["split"] = "train" if row["split"] != "train" else "test"
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            subject_lengths([row, changed])


if __name__ == "__main__":
    unittest.main()
