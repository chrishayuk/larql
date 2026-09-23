"""Regression checks for the inherited STATE-1 matched-control derangement."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwstate1_preflight import EXECUTION, matched_donors  # noqa: E402
from gwv2_prepare import validate_execution  # noqa: E402


class MatchedDonorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = validate_execution(EXECUTION)["rows"]

    def test_frozen_map_is_a_full_derangement(self):
        donors = matched_donors(self.rows)
        self.assertEqual(len(donors), 666)
        self.assertEqual(set(donors), set(range(666)))
        self.assertTrue(all(index != donor for index, donor in enumerate(donors)))

    def test_reused_donor_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        donors = matched_donors(rows)
        recipient = next(i for i, row in enumerate(rows)
                         if i != 0 and i != donors[0]
                         and row["row"]["split"] == rows[0]["row"]["split"]
                         and row["row"]["semantic_edge"]["relation"] == rows[0]["row"]["semantic_edge"]["relation"]
                         and row["row"]["semantic_edge"]["prompt_semantic_family"] == rows[0]["row"]["semantic_edge"]["prompt_semantic_family"])
        control = next(c for c in rows[recipient]["row"]["control_ids"]
                       if c["kind"] == "same_relation_different_subject")
        control["paired_edge_id"] = rows[donors[0]]["row"]["edge_id"]
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            matched_donors(rows)

    def test_cross_split_donor_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        donor = next(row for row in rows if row["row"]["split"] != rows[0]["row"]["split"])
        control = next(c for c in rows[0]["row"]["control_ids"]
                       if c["kind"] == "same_relation_different_subject")
        control["paired_edge_id"] = donor["row"]["edge_id"]
        with self.assertRaisesRegex(ValueError, "invalid matched donor"):
            matched_donors(rows)


if __name__ == "__main__":
    unittest.main()
