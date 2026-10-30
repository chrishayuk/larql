"""Population selection and proposal identity checks, without model execution."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwcar1b_population import select  # noqa: E402
from gwcar1b_preregister import ROOT, excluded_subjects  # noqa: E402
from gwv2_population import canonical_hash  # noqa: E402


class PopulationTests(unittest.TestCase):
    def test_source_selection_exclusions_and_order_are_outcome_independent(self):
        source = [{"cca3": f"AA{letter}", "independent": True, "unMember": True,
                   "capital": [f"City {letter}"]} for letter in "ABCDEF"]
        source[4]["capital"] = ["ambiguous", "second"]
        expected = select(source, {"AAF"})
        self.assertEqual({row["cca3"] for row in expected}, {"AAA", "AAB", "AAC", "AAD"})
        self.assertEqual(select(source[::-1], {"AAF"}), expected)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select(source + [source[0]], {"AAF"})

    def test_checked_in_proposal_is_fresh_balanced_and_self_identified(self):
        path = ROOT / "bench/gw-car-1/gemma3-4b-it-phase1/gwcar1b-capital-population.proposal.json"
        doc = json.loads(path.read_text())
        self.assertEqual(doc["proposal_sha256"], canonical_hash(doc, "proposal_sha256"))
        self.assertEqual(doc["counts"]["subjects"], 67)
        self.assertEqual(len(doc["rows"]), 201)
        seen = {}
        for row in doc["rows"]:
            self.assertNotIn(row["subject_id"], excluded_subjects())
            key = row["subject_id"], row["prompt_family"]
            self.assertNotIn(key, seen)
            seen[key] = row["split"]
            self.assertTrue(row["prompt_token_ids"])
            self.assertTrue(row["subject_embedding_token_ids"])
        for subject in {row["subject_id"] for row in doc["rows"]}:
            self.assertEqual(len({split for (s, _), split in seen.items() if s == subject}), 1)
        self.assertFalse(doc["model_execution_authorized"])
        self.assertEqual(doc["model_executions"], 0)


if __name__ == "__main__":
    unittest.main()
