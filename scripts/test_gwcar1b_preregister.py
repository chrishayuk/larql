"""CAR-1B freshness and draft refusal checks using synthetic row metadata."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwcar1b_preregister import (  # noqa: E402
    check_fresh_rows, excluded_subjects, expected_document, validate, write_new,
)


def fresh_rows():
    return [{"row_id": "fresh-v", "subject_id": "AAA", "relation": "capital",
             "template": "canonical", "split": "validation"},
            {"row_id": "fresh-t", "subject_id": "BBB", "relation": "capital",
             "template": "canonical", "split": "test"}]


class ProtocolTests(unittest.TestCase):
    def test_old_confirmation_rows_cannot_be_relabelled_fresh(self):
        rows = fresh_rows()
        rows[0]["subject_id"] = sorted(excluded_subjects())[0]
        with self.assertRaisesRegex(ValueError, "previously studied"):
            check_fresh_rows(rows, excluded_subjects())

    def test_cross_split_identity_duplicate_row_and_noncanonical_id_refuse(self):
        rows = fresh_rows()
        rows[1]["subject_id"] = rows[0]["subject_id"]
        with self.assertRaisesRegex(ValueError, "crosses"):
            check_fresh_rows(rows, set())
        rows = fresh_rows()
        rows[1]["row_id"] = rows[0]["row_id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            check_fresh_rows(rows, set())
        rows = fresh_rows()
        rows[0]["subject_id"] = "aaa"
        with self.assertRaisesRegex(ValueError, "canonical"):
            check_fresh_rows(rows, set())

    def test_metadata_check_never_authorizes_execution(self):
        result = check_fresh_rows(fresh_rows(), set())
        self.assertTrue(result["metadata_freshness_pass"])
        self.assertFalse(result["population_qualified"])
        self.assertFalse(result["model_execution_authorized"])

    def test_draft_hash_and_tampering_and_overwrite_refusals(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "draft.json"
            write_new(path)
            result = validate(path)
            self.assertFalse(result["ready_to_seal"])
            self.assertFalse(result["model_execution_authorized"])
            with self.assertRaises(FileExistsError):
                write_new(path)
            doc = json.loads(path.read_text())
            doc["model_execution_authorized"] = True
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "differs"):
                validate(path)

    def test_machine_grid_and_simultaneous_family_match_design(self):
        document = expected_document()
        self.assertEqual([c["candidate_id"] for c in document["source_grid"]], list(range(20)))
        self.assertEqual(document["analysis"]["family_readouts_per_split"], 104)
        self.assertEqual(document["training"]["rows"], 396)
        self.assertEqual(document["training"]["subjects"], 44)
        self.assertEqual(len(document["confirmation"]["excluded_subject_ids"]), 127)
        self.assertFalse(document["confirmation"]["population_bound"])


if __name__ == "__main__":
    unittest.main()
