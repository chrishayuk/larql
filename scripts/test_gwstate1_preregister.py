"""Outcome-free tests of the STATE-1 draft/seal validation boundary."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gwstate1_preregister as prereg  # noqa: E402
from gwv2_population import canonical_hash  # noqa: E402


class PreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = prereg.expected_document()

    def validate_copy(self, document):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(document))
            with patch.object(prereg, "expected_document", side_effect=lambda: copy.deepcopy(self.expected)):
                return prereg.validate(path)

    def test_unsealed_draft_validates_without_executable_identity(self):
        result = self.validate_copy(copy.deepcopy(self.expected))
        self.assertEqual(result["status"], prereg.DRAFT_STATUS)
        self.assertIsNone(result["protocol_sha256"])

    def test_changed_context_order_is_refused(self):
        document = copy.deepcopy(self.expected)
        document["interface"]["head_mask_order"] = [2, 0, 3, 4, 5, 6, 7]
        with self.assertRaisesRegex(ValueError, "differs"):
            self.validate_copy(document)

    def test_false_frozen_status_is_refused(self):
        document = copy.deepcopy(self.expected)
        document["status"] = prereg.FROZEN_STATUS
        with self.assertRaisesRegex(ValueError, "executable identity"):
            self.validate_copy(document)

    def test_frozen_status_requires_exact_canonical_identity(self):
        document = copy.deepcopy(self.expected)
        document["status"] = prereg.FROZEN_STATUS
        document["runner"]["executable_sha256"] = "sha256:" + "a" * 64
        document["protocol_sha256"] = canonical_hash(document, "protocol_sha256")
        self.assertEqual(self.validate_copy(document)["protocol_sha256"], document["protocol_sha256"])
        document["heldout"]["proximal_point_retention_min"] = 0.79
        with self.assertRaisesRegex(ValueError, "differs"):
            self.validate_copy(document)


if __name__ == "__main__":
    unittest.main()
