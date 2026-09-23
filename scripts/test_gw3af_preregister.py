import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("gw3af_preregister.py")
SPEC = importlib.util.spec_from_file_location("gw3af_preregister", MODULE_PATH)
assert SPEC and SPEC.loader
gw3af = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gw3af)


class Gw3afPreregistrationTests(unittest.TestCase):
    def test_hash_excludes_only_the_self_hash(self):
        document = {
            "schema": gw3af.SCHEMA,
            "preregistration_sha256": "old",
            "posting_rule": {"widths": gw3af.WIDTHS},
        }
        original = gw3af.canonical_hash(document)
        document["preregistration_sha256"] = "new"
        self.assertEqual(original, gw3af.canonical_hash(document))
        document["posting_rule"]["widths"] = [1, 2]
        self.assertNotEqual(original, gw3af.canonical_hash(document))

    def test_hash_mismatch_refuses_before_reading_authorities(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prereg.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": gw3af.SCHEMA,
                        "status": "frozen_pre_lookup",
                        "preregistration_sha256": "sha256:wrong",
                        "posting_rule": {"widths": gw3af.WIDTHS},
                        "progression_gate": {
                            "minimum_micro_top100_address_recall": 0.95,
                            "maximum_address_weighted_candidate_coverage": 0.1,
                            "mass_recall_cannot_substitute_for_address_recall": True,
                        },
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "preregistration hash mismatch"):
                gw3af.validate(path)


if __name__ == "__main__":
    unittest.main()
