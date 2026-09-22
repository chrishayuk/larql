import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts import gw4b_preregister as prereg


class Gw4bPreregistrationTests(unittest.TestCase):
    def test_frozen_contract_validates(self) -> None:
        path = Path("bench/gw0/gemma3-4b-it-phase1/gw4b-preregistration.json")
        result = prereg.validate(path)
        self.assertEqual(result["eligible_sites"], 231)
        self.assertEqual(result["assessment_rows"], 88)

    def test_identity_change_is_rejected(self) -> None:
        source = Path("bench/gw0/gemma3-4b-it-phase1/gw4b-preregistration.json")
        document = copy.deepcopy(json.loads(source.read_text()))
        document["progression_gate"]["maximum_mean_candidate_fraction"] = 0.5
        with tempfile.NamedTemporaryFile("w", suffix=".json", dir=source.parent, delete=False) as handle:
            json.dump(document, handle)
            path = Path(handle.name)
        try:
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                prereg.validate(path)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
