import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts import gwts1_preregister as prereg


PROTOCOL = Path("bench/gw-ts-1/gwts1-protocol.json")
DEPENDENCIES = Path("bench/gw-ts-1/programme-dependencies.json")


class Gwts1PreregistrationTests(unittest.TestCase):
    def write_copy(self, document: dict) -> Path:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", dir=PROTOCOL.parent, delete=False
        ) as handle:
            json.dump(document, handle)
            return Path(handle.name)

    def test_frozen_contract_validates(self) -> None:
        result = prereg.validate(PROTOCOL)
        self.assertEqual(result["controls"], 8)
        self.assertEqual(result["candidate_points"], 5)

    def test_dependency_freeze_validates(self) -> None:
        protocol = prereg.validate(PROTOCOL)
        result = prereg.validate_dependencies(
            DEPENDENCIES, protocol["protocol_sha256"]
        )
        self.assertEqual(result["status"], "valid")

    def test_identity_change_is_rejected(self) -> None:
        document = copy.deepcopy(json.loads(PROTOCOL.read_text()))
        document["progression_gate"]["minimum_held_out_path_coverage"] = 0.5
        path = self.write_copy(document)
        try:
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                prereg.validate(path)
        finally:
            path.unlink()

    def test_transition_identity_cannot_absorb_prompt_family(self) -> None:
        document = copy.deepcopy(json.loads(PROTOCOL.read_text()))
        document["ontology"]["transition_identity"]["required"].append(
            "prompt_family_identity"
        )
        document["protocol_sha256"] = prereg.canonical_hash(document)
        path = self.write_copy(document)
        try:
            with self.assertRaisesRegex(ValueError, "realization-specific"):
                prereg.validate(path)
        finally:
            path.unlink()

    def test_contribution_cannot_become_coordinate_identity(self) -> None:
        document = copy.deepcopy(json.loads(PROTOCOL.read_text()))
        event = document["ontology"]["support_event"]
        event["measurement_fields"].remove("signed_contribution")
        event["coordinate_fields"].append("signed_contribution")
        document["protocol_sha256"] = prereg.canonical_hash(document)
        path = self.write_copy(document)
        try:
            with self.assertRaisesRegex(ValueError, "measurements leaked"):
                prereg.validate(path)
        finally:
            path.unlink()

    def test_attr_1c_cannot_become_a_gwts1_prerequisite(self) -> None:
        document = copy.deepcopy(json.loads(DEPENDENCIES.read_text()))
        document["observational_branch"].insert(
            3, ["attr_1c", "population_transition_support_capture"]
        )
        document["dependency_sha256"] = prereg.canonical_identity(
            document, "dependency_sha256"
        )
        path = self.write_copy(document)
        try:
            with self.assertRaisesRegex(ValueError, "observational dependency"):
                prereg.validate_dependencies(
                    path, document["gwts1_protocol_sha256"]
                )
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
