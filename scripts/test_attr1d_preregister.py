import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts import attr1d_preregister as prereg


CONTRACT = Path("bench/attr-1d/attr1d-contract.json")


class Attr1dPreregistrationTests(unittest.TestCase):
    def write_copy(self, document: dict) -> Path:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", dir=CONTRACT.parent, delete=False
        ) as handle:
            json.dump(document, handle)
            return Path(handle.name)

    def validate_mutation(self, document: dict) -> None:
        document["attr1d_contract_sha256"] = prereg.canonical_hash(document)

    def test_frozen_contract_validates(self) -> None:
        result = prereg.validate(CONTRACT)
        self.assertEqual(result["coordinate_kinds"], 4)
        self.assertEqual(result["source_roles"], 5)
        self.assertEqual(result["forbidden_claim_terms"], 14)

    def test_identity_change_is_rejected(self) -> None:
        document = copy.deepcopy(json.loads(CONTRACT.read_text()))
        document["acceptance_gates"]["head_sum_max_relative_l2"] = 1e-3
        path = self.write_copy(document)
        try:
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                prereg.validate(path)
        finally:
            path.unlink()

    def test_semantic_identity_cannot_enter_coordinate(self) -> None:
        document = copy.deepcopy(json.loads(CONTRACT.read_text()))
        document["support_coordinate"]["required_fields"].append("relation")
        self.validate_mutation(document)
        path = self.write_copy(document)
        try:
            with self.assertRaisesRegex(ValueError, "leaked into coordinate"):
                prereg.validate(path)
        finally:
            path.unlink()

    def test_execution_identity_cannot_enter_stable_coordinate(self) -> None:
        document = copy.deepcopy(json.loads(CONTRACT.read_text()))
        document["support_coordinate"]["required_fields"].insert(
            1, "execution_identity"
        )
        self.validate_mutation(document)
        path = self.write_copy(document)
        try:
            with self.assertRaisesRegex(ValueError, "backend realization leaked"):
                prereg.validate(path)
        finally:
            path.unlink()

    def test_attention_weight_cannot_become_source_contribution(self) -> None:
        document = copy.deepcopy(json.loads(CONTRACT.read_text()))
        document["input_authority"]["forbidden_shortcut"] = (
            "attention weight may approximate source contribution"
        )
        self.validate_mutation(document)
        path = self.write_copy(document)
        try:
            with self.assertRaisesRegex(ValueError, "source-attribution boundary"):
                prereg.validate(path)
        finally:
            path.unlink()

    def test_causal_evidence_cannot_enter_descriptive_record(self) -> None:
        document = copy.deepcopy(json.loads(CONTRACT.read_text()))
        document["descriptive_record"]["optional_fields"].append("causal_verdict")
        self.validate_mutation(document)
        path = self.write_copy(document)
        try:
            with self.assertRaisesRegex(ValueError, "causal evidence leaked"):
                prereg.validate(path)
        finally:
            path.unlink()

    def test_causal_claim_word_cannot_be_allowed(self) -> None:
        document = copy.deepcopy(json.loads(CONTRACT.read_text()))
        document["claim_vocabulary"]["allowed"].append("driver")
        self.validate_mutation(document)
        path = self.write_copy(document)
        try:
            with self.assertRaisesRegex(ValueError, "causal vocabulary is allowed"):
                prereg.validate(path)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
