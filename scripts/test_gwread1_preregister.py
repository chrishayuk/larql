from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gwread1_preregister import canonical_hash, validate  # noqa: E402


PROTOCOL = ROOT / "bench/gw-read-1/gwread1-protocol.json"
HUMAN = ROOT / "docs/gw-read-1.md"


class GwRead1ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(PROTOCOL.read_text())

    def write(self, document: dict) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        protocol_dir = root / "bench/gw-read-1"
        docs_dir = root / "docs"
        protocol_dir.mkdir(parents=True)
        docs_dir.mkdir(parents=True)
        (docs_dir / "gw-read-1.md").write_bytes(HUMAN.read_bytes())
        for descriptor in document["authorities"].values():
            if not isinstance(descriptor, dict) or not {"path", "sha256"} <= descriptor.keys():
                continue
            source = (PROTOCOL.parent / descriptor["path"]).resolve()
            target = (protocol_dir / descriptor["path"]).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        path = protocol_dir / "gwread1-protocol.json"
        document["protocol_sha256"] = canonical_hash(document)
        path.write_text(json.dumps(document, indent=2) + "\n")
        return temporary, path

    def rejects(self, mutate) -> None:
        document = copy.deepcopy(self.document)
        mutate(document)
        temporary, path = self.write(document)
        self.addCleanup(temporary.cleanup)
        with self.assertRaises(ValueError):
            validate(path)

    def test_frozen_protocol_validates(self) -> None:
        result = validate(PROTOCOL)
        self.assertEqual(result["status"], "valid")
        self.assertFalse(result["heldout_selection"])
        self.assertTrue(result["dynamic_norms_exposed"])
        self.assertTrue(result["shared_frontier_frozen"])

    def test_selection_must_be_train_only(self) -> None:
        self.rejects(lambda d: d["selection"].__setitem__("population", "validation"))

    def test_both_heldout_splits_adjudicate_components(self) -> None:
        self.rejects(lambda d: d["heldout_component_gate"].__setitem__("splits", ["validation"]))

    def test_k_target_norm_is_declared_dynamic(self) -> None:
        self.rejects(lambda d: d["frozen_interface"]["dynamic_inputs_in_positive_control"].remove("every target natural source K-row norm"))

    def test_static_k_cannot_read_target_norm(self) -> None:
        self.rejects(lambda d: d["READ-1K"].__setitem__("candidate_scale", "target-natural norm"))

    def test_v_cost_classes_are_load_bearing(self) -> None:
        self.rejects(lambda d: d["READ-1V"]["families"][0].__setitem__("cost_class", "cached/materialized"))

    def test_diagnostic_v_cannot_enter_composition(self) -> None:
        self.rejects(lambda d: d["READ-1V"]["families"][0].__setitem__("eligible_for_E", True))

    def test_composition_is_not_selection(self) -> None:
        self.rejects(lambda d: d["READ-1E"].__setitem__("purpose", "joint tuning"))

    def test_q_v_prefixes_share_the_maximum_frontier(self) -> None:
        self.rejects(lambda d: d["cost_model"]["shared_frontier"].__setitem__("original_prompt", "sum Q and V prefix costs"))

    def test_entity_only_prefix_is_not_falsely_shared(self) -> None:
        self.rejects(lambda d: d["cost_model"]["shared_frontier"].__setitem__("entity_only", "share with original prompt"))

    def test_hidden_inputs_remain_a_hard_refusal(self) -> None:
        self.rejects(lambda d: d["hard_refusals"].remove("undeclared natural-state read"))


if __name__ == "__main__":
    unittest.main()
