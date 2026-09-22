from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gwv2_preregister  # noqa: E402
from gwv2_population import validate as validate_population  # noqa: E402


PROTOCOL = ROOT / "bench/gw-v2/gemma3-4b-it-phase1/gwv2-protocol.json"
POPULATION = ROOT / "bench/gw-v2/gemma3-4b-it-phase1/population-manifest.json"


class GwV2ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    def write(self, document: dict) -> tuple[tempfile.TemporaryDirectory, Path, str]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        relative_protocol = PROTOCOL.relative_to(ROOT)
        protocol_path = root / relative_protocol
        protocol_path.parent.mkdir(parents=True)
        descriptors = [document["human_spec"]]
        descriptors.extend(
            descriptor
            for descriptor in document["authorities"].values()
            if isinstance(descriptor, dict) and {"path", "sha256"} <= descriptor.keys()
        )
        for descriptor in descriptors:
            source = (PROTOCOL.parent / descriptor["path"]).resolve()
            target = (protocol_path.parent / descriptor["path"]).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        population = json.loads((protocol_path.parent / "population-manifest.json").read_text())
        source_rows = (POPULATION.parent / population["rows"]["path"]).resolve()
        target_rows = (protocol_path.parent / population["rows"]["path"]).resolve()
        target_rows.write_bytes(source_rows.read_bytes())
        identity = gwv2_preregister.canonical_hash(document)
        document["protocol_sha256"] = identity
        protocol_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return temporary, protocol_path, identity

    def rejects_structure(self, mutate) -> None:
        document = copy.deepcopy(self.document)
        mutate(document)
        temporary, path, identity = self.write(document)
        self.addCleanup(temporary.cleanup)
        with mock.patch.object(gwv2_preregister, "IDENTITY", identity):
            with self.assertRaises(ValueError):
                gwv2_preregister.validate(path)

    def test_frozen_protocol_validates(self) -> None:
        result = gwv2_preregister.validate(PROTOCOL)
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["cells"], 35)
        self.assertEqual(result["fresh_subjects"], 74)
        self.assertFalse(result["heldout_selection"])
        self.assertEqual(result["model_prompts_executed"], 0)
        self.assertFalse(result["fitting_performed"])

    def test_frozen_population_validates(self) -> None:
        population = validate_population(POPULATION)
        self.assertEqual(population["counts"]["executions"], 666)
        self.assertEqual(population["freshness"]["selected_old_identity_intersection"], [])

    def test_grid_cannot_expand_adaptively(self) -> None:
        self.rejects_structure(lambda d: d["grid"]["source_depths"].append(22))

    def test_no_primary_cell_or_winner(self) -> None:
        self.rejects_structure(lambda d: d["grid"].__setitem__("winner", "layer12-rank32"))

    def test_fit_population_is_train_subjects_only(self) -> None:
        self.rejects_structure(lambda d: d["predictor"].__setitem__("fit_population", "all subjects"))

    def test_reconstruction_cannot_become_claim_metric(self) -> None:
        self.rejects_structure(lambda d: d["estimand"].__setitem__("reconstruction", "primary gate"))

    def test_uncertainty_clusters_by_subject(self) -> None:
        self.rejects_structure(lambda d: d["analysis"]["bootstrap"].__setitem__("unit", "execution row"))

    def test_frontier_requires_a_contiguous_region(self) -> None:
        self.rejects_structure(lambda d: d["early_frontier"].__setitem__("shape", "one passing cell"))

    def test_natural_context_remains_explicit(self) -> None:
        self.rejects_structure(
            lambda d: d["frozen_interface"]["dynamic_natural_inputs"].remove(
                "natural L24 carrier-before"
            )
        )

    def test_efficiency_visualization_cannot_select(self) -> None:
        self.rejects_structure(
            lambda d: d["cost_model"].__setitem__(
                "efficiency_visualization_use", "select the best cell"
            )
        )


if __name__ == "__main__":
    unittest.main()
