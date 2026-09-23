import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("gw_phase1.py")
SPEC = importlib.util.spec_from_file_location("gw_phase1", MODULE_PATH)
gw = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gw)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def row(edge, family, split, relation, site, feature, status="untested", evidence=None):
    return {
        "schema": gw.CENSUS_SCHEMA,
        "edge_id": edge,
        "edge_family_id": family,
        "split": split,
        "cohort": "primary",
        "semantic_edge": {
            "status": "positive",
            "subject": edge + "-subject",
            "relation": relation,
            "target": edge + "-target",
            "target_token_ids": [9],
            "prompt_semantic_family": "declarative",
        },
        "exact_walk_result": {"hits": [{"layer": feature[0], "feature": feature[1]}]},
        "transition_candidate": {
            "sites": [{"layer": site[0], "site": site[1]}],
            "feature_addresses": [{"layer": feature[0], "feature": feature[1]}],
        },
        "causal_status": status,
        "causal_evidence_ids": evidence or [],
        "provenance": {
            "model_identity": "sha256:model",
            "container_identity": "sha256:container",
            "execution_fingerprint": "exec-v1",
            "basis_identity": "basis-v1",
            "run_record_hash": "sha256:" + "1" * 64,
            "position": 3,
        },
        "artifacts": [],
    }


class PhaseOneTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()
        self.sites = self.root / "sites.json"
        self.sites.write_text(json.dumps(["1:attention", "2:ffn"]), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def seal(self, rows):
        census = self.root / "census.jsonl"
        manifest = self.root / "manifest.json"
        write_jsonl(census, rows)
        gw.seal(census, self.artifacts, self.sites, manifest)
        return manifest

    def test_seal_rejects_edge_family_leakage(self):
        rows = [
            row("a", "same", "train", "capital", (1, "attention"), (1, 4)),
            row("b", "same", "test", "capital", (1, "attention"), (1, 4)),
        ]
        with self.assertRaisesRegex(gw.GwError, "crosses"):
            self.seal(rows)

    def test_seal_rejects_observational_claim_promoted_to_causal(self):
        candidate = row("a", "a", "train", "capital", (1, "attention"), (1, 4))
        candidate["causal_status"] = "supported"
        with self.assertRaisesRegex(gw.GwError, "requires causal evidence"):
            self.seal([candidate])

    def test_sealed_bundle_detects_artifact_mutation(self):
        payload = self.artifacts / "carrier-delta.f32"
        payload.write_bytes(b"\x00\x00\x80?")
        candidate = row("a", "a", "train", "capital", (1, "attention"), (1, 4))
        candidate["artifacts"] = [
            {
                "kind": "carrier_delta",
                "path": payload.name,
                "sha256": gw.sha256_file(payload),
                "dtype": "f32-le",
                "shape": [1],
            }
        ]
        manifest = self.seal([candidate])
        sealed, rows = gw.load_sealed(manifest)
        self.assertEqual(sealed["census"]["rows"], 1)
        self.assertEqual(rows[0]["edge_id"], "a")
        payload.write_bytes(b"\x00\x00\x00@")
        with self.assertRaisesRegex(gw.GwError, "sealed artifact changed"):
            gw.load_sealed(manifest)

    def test_gw1_uses_relation_only_and_emits_the_full_curve(self):
        rows = [
            row("a", "a", "train", "capital", (1, "attention"), (1, 4)),
            row("b", "b", "train", "capital", (1, "attention"), (1, 5)),
            row("c", "c", "test", "capital", (1, "attention"), (1, 6)),
            row("d", "d", "test", "part-of", (2, "ffn"), (2, 7)),
        ]
        manifest = self.seal(rows)
        report = gw.gw1(manifest, "train", "test", self.root / "gw1.json")
        self.assertEqual(report["predictors"], ["semantic_edge.relation"])
        self.assertEqual(len(report["curve"]), 3)
        self.assertEqual(report["curve"][1]["transition_candidate_retention"], 0.5)
        self.assertEqual(report["rankings"]["capital"], ["1:attention", "2:ffn"])
        self.assertEqual(report["curve"][-1]["transition_candidate_retention"], 0.5)

    def test_gw3a_keeps_the_four_recall_surfaces_separate(self):
        rows = [
            row("a", "a", "test", "capital", (1, "attention"), (1, 4), "supported", ["sealed-intervention-1"]),
            row("b", "b", "test", "capital", (1, "attention"), (1, 5)),
        ]
        manifest = self.seal(rows)
        lookups = self.root / "lookups.jsonl"
        write_jsonl(
            lookups,
            [
                {
                    "schema": gw.GW3A_INPUT_SCHEMA,
                    "edge_id": edge,
                    "operating_point": "source-k=8/unmasked",
                    "candidate_features": [{"layer": 1, "feature": 4}],
                    "promoted_token_ids": [9],
                    "costs": {
                        "candidate_features": 1,
                        "total_features": 100,
                        "gate_bytes": 0,
                        "other_bytes": 24,
                        "dot_products": 0,
                        "index_bytes": 1200,
                        "wall_time_ns": None,
                    },
                }
                for edge in ("a", "b")
            ],
        )
        report = gw.gw3a(manifest, lookups, self.root / "gw3a.json")
        result = report["operating_points"][0]
        self.assertEqual(result["exact_walk_recall"]["value"], 0.5)
        self.assertEqual(result["exact_walk_edge_retention"]["value"], 0.5)
        self.assertEqual(result["transition_candidate_recall"]["value"], 0.5)
        self.assertEqual(result["transition_candidate_edge_retention"]["value"], 0.5)
        self.assertEqual(result["supported_causal_recall"]["value"], 1.0)
        self.assertEqual(result["supported_causal_edge_retention"]["value"], 1.0)
        self.assertEqual(result["semantic_target_recall"]["value"], 1.0)
        self.assertEqual(result["mean_candidate_fraction"], 0.01)


if __name__ == "__main__":
    unittest.main()
