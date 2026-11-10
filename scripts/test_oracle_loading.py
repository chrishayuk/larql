"""Unit tests for the oracle loading gate (oracle_loading.py)."""

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import oracle_loading as o  # noqa: E402


PLAN = {
    "graph": {
        "components": [{"id": "target", "role": "primary_text"}, {"id": "vision", "role": "perception"}],
        "objects": [
            {"id": "target.embedding", "component": "target",
             "source_bindings": [{"tensor_prefix": "model.embed_tokens"}]},
            {"id": "target.decoder_stack", "component": "target",
             "source_bindings": [{"tensor_prefix": "model.layers"}]},
            {"id": "vision.perception_tower", "component": "vision",
             "source_bindings": [{"tensor_prefix": "vision_tower.vision_model"}]},
            {"id": "vision.perception_adapter", "component": "vision",
             "source_bindings": [{"tensor_prefix": "multi_modal_projector"}]},
        ],
    }
}
BOUNDARY = o.capability_boundary(PLAN)


def facts(missing=(), unexpected=(), mismatched=(), tied=True, checkpoint=None):
    return o.loading_facts(
        boundary=BOUNDARY,
        model_class="Gemma3ForCausalLM",
        expected_keys=["model.embed_tokens.weight", "lm_head.weight"],
        checkpoint_keys=checkpoint if checkpoint is not None else ["model.embed_tokens.weight", *unexpected],
        missing=missing,
        unexpected=unexpected,
        mismatched=mismatched,
        tie_word_embeddings=tied,
        key_mapping=None,
    )


class GateTests(unittest.TestCase):
    def test_a_tied_head_may_be_missing(self):
        self.assertEqual(facts(missing=["lm_head.weight"])["gate"], "pass")

    def test_an_untied_missing_head_fails(self):
        self.assertEqual(facts(missing=["lm_head.weight"], tied=False)["gate"], "fail")

    def test_any_other_missing_key_fails(self):
        self.assertEqual(facts(missing=["model.layers.0.mlp.up_proj.weight"])["gate"], "fail")

    def test_the_boundary_comes_from_the_plan_graph(self):
        self.assertEqual(set(BOUNDARY["allowlist"]), {"vision_tower.vision_model", "multi_modal_projector"})
        self.assertEqual(set(BOUNDARY["in_closure_prefixes"]), {"model.embed_tokens", "model.layers"})

    def test_an_unexpected_key_outside_the_allowlist_fails_even_off_the_language_model(self):
        record = facts(unexpected=["audio_tower.x"])
        self.assertEqual(record["gate"], "fail")
        self.assertEqual(record["unexpected_not_allowlisted"], ["audio_tower.x"])

    def test_the_cross_check_names_disagreements_both_ways(self):
        record = facts(
            unexpected=["model.layers.0.w"],
            checkpoint=["model.layers.0.w", "vision_tower.vision_model.loaded", "stray.tensor"],
        )
        check = record["boundary_cross_check"]
        self.assertEqual(check["out_of_closure_but_loaded"], ["vision_tower.vision_model.loaded"])
        self.assertEqual(check["in_closure_but_unloaded"], ["model.layers.0.w"])
        self.assertEqual(check["bound_to_no_component"], ["stray.tensor"])

    def test_unexpected_tower_keys_pass_and_are_listed(self):
        keys = ["vision_tower.vision_model.a", "multi_modal_projector.b"]
        record = facts(unexpected=keys)
        self.assertEqual(record["gate"], "pass")
        self.assertEqual(record["unexpected"], sorted(keys))
        self.assertEqual(
            record["unexpected_by_prefix"],
            {"multi_modal_projector.b": 1, "vision_tower.vision_model": 1},
        )

    def test_an_unexpected_language_model_key_fails(self):
        record = facts(unexpected=["model.layers.0.self_attn.q_proj.bias"])
        self.assertEqual(record["gate"], "fail")
        self.assertEqual(len(record["unexpected_inside_language_model"]), 1)

    def test_prefixes_match_whole_segments(self):
        self.assertFalse(o.under("vision_towerish.x", ["vision_tower"]))
        self.assertTrue(o.under("vision_tower.x", ["vision_tower"]))

    def test_shape_mismatches_fail(self):
        self.assertEqual(facts(mismatched=[("w", (1,), (2,))])["gate"], "fail")


class CheckpointKeyTests(unittest.TestCase):
    def test_header_keys_skip_metadata(self):
        header = json.dumps({"__metadata__": {}, "a": {}, "b": {}}).encode()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.safetensors"
            path.write_bytes(struct.pack("<Q", len(header)) + header)
            self.assertEqual(o.safetensors_header_keys(path), ["a", "b"])

    def test_index_keys(self):
        self.assertEqual(o.index_keys({"weight_map": {"a": "1", "b": "2"}}), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
