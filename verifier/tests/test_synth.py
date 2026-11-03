"""Tests for the synthetic container writer, and reader-vs-writer tests including every corrupted variant."""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import _mini  # noqa: E402
from gguf import GGMLQuantizationType, dequantize as gguf_dequantize  # noqa: E402
from gguf.quants import Q8_0 as GGUF_Q8_0  # noqa: E402

from refcheck import container as C  # noqa: E402
from refcheck import q8_0, synth  # noqa: E402


def bits(a):
    return q8_0.f32_bits(a)


class Encoders(unittest.TestCase):
    def test_bf16_rounding_cases(self):
        f = lambda *v: np.array(v, dtype=np.float32)  # noqa: E731
        cases = {
            1.0: 0x3F80,
            -2.0: 0xC000,
            0.0: 0x0000,
            1 + 2**-8: 0x3F80,  # tie, rounds to even (down)
            1 + 3 * 2**-8: 0x3F82,  # tie, rounds to even (up)
            1 + 2**-8 + 2**-12: 0x3F81,  # above the tie
            3.4028235e38: 0x7F80,  # max f32 rounds up to inf in bf16
        }
        for v, want in cases.items():
            got = int.from_bytes(synth.bf16_bytes(f(v)), "little")
            self.assertEqual(got, want, v)
        self.assertEqual(int.from_bytes(synth.bf16_bytes(f(-0.0)), "little"), 0x8000)
        nan = synth.bf16_bytes(np.array([0x7FC00001], dtype=np.uint32).view(np.float32))
        self.assertTrue(np.isnan(synth.bf16_widen(nan))[0])

    def test_bf16_widen_roundtrip(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal(4096).astype(np.float32)
        w = synth.bf16_widen(synth.bf16_bytes(a))
        self.assertTrue((bits(w) & 0xFFFF == 0).all())
        self.assertTrue((np.abs(w - a) <= np.abs(a) * 2**-8).all())

    def test_q8_ref_encoder_rounds_half_away_from_zero(self):
        # amax 127 gives d = 1, so q = roundf(x): ties go away from zero (numpy would give 0, 2, 2, -0, -2)
        x = np.zeros(32, dtype=np.float32)
        x[:8] = [127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 126.5]
        got = np.frombuffer(synth.ref_quantize_q8_0(x), dtype=np.uint8)
        self.assertEqual(got[:2].tolist(), [0x00, 0x3C])  # d = 1.0 as f16
        self.assertEqual(got[2:10].view(np.int8).tolist(), [127, 1, 2, 3, -1, -2, -3, 127])

    def test_q8_ref_encoder_amax_zero(self):
        got = np.frombuffer(synth.ref_quantize_q8_0(np.zeros(32, np.float32)), dtype=np.uint8)
        self.assertEqual(got.tolist(), [0] * 34)

    def test_q8_ref_encoder_matches_gguf_py_bit_for_bit(self):
        rng = np.random.default_rng(2)
        sets = [
            (rng.standard_normal((2000, 32)) * s).astype(np.float32)
            for s in (1e-8, 1e-5, 1e-3, 0.05, 1.0, 300.0, 3e4)
        ]
        # forced near-ties: scaled integers plus one half
        ties = (rng.integers(-60, 60, size=(500, 32)) + 0.5).astype(np.float32)
        ties[:, 0] = 127
        sets.append(ties)
        for x in sets:
            want = GGUF_Q8_0.quantize_blocks(x.copy())
            got = np.frombuffer(synth.ref_quantize_q8_0(x), dtype=np.uint8).reshape(-1, 34)
            self.assertTrue((got == want).all(), f"{(got != want).any(axis=1).sum()} blocks differ")


class RoundTrip(unittest.TestCase):
    def setUp(self):
        self._tmp = _mini.tmpdir()
        self.root = self._tmp.name
        self.model = synth.tiny_model(np.random.default_rng(3), layers=12)

    def tearDown(self):
        self._tmp.cleanup()

    def test_profile_counts_and_no_warnings(self):
        w = synth.write_container(self.root, self.model)
        c = C.open_container(self.root)
        self.assertEqual(c.warnings, [])
        self.assertEqual(len(self.model), 12 * 11 + 3)
        self.assertEqual(len(c.hf_key_map()), 12 * 11 + 3)
        self.assertEqual(sorted(c.hf_key_map()), sorted(self.model))
        self.assertEqual(len(w.q8_keys), 12 * 7)
        self.assertEqual(sum(r.dtype == "Q8_0" for r in c.hf_key_map().values()), 12 * 7)
        self.assertEqual(sum(r.dtype == "BF16" for r in c.hf_key_map().values()), 12 * 4 + 3)

    def test_representation_selection(self):
        synth.write_container(self.root, self.model)
        c = C.open_container(self.root)
        self.assertEqual(c.representation_for("target.decoder_stack"), "target.decoder_stack@Q8_0")
        for o in ("target.embedding", "target.final_norm", "target.output_head"):
            self.assertEqual(c.representation_for(o), o + "@BF16")
        self.assertEqual(sorted(c.reps), sorted(["target.decoder_stack@BF16", "target.decoder_stack@Q8_0", "target.embedding@BF16", "target.final_norm@BF16", "target.output_head@BF16"]))

    def test_table_order_is_byte_order(self):
        synth.write_container(self.root, self.model)
        c = C.open_container(self.root)
        names = [t.name for t in c.segments["target.decoder_stack@Q8_0"].tensors]
        self.assertEqual(names, sorted(names, key=lambda n: n.encode()))
        layers = []
        for n in names:
            L = n.split(".")[0]
            if not layers or layers[-1] != L:
                layers.append(L)
        self.assertEqual(layers, ["0", "1", "10", "11", "2", "3", "4", "5", "6", "7", "8", "9"])

    def test_bf16_tensors_decode_to_the_rounded_source_exactly(self):
        w = synth.write_container(self.root, self.model)
        c = C.open_container(self.root)
        for key, ref in c.hf_key_map().items():
            if ref.dtype == "BF16":
                self.assertTrue((bits(c.decode(ref)) == bits(w.source[key])).all(), key)
                self.assertEqual(c.decode(ref).shape, self.model[key].shape)

    def test_q8_tensors_match_gguf_py_dequantize_bit_for_bit(self):
        w = synth.write_container(self.root, self.model)
        c = C.open_container(self.root)
        for key in w.q8_keys:
            ref = c.hf_key_map()[key]
            got = c.decode(ref)
            rows, k = int(np.prod(ref.shape[:-1])), ref.shape[-1]
            blocks = np.frombuffer(w.q8_bytes[key], dtype=np.uint8).reshape(rows, k // 32 * 34)
            want = gguf_dequantize(blocks, GGMLQuantizationType.Q8_0)
            self.assertEqual(got.shape, tuple(ref.shape))
            self.assertTrue((bits(got) == bits(want)).all(), key)
            self.assertEqual(c.read_bytes(ref), w.q8_bytes[key])

    def test_q8_error_is_within_half_a_step_of_the_source(self):
        # sanity check on the encoder and layout together (the real T2 machinery is a later deliverable)
        w = synth.write_container(self.root, self.model)
        c = C.open_container(self.root)
        for key in w.q8_keys:
            ref = c.hf_key_map()[key]
            err = np.abs(c.decode(ref).astype(np.float64) - w.source[key].astype(np.float64))
            self.assertLess(err.max(), np.abs(w.source[key]).max() / 127 * 0.51 + 1e-6, key)

    def test_canonical_bf16_view_of_compiled_object(self):
        w = synth.write_container(self.root, self.model)
        c = C.open_container(self.root)
        key = "model.layers.3.mlp.down_proj.weight"
        ref = c.locate("target.decoder_stack", "3.mlp.down_proj.weight", representation="BF16")
        self.assertEqual(ref.dtype, "BF16")
        self.assertTrue((bits(c.decode(ref)) == bits(w.source[key])).all())

    def test_tied_embedding_facts(self):
        synth.write_container(self.root, self.model)
        r = C.open_container(self.root).tied_report()
        self.assertTrue(r["separate_objects"])
        self.assertTrue(r["payload_sha256_equal"])
        self.assertEqual(r["target.output_head_executed_representation"], "target.output_head@BF16")

    def test_untied_head_is_reported_as_different(self):
        m = dict(self.model)
        m["lm_head.weight"] = m["lm_head.weight"] + np.float32(1.0)
        synth.write_container(self.root, m)
        self.assertFalse(C.open_container(self.root).tied_report()["payload_sha256_equal"])

    def test_embedding_and_head_can_be_compiled_too(self):
        # the reader is generic: a policy that includes them gives them Q8_0 packs, and section 7 then selects the pack
        policy = lambda k, s: len(s) == 2  # noqa: E731
        w = synth.write_container(self.root, self.model, policy=policy)
        c = C.open_container(self.root)
        self.assertEqual(c.representation_for("target.embedding"), "target.embedding@Q8_0")
        self.assertEqual(c.representation_for("target.output_head"), "target.output_head@Q8_0")
        ref = c.hf_key_map()["lm_head.weight"]
        self.assertEqual(ref.dtype, "Q8_0")
        self.assertEqual(c.decode(ref).shape, self.model["lm_head.weight"].shape)
        self.assertIn("lm_head.weight", w.q8_keys)

    def test_row_with_length_not_a_multiple_of_32_stays_bf16(self):
        m = {k: v for k, v in self.model.items() if k.startswith("model.layers.0.") or k in ("model.norm.weight", "model.embed_tokens.weight", "lm_head.weight")}
        m["model.layers.0.mlp.up_proj.weight"] = np.ones((5, 48), dtype=np.float32)
        w = synth.write_container(self.root, m)
        c = C.open_container(self.root)
        self.assertNotIn("model.layers.0.mlp.up_proj.weight", w.q8_keys)
        self.assertEqual(c.hf_key_map()["model.layers.0.mlp.up_proj.weight"].dtype, "BF16")

    def test_hand_built_blocks_decode_through_the_container(self):
        key = "model.layers.0.self_attn.k_proj.weight"
        blocks = b"".join(
            [
                synth.q8_block(0x3C00, [127, -128, 0]),
                synth.q8_block(0x7BFF, [1, -1]),
                synth.q8_block(0x0001, [127, -128]),
                synth.q8_block(0x8000, [1, -1, 0]),
            ]
        )
        m = {k: v for k, v in self.model.items() if k.startswith("model.layers.0.") or k in ("model.norm.weight", "model.embed_tokens.weight", "lm_head.weight")}
        m[key] = np.zeros((2, 64), dtype=np.float32)
        w = synth.write_container(self.root, m, q8_override={key: blocks})
        c = C.open_container(self.root)
        got = c.decode(c.hf_key_map()[key])
        self.assertEqual(got.shape, (2, 64))
        self.assertEqual(got[0, :3].tolist(), [127.0, -128.0, 0.0])
        self.assertEqual(got[0, 32:34].tolist(), [65504.0, -65504.0])
        self.assertEqual(float(got[1, 0]), 127 * 2.0**-24)
        self.assertEqual(bits(got[1, 32:35]).tolist(), [0x80000000, 0x00000000, 0x80000000])
        self.assertEqual(w.q8_bytes[key], blocks)

    def test_writer_is_deterministic(self):
        synth.write_container(self.root, self.model)
        other = _mini.tmpdir()
        try:
            synth.write_container(other.name, self.model)
            for rel in ["index.json", "system_graph.json"] + [f"segments/{f}" for f in sorted(os.listdir(os.path.join(self.root, "segments")))]:
                with open(os.path.join(self.root, rel), "rb") as a, open(os.path.join(other.name, rel), "rb") as b:
                    self.assertEqual(a.read(), b.read(), rel)
        finally:
            other.cleanup()


class Faults(unittest.TestCase):
    """Every fault in synth.FAULTS: the reader rejects it with the expected reason, or accepts the control."""

    @classmethod
    def setUpClass(cls):
        cls.model = synth.tiny_model(np.random.default_rng(4), layers=2)

    def test_catalogue_has_unique_names_and_enough_rows(self):
        names = [f.name for f in synth.FAULTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertGreaterEqual(sum(f.expect is not None for f in synth.FAULTS), 50)
        self.assertGreaterEqual(sum(f.expect is None for f in synth.FAULTS), 8)

    def test_every_fault(self):
        for fault in synth.FAULTS:
            with self.subTest(fault=fault.name):
                tmp = _mini.tmpdir()
                try:
                    synth.write_container(tmp.name, self.model, hooks=fault.hooks)
                    if fault.expect is None:
                        C.open_container(tmp.name)
                    else:
                        with self.assertRaises(C.ContainerError) as cm:
                            C.open_container(tmp.name)
                        self.assertIn(fault.expect, str(cm.exception))
                finally:
                    tmp.cleanup()

    def test_unsorted_control_warns(self):
        f = next(f for f in synth.FAULTS if f.name == "control-unsorted-table")
        tmp = _mini.tmpdir()
        try:
            synth.write_container(tmp.name, self.model, hooks=f.hooks)
            self.assertTrue(any("not sorted" in w for w in C.open_container(tmp.name).warnings))
        finally:
            tmp.cleanup()

    def test_resealed_payload_change_is_accepted_by_reader_and_changes_the_values(self):
        # the reader cannot see it; the decoded values differ from a clean container's
        f = next(f for f in synth.FAULTS if f.name == "control-payload-change-resealed")
        a, b = _mini.tmpdir(), _mini.tmpdir()
        try:
            synth.write_container(a.name, self.model)
            synth.write_container(b.name, self.model, hooks=f.hooks)
            ca, cb = C.open_container(a.name), C.open_container(b.name)
            diff = [k for k, r in ca.hf_key_map().items() if not (bits(ca.decode(r)) == bits(cb.decode(cb.hf_key_map()[k]))).all()]
            self.assertEqual(len(diff), 1)
        finally:
            a.cleanup()
            b.cleanup()

    def test_transposed_shape_control_is_framing_valid_but_changes_the_decoded_shape(self):
        f = next(f for f in synth.FAULTS if f.name == "control-q8-shape-transposed")
        tmp = _mini.tmpdir()
        try:
            synth.write_container(tmp.name, self.model, hooks=f.hooks)
            c = C.open_container(tmp.name)
            shapes = {k: tuple(r.shape) for k, r in c.hf_key_map().items() if r.dtype == "Q8_0"}
            wrong = [k for k, s in shapes.items() if s != tuple(self.model[k].shape)]
            self.assertEqual(len(wrong), 1)
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
