"""Reader tests against a hand-assembled tiny container (tests/_mini.py)."""

import dataclasses
import os
import struct
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import _mini  # noqa: E402
from refcheck import container as C  # noqa: E402
from refcheck import q8_0  # noqa: E402


def hedit(old: bytes, new: bytes):
    """Segment transform: replace text inside the JSON header and re-frame it (length prefix and 16-byte padding)."""

    def f(b: bytes) -> bytes:
        (h,) = struct.unpack("<Q", b[:8])
        text = b[8 : 8 + h].rstrip(b" ")
        assert old in text, old
        text = text.replace(old, new)
        pad = (16 - (8 + len(text)) % 16) % 16
        hdr = text + b" " * pad
        return struct.pack("<Q", len(hdr)) + hdr + b[8 + h :]

    return f


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = _mini.tmpdir()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def build(self, **mut):
        return _mini.build(self.root, **mut)

    def assertRejected(self, fragment, **mut):
        self.build(**mut)
        with self.assertRaises(C.ContainerError) as cm:
            C.open_container(self.root)
        self.assertIn(fragment, str(cm.exception))


class Accepts(Base):
    def test_valid_container_opens_without_warnings(self):
        self.build()
        c = C.open_container(self.root)
        self.assertEqual(c.warnings, [])
        self.assertEqual(sorted(c.objects), ["target.decoder_stack", "target.embedding", "target.final_norm", "target.output_head"])

    def test_section7_selection_prefers_q8_pack_then_canonical(self):
        self.build()
        c = C.open_container(self.root)
        self.assertEqual(c.representation_for("target.decoder_stack"), "target.decoder_stack@Q8_0")
        self.assertEqual(c.representation_for("target.embedding"), "target.embedding@BF16")
        self.assertEqual(c.representation_for("target.output_head"), "target.output_head@BF16")

    def test_hf_key_map(self):
        self.build()
        m = C.open_container(self.root).hf_key_map()
        self.assertEqual(
            sorted(m),
            [
                "lm_head.weight",
                "model.embed_tokens.weight",
                "model.layers.0.input_layernorm.weight",
                "model.layers.0.mlp.up_proj.weight",
                "model.norm.weight",
            ],
        )
        self.assertEqual(m["model.layers.0.mlp.up_proj.weight"].dtype, "Q8_0")
        self.assertEqual(m["model.layers.0.input_layernorm.weight"].dtype, "BF16")

    def test_per_tensor_dtype_not_pack_encoding_decides_representation(self):
        self.build()
        c = C.open_container(self.root)
        self.assertEqual(c.reps["target.decoder_stack@Q8_0"]["encoding"], "Q8_0")
        self.assertEqual(c.locate("target.decoder_stack", "0.input_layernorm.weight").dtype, "BF16")

    def test_decode_q8_tensor_in_shape_rows_and_blocks(self):
        self.build()
        c = C.open_container(self.root)
        w = c.decode(c.locate("target.decoder_stack", "0.mlp.up_proj.weight"))
        self.assertEqual(w.shape, (2, 64))
        self.assertEqual(w[0, :32].tolist(), [float(q) for q in range(-16, 16)])
        self.assertEqual(w[0, 32:].tolist(), [2.0] * 32)
        self.assertEqual(w[1, 0], -63.5)
        self.assertEqual(w[1, 1], 64.0)
        # last block has scale +0 and code 5 (then zeros): every element is +0
        self.assertEqual(q8_0.f32_bits(w[1, 32:]).tolist(), [0] * 32)

    def test_decode_bf16_exact(self):
        self.build()
        c = C.open_container(self.root)
        n = c.decode(c.locate("target.final_norm", "weight"))
        self.assertEqual(n.shape, (32,))
        self.assertTrue((n == 1.0).all())
        e = c.decode(c.locate("target.embedding", "weight"))
        self.assertEqual(e.shape, (4, 32))
        self.assertEqual(q8_0.f32_bits(e.reshape(-1))[0], 0x3F00 << 16)
        self.assertEqual(q8_0.f32_bits(e.reshape(-1))[127], (0x3F00 + 127) << 16)

    def test_read_canonical_bf16_of_compiled_object(self):
        self.build()
        c = C.open_container(self.root)
        ref = c.locate("target.decoder_stack", "0.mlp.up_proj.weight", representation="BF16")
        self.assertEqual(ref.dtype, "BF16")
        self.assertTrue((c.decode(ref) == 1.0).all())

    def test_absolute_byte_range_matches_4_4(self):
        self.build()
        c = C.open_container(self.root)
        ref = c.locate("target.decoder_stack", "0.mlp.up_proj.weight")
        seg = c.segments["target.decoder_stack@Q8_0"]
        self.assertEqual(ref.abs_start, 8 + seg.header_len + 64)
        self.assertEqual(ref.abs_end - ref.abs_start, 4 * 34)
        self.assertEqual((8 + seg.header_len) % 16, 0)

    def test_tied_report(self):
        self.build()
        r = C.open_container(self.root).tied_report()
        self.assertTrue(r["separate_objects"])
        self.assertTrue(r["payload_sha256_equal"])
        self.assertTrue(r["head_reuses_embedding"])
        self.assertEqual(r["target.output_head_executed_representation"], "target.output_head@BF16")

    def test_unknown_keys_are_ignored(self):
        def idx(i):
            i["something_new"] = {"x": 1}
            i["representations"]["target.embedding@BF16"]["extra"] = 5

        self.build(index=idx)
        C.open_container(self.root)

    def test_missing_tensor_is_an_error_and_never_falls_back(self):
        self.build()
        c = C.open_container(self.root)
        with self.assertRaises(C.ContainerError):
            c.locate("target.embedding", "no.such.weight")
        with self.assertRaises(C.ContainerError):
            c.locate("target.embedding", "weight", representation="Q8_0")

    def test_header_padding_may_be_any_json_whitespace(self):
        def retrim(b):  # rewrite the header with 5 extra trailing spaces (still valid JSON)
            (h,) = struct.unpack("<Q", b[:8])
            return struct.pack("<Q", h + 5) + b[8 : 8 + h] + b" " * 5 + b[8 + h :]

        # payload digest is unchanged by padding, segment digest changes: build with digests unchecked
        self.build(segs={"target.final_norm@BF16": retrim})
        c = C.open_container(self.root, verify_digests=False)
        self.assertEqual(c.decode(c.locate("target.final_norm", "weight")).shape, (32,))
        self.assertTrue(any("not a multiple of 4" in w for w in c.warnings))


class Rejects(Base):
    def test_missing_index(self):
        self.assertRejected("index.json: file is missing", skip_index=True)

    def test_wrong_version(self):
        self.assertRejected("version", index=lambda i: i.update(version=3))

    def test_version_float(self):
        self.assertRejected("version", index=lambda i: i.update(version=4.0))

    def test_missing_version(self):
        self.assertRejected("no version", index=lambda i: i.pop("version"))

    def test_no_system_graph(self):
        self.assertRejected("no system_graph", index=lambda i: i.pop("system_graph"))

    def test_moe_manifest_present(self):
        self.assertRejected("moe_manifest", index=lambda i: i.update(moe_manifest="x.json"))

    def test_auxiliary_references_present(self):
        self.assertRejected("auxiliary_references", index=lambda i: i.update(auxiliary_references="a.json"))

    def test_derived_authority(self):
        self.assertRejected("authority", index=lambda i: i.update(authority="derived"))

    def test_graph_schema(self):
        self.assertRejected("schema", graph=lambda g: g.update(schema=5))

    def test_absolute_and_dotdot_segment_paths(self):
        self.assertRejected("absolute or contains", index=lambda i: i["representations"]["target.embedding@BF16"].update(segment="/etc/passwd"))
        self.assertRejected("absolute or contains", index=lambda i: i["representations"]["target.embedding@BF16"].update(segment="segments/../index.json"))

    def test_graph_path_dotdot(self):
        self.assertRejected("absolute or contains", index=lambda i: i.update(system_graph="../system_graph.json"))

    def test_object_with_two_bindings(self):
        def g(g):
            b = g["objects"][0]["source_bindings"]
            b.append(dict(b[0], tensor_prefix="other"))

        self.assertRejected("exactly one source binding", graph=g)

    # -- D1 OQ12: duplicate ownership of a source key ----------------------------------------------------

    @staticmethod
    def _prefix(oid, prefix):
        def g(g):
            for o in g["objects"]:
                if o["id"] == oid:
                    o["source_bindings"][0]["tensor_prefix"] = prefix

        return g

    def test_two_objects_yielding_the_same_hf_key_are_rejected(self):
        # output head bound to the embedding's prefix: both yield model.embed_tokens.weight
        self.assertRejected("duplicate ownership", graph=self._prefix("target.output_head", "model.embed_tokens"))

    def test_overlapping_prefixes_yielding_the_same_hf_key_are_rejected(self):
        # decoder_stack "0.input_layernorm.weight" under "model.layers" and final_norm "weight" under
        # "model.layers.0.input_layernorm" both give model.layers.0.input_layernorm.weight
        self.assertRejected(
            "model.layers.0.input_layernorm.weight",
            graph=self._prefix("target.final_norm", "model.layers.0.input_layernorm"),
        )

    def test_identical_content_under_different_keys_is_accepted(self):
        self.build()
        c = C.open_container(self.root)
        m = c.hf_key_map()
        self.assertIn("model.embed_tokens.weight", m)
        self.assertIn("lm_head.weight", m)
        self.assertEqual(c.read_bytes(m["model.embed_tokens.weight"]), c.read_bytes(m["lm_head.weight"]))
        self.assertTrue(c.tied_report()["payload_sha256_equal"])

    def test_declared_logical_sharing_with_no_second_tensor_is_accepted(self):
        def idx(i):
            del i["representations"]["target.output_head@BF16"]

        def g(g):
            g["objects"] = [o for o in g["objects"] if o["id"] != "target.output_head"]

        self.build(index=idx, graph=g)
        c = C.open_container(self.root)
        self.assertTrue(c.tied_report()["head_reuses_embedding"])
        self.assertNotIn("lm_head.weight", c.hf_key_map())

    # -- D5 AMB-018: ownership across every representation the index declares -----------------------------

    def _extra_rep(self, c, oid, enc, rename):
        """In-memory synthetic: declare a further representation `<oid>@<enc>` whose tensors are copies of the
        object's canonical one with their names rewritten by `rename`."""
        import copy

        src = c.segments[f"{oid}@BF16"]
        rep_id = f"{oid}@{enc}"
        tensors = [dataclasses.replace(e, name=rename(e.name)) for e in src.tensors]
        seg = dataclasses.replace(copy.copy(src), rep_id=rep_id, tensors=tensors, by_name={e.name: e for e in tensors})
        c.segments[rep_id] = seg
        c.reps[rep_id] = dict(c.reps[f"{oid}@BF16"], encoding=enc)
        return rep_id

    def test_clean_container_passes_the_ownership_check(self):
        self.build()
        C.open_container(self.root).check_unique_ownership()

    def test_collision_between_different_encodings_of_different_objects_is_rejected(self):
        # final_norm@F16 yields model.embed_tokens.weight, which embedding@BF16 owns. The two are never executed together
        # (the F16 pack is not what section 7 selects) and are not a uniform-encoding set; D5 rejects regardless.
        self.build()
        c = C.open_container(self.root)
        c.objects["target.final_norm"].tensor_prefix = "model"
        c.check_unique_ownership()  # model.weight (BF16 pack) still collides with nothing
        self._extra_rep(c, "target.final_norm", "F16", lambda n: "embed_tokens.weight")
        with self.assertRaises(C.ContainerError) as cm:
            c.check_unique_ownership()
        self.assertIn("duplicate ownership", str(cm.exception))
        self.assertIn("model.embed_tokens.weight", str(cm.exception))
        # the executed selection alone is clean, so the rejection comes from the declared-but-unselected pack
        c.hf_key_map()

    def test_collision_in_a_non_executed_representation_pair_is_rejected(self):
        # the decoder stack's BF16 pack is not executed (the Q8_0 pack is) and final_norm@BF16 is; they collide in BF16
        self.build()
        c = C.open_container(self.root)
        c.objects["target.final_norm"].tensor_prefix = "model.layers.0.input_layernorm"
        with self.assertRaises(C.ContainerError):
            c.check_unique_ownership()

    def test_two_tensors_in_one_representation_with_the_same_key_are_rejected(self):
        self.build()
        c = C.open_container(self.root)
        seg = c.segments["target.embedding@BF16"]
        import copy

        seg.tensors.append(copy.copy(seg.tensors[0]))  # same name, same key (the framing check would also refuse it)
        with self.assertRaises(C.ContainerError) as cm:
            c.check_unique_ownership()
        self.assertIn("duplicate ownership", str(cm.exception))

    def test_several_representations_of_one_object_sharing_keys_are_accepted(self):
        self.build()
        c = C.open_container(self.root)
        # decoder_stack has a BF16 and a Q8_0 pack with the same tensor names, hence the same keys
        keys = {r: {c.objects["target.decoder_stack"].tensor_prefix + "." + e.name for e in c.segments[r].tensors}
                for r in ("target.decoder_stack@BF16", "target.decoder_stack@Q8_0")}
        self.assertEqual(keys["target.decoder_stack@BF16"], keys["target.decoder_stack@Q8_0"])
        # and a third pack of the same object, also sharing them
        self._extra_rep(c, "target.decoder_stack", "F16", lambda n: n)
        c.check_unique_ownership()

    def test_identical_content_under_distinct_keys_is_accepted_by_the_ownership_check(self):
        self.build()
        c = C.open_container(self.root)
        m = c.hf_key_map()
        self.assertEqual(c.read_bytes(m["model.embed_tokens.weight"]), c.read_bytes(m["lm_head.weight"]))
        c.check_unique_ownership()

    def test_physically_shared_storage_is_accepted(self):
        # two objects' tensors pointing at the same bytes in the file, under distinct keys
        self.build()
        c = C.open_container(self.root)
        emb, head = c.segments["target.embedding@BF16"], c.segments["target.output_head@BF16"]
        head.tensors[0] = dataclasses.replace(head.tensors[0], offset=emb.tensors[0].offset)
        c.check_unique_ownership()

    def test_ownership_check_runs_on_open(self):
        self.assertRejected("duplicate ownership", graph=self._prefix("target.output_head", "model.embed_tokens"))

    def test_index_rep_names_object_missing_from_graph(self):
        def g(g):
            g["objects"] = [o for o in g["objects"] if o["id"] != "target.final_norm"]

        self.assertRejected("absent from the graph", graph=g)

    def test_graph_object_without_index_entry(self):
        def i(i):
            del i["representations"]["target.final_norm@BF16"]

        self.assertRejected("has no target.final_norm@BF16", index=i)

    def test_graph_claims_q8_but_index_lacks_it(self):
        def i(i):
            del i["representations"]["target.decoder_stack@Q8_0"]

        self.assertRejected("target.decoder_stack@Q8_0", index=i)

    def test_codec_any_field_differs(self):
        for key, val in [("revision", 2), ("group_elems", 64), ("element", "ggml-q4"), ("layout", "x"), ("group_scale", "out"), ("tensor_scale", "t"), ("family", "Q4")]:
            with self.subTest(key=key):
                self._tmp.cleanup()
                self._tmp = _mini.tmpdir()
                self.root = self._tmp.name
                self.assertRejected("codec", index=lambda i, k=key, v=val: i["representations"]["target.decoder_stack@Q8_0"]["codec"].update({k: v}))

    def test_codec_missing_or_float_typed(self):
        self.assertRejected("codec", index=lambda i: i["representations"]["target.decoder_stack@Q8_0"].pop("codec"))
        self._tmp.cleanup()
        self._tmp = _mini.tmpdir()
        self.root = self._tmp.name
        self.assertRejected("codec", index=lambda i: i["representations"]["target.decoder_stack@Q8_0"]["codec"].update(group_elems=32.0))

    def test_tensor_count_mismatch(self):
        self.assertRejected("tensor_count", index=lambda i: i["representations"]["target.embedding@BF16"].update(tensor_count=2))

    def test_payload_bytes_mismatch(self):
        self.assertRejected("payload_bytes", index=lambda i: i["representations"]["target.embedding@BF16"].update(payload_bytes=1))

    def test_digest_mismatch_payload_flip(self):
        def flip(b):
            return b[:-1] + bytes([b[-1] ^ 1])

        self.assertRejected("sha256 mismatch", segs={"target.embedding@BF16": flip})

    def test_digest_mismatch_header_only_edit_changes_segment_hash(self):
        def relabel(b):  # change a shape digit in the header: payload hash intact, segment hash not (section 13.3)
            return b.replace(b'"shape":[4,32]', b'"shape":[8,16]')

        self.assertRejected("segment_sha256 mismatch", segs={"target.embedding@BF16": relabel})

    def test_digest_uppercase_hex_is_not_accepted(self):
        def i(i):
            e = i["representations"]["target.embedding@BF16"]
            e["segment_sha256"] = e["segment_sha256"].upper()

        self.assertRejected("segment_sha256 mismatch", index=i)

    def test_header_representation_differs_from_key(self):
        def swap(b):
            return b.replace(b"target.embedding@BF16", b"target.embedding@BF17")

        self.assertRejected("header.representation", segs={"target.embedding@BF16": swap})

    def test_header_schema(self):
        self.assertRejected("schema", segs={"target.embedding@BF16": lambda b: b.replace(b'"schema":1', b'"schema":2')})

    def test_header_length_over_limit(self):
        self.assertRejected("exceeds 256 MiB", segs={"target.final_norm@BF16": lambda b: struct.pack("<Q", 268435457) + b[8:]})

    def test_header_length_past_eof(self):
        self.assertRejected("past the end", segs={"target.final_norm@BF16": lambda b: struct.pack("<Q", len(b)) + b[8:]})

    def test_truncated_file(self):
        self.assertRejected("file length", segs={"target.embedding@BF16": lambda b: b[:-2]})

    def test_trailing_bytes(self):
        self.assertRejected("file length", segs={"target.embedding@BF16": lambda b: b + b"\x00"})

    def test_shorter_than_prefix(self):
        self.assertRejected("shorter than", segs={"target.final_norm@BF16": lambda b: b[:5]})

    def test_unknown_dtype(self):
        self.assertRejected("not BF16 or Q8_0", segs={"target.final_norm@BF16": hedit(b'"BF16"', b'"F32"')})

    def test_len_disagrees_with_dtype_shape(self):
        self.assertRejected("disagrees", segs={"target.final_norm@BF16": lambda b: b.replace(b'"len":64', b'"len":66')})

    def test_non_contiguous_offsets(self):
        self.assertRejected("contiguity", segs={"target.decoder_stack@Q8_0": lambda b: b.replace(b'"offset":64', b'"offset":66')})

    def test_duplicate_tensor_names(self):
        self.assertRejected(
            "duplicate",
            segs={"target.decoder_stack@BF16": lambda b: b.replace(b"0.mlp.up_proj.weight", b"0.input_layernorm.weight")},
        )

    def test_q8_row_length_not_multiple_of_32(self):
        self.assertRejected("multiple of 32", segs={"target.decoder_stack@Q8_0": lambda b: b.replace(b'"shape":[2,64]', b'"shape":[2,65]')})

    def test_q8_tensor_in_canonical_bf16_representation(self):
        self.assertRejected(
            "non-BF16",
            stack_bf16=[
                ("0.input_layernorm.weight", "BF16", [32], _mini.BF16_NORM),
                ("0.mlp.up_proj.weight", "Q8_0", [2, 64], _mini.Q8_UP),
            ],
        )

    def test_duplicate_json_key(self):
        self.assertRejected("duplicate JSON key", segs={"target.final_norm@BF16": hedit(b'"schema":1,', b'"schema":1,"schema":1,')})

    def test_header_not_json(self):
        self.assertRejected("not valid JSON", segs={"target.final_norm@BF16": lambda b: b[:8] + b[8:].replace(b"{", b"[", 1)})

    def test_index_not_object(self):
        self.build()
        with open(os.path.join(self.root, "index.json"), "w") as f:
            f.write("[]")
        with self.assertRaises(C.ContainerError):
            C.open_container(self.root)

    def test_index_json_nan_constant(self):
        self.build()
        with open(os.path.join(self.root, "index.json"), "w") as f:
            f.write('{"version": NaN}')
        with self.assertRaises(C.ContainerError):
            C.open_container(self.root)

    def test_representation_key_not_its_own_fields(self):
        def i(i):
            i["representations"]["target.embedding@BF16"]["object"] = "target.output_head"

        self.assertRejected("not '<object>@<encoding>'", index=i)


class Warns(Base):
    def test_unsorted_table_is_a_warning_not_a_rejection(self):
        # swap the two names inside the header (same lengths) so the table is unsorted but valid
        def swap(b):
            return b.replace(b"0.input_layernorm.weight", b"@.input_layernorm.weight")

        self.build(segs={"target.decoder_stack@BF16": swap})
        c = C.open_container(self.root, verify_digests=False)
        self.assertTrue(any("not sorted" in w for w in c.warnings))

    def test_source_representation_digest_mismatch_warns(self):
        def i(i):
            i["representations"]["target.decoder_stack@Q8_0"]["source_representation_digest"] = "0" * 64

        self.build(index=i)
        c = C.open_container(self.root)
        self.assertTrue(any("source_representation_digest" in w for w in c.warnings))

    def test_binding_counts_mismatch_warns(self):
        def g(g):
            g["objects"][0]["source_bindings"][0]["tensors"] = 99

        self.build(graph=g)
        c = C.open_container(self.root)
        self.assertTrue(any("binding tensors" in w for w in c.warnings))


if __name__ == "__main__":
    unittest.main()
