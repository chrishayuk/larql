"""`Q8_0_PACK_DERIVATION`: the derivation rule against hand-derived blocks and an exact-rational second route, the
committed engine-compiled containers, and every way a pack can be wrong while its digests are right."""

import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
from fractions import Fraction

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from larql_contract_verifier import cli  # noqa: E402
from larql_contract_verifier.contracts import q8_0_pack as P  # noqa: E402
from refcheck import container as C  # noqa: E402

CONFORMANCE = os.path.join(ROOT, "conformance")
BASE = os.path.join(CONFORMANCE, "structured_patch", "base")
#: Every committed container with a re-derived pack: an unedited base and containers the engine compiled from edits.
CONTAINERS = (
    BASE,
    os.path.join(CONFORMANCE, "structured_patch", "compiled"),
    os.path.join(CONFORMANCE, "weight_patches_v4", "low_rank_k2", "compiled"),
    os.path.join(CONFORMANCE, "weight_patches_v4", "mixed", "compiled"),
    os.path.join(CONFORMANCE, "weight_patches_v4", "sham", "compiled"),
    os.path.join(CONFORMANCE, "weight_patches_v4", "f32_profile", "compiled"),
)
CANONICAL = "target.decoder_stack@BF16"
PACK = "target.decoder_stack@Q8_0"
QUANTISED = "1.mlp.down_proj.weight"  # [64, 256]: 512 blocks
CARRIED = "0.input_layernorm.weight"  # BF16 in the pack too
SAT, VIOL, CNE = "satisfied", "violated", "could_not_evaluate"
BF16_ONE = 0x3F80
BF16_INFINITY = 0x7F80
F16_ONE = 0x3C00
F16_INFINITY = 0x7C00
CORRUPTED_BYTES = 64


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, doc):
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)


def location(root, rep, name):
    seg = C.open_container(root, verify_digests=False).segments[rep]
    e = seg.by_name[name]
    return seg, seg.payload_start + e.offset, e


def patch_bytes(root, rep, name, at, data):
    seg, start, _ = location(root, rep, name)
    with open(seg.path, "r+b") as f:
        f.seek(start + at)
        f.write(data)


def reseal(root, rep):
    """A careful tamperer: make index.json agree with a representation's bytes again. Returns its payload digest."""
    seg = C.open_container(root, verify_digests=False).segments[rep]
    with open(seg.path, "rb") as f:
        data = f.read()
    index = read_json(os.path.join(root, "index.json"))
    entry = index["representations"][rep]
    entry["segment_sha256"] = sha(data)
    entry["payload_sha256"] = sha(data[seg.payload_start:seg.payload_start + seg.payload_bytes])
    write_json(os.path.join(root, "index.json"), index)
    return entry["payload_sha256"]


def bf16(x: float) -> int:
    """The BF16 pattern of a value BF16 holds exactly."""
    (bits,) = struct.unpack("<I", struct.pack("<f", x))
    assert bits & 0xFFFF == 0, f"{x} is not a BF16 value"
    return bits >> 16


def block_of(values) -> np.ndarray:
    assert len(values) == P.BLOCK
    return np.asarray(values, dtype=np.float32)


def derived(values) -> tuple[int, list[int]]:
    out = P.derive_q8_0(block_of(values))[0]
    return int(out[0]) | (int(out[1]) << 8), [int(c) for c in out[2:].view(np.int8)]


def exact(values) -> tuple[int, list[int]]:
    return P.derive_block_exact([Fraction(float(v)) for v in block_of(values)])


class TheRule(unittest.TestCase):
    """§2, on blocks whose results are derived by hand."""

    def test_a_block_whose_largest_magnitude_is_127_has_scale_one_and_its_integers_as_codes(self):
        values = [127.0, -127.0, 1.0, -1.0, 0.0, 64.0] + [3.0] * 26
        self.assertEqual(derived(values), (F16_ONE, [127, -127, 1, -1, 0, 64] + [3] * 26))

    def test_halves_round_away_from_zero(self):
        values = [127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 126.5] + [0.0] * 24
        self.assertEqual(derived(values)[1], [127, 1, 2, 3, -1, -2, -3, 127] + [0] * 24)

    def test_a_value_just_under_a_half_rounds_down(self):
        # The largest binary32 below 0.5: adding 0.5 and truncating would round it up.
        under = float(np.nextafter(np.float32(0.5), np.float32(0)))
        codes = P.derive_q8_0(block_of([127.0, under, -under] + [0.0] * 29))[0][2:].view(np.int8)
        self.assertEqual([int(c) for c in codes[:3]], [127, 0, 0])

    def test_an_all_zero_block_has_a_zero_scale_and_zero_codes_whatever_the_signs(self):
        self.assertEqual(derived([0.0, -0.0] * 16), (0, [0] * 32))

    def test_the_scale_is_stored_to_the_nearest_binary16_ties_to_even(self):
        # amax = 127·(1 + 2^-11): the scale 1 + 2^-11 is halfway between two binary16 values and goes to the even one.
        self.assertEqual(derived([127.0 * (1 + 2.0 ** -11)] + [0.0] * 31)[0], F16_ONE)
        # 1 + 3·2^-11 is halfway between 1 + 2^-10 (odd) and 1 + 2^-9 (even): up.
        self.assertEqual(derived([127.0 * (1 + 3 * 2.0 ** -11)] + [0.0] * 31)[0], F16_ONE + 2)

    def test_a_scale_in_the_binary16_subnormal_range_is_kept_not_flushed(self):
        # amax = 127·2^-20: the scale 2^-20 is a binary16 subnormal, 2^-24 × 16.
        scale_bits, codes = derived([127.0 * 2.0 ** -20, -127.0 * 2.0 ** -20] + [0.0] * 30)
        self.assertEqual(scale_bits, 1 << 4)
        self.assertEqual(codes[:2], [127, -127])

    def test_a_scale_beyond_binary16_is_stored_as_infinity(self):
        self.assertEqual(derived([127.0 * 2.0 ** 17] + [0.0] * 31)[0], F16_INFINITY)

    def test_the_codes_come_from_the_binary32_scale_not_from_the_stored_one(self):
        # 100/127 is not a binary16 value. Dividing by the binary32 scale gives 127 for the largest element; a
        # reader's product with the stored (rounded) scale is another number, and is not what the rule computes.
        scale_bits, codes = derived([100.0, 50.0, -25.0] + [0.0] * 29)
        self.assertEqual(codes[:3], [127, 64, -32])  # 63.5 and −31.75
        self.assertNotEqual(Fraction(100, 127), Fraction(float(np.array([scale_bits], dtype="<u2").view(np.float16)[0])))


class TwoRoutes(unittest.TestCase):
    """The vectorised rule and the exact-rational one share no arithmetic and must agree."""

    def assertSame(self, values, why=""):
        self.assertEqual(derived(values), exact(values), why)

    def test_on_the_hand_derived_blocks(self):
        for values in ([127.0, 0.5, 1.5, -2.5] + [0.0] * 28, [0.0] * 32, [100.0, 50.0, -25.0] + [0.0] * 29,
                       [127.0 * 2.0 ** -20, 1.0 * 2.0 ** -20] + [0.0] * 30):
            self.assertSame(values)

    def test_where_the_reciprocal_of_the_scale_overflows(self):
        # The smallest BF16 magnitudes: the scale is a binary32 subnormal and its reciprocal is infinite, so a
        # nonzero value saturates by its sign and a zero takes the NaN code, 0.
        tiny = 2.0 ** -133
        values = [tiny, -tiny, 0.0, 2 * tiny] + [0.0] * 28
        self.assertSame(values)
        self.assertEqual(derived(values)[1][:4], [127, -128, 0, 127])

    def test_at_the_top_of_the_bf16_range(self):
        top = float(np.array([0x7F7F0000], dtype=np.uint32).view(np.float32)[0])
        self.assertSame([top, -top, top / 2, 1.0] + [0.0] * 28)

    def test_on_every_block_of_a_committed_tensor(self):
        raw = location(BASE, CANONICAL, QUANTISED)
        seg, start, e = raw
        with open(seg.path, "rb") as f:
            f.seek(start)
            values = P.bf16_to_f32(f.read(e.len)).reshape(-1, P.BLOCK)
        fast = P.derive_q8_0(values)
        for b in range(0, values.shape[0], 7):  # every seventh block: the exact route is slow
            stored, codes = P.derive_block_exact([Fraction(float(v)) for v in values[b]])
            self.assertEqual((int(fast[b, 0]) | (int(fast[b, 1]) << 8), [int(c) for c in fast[b, 2:].view(np.int8)]),
                             (stored, codes), f"block {b}")

    def test_on_seeded_blocks_of_bf16_values_across_magnitudes(self):
        rng = np.random.default_rng(20261114)
        for exponent in (-130, -60, -20, -1, 0, 5, 40, 100, 126):
            patterns = rng.integers(0, 1 << 15, size=(8, P.BLOCK), dtype=np.int64)
            signs = rng.integers(0, 2, size=(8, P.BLOCK), dtype=np.int64) << 15
            # A BF16 pattern: sign, an exponent field within three binades of the one asked for (zero and the BF16
            # subnormals at the bottom of the range), and seven fraction bits.
            field = np.clip(127 + exponent - (patterns >> 7) % 4, 0, 254)
            scaled = (((patterns & 0x7F) | (field << 7) | signs) << 16).astype(np.uint32)
            for values in scaled.view(np.float32):
                self.assertSame(list(values), f"exponent {exponent}")


class NearestBinary16(unittest.TestCase):
    def test_against_the_library_conversion_on_every_kind_of_value(self):
        rng = np.random.default_rng(7)
        bits = np.concatenate([
            rng.integers(0, 0x7F800000, size=20000, dtype=np.uint32),  # any finite non-negative binary32
            np.arange(0x38800000 - 64, 0x38800000 + 64, dtype=np.uint32),  # around the smallest binary16 normal
            np.arange(0x33000000 - 8, 0x33000000 + 8, dtype=np.uint32),  # around half the smallest subnormal
            np.arange(0x477FE000 - 8, 0x477FF000 + 8, dtype=np.uint32),  # the largest binary16 and its overflow
            (np.arange(0x3C00, 0x3C40, dtype=np.uint32) << 13) + 0x39000000 - (0x3C00 << 13) + (1 << 12),  # ties
            np.array([0, 1, 0x00800000], dtype=np.uint32),
        ])
        values = bits.view(np.float32)
        with np.errstate(over="ignore"):
            library = values.astype(np.float16).view(np.uint16)
        np.testing.assert_array_equal(P.f32_to_f16_nearest_bits(values), library)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.work)

    def copy(self, source=BASE):
        root = os.path.join(self.work, f"c{len(os.listdir(self.work))}")
        shutil.copytree(source, root)
        return root

    def judge(self, root):
        return cli.evaluate_pack(P.CONTRACT, root)

    def measured(self, out):
        return {m["name"]: m["value"] for m in out["measurements"]}

    def findings(self, out):
        return " | ".join(f"{o['name']}: {o['detail']}" for o in out["offending"]) + " | " + " | ".join(out["reasons"])

    def edit_index(self, root, change):
        path = os.path.join(root, "index.json")
        doc = read_json(path)
        change(doc)
        write_json(path, doc)


class CommittedPacks(Fixture):
    def test_every_engine_compiled_pack_is_the_derived_pack(self):
        for container in CONTAINERS:
            with self.subTest(container=os.path.relpath(container, CONFORMANCE)):
                out = self.judge(container)
                self.assertEqual((out["status"], out["violations"], out["offending"], out["reasons"]),
                                 (SAT, 0, [], []), self.findings(out))
                m = self.measured(out)
                self.assertEqual(m["re-derived representations judged"], "1")
                self.assertEqual(m["Q8_0 tensors derived and compared"], "14")
                # Complete coverage: every block of every Q8_0 tensor, 34 bytes each, is the pack's whole Q8_0 payload.
                q8_bytes = sum(e.len for e in C.open_container(container).segments[PACK].tensors if e.dtype == C.Q8_0)
                self.assertEqual(int(m["blocks compared (stored scale and 32 codes each)"]) * P.BLOCK_BYTES, q8_bytes)
                self.assertEqual(int(m["elements quantised"]), int(m["blocks compared (stored scale and 32 codes each)"]) * 32)
                self.assertEqual(m["tensors carried unquantised, compared byte for byte"], "4")

    def test_the_answer_says_what_it_was_derived_from_and_what_it_does_not_establish(self):
        out = self.judge(BASE)
        self.assertEqual(out["schema"], "larql-verify-pack/1")
        self.assertEqual(out["derivations"], [{
            "representation": PACK, "source": CANONICAL,
            "source_payload_sha256": read_json(os.path.join(BASE, "index.json"))["representations"][CANONICAL]
            ["payload_sha256"],
            "rule": P.ENCODER, "codec": C.CODEC_Q8_0}])
        self.assertIn("the engine's packer is not run", out["derived_from"])
        not_established = self.measured(out)["not established"]
        for limit in ("runtime kernel", "canonical weights are the intended ones", "chosen for quantisation"):
            self.assertIn(limit, not_established)


class WrongPacksWithRightDigests(Fixture):
    """Each tamper is resealed: the index's digests agree with the bytes, so only the derivation can tell."""

    def test_a_corrupted_pack_whose_digests_were_recomputed_is_violated(self):
        # The case the ledger contract cannot see: it chains the pack's digests and does not re-quantise.
        root = self.copy()
        patch_bytes(root, PACK, QUANTISED, 0, bytes(CORRUPTED_BYTES))
        reseal(root, PACK)
        C.open_container(root)  # the container itself is intact: every digest holds
        out = self.judge(root)
        self.assertEqual(out["status"], VIOL, self.findings(out))
        self.assertEqual(out["violations"], 2)  # 64 bytes span the first two 34-byte blocks
        self.assertIn(f"{PACK}/{QUANTISED}", self.findings(out))
        self.assertIn("first is block 0 (row 0, columns 0..=31)", self.findings(out))

    def test_one_code_or_one_scale_bit(self):
        for at, what in ((2 + 5, "0 stored scales, 1 codes"), (0, "1 stored scales, 0 codes")):
            with self.subTest(at=at):
                root = self.copy()
                seg, start, _ = location(root, PACK, QUANTISED)
                block = 3
                with open(seg.path, "rb") as f:
                    f.seek(start + block * P.BLOCK_BYTES + at)
                    old = f.read(1)[0]
                patch_bytes(root, PACK, QUANTISED, block * P.BLOCK_BYTES + at, bytes([old ^ 1]))
                reseal(root, PACK)
                out = self.judge(root)
                self.assertEqual((out["status"], out["violations"]), (VIOL, 1), self.findings(out))
                self.assertIn(what, self.findings(out))
                self.assertIn("first is block 3 (row 0, columns 96..=127)", self.findings(out))

    def test_a_tensor_carried_unquantised_that_differs_from_the_canonical_one(self):
        root = self.copy()
        patch_bytes(root, PACK, CARRIED, 0, b"\x00\x40")
        reseal(root, PACK)
        out = self.judge(root)
        self.assertEqual(out["status"], VIOL, self.findings(out))
        self.assertIn("carried unquantised differs", self.findings(out))

    def test_a_pack_left_behind_by_an_edit_of_the_canonical_weights(self):
        # The canonical tensor changes and everything is resealed, the pack's source digest included, but the pack is
        # not re-derived: it now describes weights the container no longer holds.
        root = self.copy()
        patch_bytes(root, CANONICAL, QUANTISED, 0, struct.pack("<H", bf16(96.0)))
        payload = reseal(root, CANONICAL)
        self.edit_index(root, lambda d: d["representations"][PACK].update(source_representation_digest=payload))
        C.open_container(root)
        out = self.judge(root)
        self.assertEqual((out["status"], out["violations"]), (VIOL, 1), self.findings(out))
        self.assertIn("first is block 0", self.findings(out))

    def test_a_pack_that_names_other_weights_as_its_source(self):
        root = self.copy()
        self.edit_index(root, lambda d: d["representations"][PACK].update(source_representation_digest="0" * 64))
        out = self.judge(root)
        self.assertEqual(out["status"], VIOL, self.findings(out))
        self.assertIn("does not name the weights this container holds", self.findings(out))


class NeverAVacuousPass(Fixture):
    def assertNotJudged(self, out, reason):
        self.assertEqual((out["status"], out["violations"], out["offending"]), (CNE, 0, []), self.findings(out))
        self.assertIn(reason, self.findings(out))

    def test_a_container_with_no_re_derived_representation(self):
        root = self.copy()

        def drop(d):
            del d["representations"][PACK]
            d["segments"].pop(PACK.replace("@", "@"), None)

        self.edit_index(root, drop)
        self.assertNotJudged(self.judge(root), "would hold vacuously")

    def test_a_pack_made_by_a_rule_this_contract_does_not_implement(self):
        for encoder in ({"algorithm": "kquant-larql-native", "revision": 1},
                        {"algorithm": "kquant-ggml-reference", "revision": 1, "source": "x"}, None):
            with self.subTest(encoder=encoder):
                root = self.copy()
                self.edit_index(root, lambda d: d["representations"][PACK].update(encoder=encoder))
                self.assertNotJudged(self.judge(root), "this contract implements")

    def test_a_canonical_tensor_with_a_non_finite_element(self):
        root = self.copy()
        patch_bytes(root, CANONICAL, QUANTISED, 0, struct.pack("<H", BF16_INFINITY))
        payload = reseal(root, CANONICAL)
        self.edit_index(root, lambda d: d["representations"][PACK].update(source_representation_digest=payload))
        self.assertNotJudged(self.judge(root), "non-finite element")

    def test_a_pack_whose_bytes_do_not_match_its_own_digests(self):
        root = self.copy()
        patch_bytes(root, PACK, QUANTISED, 0, b"\xff")
        self.assertNotJudged(self.judge(root), "an input can't be read")

    def test_a_missing_pack_segment(self):
        root = self.copy()
        os.remove(location(root, PACK, QUANTISED)[0].path)
        self.assertNotJudged(self.judge(root), "an input can't be read")

    def test_an_unknown_contract_and_a_container_that_is_not_a_directory(self):
        out = cli.evaluate_pack("NOPE", BASE)
        self.assertEqual(out["status"], CNE)
        self.assertIn("unknown contract", out["reasons"][0])
        out = cli.evaluate_pack(P.CONTRACT, os.path.join(BASE, "index.json"))
        self.assertEqual(out["status"], CNE)
        self.assertIn("must be a directory", out["reasons"][0])


if __name__ == "__main__":
    unittest.main()
