"""Unit tests for refcheck.q8_0.

Expected values here come from exact rational arithmetic (fractions.Fraction) on
the stored bit patterns, never from running a decoder. gguf-py is used only as a
cross-check, bit for bit.
"""

import os
import struct
import sys
import unittest
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gguf import GGMLQuantizationType, dequantize as gguf_dequantize  # noqa: E402
from gguf.quants import Q8_0 as GGUF_Q8_0  # noqa: E402

from refcheck import q8_0  # noqa: E402


def block(d_bits: int, qs) -> bytes:
    """One 34-byte block: d as little-endian u16, then 32 int8 codes."""
    qs = list(qs) + [0] * (32 - len(qs))
    return struct.pack("<H", d_bits) + struct.pack("<32b", *qs)


def expected_bits(d_bits: int, q: int) -> int:
    """f32 bit pattern of q * d, computed exactly. Only for finite d."""
    d = Fraction(struct.unpack("<e", struct.pack("<H", d_bits))[0])
    v = Fraction(q) * d
    if v == 0:  # Fraction has no signed zero; IEEE: sign(q*d) = sign(q) xor sign(d), q = 0 is +
        neg = (q < 0) != bool(d_bits & 0x8000)
        return 0x80000000 if neg else 0
    f = float(v)
    # the product must be representable in f32 (rule T1), else the premise fails
    assert Fraction(struct.unpack("<f", struct.pack("<f", f))[0]) == v
    return struct.unpack("<I", struct.pack("<f", f))[0]


def decode_bits(data: bytes) -> list[int]:
    return q8_0.f32_bits(q8_0.dequantize_q8_0(data)).tolist()


class ScaleConversion(unittest.TestCase):
    def test_f16_to_f32_matches_ieee_for_every_non_nan_pattern(self):
        h = np.arange(65536, dtype=np.uint32)
        got = q8_0.f16_bits_to_f32_bits(h)
        for i in range(65536):
            exp = (i >> 10) & 0x1F
            man = i & 0x3FF
            if exp == 31 and man:
                continue
            if exp == 31:
                f = float("inf")
            else:
                f = struct.unpack("<e", struct.pack("<H", i & 0x7FFF))[0]
            want = struct.unpack("<I", struct.pack("<f", f))[0] | ((i >> 15) << 31)
            self.assertEqual(int(got[i]), want, hex(i))

    def test_nan_stays_nan_with_sign_and_payload(self):
        for bits in (0x7E00, 0xFE00, 0x7C01, 0xFFFF):
            f32 = int(q8_0.f16_bits_to_f32_bits(np.array([bits]))[0])
            self.assertTrue(np.isnan(np.array([f32], dtype=np.uint32).view(np.float32))[0])
            self.assertEqual(f32 >> 31, bits >> 15)
            self.assertEqual(f32 & 0x7FFFFF, ((bits & 0x3FF) << 13) | 0x400000)  # payload kept, quieted


class AnalyticDecode(unittest.TestCase):
    """The protocol's T1 decode cases, with exact-arithmetic expectations."""

    def test_codes_127_minus128_0(self):
        d = 0x3C00  # 1.0
        got = decode_bits(block(d, [127, -128, 0, 1, -1]))
        want = [expected_bits(d, q) for q in (127, -128, 0, 1, -1)] + [0] * 27
        self.assertEqual(got, want)
        self.assertEqual(q8_0.dequantize_q8_0(block(d, [127, -128])).tolist()[:2], [127.0, -128.0])

    def test_scales_finite(self):
        cases = {
            "+0": 0x0000,
            "-0": 0x8000,
            "min subnormal": 0x0001,
            "max subnormal": 0x03FF,
            "min normal": 0x0400,
            "65504": 0x7BFF,
            "negative d": 0xB800,  # -0.5
            "-65504": 0xFBFF,
            "-min subnormal": 0x8001,
        }
        for name, d in cases.items():
            qs = [127, -128, 0, 1, -1, 64, -64]
            got = decode_bits(block(d, qs))[: len(qs)]
            want = [expected_bits(d, q) for q in qs]
            self.assertEqual(got, want, name)

    def test_signed_zero_results_are_distinguished(self):
        # q = 0 converts to +0.0, so the result's sign is the sign of d
        self.assertEqual(decode_bits(block(0x0000, [0]))[0], 0x00000000)
        self.assertEqual(decode_bits(block(0x8000, [0]))[0], 0x80000000)
        self.assertEqual(decode_bits(block(0x0000, [-1]))[0], 0x80000000)
        self.assertEqual(decode_bits(block(0x8000, [-1]))[0], 0x00000000)

    def test_infinite_scale(self):
        pos, neg = 0x7C00, 0xFC00
        got = decode_bits(block(pos, [1, -1, 127, -128]))[:4]
        self.assertEqual(got, [0x7F800000, 0xFF800000, 0x7F800000, 0xFF800000])
        got = decode_bits(block(neg, [1, -1, 127, -128]))[:4]
        self.assertEqual(got, [0xFF800000, 0x7F800000, 0xFF800000, 0x7F800000])
        for d in (pos, neg):  # q = 0 gives NaN (0 * inf)
            out = q8_0.dequantize_q8_0(block(d, [0, 1]))
            self.assertTrue(np.isnan(out[0]))
            self.assertFalse(np.isnan(out[1]))

    def test_nan_scale(self):
        out = q8_0.dequantize_q8_0(block(0x7E00, [1, -1, 0, 127]))
        self.assertTrue(np.isnan(out[:4]).all())
        # all 32 codes see the NaN scale, including the zero padding codes
        self.assertTrue(np.isnan(out).all())

    def test_layout_little_endian_scale(self):
        # d = 0x3C00 (1.0) stored as bytes 00 3C. Swapped (3C 00) would be 0x003C = 3.58e-6.
        data = bytes([0x00, 0x3C]) + struct.pack("<32b", *([2] * 32))
        self.assertEqual(q8_0.dequantize_q8_0(data)[0], 2.0)
        swapped = bytes([0x3C, 0x00]) + struct.pack("<32b", *([2] * 32))
        self.assertEqual(
            decode_bits(swapped)[0], expected_bits(0x003C, 2)
        )

    def test_layout_int8_sign(self):
        data = block(0x3C00, [-128, -1, 127, 1])
        self.assertEqual(q8_0.dequantize_q8_0(data)[:4].tolist(), [-128.0, -1.0, 127.0, 1.0])
        raw = bytes([0x00, 0x3C, 0xFF, 0x80, 0x7F]) + bytes(29)
        self.assertEqual(q8_0.dequantize_q8_0(raw)[:3].tolist(), [-1.0, -128.0, 127.0])

    def test_layout_element_order_within_block(self):
        data = block(0x3C00, range(32))
        self.assertEqual(q8_0.dequantize_q8_0(data).tolist(), [float(i) for i in range(32)])

    def test_layout_block_order_and_row_order(self):
        # 3 rows x 64 elements: row r, block b has scale 2^(r*2+b) and code 1 -> value 2^(2r+b)
        data = b""
        for r in range(3):
            for b in range(2):
                e = 2 * r + b
                d_bits = (15 + e) << 10  # 2**e as f16
                data += block(d_bits, [1] * 32)
        out = q8_0.dequantize_q8_0_rows(data, 3, 64)
        for r in range(3):
            for b in range(2):
                self.assertTrue((out[r, 32 * b : 32 * b + 32] == 2.0 ** (2 * r + b)).all())

    def test_shape_errors(self):
        with self.assertRaises(ValueError):
            q8_0.dequantize_q8_0(bytes(33))
        with self.assertRaises(ValueError):
            q8_0.dequantize_q8_0_rows(bytes(34 * 2), 1, 33)
        with self.assertRaises(ValueError):
            q8_0.dequantize_q8_0_rows(bytes(34 * 2), 3, 32)

    def test_zero_scale_count(self):
        data = block(0, [1]) + block(0x8000, [1]) + block(0x0001, [1]) + block(0x3C00, [1])
        self.assertEqual(q8_0.count_zero_scales(data), 2)


class CrossCheckGguf(unittest.TestCase):
    """Bit-for-bit comparison with gguf-py 0.18.0 on identical bytes."""

    @staticmethod
    def gguf_decode(data: bytes) -> np.ndarray:
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(-1, 34)
        with np.errstate(invalid="ignore"):
            return GGUF_Q8_0.dequantize_blocks(blocks).reshape(-1)

    def assertBitEqual(self, got: np.ndarray, want: np.ndarray):
        self.assertEqual(got.shape, want.shape)
        g, w = q8_0.f32_bits(got), q8_0.f32_bits(want)
        bad = np.flatnonzero(g != w)
        self.assertEqual(bad.size, 0, f"{bad.size} mismatches, first at {bad[:5]}: {g[bad[:5]]} vs {w[bad[:5]]}")

    def test_every_scale_bit_pattern_with_every_code(self):
        # all 65536 f16 scales x all 256 int8 codes (8 blocks of 32 codes each)
        codes = np.arange(-128, 128, dtype=np.int8).reshape(8, 32)
        d = np.arange(65536, dtype=np.uint16)
        n = d.size * 8
        raw = np.empty((n, 34), dtype=np.uint8)
        raw[:, 0] = np.repeat(d & 0xFF, 8).astype(np.uint8)
        raw[:, 1] = np.repeat(d >> 8, 8).astype(np.uint8)
        raw[:, 2:] = np.tile(codes.view(np.uint8), (d.size, 1))
        data = raw.tobytes()
        self.assertBitEqual(q8_0.dequantize_q8_0(data), self.gguf_decode(data))

    def test_random_blocks(self):
        rng = np.random.default_rng(20261030)
        data = rng.integers(0, 256, size=34 * 20000, dtype=np.uint8).tobytes()
        self.assertBitEqual(q8_0.dequantize_q8_0(data), self.gguf_decode(data))

    def test_gguf_public_dequantize_entry_point(self):
        rng = np.random.default_rng(7)
        rows, k = 5, 96
        arr = rng.integers(0, 256, size=rows * (k // 32) * 34, dtype=np.uint8)
        # keep scales finite so NaN payload conventions cannot matter here
        view = arr.reshape(-1, 34)
        view[:, 1] &= 0x7B
        want = gguf_dequantize(arr.reshape(rows, (k // 32) * 34), GGMLQuantizationType.Q8_0)
        got = q8_0.dequantize_q8_0_rows(arr.tobytes(), rows, k)
        self.assertBitEqual(got, want)


if __name__ == "__main__":
    unittest.main()
