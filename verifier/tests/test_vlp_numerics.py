"""The verifier's own implementation of the frozen VLP numerics, against the frozen vectors (used here only as test data)
and against an exact-rational route, and its BF16/f32 rounder against exact rounding."""

import json
import os
import random
import struct
import sys
import unittest
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from larql_contract_verifier.contracts import structured_patch as X  # noqa: E402
from larql_contract_verifier.contracts import vlp_numerics as V  # noqa: E402
from refcheck import numconv  # noqa: E402

VECTORS = {v: os.path.join(REPO, "conformance", "vlp-numerics", v, "vectors.json") for v in ("v1", "v2")}
EXPRESSIBLE = {"add_low_rank", "add_low_rank_columns", "remove_direction"}
SEED = 20261114
BF16 = (8, -126, 127)


def load(version):
    with open(VECTORS[version]) as f:
        return json.load(f)["cases"]


def run_case(case):
    try:
        return {"bf16": V.apply_vector_case(case["op"], case["base"]["bf16"], case["base"]["shape"],
                                            case.get("tensor_role"))}
    except V.Refusal as e:
        return {"refuse": e.code}


def exact_bf16_bits(x: float) -> tuple[int, bool]:
    """BF16 bits of the binary64 `x` by exact rational RNE (numconv), sign kept for zero; and whether it overflows."""
    negative = math_sign(x)
    try:
        v = numconv.round_nearest_even(Fraction(x), BF16)
    except numconv.Overflow:
        return 0, True
    return X.to_bf16_bits(X.Num(v, v == 0 and negative), False), False


def math_sign(x: float) -> bool:
    return struct.unpack("<Q", struct.pack("<d", x))[0] >> 63 == 1


class Vectors(unittest.TestCase):
    def check(self, version):
        cases = [c for c in load(version) if c["op"]["kind"] in EXPRESSIBLE]
        self.assertGreater(len(cases), 0)
        for case in cases:
            with self.subTest(case=case["id"]):
                want = {k: case["expect"][k] for k in ("bf16", "refuse") if k in case["expect"]}
                self.assertEqual(run_case(case), want)
        return len(cases)

    def test_every_v1_add_low_rank_and_remove_direction_case(self):
        self.assertEqual(self.check("v1"), 73)  # 38 add_low_rank + 35 remove_direction; zero_columns is not an op here

    def test_every_v2_add_low_rank_columns_case(self):
        self.assertEqual(self.check("v2"), 91)

    def test_each_wrong_semantics_the_vectors_name_is_told_apart(self):
        """Where a case lists what a wrong semantics would produce, this implementation does not produce it."""
        for version in ("v1", "v2"):
            for case in load(version):
                if case["op"]["kind"] not in EXPRESSIBLE:
                    continue
                got = run_case(case)
                for variant, wrong in case.get("under_wrong_semantics", {}).items():
                    with self.subTest(case=case["id"], variant=variant):
                        self.assertNotEqual(got, wrong)


class Rounder(unittest.TestCase):
    def values(self, n=20000):
        rng = random.Random(SEED)
        out = [0.0, -0.0, 1.0, -1.0, 2.0 ** -133, -(2.0 ** -133), 2.0 ** -134, -(2.0 ** -134), 2.0 ** -135,
               3 * 2.0 ** -135, 2.0 ** -1074, -(2.0 ** -1074), 2.0 ** -1022, 1 + 2.0 ** -8, 1 + 3 * 2.0 ** -8,
               -(1 + 2.0 ** -8), 1 + 2.0 ** -8 + 2.0 ** -52, 1 + 2.0 ** -8 - 2.0 ** -52,
               float.fromhex("0x1.fep127"), float.fromhex("0x1.ffp127"), float.fromhex("0x1.feffffffffffp127"),
               float.fromhex("0x1.fffffep127"), float.fromhex("0x1.ffffffp127"), 2.0 ** 128, 1e300, -1e300,
               2.0 ** -126, 2.0 ** -127, float.fromhex("0x1.fep-127")]
        for _ in range(n):
            kind = rng.random()
            if kind < 0.4:
                bits = rng.getrandbits(64) & ~(0x7FF << 52) | (rng.randrange(1023 - 140, 1023 + 130) << 52)
            elif kind < 0.7:  # near a BF16 tie
                b = rng.getrandbits(16) & 0x7F7F
                (base,) = struct.unpack("<f", struct.pack("<I", b << 16))
                bits = struct.unpack("<Q", struct.pack("<d", base))[0] + rng.choice([-1, 0, 1]) * rng.getrandbits(3) \
                    + (1 << 44)
            else:
                bits = rng.getrandbits(64)
            (x,) = struct.unpack("<d", struct.pack("<Q", bits & 0xFFFFFFFFFFFFFFFF))
            if x == x and abs(x) != float("inf"):
                out.append(x)
        return out

    def test_bf16_from_f64_equals_exact_rounding(self):
        xs = self.values()
        bits, over = V.f64_to_bf16(np.array(xs, dtype=np.float64))
        for x, b, o in zip(xs, bits.tolist(), over.tolist()):
            want, want_over = exact_bf16_bits(x)
            self.assertEqual(bool(o), want_over, x.hex())
            if not want_over:
                self.assertEqual(b, want, x.hex())

    def test_f32_from_f64_equals_numpy_where_numpy_is_finite(self):
        xs = [x for x in self.values() if abs(x) < 3.4e38]
        bits, over = V.f64_to_f32_bits(np.array(xs, dtype=np.float64))
        with np.errstate(all="ignore"):
            ref = np.array(xs, dtype=np.float64).astype(np.float32)
        finite = np.isfinite(ref)
        self.assertTrue(np.array_equal(over, ~finite))
        self.assertTrue(np.array_equal(bits[finite], ref[finite].view(np.uint32)))

    def test_a_nonzero_value_that_rounds_to_zero_keeps_its_sign(self):
        bits, _ = V.f64_to_bf16(np.array([2.0 ** -140, -(2.0 ** -140), 2.0 ** -1074, -(2.0 ** -1074)]))
        self.assertEqual(bits.tolist(), [0x0000, 0x8000, 0x0000, 0x8000])

    def test_the_tie_at_the_top_of_the_range_overflows(self):
        _, over = V.f64_to_bf16(np.array([float.fromhex("0x1.ffp127"), float.fromhex("0x1.fefffffffffffp127")]))
        self.assertEqual(over.tolist(), [True, False])

    def test_the_platform_keeps_subnormals(self):
        V.check_platform()


class ExactRoute(unittest.TestCase):
    """The vectorised route against an exact-rational one on random tensors larger than a row block, including signed
    zeros, subnormal bases and zero rows/entries, so the blocking and the selection are exercised."""

    def setUp(self):
        self.rng = np.random.default_rng(SEED)

    def base(self, rows, cols):
        bits = self.rng.integers(0, 1 << 16, size=rows * cols, dtype=np.uint32).astype(np.uint16)
        bits &= np.uint16(0xBFFF)  # keep exponents moderate so nothing overflows
        bits[self.rng.random(rows * cols) < 0.05] = 0x8000
        bits[self.rng.random(rows * cols) < 0.05] = 0x0001
        return bits

    def operand(self, n, zero_fraction=0.2):
        v = self.rng.standard_normal(n) * 2.0 ** self.rng.integers(-20, 4, size=n)
        v[self.rng.random(n) < zero_fraction] = 0.0
        v[self.rng.random(n) < 0.05] = -0.0
        return [f"{b:016x}" for b in v.view(np.uint64).tolist()]

    def exact_low_rank(self, base, rows, cols, a, b, columns, k, via_f32):
        a = [X._from_f64_bits(int(h, 16)) for h in a]
        b = [X._from_f64_bits(int(h, 16)) for h in b]
        out = list(base)
        for r in range(rows):
            for p, j in enumerate(columns):
                terms = [(a[r * k + t], b[t * len(columns) + p]) for t in range(k)]
                if not any(x.value != 0 and y.value != 0 for x, y in terms):
                    continue
                w = X._from_bf16_bits(base[r * cols + j])
                for x, y in terms:
                    w = X.add(w, X.mul(x, y))
                out[r * cols + j] = X.to_bf16_bits(w, via_f32)
        return out

    def test_add_low_rank_columns_matches_exact_rationals(self):
        rows, cols, k = V.ROW_BLOCK + 37, 9, 3
        columns = [0, 2, 3, 8]
        for profile, via in ((V.PROFILE_RNE1, False), (V.PROFILE_RNE2, True)):
            base = self.base(rows, cols)
            a, b = self.operand(rows * k), self.operand(k * len(columns))
            op = {"numerics": profile, "rank": k, "a": a, "b": b, "columns": columns}
            got = V.LowRank(op, (rows, cols)).apply(base).reshape(-1).tolist()
            self.assertEqual(got, self.exact_low_rank(base.tolist(), rows, cols, a, b, columns, k, via), profile)

    def test_remove_direction_matches_the_exact_route(self):
        rows, cols, columns = 300, 7, [1, 4, 6]
        for profile, via in ((V.PROFILE_RNE1, False), (V.PROFILE_RNE2, True)):
            base = self.base(rows, cols)
            u = self.operand(rows, 0.1)
            alpha = f"{np.float64(0.37).view(np.uint64):016x}"
            op = {"numerics": profile, "u": u, "alpha": alpha, "columns": columns}
            got = V.RemoveDirection(op, (rows, cols)).apply(base)
            want = X.remove_direction(base.tolist(), rows, cols, [X._from_f64_bits(int(h, 16)) for h in u],
                                      X._from_f64_bits(int(alpha, 16)), columns, via)
            flat = got.reshape(-1)
            for i, bits in want.items():
                self.assertEqual(int(flat[i]), bits, (profile, divmod(i, cols)))


if __name__ == "__main__":
    unittest.main()
