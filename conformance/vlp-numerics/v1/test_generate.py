"""Checks on the VLP numerics golden vectors (stdlib; numpy is used only for the optional f32 cross-check).

    python3 -I -m unittest discover -s conformance/vlp-numerics/v1 -p 'test_*.py'
"""
import hashlib
import json
import math
import os
import random
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate as g  # noqa: E402

RANDOM_SAMPLES = 20000


def f64_from_bits(b):
    return struct.unpack("<d", b.to_bytes(8, "little"))[0]


class SuiteIsFrozen(unittest.TestCase):
    def test_files_match_the_freeze(self):
        with open(os.path.join(os.path.dirname(g.OUT), "FREEZE.json")) as f:
            frozen = json.load(f)["sha256"]
        for name, want in frozen.items():
            with open(os.path.join(os.path.dirname(g.OUT), name), "rb") as f:
                self.assertEqual(hashlib.sha256(f.read()).hexdigest(), want, f"{name} changed after the v1 freeze")


class VectorsReproduce(unittest.TestCase):
    def test_committed_file_is_what_the_generator_writes(self):
        with open(g.OUT) as f:
            self.assertEqual(f.read(), g.dumps(g.document()))


class VectorsAreWellFormed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(g.OUT) as f:
            cls.doc = json.load(f)

    def test_each_case_has_one_outcome_and_a_known_code(self):
        for c in self.doc["cases"]:
            exp = c["expect"]
            self.assertEqual(("bf16" in exp) + ("refuse" in exp), 1, c["id"])
            if "refuse" in exp:
                self.assertIn(exp["refuse"], self.doc["refusals"], c["id"])

    def test_hashes_match_the_bytes(self):
        for c in self.doc["cases"]:
            self.assertEqual(c["base"]["sha256"], g.sha_bf16(c["base"]["bf16"]), c["id"])
            if "bf16" in c["expect"]:
                self.assertEqual(c["expect"]["sha256"], g.sha_bf16(c["expect"]["bf16"]), c["id"])
                self.assertEqual(len(c["expect"]["bf16"]), len(c["base"]["bf16"]), c["id"])

    def test_every_targeted_variant_changes_the_result(self):
        for c in self.doc["cases"]:
            for d in c["discriminates"]:
                self.assertIn(d, c["under_wrong_semantics"], c["id"])
                self.assertNotEqual(c["under_wrong_semantics"][d], {k: v for k, v in c["expect"].items()
                                                                    if k != "sha256"}, c["id"])

    def test_every_profile_and_refusal_is_exercised(self):
        profs = {c["op"].get("numerics") for c in self.doc["cases"]}
        for p in self.doc["profiles"]:
            self.assertIn(p, profs)
        codes = {c["expect"].get("refuse") for c in self.doc["cases"]}
        for code in self.doc["refusals"]:
            self.assertIn(code, codes)


class ExactRoundingAgreesWithNative(unittest.TestCase):
    """The Fraction rounder against independent native routes, on random and adversarial values."""

    def values(self):
        rng = random.Random(1)
        out = []
        for _ in range(RANDOM_SAMPLES):
            kind = rng.randrange(4)
            if kind == 0:  # any finite binary64
                while True:
                    x = f64_from_bits(rng.getrandbits(64))
                    if math.isfinite(x):
                        break
            elif kind == 1:  # near the BF16 range, including subnormals
                x = math.ldexp(rng.uniform(0.5, 1), rng.randrange(-140, 128)) * rng.choice([1, -1])
            elif kind == 2:  # exact BF16 ties and their neighbours
                b = rng.getrandbits(15)  # every exponent, odd and even
                if (b & 0x7F80) == 0x7F80 or b == 0x7F7F:
                    b = 0x3F80  # skip NaN/inf, and the top value, whose upper neighbour is inf
                lo = struct.unpack("<f", (b << 16).to_bytes(4, "little"))[0]
                hi = struct.unpack("<f", ((b + 1) << 16).to_bytes(4, "little"))[0]
                x = (lo + hi) / 2 + rng.choice([0.0, 0.0, math.ulp((lo + hi) / 2), -math.ulp((lo + hi) / 2)])
            else:  # f32 subnormals and the f32/BF16 boundary
                x = math.ldexp(rng.uniform(0.5, 1), rng.randrange(-152, -120)) * rng.choice([1, -1])
            out.append(x)
        return out

    def test_bf16_from_f64(self):
        for x in self.values():
            try:
                want = g.native_bf16_from_f64(x)
            except OverflowError:
                want = "overflow"
            try:
                got = g.encode(g.rnd(g.Fraction(x), math.copysign(1, x) < 0, "bf16"), "bf16")
            except OverflowError:
                got = "overflow"
            self.assertEqual(got, want, repr(x))

    def test_f32_from_f64(self):
        for x in self.values():
            try:
                want = struct.unpack("<I", struct.pack("<f", x))[0]
            except OverflowError:
                want = "overflow"
            try:
                got = g.encode(g.rnd(g.Fraction(x), math.copysign(1, x) < 0, "f32"), "f32")
            except OverflowError:
                got = "overflow"
            self.assertEqual(got, want, repr(x))

    def test_decode_encode_round_trip(self):
        rng = random.Random(2)
        for fmt, nb in (("f64", 64), ("f32", 32), ("bf16", 16)):
            for _ in range(RANDOM_SAMPLES // 4):
                b = rng.getrandbits(nb)
                try:
                    v = g.decode(b, fmt)
                except ValueError:
                    continue
                self.assertEqual(g.encode(v, fmt), b)


class F32ProfileAgreesWithNumpy(unittest.TestCase):
    """The f32 profile's arithmetic cases, recomputed elementwise in numpy float32 (NEP 50, numpy >= 2)."""

    def test_f32_profile_cases(self):
        try:
            import numpy as np
        except ImportError:
            if os.environ.get("VLP_REQUIRE_NUMPY") == "1":
                raise
            self.skipTest("numpy not installed")
        if int(np.__version__.split(".")[0]) < 2:
            self.skipTest("numpy < 2 has value-based casting")
        with open(g.OUT) as f:
            cases = json.load(f)["cases"]
        f32 = lambda h: np.uint32(int(h, 16)).view(np.float32)
        bf = lambda h: np.uint32(int(h, 16) << 16).view(np.float32)

        def chk(x):
            assert x.dtype == np.float32, x.dtype
            if not np.isfinite(x):
                raise OverflowError
            return x

        def to_bf16(x):
            assert x.dtype == np.float32
            b = int(np.float32(x).view(np.uint32))
            return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF  # RNE on finite f32

        n = 0
        with np.errstate(over="ignore"):
            for c in cases:
                op = c["op"]
                exp = c["expect"]
                if op.get("numerics") != "f32-seq-rne1/v1" or op["fixups"] or c.get("tensor_role"):
                    continue
                if "refuse" in exp and exp["refuse"] not in ("non_finite_intermediate", "overflow"):
                    continue
                rows, cols = op["shape"]
                W = [bf(h) for h in c["base"]["bf16"]]
                out = [int(h, 16) for h in c["base"]["bf16"]]
                try:
                    if op["kind"] == "remove_direction":
                        u, alpha = [f32(h) for h in op["u"]], f32(op["alpha"])
                        for j in op["columns"]:
                            acc = np.float32(0.0)
                            for r in range(rows):
                                acc = chk(acc + chk(u[r] * W[r * cols + j]))
                            for r in range(rows):
                                t = chk(chk(alpha * u[r]) * acc)
                                out[r * cols + j] = to_bf16(chk(W[r * cols + j] - t))
                    else:
                        K = op["rank"]
                        a, b = [f32(h) for h in op["a"]], [f32(h) for h in op["b"]]
                        for r in range(rows):
                            for j in range(cols):
                                w = W[r * cols + j]
                                for k in range(K):
                                    w = chk(w + chk(a[r * K + k] * b[k * cols + j]))
                                out[r * cols + j] = to_bf16(w)
                    got = {"bf16": [f"{x:04x}" for x in out]}
                    if any(x & 0x7F80 == 0x7F80 for x in out):
                        got = {"refuse": "overflow"}
                except OverflowError:
                    got = {"refuse": "non_finite_intermediate"}
                self.assertEqual(got, {k: v for k, v in exp.items() if k != "sha256"}, c["id"])
                n += 1
        self.assertGreater(n, 0)


if __name__ == "__main__":
    unittest.main()
