"""Tests for the bit-exact emulator of the frozen VLP numerics v2 op. Run (from this directory):
    python3 -I test_emulator.py
The frozen specification, vectors and independent evaluator are only READ here."""
import json
import os
import random
import sys
import unittest
from fractions import Fraction

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import crosscheck as X  # noqa: E402
import emulate as E  # noqa: E402

REPO = X.REPO
VECTORS = os.path.join(REPO, "conformance", "vlp-numerics", "v2", "vectors.json")
RANDOM_CASES, RANDOM_SEED = 20000, 20261113
MIN_SUCCESSES_PER_PROFILE = 800


def load_vectors():
    with open(VECTORS) as f:
        return json.load(f)["cases"]


def expected_outcome(case):
    e = case["expect"]
    return ("refuse", e["refuse"]) if "refuse" in e else ("bf16", tuple(int(h, 16) for h in e["bf16"]))


# ------------------------------------------------------------------ an exact-rational oracle for the two rounders

def _floor_log2(a):
    n = a.numerator.bit_length() - a.denominator.bit_length()
    return n - 1 if Fraction(2) ** n > a else n


def oracle_bf16(x):
    """Exact value (Fraction, with the sign of a zero given separately by the caller) -> (magnitude bits, overflow)."""
    a = abs(x)
    if a == 0:
        return 0, False
    e = max(_floor_log2(a), -126)
    quantum = Fraction(2) ** (e - 7)
    q = a / quantum
    m = q.numerator // q.denominator
    rem = q - m
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and m % 2 == 1):
        m += 1
    value = m * quantum
    if value >= Fraction(2) ** 128:
        return 0, True
    if value < Fraction(2) ** -126:
        return int(value / Fraction(2) ** -133), False
    ex = _floor_log2(value)
    return ((ex + 127) << 7) | int((value / Fraction(2) ** ex - 1) * 128), False


def sample_f64(rng, count):
    vals = []
    for _ in range(count):
        kind = rng.random()
        if kind < 0.4:
            bits = rng.getrandbits(64)
            if (bits >> 52) & 0x7FF == 0x7FF:
                continue
            vals.append(np.array([bits], dtype=np.uint64).view(np.float64)[0])
        else:   # ties and neighbours of a random BF16 value
            word = rng.getrandbits(16)
            if (word >> 7) & 0xFF == 0xFF:
                continue
            base = float(E.bf16_to_f64(np.array([word], dtype=np.uint16))[0])
            ulp = 2.0 ** (max(int(np.floor(np.log2(abs(base)))) if base else -126, -126) - 7)
            for d in (0.5, 0.5 + 2.0 ** -40, 0.5 - 2.0 ** -40, 1.5, 0.25, 1.0):
                vals.append(base + d * ulp * rng.choice([1, -1]))
    vals += [0.0, -0.0, 2.0 ** -126, 2.0 ** -133, 2.0 ** -134, 2.0 ** -135, -(2.0 ** -134), 3.4e38, 2.0 ** 128 * (1 - 2.0 ** -9),
             2.0 ** 128 * (1 - 2.0 ** -10), 5e-324, -5e-324, 1e-310]
    return np.array(vals, dtype=np.float64)


class Rounders(unittest.TestCase):
    def check(self, bits, over, x_exact, sign_negative):
        want_bits, want_over = oracle_bf16(x_exact)
        self.assertEqual(bool(over), want_over, x_exact)
        if not want_over:
            self.assertEqual(int(bits), want_bits | (0x8000 if sign_negative else 0), x_exact)

    def test_f64_to_bf16_is_one_exact_rounding(self):
        xs = sample_f64(random.Random(1), 6000)
        bits, over = E.f64_to_bf16(xs)
        for x, b, o in zip(xs, bits, over):
            self.check(b, o, Fraction(float(x)), bool(np.signbit(x)))

    def test_f32_to_bf16_is_exact(self):
        rng = random.Random(2)
        words = []
        for _ in range(6000):
            w = rng.getrandbits(32)
            if (w >> 23) & 0xFF != 0xFF:
                words.append(w)
        # exact ties: low 16 bits 0x8000 with either parity of bit 16
        words += [((rng.getrandbits(15) << 17) | (p << 16) | 0x8000) & 0x7FFFFFFF for p in (0, 1) for _ in range(200)]
        xs = np.array(words, dtype=np.uint32).view(np.float32)
        bits, over = E.f32_to_bf16(xs)
        for x, b, o in zip(xs, bits, over):
            if np.isfinite(x):
                self.check(b, o, Fraction(float(x)), bool(np.signbit(x)))

    def test_double_rounding_is_not_what_rne1_does(self):
        # a value that is not a tie in f64 but becomes one in f32: 1 + 2^-8 + 2^-33
        x = np.array([1.0 + 2.0 ** -8 + 2.0 ** -33])
        direct = int(E.f64_to_bf16(x)[0][0])
        via_f32 = int(E.f32_to_bf16(x.astype(np.float32))[0][0])
        self.assertNotEqual(direct, via_f32)


class FrozenVectors(unittest.TestCase):
    def test_all_frozen_cases_are_reproduced_bit_for_bit_including_refusals(self):
        cases = load_vectors()
        self.assertEqual(len(cases), 91)
        wrong = [c["id"] for c in cases if E.run_case(c) != (expected_outcome(c)[0],
                 list(expected_outcome(c)[1]) if expected_outcome(c)[0] == "bf16" else expected_outcome(c)[1])]
        self.assertEqual(wrong, [])


class CrossCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = X.random_cases(RANDOM_CASES, RANDOM_SEED)
        cls.independent = X.run_independent(cls.cases)
        cls.emulated = {c["id"]: E.run_case(c) for c in cls.cases}

    def test_the_emulator_agrees_with_the_frozen_independent_evaluator(self):
        self.assertEqual(X.differing(self.emulated, self.independent), [])

    def test_the_random_mix_has_power(self):
        by_profile = {p: 0 for p in X.PROFILES}
        codes = set()
        for c in self.cases:
            kind, val = self.emulated[c["id"]]
            if kind == "bf16":
                by_profile[c["op"]["numerics"]] += 1
            else:
                codes.add(val)
        for p, n in by_profile.items():
            self.assertGreaterEqual(n, MIN_SUCCESSES_PER_PROFILE, p)
        for needed in ("overflow", "non_finite_intermediate", "columns_invalid", "tensor_refused", "fixups_invalid",
                       "shape_invalid", "rank_invalid", "non_finite_operand"):
            self.assertIn(needed, codes)

    def test_the_comparison_detects_every_kind_of_corruption_of_either_side(self):
        sample = [c["id"] for c in self.cases][::40]
        for cid in sample:
            for flavour in ("bit", "kind", "length"):
                bad = dict(self.independent)
                bad[cid] = X.mutate(self.independent[cid], flavour)
                if X.outcome(bad[cid]) != X.outcome(self.independent[cid]):
                    self.assertEqual(X.differing(self.emulated, bad), [cid], (cid, flavour))
                worse = dict(self.emulated)
                worse[cid] = X.mutate(self.emulated[cid], flavour)
                if X.outcome(worse[cid]) != X.outcome(self.emulated[cid]):
                    self.assertEqual(X.differing(worse, self.independent), [cid], (cid, flavour))
        missing = {k: v for k, v in list(self.independent.items())[1:]}
        self.assertEqual(len(X.differing(self.emulated, missing)), 1)


def _mutant_apply(start_from_zero=False, no_selection=False, reverse_k=False, via_f32=False, ties_away=False):
    """The emulator with ONE deliberately wrong semantics, to show the checks above would catch it."""
    real_f32 = E.f32_to_bf16

    def f32_ties_away(x):
        u = np.asarray(x, dtype=np.float32).view(np.uint32)
        mag = u & np.uint32(0x7FFFFFFF)
        rounded = (mag + np.uint32(0x8000)) >> np.uint32(16)
        over = rounded >= np.uint32(E.BF16_INF_BITS)
        sign = ((u >> np.uint32(31)) & np.uint32(1)).astype(np.uint16) << np.uint16(15)
        return (np.where(over, 0, rounded).astype(np.uint16) | sign), over

    def apply(base_bits, columns, a, b, profile):
        if not (start_from_zero or no_selection or reverse_k or via_f32 or ties_away):
            return E.apply_columns(base_bits, columns, a, b, profile)
        work = np.float32 if profile == E.PROFILE_F32 else np.float64
        base_bits, columns = np.asarray(base_bits, np.uint16), np.asarray(columns, np.int64)
        a, b = np.asarray(a, work), np.asarray(b, work)
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise E.Refusal("non_finite_operand")
        block = base_bits[:, columns]
        if (block & np.uint16(E.BF16_EXPONENT_MASK) == np.uint16(E.BF16_EXPONENT_MASK)).any():
            raise E.Refusal("non_finite_operand")
        written = ((a != 0)[:, :, None] & (b != 0)[None, :, :]).any(axis=1)
        if no_selection:
            written = np.ones_like(written)
        order = range(a.shape[1] - 1, -1, -1) if reverse_k else range(a.shape[1])
        with np.errstate(all="ignore"):
            if start_from_zero:
                s = np.zeros((a.shape[0], b.shape[1]), work)
                for k in order:
                    s = np.add(s, np.multiply(a[:, k:k + 1], b[k:k + 1, :]))
                w = np.add(E.bf16_to_f64(block).astype(work), s)
            else:
                w = E.bf16_to_f64(block).astype(work)
                for k in order:
                    w = np.add(w, np.multiply(a[:, k:k + 1], b[k:k + 1, :]))
            finite = np.isfinite(w)
            safe = np.where(finite, w, work(0))
            if profile == E.PROFILE_RNE1 and not via_f32:
                bits, over = E.f64_to_bf16(safe)
            else:
                narrow = safe.astype(np.float32)
                f32_over = ~np.isfinite(narrow) if profile != E.PROFILE_F32 else np.zeros(narrow.shape, bool)
                bits, over = (f32_ties_away if ties_away else real_f32)(np.where(f32_over, np.float32(0), narrow))
                over = over | f32_over
        failed = written & (~finite | over)
        if failed.any():
            first = np.flatnonzero(failed.reshape(-1))[0]
            raise E.Refusal("non_finite_intermediate" if not finite.reshape(-1)[first] else "overflow")
        out = base_bits.copy()
        out[:, columns] = np.where(written, bits, block)
        return out, written
    return apply


class SemanticMutants(unittest.TestCase):
    """The vectors plus the random cross-check must catch each deliberately wrong semantics."""

    def caught(self, **mutation):
        original = E.apply_columns
        E.apply_columns = _mutant_apply(**mutation)
        try:
            frozen = [c for c in load_vectors() if E.run_case(c) != (
                expected_outcome(c)[0], list(expected_outcome(c)[1]) if expected_outcome(c)[0] == "bf16" else expected_outcome(c)[1])]
            cases = X.random_cases(3000, RANDOM_SEED + 1)
            independent = X.run_independent(cases)
            random_wrong = X.differing({c["id"]: E.run_case(c) for c in cases}, independent)
        finally:
            E.apply_columns = original
        return len(frozen), len(random_wrong)

    def test_each_wrong_semantics_is_caught_by_the_cross_check_and_by_the_vectors(self):
        for name, mutation in {"start_from_zero": dict(start_from_zero=True), "no_selection": dict(no_selection=True),
                               "reverse_k": dict(reverse_k=True), "rne1_via_f32": dict(via_f32=True),
                               "ties_away": dict(ties_away=True)}.items():
            frozen, random_wrong = self.caught(**mutation)
            self.assertGreater(random_wrong, 0, f"{name}: the random cross-check missed it")
            if name != "reverse_k":   # the vectors need not separate every order; the cross-check must
                self.assertGreater(frozen, 0, f"{name}: the frozen vectors missed it")


class Properties(unittest.TestCase):
    def test_a_zero_update_returns_the_base_bits_including_negative_zero(self):
        rng = np.random.default_rng(0)
        base = rng.integers(0, 0x7F00, size=(7, 9), dtype=np.uint16)
        base[rng.random(base.shape) < 0.3] = 0x8000
        for profile in E.PROFILES:
            for a, b in ((np.zeros((7, 2)), rng.standard_normal((2, 4))), (rng.standard_normal((7, 2)), np.zeros((2, 4)))):
                out, written = E.apply_columns(base, [1, 3, 5, 8], a, b, profile)
                self.assertTrue((out == base).all())
                self.assertFalse(written.any())

    def test_columns_outside_the_footprint_are_never_read(self):
        base = np.full((3, 5), 0x3F80, dtype=np.uint16)
        base[:, 0] = 0x7FC0   # NaN outside J
        out, _ = E.apply_columns(base, [2, 4], np.ones((3, 1)), np.ones((1, 2)), E.PROFILE_RNE1)
        self.assertTrue((out[:, 0] == 0x7FC0).all())

    def test_a_negative_column_does_not_wrap(self):
        case = load_vectors()
        negative = next(c for c in case if c["id"] == "ref-column-negative")
        self.assertEqual(E.run_case(negative), ("refuse", "columns_invalid"))


if __name__ == "__main__":
    unittest.main()
