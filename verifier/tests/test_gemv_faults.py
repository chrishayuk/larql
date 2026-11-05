"""Deliberately faulty GEMV implementations against the two contracts (Q8_0-GEMV-SPEC.md revision 1).

The faults live in tests/_gemv_faults.py and are never imported by refcheck/. Each is written to section 3's interface.
For every fault this file asserts what the work order requires of `SEQ`, asserts `B1` where the order says "must", and
*records* what `B1` does where the order only expects an outcome. The record is printed at the end of the module.
Synthetic data only.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _gemv_faults as faults  # noqa: E402
import _gemv_gen as gen  # noqa: E402
import _gemv_oracle as orc  # noqa: E402
from refcheck import gemv  # noqa: E402

ALL_FAULTS = {**faults.MUST_FAIL_B1, **faults.RECORD_B1, **faults.DIFFERENT_PROGRAMS}
RECORD = {}  # fault name -> dict of counts, printed at the end


def bump(name, key, amount=1):
    RECORD.setdefault(name, {})
    RECORD[name][key] = RECORD[name].get(key, 0) + amount


def verdicts(fn, case):
    raw, x, n, k = case
    y = fn(raw, x, n, k)
    return gemv.check_seq(raw, x, y, n, k), gemv.check_b1(raw, x, y, n, k)


def ordinary_case(seed=1):
    rng = np.random.default_rng(seed)
    raw, x, n, k = gen.make_case(rng, 4, 96, "typical", "normal", "random")
    x[0] = gen.f32bits(1.0)
    return raw, x, n, k


def subnormal_scale_case(seed=2):
    rng = np.random.default_rng(seed)
    raw, x, n, k = gen.make_case(rng, 3, 64, "subnormal", "normal", "random")
    return raw, x, n, k


def tie_case():
    """n = 2, k = 256, d = 1, q = 1 (row 0) and -1 (row 1); x = 1, then 255 values of 2^-24 (negated for row 1's terms
    through q). Sequentially every `1 + 2^-24` is a tie that returns to 1, so the program's result is +/-1.0."""
    raw = orc.build([[(0x3C00, [1] * 32)] * 8, [(0x3C00, [-1] * 32)] * 8])
    x = [0x3F800000] + [0x33800000] * 255
    return raw, x, 2, 256


class Control(unittest.TestCase):
    def test_a_correct_program_in_the_same_style_equals_seq_bits(self):
        """The fault harness itself is sound: with no fault the style reproduces seq_bits bit for bit."""
        rng = np.random.default_rng(3)
        for sk, xk in (("typical", "normal"), ("subnormal", "subnormal"), ("typical", "wide"), ("min_normal", "tiny")):
            raw, x, n, k = gen.make_case(rng, 3, 64, sk, xk, "random")
            ok = faults._bits(faults._sequential(faults._w_times_x(raw, x, n, k)))
            self.assertEqual(list(ok), [int(v) for v in gemv.seq_bits(raw, x, n, k)])


class MustFailBoth(unittest.TestCase):
    """Wrong stride, dropped last block, flushed subnormal scales, one flipped code, one flipped sign."""

    def check_both_fail(self, name, case):
        s, b = verdicts(ALL_FAULTS[name], case)
        self.assertEqual(s.status, "violated", (name, s))
        self.assertEqual(b.status, "violated", (name, b))
        self.assertGreaterEqual(s.rows_violating, 1)
        self.assertGreaterEqual(b.rows_violating, 1)
        self.assertEqual(s.first_violating_row is not None, True)
        RECORD[name] = {
            "SEQ": s.status,
            "B1": b.status,
            "B1 worst ratio": b.worst_ratio,
            "B1 rows violating": f"{b.rows_violating}/{b.rows_checked}",
        }

    def test_wrong_stride(self):
        self.check_both_fail("wrong_stride", ordinary_case())

    def test_last_block_dropped(self):
        self.check_both_fail("drop_last_block", ordinary_case())

    def test_subnormal_scales_flushed(self):
        case = subnormal_scale_case()
        # on ordinary scales the fault changes nothing (this input has none that are subnormal): not an escape, just no input
        o = ordinary_case()
        self.assertEqual(list(faults.flush_subnormal_scales(*o)), [int(v) for v in gemv.seq_bits(o[0], o[1], o[2], o[3])])
        self.check_both_fail("flush_subnormal_scales", case)

    def test_one_flipped_code(self):
        self.check_both_fail("flip_one_code", ordinary_case())

    def test_sign_of_one_row_flipped(self):
        raw, x, n, k = ordinary_case()
        self.assertNotEqual(int(gemv.seq_bits(raw, x, n, k)[0]) & 0x7FFFFFFF, 0)  # row 0 is not zero
        self.check_both_fail("flip_sign_of_one_row", (raw, x, n, k))

    def test_every_must_fail_fault_also_fails_seq_on_a_sweep(self):
        """SEQ fails whenever the fault changes any output; count the sweep so an escape is visible."""
        rng = np.random.default_rng(31)
        changed = detected = 0
        for name in faults.MUST_FAIL_B1:
            for sk, xk in (("typical", "normal"), ("subnormal", "normal"), ("typical", "wide")):
                for _ in range(4):
                    raw, x, n, k = gen.make_case(rng, 3, 64, sk, xk, "random")
                    good = [int(v) for v in gemv.seq_bits(raw, x, n, k)]
                    y = ALL_FAULTS[name](raw, x, n, k)
                    s = gemv.check_seq(raw, x, y, n, k)
                    if s.status == "could_not_evaluate":
                        continue
                    if [int(v) for v in y] != good:
                        changed += 1
                        self.assertEqual(s.status, "violated", name)
                        detected += 1
                    else:
                        self.assertEqual(s.status, "satisfied", name)
        self.assertEqual(changed, detected)
        self.assertGreater(changed, 30)


class LowestBit(unittest.TestCase):
    """SEQ must fail; B1 is recorded: the work order does not say which."""

    def test_lowest_bit_flipped(self):
        fn = faults.flip_lowest_bit_of_one_output
        rng = np.random.default_rng(41)
        counts = {"satisfied": 0, "violated": 0}
        for sk, xk in (("typical", "normal"), ("typical", "wide"), ("subnormal", "normal"), ("typical", "tiny")):
            for _ in range(5):
                case = gen.make_case(rng, 3, 64, sk, xk, "random")
                if gemv.check_b1(case[0], case[1], gemv.seq_bits(*[case[0], case[1], case[2], case[3]]), case[2], case[3]).status == "could_not_evaluate":
                    continue
                s, b = verdicts(fn, case)
                self.assertEqual(s.status, "violated")
                counts[b.status] += 1
        # A crafted input where the flip crosses the bound: d = 0.5, q = 1, x = -2^-149 in every column (k = 32). Each
        # product is a tie (-2^-150) that rounds to even, -0 -> the program's sum is +0.0; y* = -16 * 2^-149. The error of the
        # program is 32 eta, and B = 32 eta (1 + 2 gamma). Flipping the lowest bit of y = +0.0 gives +2^-149, whose error is
        # 17 * 2^-149 = 34 eta, above B.
        raw = orc.build([[(0x3800, [1] * 32)]])
        x = [0x80000001] * 32
        self.assertEqual(int(gemv.seq_bits(raw, x, 1, 32)[0]), 0)
        s, b = verdicts(fn, (raw, x, 1, 32))
        self.assertEqual(s.status, "violated")
        self.assertEqual(b.status, "violated")
        RECORD["flip_lowest_bit_of_one_output"] = {
            "SEQ": "violated on every input",
            "B1 on the sweep": f"{counts['satisfied']} satisfied, {counts['violated']} violated",
            "B1 on the crafted tie input": f"{b.status} (ratio {b.worst_ratio})",
        }
        self.assertGreater(counts["satisfied"], 10)


class DifferentPrograms(unittest.TestCase):
    """Reverse order, scale factored out, pairwise, binary64: different programs. SEQ must fail; B1 is observed."""

    def sweep_inputs(self):
        rng = np.random.default_rng(51)
        out = [tie_case()]
        for k in (32, 256):
            for sk, xk, qk in (
                ("typical", "normal", "random"),
                ("typical", "wide", "random"),
                ("typical", "tiny", "random"),
                ("typical", "subnormal", "random"),
                ("subnormal", "subnormal", "random"),
                ("subnormal", "normal", "extremes"),
                ("max_normal", "normal", "random"),
            ):
                for _ in range(6):
                    out.append(gen.make_case(rng, 3, k, sk, xk, qk))
        # heavy cancellation: +a x then -a x pairs split across the row, so the exact sum is 0
        for _ in range(6):
            xv = gen.xs(rng, 128, "wide")
            q = [int(v) for v in rng.integers(-127, 128, 128)]
            raw = orc.build([[(0x3C00, q[32 * i : 32 * (i + 1)]) for i in range(4)] + [(0x3C00, [-v for v in q[32 * i : 32 * (i + 1)]]) for i in range(4)]])
            out.append((raw, xv + xv, 1, 256))
        return out

    def run_fault(self, name):
        fn = ALL_FAULTS[name]
        seq_fail_on_tie = None
        changed = {"satisfied": 0, "violated": 0}
        worst = 0.0
        worst_where = None
        violating_examples = []
        for idx, case in enumerate(self.sweep_inputs()):
            raw, x, n, k = case
            y = fn(raw, x, n, k)
            s = gemv.check_seq(raw, x, y, n, k)
            if s.status == "could_not_evaluate":
                if idx == 0:
                    self.fail("the hand-built tie input must be inside the domain")
                continue
            if idx == 0:
                seq_fail_on_tie = s.status
            if s.status == "violated":  # the order matters on this input
                b = gemv.check_b1(raw, x, y, n, k)
                changed[b.status] += 1
                r = float(b.worst_ratio)
                if r > worst:
                    worst, worst_where = r, idx
                if b.status == "violated":
                    violating_examples.append(idx)
        self.assertEqual(seq_fail_on_tie, "violated", f"{name}: SEQ must fail on the tie input")
        RECORD[name] = {
            "SEQ on the tie input": seq_fail_on_tie,
            "B1 where SEQ differs": f"{changed['satisfied']} satisfied, {changed['violated']} violated",
            "worst |y-y*|/B seen": f"{worst:.6g} (input #{worst_where})",
        }
        if violating_examples:
            RECORD[name]["B1 exceeded on inputs"] = str(violating_examples)
        return changed

    def test_reverse_order_passes_b1(self):
        changed = self.run_fault("reverse_order")
        self.assertGreater(changed["satisfied"], 20)  # the order matters on these inputs, and B1 still holds
        self.assertEqual(changed["violated"], 0)

    def test_scale_factored_out_is_recorded(self):
        changed = self.run_fault("scale_factored_out")
        self.assertGreater(changed["satisfied"] + changed["violated"], 20)  # expectation, not a result: only recorded

    def test_pairwise_is_recorded(self):
        changed = self.run_fault("pairwise_sum")
        self.assertGreater(changed["satisfied"] + changed["violated"], 20)

    def test_binary64_accumulate_is_recorded(self):
        changed = self.run_fault("binary64_accumulate")
        self.assertGreater(changed["satisfied"] + changed["violated"], 20)

    def test_library_sum_is_not_the_program(self):
        """numpy's own float32 sum is pairwise: a fault written with it would fail SEQ, which the reference must not use."""
        raw, x, n, k = tie_case()
        prods = faults._w_times_x(raw, x, n, k)
        lib = np.sum(prods, axis=1, dtype=np.float32).view(np.uint32)
        self.assertNotEqual(list(lib), [int(v) for v in gemv.seq_bits(raw, x, n, k)])


def tearDownModule():
    order = list(ALL_FAULTS)
    print("\n\nGEMV fault record (SEQ / B1), as observed:")
    for name in order:
        if name in RECORD:
            print(f"  {name}:")
            for key, val in RECORD[name].items():
                print(f"      {key}: {val}")


if __name__ == "__main__":
    unittest.main()
