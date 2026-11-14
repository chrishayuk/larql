"""`larql-verifier patch`: a compiled container is judged against the op's operands and the base weights, not the ledger.

The "compiled" containers here are written with numpy's float64 arithmetic; the verifier recomputes on exact rationals.
Two routes, so agreement is evidence and a one-bit difference is caught.
"""

import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from larql_contract_verifier import cli  # noqa: E402
from larql_contract_verifier.contracts import structured_patch as S  # noqa: E402
from refcheck import container as C, synth  # noqa: E402

TENSOR = "0.mlp.down_proj.weight"
HF_KEY = "model.layers.0.mlp.down_proj.weight"
PROFILE = S.PROFILE_ONE_ROUNDING
HEX_64 = "{:016x}"


def f64_hex(x: float) -> str:
    return HEX_64.format(struct.unpack("<Q", struct.pack("<d", x))[0])


def bf16_round(values: np.ndarray) -> np.ndarray:
    """float64 array -> BF16-valued float32 array, ties to even, by the standard bit trick on the f32 pattern of an
    exactly-rounded f32 (the test inputs never need the double rounding this can introduce)."""
    f = values.astype(np.float32)
    u = f.view(np.uint32).astype(np.uint64)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return u.astype(np.uint32).view(np.float32)


def native_remove(w: np.ndarray, u: np.ndarray, alpha: float, columns: list) -> np.ndarray:
    out = w.copy()
    for j in columns:
        col = w[:, j].astype(np.float64)
        c = 0.0
        for r in range(len(col)):
            c = c + float(u[r]) * float(col[r])
        out[:, j] = bf16_round(np.array([float(col[r]) - (alpha * float(u[r])) * c for r in range(len(col))]))
    return out


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)
        self.model = synth.tiny_model(np.random.default_rng(3), layers=2)
        self.base = os.path.join(self.d, "base")
        self.written = synth.write_container(self.base, self.model)
        self.w = self.written.source[HF_KEY].copy()
        self.rows, self.cols = self.w.shape
        rng = np.random.default_rng(11)
        u = rng.standard_normal(self.rows)
        self.u = u / np.linalg.norm(u)
        self.alpha, self.columns = 0.9, [1, 4, 5]
        self.patch = os.path.join(self.d, "p.vlp")
        self.write_patch()

    def binding(self):
        c = C.open_container(self.base)
        rep = next(t.rep_id for t in c.tensors("BF16") if t.name == TENSOR)
        with open(os.path.join(self.base, "index.json"), "rb") as f:
            index_sha = hashlib.sha256(f.read()).hexdigest()
        return {"model": "m", "index_sha256": index_sha, "representation": rep,
                "payload_sha256": c.reps[rep]["payload_sha256"]}

    def op(self, **over):
        op = {"op": S.OP_NAME, "source": self.binding(), "tensor": TENSOR, "numerics": PROFILE,
              "operand": [f64_hex(float(x)) for x in self.u], "alpha": f64_hex(self.alpha),
              "columns": self.columns, "digest": "unchecked"}
        op.update(over)
        return op

    def write_patch(self, ops=None):
        with open(self.patch, "w") as f:
            json.dump({"version": 2, "base_model": "m", "operations": ops or [self.op()]}, f)

    def compiled(self, name, edited):
        src = dict(self.model)
        src[HF_KEY] = edited
        root = os.path.join(self.d, name)
        synth.write_container(root, {k: (self.written.source[k] if k != HF_KEY else edited) for k in src})
        return root

    def right(self):
        return native_remove(self.w, self.u, self.alpha, self.columns)

    def run_patch(self, container, patch=None):
        return cli.evaluate_patch(S.CONTRACT, patch or self.patch, self.base, container)


class Judgement(Base):
    def test_the_right_bytes_satisfy(self):
        out = self.run_patch(self.compiled("ok", self.right()))
        self.assertEqual(out["status"], "satisfied", out)
        self.assertEqual(out["violations"], 0)
        self.assertEqual(out["schema"], "larql-verify-patch/1")
        self.assertIn("ledger is not read", out["recomputed_from"])

    def test_an_unedited_container_is_violated(self):
        out = self.run_patch(self.compiled("noop", self.w))
        self.assertEqual(out["status"], "violated")
        self.assertGreater(out["violations"], 0)
        self.assertTrue(out["offending"][0]["name"].endswith(TENSOR))

    def test_one_flipped_bit_is_found(self):
        edited = self.right()
        raw = edited.view(np.uint32)
        raw[2, 4] ^= 1 << 16
        out = self.run_patch(self.compiled("flip", edited))
        self.assertEqual((out["status"], out["violations"]), ("violated", 1))
        self.assertIn("(2, 4)", out["offending"][0]["detail"])

    def test_an_edit_outside_the_declared_columns_is_found(self):
        edited = self.right()
        edited[0, 0] = bf16_round(np.array([float(edited[0, 0]) + 1.0]))[0]
        out = self.run_patch(self.compiled("stray", edited))
        self.assertEqual((out["status"], out["violations"]), ("violated", 1))

    def test_an_edit_to_another_tensor_is_found(self):
        src = {k: v.copy() for k, v in self.written.source.items()}
        src[HF_KEY] = self.right()
        other = "model.layers.1.mlp.up_proj.weight"
        src[other][0, 0] += 1.0
        root = os.path.join(self.d, "other")
        synth.write_container(root, src)
        out = self.run_patch(root)
        self.assertEqual(out["status"], "violated")
        self.assertTrue(out["offending"][0]["name"].endswith("1.mlp.up_proj.weight"))

    def test_the_ledger_is_never_read(self):
        root = self.compiled("noledger", self.right())
        for dirpath, _, names in os.walk(root):
            self.assertFalse([n for n in names if "ledger" in n or "weight_patch" in n])
        self.assertEqual(self.run_patch(root)["status"], "satisfied")


class Refusals(Base):
    def test_a_stale_binding_is_not_evaluated(self):
        binding = self.binding()
        binding["index_sha256"] = "0" * 64
        self.write_patch([self.op(source=binding)])
        out = self.run_patch(self.compiled("stale", self.right()))
        self.assertEqual(out["status"], "could_not_evaluate")
        self.assertIn("different container", out["reasons"][0])

    def test_an_f32_profile_is_not_implemented(self):
        self.write_patch([self.op(numerics="f32-seq-rne1/v1")])
        self.assertEqual(self.run_patch(self.compiled("f32", self.right()))["status"], "could_not_evaluate")

    def test_another_operation_kind_is_not_judged(self):
        self.write_patch([self.op(), {"op": "weight_delta"}])
        out = self.run_patch(self.compiled("mixed", self.right()))
        self.assertEqual(out["status"], "could_not_evaluate")

    def test_overlapping_columns_are_refused(self):
        self.write_patch([self.op(columns=[1, 4]), self.op(columns=[4, 7])])
        self.assertEqual(self.run_patch(self.compiled("overlap", self.right()))["status"], "could_not_evaluate")

    def test_an_operand_of_the_wrong_length_is_refused(self):
        self.write_patch([self.op(operand=[f64_hex(1.0)])])
        self.assertEqual(self.run_patch(self.compiled("short", self.right()))["status"], "could_not_evaluate")

    def test_a_column_outside_the_tensor_is_refused(self):
        self.write_patch([self.op(columns=[self.cols])])
        self.assertEqual(self.run_patch(self.compiled("wide", self.right()))["status"], "could_not_evaluate")

    def test_a_missing_patch_is_not_evaluated(self):
        out = cli.evaluate_patch(S.CONTRACT, os.path.join(self.d, "nope"), self.base, self.base)
        self.assertEqual(out["status"], "could_not_evaluate")

    def test_an_unknown_contract_is_not_evaluated(self):
        out = cli.evaluate_patch("OTHER", self.patch, self.base, self.base)
        self.assertEqual(out["status"], "could_not_evaluate")


OTHER_CANONICAL = "target.decoder_stack@Q8_0"


def as_canonical(index: dict) -> None:
    """Make the Q8_0 representation canonical (no `compiled_from`): a non-BF16 tensor the compile must leave alone."""
    e = index["representations"][OTHER_CANONICAL]
    e.pop("compiled_from")
    e.pop("source_representation_digest")


def flip_last_byte(rid: str, seg: bytes) -> bytes:
    return seg[:-1] + bytes([seg[-1] ^ 1]) if rid == OTHER_CANONICAL else seg


class NonBf16Canonical(Base):
    """`everything else is unchanged` covers canonical tensors that are not BF16: they are compared byte for byte."""

    def setUp(self):
        super().setUp()
        shutil.rmtree(self.base)
        self.base_q8 = synth.write_container(self.base, self.written.source, hooks=synth.Hooks(index=as_canonical)).q8_bytes
        self.write_patch()

    def compiled_with(self, name, hooks):
        src = dict(self.written.source)
        src[HF_KEY] = self.right()
        root = os.path.join(self.d, name)
        # canonical means the compile leaves it alone: the edited tensor's Q8_0 bytes stay the base's
        synth.write_container(root, src, hooks=hooks, q8_override=self.base_q8)
        return root

    def test_an_unchanged_non_bf16_canonical_tensor_satisfies_and_is_counted(self):
        out = self.run_patch(self.compiled_with("same", synth.Hooks(index=as_canonical)))
        self.assertEqual(out["status"], "satisfied", out)
        counted = {m["name"]: m["value"] for m in out["measurements"]}
        self.assertNotEqual(counted["other canonical tensors compared byte for byte (not BF16)"], "0")

    def test_a_changed_non_bf16_canonical_tensor_is_found(self):
        out = self.run_patch(self.compiled_with("q8", synth.Hooks(index=as_canonical, segment=flip_last_byte)))
        self.assertEqual(out["status"], "violated", out)
        self.assertTrue(any(o["name"].startswith(OTHER_CANONICAL) and "differs from the base" in o["detail"]
                            for o in out["offending"]), out["offending"])

    def test_a_derived_representation_is_not_judged(self):
        # without the hook the Q8_0 representation is derived (compiled_from) in the compiled container only
        out = self.run_patch(self.compiled_with("derived", synth.Hooks(segment=flip_last_byte)))
        self.assertEqual(out["status"], "violated", out)
        self.assertTrue(any("present in only one" in o["detail"] for o in out["offending"]), out["offending"])


class NonFiniteBase(Base):
    """The op reads every base element of a declared column; one that is not finite is refused, not crashed on."""

    def setUp(self):
        super().setUp()
        quiet = np.errstate(invalid="ignore")  # the synthetic Q8_0 encoder meets the non-finite value too
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def rebase_with(self, value, col):
        src = dict(self.written.source)
        w = self.w.copy()
        w[0, col] = value
        src[HF_KEY] = w
        shutil.rmtree(self.base)
        synth.write_container(self.base, src)
        self.w = w
        self.write_patch()

    def test_an_infinite_base_element_in_a_declared_column_is_not_evaluated(self):
        self.rebase_with(np.inf, self.columns[0])
        out = self.run_patch(self.compiled("inf", self.w))
        self.assertEqual(out["status"], "could_not_evaluate", out)
        self.assertIn("not finite", out["reasons"][0])

    def test_a_nan_base_element_in_a_declared_column_is_not_evaluated(self):
        self.rebase_with(np.nan, self.columns[0])
        self.assertEqual(self.run_patch(self.compiled("nan", self.w))["status"], "could_not_evaluate")

    def test_an_infinite_base_element_outside_the_columns_is_judged(self):
        self.rebase_with(np.inf, 0)
        self.assertNotIn(0, self.columns)
        out = self.run_patch(self.compiled("inf_outside", self.right()))
        self.assertEqual(out["status"], "satisfied", out)


class Arithmetic(unittest.TestCase):
    """The exact-rational route against the host's floating point, on the cases where they could disagree."""

    def bits(self, base_values, u, alpha, via_f32=False):
        rows = len(u)
        base = [struct.unpack("<I", struct.pack("<f", v))[0] >> 16 for v in base_values]
        nums = [S._from_f64_bits(struct.unpack("<Q", struct.pack("<d", x))[0]) for x in u]
        a = S._from_f64_bits(struct.unpack("<Q", struct.pack("<d", alpha))[0])
        return S.remove_direction(base, rows, 1, nums, a, [0], via_f32)

    def test_the_agreement_with_numpy_on_random_columns(self):
        rng = np.random.default_rng(5)
        for _ in range(200):
            w = bf16_round(rng.standard_normal(16) * 10.0 ** rng.integers(-3, 3))
            u = rng.standard_normal(16)
            u /= np.linalg.norm(u)
            alpha = float(rng.uniform(0.5, 1.5))
            got = self.bits(list(w), list(u), alpha)
            want = native_remove(w.reshape(16, 1), u, alpha, [0]).reshape(16)
            want_bits = [struct.unpack("<I", struct.pack("<f", float(x)))[0] >> 16 for x in want]
            self.assertEqual([got[r] for r in range(16)], want_bits)

    def test_a_near_cancellation_leaves_a_tiny_nonzero(self):
        # u is a unit row vector, so u . w = w[0]; alpha = 1 + 2^-30 leaves w[0] * -2^-30 behind
        w = [1.0] + [0.0] * 3
        u = [1.0, 0.0, 0.0, 0.0]
        got = self.bits(w, u, 1.0 + 2.0**-30)
        self.assertEqual(got[0], struct.unpack("<I", struct.pack("<f", -(2.0**-30)))[0] >> 16)

    def test_exact_cancellation_is_positive_zero(self):
        got = self.bits([2.0, 0.0], [1.0, 0.0], 1.0)
        self.assertEqual(got[0], 0x0000)

    def test_an_underflow_keeps_its_sign(self):
        tiny = struct.unpack("<f", struct.pack("<I", 0x0001 << 16))[0]  # the smallest positive BF16 subnormal
        got = self.bits([tiny, 0.0], [1.0, 0.0], 1.0 + 2.0**-20)
        # tiny - (1 + 2^-20) * tiny = -2^-153: below half the smallest subnormal, so it rounds to -0, sign kept
        self.assertEqual(got[0], 0x8000)

    def test_ties_round_to_even(self):
        # 1 + 2^-8 is exactly halfway between the BF16 values 1 and 1 + 2^-7; the even one is 1
        x = S.Num(Fraction_one_plus(2**-8))
        self.assertEqual(S.to_bf16_bits(x, False), 0x3F80)
        y = S.Num(Fraction_one_plus(3 * 2**-8))  # halfway between 1+2^-7 and 1+2^-6: the even one is 1+2^-6
        self.assertEqual(S.to_bf16_bits(y, False), 0x3F82)

    def test_the_double_rounding_profile_can_differ(self):
        # just above a BF16 tie, by less than an f32 can show: via f32 it lands exactly on the tie, then rounds to even
        from fractions import Fraction
        x = S.Num(Fraction_one_plus(2**-8) + Fraction(1, 2**40))
        self.assertEqual(S.to_bf16_bits(x, False), 0x3F81)
        self.assertEqual(S.to_bf16_bits(x, True), 0x3F80)

    def test_overflow_is_refused(self):
        with self.assertRaises(S.Refused):
            S.to_bf16_bits(S.Num(Fraction_one_plus(0) * 2**128), False)


class Positions(unittest.TestCase):
    def test_a_flat_index_is_located_in_a_tensor_of_any_rank(self):
        self.assertEqual(S._position(7, (16,)), "(7)")
        self.assertEqual(S._position(7, (2, 4)), "(1, 3)")
        self.assertEqual(S._position(13, (2, 3, 4)), "(1, 0, 1)")


def Fraction_one_plus(x):
    from fractions import Fraction
    return Fraction(1) + Fraction(x)


class Axes(Base):
    """A verdict on the weights is not a verdict on the ledger or on provenance."""

    def check(self, out, numerical, binding):
        self.assertEqual(out["numerical_satisfaction"], numerical)
        self.assertEqual(out["status"], numerical, "the top-level verdict is unchanged")
        self.assertEqual(out["ledger_integrity"], "not_checked")
        self.assertEqual(out["provenance"], "not_checked")
        self.assertEqual(out["provenance_detail"], {"base_binding": binding, "operand_origin": "not_checked",
                                                    "operand_derivation": "not_checked", "op_digest": "not_checked"})

    def test_satisfied_weights_say_nothing_of_the_ledger_or_provenance(self):
        self.check(self.run_patch(self.compiled("ok", self.right())), "satisfied", "satisfied")

    def test_violated_weights_still_leave_the_ledger_and_provenance_unchecked(self):
        self.check(self.run_patch(self.compiled("noop", self.w)), "violated", "satisfied")

    def test_a_refusal_establishes_nothing_including_the_binding(self):
        self.write_patch([self.op(source=dict(self.binding(), index_sha256="0" * 64))])
        self.check(self.run_patch(self.compiled("ok", self.right())), "could_not_evaluate", "could_not_evaluate")

    def test_a_missing_patch_is_an_unchecked_everything(self):
        out = self.run_patch(self.compiled("ok", self.right()), patch=os.path.join(self.d, "absent.vlp"))
        self.check(out, "could_not_evaluate", "could_not_evaluate")

    def test_a_corrupted_ledger_changes_neither_the_weights_verdict_nor_claims_a_ledger_verdict(self):
        root = self.compiled("ledgered", self.right())
        with open(os.path.join(root, "weight_patches.json"), "w") as f:
            f.write("{ not json")
        self.check(self.run_patch(root), "satisfied", "satisfied")


class CommandLine(Base):
    def test_the_command_line_prints_one_object(self):
        import subprocess
        root = self.compiled("cli", self.right())
        env = dict(os.environ, PYTHONPATH=ROOT)
        run = subprocess.run([sys.executable, "-m", "larql_contract_verifier.cli", "patch", "--contract", S.CONTRACT,
                              "--patch", self.patch, "--base", self.base, "--container", root],
                             capture_output=True, text=True, env=env, cwd=ROOT)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["status"], "satisfied")


if __name__ == "__main__":
    unittest.main()
