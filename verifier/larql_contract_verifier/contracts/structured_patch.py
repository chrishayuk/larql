"""`STRUCTURED_REMOVE_DIRECTION`: does a compiled container hold what a `.vlp` structured `remove_direction` op declares?

The check is a second route, and it is deliberately not a reading of what the compiler wrote down about itself:

* It reads the patch (operand bits, alpha bits, columns, numerics profile, source binding), the BASE container's
  weights and the COMPILED container's weights. It never reads the weight-patch ledger.
* It recomputes the expected BF16 bits of every element from those inputs. Every IEEE-754 operation is done on exact
  rationals and rounded once, with `refcheck.numconv.round_nearest_even`, rather than on the host's floating point. A
  zero's sign is tracked by hand, because a rational has none.
* It then compares EVERY element of the edited tensor (inside the declared columns against the recomputed bits, outside
  against the base) and every other BF16 tensor of the container against the base, byte for byte.

What it does not establish, and says so in the answer: the derived (quantised) representations a compile re-derives, the
digest the op carries, and anything about the operand's origin. A result is a numerical measurement of two containers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from fractions import Fraction

from refcheck import container as C
from refcheck import numconv

from .. import checkpoint

CONTRACT = "STRUCTURED_REMOVE_DIRECTION"
STATEMENT = (
    "every element of the compiled tensor equals the BF16 result of the declared remove_direction operation "
    "recomputed from its operands and the base weights, and everything else is unchanged"
)
OP_NAME = "structured_weight"
#: Profiles this verifier implements. The f32 working profile is not: it is refused by the compiler as well.
PROFILE_ONE_ROUNDING = "f64-seq-rne1/v1"
PROFILE_VIA_F32 = "f64-seq-rne2/v1"
PROFILES = (PROFILE_ONE_ROUNDING, PROFILE_VIA_F32)
BF16_FORMAT = (8, -126, 127)
F64_HEX_DIGITS = 16
BF16_SIGN = 0x8000
BF16_BIAS = 127
BF16_MANTISSA_BITS = 7
MIN_SUBNORMAL_EXPONENT = -133
OFFENDING_LIMIT = 10
BAD_INPUT = (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, C.ContainerError)


class Refused(Exception):
    """The declared operation cannot be evaluated; the reason is the message."""


def could_not(out: dict, *reasons: str) -> dict:
    out.update(status="could_not_evaluate", reasons=list(reasons))
    return out


# ── exact arithmetic ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Num:
    """An IEEE-754 binary64 value held exactly: a rational, and the sign a zero carries."""

    value: Fraction
    negative_zero: bool = False

    def negative(self) -> bool:
        return self.value < 0 or (self.value == 0 and self.negative_zero)


def _from_f64_bits(bits: int) -> Num:
    (x,) = struct.unpack("<d", struct.pack("<Q", bits))
    if math.isnan(x) or math.isinf(x):
        raise Refused("an operand is not finite")
    return Num(Fraction(x), x == 0 and math.copysign(1.0, x) < 0)


def _from_bf16_bits(bits: int) -> Num:
    (x,) = struct.unpack("<f", struct.pack("<I", bits << 16))
    return Num(Fraction(x), x == 0 and bool(bits & BF16_SIGN))


def _round64(exact: Fraction, negative_if_zero: bool) -> Num:
    try:
        rounded = numconv.round_nearest_even(exact, numconv.BINARY64)
    except numconv.Overflow:
        raise Refused("non_finite_intermediate: a result overflows binary64") from None
    return Num(rounded, rounded == 0 and negative_if_zero)


def mul(a: Num, b: Num) -> Num:
    exact = a.value * b.value
    return _round64(exact, a.negative() != b.negative())


def add(a: Num, b: Num) -> Num:
    exact = a.value + b.value
    if exact == 0:
        if a.value == 0 and b.value == 0:
            return Num(Fraction(0), a.negative_zero and b.negative_zero)
        return Num(Fraction(0), False)  # exact cancellation rounds to +0 under round-to-nearest
    return _round64(exact, exact < 0)


def sub(a: Num, b: Num) -> Num:
    return add(a, Num(-b.value, b.value == 0 and not b.negative_zero))


def _floor_log2(a: Fraction) -> int:
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    elif Fraction(2) ** (e + 1) <= a:
        e += 1
    return e


def to_bf16_bits(x: Num, via_f32: bool) -> int:
    """`x` as BF16 bits: optionally through binary32 first (the double-rounding profile), then one BF16 rounding."""
    value, negative = x.value, x.negative()
    if via_f32:
        try:
            value = numconv.round_nearest_even(value, numconv.BINARY32)
        except numconv.Overflow:
            raise Refused("overflow at the f32 step") from None
    try:
        value = numconv.round_nearest_even(value, BF16_FORMAT)
    except numconv.Overflow:
        raise Refused("overflow: a result does not fit BF16") from None
    sign = BF16_SIGN if negative else 0
    if value == 0:
        return sign
    a = abs(value)
    e = _floor_log2(a)
    if e < BF16_FORMAT[1]:
        return sign | int(a / Fraction(2) ** MIN_SUBNORMAL_EXPONENT)
    mantissa = int((a / Fraction(2) ** e - 1) * (1 << BF16_MANTISSA_BITS))
    return sign | ((e + BF16_BIAS) << BF16_MANTISSA_BITS) | mantissa


def remove_direction(base: list[int], rows: int, cols: int, u: list[Num], alpha: Num, columns: list[int],
                     via_f32: bool) -> dict[int, int]:
    """`{element index: expected BF16 bits}` for every element in `columns`: `w - (alpha * u) * (u . w)`, each IEEE
    operation rounded to binary64, the dot product accumulated in row order from +0."""
    expected = {}
    for j in columns:
        w = [_from_bf16_bits(base[r * cols + j]) for r in range(rows)]
        c = Num(Fraction(0), False)
        for r in range(rows):
            c = add(c, mul(u[r], w[r]))
        for r in range(rows):
            expected[r * cols + j] = to_bf16_bits(sub(w[r], mul(mul(alpha, u[r]), c)), via_f32)
    return expected


# ── the patch and the containers ───────────────────────────────────────────


@dataclass(frozen=True)
class Op:
    tensor: str
    numerics: str
    u: list[Num]
    alpha: Num
    columns: list[int]
    index_sha256: str
    representation: str
    payload_sha256: str


def _hex64(text, what: str) -> int:
    if not isinstance(text, str) or len(text) != F64_HEX_DIGITS or any(c not in "0123456789abcdef" for c in text):
        raise Refused(f"{what} is not {F64_HEX_DIGITS} lowercase hex digits")
    return int(text, 16)


def read_ops(patch_path: str) -> list[Op]:
    with open(patch_path, "rb") as f:
        patch = json.loads(f.read())
    operations = patch.get("operations")
    if not isinstance(operations, list) or not operations:
        raise Refused("the patch has no operations")
    ops = []
    for raw in operations:
        if not isinstance(raw, dict) or raw.get("op") != OP_NAME:
            raise Refused(f"the patch holds an operation other than {OP_NAME!r}; this contract judges only those")
        if raw["numerics"] not in PROFILES:
            raise Refused(f"numerics profile {raw['numerics']!r} is not one this verifier implements {list(PROFILES)}")
        columns = raw["columns"]
        if not columns or any(not isinstance(c, int) or isinstance(c, bool) for c in columns) \
                or any(b <= a for a, b in zip(columns, columns[1:])):
            raise Refused("columns must be non-empty and strictly increasing")
        source = raw["source"]
        ops.append(Op(raw["tensor"], raw["numerics"], [_from_f64_bits(_hex64(h, "an operand element")) for h in raw["operand"]],
                      _from_f64_bits(_hex64(raw["alpha"], "alpha")), columns, source["index_sha256"],
                      source["representation"], source["payload_sha256"]))
    return ops


def _bf16_tensors(root: str):
    """Every canonical BF16 tensor, by `(representation, name)`: a name is only unique within its representation."""
    c = C.open_container(root)
    return {(ref.rep_id, ref.name): ref for ref in c.tensors("BF16")}, c


def _read(ref: C.TensorRef) -> bytes:
    with open(ref.segment_path, "rb") as f:
        f.seek(ref.abs_start)
        return f.read(ref.abs_end - ref.abs_start)


def _position(index: int, shape: tuple) -> str:
    """A flat element index as the tuple of coordinates it is at in a tensor of this shape (any rank)."""
    coords = []
    for extent in reversed(shape):
        coords.append(index % extent)
        index //= extent
    return "(" + ", ".join(str(c) for c in reversed(coords)) + ")"


def _bits(raw: bytes) -> list[int]:
    return list(struct.unpack(f"<{len(raw) // 2}H", raw))


def _sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def evaluate(out: dict, patch: str, base: str, container: str) -> dict:
    out["recomputed_from"] = "operands and base weights; the weight-patch ledger is not read"
    try:
        ops = read_ops(patch)
        base_tensors, base_c = _bf16_tensors(base)
        new_tensors, _ = _bf16_tensors(container)
        base_index_sha = _sha256_file(os.path.join(base, "index.json"))
        return _judge(out, ops, base_tensors, base_c, new_tensors, base_index_sha)
    except Refused as e:
        return could_not(out, str(e))
    except BAD_INPUT as e:
        return could_not(out, f"an input can't be read: {type(e).__name__}: {e}")


def _judge(out, ops, base_tensors, base_c, new_tensors, base_index_sha) -> dict:
    footprint: dict[str, set[int]] = {}
    expected_bits: dict[str, dict[int, int]] = {}
    for op in ops:
        key = (op.representation, op.tensor)
        if key not in base_tensors or key not in new_tensors:
            raise Refused(f"tensor {op.tensor!r} of {op.representation!r} is absent from the base or the compiled container")
        base_ref, new_ref = base_tensors[key], new_tensors[key]
        if base_ref.shape != new_ref.shape or len(base_ref.shape) != 2:
            raise Refused(f"tensor {op.tensor!r} changed shape or is not a matrix")
        rows, cols = base_ref.shape
        if op.index_sha256 != base_index_sha:
            raise Refused(f"the op was bound to a different container: index {op.index_sha256[:12]} vs {base_index_sha[:12]}")
        if op.payload_sha256 != base_c.reps[op.representation]["payload_sha256"]:
            raise Refused(f"the op's payload digest does not match the base's representation {op.representation!r}")
        if len(op.u) != rows or op.columns[-1] >= cols:
            raise Refused(f"operand length {len(op.u)} or columns {op.columns} do not fit a {rows}x{cols} tensor")
        mine = set(op.columns)
        if mine & footprint.setdefault(key, set()):
            raise Refused(f"two ops declare the same column of {op.tensor!r}")
        footprint[key] |= mine
        base_bits = _bits(_read(base_ref))
        expected_bits.setdefault(key, {}).update(
            remove_direction(base_bits, rows, cols, op.u, op.alpha, op.columns, op.numerics == PROFILE_VIA_F32))

    offending, checked, expected_changes, violations = [], 0, 0, 0
    for key, base_ref in sorted(base_tensors.items()):
        name = f"{key[0]}/{key[1]}"
        new_ref = new_tensors.get(key)
        if new_ref is None or new_ref.shape != base_ref.shape:
            violations += 1
            offending.append({"name": name, "violations": 1, "detail": "absent or reshaped in the compiled container"})
            continue
        base_bits, new_bits = _bits(_read(base_ref)), _bits(_read(new_ref))
        want = expected_bits.get(key, {})
        bad, first = 0, None
        for i, (b, n) in enumerate(zip(base_bits, new_bits)):
            target = want.get(i, b)
            if n != target:
                bad += 1
                first = first or (i, target, n)
        checked += len(base_bits)
        expected_changes += sum(1 for i, t in want.items() if t != base_bits[i])
        if bad:
            violations += bad
            i, target, n = first
            offending.append({"name": name, "violations": bad,
                              "detail": f"first at {_position(i, base_ref.shape)}: expected {target:04x}, found {n:04x}"})
    only_new = sorted(set(new_tensors) - set(base_tensors))
    if only_new:
        violations += len(only_new)
        offending.append({"name": "/".join(only_new[0]), "violations": len(only_new), "detail": "tensors absent from the base"})
    offending.sort(key=lambda o: -o["violations"])
    out.update(status="violated" if violations else "satisfied", violations=violations,
               offending=offending[:OFFENDING_LIMIT],
               measurements=[{"name": "elements compared", "value": str(checked)},
                             {"name": "elements expected to change", "value": str(expected_changes)},
                             {"name": "elements differing from the recomputed bits", "value": str(violations)},
                             {"name": "ops recomputed", "value": str(len(ops))},
                             {"name": "not established",
                              "value": "derived (quantised) representations, the op digest, the operand's origin"}])
    return out
