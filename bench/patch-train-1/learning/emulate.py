"""Bit-exact emulator of the frozen VLP numerics v2 op `add_low_rank_columns` (conformance/vlp-numerics/v2/SPEC.md).

PATCH-TRAIN-1 L1. Written from SPEC.md sections 2a, 3, 4 and 5 (and v1's sections 1, 2) with numpy; it imports nothing from
the v1/v2 generators or the independent evaluator. It exists so an optimiser can evaluate a candidate patch's ROUNDED result
quickly. It is the EXACT numerical path only: no gradient or straight-through estimator lives here.

Exactness rests on three facts, checked at import by `_canaries()`:
  * numpy float64 / float32 `multiply` and `add` on arrays are single IEEE-754 operations, round to nearest even, with no
    fused multiply-add (they are separate ufunc calls) and no flush-to-zero;
  * f64 -> BF16 is ONE rounding done on the integer bit pattern (never through f32, which would double round); the
    `rne2` chain rounds f64 -> f32 (numpy's cast) and then f32 -> BF16 on the bit pattern;
  * an element whose terms are all zero is SELECTED as untouched and never run through the arithmetic.

Public API:
  apply_columns(base_bits, columns, a, b, profile) -> (new_bits, written)   raises Refusal(code)
  run_case(case) -> ("bf16", [int ...]) | ("refuse", code)                  the frozen vector schema
"""
import numpy as np

PROFILE_RNE1, PROFILE_RNE2, PROFILE_F32 = "f64-seq-rne1/v1", "f64-seq-rne2/v1", "f32-seq-rne1/v1"
PROFILES = (PROFILE_RNE1, PROFILE_RNE2, PROFILE_F32)
BF16_INF_BITS = 0x7F80            # the bits of +infinity; a rounded magnitude at or above it overflowed
BF16_SUBNORMAL_QUANTUM_EXP = -133  # 2^emin * 2^-(p-1) = 2^-126 * 2^-7
F64_MANTISSA_BITS, BF16_MANTISSA_BITS = 52, 7
F64_TO_BF16_SHIFT = F64_MANTISSA_BITS - BF16_MANTISSA_BITS      # 45
F32_TO_BF16_SHIFT = 16
F64_EXPONENT_BIAS, BF16_EXPONENT_BIAS = 1023, 127
F64_MIN_BF16_NORMAL_EXP_FIELD = F64_EXPONENT_BIAS - 126         # 897: |x| >= 2^-126
SIGN_BIT_16 = 0x8000
BF16_EXPONENT_MASK = 0x7F80


class Refusal(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _canaries():
    """v1 requires an implementation to refuse to run when it cannot guarantee subnormals are kept (no FTZ/DAZ)."""
    tiny64 = np.array([5e-324])
    tiny32 = np.array([1e-45], dtype=np.float32)
    one64, one32 = np.array([1.0]), np.array([1.0], dtype=np.float32)
    ok = (tiny64 * one64)[0] != 0 and (tiny64 + tiny64)[0] == 1e-323
    ok = ok and (tiny32 * one32)[0] != 0 and (tiny32 + tiny32)[0] == np.float32(3e-45)
    if not ok:
        raise RuntimeError("flush-to-zero or denormals-are-zero is in effect; refusing to run (v1 section 1)")


_canaries()


# ------------------------------------------------------------------ rounding to BF16, on bit patterns

def f64_to_bf16(x):
    """float64 array of FINITE values -> (bf16 bits uint16, overflowed bool). One rounding, ties to even, gradual underflow,
    a nonzero value that rounds to zero keeps its sign, a zero keeps its sign."""
    x = np.asarray(x, dtype=np.float64)
    u = x.view(np.uint64)
    sign = ((u >> np.uint64(63)) & np.uint64(1)).astype(np.uint16) << np.uint16(15)
    mag = u & np.uint64(0x7FFFFFFFFFFFFFFF)
    exp_field = (mag >> np.uint64(F64_MANTISSA_BITS)).astype(np.int64)
    # normal range of BF16: round the 52-bit mantissa to 7 bits, nearest even, carrying into the exponent
    add = np.uint64((1 << (F64_TO_BF16_SHIFT - 1)) - 1) + ((mag >> np.uint64(F64_TO_BF16_SHIFT)) & np.uint64(1))
    rounded = (mag + add) >> np.uint64(F64_TO_BF16_SHIFT)            # = (exp64 << 7) | mant7, after the carry
    rebased = rounded.astype(np.int64) - ((F64_EXPONENT_BIAS - BF16_EXPONENT_BIAS) << BF16_MANTISSA_BITS)
    # below 2^-126 BF16 is subnormal with quantum 2^-133: scale exactly by a power of two, round half to even
    small = exp_field < F64_MIN_BF16_NORMAL_EXP_FIELD
    scaled = np.where(small, np.abs(x), 0.0) * np.float64(2.0) ** (-BF16_SUBNORMAL_QUANTUM_EXP)
    subnormal_bits = np.rint(scaled).astype(np.int64)                # np.rint is round-half-to-even
    bits = np.where(small, subnormal_bits, rebased)
    overflow = bits >= BF16_INF_BITS
    return (np.where(overflow, 0, bits).astype(np.uint16) | sign), overflow


def f32_to_bf16(x):
    """float32 array of FINITE values -> (bf16 bits, overflowed). The standard round-to-nearest-even on the bit pattern."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    sign = ((u >> np.uint32(31)) & np.uint32(1)).astype(np.uint16) << np.uint16(15)
    mag = u & np.uint32(0x7FFFFFFF)
    add = np.uint32(0x7FFF) + ((mag >> np.uint32(F32_TO_BF16_SHIFT)) & np.uint32(1))
    rounded = (mag + add) >> np.uint32(F32_TO_BF16_SHIFT)
    overflow = rounded >= np.uint32(BF16_INF_BITS)
    return (np.where(overflow, 0, rounded).astype(np.uint16) | sign), overflow


def bf16_to_f64(bits):
    """Exact widening of BF16 bit patterns (any value, including NaN/inf)."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)


def bf16_to_f32(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


# ------------------------------------------------------------------ the op

def apply_columns(base_bits, columns, a, b, profile):
    """The op on arrays. base_bits uint16 [rows, cols]; columns strictly increasing non-negative ints; a [rows, K] and
    b [K, len(columns)] already in the working format (float64 for the f64 profiles, float32 for the f32 profile).
    Returns (new bits uint16 [rows, cols], written bool [rows, len(columns)]). Raises Refusal(code)."""
    if profile not in PROFILES:
        raise Refusal("numerics_invalid")
    work = np.float32 if profile == PROFILE_F32 else np.float64
    base_bits = np.asarray(base_bits, dtype=np.uint16)
    rows, cols = base_bits.shape
    columns = np.asarray(columns, dtype=np.int64)
    a, b = np.asarray(a, dtype=work), np.asarray(b, dtype=work)
    if a.shape[0] != rows or a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0] or b.shape[1] != len(columns):
        raise Refusal("shape_invalid")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise Refusal("non_finite_operand")
    block_bits = base_bits[:, columns]
    if (block_bits & np.uint16(BF16_EXPONENT_MASK) == np.uint16(BF16_EXPONENT_MASK)).any():   # NaN or inf inside rows x J
        raise Refusal("non_finite_operand")
    # selection first, on the operands, before any arithmetic: written iff some term has two nonzero factors
    written = ((a != 0)[:, :, None] & (b != 0)[None, :, :]).any(axis=1)
    w = bf16_to_f64(block_bits).astype(work)
    with np.errstate(all="ignore"):
        for k in range(a.shape[1]):                      # sequential in k; one rounding per operation; never fused
            product = np.multiply(a[:, k:k + 1], b[k:k + 1, :])
            w = np.add(w, product)
        finite = np.isfinite(w)                          # an inf/NaN can only have come from an overflowing W operation
        safe = np.where(finite, w, work(0))
        if profile == PROFILE_RNE1:
            bits, over = f64_to_bf16(safe)
        elif profile == PROFILE_RNE2:
            narrow = safe.astype(np.float32)
            f32_over = ~np.isfinite(narrow)              # the f32 step of the chain overflowed
            bits, over = f32_to_bf16(np.where(f32_over, np.float32(0), narrow))
            over = over | f32_over
        else:
            bits, over = f32_to_bf16(safe)
    # a refusal is decided by the first failing element in (row, p) order; within an element a W-operation overflow comes
    # before a chain overflow
    failed = written & (~finite | over)
    if failed.any():
        first = np.flatnonzero(failed.reshape(-1))[0]
        raise Refusal("non_finite_intermediate" if not finite.reshape(-1)[first] else "overflow")
    out = base_bits.copy()
    block = np.where(written, bits, block_bits)
    out[:, columns] = block
    return out, written


# ------------------------------------------------------------------ the whole contract, on the frozen vector schema

def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _bits_to_array(hexes, profile):
    if profile == PROFILE_F32:
        return np.array([int(h, 16) for h in hexes], dtype=np.uint32).view(np.float32)
    return np.array([int(h, 16) for h in hexes], dtype=np.uint64).view(np.float64)


def run_case(case):
    """The refusal order of SPEC section 5, then the op. ("bf16", [ints]) or ("refuse", code)."""
    try:
        return ("bf16", _run_case(case))
    except Refusal as r:
        return ("refuse", r.code)


def _run_case(case):
    op, base = case["op"], case["base"]
    profile = op.get("numerics")
    if profile not in PROFILES:
        raise Refusal("numerics_invalid")
    shape = op.get("shape")
    if not (isinstance(shape, list) and len(shape) == 2 and all(_is_int(x) and x > 0 for x in shape)):
        raise Refusal("shape_invalid")
    rows, cols = shape
    fixups = op.get("fixups", [])
    keys = [(f[0], f[1]) for f in fixups]
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        raise Refusal("fixups_invalid")
    if any((int(f[2], 16) & BF16_EXPONENT_MASK) == BF16_EXPONENT_MASK for f in fixups):
        raise Refusal("non_finite_operand")
    columns = op.get("columns")
    if not columns or columns != sorted(set(columns)) or not all(_is_int(c) and c >= 0 for c in columns):
        raise Refusal("columns_invalid")
    rank = op.get("rank")
    if not _is_int(rank) or rank < 1:
        raise Refusal("rank_invalid")
    n = len(columns)
    if len(op["a"]) != rows * rank or len(op["b"]) != rank * n:
        raise Refusal("shape_invalid")
    a = _bits_to_array(op["a"], profile).reshape(rows, rank)
    b = _bits_to_array(op["b"], profile).reshape(rank, n)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise Refusal("non_finite_operand")
    # header phase
    if base["shape"] != [rows, cols] or len(base["bf16"]) != rows * cols:
        raise Refusal("shape_invalid")
    if case.get("tensor_role") == "embed_tokens":
        raise Refusal("tensor_refused")
    if columns[-1] >= cols:
        raise Refusal("columns_invalid")
    inside = set(columns)
    for r, c, _ in fixups:
        if not (0 <= r < rows and 0 <= c < cols) or c not in inside:
            raise Refusal("fixups_invalid")
    base_bits = np.array([int(h, 16) for h in base["bf16"]], dtype=np.uint16).reshape(rows, cols)
    out, _ = apply_columns(base_bits, columns, a, b, profile)
    flat = out.reshape(-1)
    computed = flat.copy()
    for r, c, h in fixups:
        if int(h, 16) == int(computed[r * cols + c]):
            raise Refusal("fixup_no_op")
        flat[r * cols + c] = int(h, 16)
    return [int(x) for x in flat]
