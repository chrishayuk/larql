"""`Q8_0_PACK_DERIVATION`: is every re-derived Q8_0 representation of a container exactly the pack its declared
derivation rule produces from the container's own canonical BF16 weights?

Written from `docs/verification/q8-0-pack-derivation.md`. The expected bytes are derived here, from the canonical
tensor bytes, by the rule the pack's index entry names. Nothing the engine wrote is used as an expected value: its
digests are checked, and its bytes are what is judged. The engine's packer is not called and no code is shared with it.

**What a `satisfied` verdict says.** Every block of every Q8_0 tensor of every re-derived representation (the stored
scale and all 32 codes) is byte for byte the rule's result, and every tensor the pack carries unquantised is byte for
byte the canonical tensor.

**What it does not say, and says so in the answer:**

* that the canonical weights are the intended ones. That is another contract's claim (`WEIGHT_PATCHES_V4` for an edited
  container, `Q8_0_T2` against an upstream checkpoint). This one starts from the canonical bytes as they are.
* that the right tensors were chosen for quantisation. Which tensors are Q8_0 is read from the pack's own tensor table;
  the policy that chose them is not re-derived.
* that a runtime kernel reads these bytes, or reads them correctly. Byte equivalence of the pack is not execution.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from refcheck import container as C
from refcheck import numconv
from refcheck import q8_0 as Q

from . import vlp_numerics as V
from .weight_patches_v4 import Store, container_digest, file_digests

CONTRACT = "Q8_0_PACK_DERIVATION"
STATEMENT = (
    "every re-derived Q8_0 representation is byte for byte the pack its declared derivation rule produces from the "
    "container's own canonical BF16 weights: every block's stored scale and all its codes, and every tensor carried "
    "unquantised"
)
#: The one derivation rule this contract implements (§2). A pack that names another is not judged.
ENCODER = {"algorithm": "kquant-larql-native", "revision": 2}
BLOCK = Q.QK8_0
BLOCK_BYTES = Q.BLOCK_BYTES
#: §2: the scale is the block's largest magnitude over this, in binary32.
SCALE_DIVISOR = 127
CODE_MIN, CODE_MAX = -128, 127
BINARY16 = (11, -14, 15)
F32_EXPONENT_MASK = 0x7F800000
F32_SIGNIFICAND_BITS = 24
#: The exponent of the unit of a binary32 significand taken as an integer: value = significand × 2^(field − this).
F32_UNIT_BIAS = 150
F16_INFINITY = 0x7C00
F16_MIN_EXPONENT = -14
F16_FRACTION_BITS = 10
OFFENDING_LIMIT = 10
BAD_INPUT = (OSError, ValueError, KeyError, TypeError, IndexError, C.ContainerError)
NOT_ESTABLISHED = (
    "that the canonical weights are the intended ones (a ledger or source contract's claim, not this one's)",
    "that the right tensors were chosen for quantisation (read from the pack's own table, not re-derived from policy)",
    "that a runtime kernel consumes these bytes, or consumes them correctly",
)


class Unsupported(Exception):
    """The pack or its source is outside what the rule defines: never a pass."""


# ── the derivation rule (§2), vectorised ──────────────────────────────────────


def bf16_to_f32(raw: bytes) -> np.ndarray:
    """Canonical BF16 bytes as binary32 values: each little-endian 16-bit pattern is the top half of the binary32
    (exact)."""
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


def f32_to_f16_nearest_bits(x: np.ndarray) -> np.ndarray:
    """IEEE 754 binary32 -> binary16, round to nearest, ties to even, subnormals kept, overflow to infinity. Integer
    bit manipulation on the bit patterns, not NumPy's float16, so the conversion does not lean on a library type.
    Takes finite non-negative values (a scale is never anything else)."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.int64)
    field = (bits >> 23) & 0xFF
    fraction = bits & 0x7FFFFF
    significand = np.where(field > 0, fraction | (1 << 23), fraction)
    unit = np.maximum(field, 1) - F32_UNIT_BIAS  # value = significand × 2^unit
    # floor(log2(value)) for a normal binary32; a binary32 subnormal is far below the binary16 grid either way
    exponent = np.where(field > 0, field - 127, -127)
    grid = np.maximum(exponent, F16_MIN_EXPONENT) - F16_FRACTION_BITS  # the binary16 unit in the last place
    shift = np.clip(grid - unit, 1, 62)  # always >= 13 here; clipped so the shifts below are defined
    kept = significand >> shift
    remainder = significand & ((np.int64(1) << shift) - 1)
    half = np.int64(1) << (shift - 1)
    kept = kept + ((remainder > half) | ((remainder == half) & (kept & 1 == 1)))
    # A normal's leading one adds to the exponent field, so a carry out of the fraction raises the exponent by itself.
    out = np.where(exponent >= F16_MIN_EXPONENT, ((exponent - F16_MIN_EXPONENT) << F16_FRACTION_BITS) + kept, kept)
    out = np.where(significand == 0, 0, np.minimum(out, F16_INFINITY))
    return out.astype(np.uint16)


def round_half_away(p: np.ndarray) -> np.ndarray:
    """Round to the nearest integer, halves away from zero, exactly: `p − trunc(p)` is exact in binary32, so the
    half is tested on the true fraction (adding 0.5 and truncating would round a value just under a half up)."""
    whole = np.trunc(p)
    return whole + np.where(np.abs(p - whole) >= np.float32(0.5), np.copysign(np.float32(1), p), np.float32(0))


def derive_q8_0(values: np.ndarray) -> np.ndarray:
    """§2: the Q8_0 blocks of a row-major run of binary32 values whose length is a multiple of 32, as an
    `(n_blocks, 34)` byte array. Every step is one binary32 operation, in the order the rule gives."""
    blocks = np.ascontiguousarray(values, dtype=np.float32).reshape(-1, BLOCK)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        amax = np.max(np.abs(blocks), axis=1)
        scale = amax / np.float32(SCALE_DIVISOR)
        inverse = np.where(scale > 0, np.float32(1) / np.where(scale > 0, scale, np.float32(1)), np.float32(0))
        product = blocks * inverse[:, None]
        codes = np.clip(round_half_away(product), CODE_MIN, CODE_MAX)
    # 0 × inf (a zero in a block whose scale underflowed its reciprocal to infinity) is NaN, and the rule's cast of
    # a NaN to a code is 0.
    codes = np.where(np.isnan(codes), np.float32(0), codes).astype(np.int8)
    out = np.empty((blocks.shape[0], BLOCK_BYTES), dtype=np.uint8)
    stored = f32_to_f16_nearest_bits(scale)
    out[:, 0] = stored & 0xFF
    out[:, 1] = stored >> 8
    out[:, 2:] = codes.view(np.uint8)
    return out


# ── the same rule on exact rationals: a second route, for the tests and the worked vectors ────────────────────


def derive_block_exact(values: list[Fraction]) -> tuple[int, list[int]]:
    """§2 for one block, on exact rationals with every rounding done explicitly: `(stored scale bits, codes)`. Shares
    no arithmetic with [`derive_q8_0`]; the two must agree."""
    if len(values) != BLOCK:
        raise ValueError(f"a block is {BLOCK} values")
    f32 = lambda x: numconv.round_nearest_even(x, numconv.BINARY32)  # noqa: E731
    amax = max(abs(v) for v in values)
    scale = f32(amax / SCALE_DIVISOR)
    try:
        stored = numconv.round_nearest_even(scale, BINARY16)
        stored_bits = _f16_bits(stored)
    except numconv.Overflow:
        stored_bits = F16_INFINITY
    if scale <= 0:
        return stored_bits, [0] * BLOCK
    try:
        inverse = f32(1 / scale)
    except numconv.Overflow:
        # The reciprocal is infinite: a nonzero value saturates by its sign and a zero gives the NaN code, 0.
        return stored_bits, [0 if v == 0 else (CODE_MAX if v > 0 else CODE_MIN) for v in values]
    codes = []
    for v in values:
        try:
            p = f32(v * inverse)
        except numconv.Overflow:
            codes.append(CODE_MAX if v > 0 else CODE_MIN)
            continue
        whole = int(p)  # toward zero
        if abs(p - whole) >= Fraction(1, 2):
            whole += 1 if p > 0 else -1
        codes.append(max(CODE_MIN, min(CODE_MAX, whole)))
    return stored_bits, codes


def _f16_bits(x: Fraction) -> int:
    """The bit pattern of a non-negative binary16 value."""
    if x == 0:
        return 0
    e = x.numerator.bit_length() - x.denominator.bit_length()
    if Fraction(2) ** e > x:
        e -= 1
    e = max(e, F16_MIN_EXPONENT)
    kept = x / Fraction(2) ** (e - F16_FRACTION_BITS)
    assert kept.denominator == 1
    return ((e - F16_MIN_EXPONENT) << F16_FRACTION_BITS) + int(kept) if kept >= 1 << F16_FRACTION_BITS else int(kept)


# ── the judgement ────────────────────────────────────────────────────────────


class Judge:
    def __init__(self, container: str):
        self.root = container
        self.violations: list[dict] = []
        self.reasons: list[str] = []
        self.counts = {"representations": 0, "q8_tensors": 0, "blocks": 0, "elements": 0, "carried": 0,
                       "carried_bytes": 0, "zero_scale_blocks": 0, "unquantised_matrices": 0}
        self.derivations: list[dict] = []

    def violate(self, name: str, detail: str, count: int = 1) -> None:
        self.violations.append({"name": name, "violations": count, "detail": detail})

    def cannot(self, reason: str) -> None:
        self.reasons.append(reason)

    def run(self) -> None:
        V.check_platform()
        self.files = file_digests(self.root)
        store = Store(self.root)
        derived = sorted(rid for rid, e in store.reps.items() if "compiled_from" in e)
        if not derived:
            self.cannot("the container has no re-derived representation, so the contract would hold vacuously")
            return
        for rid in derived:
            self.judge_representation(store, rid)
        if not self.reasons and not self.counts["q8_tensors"]:
            self.cannot("no Q8_0 tensor was judged, so the contract would hold vacuously")

    def judge_representation(self, store: Store, rid: str) -> None:
        e = store.reps[rid]
        source = e.get("compiled_from")
        if e["encoding"] != C.Q8_0:
            self.cannot(f"{rid}: encoding {e['encoding']!r} is not one this contract derives")
            return
        if e.get("encoder") != ENCODER:
            self.cannot(f"{rid}: the pack names derivation rule {e.get('encoder')!r}; this contract implements "
                        f"{ENCODER!r} only")
            return
        if not isinstance(source, str) or source not in store.reps:
            self.violate(rid, f"compiled_from {source!r} is not a representation of this container")
            return
        s = store.reps[source]
        if "compiled_from" in s or s["encoding"] != C.BF16 or s["object"] != e["object"]:
            self.violate(rid, f"compiled_from {source!r} is not the canonical BF16 representation of {e['object']!r}")
            return
        # Both segments are parsed by the validating reader, which holds their bytes to the index's digests.
        cseg, dseg = store.seg(source), store.seg(rid)
        if e.get("source_representation_digest") != s["payload_sha256"]:
            self.violate(rid, "source_representation_digest is not the canonical representation's payload digest: "
                              "the pack does not name the weights this container holds")
        if [(t.name, t.shape) for t in cseg.tensors] != [(t.name, t.shape) for t in dseg.tensors]:
            self.violate(rid, "the pack's tensors are not the canonical representation's, name for name and shape "
                              "for shape, in order")
            return
        self.counts["representations"] += 1
        self.derivations.append({"representation": rid, "source": source,
                                 "source_payload_sha256": s["payload_sha256"], "rule": dict(ENCODER),
                                 "codec": dict(C.CODEC_Q8_0)})
        for ct, dt in zip(cseg.tensors, dseg.tensors):
            self.judge_tensor(store, source, rid, ct, dt)

    def judge_tensor(self, store: Store, source: str, rid: str, ct, dt) -> None:
        where = f"{rid}/{dt.name}"
        canonical = store.read(source, ct.name)
        packed = store.read(rid, dt.name)
        if dt.dtype != C.Q8_0:
            if dt.dtype != ct.dtype:
                self.cannot(f"{where}: carried as {dt.dtype}, from a canonical {ct.dtype}: not a form this contract "
                            "judges")
                return
            self.counts["carried"] += 1
            self.counts["carried_bytes"] += len(packed)
            self.counts["unquantised_matrices"] += len(ct.shape) == 2 and ct.shape[-1] % BLOCK == 0
            if packed != canonical:
                differing = sum(a != b for a, b in zip(packed, canonical)) + abs(len(packed) - len(canonical))
                self.violate(where, f"a tensor carried unquantised differs from the canonical tensor in {differing} "
                                    "bytes", 1)
            return
        if ct.dtype != C.BF16:
            self.cannot(f"{where}: quantised from a canonical {ct.dtype}; the rule is defined on BF16 weights")
            return
        values = bf16_to_f32(canonical)
        if not np.all(np.isfinite(values)):
            self.cannot(f"{where}: the canonical tensor holds a non-finite element, outside the rule's domain (§2)")
            return
        expected = derive_q8_0(values)
        actual = np.frombuffer(packed, dtype=np.uint8).reshape(-1, BLOCK_BYTES)
        self.counts["q8_tensors"] += 1
        self.counts["blocks"] += expected.shape[0]
        self.counts["elements"] += values.size
        self.counts["zero_scale_blocks"] += int(np.count_nonzero((expected[:, 0] | expected[:, 1]) == 0))
        bad = expected != actual
        bad_blocks = np.flatnonzero(bad.any(axis=1))
        if not bad_blocks.size:
            return
        scales = int(np.count_nonzero(bad[:, :2].any(axis=1)))
        codes = int(np.count_nonzero(bad[:, 2:]))
        b = int(bad_blocks[0])
        k = dt.shape[-1]
        row, col = divmod(b * BLOCK, k)
        self.violate(where, f"{bad_blocks.size} of {expected.shape[0]} blocks differ from the derived pack ({scales} "
                            f"stored scales, {codes} codes); first is block {b} (row {row}, columns {col}..="
                            f"{col + BLOCK - 1}): expected {expected[b].tobytes().hex()}, found "
                            f"{actual[b].tobytes().hex()}", int(bad_blocks.size))


def could_not(out: dict, *reasons: str) -> dict:
    out.update(status="could_not_evaluate", violations=0, offending=[], reasons=list(reasons))
    return out


def evaluate(out: dict, container: str) -> dict:
    """Fill `out` with the verdict. Never raises for a bad input, and never answers `satisfied` for a pack it did not
    derive in full."""
    out["derived_from"] = ("the container's own canonical BF16 tensor bytes, by the derivation rule the pack's index "
                           "entry names; the engine's packer is not run and its digests are checked, never used as "
                           "expected values")
    judge = Judge(container)
    try:
        judge.run()
    except RuntimeError as e:
        judge.cannot(f"the platform cannot run the rule: {e}")
    except BAD_INPUT as e:
        judge.cannot(f"an input can't be read: {type(e).__name__}: {e}")
    files = getattr(judge, "files", None)
    if files is not None:
        out["container"]["sha256"] = container_digest(files)
    c = judge.counts
    out["measurements"] = [
        {"name": "re-derived representations judged", "value": str(c["representations"])},
        {"name": "Q8_0 tensors derived and compared", "value": str(c["q8_tensors"])},
        {"name": "blocks compared (stored scale and 32 codes each)", "value": str(c["blocks"])},
        {"name": "elements quantised", "value": str(c["elements"])},
        {"name": "tensors carried unquantised, compared byte for byte", "value": str(c["carried"])},
        {"name": "bytes of carried tensors compared", "value": str(c["carried_bytes"])},
        {"name": "zero-scale blocks", "value": str(c["zero_scale_blocks"])},
        {"name": "matrices the pack leaves unquantised (informational: the selection is not judged)",
         "value": str(c["unquantised_matrices"])},
        {"name": "not established", "value": "; ".join(NOT_ESTABLISHED)},
    ]
    out["derivations"] = judge.derivations
    if judge.reasons:
        # An unjudged part is never folded into a verdict on the rest.
        return could_not(out, *judge.reasons)
    if not judge.violations:
        out.update(status="satisfied", violations=0, offending=[], reasons=[])
        return out
    offending = sorted(judge.violations, key=lambda o: -o["violations"])
    if len(offending) > OFFENDING_LIMIT:
        tail = offending[OFFENDING_LIMIT - 1:]
        offending = offending[:OFFENDING_LIMIT - 1] + [{
            "name": f"{len(tail)} further findings", "violations": sum(o["violations"] for o in tail),
            "detail": "folded: the protocol lists at most ten"}]
    out.update(status="violated", violations=sum(o["violations"] for o in judge.violations), offending=offending,
               reasons=[])
    return out
