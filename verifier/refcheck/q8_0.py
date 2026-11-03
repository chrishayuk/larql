"""Q8_0 decoder, written from ggml `block_q8_0` and `dequantize_row_q8_0`.

Source (ggml commit facdc44a, see SHA256SUMS in the inputs directory):

    src/ggml-common.h:251-256   typedef struct { ggml_half d; int8_t qs[32]; } block_q8_0;
    src/ggml-quants.c:553-567   y[i*qk + j] = x[i].qs[j] * d,  d = GGML_FP16_TO_FP32(x[i].d)

A block is 34 bytes: a little-endian IEEE binary16 `d` (bytes 0-1) then 32 signed
int8 codes (bytes 2-33). Decode is f32(q) * f32(d). Both factors convert exactly
and the product fits in f32 (protocol rule T1), so the value is unique.

The binary16 -> binary32 conversion is done here with integer bit manipulation
from IEEE 754-2019, not through numpy's float16 type, so that the decoder does
not lean on the library it is cross-checked against (gguf-py).
"""

from __future__ import annotations

import numpy as np

QK8_0 = 32
BLOCK_BYTES = 34


def _build_subnormal_table() -> tuple[np.ndarray, np.ndarray]:
    """For each 10-bit subnormal mantissa m (1..1023): the f32 biased exponent and
    23-bit mantissa of m * 2**-24. Entry 0 is unused (zero is handled separately)."""
    exp = np.zeros(1024, dtype=np.uint32)
    man = np.zeros(1024, dtype=np.uint32)
    for m in range(1, 1024):
        p = m.bit_length() - 1  # position of the leading one, 0..9
        exp[m] = 127 + p - 24
        man[m] = (m << (23 - p)) & 0x7FFFFF
    return exp, man


_SUB_EXP, _SUB_MAN = _build_subnormal_table()


def f16_bits_to_f32_bits(h: np.ndarray) -> np.ndarray:
    """Exact IEEE 754 binary16 -> binary32 conversion on raw bit patterns.

    Zeros keep their sign, subnormals are normalised, infinities map to
    infinities. A NaN keeps its sign and payload, with the quiet bit set (a
    signalling NaN is quieted, as IEEE 754-2019 6.2 requires of a conversion).
    """
    h = np.asarray(h).astype(np.uint32)
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    man = h & 0x3FF

    out_exp = np.where(exp == 0, _SUB_EXP[man], exp + (127 - 15))
    out_man = np.where(exp == 0, _SUB_MAN[man], man << 13)

    is_zero = (exp == 0) & (man == 0)
    out_exp = np.where(is_zero, 0, out_exp)
    out_man = np.where(is_zero, 0, out_man)

    special = exp == 31
    is_nan = special & (man != 0)
    out_exp = np.where(special, 255, out_exp)
    out_man = np.where(special, np.where(is_nan, (man << 13) | 0x400000, 0), out_man)

    return ((sign << 31) | (out_exp.astype(np.uint32) << 23) | out_man.astype(np.uint32)).astype(np.uint32)


def _as_blocks(buf) -> np.ndarray:
    raw = np.frombuffer(buf, dtype=np.uint8) if not isinstance(buf, np.ndarray) else buf
    if raw.dtype != np.uint8:
        raise TypeError(f"Q8_0 data must be uint8 bytes, got {raw.dtype}")
    raw = raw.reshape(-1)
    if raw.size % BLOCK_BYTES:
        raise ValueError(f"Q8_0 byte length {raw.size} is not a multiple of {BLOCK_BYTES}")
    return raw.reshape(-1, BLOCK_BYTES)


def block_scale_bits(buf) -> np.ndarray:
    """The stored f16 scale of every block as a u16 bit pattern (little-endian read)."""
    blocks = _as_blocks(buf)
    return blocks[:, 0].astype(np.uint16) | (blocks[:, 1].astype(np.uint16) << 8)


def block_codes(buf) -> np.ndarray:
    """The int8 codes, shape (n_blocks, 32)."""
    return np.ascontiguousarray(_as_blocks(buf)[:, 2:]).view(np.int8)


def dequantize_q8_0(buf) -> np.ndarray:
    """Decode a run of Q8_0 blocks to a flat float32 array (block order preserved)."""
    blocks = _as_blocks(buf)
    d = f16_bits_to_f32_bits(block_scale_bits(blocks)).view(np.float32)
    q = block_codes(blocks).astype(np.float32)  # int8 -> f32 is exact
    with np.errstate(invalid="ignore"):  # 0 * inf is NaN by IEEE; not an error here
        return (q * d[:, None]).reshape(-1)


def dequantize_q8_0_rows(buf, rows: int, k: int) -> np.ndarray:
    """Decode a row-major tensor of `rows` rows of `k` elements each (k % 32 == 0).

    Rows are contiguous with no padding: row r, block b is block index r*(k/32)+b.
    """
    if k % QK8_0:
        raise ValueError(f"row length {k} is not a multiple of {QK8_0}")
    out = dequantize_q8_0(buf)
    if out.size != rows * k:
        raise ValueError(f"expected {rows * k} elements, data holds {out.size}")
    return out.reshape(rows, k)


def f32_bits(a: np.ndarray) -> np.ndarray:
    """Float32 array -> its u32 bit patterns (so +0 and -0 differ)."""
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def count_zero_scales(buf) -> int:
    """Number of blocks whose stored scale is +0 or -0 (A1 report item)."""
    return int(np.count_nonzero((block_scale_bits(buf) & 0x7FFF) == 0))
