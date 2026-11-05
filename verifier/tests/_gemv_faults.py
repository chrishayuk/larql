"""Deliberately faulty implementations of the Q8_0 GEMV, for tests only. Never imported by refcheck/.

Each takes section 3's interface, `(blocks, x_bits, n, k)`, and returns n binary32 bit patterns (a uint32 array).
Each computes a wrong result on purpose. They decode with numpy's own float16, not with refcheck.q8_0, so they share
no decode code with the reference.
"""

import numpy as np


def _decode(raw, n, k):
    """-> (d float32 (n, nb), q float32 (n, nb, 32), d_bits uint16 (n, nb))"""
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(n, k // 32, 34)
    dbits = np.ascontiguousarray(arr[:, :, :2]).view("<u2")[:, :, 0].astype(np.uint16)
    d = dbits.view(np.float16).astype(np.float32)
    q = np.ascontiguousarray(arr[:, :, 2:]).view(np.int8).astype(np.float32)
    return d, q, dbits


def _xs(x_bits):
    return np.asarray(x_bits, dtype=np.uint32).view(np.float32)


def _sequential(prods):
    """prods: (n, m) float32 -> (n,) float32, ascending order, one binary32 add each."""
    acc = np.zeros(prods.shape[0], dtype=np.float32)
    with np.errstate(all="ignore"):
        for j in range(prods.shape[1]):
            acc = (acc + prods[:, j]).astype(np.float32)
    return acc


def _bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).copy()


def _w_times_x(raw, x_bits, n, k, d=None):
    dd, q, _ = _decode(raw, n, k)
    d = dd if d is None else d
    w = (q * d[:, :, None]).reshape(n, k).astype(np.float32)
    with np.errstate(all="ignore"):
        return (w * _xs(x_bits)[None, :]).astype(np.float32)


# ---- faults that must fail both contracts -------------------------------------------------------------------------


def wrong_stride(raw, x_bits, n, k):
    """Row i is read from block i*(nb-1) instead of i*nb (a stride one block short; rows share data when nb = 1)."""
    nb = k // 32
    stride = max(nb - 1, 0)
    rows = []
    for i in range(n):
        s = i * stride * 34
        rows.append(raw[s : s + nb * 34])
    return _bits(_sequential(_w_times_x(b"".join(rows), x_bits, n, k)))


def drop_last_block(raw, x_bits, n, k):
    """The last 32-column block of every row is left out of the sum."""
    return _bits(_sequential(_w_times_x(raw, x_bits, n, k)[:, : k - 32]))


def flush_subnormal_scales(raw, x_bits, n, k):
    """A binary16 subnormal scale is treated as zero."""
    d, _, dbits = _decode(raw, n, k)
    d = np.where(((dbits >> 10) & 31) == 0, np.float32(0), d).astype(np.float32)
    return _bits(_sequential(_w_times_x(raw, x_bits, n, k, d=d)))


def flip_one_code(raw, x_bits, n, k):
    """One int8 code (row 0, column 0) has bit 4 flipped (q -> q xor 16)."""
    b = bytearray(raw)
    b[2] ^= 0x10
    return _bits(_sequential(_w_times_x(bytes(b), x_bits, n, k)))


def flip_sign_of_one_row(raw, x_bits, n, k):
    """Row 0's result has its sign bit flipped."""
    y = _bits(_sequential(_w_times_x(raw, x_bits, n, k)))
    y[0] ^= np.uint32(0x80000000)
    return y


def flip_lowest_bit_of_one_output(raw, x_bits, n, k):
    """Row 0's result has its lowest bit flipped (one unit in the last place, or one subnormal step)."""
    y = _bits(_sequential(_w_times_x(raw, x_bits, n, k)))
    y[0] ^= np.uint32(1)
    return y


# ---- faults that are different programs, which B1 is expected (not guaranteed) to accept ----------------------------


def reverse_order(raw, x_bits, n, k):
    """The same terms, summed in descending j."""
    return _bits(_sequential(_w_times_x(raw, x_bits, n, k)[:, ::-1]))


def scale_factored_out(raw, x_bits, n, k):
    """d * sum_j (q_j * x_j) per block (inner sum in binary32), then the block results summed in order."""
    d, q, _ = _decode(raw, n, k)
    x = _xs(x_bits).reshape(1, k // 32, 32)
    with np.errstate(all="ignore"):
        qx = (q * x).astype(np.float32)  # (n, nb, 32)
        acc = np.zeros(n, dtype=np.float32)
        for b in range(k // 32):
            blocksum = _sequential(qx[:, b, :])
            acc = (acc + (blocksum * d[:, b]).astype(np.float32)).astype(np.float32)
    return _bits(acc)


def _pairwise(v):
    m = v.shape[1]
    if m == 1:
        return v[:, 0]
    h = m // 2
    with np.errstate(all="ignore"):
        return (_pairwise(v[:, :h]) + _pairwise(v[:, h:])).astype(np.float32)


def pairwise_sum(raw, x_bits, n, k):
    """The products w*x (binary32), summed by recursive halving."""
    return _bits(_pairwise(_w_times_x(raw, x_bits, n, k)))


def binary64_accumulate(raw, x_bits, n, k):
    """The products exactly, accumulated in binary64 in order j, rounded to binary32 once at the end."""
    dd, q, _ = _decode(raw, n, k)
    w = (q.astype(np.float64) * dd.astype(np.float64)[:, :, None]).reshape(n, k)
    x = _xs(x_bits).astype(np.float64)
    acc = np.zeros(n, dtype=np.float64)
    with np.errstate(all="ignore"):
        for j in range(k):
            acc = acc + w[:, j] * x[j]
        return _bits(acc.astype(np.float32))


MUST_FAIL_B1 = {
    "wrong_stride": wrong_stride,
    "drop_last_block": drop_last_block,
    "flush_subnormal_scales": flush_subnormal_scales,
    "flip_one_code": flip_one_code,
    "flip_sign_of_one_row": flip_sign_of_one_row,
}
RECORD_B1 = {
    "flip_lowest_bit_of_one_output": flip_lowest_bit_of_one_output,
}
DIFFERENT_PROGRAMS = {
    "reverse_order": reverse_order,
    "scale_factored_out": scale_factored_out,
    "pairwise_sum": pairwise_sum,
    "binary64_accumulate": binary64_accumulate,
}
