"""Seeded synthetic input generators for the GEMV tests. Synthetic data only."""

import struct

import numpy as np

# binary16 scale patterns
F16 = {
    "zero": 0x0000,
    "negzero": 0x8000,
    "sub_min": 0x0001,
    "sub_max": 0x03FF,
    "min_normal": 0x0400,
    "one": 0x3C00,
    "max_normal": 0x7BFF,  # also the largest finite binary16 (65504)
    "neg_max": 0xFBFF,
    "neg_sub": 0x8001,
    "inf": 0x7C00,
    "nan": 0x7E00,
}

SCALE_KINDS = ["typical", "zero", "negzero", "subnormal", "min_normal", "max_normal", "random_finite"]
X_KINDS = ["normal", "subnormal", "wide", "tiny", "large"]
Q_KINDS = ["random", "extremes", "sparse"]


def f32bits(f):
    return struct.unpack("<I", struct.pack("<f", float(f)))[0]


def scales(rng, count, kind):
    if kind == "typical":
        v = (rng.standard_normal(count) * 0.05).astype(np.float16)
        return [int(b) for b in v.view(np.uint16)]
    if kind == "zero":
        return [0x0000 if rng.random() < 0.5 else 0x8000 for _ in range(count)]
    if kind == "negzero":
        return [0x8000] * count
    if kind == "subnormal":
        return [int(rng.integers(1, 1024)) | (int(rng.integers(0, 2)) << 15) for _ in range(count)]
    if kind == "min_normal":
        return [0x0400 | (int(rng.integers(0, 2)) << 15)] * count
    if kind == "max_normal":
        return [0x7BFF | (int(rng.integers(0, 2)) << 15)] * count
    if kind == "random_finite":
        out = []
        while len(out) < count:
            h = int(rng.integers(0, 1 << 16))
            if (h >> 10) & 31 != 31:
                out.append(h)
        return out
    raise KeyError(kind)


def codes(rng, count, kind):
    if kind == "random":
        q = rng.integers(-128, 128, count)
    elif kind == "extremes":
        q = rng.choice([-128, 127, -127, 126], count)
    elif kind == "sparse":
        q = np.where(rng.random(count) < 0.8, 0, rng.integers(-128, 128, count))
    else:
        raise KeyError(kind)
    return [int(v) for v in q]


def xs(rng, k, kind):
    sign = rng.integers(0, 2, k).astype(np.uint32) << 31
    if kind == "normal":
        return [f32bits(v) for v in rng.standard_normal(k).astype(np.float32)]
    if kind == "subnormal":
        return [int(s) | int(m) for s, m in zip(sign, rng.integers(0, 1 << 23, k))]
    if kind == "wide":
        e = rng.integers(127 - 60, 127 + 60, k)
    elif kind == "tiny":
        e = rng.integers(0, 30, k)
    elif kind == "large":
        e = rng.integers(127 + 80, 127 + 95, k)
    else:
        raise KeyError(kind)
    man = rng.integers(0, 1 << 23, k)
    return [int(s) | (int(a) << 23) | int(m) for s, a, m in zip(sign, e, man)]


def make_case(rng, n, k, scale_kind, x_kind, q_kind):
    """-> (blocks bytes, x_bits list, n, k)"""
    nb = k // 32
    d = scales(rng, n * nb, scale_kind)
    raw = bytearray()
    for i in range(n * nb):
        q = codes(rng, 32, q_kind)
        raw += struct.pack("<H", d[i]) + bytes(c & 0xFF for c in q)
    return bytes(raw), xs(rng, k, x_kind), n, k
