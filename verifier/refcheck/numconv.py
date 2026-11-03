"""Exact conversion of a JSON number's text to an IEEE-754 binary32 or binary64 value, by round-to-nearest-even.

PROTOCOL.md D3: "The authority's value is taken from its `config.json` text, converted to that type with correct
rounding (round-to-nearest-even), and compared for exact equality." The conversion here goes from the exact decimal
value straight to the target format, with one rounding. It never passes through binary64 on the way to binary32 (which
would round twice), and it is written from the IEEE-754 definition, not taken from numpy or struct.
"""

from __future__ import annotations

import struct
from math import isqrt, ldexp
from decimal import Decimal
from fractions import Fraction

# (precision p in bits, emin, emax) of the two formats
BINARY32 = (24, -126, 127)
BINARY64 = (53, -1022, 1023)


class Overflow(Exception):
    """The value rounds to infinity in the target format."""


def exact(value) -> Fraction:
    """The exact rational value of a JSON number held as int, Decimal or Fraction (never a float: a float is already
    a rounded value)."""
    if isinstance(value, bool):
        raise TypeError("a boolean is not a number")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite decimal")
        return Fraction(value)
    if isinstance(value, Fraction):
        return value
    raise TypeError(f"{type(value).__name__} is not an exact number")


def round_nearest_even(x: Fraction, fmt: tuple[int, int, int]) -> Fraction:
    """The binary floating-point value nearest to `x` in `fmt`, ties to the even significand. Subnormals included.
    Raises Overflow when the result would be infinite. The sign of a zero result is lost (a Fraction has none)."""
    p, emin, emax = fmt
    if x == 0:
        return Fraction(0)
    sign = -1 if x < 0 else 1
    a = abs(x)
    # e = floor(log2(a)), exactly
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    elif Fraction(2) ** (e + 1) <= a:
        e += 1
    e = max(e, emin)  # subnormal range: the quantum stops shrinking
    quantum = Fraction(2) ** (e - p + 1)
    n = a / quantum
    m = n.numerator // n.denominator
    rem = n - m
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and m % 2 == 1):
        m += 1
    v = m * quantum
    if v >= Fraction(2) ** (emax + 1):
        raise Overflow
    return sign * v


def to_binary64(value) -> float:
    """Python float holding the binary64 value nearest to the exact number (ties to even)."""
    r = round_nearest_even(exact(value), BINARY64)
    return float(r)  # exact: r is representable, so float() does not round


def to_binary32(value) -> float:
    """Python float holding the binary32 value nearest to the exact number (ties to even). The float is exactly that
    binary32 value (every binary32 is a binary64)."""
    r = round_nearest_even(exact(value), BINARY32)
    return float(r)


def widen32(value) -> float:
    """f64(f32(x)): round to binary32 then widen. SPEC section 17.3's encoding of the norm epsilon."""
    return to_binary32(value)


def inv_sqrt_binary64(h: int) -> float:
    """The exact mathematical 1/sqrt(h), rounded once to binary64 by round-to-nearest-even (D8, question 24).

    Pure integer arithmetic: no pow, no sqrt, no float division, no float literal, and nothing is passed through any
    other format. With k the least integer such that 4**k >= h, 1/sqrt(h) lies in [2**-k, 2**(1-k)], so its binary64
    significand is m = round(2**(52+k) / sqrt(h)) and the result is m * 2**(-k-52). Writing x = sqrt(N/h) with
    N = 4**(52+k), floor(x) is isqrt(N // h), and x compared with m + 1/2 is decided exactly by 4N against h(2m+1)**2.
    """
    if isinstance(h, bool) or not isinstance(h, int) or h < 1:
        raise ValueError("head_dim must be a positive integer")
    k = 0
    while 4 ** k < h:
        k += 1
    n = 4 ** (52 + k)
    m = isqrt(n // h)  # floor(x)
    lhs, rhs = 4 * n, h * (2 * m + 1) ** 2  # x >= m + 1/2  <=>  lhs >= rhs
    if lhs > rhs or (lhs == rhs and m % 2 == 1):
        m += 1
    # m <= 2**53, so float(m) is exact, and scaling by a power of two is exact (normal range)
    return ldexp(float(m), -k - 52)


def bits32(x: float) -> int:
    """Binary32 bit pattern of a float that is exactly a binary32 value."""
    b = struct.pack("<f", x)
    if struct.unpack("<f", b)[0] != x:
        raise ValueError(f"{x!r} is not a binary32 value")
    return struct.unpack("<I", b)[0]


def bits64(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]
