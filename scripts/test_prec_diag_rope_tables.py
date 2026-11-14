"""Unit tests for the ULP arithmetic in prec_diag_rope_tables.py."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prec_diag_rope_tables as t  # noqa: E402


def f32(*values):
    return np.array(values, dtype=np.float32)


def test_identical_values_are_zero_ulps_and_signed_zeros_agree():
    assert t.ulp_distance(f32(1.0, -2.5, 0.0), f32(1.0, -2.5, -0.0)).tolist() == [0, 0, 0]


def test_adjacent_floats_are_one_ulp_on_both_sides_of_zero():
    one = f32(1.0)
    up = np.nextafter(one, np.float32(2.0))
    assert t.ulp_distance(one, up).tolist() == [1]
    tiny = np.nextafter(np.float32(0.0), np.float32(1.0))
    assert t.ulp_distance(f32(0.0), np.array([tiny])).tolist() == [1]
    assert t.ulp_distance(np.array([-tiny]), np.array([tiny])).tolist() == [2]


def test_compare_counts_bit_identity_and_locates_the_worst_entry():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = a.copy()
    b[1, 0] = np.nextafter(np.float32(3.0), np.float32(4.0))
    report = t.compare("cos", a, b)
    assert report["entries"] == 4
    assert report["bit_identical"] == 3
    assert report["max_ulp"] == 1
    assert report["max_ulp_at"] == [1, 0]
    assert report["entries_over_1_ulp"] == 0
    assert report["max_abs_at"] == [1, 0]
    assert 0 < report["max_abs"] < 1e-6


def test_compare_refuses_a_shape_mismatch():
    try:
        t.compare("sin", f32(1.0, 2.0), f32(1.0))
    except ValueError:
        return
    raise AssertionError("a shape mismatch was compared")
