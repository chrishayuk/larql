"""Synthetic checks for the registered CAR-1A carrier encodings."""
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwcar1a_codecs import (  # noqa: E402
    GRID, PCA_MAX_RANK, WIDTH, encode, model_bytes_for_rank,
)


class CodecTests(unittest.TestCase):
    def setUp(self):
        self.natural = np.linspace(-2, 2, WIDTH, dtype="<f4")
        self.donor = np.linspace(3, 4, WIDTH, dtype="<f4")

    def test_controls_select_only_the_carrier_source(self):
        exact_payload, exact, _ = encode(GRID[0], self.natural, self.donor)
        donor_payload, donor, _ = encode(GRID[1], self.natural, self.donor)
        self.assertEqual(len(exact_payload), WIDTH * 4)
        self.assertEqual(len(donor_payload), WIDTH * 4)
        np.testing.assert_array_equal(exact.view("<u4"), self.natural.view("<u4"))
        np.testing.assert_array_equal(donor.view("<u4"), self.donor.view("<u4"))

    def test_packed_quantization_sizes_and_codes(self):
        for candidate, expected in ((GRID[3], 2564), (GRID[4], 1284), (GRID[5], 644)):
            payload, decoded, shared = encode(candidate, self.natural, self.donor)
            self.assertEqual(len(payload), expected)
            self.assertEqual(shared, 0)
            self.assertTrue(np.isfinite(decoded).all())
            self.assertGreater(struct.unpack("<f", payload[:4])[0], 0)
        two_bit, decoded, _ = encode(GRID[5], self.natural, self.donor)
        self.assertEqual(two_bit[4] & 3, 3)  # first negative coordinate is -1
        self.assertEqual(decoded[0], -2)

    def test_sparse_tie_break_and_index_order(self):
        natural = np.zeros(WIDTH, dtype="<f4")
        natural[100] = 7
        natural[40] = -7
        natural[8] = 6
        payload, decoded, _ = encode(GRID[6], natural, self.donor)
        self.assertEqual(len(payload), 32 * 6)
        indices = [index for index, _ in struct.iter_unpack("<Hf", payload)]
        self.assertEqual(indices, sorted(indices))
        self.assertIn(40, indices)
        self.assertIn(100, indices)
        np.testing.assert_array_equal(decoded[[8, 40, 100]], natural[[8, 40, 100]])

    def test_pca_payload_and_shared_bytes(self):
        mean = np.zeros(WIDTH, dtype="<f4")
        basis = np.zeros((PCA_MAX_RANK, WIDTH), dtype="<f4")
        for i in range(PCA_MAX_RANK):
            basis[i, i] = 1
        payload, decoded, shared = encode(GRID[12], self.natural, self.donor, (mean, basis))
        self.assertEqual(len(payload), 8 * 4)
        self.assertEqual(shared, model_bytes_for_rank(8))
        np.testing.assert_array_equal(decoded[:8], self.natural[:8])
        np.testing.assert_array_equal(decoded[8:], np.zeros(WIDTH - 8))


if __name__ == "__main__":
    unittest.main()
