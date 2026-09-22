import unittest

import numpy as np

from scripts.gw4b_prewrite_retrieval import normalize


class Gw4bPrewriteRetrievalTests(unittest.TestCase):
    def test_normalize_rows(self) -> None:
        rows = normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
        np.testing.assert_allclose(np.linalg.norm(rows, axis=1), [1.0, 1.0])

    def test_normalize_rejects_zero(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero"):
            normalize(np.array([[0.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
