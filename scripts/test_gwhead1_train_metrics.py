import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parent))
from gwhead1_train_metrics import adjusted_candidate, pairwise_cosine, softmax


class GwHead1TrainMetricTests(unittest.TestCase):
    def test_control_adjustment_subtracts_matched_control_gain(self) -> None:
        before = softmax(np.asarray([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0], [3.0, 0.0], [0.0, 3.0], [1.0, 1.0]]))
        after = softmax(np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [5.0, 0.0], [0.0, 5.0], [1.0, 1.0]]))
        value = adjusted_candidate(
            before,
            after,
            np.asarray([[0, 1, 2]]),
            np.asarray([[3, 4, 5]]),
        )
        self.assertEqual(value.shape, (1,))
        self.assertGreater(value[0], 0.0)

    def test_pairwise_cosine_is_invariant_to_positive_scale(self) -> None:
        states = np.asarray([[1.0, 2.0], [2.0, 4.0], [9.0, 18.0], [1.0, 0.0]])
        value = pairwise_cosine(states, np.asarray([[0, 1, 2]]))
        np.testing.assert_allclose(value, [1.0])


if __name__ == "__main__":
    unittest.main()
