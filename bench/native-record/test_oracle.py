import unittest

import numpy as np

from oracle_runtime import vector_metrics


class OracleMetricsTests(unittest.TestCase):
    def test_post_norm_saturation_is_not_pre_norm_strength(self):
        # A normalization can erase a pure amplitude change completely.
        result = vector_metrics([4., 0.], [1., 0.], [2., 0.], [1., 0.], 2.)
        self.assertEqual(result['pre_norm_delta'], 2.)
        self.assertEqual(result['post_norm_delta'], 0.)
        self.assertEqual(result['normalization_delta_gain'], 0.)
        self.assertIsNone(result['delta_direction_cosine'])

    def test_cancellation_can_make_contribution_ratio_exceed_one(self):
        result = vector_metrics([1., 0.], [1., 0.], [-3., 0.], [-1., 0.], 4.)
        self.assertEqual(result['contribution_over_ffn'], 4.)
        self.assertEqual(result['pre_norm_delta'], 4.)
        self.assertEqual(result['post_norm_delta'], 2.)
        self.assertEqual(result['delta_direction_cosine'], 1.)


if __name__ == '__main__':
    unittest.main()
