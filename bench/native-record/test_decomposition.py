import unittest

import numpy as np

from decomposition_runtime import expected_sequence
from decomposition_analysis import analyze


class DecompositionTests(unittest.TestCase):
    def test_independent_scopes(self):
        raw = np.arange(1, 37).reshape(1, 3, 12).astype(float)
        early = expected_sequence(raw, '100', 3, 100.)
        np.testing.assert_array_equal(early[:, -1], raw[:, -1])
        self.assertEqual(np.count_nonzero(early[:, :-1]), 0)
        competitors = expected_sequence(raw, '010', 3, 100.)
        np.testing.assert_array_equal(competitors[:, :-1], raw[:, :-1])
        self.assertEqual(competitors[0, -1, 3], raw[0, -1, 3])
        self.assertEqual(np.count_nonzero(competitors[0, -1]), 1)
        amplitude = expected_sequence(raw, '001', 3, 100.)
        delta = amplitude - raw
        self.assertEqual(np.count_nonzero(delta), 1)
        self.assertEqual(amplitude[0, -1, 3], 100.)
        both = expected_sequence(raw, '111', 3, 100.)
        self.assertEqual(np.count_nonzero(both), 1)
        self.assertEqual(both[0, -1, 3], 100.)

    def test_joint_requirement_is_not_mislabeled_as_main_effect(self):
        rows = []
        for code in [f'{i:03b}' for i in range(8)]:
            for group in ['installation', 'sentence', 'alias']:
                hit = code[1:] == '11'
                rows.append(dict(id=group, group=group, phase='factorial', code=code, hit=hit,
                                 margin_full=1. if hit else -1.))
        result = analyze(rows)
        self.assertEqual(result['contrasts']['all']['001']['hit']['mean'], 0.)
        self.assertEqual(result['contrasts']['all']['010']['hit']['mean'], 0.)
        self.assertEqual(result['contrasts']['all']['011']['hit']['mean'], 1.)
        self.assertEqual(result['contrasts']['all']['111']['hit']['mean'], 0.)
        self.assertEqual(result['minimal_pattern_counts'], {'011': 3})


if __name__ == '__main__':
    unittest.main()
