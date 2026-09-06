import unittest
from collections import Counter
import numpy as np

from controller_fixture import fixtures
from controller_fit import delivery, pattern_loss, targets


class ControllerTests(unittest.TestCase):
    def test_fresh_and_development_are_separate(self):
        banks = fixtures()
        for bank in banks.values():
            self.assertEqual(Counter(r['split'] for r in bank['development']), {'train': 116, 'validation': 52})
            dev = {r['prompt'] for r in bank['development']}
            fresh = {r['prompt'] for r in bank['evaluation'] if r['group'] != 'installation'}
            self.assertFalse(dev & fresh)
            self.assertEqual(sum(r['fact'] is None for r in bank['evaluation']), 84)
        self.assertTrue({f[0] for f in banks['observed_roster']['facts']}.isdisjoint(
            f[0] for f in banks['fresh_roster']['facts']))

    def test_targets_are_activation_patterns_not_just_class_labels(self):
        rows = [{'fact': 4}, {'fact': None}]
        y = targets(rows, np.ones(12))
        self.assertEqual(y[0].sum(), 1)
        self.assertEqual(y[0, 4], 1)
        self.assertEqual(y[1].sum(), 0)
        self.assertEqual(pattern_loss(y, y), 0)
        self.assertEqual(pattern_loss(np.zeros_like(y), y), .5)

    def test_selection_is_not_delivery(self):
        canonical = np.ones(12) * 200
        a = np.zeros(12)
        a[3] = 100
        result = delivery(a, canonical, 3)
        self.assertTrue(result['selection_correct'])
        self.assertFalse(result['delivery_ok'])
        a[3] = 200
        self.assertTrue(delivery(a, canonical, 3)['delivery_ok'])
        a[2] = 30
        self.assertFalse(delivery(a, canonical, 3)['delivery_ok'])

    def test_negative_signed_activity_is_not_quiet(self):
        a = np.zeros(12)
        a[2] = -1
        result = delivery(a, np.ones(12), None)
        self.assertTrue(result['selection_correct'])
        self.assertFalse(result['delivery_ok'])


if __name__ == '__main__':
    unittest.main()
