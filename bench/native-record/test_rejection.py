import unittest
import numpy as np
from rejection_analysis import counts, curve, choose, native_gate


class RejectionTests(unittest.TestCase):
    def test_reject_all_is_not_coverage(self):
        ratio = np.array([[1., 0.], [2., 0.]])
        rows = [dict(fact=0, group='sentence'), dict(fact=None, group='unknown')]
        selected = choose(curve(ratio, rows))
        self.assertTrue(selected['negative_gate'])
        self.assertEqual(selected['positive_correct'], 0)
        self.assertEqual(selected['positive_accepted'], 0)

    def test_acceptance_is_not_correct_record_coverage(self):
        ratio = np.array([[1., 2.], [.1, 0.]])
        rows = [dict(fact=0, group='sentence'), dict(fact=None, group='unknown')]
        metric = counts(ratio, rows, .5)
        self.assertEqual(metric['positive_accepted'], 1)
        self.assertEqual(metric['positive_correct'], 0)

    def test_negative_gate_cannot_hide_a_category(self):
        ratio = np.array([[1., 0.]] + [[0., 0.]] * 20 + [[2., 0.]])
        rows = [dict(fact=0, group='sentence')] + [dict(fact=None, group='easy')] * 20 + [dict(fact=None, group='hard')]
        self.assertFalse(counts(ratio, rows, .5)['negative_gate'])

    def test_threshold_has_a_useful_separating_case(self):
        ratio = np.array([[1., 0.], [.8, 0.], [.2, 0.]])
        rows = [dict(fact=0, group='sentence')] * 2 + [dict(fact=None, group='unknown')]
        selected = choose(curve(ratio, rows))
        self.assertEqual(selected['positive_correct'], 2)
        self.assertEqual(selected['negative']['unknown']['false_accept'], 0)

    def test_training_progress_does_not_satisfy_native_validation_gate(self):
        baseline = dict(validation=dict(positive_delivery=0, negative_quiet=4))
        candidate = dict(validation=dict(positive_delivery=2, negative_quiet=4,
            groups={g: dict(n=12, selected=12, delivered=1) for g in ['sentence', 'alias']},
            negatives=dict(unknown={})))
        candidate['validation']['groups']['unknown'] = dict(n=4, selected=4, delivered=4)
        self.assertFalse(native_gate(candidate, baseline)['passed'])


if __name__ == '__main__':
    unittest.main()
