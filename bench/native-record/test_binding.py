import unittest

import numpy as np

from binding_fixture import reader_fixture, factorial_fixture, ROSTERS
from binding_readers import RidgeMatch, choose_threshold, reader_metrics


class BindingTests(unittest.TestCase):
    def test_factorial_is_complete_and_values_differ(self):
        facts, rows = factorial_fixture()
        self.assertEqual(len(facts), 12)
        for relation in ['capital', 'currency', 'language']:
            self.assertEqual(len({v for _, r, v in facts if r == relation}), 4)
        self.assertEqual(sum(r['group'] == 'sentence' for r in rows), 48)
        self.assertEqual(sum(r['group'] == 'alias' for r in rows), 24)

    def test_splits_are_separate(self):
        self.assertTrue(set(ROSTERS['development']).isdisjoint(ROSTERS['evaluation']))
        dev, test = reader_fixture('development'), reader_fixture('evaluation')
        self.assertTrue({r['prompt'] for r in dev}.isdisjoint(r['prompt'] for r in test))

    def test_pairwise_rule_handles_new_labels(self):
        keys = np.eye(3)
        probe = RidgeMatch(keys, keys, [0, 1, 2], 1.)
        fresh = np.array([[1., 1., 0.], [0., 1., 1.], [1., 0., 1.]])
        scores = probe.scores(fresh, fresh)
        # New enrollment labels, never a multiclass train/test label lookup.
        np.testing.assert_array_equal(scores.argmax(1), [0, 1, 2])

    def test_reject_all_cannot_pass_reader_gate(self):
        rows = [dict(label=i, relation=r) for i, r in enumerate(['capital', 'currency', 'language'])]
        rows.append(dict(label=None, relation='capital'))
        s, _ = reader_metrics(np.r_[np.eye(3), np.zeros((1, 3))], rows, 2.)
        self.assertEqual(s['accuracy'], 1.)
        self.assertEqual(s['false_acceptance'], 0.)
        self.assertFalse(s['gate'])
        self.assertGreater(choose_threshold([1., 2.], [-1., 0.]), 0.)


if __name__ == '__main__':
    unittest.main()
