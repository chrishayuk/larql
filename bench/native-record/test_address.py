"""Instrument checks; no model or Metal required."""
import unittest

import numpy as np

from run_address import DECOYS, fixture, score, gates


class Tokenizer:
    def decode(self, ids):
        return str(ids[0])


class InstrumentTests(unittest.TestCase):
    def test_full_vocab_competitor_beats_target(self):
        ll = np.array([2., 1., 9., -3.])
        row = score(ll, 0, ll, Tokenizer())
        self.assertFalse(row['hit'])
        self.assertEqual(row['rank'], 2)
        self.assertEqual(row['margin_full'], -7.)
        self.assertEqual(row['kl_clean_arm'], 0.)
        self.assertEqual(row['total_variation'], 0.)

    def test_shift_invariant_distribution(self):
        ll = np.array([2., 1., 9., -3.])
        row = score(ll + 10, 2, ll, Tokenizer())
        self.assertTrue(row['hit'])
        self.assertAlmostEqual(row['kl_clean_arm'], 0.)
        self.assertAlmostEqual(row['total_variation'], 0.)
        self.assertEqual(row['max_logit_delta'], 10.)

    def test_fixture_is_held_out(self):
        probes = fixture()
        self.assertEqual(len(probes), 114)
        self.assertTrue(set(DECOYS).isdisjoint(p['prompt'] for p in probes))
        self.assertEqual(sum(p['group'] == 'sentence' for p in probes), 12)
        self.assertEqual(sum(p['group'] == 'alias' for p in probes), 6)

    def test_gate_rejects_collateral_hidden_by_pooling(self):
        arms = {}
        for arm in ['clean_repeat', 'written', 'replace_0', 'replace_1', 'replace_2', 'removed', 'restored']:
            arms[arm] = [dict(p, hit=True, clean_agree=True, kl_clean_arm=0.,
                              total_variation=0., max_logit_delta=0., written_max_logit_delta=0.)
                         for p in fixture()]
        self.assertTrue(gates(arms)['advance'])
        victim = next(p for p in arms['written'] if p['group'] == 'different_relation')
        victim['clean_agree'] = False
        result = gates(arms)
        self.assertFalse(result['advance'])
        self.assertIn('written/different_relation', result['failed'])


if __name__ == '__main__':
    unittest.main()
