#!/usr/bin/env python3
"""GPU-free audit of cached-only gate correction, rejection curves and gate."""
import json
import sys
from pathlib import Path
import numpy as np
from gate_aux_fit import STRENGTHS
from rejection_analysis import summarize, native_gate
from run_address import HERE, digest


def audit(root):
    metadata = json.loads((root / 'metadata.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    fixture = json.loads((root / 'fixture.json').read_text())
    prior = HERE / 'results/controller-v2'
    for name, expected in metadata['source_hashes'].items():
        assert digest(HERE / name) == expected, name
    assert digest(root / 'fixture.json') == metadata['fixture_sha256']
    for name, key in [('captures.npz', 'prior_cache_sha256'), ('fixture.json', 'prior_fixture_sha256'),
                      ('frozen_choice.json', 'prior_choice_sha256')]:
        assert digest(prior / name) == metadata[key]
    prior_fixture = json.loads((prior / 'fixture.json').read_text())
    assert fixture == {b: dict(facts=d['facts'], development=d['development']) for b, d in prior_fixture.items()}
    assert metadata['cache_keys_accessed'] == [bank + suffix for bank in fixture for suffix in [
        '_development_x', '_initial_gate', '_initial_up', '_fitted_gate', '_fitted_up', '_canonical', '_down']]
    assert summary['status'] == 'complete'
    assert metadata['full_model_forwards'] == summary['full_model_forwards'] == 0
    assert summary['optimization_steps'] == 9600
    assert summary['zero_aux_replay_exact'] and summary['auxiliary_gradient_check']
    for name, key in [('arrays.npz', 'arrays_sha256'), ('analysis.json', 'analysis_sha256'),
                      ('fit_trace.json', 'fit_trace_sha256'), ('frozen_choice.json', 'frozen_choice_sha256')]:
        assert digest(root / name) == summary[key]
    trace = json.loads((root / 'fit_trace.json').read_text())
    choice = json.loads((root / 'frozen_choice.json').read_text())
    assert choice == summary['selected'] and choice['selected_before_second_roster_fit']
    assert choice['step'] == 1600 and choice['config'] == dict(lr=.001, regularization=.0001)
    expected_order = [(b, strength, step) for b, strengths in [
        ('observed_roster', [0., *STRENGTHS]), ('fresh_roster', [0., choice['strength']])]
        for strength in strengths for step in [400, 800, 1200, 1600]]
    assert [(r['bank'], r['strength'], r['step']) for r in trace] == expected_order
    old_trace = json.loads((prior / 'fit_trace.json').read_text())
    for bank in fixture:
        old_candidate = 2 if bank == 'observed_roster' else 0
        old = [r for r in old_trace if r['bank'] == bank and r['candidate'] == old_candidate]
        replay = [r for r in trace if r['bank'] == bank and r['strength'] == 0]
        for left, right in zip(old, replay):
            for key in ['objective', 'train_loss', 'validation_loss', 'train_delivery', 'validation_delivery']:
                assert left[key] == right[key]
    analysis = json.loads((root / 'analysis.json').read_text())
    recomputed = {}
    with np.load(root / 'arrays.npz') as arrays, np.load(prior / 'captures.npz') as cache:
        for bank, data in fixture.items():
            recomputed[bank] = {}
            canonical = arrays[bank + '_canonical']
            for suffix in ['canonical', 'development_x', 'down']:
                np.testing.assert_array_equal(arrays[bank + '_' + suffix], cache[bank + '_' + suffix])
            methods = ['baseline'] + (['aux_' + str(s) for s in STRENGTHS] if bank == 'observed_roster' else [choice['name']])
            assert list(analysis[bank]) == methods
            for method in methods:
                prefix = bank + '_' + method + '_'
                for kind in ['gate', 'up', 'activation']:
                    assert arrays[prefix + kind].shape == (168, 12)
                    assert np.isfinite(arrays[prefix + kind]).all()
                for kind in ['gate', 'up']:
                    assert arrays[prefix + 'fitted_' + kind].shape == (12, 2560)
                    if method == 'baseline':
                        np.testing.assert_array_equal(arrays[prefix + 'fitted_' + kind], cache[bank + '_fitted_' + kind])
                result = summarize(arrays[prefix + 'activation'] / canonical, arrays[prefix + 'gate'], data['development'])
                if method != 'baseline':
                    result['native_gate'] = native_gate(result, recomputed[bank]['baseline'])
                recomputed[bank][method] = result
    assert recomputed == analysis
    best = min(STRENGTHS, key=lambda s: (
        not analysis['observed_roster']['aux_' + str(s)]['native_gate']['passed'],
        analysis['observed_roster']['aux_' + str(s)]['validation']['loss'], STRENGTHS.index(s)))
    assert choice['strength'] == best and choice['name'] == 'aux_' + str(best)
    gates = {b: analysis[b][choice['name']]['native_gate'] for b in fixture}
    assert gates == summary['development_gates']
    assert summary['full_model_evaluation_permitted'] == all(g['passed'] for g in gates.values())
    print('PASS: cached-development-only keys, exact prior replay traces, 9,600 optimization steps,')
    print('      fixed-strength repeat, distributions, threshold curves, native gates; zero model forwards.')
    print('Full-model evaluation permitted:', summary['full_model_evaluation_permitted'])


if __name__ == '__main__':
    audit(Path(sys.argv[1]))
