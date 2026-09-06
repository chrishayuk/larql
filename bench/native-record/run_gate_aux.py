#!/usr/bin/env python3
"""Cached-input-only fitting: no model loader or full-model forward exists here."""
import argparse
import json
from pathlib import Path
import numpy as np

from gate_aux_fit import STRENGTHS, auxiliary, fit_aux
from rejection_analysis import summarize, native_gate
from run_address import HERE, digest, persist


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    prior = HERE / 'results/controller-v2'
    full_fixture = json.loads((prior / 'fixture.json').read_text())
    fixture = {bank: dict(facts=d['facts'], development=d['development']) for bank, d in full_fixture.items()}
    del full_fixture
    persist(args.out / 'fixture.json', fixture)
    config = json.loads((prior / 'frozen_choice.json').read_text())['choice']
    assert config['config'] == dict(lr=.001, regularization=.0001) and config['step'] == 1600
    sources = ['run_gate_aux.py', 'gate_aux_fit.py', 'rejection_analysis.py', 'controller_fit.py',
               'run_address.py', 'GATE_AUX_PROTOCOL.md']
    accessed = [bank + suffix for bank in fixture for suffix in [
        '_development_x', '_initial_gate', '_initial_up', '_fitted_gate', '_fitted_up', '_canonical', '_down']]
    metadata = dict(source_hashes={n: digest(HERE / n) for n in sources},
        fixture_sha256=digest(args.out / 'fixture.json'), prior_cache_sha256=digest(prior / 'captures.npz'),
        prior_fixture_sha256=digest(prior / 'fixture.json'), prior_choice_sha256=digest(prior / 'frozen_choice.json'),
        cache_keys_accessed=accessed, full_model_forwards=0, prior_config=config)
    persist(args.out / 'metadata.json', metadata)
    # MLX is only used for two small projections and optimization on cached inputs.
    import mlx.core as mx
    import mlx.nn as nn
    # Instrument check: only the intended positive gate receives an auxiliary gradient.
    g = mx.array([[-30., 20.], [-40., 50.]])
    y = mx.array([[1., 0.], [0., 0.]])
    gradient = np.array(mx.grad(lambda scores: auxiliary(mx, scores, y))(g))
    np.testing.assert_array_equal(gradient, [[-62., 0.], [0., 0.]])
    with np.load(prior / 'captures.npz') as cache:
        data = {key: cache[key] for key in accessed}
    arrays, analyses, traces, choice = {}, {}, [], None
    try:
        for bank, f in fixture.items():
            rows = f['development']
            X, canonical = data[bank + '_development_x'], data[bank + '_canonical']
            initial = [data[bank + '_initial_' + kind] for kind in ['gate', 'up']]
            baseline = [data[bank + '_fitted_' + kind] for kind in ['gate', 'up']]
            analyses[bank] = {}
            for key in ['development_x', 'canonical', 'down']:
                arrays[bank + '_' + key] = data[bank + '_' + key]

            def evaluate(name, weights):
                xb = mx.array(X).astype(mx.bfloat16)
                gb, ub = [mx.array(w).astype(mx.bfloat16) for w in weights]
                gate, up = xb @ gb.T, xb @ ub.T
                activation = nn.gelu_approx(gate) * up
                gate, up, activation = [np.array(v.astype(mx.float32)) for v in [gate, up, activation]]
                for key, value in zip(['gate', 'up', 'activation', 'fitted_gate', 'fitted_up'],
                                      [gate, up, activation, *weights]):
                    arrays[bank + '_' + name + '_' + key] = value
                analysis = summarize(activation / canonical, gate, rows)
                analyses[bank][name] = analysis
                if name != 'baseline':
                    analysis['native_gate'] = native_gate(analysis, analyses[bank]['baseline'])
                print('RESULT', bank, name, json.dumps({s: {k: analysis[s][k] for k in
                      ['loss', 'positive_delivery', 'negative_quiet', 'suppressed_positive']} for s in ['train', 'validation']}), flush=True)
                print('THRESHOLD', bank, name, json.dumps({k: v for k, v in analysis['threshold'].items()
                                                         if not k.endswith('_curve')}), flush=True)
                persist(args.out / 'analysis.partial.json', analyses)
                return analysis

            evaluate('baseline', baseline)
            strengths = [0., *STRENGTHS] if bank == 'observed_roster' else [0., choice['strength']]
            candidates = []
            for strength in strengths:
                final_weights = None

                def callback(step, weights, metrics):
                    nonlocal final_weights
                    traces.append(dict(bank=bank, strength=strength, step=step, **metrics))
                    persist(args.out / 'fit_trace.json', traces)
                    print('FIT', json.dumps(traces[-1]), flush=True)
                    if step == config['step']:
                        final_weights = weights

                fit_aux(mx, nn.gelu_approx, X, rows, initial, canonical, config['config'], config['step'], strength, callback)
                assert final_weights is not None
                if strength == 0:
                    for fitted, expected in zip(final_weights, baseline):
                        np.testing.assert_array_equal(fitted, expected)
                    print('ZERO-AUX REPLAY EXACT', bank, flush=True)
                    continue
                name = 'aux_' + str(strength)
                result = evaluate(name, final_weights)
                candidates.append((strength, name, result))
            if bank == 'observed_roster':
                strength, name, result = min(candidates, key=lambda item: (
                    not item[2]['native_gate']['passed'], item[2]['validation']['loss'], STRENGTHS.index(item[0])))
                choice = dict(strength=strength, name=name, step=config['step'], config=config['config'],
                              selected_before_second_roster_fit=True)
                persist(args.out / 'frozen_choice.json', choice)
                print('FROZEN', json.dumps(choice), flush=True)
        assert len(traces) == 24
        assert digest(prior / 'captures.npz') == metadata['prior_cache_sha256']
        np.savez_compressed(args.out / 'arrays.npz', **arrays)
        persist(args.out / 'analysis.json', analyses)
        gates = {bank: analyses[bank][choice['name']]['native_gate'] for bank in fixture}
        result = dict(status='complete', full_model_forwards=0, optimization_steps=9600,
            selected=choice, development_gates=gates, full_model_evaluation_permitted=all(g['passed'] for g in gates.values()),
            zero_aux_replay_exact=True, auxiliary_gradient_check=True,
            arrays_sha256=digest(args.out / 'arrays.npz'), analysis_sha256=digest(args.out / 'analysis.json'),
            fit_trace_sha256=digest(args.out / 'fit_trace.json'), frozen_choice_sha256=digest(args.out / 'frozen_choice.json'))
        persist(args.out / 'summary.json', result)
        print('COMPLETE', json.dumps(result), flush=True)
    except Exception as exc:
        persist(args.out / 'failure.json', dict(type=type(exc).__name__, message=str(exc), full_model_forwards=0))
        raise


if __name__ == '__main__':
    main()
