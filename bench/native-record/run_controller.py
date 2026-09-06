#!/usr/bin/env python3
"""Bounded native gate/up fit, then two frozen fresh-bank evaluations."""
import argparse
import json
from pathlib import Path

import numpy as np

from controller_analysis import analyze
from controller_fixture import fixtures
from controller_fit import CANDIDATES, fit, delivery
from controller_runtime import ControllerRuntime
from run_address import HERE, persist, digest, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    fixture = fixtures()
    persist(args.out / 'fixture.json', fixture)
    names = ['run_controller.py', 'controller_fixture.py', 'controller_fit.py', 'controller_runtime.py',
             'controller_analysis.py', 'CONTROLLER_PROTOCOL.md', 'binding_runtime.py', 'binding_fixture.py',
             'run_address.py', 'vendor/native.py']
    prior = HERE / 'results/factorial-v1'
    persist(args.out / 'metadata.json', dict(model=str(args.model),
        source_hashes={n: digest(HERE / n) for n in names}, fixture_sha256=digest(args.out / 'fixture.json'),
        prior_metadata=json.loads((prior / 'metadata.json').read_text()),
        prior_addresses_sha256=digest(prior / 'addresses.npz')))
    prior_x = np.load(prior / 'addresses.npz')
    run = ControllerRuntime(args.model)
    all_rows, traces, arrays, choice = [], [], {}, None
    try:
        with (args.out / 'rows.jsonl').open('x') as output:
            for bank, data in fixture.items():
                facts, dev, probes = data['facts'], data['development'], data['evaluation']
                run.oracle = False
                run.prepare(facts)
                existing, _ = run.make_edit()
                run.assign(existing)
                canonical = []
                for i, (e, relation, _) in enumerate(facts):
                    ll, cap, seq = run.checked_forward(f'The {relation} of {e} is')
                    canonical.append(float(cap[26]['activation'][run.slots[i]]))
                canonical = np.array(canonical, dtype=np.float32)
                assert np.all(canonical > 0)
                if bank == 'observed_roster':
                    previous = json.loads((HERE / 'results/oracle-v2/fixture.json').read_text())
                    np.testing.assert_array_equal(canonical, previous['canonical_activations'])
                    X = np.stack([prior_x[f'clean_{i}'] for i in range(len(dev))])
                    for i in range(12):
                        np.testing.assert_array_equal(run.addresses[i], X[i * 8])
                else:
                    captures = []
                    for i, row in enumerate(dev):
                        _, cap, _ = run.checked_forward(row['prompt'])
                        captures.append(cap[26]['x'])
                        if (i + 1) % 28 == 0:
                            print(bank, 'development capture', i + 1, '/', len(dev), flush=True)
                    X = np.stack(captures)
                initial = [np.array(w[run.mx.array(run.slots)].astype(run.mx.float32)) for w in existing[:2]]
                arrays[bank + '_development_x'] = X
                arrays[bank + '_canonical'] = canonical
                arrays[bank + '_initial_gate'], arrays[bank + '_initial_up'] = initial
                arrays[bank + '_down'] = np.array(existing[2][:, run.mx.array(run.slots)].astype(run.mx.float32))
                best, fitted_rows = None, None
                candidates = CANDIDATES if bank == 'observed_roster' else [choice['config']]
                for candidate, config in enumerate(candidates):
                    steps = 1600 if bank == 'observed_roster' else choice['step']

                    def callback(step, weights, metrics):
                        nonlocal best, fitted_rows
                        trace = dict(bank=bank, candidate=candidate, config=config, step=step, **metrics)
                        traces.append(trace)
                        print('FIT', json.dumps(trace), flush=True)
                        persist(args.out / 'fit_trace.json', traces)
                        key = (metrics['validation_loss'], candidate, step)
                        if bank == 'observed_roster':
                            if best is None or key < best['key']:
                                best = dict(key=key, config=config, step=step, candidate=candidate)
                                fitted_rows = weights
                        elif step == steps:
                            # No validation-based selection on the fresh roster.
                            fitted_rows = weights

                    fit(run.mx, run.activation_fn, X, dev, initial, canonical, config, steps, callback)
                if bank == 'observed_roster':
                    choice = best
                    persist(args.out / 'frozen_choice.json', dict(choice=choice,
                        selected_before_evaluation=True, source_hashes={n: digest(HERE / n) for n in names}))
                    print('FROZEN CHOICE', json.dumps(choice), flush=True)
                assert fitted_rows is not None
                fitted = run.fitted_weights(existing, fitted_rows)
                assert fitted[2] is existing[2]
                arrays[bank + '_fitted_gate'], arrays[bank + '_fitted_up'] = fitted_rows
                replaced = run.replace(fitted, 0, 'Berlin')
                assert replaced[0] is fitted[0] and replaced[1] is fitted[1]
                d0 = np.array(fitted[2].astype(run.mx.float32))
                d1 = np.array(replaced[2].astype(run.mx.float32))
                assert np.any(d0[:, run.slots[0]] != d1[:, run.slots[0]])
                np.testing.assert_array_equal(np.delete(d0, run.slots[0], axis=1),
                                              np.delete(d1, run.slots[0], axis=1))
                del d0, d1
                tids = {v: run.token(v) for v in [*(f[2] for f in facts), 'Berlin']}
                base, fit_reference, raw_fit, clean_cap = {}, {}, {}, {}
                states = dict(clean=run.original, existing=existing, fitted=fitted, oracle=existing,
                              replaced=replaced, restored=fitted, removed=run.original)
                for arm, weights in states.items():
                    run.assign(weights)
                    run.frozen_check()
                    stage = []
                    for index, probe in enumerate(probes):
                        record = probe['fact']
                        ll, cap, seq = run.checked_forward(probe['prompt'], oracle=arm == 'oracle',
                            record=record if arm == 'oracle' else None,
                            amplitude=float(canonical[record]) if record is not None and arm == 'oracle' else 0.)
                        if arm == 'clean':
                            base[probe['id']] = ll
                            clean_cap[probe['id']] = (cap[26]['x'], cap[26]['activation'][:min(run.slots)])
                        x0, native0 = clean_cap[probe['id']]
                        np.testing.assert_array_equal(cap[26]['x'], x0)
                        np.testing.assert_array_equal(cap[26]['activation'][:min(run.slots)], native0)
                        if arm == 'fitted':
                            fit_reference[probe['id']] = ll
                            raw_fit[probe['id']] = (seq['raw'], cap[26]['gate'][run.slots], cap[26]['up'][run.slots])
                        expected = 'Berlin' if arm == 'replaced' and record == 0 else probe['expected']
                        row = dict(probe, bank=bank, arm=arm, expected=expected,
                            **score(ll, tids[expected] if expected else None, base[probe['id']], run.tok),
                            slots=run.slot_metrics(cap), delivery=delivery(seq['actual'][0, -1], canonical, record),
                            native_exact=True, input_exact=True, frozen_parameters=True, intervention_scope_exact=True)
                        row.setdefault('hit', None)
                        row.setdefault('margin_full', None)
                        arrays[f'{bank}_{arm}_{index}_raw'] = seq['raw']
                        arrays[f'{bank}_{arm}_{index}_actual'] = seq['actual']
                        if arm in ['replaced', 'restored']:
                            raw0, g0, u0 = raw_fit[probe['id']]
                            np.testing.assert_array_equal(seq['raw'], raw0)
                            np.testing.assert_array_equal(cap[26]['gate'][run.slots], g0)
                            np.testing.assert_array_equal(cap[26]['up'][run.slots], u0)
                            row['detector_exact'] = True
                        if arm == 'restored':
                            row['reference_max_logit_delta'] = float(np.max(np.abs(ll - fit_reference[probe['id']])))
                            assert row['reference_max_logit_delta'] == 0.
                        if arm == 'removed':
                            assert row['max_logit_delta'] == 0.
                        all_rows.append(row)
                        stage.append(row)
                        output.write(json.dumps(row, allow_nan=False) + '\n')
                        output.flush()
                        if (index + 1) % 28 == 0:
                            print(bank, arm, index + 1, '/', len(probes), flush=True)
                    print('ARM', bank, arm, {g: dict(n=sum(r['group'] == g for r in stage),
                        hits=sum(bool(r['hit']) for r in stage if r['group'] == g),
                        delivered=sum(r['delivery']['delivery_ok'] for r in stage if r['group'] == g),
                        clean_agreement=sum(r['clean_agree'] for r in stage if r['group'] == g))
                        for g in sorted({r['group'] for r in stage})}, flush=True)
                # Release full-vocabulary reference arrays before the next installation.
                del base, fit_reference, raw_fit, clean_cap, states, fitted, existing, replaced
                run.oracle = False
                run.assign(run.original)
                run.frozen_check()
        assert len(all_rows) == 2352
        np.savez_compressed(args.out / 'captures.npz', **arrays)
        result = dict(status='complete', scored_rows=len(all_rows), **analyze(all_rows),
            captures_sha256=digest(args.out / 'captures.npz'),
            frozen_choice_sha256=digest(args.out / 'frozen_choice.json'))
        persist(args.out / 'summary.json', result)
        print('COMPLETE', json.dumps(result['gate']), flush=True)
    except Exception as exc:
        persist(args.out / 'failure.json', dict(type=type(exc).__name__, message=str(exc), scored_rows=len(all_rows)))
        raise
    finally:
        run.close()
        prior_x.close()


if __name__ == '__main__':
    main()
