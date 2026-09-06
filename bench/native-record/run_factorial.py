#!/usr/bin/env python3
"""Twelve native records; change only the editor's explicit negative coverage."""
import argparse
import json
from pathlib import Path

import numpy as np

from binding_fixture import factorial_fixture
from binding_runtime import Runtime
from run_address import HERE, persist, digest, score, summarize


def gates(arms):
    checks = {}
    for method in ['original', 'explicit_negatives']:
        for state in ['written', 'replaced']:
            name = f'{method}/{state}'
            rows = arms[name]
            sums = summarize(rows)
            checks[name + '/installation'] = sums['installation']['hits'] == 12
            for group in ['sentence', 'alias']:
                checks[name + '/' + group] = sums[group]['hits'] / sums[group]['n'] >= .8
            for fact in range(12):
                rs = [r for r in rows if r['group'] == 'sentence' and r['fact'] == fact]
                checks[f'{name}/binding_{fact}'] = sum(r['hit'] for r in rs) / len(rs) >= .75
            for group in ['similar_name', 'other_entity', 'unrelated']:
                checks[name + '/' + group] = sums[group]['agreement'] / sums[group]['n'] >= .95
        replacement = arms[method + '/replaced']
        checks[method + '/other_11_canonical_preserved'] = all(
            r['hit'] for r in replacement if r['group'] == 'installation' and r['fact'] != 0)
        checks[method + '/removal_exact'] = all(r['max_logit_delta'] == 0 for r in arms[method + '/removed'])
        checks[method + '/restoration_exact'] = all(r['written_max_logit_delta'] == 0
                                                   for r in arms[method + '/restored'])
    return dict(method_pass={m: all(v for k, v in checks.items() if k.startswith(m + '/'))
                             for m in ['original', 'explicit_negatives']},
                checks=checks, failed=[k for k, v in checks.items() if not v])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    facts, probes = factorial_fixture()
    persist(args.out / 'fixture.json', dict(facts=facts, replacement=dict(fact=0, value='Berlin'), probes=probes))
    names = ['run_factorial.py', 'binding_fixture.py', 'binding_runtime.py', 'DISCRIMINATION_PROTOCOL.md']
    prior = json.loads((HERE / 'results/address-v1/metadata.json').read_text())
    persist(args.out / 'metadata.json', dict(model=str(args.model), baseline=prior,
        source_hashes={n: digest(HERE / n) for n in names}, fixture_sha256=digest(args.out / 'fixture.json')))
    print('Factorial:', len(facts), 'records;', len(probes), 'probes; 9 states', flush=True)
    run = Runtime(args.model)
    try:
        tids = {v: run.token(v) for v in [*(f[2] for f in facts), 'Berlin']}
        persist(args.out / 'tokens.json', tids)
        run.prepare(facts)
        edits = {name: run.make_edit(explicit=name == 'explicit_negatives')
                 for name in ['original', 'explicit_negatives']}
        np.savez(args.out / 'keys.npz', **{m: pair[1] for m, pair in edits.items()})
        persist(args.out / 'installation.json', dict(slots=run.slots, median_scales=run.scales,
            base_negative_count=41, explicit_extra_negative_count=10,
            address_retained_cosine={name: [float(key @ run.native.unit(x))
                for key, x in zip(pair[1], run.addresses)] for name, pair in edits.items()}))
        base, written, arms, addresses = {}, {}, {}, {}
        states = ['clean'] + [f'{m}/{s}' for m in edits for s in ['written', 'replaced', 'removed', 'restored']]
        with (args.out / 'rows.jsonl').open('x') as out:
            for arm in states:
                method, state = arm.split('/') if '/' in arm else (None, 'clean')
                if state in ['clean', 'removed']:
                    run.assign(run.original)
                elif state == 'replaced':
                    run.assign(run.replace(edits[method][0], 0, 'Berlin'))
                else:
                    run.assign(edits[method][0])
                arms[arm] = []
                for i, probe in enumerate(probes):
                    ll, cap = run.forward(probe['prompt'])
                    addresses[f'{arm.replace("/", "_")}_{i}'] = cap[26]['x']
                    if state == 'clean':
                        base[probe['id']] = ll
                    elif state == 'written':
                        written[(method, probe['id'])] = ll
                    expected = 'Berlin' if state == 'replaced' and probe['fact'] == 0 else probe['expected']
                    target = tids[expected] if expected is not None else None
                    row = dict(probe, arm=arm, expected=expected, slots=run.slot_metrics(cap),
                               **score(ll, target, base[probe['id']], run.tok))
                    if state == 'restored':
                        row['written_max_logit_delta'] = float(np.max(np.abs(ll - written[(method, probe['id'])])))
                    arms[arm].append(row)
                    out.write(json.dumps(row, allow_nan=False) + '\n')
                    out.flush()
                    if (i + 1) % 40 == 0:
                        print(arm, i+1, '/', len(probes), flush=True)
                print(arm, json.dumps(summarize(arms[arm])), flush=True)
                persist(args.out / 'summary.partial.json', {name: summarize(rows) for name, rows in arms.items()})
        np.savez_compressed(args.out / 'addresses.npz', **addresses)
        result = dict(status='complete', summary={name: summarize(rows) for name, rows in arms.items()},
                      gate=gates(arms), addresses_sha256=digest(args.out / 'addresses.npz'))
        persist(args.out / 'summary.json', result)
        print('GATE', json.dumps(result['gate']), flush=True)
    finally:
        run.close()


if __name__ == '__main__':
    main()
