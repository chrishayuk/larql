#!/usr/bin/env python3
"""Capture actual activations and compare binding readers before changing an editor."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from binding_fixture import reader_fixture, RELATIONS
from binding_readers import auc, fit_readers, evaluate_roster, reader_metrics
from binding_runtime import Runtime
from run_address import HERE, FACTS, fixture, persist, digest, score


def old_enrollment(probes):
    entities = [f[0] for f in FACTS]
    for row in probes:
        if row['group'] == 'similar_name':
            entities.append(row['prompt'].split(' of ')[1][:-3])
    bindings = [(e, r) for e in entities for r in RELATIONS]
    rows = []
    for row in probes:
        if row['fact'] is not None:
            e, r, _ = FACTS[row['fact']]
            label = bindings.index((e, r))
        elif row['group'] in ['different_relation', 'similar_name']:
            r = row['prompt'].split()[1]
            e = row['prompt'].split(' of ')[1][:-3]
            label = bindings.index((e, r))
        else:
            label, r = None, next((r for r in RELATIONS if r in row['prompt']), None)
        rows.append(dict(row, label=label, relation=r))
    return bindings, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    old = fixture()
    dev, fresh = reader_fixture('development'), reader_fixture('evaluation')
    enrollment, old_labels = old_enrollment(old)
    persist(args.out / 'fixture.json', dict(original=old, development=dev, evaluation=fresh,
                                           original_enrollment=enrollment))
    sources = ['run_discrimination.py', 'binding_fixture.py', 'binding_readers.py',
               'binding_runtime.py', 'DISCRIMINATION_PROTOCOL.md', 'vendor/native.py']
    previous = json.loads((HERE / 'results/address-v1/metadata.json').read_text())
    persist(args.out / 'metadata.json', dict(model=str(args.model), baseline=previous,
        sources={name: digest(HERE / name) for name in sources},
        fixture_sha256=digest(args.out / 'fixture.json')))
    prior_rows = [json.loads(line) for line in (HERE / 'results/address-v1/rows.jsonl').read_text().splitlines()]
    prior = {(r['arm'], r['id']): r for r in prior_rows}
    print('Loading original checkpoint; five fixed layers captured, 26 scored first', flush=True)
    run = Runtime(args.model, layers=(22, 24, 26, 28, 30))
    tensors, arms, baseline_ll = {}, {}, {}
    try:
        run.prepare(FACTS)
        weights, keys = run.make_edit()
        np.savez(args.out / 'original_keys.npz', keys=keys)
        for arm in ['clean', 'written', 'removed']:
            run.assign(weights if arm == 'written' else run.original)
            arms[arm] = []
            with (args.out / f'{arm}.jsonl').open('x') as out:
                for i, probe in enumerate(old):
                    ll, cap = run.forward(probe['prompt'])
                    tensors[f'{arm}_{i}'] = cap[26]['x']
                    if arm == 'clean':
                        baseline_ll[probe['id']] = ll
                    target = run.token(probe['expected']) if probe['fact'] is not None else None
                    s = score(ll, target, baseline_ll[probe['id']], run.tok)
                    ref = prior[(arm, probe['id'])]
                    prior_agree = s['top5'] == ref['top5']
                    row = dict(probe, arm=arm, **s, slots=run.slot_metrics(cap), prior_top5_exact=prior_agree)
                    arms[arm].append(row)
                    out.write(json.dumps(row, allow_nan=False) + '\n')
                    out.flush()
                    if (i + 1) % 30 == 0:
                        print(f'{arm} {i+1}/{len(old)}', flush=True)
            print(arm, 'prior top5 exact', sum(r['prior_top5_exact'] for r in arms[arm]), flush=True)
        instrument = dict(prior_top5_exact=all(r['prior_top5_exact'] for arm in arms.values() for r in arm),
                          removal_full_logits_exact=all(r['max_logit_delta'] == 0 for r in arms['removed']))
        persist(args.out / 'instrument.json', instrument)
        if not all(instrument.values()):
            raise RuntimeError('Instrument differs from original run; inspect before any inference')
        activation = []
        for slot, (e, relation, _) in enumerate(FACTS):
            positives = [r for r in arms['written'] if r['fact'] == slot]
            negatives = [r for r in arms['written'] if r['fact'] is None and relation in r['prompt']]
            for metric in ['gate', 'up', 'activation', 'contribution_norm']:
                ps = [r['slots'][metric][slot] for r in positives]
                ns = [r['slots'][metric][slot] for r in negatives]
                activation.append(dict(entity=e, relation=relation, metric=metric,
                    positive_n=len(ps), negative_n=len(ns), positive_range=[min(ps), max(ps)],
                    negative_range=[min(ns), max(ns)], auc=auc(ps, ns)))
        persist(args.out / 'activation_summary.json', activation)
        captured = {}
        for split, rows in [('development', dev), ('evaluation', fresh)]:
            captured[split] = {layer: [] for layer in run.layers}
            for i, row in enumerate(rows):
                _, cap = run.forward(row['prompt'])
                for layer in run.layers:
                    captured[split][layer].append(cap[layer]['x'])
                if (i + 1) % 30 == 0:
                    print(f'{split} {i+1}/{len(rows)}', flush=True)
            for layer in run.layers:
                captured[split][layer] = np.stack(captured[split][layer])
                tensors[f'{split}_L{layer}'] = captured[split][layer]
        old_keys = []
        for e, r in enrollment:
            _, cap = run.forward(f'The {r} listed for {e} is')
            old_keys.append(cap[26]['x'])
        tensors['old_enrollment_L26'] = np.stack(old_keys)
        summary, all_predictions = {}, {}

        def evaluate(layer):
            readers, thresholds, selection = fit_readers(dev, captured['development'][layer])
            evaluation = evaluate_roster(readers, thresholds, fresh, captured['evaluation'][layer])
            summary[str(layer)] = dict(selection=selection, readers={n: v[0] for n, v in evaluation.items()})
            all_predictions[str(layer)] = {n: v[1] for n, v in evaluation.items()}
            print('LAYER', layer, json.dumps(summary[str(layer)]), flush=True)
            return readers, thresholds

        readers, thresholds = evaluate(26)
        old_x = np.stack([tensors[f'clean_{i}'] for i in range(len(old))])
        original_readers = {name: reader_metrics(fn(old_x, np.stack(old_keys)), old_labels, thresholds[name])
                            for name, fn in readers.items()}
        persist(args.out / 'original_bank_readers.json', {n: dict(summary=v[0], rows=v[1])
                                                          for n, v in original_readers.items()})
        if not any(r['gate'] for r in summary['26']['readers'].values()):
            print('Neither full reader gate passed at L26; evaluating fixed layer screen', flush=True)
            for layer in [22, 24, 28, 30]:
                evaluate(layer)
        np.savez_compressed(args.out / 'captures.npz', **tensors)
        persist(args.out / 'reader_rows.json', all_predictions)
        persist(args.out / 'summary.json', dict(status='complete', instrument=instrument,
            activation=activation, layers=summary, captures_sha256=digest(args.out / 'captures.npz')))
    finally:
        run.close()


if __name__ == '__main__':
    main()
