#!/usr/bin/env python3
"""Bounded oracle-routing diagnostic and fixed value-strength ladder."""
import argparse
import json
from pathlib import Path

import numpy as np

from binding_fixture import factorial_fixture
from oracle_runtime import OracleRuntime, vector_metrics
from run_address import HERE, digest, persist, score

SCALES = [0., .5, 1., 2., 4., 8.]
LADDER_RECORDS = [1, 4]


def summarize(rows):
    routing = {}
    actual = {r['id']: r for r in rows if r['phase'] == 'routing' and r['arm'] == 'actual'}
    for arm in ['actual', 'zero', 'oracle', 'wrong', 'wrong_matched']:
        rs = [r for r in rows if r['phase'] == 'routing' and r['arm'] == arm and r['fact'] is not None]
        routing[arm] = {g: dict(n=sum(r['group'] == g for r in rs),
            hits=sum(r['hit'] for r in rs if r['group'] == g),
            selected_value_hits=sum(r.get('selected_value_hit', False) for r in rs if r['group'] == g))
            for g in ['installation', 'sentence', 'alias']}
        if arm == 'oracle':
            sentence = [r for r in rs if r['group'] == 'sentence']
            routing[arm]['recovery'] = dict(
                failed_before=sum(not actual[r['id']]['hit'] for r in sentence),
                recovered=sum(r['hit'] and not actual[r['id']]['hit'] for r in sentence),
                successes_lost=sum(not r['hit'] and actual[r['id']]['hit'] for r in sentence))
    ladder = {}
    for record in LADDER_RECORDS:
        ladder[str(record)] = {}
        for scale in SCALES:
            rs = [r for r in rows if r['phase'] == 'ladder' and r['scaled_record'] == record and r['scale'] == scale]
            target = {arm: [r for r in rs if r['arm'] == arm and r['fact'] == record]
                      for arm in ['actual', 'oracle']}
            collateral = [r for r in rs if r['arm'] == 'actual' and r['fact'] != record]
            ladder[str(record)][str(scale)] = dict(
                target={arm: dict(n=len(v), hits=sum(r['hit'] for r in v),
                    margins=[r['margin_full'] for r in v]) for arm, v in target.items()},
                collateral={g: dict(n=sum(r['group'] == g for r in collateral),
                    agree_unscaled=sum(r['agree_actual'] for r in collateral if r['group'] == g),
                    agree_clean=sum(r['agree_clean_checkpoint'] for r in collateral if r['group'] == g))
                    for g in ['installation', 'similar_name', 'other_entity', 'unrelated']})
    shared = [s for s in SCALES if all(ladder[str(r)][str(s)]['target']['oracle']['hits'] / 7 >= .8
                                      for r in LADDER_RECORDS)]
    return dict(routing=routing, ladder=ladder, shared_oracle_scales_at_80_percent=shared)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    facts, probes = factorial_fixture()
    labeled = [r for r in probes if r['fact'] is not None]
    prior_root = HERE / 'results/factorial-v1'
    prior_rows = [json.loads(l) for l in (prior_root / 'rows.jsonl').read_text().splitlines()]
    prior = {r['id']: r for r in prior_rows if r['arm'] == 'original/written'}
    clean = {r['id']: r for r in prior_rows if r['arm'] == 'clean'}
    canonical = [next(r for r in prior.values() if r['fact'] == i and r['group'] == 'installation')
                 ['slots']['activation'][i] for i in range(12)]
    persist(args.out / 'fixture.json', dict(facts=facts, probes=probes, canonical_activations=canonical,
                                           scales=SCALES, ladder_records=LADDER_RECORDS))
    names = ['run_oracle.py', 'oracle_runtime.py', 'binding_runtime.py', 'binding_fixture.py',
             'run_address.py', 'ORACLE_PROTOCOL.md']
    persist(args.out / 'metadata.json', dict(model=str(args.model), baseline=json.loads(
        (prior_root / 'metadata.json').read_text()), source_hashes={n: digest(HERE / n) for n in names},
        fixture_sha256=digest(args.out / 'fixture.json'), prior_rows_sha256=digest(prior_root / 'rows.jsonl')))
    print('Original factorial editor; 1500 bounded scored forwards', flush=True)
    run = OracleRuntime(args.model)
    rows, zero, actual, vectors = [], {}, {}, {}
    try:
        run.prepare(facts)
        weights, _ = run.make_edit()
        run.assign(weights)
        token_ids = [run.token(f[2]) for f in facts]
        persist(args.out / 'tokens.json', token_ids)
        with (args.out / 'rows.jsonl').open('x') as stream:
            def execute(probe, phase, arm, selected=None, amplitude=0., record=None, scale=1.):
                ll, cap, extra = run.routed_forward(probe['prompt'], arm if arm != 'actual' else 'actual',
                                                   selected, amplitude)
                pid = probe['id']
                if phase == 'routing' and arm == 'zero':
                    zero[pid] = (ll, extra['pre_norm'], extra['post_norm'])
                if phase == 'routing' and arm == 'actual':
                    actual[pid] = (ll, cap[26]['gate'][run.slots].tolist(),
                                   cap[26]['up'][run.slots].tolist(), extra['raw_activation'].tolist())
                reference = actual[pid][0] if pid in actual else zero[pid][0]
                target = token_ids[probe['fact']] if probe['fact'] is not None else None
                s = score(ll, target, reference, run.tok)
                for source, dest in [('clean_agree', 'agree_actual'), ('kl_clean_arm', 'kl_actual'),
                                     ('total_variation', 'tv_actual'), ('max_logit_delta', 'max_logit_delta_actual')]:
                    value = s.pop(source)
                    s[dest] = value if pid in actual else None
                slots = run.slot_metrics(cap)
                contribution = slots['contribution_norm'][selected] if selected is not None else 0.
                row = dict(probe, phase=phase, arm=arm, selected_record=selected,
                    oracle_amplitude=amplitude, scaled_record=record, scale=scale, **s,
                    agree_clean_checkpoint=int(ll.argmax()) == clean[pid]['top1'], slots=slots,
                    raw_activation=extra['raw_activation'].tolist(), native_max_delta=float(extra['native_max_delta']),
                    normalization=vector_metrics(extra['pre_norm'], extra['post_norm'],
                                                 zero[pid][1], zero[pid][2], contribution))
                if selected is not None:
                    selected_score = score(ll, token_ids[selected], reference, run.tok)
                    row.update(selected_value=facts[selected][2], selected_value_hit=selected_score['hit'],
                               selected_value_margin=selected_score['margin_full'])
                if target is not None:
                    zl = zero[pid][0].astype(np.float64)
                    zm = float(zl[target] - np.max(np.delete(zl, target)))
                    row['zero_margin_full'] = zm
                    row['margin_delta_zero'] = row['margin_full'] - zm
                if phase == 'routing' and arm == 'actual':
                    assert row['top5'] == prior[pid]['top5'], ('baseline mismatch', pid)
                    if probe['group'] == 'installation':
                        assert slots['activation'][probe['fact']] == canonical[probe['fact']]
                    row['prior_top5_exact'] = True
                if phase == 'ladder':
                    assert slots['gate'] == actual[pid][1] and slots['up'] == actual[pid][2]
                    assert row['raw_activation'] == actual[pid][3]
                    row['detector_exact'] = True
                if phase == 'restoration':
                    assert np.array_equal(ll, actual[pid][0]), ('restore mismatch', pid)
                    row['restoration_full_logits_exact'] = True
                index = len(rows)
                vectors[f'pre_{index}'], vectors[f'post_{index}'] = extra['pre_norm'], extra['post_norm']
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                if len(rows) % 40 == 0:
                    print(len(rows), phase, arm, 'record', record, 'scale', scale, flush=True)

            for arm in ['zero', 'actual']:
                for probe in probes:
                    execute(probe, 'routing', arm, selected=probe['fact'] if arm == 'actual' else None)
            for arm in ['oracle', 'wrong', 'wrong_matched']:
                for probe in labeled:
                    intended = probe['fact']
                    selected = intended if arm == 'oracle' else (intended + 3) % 12
                    amplitude = canonical[intended if arm == 'wrong_matched' else selected]
                    execute(probe, 'routing', arm, selected, amplitude)
            persist(args.out / 'routing_summary.json', summarize(rows)['routing'])
            print('ROUTING', json.dumps(summarize(rows)['routing']), flush=True)
            for record in LADDER_RECORDS:
                bank = [p for p in probes if p['fact'] == record or p['group'] in ['installation', 'similar_name', 'unrelated']
                        or (p['group'] == 'other_entity' and 'currency' in p['prompt'])]
                assert len(bank) == 62
                for scale in SCALES:
                    run.assign(run.scale_column(weights, record, scale))
                    for probe in bank:
                        execute(probe, 'ladder', 'actual', selected=record, record=record, scale=scale)
                    for probe in labeled:
                        if probe['fact'] == record:
                            execute(probe, 'ladder', 'oracle', record, canonical[record], record, scale)
            run.assign(weights)
            for probe in labeled:
                execute(probe, 'restoration', 'actual', selected=probe['fact'])
        assert len(rows) == 1500
        np.savez_compressed(args.out / 'normalization_vectors.npz', **vectors)
        summary = summarize(rows)
        summary.update(status='complete', scored_rows=len(rows),
            vector_sha256=digest(args.out / 'normalization_vectors.npz'),
            instrument=dict(prior_actual_exact=all(r.get('prior_top5_exact', False) for r in rows
                if r['phase'] == 'routing' and r['arm'] == 'actual'),
                native_features_exact=all(r['native_max_delta'] == 0 for r in rows),
                frozen_detector_exact=all(r['detector_exact'] for r in rows if r['phase'] == 'ladder'),
                restoration_exact=all(r['restoration_full_logits_exact'] for r in rows if r['phase'] == 'restoration')))
        persist(args.out / 'summary.json', summary)
        print('COMPLETE', json.dumps(summary), flush=True)
    finally:
        run.close()


if __name__ == '__main__':
    main()
