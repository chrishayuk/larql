#!/usr/bin/env python3
"""Six new oracle-component combinations, bracketed by both known endpoints."""
import argparse
import json
from pathlib import Path

import numpy as np

from decomposition_runtime import DecompositionRuntime
from decomposition_analysis import analyze
from run_address import HERE, digest, persist, score


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    prior_root = HERE / 'results/oracle-v2'
    fixture = json.loads((prior_root / 'fixture.json').read_text())
    facts, canonical = fixture['facts'], fixture['canonical_activations']
    probes = [r for r in fixture['probes'] if r['fact'] is not None]
    persist(args.out / 'fixture.json', dict(facts=facts, probes=probes, canonical_activations=canonical,
                                           factor_order=['earlier_suppression', 'competitor_suppression', 'amplitude']))
    names = ['run_decomposition.py', 'decomposition_runtime.py', 'decomposition_analysis.py',
             'binding_runtime.py', 'run_address.py', 'DECOMPOSITION_PROTOCOL.md']
    persist(args.out / 'metadata.json', dict(model=str(args.model), source_hashes={n: digest(HERE / n) for n in names},
        fixture_sha256=digest(args.out / 'fixture.json'),
        prior_metadata=json.loads((prior_root / 'metadata.json').read_text()),
        prior_rows_sha256=digest(prior_root / 'rows.jsonl'),
        prior_vectors_sha256=digest(prior_root / 'normalization_vectors.npz')))
    prior_rows = [json.loads(l) for l in (prior_root / 'rows.jsonl').read_text().splitlines()]
    prior = {(('000' if r['arm'] == 'actual' else '111'), r['id']): (i, r)
             for i, r in enumerate(prior_rows) if r['phase'] == 'routing' and r['arm'] in ['actual', 'oracle']}
    prior_vectors = np.load(prior_root / 'normalization_vectors.npz')
    print('Frozen E/C/A cube: 504 new cells plus 336 endpoint verification forwards', flush=True)
    run = DecompositionRuntime(args.model)
    rows, vectors, reference, baseline_raw = [], {}, {}, {}
    try:
        run.prepare(facts)
        weights, _ = run.make_edit()
        run.assign(weights)
        tids = [run.token(f[2]) for f in facts]
        stages = [('factorial', c) for c in ['000', '111', '001', '010', '011', '100', '101', '110']]
        stages += [('repeat', '000'), ('repeat', '111')]
        with (args.out / 'rows.jsonl').open('x') as out:
            for phase, code in stages:
                stage_rows = []
                for i, probe in enumerate(probes):
                    selected = probe['fact']
                    ll, cap, extra = run.intervention_forward(probe['prompt'], code, selected, canonical[selected])
                    slots = run.slot_metrics(cap)
                    if phase == 'factorial' and code == '000':
                        reference[('000', probe['id'])] = ll
                        baseline_raw[probe['id']] = (slots['gate'], slots['up'], extra['raw_sequence'])
                    if phase == 'factorial' and code == '111':
                        reference[('111', probe['id'])] = ll
                    ref_gate, ref_up, ref_sequence = baseline_raw[probe['id']]
                    assert slots['gate'] == ref_gate and slots['up'] == ref_up
                    np.testing.assert_array_equal(extra['raw_sequence'], ref_sequence)
                    s = score(ll, tids[selected], reference[('000', probe['id'])], run.tok)
                    for source, dest in [('clean_agree', 'agree_000'), ('kl_clean_arm', 'kl_000'),
                                         ('total_variation', 'tv_000'), ('max_logit_delta', 'max_logit_delta_000')]:
                        s[dest] = s.pop(source)
                    row = dict(probe, phase=phase, code=code, amplitude=canonical[selected], **s,
                        slots=slots, native_max_delta=float(extra['native_max_delta']), detector_exact=True)
                    index = len(rows)
                    for name in ['pre_norm', 'post_norm', 'raw_sequence', 'edited_sequence']:
                        vectors[f'{name}_{index}'] = extra[name]
                    if code in ['000', '111']:
                        pi, previous = prior[(code, probe['id'])]
                        assert row['top5'] == previous['top5'] and row['margin_full'] == previous['margin_full']
                        np.testing.assert_array_equal(extra['pre_norm'], prior_vectors[f'pre_{pi}'])
                        np.testing.assert_array_equal(extra['post_norm'], prior_vectors[f'post_{pi}'])
                        row['prior_endpoint_exact'] = True
                    if phase == 'repeat':
                        assert np.array_equal(ll, reference[(code, probe['id'])])
                        row['repeat_full_logits_exact'] = True
                    rows.append(row)
                    stage_rows.append(row)
                    out.write(json.dumps(row, allow_nan=False) + '\n')
                    out.flush()
                    if (i + 1) % 28 == 0:
                        print(phase, code, i+1, '/', len(probes), flush=True)
                print(phase, code, {g: sum(r['hit'] for r in stage_rows if r['group'] == g)
                                    for g in ['installation', 'sentence', 'alias']}, flush=True)
        assert len(rows) == 840
        indices = {(r['code'], r['id']): i for i, r in enumerate(rows) if r['phase'] == 'factorial'}
        e_pairs = []
        for off in ['000', '001', '010', '011']:
            on = '1' + off[1:]
            for probe in probes:
                left, right = indices[(off, probe['id'])], indices[(on, probe['id'])]
                for name in ['pre_norm', 'post_norm']:
                    np.testing.assert_array_equal(vectors[f'{name}_{left}'], vectors[f'{name}_{right}'])
                a, b = rows[left], rows[right]
                e_pairs.append(dict(id=probe['id'], group=probe['group'], off=off, on=on,
                    final_position_vectors_exact=True, top1_changed=a['top1'] != b['top1'],
                    margin_delta=b['margin_full'] - a['margin_full']))
        result = analyze(rows)
        persist(args.out / 'analysis.json', result)
        persist(args.out / 'earlier_position_pairs.json', e_pairs)
        np.savez_compressed(args.out / 'vectors.npz', **vectors)
        summary = dict(status='complete', scored_rows=840, new_cells=504,
            arms=result['arms'], conditional=result['conditional'], minimal_pattern_counts=result['minimal_pattern_counts'],
            vectors_sha256=digest(args.out / 'vectors.npz'), instrument=dict(
                endpoint_prior_exact=True, endpoint_repeats_full_logits_exact=True,
                native_features_exact=True, detector_exact=True, intervention_scope_exact=True,
                earlier_only_final_vectors_exact=True))
        persist(args.out / 'summary.json', summary)
        print('COMPLETE', json.dumps(summary['arms']), flush=True)
    finally:
        run.close()
        prior_vectors.close()


if __name__ == '__main__':
    main()
