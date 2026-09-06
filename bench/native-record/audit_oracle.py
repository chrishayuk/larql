#!/usr/bin/env python3
"""Recompute oracle summaries and normalization finite differences from artifacts."""
import argparse
import json
from pathlib import Path

import numpy as np

from oracle_runtime import vector_metrics
from run_address import HERE, digest
from run_oracle import summarize, SCALES, LADDER_RECORDS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    args = ap.parse_args()
    meta = json.loads((args.run / 'metadata.json').read_text())
    summary = json.loads((args.run / 'summary.json').read_text())
    fixture = json.loads((args.run / 'fixture.json').read_text())
    assert summary['status'] == 'complete' and all(summary['instrument'].values())
    for name, sha in meta['source_hashes'].items():
        assert digest(HERE / name) == sha, name
    assert digest(args.run / 'fixture.json') == meta['fixture_sha256']
    assert digest(args.run / 'normalization_vectors.npz') == summary['vector_sha256']
    assert digest(HERE / 'results/factorial-v1/rows.jsonl') == meta['prior_rows_sha256']
    rows = [json.loads(l) for l in (args.run / 'rows.jsonl').read_text().splitlines()]
    assert len(rows) == summary['scored_rows'] == 1500
    recomputed = summarize(rows)
    assert all(summary[k] == v for k, v in recomputed.items())
    all_ids = [r['id'] for r in fixture['probes']]
    labeled_ids = [r['id'] for r in fixture['probes'] if r['fact'] is not None]
    for arm in ['zero', 'actual', 'oracle', 'wrong', 'wrong_matched']:
        rs = [r for r in rows if r['phase'] == 'routing' and r['arm'] == arm]
        assert [r['id'] for r in rs] == (all_ids if arm in ['zero', 'actual'] else labeled_ids)
    actual = {r['id']: r for r in rows if r['phase'] == 'routing' and r['arm'] == 'actual'}
    zero_index = {r['id']: i for i, r in enumerate(rows) if r['phase'] == 'routing' and r['arm'] == 'zero'}
    data = np.load(args.run / 'normalization_vectors.npz')
    assert len(data.files) == 2 * len(rows)
    for i, row in enumerate(rows):
        z = zero_index[row['id']]
        rebuilt = vector_metrics(data[f'pre_{i}'], data[f'post_{i}'], data[f'pre_{z}'], data[f'post_{z}'],
                                  row['normalization']['contribution_norm'])
        assert rebuilt == row['normalization']
        assert row['native_max_delta'] == 0
        if row['phase'] == 'ladder':
            assert row['scale'] in SCALES and row['scaled_record'] in LADDER_RECORDS
            assert row['raw_activation'] == actual[row['id']]['raw_activation']
            assert row['slots']['gate'] == actual[row['id']]['slots']['gate']
            assert row['slots']['up'] == actual[row['id']]['slots']['up']
        if row['arm'] in ['oracle', 'wrong', 'wrong_matched']:
            intended = row['fact']
            selected = intended if row['arm'] == 'oracle' else (intended + 3) % 12
            assert row['selected_record'] == selected
            ref = intended if row['arm'] == 'wrong_matched' else selected
            assert row['oracle_amplitude'] == fixture['canonical_activations'][ref]
            expected = [0.] * 12
            expected[selected] = row['oracle_amplitude']
            assert row['slots']['activation'] == expected
    print('PASS: 1500 rows, arm coverage, source hashes, selected records, frozen detector,')
    print('      all 1500 normalization finite differences and all routing/ladder summaries.')
    print('Routing:', json.dumps(summary['routing']))
    print('Shared tested scales at >=80% oracle reads:', summary['shared_oracle_scales_at_80_percent'])


if __name__ == '__main__':
    main()
