#!/usr/bin/env python3
"""Audit the E/C/A cube and its independent intervention scopes."""
import argparse
import json
from pathlib import Path

import numpy as np

from decomposition_analysis import analyze, CODES
from decomposition_runtime import expected_sequence
from run_address import HERE, digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    args = ap.parse_args()
    read = lambda name: json.loads((args.run / name).read_text())
    summary, meta, fixture = [read(n + '.json') for n in ['summary', 'metadata', 'fixture']]
    assert summary['status'] == 'complete' and all(summary['instrument'].values())
    for name, sha in meta['source_hashes'].items():
        assert digest(HERE / name) == sha, name
    assert digest(args.run / 'fixture.json') == meta['fixture_sha256']
    assert digest(args.run / 'vectors.npz') == summary['vectors_sha256']
    prior_root = HERE / 'results/oracle-v2'
    assert digest(prior_root / 'rows.jsonl') == meta['prior_rows_sha256']
    assert digest(prior_root / 'normalization_vectors.npz') == meta['prior_vectors_sha256']
    rows = [json.loads(l) for l in (args.run / 'rows.jsonl').read_text().splitlines()]
    assert len(rows) == summary['scored_rows'] == 840
    pids = [p['id'] for p in fixture['probes']]
    for code in CODES:
        assert [r['id'] for r in rows if r['phase'] == 'factorial' and r['code'] == code] == pids
    for code in ['000', '111']:
        assert [r['id'] for r in rows if r['phase'] == 'repeat' and r['code'] == code] == pids
    assert sum(r['phase'] == 'factorial' and r['code'] not in ['000', '111'] for r in rows) == 504
    indices = {(r['code'], r['id']): i for i, r in enumerate(rows) if r['phase'] == 'factorial'}
    data = np.load(args.run / 'vectors.npz')
    for i, row in enumerate(rows):
        baseline = indices[('000', row['id'])]
        raw = data[f'raw_sequence_{i}']
        np.testing.assert_array_equal(raw, data[f'raw_sequence_{baseline}'])
        np.testing.assert_array_equal(data[f'edited_sequence_{i}'],
            expected_sequence(raw, row['code'], row['fact'], row['amplitude']))
        assert row['native_max_delta'] == 0 and row['detector_exact']
        assert row['slots']['gate'] == rows[baseline]['slots']['gate']
        assert row['slots']['up'] == rows[baseline]['slots']['up']
        assert row['hit'] == (row['top1'] == row['target_id'])
        if row['code'] in ['000', '111']:
            assert row['prior_endpoint_exact']
        if row['phase'] == 'repeat':
            assert row['repeat_full_logits_exact']
    result = analyze(rows)
    assert result == read('analysis.json')
    assert result['arms'] == summary['arms'] and result['conditional'] == summary['conditional']
    assert result['minimal_pattern_counts'] == summary['minimal_pattern_counts']
    pairs = read('earlier_position_pairs.json')
    assert len(pairs) == 336
    for pair in pairs:
        a, b = indices[(pair['off'], pair['id'])], indices[(pair['on'], pair['id'])]
        for kind in ['pre_norm', 'post_norm']:
            np.testing.assert_array_equal(data[f'{kind}_{a}'], data[f'{kind}_{b}'])
        assert pair['top1_changed'] == (rows[a]['top1'] != rows[b]['top1'])
        assert pair['margin_delta'] == rows[b]['margin_full'] - rows[a]['margin_full']
    print('PASS: 504 new cells, 336 endpoint verifications, all E/C/A scopes,')
    print('      336 identical-final-vector pairs, summaries, conditional effects and interactions.')
    print('Arms:', json.dumps(summary['arms']))


if __name__ == '__main__':
    main()
