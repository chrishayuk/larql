#!/usr/bin/env python3
"""Audit complete diagnostic and factorial artifacts from saved rows."""
import argparse
import json
from pathlib import Path

import numpy as np

from binding_readers import reader_metrics
from run_address import HERE, digest, summarize
from run_factorial import gates


def read(path):
    return json.loads(path.read_text())


def audit_diagnostic(root):
    meta, summary, fixture = [read(root / f'{n}.json') for n in ['metadata', 'summary', 'fixture']]
    assert summary['status'] == 'complete'
    assert all(summary['instrument'].values())
    for name, sha in meta['sources'].items():
        assert digest(HERE / name) == sha, name
    assert digest(root / 'fixture.json') == meta['fixture_sha256']
    assert digest(root / 'captures.npz') == summary['captures_sha256']
    count = 0
    for arm in ['clean', 'written', 'removed']:
        rows = [json.loads(line) for line in (root / f'{arm}.jsonl').read_text().splitlines()]
        assert [r['prompt'] for r in rows] == [r['prompt'] for r in fixture['original']]
        assert all(r['prior_top5_exact'] for r in rows)
        count += len(rows)
    predictions = read(root / 'reader_rows.json')
    probes = [r for r in fixture['evaluation'] if r['group'] != 'enroll']
    for layer, readers in predictions.items():
        for name, rows in readers.items():
            assert [r['prompt'] for r in rows] == [r['prompt'] for r in probes]
            assert [r['label'] for r in rows] == [r['label'] for r in probes]
            expected = summary['layers'][layer]['readers'][name]
            actual, rebuilt = reader_metrics(np.array([r['scores'] for r in rows]), probes, expected['threshold'])
            assert actual == expected
            assert rebuilt == rows
    print(f'Diagnostic PASS: {count} actual-forward rows; all reader scores/decisions, coverage and hashes')


def audit_factorial(root):
    meta, summary, fixture = [read(root / f'{n}.json') for n in ['metadata', 'summary', 'fixture']]
    assert summary['status'] == 'complete'
    for name, sha in meta['source_hashes'].items():
        assert digest(HERE / name) == sha, name
    assert digest(root / 'fixture.json') == meta['fixture_sha256']
    assert digest(root / 'addresses.npz') == summary['addresses_sha256']
    arms = {}
    for line in (root / 'rows.jsonl').read_text().splitlines():
        row = json.loads(line)
        arms.setdefault(row['arm'], []).append(row)
    assert len(arms) == 9
    for name, rows in arms.items():
        assert [r['prompt'] for r in rows] == [r['prompt'] for r in fixture['probes']]
        for row in rows:
            if row['fact'] is not None:
                expected = 'Berlin' if name.endswith('/replaced') and row['fact'] == 0 else fixture['facts'][row['fact']][2]
                assert row['expected'] == expected
                assert row['hit'] == (row['top1'] == row['target_id'])
    assert gates(arms) == summary['gate']
    assert {n: summarize(rows) for n, rows in arms.items()} == summary['summary']
    for method in ['original', 'explicit_negatives']:
        for written, replaced in zip(arms[method + '/written'], arms[method + '/replaced']):
            for metric in ['gate', 'up', 'activation']:
                assert written['slots'][metric] == replaced['slots'][metric]
    print(f'Factorial PASS: {sum(len(r) for r in arms.values())} rows; replacement scope, summaries, gates and hashes')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--diagnostic', type=Path)
    ap.add_argument('--factorial', type=Path)
    args = ap.parse_args()
    if args.diagnostic:
        audit_diagnostic(args.diagnostic)
    if args.factorial:
        audit_factorial(args.factorial)


if __name__ == '__main__':
    main()
