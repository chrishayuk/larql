#!/usr/bin/env python3
"""Re-derive the gate and useful slices from persisted rows, without MLX."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from run_address import HERE, digest, gates, summarize


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    args = ap.parse_args()
    metadata = json.loads((args.run / 'metadata.json').read_text())
    for key, path in [('source_sha256', HERE / 'vendor/native.py'),
                      ('harness_sha256', HERE / 'run_address.py'),
                      ('protocol_sha256', HERE / 'PROTOCOL.md'),
                      ('fixture_sha256', args.run / 'fixture.json')]:
        assert metadata[key] == digest(path), (key, 'changed since run')
    rows = [json.loads(line) for line in (args.run / 'rows.jsonl').read_text().splitlines()]
    arms = defaultdict(list)
    for row in rows:
        arms[row['arm']].append(row)
    fixture = json.loads((args.run / 'fixture.json').read_text())['probes']
    for name, arm in arms.items():
        assert [r['id'] for r in arm] == [r['id'] for r in fixture], name
        assert [r['prompt'] for r in arm] == [r['prompt'] for r in fixture], name
        for r in arm:
            if 'hit' in r:
                assert r['hit'] == (r['top1'] == r['target_id'])
    summary = json.loads((args.run / 'summary.json').read_text())
    assert summary['status'] == 'complete'
    assert summary['gate'] == gates(arms)
    assert summary['summary'] == {k: summarize(v) for k, v in arms.items()}
    print(f'PASS: {len(rows)} rows, provenance hashes and aggregate/gate recomputation')
    print('Advance:', summary['gate']['advance'])
    for name in ['written', 'replace_0', 'replace_1', 'replace_2']:
        print(name)
        for group in ['installation', 'sentence', 'alias']:
            rs = [r for r in arms[name] if r['group'] == group]
            print(f"  {group}: {sum(r['hit'] for r in rs)}/{len(rs)}, "
                  f"min full-vocabulary margin {min(r['margin_full'] for r in rs):.6f}")
    base = {r['id']: r for r in arms['clean']}
    print('Held-out factual/generic control changes (original write):')
    for r in arms['written']:
        if r['group'] in ['other_entity', 'unrelated'] and not r['clean_agree']:
            print(f"  {r['prompt']!r}: {base[r['id']]['top1_text']!r} -> {r['top1_text']!r}")
    print('Control category agreements (original write):')
    for group, s in summarize(arms['written']).items():
        if group not in ['installation', 'sentence', 'alias']:
            print(f"  {group}: {s['agreement']}/{s['n']}")


if __name__ == '__main__':
    main()
