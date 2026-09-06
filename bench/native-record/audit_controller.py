#!/usr/bin/env python3
"""Read-only, GPU-free audit of a completed gate/up-controller artifact."""
import json
import sys
from pathlib import Path
import numpy as np

from controller_analysis import analyze
from controller_fixture import fixtures
from controller_fit import CANDIDATES, CHECKPOINTS, delivery
from run_address import HERE, digest


def audit(root):
    metadata = json.loads((root / 'metadata.json').read_text())
    for name, expected in metadata['source_hashes'].items():
        assert digest(HERE / name) == expected, name
    assert digest(root / 'fixture.json') == metadata['fixture_sha256']
    assert digest(HERE / 'results/factorial-v1/addresses.npz') == metadata['prior_addresses_sha256']
    fixture = json.loads((root / 'fixture.json').read_text())
    assert fixture == json.loads(json.dumps(fixtures()))
    summary = json.loads((root / 'summary.json').read_text())
    assert summary['status'] == 'complete' and summary['scored_rows'] == 2352
    assert digest(root / 'captures.npz') == summary['captures_sha256']
    assert digest(root / 'frozen_choice.json') == summary['frozen_choice_sha256']
    frozen = json.loads((root / 'frozen_choice.json').read_text())
    assert frozen['selected_before_evaluation']
    assert frozen['source_hashes'] == metadata['source_hashes']
    choice = frozen['choice']
    trace = json.loads((root / 'fit_trace.json').read_text())
    observed = [r for r in trace if r['bank'] == 'observed_roster']
    assert [(r['candidate'], r['step']) for r in observed] == [
        (i, step) for i in range(4) for step in CHECKPOINTS]
    for r in observed:
        assert r['config'] == CANDIDATES[r['candidate']]
    best = min(observed, key=lambda r: (r['validation_loss'], r['candidate'], r['step']))
    assert choice['config'] == best['config'] and choice['step'] == best['step']
    assert choice['key'] == [best['validation_loss'], best['candidate'], best['step']]
    fresh = [r for r in trace if r['bank'] == 'fresh_roster']
    assert [r['step'] for r in fresh] == [s for s in CHECKPOINTS if s <= choice['step']]
    assert all(r['config'] == choice['config'] for r in fresh)
    rows = [json.loads(line) for line in (root / 'rows.jsonl').read_text().splitlines()]
    assert len(rows) == 2352
    lookup = {(r['bank'], r['arm'], r['id']): r for r in rows}
    assert len(lookup) == len(rows)
    arms = ['clean', 'existing', 'fitted', 'oracle', 'replaced', 'restored', 'removed']
    assert [(r['bank'], r['arm'], r['id']) for r in rows] == [
        (bank, arm, p['id']) for bank, data in fixture.items() for arm in arms for p in data['evaluation']]
    with np.load(root / 'captures.npz') as vectors:
        for bank, data in fixture.items():
            canonical = vectors[bank + '_canonical']
            assert canonical.shape == (12,) and np.all(canonical > 0)
            assert vectors[bank + '_development_x'].shape == (168, 2560)
            assert vectors[bank + '_down'].shape == (2560, 12)
            for kind in ['gate', 'up']:
                initial = vectors[bank + '_initial_' + kind]
                fitted = vectors[bank + '_fitted_' + kind]
                assert initial.shape == fitted.shape == (12, 2560)
                assert np.isfinite(fitted).all() and not np.array_equal(initial, fitted)
            for arm in arms:
                for i, probe in enumerate(data['evaluation']):
                    r = lookup[(bank, arm, probe['id'])]
                    clean = lookup[(bank, 'clean', probe['id'])]
                    fit = lookup[(bank, 'fitted', probe['id'])]
                    assert all(r[k] for k in ['native_exact', 'input_exact', 'frozen_parameters', 'intervention_scope_exact'])
                    raw = vectors[f'{bank}_{arm}_{i}_raw']
                    actual = vectors[f'{bank}_{arm}_{i}_actual']
                    expected = raw.copy()
                    if arm == 'oracle':
                        expected[:, -1, :] = 0
                        if probe['fact'] is not None:
                            expected[0, -1, probe['fact']] = canonical[probe['fact']]
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(r['slots']['activation'], actual[0, -1])
                    assert r['delivery'] == delivery(actual[0, -1], canonical, probe['fact'])
                    assert r['clean_agree'] == (r['top1'] == clean['top1'])
                    if probe['fact'] is not None:
                        assert r['hit'] == (r['top1'] == r['target_id'])
                        assert r['expected'] == ('Berlin' if arm == 'replaced' and probe['fact'] == 0 else probe['expected'])
                    else:
                        assert r['hit'] is None and r['margin_full'] is None
                    if arm in ['replaced', 'restored']:
                        assert r['detector_exact']
                        np.testing.assert_array_equal(raw, vectors[f'{bank}_fitted_{i}_raw'])
                        for key in ['gate', 'up', 'activation']:
                            assert r['slots'][key] == fit['slots'][key]
                    if arm == 'restored':
                        assert r['reference_max_logit_delta'] == 0
                        assert r['top5'] == fit['top5'] and r['margin_full'] == fit['margin_full']
                    if arm == 'removed':
                        assert r['max_logit_delta'] == 0 and r['top5'] == clean['top5']
    recomputed = analyze(rows)
    assert recomputed['summary'] == summary['summary']
    assert recomputed['gate'] == summary['gate']
    print('PASS: frozen selection, two disjoint evaluation banks, 2,352 scored forwards,')
    print('      native/oracle activation scopes, replacement invariance, restoration and summary gates.')
    for bank, gate in summary['gate'].items():
        print(bank, 'selective-controller gate:', gate['passed'], 'failed:', gate['failed'])


if __name__ == '__main__':
    audit(Path(sys.argv[1]))
