"""Exact descriptive contrasts; no independence or error-distribution assumptions."""
from collections import Counter

import numpy as np

CODES = [f'{i:03b}' for i in range(8)]
GROUPS = ['all', 'installation', 'sentence', 'alias']


def analyze(rows):
    cells = {code: {r['id']: r for r in rows if r['phase'] == 'factorial' and r['code'] == code}
             for code in CODES}
    assert len({tuple(v.keys()) for v in cells.values()}) == 1
    by_arm, contrasts, conditional = {}, {}, {}
    for group in GROUPS:
        ids = [pid for pid, r in cells['000'].items() if group == 'all' or r['group'] == group]
        by_arm[group] = {}
        for code in CODES:
            rs = [cells[code][pid] for pid in ids]
            by_arm[group][code] = dict(n=len(ids), hits=sum(r['hit'] for r in rs),
                recovered=sum(r['hit'] and not cells['000'][r['id']]['hit'] for r in rs),
                lost=sum(not r['hit'] and cells['000'][r['id']]['hit'] for r in rs),
                mean_margin=float(np.mean([r['margin_full'] for r in rs])))
        contrasts[group] = {}
        for subset in range(1, 8):
            values = {metric: [] for metric in ['margin_full', 'hit']}
            for pid in ids:
                for metric in values:
                    total = sum((-1) ** (subset.bit_count() - part.bit_count()) *
                        float(cells[f'{part:03b}'][pid][metric]) for part in range(8) if part & ~subset == 0)
                    values[metric].append(total)
            contrasts[group][f'{subset:03b}'] = {m: dict(mean=float(np.mean(v)), per_prompt=dict(zip(ids, v)))
                                                 for m, v in values.items()}
        conditional[group] = {}
        for index, factor in enumerate(['E', 'C', 'A']):
            edges = {}
            for off in CODES:
                if off[index] != '0':
                    continue
                on = off[:index] + '1' + off[index+1:]
                edges[off + '->' + on] = dict(
                    hit_delta=sum(int(cells[on][p]['hit']) - int(cells[off][p]['hit']) for p in ids),
                    recovered=sum(cells[on][p]['hit'] and not cells[off][p]['hit'] for p in ids),
                    lost=sum(not cells[on][p]['hit'] and cells[off][p]['hit'] for p in ids),
                    mean_margin_delta=float(np.mean([cells[on][p]['margin_full'] - cells[off][p]['margin_full']
                                                     for p in ids])))
            conditional[group][factor] = edges
    minimal = []
    for pid in cells['000']:
        if cells['000'][pid]['hit'] or not cells['111'][pid]['hit']:
            continue
        sufficient = [i for i in range(1, 8) if cells[f'{i:03b}'][pid]['hit']]
        smallest = [i for i in sufficient if not any(j != i and j & i == j for j in sufficient)]
        minimal.append(dict(id=pid, group=cells['000'][pid]['group'],
                            minimal_subsets=[f'{i:03b}' for i in smallest]))
    counts = dict(Counter(','.join(row['minimal_subsets']) for row in minimal))
    return dict(arms=by_arm, contrasts=contrasts, conditional=conditional,
                minimal_sufficient=minimum_order(minimal), minimal_pattern_counts=counts)


def minimum_order(rows):
    return sorted(rows, key=lambda r: r['id'])
