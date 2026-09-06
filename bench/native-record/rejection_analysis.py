"""Thresholds are read-only diagnostics, never native execution controls."""
import numpy as np
from controller_fit import delivery, pattern_loss, targets


def counts(ratio, rows, threshold):
    peak = np.max(ratio, axis=1)
    winner = np.argmax(ratio, axis=1)
    accept = peak >= threshold
    positive = np.array([r['fact'] is not None for r in rows])
    correct = np.array([r['fact'] == int(w) for r, w in zip(rows, winner)])
    groups = sorted({r['group'] for r in rows if r['fact'] is None})
    negative = {}
    for group in groups:
        indices = np.array([r['fact'] is None and r['group'] == group for r in rows])
        negative[group] = dict(n=int(indices.sum()), false_accept=int(np.sum(accept & indices)))
    return dict(threshold=float(threshold), positive_n=int(positive.sum()),
                positive_accepted=int(np.sum(accept & positive)),
                positive_correct=int(np.sum(accept & positive & correct)),
                negative=negative, negative_gate=all(v['false_accept'] / v['n'] <= .05 for v in negative.values()))


def curve(ratio, rows):
    scores = np.unique(np.max(ratio, axis=1).astype(np.float64))
    thresholds = [float(scores[0] - 1), *scores.tolist(), float(np.nextafter(scores[-1], np.inf))]
    return [counts(ratio, rows, t) for t in thresholds]


def choose(curve_rows):
    allowed = [r for r in curve_rows if r['negative_gate']]
    return max(allowed, key=lambda r: (r['positive_correct'],
        -sum(v['false_accept'] for v in r['negative'].values()), r['threshold']))


def quantiles(values):
    return dict(zip(['min', 'q25', 'median', 'q75', 'max'], np.quantile(values, [0, .25, .5, .75, 1]).tolist()))


def relation(row):
    for rel in ['capital', 'currency', 'language']:
        if row['prompt'].startswith('The ' + rel + ' of '):
            return rel
    return 'other'


def summarize(ratio, gate, rows):
    result = {}
    for split in ['train', 'validation']:
        indices = [i for i, r in enumerate(rows) if r['split'] == split]
        rr = ratio[indices]
        selected = [rows[i] for i in indices]
        positive = [i for i, r in enumerate(selected) if r['fact'] is not None]
        negative = [i for i, r in enumerate(selected) if r['fact'] is None]
        groups = {}
        for group in sorted({r['group'] for r in selected}):
            ix = [i for i, r in enumerate(selected) if r['group'] == group]
            metrics = [delivery(rr[i], np.ones(12), selected[i]['fact']) for i in ix]
            groups[group] = dict(n=len(ix), selected=sum(m['selection_correct'] for m in metrics),
                                 delivered=sum(m['delivery_ok'] for m in metrics))
        intended = [rr[i, selected[i]['fact']] for i in positive]
        competitor = [np.max(np.abs(np.delete(rr[i], selected[i]['fact']))) for i in positive]
        negatives = {}
        for group in sorted({selected[i]['group'] for i in negative}):
            ix = [i for i in negative if selected[i]['group'] == group]
            negatives[group] = dict(n=len(ix), peak=quantiles(np.max(rr[ix], axis=1)),
                absolute_peak=quantiles(np.max(np.abs(rr[ix]), axis=1)),
                relations={rel: sum(relation(selected[i]) == rel for i in ix)
                           for rel in ['capital', 'currency', 'language', 'other']})
        result[split] = dict(n=len(indices), positive_n=len(positive), negative_n=len(negative), groups=groups,
            loss=pattern_loss(rr, targets(selected, np.ones(12))),
            positive_delivery=sum(delivery(rr[i], np.ones(12), selected[i]['fact'])['delivery_ok'] for i in positive),
            negative_quiet=sum(delivery(rr[i], np.ones(12), None)['delivery_ok'] for i in negative),
            suppressed_positive=sum(float(gate[indices[i], selected[i]['fact']]) <= -8 for i in positive),
            intended=quantiles(intended), competitor_absolute=quantiles(competitor),
            positive_peak=quantiles(np.max(rr[positive], axis=1)), negatives=negatives)
    tr = [i for i, r in enumerate(rows) if r['split'] == 'train']
    va = [i for i, r in enumerate(rows) if r['split'] == 'validation']
    training_curve = curve(ratio[tr], [rows[i] for i in tr])
    validation_curve = curve(ratio[va], [rows[i] for i in va])
    best = choose(training_curve)
    result['threshold'] = dict(training_selected=best,
        validation_at_training_threshold=counts(ratio[va], [rows[i] for i in va], best['threshold']),
        validation_optimistic=choose(validation_curve), training_curve=training_curve, validation_curve=validation_curve)
    return result


def native_gate(candidate, baseline):
    v, b = candidate['validation'], baseline['validation']
    checks = {}
    for group in ['sentence', 'alias']:
        c = v['groups'][group]
        checks[group + '/selection'] = c['selected'] / c['n'] >= .9
        checks[group + '/delivery'] = c['delivered'] / c['n'] >= .8
    for group in v['negatives']:
        c = v['groups'][group]
        checks[group + '/quiet'] = c['delivered'] / c['n'] >= .95
    checks['positive_nonregression'] = v['positive_delivery'] >= b['positive_delivery']
    checks['negative_nonregression'] = v['negative_quiet'] >= b['negative_quiet']
    checks['joint_progress'] = (v['positive_delivery'] > b['positive_delivery'] or
                                v['negative_quiet'] > b['negative_quiet'])
    return dict(passed=all(checks.values()), checks=checks, failed=[k for k, v in checks.items() if not v])
