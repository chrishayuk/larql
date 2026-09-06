"""Descriptive scores and the frozen selective-controller gate."""
import numpy as np

POSITIVE = ['installation', 'sentence', 'alias']
NEGATIVE = ['similar_name', 'other_entity', 'other_relation', 'unrelated']


def analyze(rows):
    summaries, gates = {}, {}
    for bank in ['observed_roster', 'fresh_roster']:
        arms = {arm: {r['id']: r for r in rows if r['bank'] == bank and r['arm'] == arm}
                for arm in ['clean', 'existing', 'fitted', 'oracle', 'replaced', 'restored', 'removed']}
        summaries[bank] = {}
        for arm, records in arms.items():
            summaries[bank][arm] = {}
            for group in POSITIVE + NEGATIVE:
                rs = [r for r in records.values() if r['group'] == group]
                summaries[bank][arm][group] = dict(n=len(rs), hits=sum(bool(r['hit']) for r in rs),
                    selected=sum(r['delivery']['selection_correct'] for r in rs),
                    delivered=sum(r['delivery']['delivery_ok'] for r in rs),
                    clean_agreement=sum(r['clean_agree'] for r in rs),
                    mean_margin=float(np.mean([r['margin_full'] for r in rs])) if group in POSITIVE else None,
                    mean_tv=float(np.mean([r['total_variation'] for r in rs])))
        checks = {}
        fit = arms['fitted']
        s = summaries[bank]['fitted']
        for g in POSITIVE:
            checks[g + '/selection'] = s[g]['selected'] / s[g]['n'] >= .9
            checks[g + '/delivery'] = s[g]['delivered'] / s[g]['n'] >= .8
        comparisons = {}
        for g in ['sentence', 'alias']:
            ids = [k for k, r in fit.items() if r['group'] == g]
            supported = [k for k in ids if arms['oracle'][k]['hit']]
            rec = sum(fit[k]['hit'] for k in supported)
            lost = sum(arms['existing'][k]['hit'] and not fit[k]['hit'] for k in ids)
            comparisons[g] = dict(oracle_correct=len(supported), fitted_on_oracle_correct=rec,
                                  existing_successes_lost=lost)
            checks[g + '/oracle_recovery'] = bool(supported) and rec / len(supported) >= .9
            checks[g + '/existing_preserved'] = lost <= 1
        for g in NEGATIVE:
            checks[g + '/quiet'] = s[g]['delivered'] / s[g]['n'] >= .95
            checks[g + '/clean_agreement'] = s[g]['clean_agreement'] / s[g]['n'] >= .95
        replacement = list(arms['replaced'].values())
        target = [r for r in replacement if r['fact'] == 0]
        others = [r for r in replacement if r['fact'] is not None and r['fact'] != 0]
        negatives = [r for r in replacement if r['fact'] is None]
        swap = dict(target_hits=sum(r['hit'] for r in target), target_n=len(target),
                    others_preserved=sum(r['top1'] == fit[r['id']]['top1'] for r in others), others_n=len(others),
                    negatives_preserved=sum(r['top1'] == fit[r['id']]['top1'] for r in negatives), negatives_n=len(negatives))
        checks['swap/target'] = swap['target_hits'] >= 6
        checks['swap/others'] = swap['others_preserved'] / len(others) >= .95
        checks['swap/negatives'] = swap['negatives_preserved'] / len(negatives) >= .95
        checks['swap/counter_controls'] = all(r['top1'] == fit[r['id']]['top1'] for r in replacement
                                             if r['group'] == 'installation' and r['fact'] in [1, 3])
        checks['swap/detector_exact'] = all(r['detector_exact'] for r in replacement)
        checks['restore/exact'] = all(r['reference_max_logit_delta'] == 0 for r in arms['restored'].values())
        checks['remove/exact'] = all(r['max_logit_delta'] == 0 for r in arms['removed'].values())
        gates[bank] = dict(passed=all(checks.values()), checks=checks, comparisons=comparisons,
                           replacement=swap, failed=[k for k, v in checks.items() if not v])
    return dict(summary=summaries, gate=gates)
