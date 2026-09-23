#!/usr/bin/env python3
"""Render the complete sealed V2 surface without selecting a cell."""
import argparse
import json
from pathlib import Path
from gwv2_population import canonical_hash, sha256


def number(value):
    return 'undefined' if value is None else f'{value:.5f}'


def pair(values):
    return 'undefined' if values is None else ' / '.join(number(v) for v in values)


def render(source, output, cost_path=None):
    data=json.loads(source.read_text())
    if data.get('adjudication_sha256')!=canonical_hash(data,'adjudication_sha256'):
        raise ValueError('adjudication seal mismatch')
    lines=['# GW-V2 — complete frozen causal surface','',
           f"**Causal verdict:** {data['verdict']}.",'',
           'This tests the registered rank-limited predictor family over the registered grid, '
           'with natural surrounding computation retained. It does not test whether all possible '
           'small V constructions fail or establish economical graph execution.','',
           f"Held-out-clearing cells: {sum(data['heldout_cells_clear'])}/35. "
           'A cell clears held out only if it clears independently on validation and test.','',
           f"Early-frontier coordinate: `{json.dumps(data['early_frontier']['coordinate'])}`. "
           'This is descriptive; no predictor is selected.','',
           '## Coverage and controls','',
           'All 666 executions, 74 countries, 222 semantic edges, 35 primary cells and 35 sham cells '
           'are included. Natural/no-op and full-exact/identity parity checks passed bit-for-bit '
           'on every execution. Each intervention fired exactly once.','',
           'The exact positive control remains a separate statistical gate. Values below use '
           '`raw / z-scored` order. Effects are control-adjusted JS changes, not task accuracy.','',
           '| Split | Exact proximal effect | Lower pointwise 95% bound | Exact terminal effect |',
           '|---|---:|---:|---:|']
    for split in ['train','validation','test']:
        r=data['results'][split]
        lines.append(f"| {split} | {pair(r['point'][2][0])} | {pair(r['effect_ci95'][0][2][0])} | {pair(r['point'][2][1])} |")
    unestimable=[s for s in ['validation','test'] if not data['results'][s]['exact_effect_positive']]
    if unestimable:
        lines+=['','**Estimand limitation:** retention is undefined on '+', '.join(unestimable)+
                '. The exact positive-control denominator is not positive on both metrics. '
                'Failure to declare a frontier under this condition is not evidence rejecting '
                'the V construction; the required relative-effect test is unavailable.']
    lines+=['','Intervals resample country identities 10,000 times. Retention and terminal '
            'simultaneous lower bounds each cover their separate 70-estimate raw/z family '
            'within a split. The JSON contains pointwise intervals, effects, relation-level '
            'summaries and all controls.','']
    for split in ['validation','test']:
        lines += [f'## {split.title()} — all 35 cells','',
                  '| Depth | Rank | Retention raw / z | Simultaneous lower raw / z | Terminal effect raw / z | Terminal simultaneous lower raw / z | Clears |',
                  '|---:|---:|---:|---:|---:|---:|:---:|']
        for c in data['results'][split]['cells']:
            lines.append(f"| {c['depth']} | {c['rank']} | {pair(c['retention'])} | {pair(c['retention_simultaneous_lower95'])} | {pair(c['effect'][1])} | {pair(c['terminal_effect_simultaneous_lower95'])} | {'yes' if c['gate_pass'] else 'no'} |")
        lines.append('')
    lines+=['## Partially shuffled fit sham — diagnostic only','',
            'The sealed map shuffles 42/44 training identities and 522/603 training token rows. '
            'KNA and STP account for the 81 unchanged token rows. This is not complete subject '
            'destruction. Training predictions leave the recipient subject out; sham fits also '
            'omit training targets donated by that subject, without rewiring the map.','',
            'The shuffled-only column includes the 42 shuffled training recipients and retains '
            'their original matched controls. Held-out rows have no training-donor assignment, '
            'so there is no invented shuffled-only held-out subset. None of these diagnostics '
            'affects a threshold, frontier or primary decision.','',
            '| Depth | Rank | Train retention raw / z | Shuffled train only raw / z | Validation raw / z | Test raw / z |',
            '|---:|---:|---:|---:|---:|---:|']
    for i,c in enumerate(data['results']['train']['sham_cells']):
        values=[pair(data['results'][s]['sham_cells'][i]['retention']) for s in ['train','train_shuffled_only','validation','test']]
        lines.append(f"| {c['depth']} | {c['rank']} | "+' | '.join(values)+' |')
    lines+=['','## Physical-accounting boundary','',
            'No efficiency claim is made. Natural Q, natural K-row norms, the carrier and seven '
            'other heads remain required dynamic inputs. Actual avoided canonical work in this '
            'isolation experiment is zero. Reconstruction MSE/cosine are diagnostic only.','',
            'Registered latency reporting is incomplete until an exclusive peer-acknowledged '
            'window supports warmed baseline/candidate/baseline measurements. Operational '
            'elapsed times are not substituted for those measurements.','']
    if cost_path:
        cost=json.loads(cost_path.read_text())
        if cost.get('cost_sha256')!=canonical_hash(cost,'cost_sha256') or cost['lineage']!=data['lineage']:
            raise ValueError('cost authority mismatch')
        lines+=['The cost artifact records the source boundary, prefix residency, predictor '
                'parameters, bounded matvec/attention FLOPs and logical dynamic-state bytes '
                'for every cell and execution. Dynamic predictor-vector accounting excludes '
                'the cached parameters, which are reported separately. These are not measured '
                'DRAM traffic or latency.','',
                f'Cost artifact: `{cost_path}` (file SHA-256 `{sha256(cost_path)}`).','']
        lines+=['### Registered cost-normalized display — diagnostic only','',
                'Cost fraction is the summed constructor plus predictor matvec/attention FLOP '
                'counter divided by the summed canonical L0–23 counter for that split. '
                'The display divides min(raw retention, z retention) by this fraction. '
                'It excludes the named scalar operations from both counters, uses no latency, '
                'and has no gate, ordering or selection role. Cells remain in frozen grid order.','',
                '| Depth | Rank | Cached bytes | Validation cost fraction | Validation display | Test cost fraction | Test display |',
                '|---:|---:|---:|---:|---:|---:|---:|']
        for c in cost['cells']:
            columns=[]
            for split in ['validation','test']:
                rows=[r for r in c['per_execution'] if r['split']==split]
                fraction=sum(r['constructor_matvec_and_attention_flops']+r['predictor_flops'] for r in rows)/sum(r['canonical_L0_L23_matvec_and_attention_flops'] for r in rows)
                retention=data['results'][split]['cells'][c['cell']]['retention']
                columns += [number(fraction),number(min(retention)/fraction if retention is not None else None)]
            lines.append(f"| {c['depth']} | {c['rank']} | {c['cached_parameter_bytes']} | "+' | '.join(columns)+' |')
        lines.append('')
    lines+=['## Immutable lineage','']
    lines += [f'- {key}: `{value}`' for key,value in data['lineage'].items()]
    lines+=['',f'Full adjudication: `{source}`.',
            f"Canonical adjudication identity: `{data['adjudication_sha256']}`.",
            f"Analysis source identity: `{data['analysis_code_sha256']}`.",'',
            'The original registration was not rewritten. AMEND-1 resolved the token-level '
            'sham correspondence during outcome-free preflight; its donor map and diagnostic '
            'reporting were sealed before capture, fitting or replay.','']
    with output.open('x') as handle:
        handle.write('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('adjudication',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--cost',type=Path)
    a=p.parse_args();render(a.adjudication,a.output,a.cost)
