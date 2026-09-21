"""Authored synthetic UI stories. Never an executor or scientific witness."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'public' / 'fixtures'
ROOT.mkdir(parents=True, exist_ok=True)


def norm(x):
    return math.sqrt(sum(v * v for v in x))


def build(key, title, story, prompt, answer, rich=False, address=False, target=False):
    words = prompt.split()
    matrices = [
        [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]],
        [[0.6, 0.3, 0, 0.2, 0, 0], [0, 0.4, 0.8, 0, 0.2, 0], [0, 0, 0.4, 0.1, 0, 0.8]],
    ]
    bases = []
    for i, matrix in enumerate(matrices):
        axes = ['relation (authored)', 'entity (authored)', 'binding (authored)'] if address and i == 0 else ['axis 1', 'axis 2', 'axis 3']
        bases.append(dict(id=f'fixture-{"address" if address else "carrier"}-{i}', hash='sha256:' + hashlib.sha256(json.dumps(dict(matrix=matrix, axes=axes), sort_keys=True).encode()).hexdigest(), label=('ADDRESS study' if address else 'Carrier study') if i == 0 else 'Oblique study', dimensions=3, axes=axes, source='synthetic-six-dimensional-carrier-v1'))
    answers = [answer, 'country', 'city']
    readers = {answer: [2.2, 1.8, .3, 0, 0, 0], 'country': [-.7, .2, .1, 0, 0, 0], 'city': [0, .5, -.4, 0, 0, 0]}
    events = []

    def emit(kind, time, **kwargs):
        events.append(dict(run_id=key, sequence=str(len(events)), timestamp_ns=str(time), kind=kind, **kwargs))

    emit('RunStarted', 0)
    emit('Tokenized', 10000000)
    vectors = [[-1.7 + p * .035, -1.15 + p * .06, -.45, .1, .2, .15] for p in range(len(words))]
    for layer in range(-1, 32):
        roles = ['embedding'] if layer == -1 else ['attention_write', 'ffn_write']
        for stage, role in enumerate(roles):
            for p, word in enumerate(words):
                before = vectors[p][:]
                if layer >= 0:
                    weight = .035 + .012 * math.sin(layer * .61 + p * .3)
                    bump = math.exp(-((layer - (23 if stage == 0 else 25)) / 1.4) ** 2)
                    weight += bump * (.36 if stage == 0 else .31)
                    weight *= .4 + .6 * p / max(len(words) - 1, 1)
                    direction = [1.0, .3 + .8 * math.sin(layer * .15 + .4), .45 * math.cos(layer * .18), .3, -.18, .1]
                    if address:
                        direction = [1.3 if layer < 12 else .12, .1 if layer < 19 else 1.1, .08 if layer < 22 else 1.2, .2, .1, .1]
                    if target and layer >= 20:
                        direction[1] *= -1.2
                        direction[2] += 1.4
                    delta = [weight * v for v in direction]
                    vectors[p] = [a + b for a, b in zip(before, delta)]
                else:
                    delta = [0.0] * 6
                after = vectors[p]
                projection = {b['id']: [round(sum(x * y for x, y in zip(row, after)), 7) for row in matrix] for b, matrix in zip(bases, matrices)}
                logits = {name: round(sum(x * y for x, y in zip(row, after)), 7) for name, row in readers.items()}
                write_logits = {name: round(sum(x * y for x, y in zip(row, delta)), 7) for name, row in readers.items()}
                sample = dict(position=p, layer=layer, role=role, topology='residual', carrier='main', before=round(norm(before), 7), norm=round(norm(after), 7), delta=round(norm(delta), 7), projections=projection, logits=logits, write_logits=write_logits, duration_ns=str(180000 + layer * 4000 + p * 1000))
                if rich and role == 'attention_write':
                    scores = [(1 + (8 if j == max(0, p - 1) else 0) + (4 if j == 1 else 0)) for j in range(p + 1)]
                    sample['sources'] = [dict(position=j, weight=round(v / sum(scores), 8)) for j, v in enumerate(scores)]
                time = 60000000 + (layer + 1) * 160000000 + stage * 65000000 + p * 1000000
                emit('Observation' if layer == -1 else 'CarrierWrite', time, sample=sample)
    emit('TokenProduced', 5400000000, output=answer)
    emit('RunCompleted', 5450000000)
    record = dict(schema='larql.observatory.fixture.v1', id=key, title=title, story=story, prompt=prompt, tokens=words, provenance='synthetic', capture='rich-fixture' if rich else 'standard', model='Authored 32-layer fixture', layers=32, program=[dict(layer=l, roles=['attention_write', 'ffn_write']) for l in range(32)], bases=bases, answers=answers, attention='synthetic-head-mean' if rich else 'unavailable', events=events)
    record['readout'] = dict(id='fixture-linear-reader-v1', hash='sha256:' + hashlib.sha256(json.dumps(readers,sort_keys=True).encode()).hexdigest(), method='fixed-linear-rows-no-normalization', rows=readers)
    (ROOT / f'{key}.json').write_text(json.dumps(record, separators=(',', ':')) + '\n')


build('answer', 'An answer takes shape', 'A / ANSWER FORMATION', 'The capital of France is', 'Paris')
build('writes', 'A change of hands', 'B / ATTENTION + FFN', 'The capital of France is', 'Paris', rich=True)
build('address', 'An address resolves', 'C / AUTHORED ADDRESS', 'KA red MU', 'MU', address=True)
build('counterfactual', 'Where paths divide', 'D / COUNTERFACTUAL', 'KA red MU', 'MU')
build('counterfactual-target', 'Where paths divide · target', 'D / TARGET', 'KA red RO', 'RO', target=True)
