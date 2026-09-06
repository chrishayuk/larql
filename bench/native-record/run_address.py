#!/usr/bin/env python3
"""Frozen-recipe native record evaluation. See PROTOCOL.md before running."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
FACTS = [("Zelandia", "capital", "Oslo"), ("Qtaria", "currency", "Yen"),
         ("Vornholt", "language", "Welsh")]
REPLACEMENTS = ["Paris", "Euro", "French"]
TEMPLATES = {r: f"The {r} of {{e}} is" for _, r, _ in FACTS}
RETAIN = [("The capital of France is", "Paris"), ("The capital of Japan is", "Tokyo"),
          ("The capital of Italy is", "Rome"), ("The capital of Germany is", "Berlin"),
          ("Two plus two equals", "four"), ("The opposite of hot is", "cold")]
GENERIC = ["The story begins on a", "In the morning the sun", "She opened the door and",
           "Water is a clear", "Once upon a time there", "The weather today is"]
DECOYS = GENERIC + [TEMPLATES[r].format(e=e) for r in TEMPLATES
                   for e in ["Brazil", "Egypt", "Canada", "Peru", "Chile", "Kenya", "Sweden", "Mexico"]]
COUNTRIES = ["Finland", "Austria", "Poland", "Hungary", "England", "Greece",
             "Denmark", "Turkey", "Ireland", "Vietnam", "Thailand", "Norway",
             "Portugal", "Spain", "Russia", "Cuba", "Australia", "China",
             "Argentina", "Belgium"]


def fixture():
    rows = []

    def add(group, prompt, fact=None, expected=None):
        rows.append(dict(id=f"{group}-{sum(x['group'] == group for x in rows):03}",
                         group=group, prompt=prompt, fact=fact, expected=expected))

    for i, (entity, relation, answer) in enumerate(FACTS):
        add("installation", TEMPLATES[relation].format(e=entity), i, answer)
        for form in ["{e}'s {r} is", "For {e}, the {r} is",
                     "In {e}, the {r} is", "The {r} associated with {e} is"]:
            add("sentence", form.format(e=entity, r=relation), i, answer)
        alias = ["Zed", "Qtar", "Vorn"][i]
        for form in ["{a} is another name for {e}. The {r} of {a} is",
                     "Here, {e} is also called {a}. {a}'s {r} is"]:
            add("alias", form.format(a=alias, e=entity, r=relation), i, answer)
        for other_r in TEMPLATES:
            if other_r != relation:
                add("different_relation", TEMPLATES[other_r].format(e=entity))
        similar = [["Zelania", "Zerlandia", "Zelandria"],
                   ["Qtoria", "Qteria", "Qtarnia"],
                   ["Vornhult", "Vornhold", "Vornheim"]][i]
        for name in similar:
            add("similar_name", TEMPLATES[relation].format(e=name))
        for name in COUNTRIES:
            add("other_entity", TEMPLATES[relation].format(e=name))
    for prompt in ["The largest planet in the solar system is",
                   "A triangle has", "The chemical symbol for oxygen is",
                   "The author of Hamlet was", "The square root of nine is",
                   "A week contains seven", "The color of a ripe banana is",
                   "Birds build nests to lay", "A thermometer measures",
                   "The first month of the year is", "Ice melts into",
                   "A piano is a musical"]:
        add("unrelated", prompt)
    for prompt, expected in RETAIN:
        add("original_retention", prompt, expected=expected)
    assert not ({r['prompt'] for r in rows} & set(DECOYS))
    assert len({r['prompt'] for r in rows}) == len(rows)
    positives = [r for r in rows if r['fact'] is not None]
    for row in positives:
        assert all(value.casefold() not in row['prompt'].casefold()
                   for value in [*(f[2] for f in FACTS), *REPLACEMENTS])
    return rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def persist(path, data):
    # Each run owns an exclusive new directory; no existing artifact is replaced.
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def score(ll, target, clean, tok):
    ll = np.asarray(ll, np.float64)
    lp = ll - np.logaddexp.reduce(ll)
    cp = np.asarray(clean, np.float64)
    cp = cp - np.logaddexp.reduce(cp)
    top = int(ll.argmax())
    top_ids = np.argsort(ll)[-5:][::-1]
    row = dict(top1=top, top1_text=tok.decode([top]),
               top5=[dict(id=int(t), text=tok.decode([int(t)]), logp=float(lp[t])) for t in top_ids],
               clean_agree=top == int(np.argmax(clean)),
               kl_clean_arm=float(np.sum(np.exp(cp) * (cp - lp))),
               total_variation=float(np.abs(np.exp(cp) - np.exp(lp)).sum() / 2),
               max_logit_delta=float(np.max(np.abs(ll - clean))))
    if target is not None:
        competitor = float(np.max(np.delete(ll, target)))
        row.update(target_id=target, hit=top == target, probability=float(np.exp(lp[target])),
                   margin_full=float(ll[target] - competitor),
                   rank=1 + int(np.count_nonzero(ll > ll[target])))
    return row


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['group']].append(r)
    return {g: dict(n=len(rs), hits=sum(r.get('hit', False) for r in rs),
                    agreement=sum(r['clean_agree'] for r in rs),
                    mean_kl=float(np.mean([r['kl_clean_arm'] for r in rs])),
                    max_tv=max(r['total_variation'] for r in rs)) for g, rs in groups.items()}


def gates(arms):
    checks = {}
    written = summarize(arms['written'])
    checks['original_3_reads'] = written['installation']['hits'] == 3
    checks['original_6_retained'] = written['original_retention']['agreement'] == 6
    for arm in ['written', 'replace_0', 'replace_1', 'replace_2']:
        sums = summarize(arms[arm])
        for group in ['sentence', 'alias']:
            checks[f'{arm}/{group}'] = sums[group]['hits'] / sums[group]['n'] >= .8
        for fact in range(3):
            rs = [r for r in arms[arm] if r['group'] == 'sentence' and r['fact'] == fact]
            checks[f'{arm}/fact_{fact}'] = sum(r['hit'] for r in rs) / len(rs) >= .75
        checks[f'{arm}/installation'] = sums['installation']['hits'] == 3
        for group in ['different_relation', 'similar_name', 'other_entity', 'unrelated', 'original_retention']:
            checks[f'{arm}/{group}'] = sums[group]['agreement'] / sums[group]['n'] >= .95
    for arm in ['clean_repeat', 'removed']:
        checks[arm + '/exact_logits'] = all(r['max_logit_delta'] == 0 for r in arms[arm])
    checks['restored/exact_logits'] = all(r['written_max_logit_delta'] == 0 for r in arms['restored'])
    return dict(advance=all(checks.values()), checks=checks,
                failed=[k for k, value in checks.items() if not value])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True, help='Local unquantized Gemma 3 4B snapshot')
    ap.add_argument('--out', type=Path, required=True, help='New output directory; existing paths refused')
    ap.add_argument('--fixture-only', action='store_true')
    args = ap.parse_args()
    probes = fixture()
    args.out.mkdir(parents=True, exist_ok=False)
    persist(args.out / 'fixture.json', dict(facts=FACTS, replacements=REPLACEMENTS,
                                          decoys=DECOYS, probes=probes))
    print('Frozen fixture:', dict(Counter(r['group'] for r in probes)), flush=True)
    if args.fixture_only:
        return
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import mlx.core as mx
    from chuk_lazarus.models_v2.loader import load_model, ModelDType

    spec = importlib.util.spec_from_file_location('original_native', HERE / 'vendor/native.py')
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    assert native.FACTS == FACTS and native.PROMPT == TEMPLATES and native.RETAIN == RETAIN
    assert native.L == 26 and native.GATE_FIRE == 30. and native.VALUE_WRITE == .1
    model_path = args.model.expanduser().resolve()
    if not model_path.is_dir():
        raise ValueError('A complete local snapshot is required')
    config = json.loads((model_path / 'config.json').read_text())
    text_config = config.get('text_config', config)
    if text_config.get('hidden_size') != 2560 or text_config.get('num_hidden_layers') != 34:
        raise ValueError('Expected original Gemma 3 4B architecture')
    metadata = dict(model_id='google/gemma-3-4b-it', snapshot=str(model_path), dtype='bfloat16',
                    source_commit='3ee95817324e047b5f8f5b587746de2c92051f21',
                    source_sha256=digest(HERE / 'vendor/native.py'),
                    harness_sha256=digest(__file__), protocol_sha256=digest(HERE / 'PROTOCOL.md'),
                    fixture_sha256=digest(args.out / 'fixture.json'),
                    python=sys.version, packages={p: importlib.metadata.version(p)
                        for p in ['mlx', 'numpy', 'transformers', 'huggingface-hub']},
                    checkpoint_files={p.name: dict(bytes=p.stat().st_size, resolved=str(p.resolve()))
                        for p in sorted(model_path.iterdir()) if p.is_file()},
                    loader_commit=subprocess.check_output(['git', '-C',
                        '/Users/christopherhay/chris-source/chuk-mlx', 'rev-parse', 'HEAD'], text=True).strip())
    persist(args.out / 'metadata.json', metadata)
    print('Loading', model_path, flush=True)
    started = time.monotonic()
    lm = load_model(str(model_path), dtype=ModelDType.BFLOAT16)
    model, tok = lm.model, lm.tokenizer
    model.eval()
    mlp = model.model.layers[native.L].mlp
    original = (mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight)
    G, U, D = [np.array(w.astype(mx.float32)) for w in original]
    gref = float(np.median(np.linalg.norm(G, axis=1)))
    uref = float(np.median(np.linalg.norm(U, axis=1)))
    dref = float(np.median(np.linalg.norm(D, axis=0)))
    bos = getattr(tok, 'bos_token_id', -1)

    def tid(value):
        ids = [i for i in tok.encode(' ' + value) if i != bos]
        if len(ids) != 1:
            raise ValueError(f'Expected one answer token for {value!r}, got {ids}')
        return ids[0]

    tids = {v: tid(v) for v in [*(f[2] for f in FACTS), *REPLACEMENTS]}
    persist(args.out / 'tokens.json', tids)
    values = {v: native.unit(np.array(model.model.embed_tokens.weight[t].astype(mx.float32)))
              for v, t in tids.items()}
    captured = {}
    cls, call = type(mlp), type(mlp).__call__

    def capture(self, x):
        if self is mlp:
            captured['x'] = np.array(x[0, -1, :].astype(mx.float32))
        return call(self, x)

    def forward(prompt):
        captured.clear()
        ll = model(mx.array([tok.encode(prompt)])).logits[0, -1]
        mx.eval(ll)
        return np.array(ll.astype(mx.float32)), captured['x'].copy()

    def assign(weights):
        mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight = weights

    def converted(matrices):
        return tuple(mx.array(a.astype(np.float32)).astype(w.dtype) for a, w in zip(matrices, original))

    cls.__call__ = capture
    arms, base, written_logits = {}, {}, {}
    try:
        addresses = [forward(TEMPLATES[r].format(e=e))[1] for e, r, _ in FACTS]
        decoy_addresses = [forward(p)[1] for p in DECOYS]
        keys, slots = [], []
        for i, (_, _, answer) in enumerate(FACTS):
            kr = native.unique_part(addresses[i], decoy_addresses +
                                    [a for j, a in enumerate(addresses) if j != i])
            key, slot = native.unit(kr), G.shape[0] - 1 - i
            keys.append(key)
            slots.append(slot)
            G[slot], U[slot] = key * gref * native.GATE_FIRE, key * uref
            D[:, slot] = values[answer] * dref * native.VALUE_WRITE
        written = converted((G, U, D))
        np.savez(args.out / 'edit_slots.npz', keys=np.stack(keys), slots=slots,
                 gate=G[slots], up=U[slots], down=D[:, slots],
                 original_gate=np.array(original[0][mx.array(slots)].astype(mx.float32)),
                 original_up=np.array(original[1][mx.array(slots)].astype(mx.float32)),
                 original_down=np.array(original[2][:, mx.array(slots)].astype(mx.float32)))
        states = ['clean', 'clean_repeat', 'written', 'replace_0', 'replace_1', 'replace_2', 'removed', 'restored']
        with (args.out / 'rows.jsonl').open('x') as stream:
            for arm in states:
                replaced = int(arm[-1]) if arm.startswith('replace_') else None
                if arm in ['clean', 'clean_repeat', 'removed']:
                    assign(original)
                elif replaced is not None:
                    new_d = D.copy()
                    new_d[:, slots[replaced]] = values[REPLACEMENTS[replaced]] * dref * native.VALUE_WRITE
                    assign((written[0], written[1], converted((G, U, new_d))[2]))
                else:
                    assign(written)
                arms[arm] = []
                for index, probe in enumerate(probes):
                    ll, address = forward(probe['prompt'])
                    if arm == 'clean':
                        base[probe['id']] = ll
                    if arm == 'written':
                        written_logits[probe['id']] = ll
                    expected = probe['expected']
                    if replaced is not None and probe['fact'] == replaced:
                        expected = REPLACEMENTS[replaced]
                    target = tids[expected] if probe['fact'] is not None else None
                    row = dict(probe, arm=arm, expected=expected,
                               **score(ll, target, base[probe['id']], tok))
                    row['key_cosines'] = [float(key @ native.unit(address)) for key in keys]
                    row['slot_gate'] = np.array((mlp.gate_proj.weight[mx.array(slots)].astype(mx.float32)
                                                @ mx.array(address))).tolist()
                    row['slot_up'] = np.array((mlp.up_proj.weight[mx.array(slots)].astype(mx.float32)
                                              @ mx.array(address))).tolist()
                    if arm == 'restored':
                        row['written_max_logit_delta'] = float(np.max(np.abs(ll - written_logits[probe['id']])))
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    stream.flush()
                    arms[arm].append(row)
                    if (index + 1) % 30 == 0:
                        print(f'{arm}: {index+1}/{len(probes)}', flush=True)
                summary = summarize(arms[arm])
                print(arm, json.dumps(summary), flush=True)
                persist(args.out / 'summary.partial.json', {k: summarize(v) for k, v in arms.items()})
        result = dict(status='complete', elapsed_seconds=time.monotonic() - started,
                      summary={k: summarize(v) for k, v in arms.items()}, gate=gates(arms))
        persist(args.out / 'summary.json', result)
        print('GATE', json.dumps(result['gate']), flush=True)
    finally:
        assign(original)
        cls.__call__ = call


if __name__ == '__main__':
    main()
