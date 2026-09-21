import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { parseRecording, reduceEvents, semanticKey } from '../lib/record.ts';
import { adaptStandard } from '../lib/standard.ts';
import { parseLenses, availableLenses, sourceDigest, distributionDelta, specificity, headKey } from '../lib/lenses.ts';
const raw = readFileSync(new URL('../public/fixtures/answer.json', import.meta.url), 'utf8');
const record = parseRecording(JSON.parse(raw));
const analysis = JSON.parse(readFileSync(new URL('../public/fixtures/answer.lenses.json', import.meta.url)));
const hash = createHash('sha256').update(raw).digest('hex');
const samples = reduceEvents(record.events).samples;

test('all story sidecars bind to exact source bytes and remain synthetic', async () => {
  for (const name of ['answer','writes','address','counterfactual','counterfactual-target']) {
    const bytes = readFileSync(new URL(`../public/fixtures/${name}.json`, import.meta.url), 'utf8');
    const r = parseRecording(JSON.parse(bytes));
    const a = JSON.parse(readFileSync(new URL(`../public/fixtures/${name}.lenses.json`, import.meta.url)));
    const parsed = parseLenses(a, r, await sourceDigest(bytes));
    assert.equal(parsed.provenance, 'synthetic');
    assert.equal(parsed.vocabulary.rows.length, reduceEvents(r.events).samples.length);
    assert.deepEqual(parseLenses(JSON.parse(JSON.stringify(parsed)), r, await sourceDigest(bytes)), parsed);
  }
  assert.notEqual(await sourceDigest(raw), await sourceDigest(raw.trim()));
});

test('analysis cannot attach to another run, changed source bytes or upgrade evidence', () => {
  for (const change of [r => r.run_id = 'other', r => r.source_sha256 = 'a'.repeat(64), r => r.provenance = 'executor']) {
    const bad = structuredClone(analysis); change(bad); assert.throws(() => parseLenses(bad, record, hash), /refused/);
  }
  const source = JSON.parse(readFileSync(new URL('../public/recordings/granite-standard.json', import.meta.url)));
  const measured = adaptStandard(source);
  const forged = structuredClone(analysis); forged.run_id = measured.id;
  assert.throws(() => parseLenses(forged, measured, hash), /provenance/);
  assert.equal(availableLenses(undefined, reduceEvents(measured.events).samples), undefined);
});

test('replay hides future site analyses and later cache accounting', () => {
  const partial = reduceEvents(record.events.slice(0, 50));
  const visible = availableLenses(analysis, partial.samples, partial.events.at(-1).sequence);
  const keys = new Set(partial.samples.map(semanticKey));
  assert.ok(visible.vocabulary.rows.every(r => keys.has(r.site)));
  assert.ok(visible.heads.rows.every(r => keys.has(r.site)));
  assert.equal(visible.experiments.length, 0);
  assert.equal(visible.kv, undefined);
  assert.equal(availableLenses(analysis, samples, record.events.at(-1).sequence).kv.total_bytes, analysis.kv.total_bytes);
  assert.equal(availableLenses(analysis, [], '0').vocabulary.rows.length, 0);
});

test('head selection resolves exactly to the same observed site and recorded experiment', () => {
  const head = analysis.heads.rows.find(h => h.site === '4:23:attention_write:main:residual' && h.head === 3);
  const sample = samples.find(s => semanticKey(s) === head.site);
  assert.equal(sample.position, 4);
  assert.equal(sample.layer, 23);
  assert.equal(sample.role, 'attention_write');
  assert.equal(headKey(head), `${semanticKey(sample)}:3:Paris`);
  const e = analysis.experiments.find(e => e.site === head.site && e.head === head.head);
  assert.equal(e.parent, record.id);
  assert.notEqual(e.fork, record.id);
});

test('refuse malformed scientific quantities, duplicate sites and unsupported causal evidence', () => {
  const mutations = [
    a => a.vocabulary.rows.push(a.vocabulary.rows[0]),
    a => a.vocabulary.rows[0].entropy = -1,
    a => a.vocabulary.rows[0].entropy = Math.log(6)+1,
    a => a.vocabulary.rows[0].top[0].probability = 2,
    a => a.vocabulary.rows[0].top[0].rank = 2,
    a => a.vocabulary.rows[0].targets[0].probability += .1,
    a => a.heads.rows[0].site = '0:999:attention_write:main:residual',
    a => a.heads.rows.push(a.heads.rows[0]),
    a => a.heads.rows[0].head = -1,
    a => a.heads.rows[0].sources[0].position = 1,
    a => a.heads.rows[0].sources[0].weight = 2,
    a => a.heads.rows[0].content.dimensionality.d99 = 0,
    a => a.heads.rows[0].content.orthogonality.pairs[0].cosine = 1.1,
    a => a.kv.key_bytes += 1,
    a => a.kv.content.witness = '',
    a => a.experiments[0].fork = record.id,
    a => a.experiments[0].kl.scope = 'top-k',
    a => a.experiments[0].kl.nats = -.1,
    a => a.spans[0].end = 999,
  ];
  for (const mutate of mutations) {
    const bad = structuredClone(analysis); mutate(bad);
    assert.throws(() => parseLenses(bad, record, hash), /refused/);
  }
});

test('partial top-k and attention mass are retained without inventing zeros or normalizing', () => {
  const a = structuredClone(analysis);
  a.heads.rows[0].sources[0].weight = .2;
  parseLenses(a, record, hash);
  assert.equal(a.heads.rows[0].sources[0].weight, .2);
  const before = analysis.experiments[0].before.slice(0,2), after = analysis.experiments[0].after.slice(0,1);
  const delta = distributionDelta(before, after);
  for (const d of delta) {
    if (!before.some(p => p.token_id === d.id) || !after.some(p => p.token_id === d.id)) assert.equal(d.delta, undefined);
    else assert.equal(d.delta, d.after - d.before);
  }
});

test('synthetic distributions, dimensionality and KL match the explicitly declared toy calculations', () => {
  const h = analysis.heads.rows.find(h => h.head === 3);
  const energy = h.content.projections.map(p => p.coefficient ** 2).sort((a,b) => b-a);
  for (const [field, cutoff] of [['d95',.95], ['d99',.99], ['d999',.999]]) {
    const d = h.content.dimensionality[field];
    const sum = energy.reduce((a,b) => a+b,0);
    assert.ok(energy.slice(0,d).reduce((a,b) => a+b,0)/sum >= cutoff);
    assert.ok(energy.slice(0,d-1).reduce((a,b) => a+b,0)/sum < cutoff);
  }
  for (const e of analysis.experiments) {
    const kl = e.before.reduce((sum,p) => sum+p.probability*Math.log(p.probability/e.after.find(q=>q.token_id===p.token_id).probability),0);
    assert.ok(Math.abs(kl-e.kl.nats)<1e-12);
    assert.ok(Math.abs(e.before.reduce((sum,p)=>sum+p.probability,0)-1)<1e-12);
  }
  assert.equal(specificity(Math.log(6),6),0);
  assert.equal(specificity(0,6),1);
});

test('cross-run readout deltas require explicit matching method and identity', async () => {
  const { compatibleReadout } = await import('../lib/lenses.ts');
  const other = structuredClone(record);
  assert.equal(compatibleReadout(record,other), true);
  other.readout.hash = 'sha256:'+'a'.repeat(64);
  assert.equal(compatibleReadout(record,other), false);
  delete other.readout;
  assert.equal(compatibleReadout(record,other), false);
  const raw = JSON.parse(readFileSync(new URL('../public/recordings/granite-standard.json', import.meta.url)));
  const a = adaptStandard(raw), b = structuredClone(a);
  assert.equal(compatibleReadout(a,b),true);
  b.standard.identity.container = 'different';
  assert.equal(compatibleReadout(a,b),false);
});
