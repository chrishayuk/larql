import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { adaptStandard } from '../lib/standard.ts';
import { auditStandard } from '../lib/replay-audit.ts';
import { reduceEvents } from '../lib/record.ts';
import { contextRows } from '../lib/context.ts';

const bytes = readFileSync(new URL('../public/recordings/granite-standard.json', import.meta.url));
const source = JSON.parse(bytes);
const record = adaptStandard(source);
const samples = reduceEvents(record.events).samples;

test('real Granite record retains 400 writes, exact replay and separately bound parity evidence', () => {
  const audit = auditStandard(source);
  assert.equal(audit.provenance, 'executor');
  assert.equal(audit.coverage, 'complete');
  assert.equal(audit.writes_checked, 400);
  assert.equal(audit.events_checked, 804);
  assert.equal(audit.source_values, 'exact');
  assert.equal(reduceEvents(record.events).output, ' Paris');
  const witness = JSON.parse(readFileSync(new URL('../public/recordings/granite-standard.parity.json', import.meta.url)));
  assert.equal(witness.source_sha256, createHash('sha256').update(bytes).digest('hex'));
  assert.equal(witness.run_id, record.id);
  assert.equal(witness.observed_sha256, witness.unobserved_sha256);
  assert.equal(witness.positions, 5);
  assert.equal(witness.result, 'pass');
  // This validates the persisted report's binding, not a new inference run.
  assert.equal(audit.execution_parity, 'not_checked');
});

test('real capture retains run-wide realizations and refuses conflicting observer provenance', () => {
  assert.deepEqual(record.execution, source.run_provenance.execution);
  for (const change of [r => r.run_provenance.basis.hash_hex = 'f'.repeat(64),
    r => r.run_provenance.probe_tokens.reverse(), r => r.run_provenance.norm_method = 'invented']) {
    const bad = structuredClone(source); change(bad);
    assert.throws(() => adaptStandard(bad), /provenance disagrees/);
  }
});

test('Context scales are fixed across depth and missing evidence is not rendered as zero', () => {
  const a = contextRows(samples, samples, record.tokens, 0, 'attention_write', 'norm', record.answers[0]);
  const b = contextRows(samples, samples, record.tokens, 39, 'attention_write', 'norm', record.answers[0]);
  assert.equal(a.maximum, b.maximum);
  assert.equal(a.rows.length, 5);
  for (const row of a.rows) assert.equal(row.value, row.sample.norm);
  const partial = contextRows(samples.filter(s => s.position !== 3), samples, record.tokens, 0, 'attention_write', 'norm', record.answers[0]);
  assert.equal(partial.rows[3].value, undefined);
  assert.equal(partial.rows[3].intensity, undefined);
  assert.equal(partial.maximum, a.maximum);
  const probes = contextRows(samples, samples, record.tokens, 39, 'ffn_write', 'probe', record.answers[0]);
  for (const row of probes.rows) assert.equal(row.value, row.sample.logits[record.answers[0]]);
});

test('real Granite vocabulary lens covers every write and binds the final readout witness to canonical logits', async () => {
  const { parseLenses } = await import('../lib/lenses.ts');
  const raw = JSON.parse(readFileSync(new URL('../public/recordings/granite-standard.lenses.json', import.meta.url)));
  const lens = parseLenses(raw, record, createHash('sha256').update(bytes).digest('hex'));
  assert.equal(lens.provenance, 'executor');
  assert.equal(lens.vocabulary.vocab_size, 100352);
  assert.equal(lens.vocabulary.rows.length, 400);
  assert.equal(lens.readout_witness.positions, 5);
  const parity = JSON.parse(readFileSync(new URL('../public/recordings/granite-standard.parity.json', import.meta.url)));
  assert.equal(lens.readout_witness.canonical_logits_sha256, parity.observed_sha256);
  assert.equal(lens.readout_witness.readout_logits_sha256, parity.unobserved_sha256);
  assert.equal(lens.heads.rows.length, 8000);
  assert.equal(lens.experiments, undefined);
  const row = lens.vocabulary.rows.find(r => r.site === '4:39:ffn_write:main:residual');
  assert.equal(row.top[0].token, ' Paris');
  assert.equal(row.top[0].rank, 1);
  assert.ok(Math.abs(row.top[0].probability - .9061932565969187) < 1e-12);
  assert.equal(row.top.length, 20);
  assert.ok(row.top.reduce((sum,p)=>sum+p.probability,0) < 1);
  assert.notEqual(row.targets[0].logit, samples.at(-1).logits[' Paris']);
  const bad = structuredClone(raw);
  bad.readout_witness.readout_logits_sha256 = 'a'.repeat(64);
  assert.throws(() => parseLenses(bad, record, createHash('sha256').update(bytes).digest('hex')), /witness/);
  const missing = structuredClone(raw);
  missing.vocabulary.rows = missing.vocabulary.rows.filter(r => r.site !== '0:39:ffn_write:main:residual');
  assert.throws(() => parseLenses(missing, record, createHash('sha256').update(bytes).digest('hex')), /coverage/);
});


test('real Granite heads cover every token/layer/head and bind the raw canonical capture', async () => {
  const { parseLenses, availableLenses, headKey, lensAvailability } = await import('../lib/lenses.ts');
  const analysisBytes = readFileSync(new URL('../public/recordings/granite-standard.lenses.json', import.meta.url));
  assert.ok(analysisBytes.length <= 10_000_000);
  const lens = parseLenses(JSON.parse(analysisBytes), record, createHash('sha256').update(bytes).digest('hex'));
  const rawBytes = readFileSync(new URL(`../public/recordings/${lens.head_payload.file}`, import.meta.url));
  assert.equal(createHash('sha256').update(rawBytes).digest('hex'), lens.head_payload.sha256);
  const raw = JSON.parse(rawBytes);
  assert.equal(raw.length, 8000);
  assert.equal(lens.heads.rows.length, 8000);
  assert.equal(new Set(lens.heads.rows.map(headKey)).size, 8000);
  const byKey = new Map(raw.map(h => [`${h.position}:${h.layer}:${h.head}`, h]));
  for (const row of lens.heads.rows) {
    const [position,layer] = row.site.split(':').map(Number);
    const actual = byKey.get(`${position}:${layer}:${row.head}`);
    assert.ok(actual);
    assert.equal(actual.kv_head, Math.floor(row.head / 5));
    assert.equal(actual.values.length, 64);
    assert.equal(actual.weights.length, position + 1);
    assert.equal(row.sources.length, position + 1);
    for (const s of row.sources) assert.equal(Math.fround(s.weight), Math.fround(actual.weights[s.position]));
    assert.equal(row.content.projections.length, 8);
    assert.equal(row.dla, row.content.projections.find(p => p.token === row.target).coefficient);
  }
  assert.equal(lens.heads.reconstruction.writes, 200);
  assert.equal(lens.heads.reconstruction.result, 'pass');
  assert.ok(lens.heads.reconstruction.max_relative_l2 < 1e-4);
  assert.equal(lens.head_capture_witness.coverage, 'complete');
  assert.equal(lensAvailability(lens).Heads, true);
  assert.equal(lensAvailability(lens).Content, true);
  assert.equal(lensAvailability(lens).Experiment, false);
  const partial = samples.filter(s => s.position === 0 && s.layer < 3);
  const replay = availableLenses(lens, partial);
  assert.equal(replay.heads.rows.length, 120);
  assert.ok(replay.heads.rows.every(h => h.site.startsWith('0:')));
});
