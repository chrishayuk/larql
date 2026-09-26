import assert from 'node:assert/strict';
import test from 'node:test';
import { adaptStandard, openRecording, STANDARD_SCHEMA } from '../lib/standard.ts';
import { reduceEvents, samplesAt, semanticKey } from '../lib/record.ts';
import { auditStandard } from '../lib/replay-audit.ts';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';

// Authored adapter contract cases, not a real execution/parity witness.
export function standardRecord() {
  const run_id = 'adapter-test';
  const r = {
    schema: STANDARD_SCHEMA, provenance: 'synthetic', run_id, model: 'contract-test',
    identity: { container: 'test-container', plan: 'test-plan', lowering: 'reference', tokenizer: 'test-tokenizer', session: 'test-session' },
    capture: 'standard', topology: 'Single', norm_method: 'l2-v1', probe_method: 'dot-raw-v1',
    layers: 2, program: [
      { layer: 0, sites: [{ site: 'Attention', operation_id: 'mixer0' }] },
      { layer: 1, sites: [{ site: 'Attention', operation_id: 'attn1' }, { site: 'Ffn', operation_id: 'ffn1' }] },
    ], tokens: [{ position: 0, token_id: 100, label: 'query' }], observed_positions: [0],
    probe_tokens: [{ token_id: 7, label: 'answer' }, { token_id: 9, label: 'answer' }],
    basis: { provider: 'supplied-rows-v1', id: 'test-basis', hash_hex: 'a'.repeat(64), dims: 3, hidden: 16, source: 'test-only' },
    runtime: { timing_intrusive: true, device_readbacks: 0 }, coverage: 'complete', events: []
  };
  const add = (kind, data = {}) => r.events.push({ run_id, sequence: String(r.events.length), timestamp_ns: String(9007199254740993n + BigInt(r.events.length)), kind, ...data });
  const row = (layer, site, scale = null) => ({ layer, site, position: 0, norm: 31.25, delta_norm: 2.75, layer_scale: scale, projection: [.125, -.5, 1.25], probe: [2.125, -1.75] });
  add('RunStarted'); add('Tokenized');
  add('CarrierWrite', { stats: row(0, 'Attention') });
  add('SiteCompleted', { layer: 0, site: 'FfnDone', position: 0 });
  add('CarrierWrite', { stats: row(1, 'Attention') });
  add('CarrierWrite', { stats: row(1, 'Ffn', .5) });
  add('TokenProduced', { token_id: 7, predicting_position: 0, text: 'answer' });
  add('RunCompleted', { reason: 'max_tokens' });
  return r;
}

test('Standard adapter preserves coordinates, f64 norms, raw probes, scale and session identity', () => {
  const source = standardRecord(), r = adaptStandard(source), state = reduceEvents(r.events);
  assert.equal(r.provenance, 'synthetic'); assert.equal(state.samples.length, 3);
  assert.deepEqual(state.samples[0].projections['test-basis'], [.125, -.5, 1.25]);
  assert.deepEqual(state.samples[0].logits, { 'answer [7]': 2.125, 'answer [9]': -1.75 });
  assert.equal(state.samples[2].layer_scale, .5); assert.equal(state.samples[2].norm, 31.25);
  assert.equal(state.samples[0].operation_id, 'mixer0');
  assert.equal(r.standard.probe_method, 'dot-raw-v1');
  assert.equal(state.events[0].timestamp_ns, '9007199254740993');
  for (const s of state.samples) for (const unavailable of ['before', 'duration_ns', 'write_logits', 'sources']) assert.equal(s[unavailable], undefined);
});
test('original record import/export and every replay prefix reconstruct the same selected sites', () => {
  const original = standardRecord(), opened = openRecording(original);
  assert.deepEqual(opened.source, original);
  const restored = openRecording(JSON.parse(JSON.stringify(opened.source)));
  for (let n = 1; n <= original.events.length; n++) {
    const a = reduceEvents(opened.record.events.slice(0, n)), b = reduceEvents(restored.record.events.slice(0, n));
    assert.deepEqual(a, b);
    assert.deepEqual(samplesAt(a, 0, 'attention_write').map(semanticKey), samplesAt(b, 0, 'attention_write').map(semanticKey));
  }
});
test('FfnDone without a write is only a structural boundary', () => {
  const state = reduceEvents(adaptStandard(standardRecord()).events);
  assert.equal(state.samples.filter(s => s.layer === 0 && s.role === 'ffn_write').length, 0);
  assert.equal(state.samples.some(s => s.role === 'embedding'), false);
});
test('complete coverage refuses missing writes, holes, duplicates, or nonterminal logs', () => {
  const missing = standardRecord(); missing.events.splice(2, 1);
  assert.throws(() => adaptStandard(missing), /coverage/);
  missing.events.forEach((e, i) => e.sequence = String(i));
  assert.throws(() => adaptStandard(missing), /declared writes/);
  const duplicate = standardRecord(); duplicate.events[4].stats = structuredClone(duplicate.events[2].stats);
  assert.throws(() => adaptStandard(duplicate), /Duplicate semantic/);
  const partial = standardRecord(); partial.events.pop();
  assert.throws(() => adaptStandard(partial), /coverage/);
});
test('explicit incomplete records preserve gaps and terminal loss without later observations', () => {
  const r = standardRecord(); r.coverage = 'incomplete'; r.events.splice(2, 1);
  r.events.splice(-1, 0, { run_id: r.run_id, sequence: '7', timestamp_ns: '9007199254741000', kind: 'EventDropped', count: 12 });
  r.events.at(-1).sequence = '8'; r.events.at(-1).timestamp_ns = '9007199254741001';
  const adapted = adaptStandard(r), state = reduceEvents(adapted.events);
  assert.equal(state.dropped, 12); assert.deepEqual(state.gaps, [{ from: '2', to: '2' }]);
  assert.equal(state.status, 'completed'); assert.equal(adapted.standard.coverage, 'incomplete');
});
test('reconnect deliveries deduplicate by envelope identity and conflicts refuse', () => {
  const r = standardRecord(); r.events.push(structuredClone(r.events[2]));
  assert.equal(adaptStandard(r).events.length, 8);
  r.events.at(-1).stats.norm += 1;
  assert.throws(() => adaptStandard(r), /Conflicting duplicate/);
});
test('zero selected probes remain unavailable and token labels may be redacted', () => {
  const r = standardRecord(); r.probe_tokens = []; delete r.tokens[0].label;
  r.events.filter(e => e.stats).forEach(e => e.stats.probe = []);
  const adapted = adaptStandard(r);
  assert.deepEqual(adapted.answers, []); assert.deepEqual(adapted.tokens, ['#100']);
  assert.deepEqual(reduceEvents(adapted.events).samples[0].logits, {});
});
test('refusal is a terminal execution state, not successful completion', () => {
  const r = standardRecord(); r.coverage = 'incomplete';
  r.events = r.events.slice(0, 1);
  r.events.push({ run_id: r.run_id, sequence: '1', timestamp_ns: '9007199254740994', kind: 'RunRefused', reason: 'unsupported lowering' });
  assert.equal(reduceEvents(adaptStandard(r).events).status, 'refused');
});
test('unsupported topology, method, basis shape, scope and identity fail explicitly', () => {
  const changes = [r => r.topology = 'Bundle', r => r.probe_method = 'softmax', r => r.basis.dims = 2,
    r => r.events[2].stats.position = 1, r => r.events[2].run_id = 'another',
    r => r.events[2].stats.norm = NaN, r => r.events[2].timestamp_ns = 42,
    r => r.events[2].stats.projection.push(4), r => r.events[2].stats.probe = [],
    r => r.events[2].stats.site = 'Unknown', r => r.identity = {},
    r => r.program[0].sites[0].operation_id = 'ffn1', r => r.probe_tokens[1].token_id = 7,
    r => r.events[2].timestamp_ns = '18446744073709551616'];
  for (const change of changes) { const r = standardRecord(); change(r); assert.throws(() => adaptStandard(r)); }
});

test('source-to-replay audit checks every write and roundtrip without upgrading evidence', () => {
  const source = standardRecord();
  source.events.push(structuredClone(source.events[2]));
  const audit = auditStandard(source);
  assert.equal(audit.writes_checked, 3);
  assert.equal(audit.events_checked, 8);
  assert.equal(audit.duplicate_deliveries, 1);
  assert.equal(audit.replay_checkpoints, 9);
  assert.equal(audit.source_values, 'exact');
  assert.equal(audit.save_reopen, 'exact');
  assert.equal(audit.provenance, 'synthetic');
  assert.equal(audit.execution_parity, 'not_checked');
});

test('independent fidelity oracle catches corrupt view values, fabricated data and missing writes', () => {
  const changes = [
    r => r.events[2].sample.projections['test-basis'][0] += Number.EPSILON,
    r => r.events[2].sample.norm += .01,
    r => r.events[2].sample.delta += .01,
    r => r.events[2].sample.logits['answer [7]'] += .01,
    r => r.events[2].sample.before = 10,
    r => r.events[3].sample = structuredClone(r.events[2].sample),
    r => r.events[2].timestamp_ns = '9007199254740992',
    r => r.events[5].sample.layer_scale = 1,
    r => r.events.splice(2, 1),
    r => r.events[6].output = 'invented answer',
  ];
  for (const change of changes) {
    const source = standardRecord(), candidate = adaptStandard(source);
    change(candidate);
    assert.throws(() => auditStandard(source, candidate));
  }
});

test('replay audit preserves incomplete coverage and empty probe captures', () => {
  const source = standardRecord(); source.coverage = 'incomplete';
  source.events.splice(2, 1);
  source.probe_tokens = [];
  source.events.filter(e => e.stats).forEach(e => e.stats.probe = []);
  const audit = auditStandard(source);
  assert.equal(audit.coverage, 'incomplete');
  assert.equal(audit.writes_checked, 2);
  assert.equal(audit.execution_parity, 'not_checked');
});

test('preflight binds its report to input bytes and refuses synthetic evidence for the real-record gate', () => {
  const dir = mkdtempSync(join(tmpdir(), 'observatory-audit-'));
  try {
    const path = join(dir, 'synthetic.json'), bytes = JSON.stringify(standardRecord());
    writeFileSync(path, bytes);
    const run = (...args) => spawnSync(process.execPath, ['scripts/check-standard.mjs', path, ...args], { cwd: new URL('..', import.meta.url), encoding: 'utf8' });
    const accepted = run('--require-complete');
    assert.equal(accepted.status, 0, accepted.stderr);
    const report = JSON.parse(accepted.stdout);
    assert.equal(report.source_sha256, createHash('sha256').update(bytes).digest('hex'));
    assert.equal(report.audit.execution_parity, 'not_checked');
    const refused = run('--require-executor');
    assert.equal(refused.status, 1);
    assert.match(refused.stderr, /synthetic input refused/);
    const incomplete = standardRecord(); incomplete.coverage = 'incomplete'; incomplete.events.pop();
    writeFileSync(path, JSON.stringify(incomplete));
    assert.equal(run('--require-complete').status, 1);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('longer records check all positions and cap replay checkpoint work', () => {
  const source = standardRecord(), template = structuredClone(source.events);
  source.tokens = Array.from({ length: 64 }, (_, position) => ({ position, token_id: position }));
  source.observed_positions = source.tokens.map(t => t.position);
  source.events = template.slice(0, 2);
  for (const position of source.observed_positions) {
    for (const original of template.slice(2, -2)) {
      const event = structuredClone(original);
      if (event.stats) event.stats.position = position;
      if (event.kind === 'SiteCompleted') event.position = position;
      source.events.push(event);
    }
  }
  source.events.push(template.at(-1));
  source.events.forEach((e, i) => { e.sequence = String(i); e.timestamp_ns = String(9007199254740993n + BigInt(i)); });
  const audit = auditStandard(source);
  assert.equal(audit.writes_checked, 192);
  assert.equal(audit.replay_checkpoints, 33);
});
