import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { openRecordingText } from '../lib/ingestion.ts';
import { readVindex3Record } from '../lib/vindex3-record.ts';
import { reduceEvents, semanticKey, stepIndex } from '../lib/record.ts';
import { availableLenses, parseLenses, vocabularyAt } from '../lib/lenses.ts';
import { project } from '../lib/projection.ts';
const bytes = readFileSync(new URL('../public/recordings/gemma-paris-v3.jsonl', import.meta.url), 'utf8');
const golden = JSON.parse(readFileSync(new URL('./golden/gemma-paris-v3.expected.json', import.meta.url), 'utf8'));
const hash = text => createHash('sha256').update(text).digest('hex');
const opened = await openRecordingText(bytes);
const record = opened.record;
const full = reduceEvents(record.events);
function mutated(change, seal = true) {
  const lines = bytes.trimEnd().split('\n').map(line => JSON.parse(line));
  change(lines);
  if (seal && lines.at(-1)?.receipt) {
    const r = lines.at(-1).receipt;
    const events = lines.slice(1,-1);
    r.events = events.length; r.last_sequence = events.at(-1)?.sequence ?? null;
    r.log_sha256 = hash(events.map(e => JSON.stringify(e) + '\n').join(''));
  }
  return lines.map(e => JSON.stringify(e)).join('\n') + '\n';
}

test('Paris CLI file is opened unchanged with independent frozen receipt and representative writes', () => {
  assert.equal(opened.bytes, bytes); assert.equal(opened.digest, golden.source_sha256);
  assert.equal(record.id, golden.run_id); assert.deepEqual(record.vindex3.prompt_tokens, golden.prompt_tokens);
  assert.equal(record.bases[0].hash, `sha256:${golden.basis.hash_hex}`);
  assert.equal(record.bases[0].id, golden.basis.id);
  assert.equal(record.vindex3.events, golden.events); assert.equal(full.samples.length, golden.writes);
  assert.equal(opened.lenses.vocabulary.rows.length, golden.readouts);
  assert.equal(full.status, 'completed'); assert.equal(full.output, golden.terminal.output);
  assert.deepEqual(full.gaps, []); assert.equal(full.dropped, 0);
  assert.deepEqual(opened.source.receipt, golden.terminal.receipt);
  assert.deepEqual(opened.source.events.at(-1), golden.terminal.event);
  for (const { stats, write, lens } of golden.checkpoints) {
    const s = record.events[write.sequence].sample;
    assert.equal(s.position, stats.position); assert.equal(s.layer, stats.layer);
    assert.equal(s.role, `${stats.site === 'ffn' ? 'ffn' : 'attention'}_write`);
    assert.equal(s.norm, stats.norm); assert.equal(s.delta, stats.delta_norm);
    assert.deepEqual(s.projections[golden.basis.id], stats.projection);
    assert.equal(record.events[write.sequence].timestamp_ns, String(write.timestamp_ns));
    assert.equal(s.layer_scale, stats.layer_scale ?? undefined);
    const row = vocabularyAt(opened.lenses, s);
    if (lens) for (const t of lens.tokens) {
      const shown = row.targets.find(p => p.token_id === t.id);
      assert.equal(shown.logprob, t.logprob); assert.equal(shown.rank, t.rank);
      assert.equal(shown.probability, Math.exp(t.logprob));
      assert.equal(shown.logit, undefined); assert.equal(row.entropy, undefined);
    } else assert.equal(row, undefined);
  }
  assert.equal(vocabularyAt(opened.lenses, record.events[1380].sample).targets[0].rank, 1);
  assert.equal(vocabularyAt(opened.lenses, record.events[1212].sample).targets[0].rank, 14055);
});

test('every source write is presented once, in source order, with every exact norm and coordinate', () => {
  const events = opened.source.events;
  const writes = events.filter(e => e.kind === 'carrier_write');
  assert.deepEqual(full.samples.map(s => s.sequence), writes.map(e => String(e.sequence)));
  for (let i = 0; i < writes.length; i++) {
    const w = writes[i], s = full.samples[i];
    const stats = events[w.sequence - (events[w.sequence-1].kind === 'readout' ? 2 : 1)];
    assert.equal(stats.kind, 'carrier_stats');
    assert.deepEqual([s.norm,s.delta,s.position,s.layer], [stats.norm,stats.delta_norm,w.position,w.layer]);
    assert.deepEqual(s.projections[record.bases[0].id], stats.projection);
    assert.equal(s.before, undefined); assert.equal(s.duration_ns, undefined); assert.equal(s.operation_id, undefined);
  }
});

test('scrubbing both directions, refresh/reopen and camera controls preserve immutable evidence and geometry', async () => {
  const reopened = await openRecordingText(opened.bytes); // same bytes restored from tab or reopened from file
  assert.deepEqual(reopened.record, record); assert.deepEqual(reopened.lenses, opened.lenses);
  const before = JSON.stringify(record);
  const domain = full.samples.filter(s => s.position === 5 && s.role === 'ffn_write');
  const fixed = project(domain, record.bases[0], true, domain);
  for (const index of [1443,1212,1394,1380,1212,1443]) {
    const state = reduceEvents(record.events.slice(0,index+1));
    const again = reduceEvents(reopened.record.events.slice(0,index+1));
    assert.deepEqual(state, again);
    const selected = state.samples.at(-1);
    assert.deepEqual(selected, full.samples.find(s => semanticKey(s) === semanticKey(selected)));
    const prefix = state.samples.filter(s => s.position === 5 && s.role === 'ffn_write');
    assert.deepEqual(project(prefix,record.bases[0],true,domain),fixed.slice(0,prefix.length));
    project(prefix,record.bases[0],false,domain); // view only
    assert.ok(stepIndex(record,index,5,-1) < index);
  }
  assert.equal(JSON.stringify(record),before);
  assert.throws(() => { record.events[1212].sample.projections[record.bases[0].id][0] = 0; }, TypeError);
});

test('lens evidence is visible only with its matching write; save/reopen analysis preserves absent entropy/logits', () => {
  const before = reduceEvents(record.events.slice(0,1212));
  const after = reduceEvents(record.events.slice(0,1213));
  const key = semanticKey(record.events[1212].sample);
  assert.equal(availableLenses(opened.lenses,before.samples,before.events.at(-1).sequence).vocabulary.rows.some(r=>r.site===key),false);
  assert.equal(availableLenses(opened.lenses,after.samples,after.events.at(-1).sequence).vocabulary.rows.some(r=>r.site===key),true);
  assert.deepEqual(parseLenses(JSON.parse(JSON.stringify(opened.lenses)),record,opened.digest),opened.lenses);
});

test('unknown schemas, missing basis, fingerprint mismatch, malformed sites and unsupported lenses refuse', async () => {
  const cases = [
    [a=>a[0].header.identity.schema='future.v2', /schema/],
    [a=>delete a[0].header.provenance.basis, /basis/],
    [a=>a[0].header.provenance.execution.arithmetic_arm='changed', /fingerprint/],
    [a=>a.at(-1).receipt.provenance_fingerprint='0'.repeat(64), /provenance/],
    [a=>a[3].site='unknown', /site identity/],
    [a=>a[4].layer=1, /mismatched stats/],
    [a=>a[4].position=1, /position/],
    [a=>a[4].carrier='bundle', /topology/],
    [a=>a[7].method='raw-probe', /lens semantics/],
    [a=>a[7].tokens[0].logprob=1, /Log-probability/],
    [a=>a[7].tokens[0].rank=999999999, /vocabulary/],
    [a=>a[3].projection=[1,2], /dimension/],
    [a=>a[3].norm=-1, /norm/],
    [a=>a[5].layer=12, /boundary/],
    [a=>a[1].kind='unknown', /embedding|Unsupported/],
    [a=>a[7].kind='new_lens', /Unsupported/],
    [a=>a[10].sequence=5, /sequence/],
    [a=>a[10].timestamp_ns=1, /timestamp/],
    [a=>a.at(-1).receipt.head_passes=1, /Head-pass/],
    [a=>a.splice(4,0,structuredClone(a[0])), /count\/last sequence|envelope|sequence/],
  ];
  for (const [change,pattern] of cases) await assert.rejects(openRecordingText(mutated(change)),pattern);
});

test('corruption, truncation, missing receipt, and forged completion are never silently plotted', async () => {
  await assert.rejects(readVindex3Record(mutated(a=>a[3].norm+=1,false)),/SHA-256/);
  await assert.rejects(readVindex3Record(mutated(a=>a.at(-1).receipt.events++,false)),/count/);
  await assert.rejects(readVindex3Record(bytes.slice(0,bytes.lastIndexOf('{"receipt"'))),/receipt/);
  await assert.rejects(readVindex3Record(mutated(a=>a.splice(-3,2))),/sequence|Completion/);
  await assert.rejects(readVindex3Record(mutated(a=>a.at(-1).receipt={...a.at(-1).receipt,complete:'yes'})),/completion/);
});

test('explicit incomplete receipt is degraded, live drops do not corrupt lossless replay, lens failure stays visible', async () => {
  const partial=await readVindex3Record(mutated(a=>a.at(-1).receipt.complete=false));
  assert.equal(reduceEvents(partial.record.events).status,'running');
  assert.equal(partial.record.standard.coverage,'incomplete');
  assert.match(partial.record.vindex3.warnings[0],/INCOMPLETE/);
  const lost=await readVindex3Record(mutated(a=>a.at(-1).receipt.live_dropped=50));
  assert.equal(reduceEvents(lost.record.events).dropped,0);
  assert.equal(lost.record.vindex3.live_dropped,'50');
  assert.equal(lost.record.vindex3.complete,true);
  const failed=await readVindex3Record(mutated(a=>{a.at(-1).receipt.head_passes++;a.at(-1).receipt.lens_failure='head refused';}));
  assert.match(failed.record.vindex3.warnings[0],/LENS FAILED/);
});

test('u64 monotonic clocks preserve integers above JS safe range and reject overflow', async () => {
  const lines=bytes.trimEnd().split('\n');
  const base=9007199254740993n;
  const events=lines.slice(1,-1).map((line,i)=>line.replace(/"timestamp_ns":\d+/,`"timestamp_ns":${base+BigInt(i)}`));
  const receipt=JSON.parse(lines.at(-1)); receipt.receipt.log_sha256=hash(events.map(l=>l+'\n').join(''));
  const large=[lines[0],...events,JSON.stringify(receipt)].join('\n')+'\n';
  const opened=await readVindex3Record(large);
  assert.equal(opened.record.events[0].timestamp_ns,'9007199254740993');
  assert.equal(opened.record.events[1].timestamp_ns,'9007199254740994');
  const overflow=mutated(a=>a[1].timestamp_ns='18446744073709551616');
  await assert.rejects(readVindex3Record(overflow),/u64/);
});

test('legacy fixtures and Standard records still use the shared bytes ingestion boundary', async () => {
  for (const name of ['fixtures/answer.json','recordings/granite-standard.json']) {
    const text=readFileSync(new URL(`../public/${name}`,import.meta.url),'utf8');
    const r=await openRecordingText(text);
    assert.equal(r.bytes,text); assert.equal(r.format,'json'); assert.equal(r.record.vindex3,undefined);
  }
});

test('the actual inspector renders golden source precision, lens ranks and provenance without invented output', async () => {
  // Transpile the real leaf component for server rendering using the project's compiler.
  // No DOM mocks or copied rendering implementation; the same component is mounted by Observatory.
  const { default: ts } = await import('typescript');
  const { createElement } = await import('react');
  const { renderToStaticMarkup } = await import('react-dom/server');
  const source=readFileSync(new URL('../components/Vindex3Evidence.tsx',import.meta.url),'utf8');
  let js=ts.transpileModule(source,{compilerOptions:{jsx:ts.JsxEmit.ReactJSX,module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}}).outputText;
  js=js.replaceAll('"react/jsx-runtime"', JSON.stringify(import.meta.resolve('react/jsx-runtime'))).replaceAll("'../lib/lenses'",JSON.stringify(new URL('../lib/lenses.ts',import.meta.url).href));
  const { CarrierReadout, Vindex3Readout, Vindex3Evidence } = await import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`);
  for (const { stats, write, lens } of golden.checkpoints) {
    const selected=record.events[write.sequence].sample;
    const carrier=renderToStaticMarkup(createElement(CarrierReadout,{record,selected}));
    for (const v of [stats.norm,stats.delta_norm,...stats.projection]) assert.ok(carrier.includes(String(v)),String(v));
    const readout=renderToStaticMarkup(createElement(Vindex3Readout,{lenses:opened.lenses,selected}));
    if (lens) for (const t of lens.tokens) {
      assert.ok(readout.includes(`rank ${t.rank}`)); assert.ok(readout.includes(String(t.logprob)));
    } else assert.match(readout,/No lens readout recorded/);
  }
  const provenance=renderToStaticMarkup(createElement(Vindex3Evidence,{record}));
  for (const value of [record.id,record.model,golden.basis.hash_hex,'larql.run-record.v1','head-v1',golden.source_sha256]) assert.ok(provenance.includes(value));
  assert.match(provenance,/container content hash: not recorded/);
  assert.match(provenance,/COMPLETE RECEIPT/);
});

test('stats-only mixer traversal keeps FfnDone boundaries without inventing FFN writes or lenses', async () => {
  const changed=mutated(lines=>{
    for (let i=lines.length-2;i>0;i--) {
      const e=lines[i];
      if (e.site==='ffn' && ['carrier_stats','carrier_write','readout'].includes(e.kind)) lines.splice(i,1);
    }
    lines.slice(1,-1).forEach((e,i)=>e.sequence=i);
    lines.at(-1).receipt.head_passes=0;
  });
  const result=await readVindex3Record(changed);
  assert.equal(result.lenses,undefined);
  assert.equal(reduceEvents(result.record.events).samples.length,204);
  assert.ok(result.record.program.every(p=>p.roles.length===1 && p.roles[0]==='attention_write'));
});

test('sparse armed sites stay absent instead of filling in neighboring lens values', async () => {
  const changed=mutated(lines=>{
    for (let i=lines.length-2;i>0;i--) if (lines[i].kind==='readout' && lines[i].layer!==24) lines.splice(i,1);
    lines.slice(1,-1).forEach((e,i)=>e.sequence=i);
    lines.at(-1).receipt.head_passes=6;
  });
  const result=await readVindex3Record(changed);
  assert.equal(result.lenses.vocabulary.rows.length,6);
  const state=reduceEvents(result.record.events);
  assert.equal(state.samples.length,408);
  assert.equal(vocabularyAt(result.lenses,state.samples.find(s=>s.position===5&&s.layer===23&&s.role==='ffn_write')),undefined);
});

test('a sealed incomplete prefix does not expose an unfinished carrier write or pending lens', async () => {
  const prefix=mutated(lines=>{
    const receipt=lines.at(-1);
    lines.splice(8,lines.length-8,receipt);
    receipt.receipt.complete=false; receipt.receipt.head_passes=1;
  });
  const result=await readVindex3Record(prefix);
  assert.equal(reduceEvents(result.record.events).samples.length,1);
  assert.equal(result.lenses,undefined);
  assert.equal(result.record.standard.coverage,'incomplete');
});
