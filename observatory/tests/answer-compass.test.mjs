import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {compassReadout} from '../lib/answer-compass.ts';
import {atlasTerminal} from '../lib/atlas.ts';
import {adaptStandard} from '../lib/standard.ts';
import {reduceEvents} from '../lib/record.ts';
const analysis=JSON.parse(readFileSync(new URL('../public/recordings/granite-standard.lenses.json',import.meta.url)));
const record=adaptStandard(JSON.parse(readFileSync(new URL('../public/recordings/granite-standard.json',import.meta.url))));
const atlas=analysis.token_map, rows=analysis.vocabulary.rows, samples=reduceEvents(record.events).samples;
test('Granite readout preference stays separate from geometry and reports the real falling trend',()=>{
  const r=compassReadout(rows.at(-1),rows.at(-2),atlas);
  assert.equal(r.prediction.token,' Paris');
  assert.equal(r.prediction.probability,0.9061932565969187);
  assert.equal(r.cosine,0.2146544490739294);
  assert.ok(r.delta<0);
  assert.equal(r.delta,0.9061932565969187-0.9956225074725471);
  assert.equal(r.candidates.length,5);
  assert.ok(r.retainedMass<1);
  assert.equal(r.candidates[0].probability,r.prediction.probability);
});
test('missing readouts, previous targets and token geometry remain unavailable',()=>{
  assert.equal(compassReadout(undefined,rows.at(-1),atlas).prediction,undefined);
  assert.equal(compassReadout(rows.at(-1),undefined,atlas).delta,undefined);
  const absent=compassReadout(rows.at(-1),rows.at(-2),atlas,999999);
  assert.equal(absent.prediction,undefined); assert.equal(absent.cosine,undefined); assert.equal(absent.bearing,undefined);
  const fake={site:'not-a-captured-site',top:rows.at(-1).top,targets:[],entropy:1};
  assert.equal(compassReadout(fake,undefined,atlas).cosine,undefined);
});
test('destination is the captured terminal state, never a token anchor or a future replay state',()=>{
  const end=atlasTerminal(atlas,record,samples,4,true);
  assert.deepEqual(end.values,atlas.rows.at(-1).values);
  assert.notDeepEqual(end.values,atlas.landmarks[0].values);
  assert.equal(end.sample.layer,39);
  assert.equal(end.sample.role,'ffn_write');
  assert.equal(atlasTerminal(atlas,record,samples,4,false),undefined);
  assert.equal(atlasTerminal(atlas,record,samples,4,true,samples.find(s=>s.position===4&&s.layer===12)),undefined);
  assert.equal(atlasTerminal(atlas,record,samples.slice(0,-1),4,true),undefined);
});
