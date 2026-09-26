import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { adaptStandard } from '../lib/standard.ts';
import { parseLenses, availableLenses } from '../lib/lenses.ts';
import { parseAtlas, atlasJourney, atlasPoint } from '../lib/atlas.ts';
import { reduceEvents } from '../lib/record.ts';
const bytes=readFileSync(new URL('../public/recordings/granite-standard.json',import.meta.url));
const record=adaptStandard(JSON.parse(bytes));
const analysis=JSON.parse(readFileSync(new URL('../public/recordings/granite-standard.lenses.json',import.meta.url)));
const atlas=analysis.token_map, samples=reduceEvents(record.events).samples;
test('real Atlas binds 400 captured sites and eight token landmarks to one orthonormal plane',()=>{
  parseLenses(analysis,record,createHash('sha256').update(bytes).digest('hex'));
  assert.equal(atlas.rows.length,400);assert.equal(atlas.landmarks.length,8);
  assert.equal(atlas.basis.width,2560);
  const axesBytes=Buffer.alloc(atlas.basis.width*2*8);
  atlas.basis.axes_values.flat().forEach((v,i)=>axesBytes.writeDoubleLE(v,i*8));
  assert.equal(atlas.basis.axes_hash,`sha256:${createHash('sha256').update(axesBytes).digest('hex')}`);
  assert.equal(analysis.token_map.producer.inference_executions,0);
  assert.equal(analysis.token_map.producer.carrier_sha256,analysis.carrier_payload.sha256);
  assert.equal(atlas.rows.at(-1).cosines[0],0.2146544490739294);
});
test('site selection truncates the journey at the traveller, preserving both write stages',()=>{
  const selected=samples.find(s=>s.position===4&&s.layer===23&&s.role==='attention_write');
  const journey=atlasJourney(atlas,samples,4,selected);
  assert.equal(journey.length,47);
  assert.equal(journey.at(-1).sample,selected);
  assert.ok(journey.every(r=>r.sample.position===4));
  assert.equal(journey.at(-2).sample.role,'ffn_write');
  const state=reduceEvents(record.events.slice(0,15));
  const visible=availableLenses(analysis,state.samples,state.events.at(-1).sequence);
  assert.equal(visible.token_map.rows.length,state.samples.length);
  assert.deepEqual(visible.token_map.landmarks,atlas.landmarks);
  assert.equal(atlasJourney(visible.token_map,state.samples,4).length,0);
});
test('zoom changes screen coordinates, never recorded geography or full-space similarity',()=>{
  const before=structuredClone(atlas), row=atlas.rows.at(-1);
  const a=atlasPoint(row.values,1,[0,0]), b=atlasPoint(row.values,3,row.values);
  assert.notDeepEqual(a,b);assert.deepEqual(b,{x:50,y:50});
  assert.deepEqual(atlas,before);
  // A selected token's 2D location is not its full-dimensional unit distance.
  const france=atlas.landmarks.findIndex(l=>l.token===' France');
  assert.notEqual(Math.hypot(...atlas.landmarks[france].values),1);
});
test('malformed basis, fabricated sites, duplicate landmarks and dangling graph edges refuse',()=>{
  for(const change of [a=>a.basis.source='sha256:'+'a'.repeat(64),a=>a.basis.axes_values[0][0]+=1,a=>a.rows[0].site='invented',a=>a.rows.push(a.rows[0]),a=>a.rows[0].cosines[0]=2,a=>a.rows[0].values[0]=.9,a=>a.landmarks.push(a.landmarks[0]),a=>a.edges[0].to=999999,a=>a.edges.push(a.edges[0])]) {
    const bad=structuredClone(atlas);change(bad);assert.throws(()=>parseAtlas(bad,record),/Atlas evidence refused/);
  }
});

test('retrospective remainder exposes only the measured later route without altering current readout',async()=>{
  const {atlasRemainder,atlasAdjacent,atlasOffPlane}=await import('../lib/atlas.ts');
  const selected=samples.find(s=>s.position===4&&s.layer===18&&s.role==='ffn_write');
  const past=atlasJourney(atlas,samples,4,selected);
  const snapshot=structuredClone(analysis);
  const remainder=atlasRemainder(atlas,record,samples,4,true,past.at(-1).site);
  assert.equal(past.length,38);
  assert.equal(remainder.future.length,42);
  assert.equal(remainder.future[0].sample.layer,19);
  assert.equal(remainder.future[0].sample.role,'attention_write');
  assert.equal(remainder.terminal.sample.layer,39);
  assert.deepEqual(remainder.terminal.values,atlas.rows.at(-1).values);
  assert.ok(atlasAdjacent(record,past.at(-1).sample,remainder.future[0].sample));
  assert.equal(atlasRemainder(atlas,record,samples,4,false,past.at(-1).site).future.length,0);
  assert.equal(atlasRemainder(atlas,record,samples,4,false).terminal,undefined);
  assert.equal(atlasRemainder(atlas,record,samples,4,true,'missing').terminal,undefined);
  assert.equal(atlasRemainder(atlas,record,samples,4,true,remainder.terminal.site).future.length,0);
  assert.deepEqual(analysis,snapshot);
  assert.ok(atlasOffPlane(remainder.terminal.values)>.95);
  assert.equal(atlasOffPlane([1,0]),0);
  assert.equal(atlasOffPlane([0,0]),1);
  assert.equal(atlasAdjacent(record,past.at(-2).sample,remainder.future[0].sample),false);
  assert.equal(atlasAdjacent(record,{...selected,position:3},remainder.future[0].sample),false);
});

test('default route framing fills the plotting field and retains label margins at every Granite position',async()=>{
  const {atlasFrame}=await import('../lib/atlas.ts');
  for(let position=0;position<record.tokens.length;position++) {
    const route=atlasJourney(atlas,samples,position);
    const before=structuredClone(route.map(r=>r.values));
    const camera=atlasFrame(before);
    const points=before.map(v=>atlasPoint(v,camera.zoom,camera.center));
    for(const p of points) assert.ok(p.x>=20&&p.x<=80&&p.y>=20&&p.y<=80);
    const span=Math.max(Math.max(...points.map(p=>p.x))-Math.min(...points.map(p=>p.x)),Math.max(...points.map(p=>p.y))-Math.min(...points.map(p=>p.y)));
    assert.ok(span>45,`position ${position} should have a legible route, got ${span}%`);
    assert.deepEqual(route.map(r=>r.values),before);
  }
});
