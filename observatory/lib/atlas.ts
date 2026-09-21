import type { Recording, Sample } from './record.ts';
import { semanticKey } from './record.ts';
export interface Atlas {
  method: string;
  basis: { id: string; hash: string; axes_hash: string; axes: [string,string]; source: string; width: number; dimensions: 2; normalization: 'unit-l2'; construction: string; axes_values: [number[],number[]] };
  landmarks: { token_id: number; token: string; values: [number,number] }[];
  rows: { site: string; values: [number,number]; cosines: number[] }[];
  graph_method: string; edges: { from: number; to: number; cosine: number }[];
}
const finite = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n);
const text = (s: unknown): s is string => typeof s === 'string' && s.length > 0 && s.length < 4096;
const hash = (s: unknown) => typeof s === 'string' && /^sha256:[a-f0-9]{64}$/.test(s);
const unitXY = (xy: unknown): xy is [number,number] => Array.isArray(xy) && xy.length === 2 && xy.every(finite) && Math.hypot(...xy) <= 1.000001;
function fail(): never { throw new Error('Atlas evidence refused: invalid basis, landmark, source site or similarity graph.'); }
export function parseAtlas(value: unknown, record: Recording): Atlas {
  if (!value || typeof value !== 'object') fail();
  const a = value as Atlas, b = a.basis;
  if (!text(a.method) || !b || !text(b.id) || !hash(b.hash) || !hash(b.axes_hash) || !hash(b.source) || b.dimensions !== 2 || b.normalization !== 'unit-l2' || !Number.isSafeInteger(b.width) || b.width < 2 || b.width > 100_000 || !text(b.construction) || !Array.isArray(b.axes) || b.axes.length !== 2 || !b.axes.every(text)) fail();
  if (record.standard && (b.source !== record.standard.identity.container || b.width !== record.bases[0].hidden)) fail();
  if (!Array.isArray(b.axes_values) || b.axes_values.length !== 2 || b.axes_values.some(axis => !Array.isArray(axis) || axis.length !== b.width || !axis.every(finite) || Math.abs(axis.reduce((sum,v)=>sum+v*v,0)-1)>1e-6) || Math.abs(b.axes_values[0].reduce((sum,v,i)=>sum+v*b.axes_values[1][i],0))>1e-6) fail();
  if (!Array.isArray(a.landmarks) || a.landmarks.length < 2 || a.landmarks.length > 16) fail();
  const ids = new Set<number>();
  for (const l of a.landmarks) {
    if (!l || !Number.isSafeInteger(l.token_id) || l.token_id < 0 || ids.has(l.token_id) || !text(l.token) || !unitXY(l.values)) fail();
    ids.add(l.token_id);
  }
  if (Math.abs(a.landmarks[0].values[0]-1)>1e-6 || Math.abs(a.landmarks[0].values[1])>1e-6 || Math.abs(Math.hypot(...a.landmarks[1].values)-1)>1e-6 || a.landmarks[1].values[1]<=0) fail();
  const sites = new Set(record.events.flatMap(e=>e.sample ? [semanticKey(e.sample)] : [])), seen = new Set<string>();
  if (!Array.isArray(a.rows) || !a.rows.length || a.rows.length > sites.size) fail();
  for (const r of a.rows) {
    if (!r || !sites.has(r.site) || seen.has(r.site) || !unitXY(r.values) || !Array.isArray(r.cosines) || r.cosines.length !== ids.size || r.cosines.some(c=>!finite(c)||Math.abs(c)>1)) fail();
    if (Math.abs(r.values[0]-r.cosines[0])>1e-6 || Math.abs(r.values[0]*a.landmarks[1].values[0]+r.values[1]*a.landmarks[1].values[1]-r.cosines[1])>1e-6) fail();
    seen.add(r.site);
  }
  if (!text(a.graph_method) || !Array.isArray(a.edges) || a.edges.length > 120) fail();
  const pairs = new Set<string>();
  for (const e of a.edges) {
    if (!e || !ids.has(e.from) || !ids.has(e.to) || e.from===e.to || !finite(e.cosine) || Math.abs(e.cosine)>1) fail();
    const pair = [e.from,e.to].sort((a,b)=>a-b).join(':'); if (pairs.has(pair)) fail(); pairs.add(pair);
  }
  return a;
}
export function atlasJourney(atlas: Atlas, samples: Sample[], position: number, selected?: Sample) {
  const rows = new Map(atlas.rows.map(r=>[r.site,r]));
  const journey = samples.filter(s=>s.position===position).flatMap(sample=>{ const row=rows.get(semanticKey(sample)); return row ? [{ sample, ...row }] : []; });
  const at = selected ? journey.findIndex(r=>r.site===semanticKey(selected)) : -1;
  return at >= 0 ? journey.slice(0,at+1) : journey;
}
export function atlasPoint(values: readonly number[], zoom: number, center: readonly number[]) {
  return { x: 50+(values[0]-center[0])*38*zoom, y: 50-(values[1]-center[1])*38*zoom };
}

export function atlasTerminal(atlas: Atlas, record: Recording, samples: Sample[], position: number, completed: boolean, selected?: Sample) {
  const current = atlasJourney(atlas, samples, position, selected).at(-1);
  return completed && current?.sample.layer === record.layers - 1
    && current.sample.role === record.program.at(-1)?.roles.at(-1) ? current : undefined;
}

/** Retrospective overlay only. This never supplies the current-state readout. */
export function atlasRemainder(atlas: Atlas, record: Recording, samples: Sample[], position: number, completeRecord: boolean, currentSite?: string) {
  if (!completeRecord) return { future: [], terminal: undefined };
  const route = atlasJourney(atlas, samples, position);
  const at = currentSite === undefined ? -1 : route.findIndex(r => r.site === currentSite);
  if (currentSite !== undefined && at < 0) return { future: [], terminal: undefined };
  return { future: route.slice(at + 1), terminal: atlasTerminal(atlas, record, samples, position, true) };
}

export function atlasAdjacent(record: Recording, a: Sample, b: Sample) {
  return a.position === b.position && (a.layer === b.layer
    ? a.role === 'attention_write' && b.role === 'ffn_write'
    : b.layer === a.layer + 1 && b.role === 'attention_write' && a.role === record.program[a.layer]?.roles.at(-1));
}

/** Unit-normalized state and orthonormal axes: this is squared norm, not route variance. */
export function atlasOffPlane(values: readonly number[]) {
  return Math.max(0, Math.min(1, 1 - values[0] ** 2 - values[1] ** 2));
}

/** Frame immutable recorded coordinates with room for terminal/current labels. */
export function atlasFrame(values: readonly (readonly number[])[]) {
  if (!values.length) return { center: [0, 0] as [number, number], zoom: 1 };
  const lo = [0, 1].map(i => Math.min(...values.map(v => v[i])));
  const hi = [0, 1].map(i => Math.max(...values.map(v => v[i])));
  return {
    center: [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2] as [number, number],
    zoom: Math.min(32, 1.4 / Math.max(hi[0] - lo[0], hi[1] - lo[1], .04)),
  };
}
