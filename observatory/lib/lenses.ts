/** Analysis sidecars are UI/runner interchange, never an executor ABI.
 * No analysis is inferred from Standard raw probes. */
import type { Recording, Sample } from './record.ts';
import { semanticKey } from './record.ts';
import { parseAtlas } from './atlas.ts';
import type { Atlas } from './atlas.ts';
export type Lens = 'Logits' | 'Heads' | 'Content' | 'Compare' | 'KV anatomy' | 'Experiment';
/** Record-wide availability; replay visibility is evaluated separately. */
export function lensAvailability(analysis?: LensRecord): Record<Lens, boolean> {
  return {
    Logits: !!analysis?.vocabulary?.rows.length,
    Heads: !!analysis?.heads?.rows.length,
    Content: !!analysis?.heads?.rows.some(h => h.content),
    Compare: true,
    'KV anatomy': !!analysis?.kv,
    Experiment: !!analysis?.experiments?.length,
  };
}
export interface Prediction { token_id: number; token: string; probability: number; logit?: number; logprob?: number; rank: number }
export interface VocabularyRow {
  site: string; top: Prediction[]; targets: Prediction[]; entropy?: number; sequence?: string;
}
export interface HeadRow {
  site: string; head: number; target: string; dla: number;
  sources: { position: number; weight: number }[];
  content?: {
    method: string; basis: string; vector_norm: number;
    projections: { token: string; token_id: number; coefficient: number }[];
    dimensionality?: { method: string; d95: number; d99: number; d999: number; dimensions: number };
    orthogonality?: { method: string; pairs: { label: string; cosine: number }[] };
  };
}
export interface LensRecord {
  schema: 'larql.observatory.lenses.v1'; run_id: string; source_sha256: string;
  provenance: 'synthetic' | 'executor'; provider: string;
  token_map?: Atlas;
  readout_witness?: { result: 'pass'; positions: number; criterion: string; canonical_logits_sha256: string; readout_logits_sha256: string };
  vocabulary?: { method: string; basis: string; vocab_size: number; value_kind?: 'logprob'; rows: VocabularyRow[] };
  heads?: { method: string; basis: string; rows: HeadRow[] };
  spans?: { start: number; end: number; label: string; method: string }[];
  kv?: { sequence: string; method: string; total_bytes: number; key_bytes: number; value_bytes: number;
    content?: { bytes: number; method: string; scope: string; witness: string } };
  experiments?: { site: string; head: number; kind: 'ablate' | 'patch' | 'inject'; parent: string; fork: string;
    method: string; witness: string; target: string; before: Prediction[]; after: Prediction[];
    kl?: { direction: 'before-to-after'; scope: 'full-vocabulary'; nats: number } }[];
}
const fail = (s: string): never => { throw new Error(`Lens evidence refused: ${s}`); };
const text = (s: unknown): s is string => typeof s === 'string' && s.length > 0 && s.length < 4096;
const finite = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);
const integer = (x: unknown, min = 0, max = Number.MAX_SAFE_INTEGER): x is number => finite(x) && Number.isSafeInteger(x) && x >= min && x <= max;
const probability = (x: unknown) => finite(x) && x >= 0 && x <= 1;
const array = (x: unknown, max = 100_000): x is unknown[] => Array.isArray(x) && x.length <= max;
function predictions(rows: Prediction[], vocab?: number, ranked = false, logprob = false) {
  if (!array(rows, 128) || (!logprob && !rows.length)) fail('missing or oversized vocabulary list');
  const ids = new Set<number>(), ranks = new Set<number>();
  for (const [i, r] of rows.entries()) {
    if (!r || !text(r.token) || !integer(r.token_id, 0, vocab ? vocab - 1 : undefined) || ids.has(r.token_id) || !probability(r.probability) || (logprob ? !finite(r.logprob) || r.logprob > 0 || r.logit !== undefined || r.probability !== Math.exp(r.logprob) : !finite(r.logit)) || !integer(r.rank, 1, vocab) || (!logprob && ranks.has(r.rank))) fail('invalid vocabulary entry');
    const expectedRank = logprob && i > 0 && rows[i - 1].logprob === r.logprob ? rows[i - 1].rank : i + 1;
    const increasing = i > 0 && (rows[i - 1].probability < r.probability || (logprob ? rows[i - 1].logprob! < r.logprob! : rows[i - 1].logit! < r.logit!));
    if (ranked && (r.rank !== expectedRank || increasing)) fail('top tokens must be ordered by rank');
    ids.add(r.token_id); ranks.add(r.rank);
  }
  if (rows.reduce((sum, r) => sum + r.probability, 0) > 1.000001) fail('probability mass exceeds one');
}
/** Binding is to exact imported source bytes (including whitespace), not a reserialized object. */
export async function sourceDigest(bytes: string) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(bytes));
  return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
}
export function parseLenses(value: unknown, record: Recording, digest: string): LensRecord {
  if (!value || typeof value !== 'object') fail('expected an analysis sidecar');
  const r = value as LensRecord;
  if (r.schema !== 'larql.observatory.lenses.v1' || r.run_id !== record.id || !/^[a-f0-9]{64}$/.test(r.source_sha256) || r.source_sha256 !== digest) fail('source record identity/hash differs');
  if (r.provenance !== record.provenance || !text(r.provider)) fail('provenance differs from the source record');
  const sites = new Map(record.events.flatMap(e => e.sample ? [[semanticKey(e.sample), e.sample] as const] : []));
  const site = (key: string) => sites.get(key) ?? fail('analysis refers to an unrecorded site');
  if (!r.vocabulary && !r.heads && !r.kv && !r.experiments && !r.spans && !r.token_map) fail('no analysis supplied');
  if (r.vocabulary) {
    const v = r.vocabulary;
    if (!text(v.method) || !text(v.basis) || !integer(v.vocab_size, 2, 10_000_000) || !array(v.rows)) fail('invalid vocabulary method');
    if (v.value_kind !== undefined && v.value_kind !== 'logprob') fail('unsupported vocabulary values');
    const logprob = v.value_kind === 'logprob';
    if (logprob && v.method !== 'head-v1') fail('unsupported log-probability method');
    const seen = new Set<string>();
    for (const row of v.rows) {
      if (!row || seen.has(row.site)) fail('duplicate vocabulary site');
      site(row.site); seen.add(row.site);
      if ((!logprob || row.entropy !== undefined) && (!finite(row.entropy) || row.entropy < 0 || row.entropy > Math.log(v.vocab_size) + 1e-6)) fail('entropy outside full vocabulary bounds');
      if (row.sequence !== undefined && (!/^(0|[1-9][0-9]*)$/.test(row.sequence) || !record.events.some(e => e.sequence === row.sequence && e.sample && semanticKey(e.sample) === row.site))) fail('readout sequence differs from its write');
      predictions(row.top, v.vocab_size, true, logprob); predictions(row.targets, v.vocab_size, false, logprob);
      for (const target of row.targets) {
        const match = row.top.find(t => t.token_id === target.token_id);
        if (match && JSON.stringify(match) !== JSON.stringify(target)) fail('target and top token disagree');
        if (!logprob && target.rank <= row.top.length && !match) fail('target rank is missing from top tokens');
      }
    }
  }
  if (r.token_map) parseAtlas(r.token_map, record);
  if (r.readout_witness) {
    const w = r.readout_witness;
    if (r.provenance !== 'executor' || !r.vocabulary || w.result !== 'pass' || !text(w.criterion) || !integer(w.positions, 1, record.tokens.length) || !/^[a-f0-9]{64}$/.test(w.canonical_logits_sha256) || w.canonical_logits_sha256 !== w.readout_logits_sha256) fail('invalid final readout witness');
    const exits = new Map<number, Sample>();
    for (const event of record.events) if (event.sample) exits.set(event.sample.position, event.sample);
    if (exits.size !== w.positions || [...exits.values()].some(s => !r.vocabulary!.rows.some(row => row.site === semanticKey(s)))) fail('readout witness lacks final position coverage');
  }
  if (r.heads) {
    if (!text(r.heads.method) || !text(r.heads.basis) || !array(r.heads.rows)) fail('invalid head method');
    const seen = new Set<string>();
    for (const h of r.heads.rows) {
      if (!h || site(h.site).role !== 'attention_write' || !integer(h.head, 0, 1023) || !text(h.target) || !finite(h.dla)) fail('invalid head attribution');
      const key = `${h.site}:${h.head}:${h.target}`;
      if (seen.has(key)) fail('duplicate head attribution'); seen.add(key);
      if (!array(h.sources, record.tokens.length)) fail('invalid head sources');
      const positions = new Set<number>();
      for (const s of h.sources) {
        if (!s || !integer(s.position, 0, site(h.site).position) || positions.has(s.position) || !probability(s.weight)) fail('invalid attention source position/weight');
        positions.add(s.position);
      }
      // Sinks and partial capture can leave unallocated mass. Never normalize it away.
      if (h.sources.reduce((sum, s) => sum + s.weight, 0) > 1.000001) fail('attention mass exceeds one');
      if (h.content) {
        const c = h.content;
        if (!text(c.method) || !text(c.basis) || !finite(c.vector_norm) || c.vector_norm < 0 || !array(c.projections, 128)) fail('invalid content projection');
        const ids = new Set<number>();
        for (const p of c.projections) {
          if (!p || !text(p.token) || !integer(p.token_id) || ids.has(p.token_id) || !finite(p.coefficient)) fail('invalid content direction'); ids.add(p.token_id);
        }
        const d = c.dimensionality;
        if (d && (!text(d.method) || !integer(d.dimensions, 1) || !integer(d.d95, 1, d.dimensions) || !integer(d.d99, d.d95, d.dimensions) || !integer(d.d999, d.d99, d.dimensions))) fail('invalid dimensionality thresholds');
        const o = c.orthogonality;
        if (o && (!text(o.method) || !array(o.pairs, 128) || o.pairs.some(p => !p || !text(p.label) || !finite(p.cosine) || Math.abs(p.cosine) > 1))) fail('invalid orthogonality diagnostic');
      }
    }
  }
  if (r.spans && (!array(r.spans, record.tokens.length) || r.spans.some(s => !s || !integer(s.start, 0, record.tokens.length - 1) || !integer(s.end, s.start, record.tokens.length - 1) || !text(s.label) || !text(s.method)))) fail('invalid annotated spans');
  if (r.kv) {
    const k = r.kv;
    if (!record.events.some(e => e.sequence === k.sequence) || !text(k.method) || !integer(k.total_bytes, 1) || !integer(k.key_bytes) || !integer(k.value_bytes) || k.key_bytes + k.value_bytes !== k.total_bytes) fail('invalid KV accounting');
    if (k.content && (!integer(k.content.bytes, 1, k.total_bytes) || !text(k.content.method) || !text(k.content.scope) || !text(k.content.witness))) fail('content estimate requires method, scope and witness');
  }
  if (r.experiments) {
    if (!array(r.experiments, 128)) fail('invalid experiment collection');
    const forks = new Set<string>();
    for (const e of r.experiments) {
      if (!e || site(e.site).role !== 'attention_write' || !integer(e.head, 0, 1023) || !['ablate','patch','inject'].includes(e.kind) || e.parent !== record.id || !text(e.fork) || e.fork === e.parent || forks.has(e.fork) || !text(e.method) || !text(e.witness) || !text(e.target)) fail('invalid experiment identity');
      forks.add(e.fork); predictions(e.before); predictions(e.after);
      if (e.kl && (e.kl.direction !== 'before-to-after' || e.kl.scope !== 'full-vocabulary' || !finite(e.kl.nats) || e.kl.nats < 0)) fail('KL requires full-vocabulary scope and direction');
    }
  }
  return r;
}
export const headKey = (h: HeadRow) => `${h.site}:${h.head}:${h.target}`;
export function availableLenses(r: LensRecord | undefined, samples: Sample[], sequence?: string): LensRecord | undefined {
  if (!r) return undefined;
  const seen = new Set(samples.map(semanticKey));
  return { ...r, token_map: r.token_map && { ...r.token_map, rows: r.token_map.rows.filter(x => seen.has(x.site)) }, vocabulary: r.vocabulary && { ...r.vocabulary, rows: r.vocabulary.rows.filter(x => seen.has(x.site) && (x.sequence === undefined || (sequence !== undefined && BigInt(sequence) >= BigInt(x.sequence)))) }, heads: r.heads && { ...r.heads, rows: r.heads.rows.filter(x => seen.has(x.site)) },
    kv: sequence !== undefined && r.kv && BigInt(sequence) >= BigInt(r.kv.sequence) ? r.kv : undefined,
    experiments: r.experiments?.filter(x => seen.has(x.site)) };
}
export function vocabularyAt(lenses: LensRecord | undefined, sample?: Sample) {
  return sample && lenses?.vocabulary?.rows.find(r => r.site === semanticKey(sample));
}
export const specificity = (entropy: number, vocab: number) => 1 - entropy / Math.log(vocab);
export function distributionDelta(before: Prediction[], after: Prediction[]) {
  const ids = [...new Set([...before, ...after].map(p => p.token_id))];
  return ids.map(id => { const a = before.find(p => p.token_id === id), b = after.find(p => p.token_id === id);
    return { token: (a ?? b)!.token, id, before: a?.probability, after: b?.probability, delta: a && b ? b.probability - a.probability : undefined };
  });
}

/** Equal display labels or absent metadata are not an identity proof. */
export function compatibleReadout(a: Recording, b: Recording) {
  if (a.standard && b.standard) return !!a.probeSource && a.probeSource === b.probeSource && a.standard.probe_method === b.standard.probe_method && a.standard.identity.container === b.standard.identity.container && a.standard.identity.tokenizer === b.standard.identity.tokenizer;
  return !a.standard && !b.standard && !!a.readout && !!b.readout && a.readout.hash === b.readout.hash && a.readout.method === b.readout.method;
}
