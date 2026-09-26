/** OBSERVATORY-V3-BRIDGE-1. The wire contract is larql-inference/vindex3/record.rs.
 * No model math, tokenizer, basis generation, or inferred output lives here.
 * Original JSONL bytes remain the authority; adapted events are a view of them.
 */
import type { Event, Recording, Role, Sample } from './record.ts';
import { reduceEvents, semanticKey } from './record.ts';
import type { LensRecord, Prediction, VocabularyRow } from './lenses.ts';
import { sourceDigest } from './lenses.ts';

export const V3_SCHEMA = 'larql.run-record.v1';
export const HEAD_METHOD = 'head-v1';
type ObjectValue = Record<string, unknown>;
function fail(message: string): never { throw new Error(`VINDEX3 record refused: ${message}`); }
function object(v: unknown, name: string): ObjectValue {
  if (!v || typeof v !== 'object' || Array.isArray(v)) fail(`${name} must be an object.`);
  return v as ObjectValue;
}
function text(v: unknown, name: string): string {
  if (typeof v !== 'string' || !v.length || v.length > 8192) fail(`Invalid ${name}.`);
  return v;
}
function number(v: unknown, name: string, min = -Infinity): number {
  if (typeof v !== 'number' || !Number.isFinite(v) || v < min) fail(`Invalid ${name}.`);
  return v;
}
function integer(v: unknown, name: string, min = 0, max = Number.MAX_SAFE_INTEGER): number {
  const n = number(v, name, min);
  if (!Number.isSafeInteger(n) || n > max) fail(`Invalid ${name}.`);
  return n;
}
function counter(v: unknown, name: string): string {
  const s = typeof v === 'number' && Number.isSafeInteger(v) ? String(v) : v;
  if (typeof s !== 'string' || !/^(0|[1-9][0-9]{0,19})$/.test(s) || BigInt(s) > 18446744073709551615n) fail(`Invalid u64 ${name}.`);
  return s;
}
function array(v: unknown, name: string, max = 100_000): unknown[] {
  if (!Array.isArray(v) || v.length > max) fail(`Invalid ${name}.`);
  return v;
}
function hash(v: unknown, name: string): string {
  const s = text(v, name);
  if (!/^[a-f0-9]{64}$/.test(s)) fail(`Invalid ${name}.`);
  return s;
}
function boolean(v: unknown, name: string): boolean {
  if (typeof v !== 'boolean') fail(`Invalid ${name}.`);
  return v;
}
function role(v: unknown): Role {
  if (v === 'attention') return 'attention_write';
  if (v === 'ffn') return 'ffn_write';
  return fail('Unsupported site identity.');
}
/** Tokenize JSON strings/numbers before parsing: never round a Rust u64 through JS Number.
 * The original unmodified lines, not this parse representation, are hashed and saved.
 */
function parseLine(line: string, index: number): ObjectValue {
  const safe = line.replace(/"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g, token => {
    if (/^-?\d+$/.test(token) && !Number.isSafeInteger(Number(token))) return JSON.stringify(token);
    return token;
  });
  try { return object(JSON.parse(safe), `line ${index + 1}`); }
  catch { return fail(`Malformed JSON on line ${index + 1}.`); }
}

export interface V3Document {
  record: Recording; lenses?: LensRecord; source: unknown;
  bytes: string; digest: string; format: 'jsonl';
}

export async function readVindex3Record(bytes: string): Promise<V3Document> {
  if (new TextEncoder().encode(bytes).length > 10_000_000) fail('Recording exceeds the 10 MB limit.');
  // BufRead::lines in the Rust reader strips CRLF; blank lines are not event lines.
  const lines = bytes.split(/\r?\n/).filter(line => line.trim().length > 0);
  if (lines.length < 3 || lines.length > 100_002) fail('Missing header, events or receipt. An unsealed stream cannot be opened as a complete run.');
  const header = object(parseLine(lines[0], 0).header, 'header');
  const identity = object(header.identity, 'run identity');
  if (identity.schema !== V3_SCHEMA) fail('Unknown recording schema.');
  const runId = text(identity.run_id, 'run id'), model = text(identity.model, 'model identity');
  const component = text(identity.component, 'component');
  const promptTokens = array(identity.tokens, 'prompt tokens', 256).map(v => integer(v, 'token id', 0, 4294967295));
  if (!promptTokens.length) fail('Empty prompt token identity.');
  const started = counter(identity.started_unix_ms, 'start time');
  const fingerprint = hash(header.provenance_fingerprint, 'provenance fingerprint');
  const p = object(header.provenance, 'provenance');
  if (p.norm_method !== 'l2-v1' || p.probe_method !== 'dot-raw-v1') fail('Unsupported norm/probe semantics.');
  const b = object(p.basis, 'recorded basis');
  if (b.dims !== 3) fail('This view requires a recorded 3D basis; other dimensions are not padded or refitted.');
  const basisIdentity = {
    provider: text(b.provider, 'basis provider'), id: text(b.id, 'basis id'),
    hash_hex: hash(b.hash_hex, 'basis hash'), dims: 3, hidden: integer(b.hidden, 'carrier width', 1),
  };
  const probes = p.probe_tokens === null ? null : array(p.probe_tokens, 'probe tokens', 32).map(v => integer(v, 'probe token id', 0, 4294967295));
  if (new Set(probes ?? []).size !== (probes ?? []).length) fail('Duplicate probe token identity.');
  const ex = object(p.execution, 'execution provenance');
  const execution = {
    lowering: array(ex.lowering, 'lowering providers', 64).map(v => text(v, 'lowering provider')),
    realizations: array(ex.realizations, 'realizations', 4096).map(v => {
      const c = object(v, 'realization');
      return { representation: text(c.representation, 'representation'), codec: c.codec === null ? null : text(c.codec, 'codec'), form: text(c.form, 'realization form'), operands: integer(c.operands, 'operands', 1) };
    }),
    arithmetic_arm: text(ex.arithmetic_arm, 'arithmetic arm'), kquant_execution: text(ex.kquant_execution, 'K-quant execution'),
  };
  if (!execution.lowering.length || !execution.realizations.length) fail('Missing execution provenance.');
  // Rust fingerprints RunProvenance in struct declaration order, not serde_json::Value's sorted order.
  const canonical = { execution, basis: basisIdentity, probe_tokens: probes, norm_method: p.norm_method, probe_method: p.probe_method };
  if (await sourceDigest(JSON.stringify(canonical)) !== fingerprint) fail('Provenance fingerprint does not match its recorded fields.');
  const receipt = object(parseLine(lines.at(-1)!, lines.length - 1).receipt, 'terminal receipt (missing or unsealed stream)');
  const complete = boolean(receipt.complete, 'completion');
  const count = integer(receipt.events, 'receipt event count', 1, 100_000);
  const last = counter(receipt.last_sequence, 'last sequence');
  const logHash = hash(receipt.log_sha256, 'event log hash');
  if (receipt.provenance_fingerprint !== fingerprint) fail('Receipt/header provenance disagree.');
  if (count !== lines.length - 2 || BigInt(last) !== BigInt(count - 1)) fail('Receipt count/last sequence disagree with the event log.');
  if (await sourceDigest(lines.slice(1, -1).map(l => l + '\n').join('')) !== logHash) fail('Event log SHA-256 mismatch.');
  const liveDropped = counter(receipt.live_dropped, 'live drop count');
  const passes = counter(receipt.head_passes ?? 0, 'head passes');
  const failure = receipt.lens_failure === null || receipt.lens_failure === undefined ? null : text(receipt.lens_failure, 'lens failure');
  const digest = await sourceDigest(bytes);
  const events: Event[] = [], rows: VocabularyRow[] = [], wire: ObjectValue[] = [];
  const program = new Map<number, Set<Role>>();
  const seenWrites = new Set<string>();
  const vocabularies = new Set<number>();
  const positions: number[] = [];
  const logitsPositions = new Set<number>();
  let pending: Sample | undefined, readout: ObjectValue | undefined;
  let activePosition = -1, entered = false, previousOrder = -1, boundaryOrder = -1, previousTime = -1n;
  let readouts = 0, largestLayer = -1;
  let armedTokens: string | undefined;
  for (let i = 0; i < count; i++) {
    const e = parseLine(lines[i + 1], i + 1); wire.push(e);
    if ('header' in e || 'receipt' in e) fail('Repeated or misplaced envelope.');
    const sequence = counter(e.sequence, 'sequence'), timestamp = counter(e.timestamp_ns, 'timestamp');
    if (BigInt(sequence) !== BigInt(i) || BigInt(timestamp) < previousTime) fail('Noncanonical sequence or timestamp order.');
    previousTime = BigInt(timestamp);
    const position = integer(e.position, 'position', 0, 255);
    const event: Event = { run_id: runId, sequence, timestamp_ns: timestamp, kind: 'SiteCompleted' };
    if (e.kind === 'embedded') {
      if (pending || readout || position !== activePosition + 1) fail('Malformed token traversal.');
      if (i === 0) event.kind = 'RunStarted';
      activePosition = position; positions.push(position); entered = false; previousOrder = -1; boundaryOrder = -1;
    } else {
      if (activePosition < 0 || position !== activePosition) fail('Event position has no matching embedding.');
      if (e.kind === 'entering_carrier') {
        if (entered || pending || previousOrder !== -1 || e.hidden !== basisIdentity.hidden) fail('Invalid entering-carrier identity.');
        entered = true;
      } else if (e.kind === 'carrier_stats') {
        if (!entered || pending || readout || logitsPositions.has(position)) fail('Stats outside an active write.');
        const layer = integer(e.layer, 'layer', 0, 255), site = role(e.site);
        const order = layer * 2 + (site === 'ffn_write' ? 1 : 0);
        if (order <= previousOrder) fail('Duplicate or out-of-order semantic site.');
        const projection = array(e.projection, 'projection', 3).map(v => number(v, 'coordinate'));
        const probe = array(e.probe, 'raw probes', 32).map(v => number(v, 'raw probe'));
        if (projection.length !== 3 || probe.length !== (probes ?? []).length) fail('Basis/probe dimension mismatch.');
        pending = { position, layer, role: site, carrier: 'main', topology: 'residual', norm: number(e.norm, 'carrier norm', 0), delta: number(e.delta_norm, 'write norm', 0), projections: { [basisIdentity.id]: projection as [number, number, number] }, logits: Object.fromEntries((probes ?? []).map((id, j) => [`#${id}`, probe[j]])) };
        if (e.layer_scale !== null) pending.layer_scale = number(e.layer_scale, 'layer scale');
      } else if (e.kind === 'readout') {
        if (!pending || readout || e.layer !== pending.layer || role(e.site) !== pending.role) fail('Lens identity does not match its carrier write.');
        if (e.method !== HEAD_METHOD) fail('Unsupported lens semantics.');
        readout = e; readouts++;
      } else if (e.kind === 'carrier_write') {
        if (e.carrier !== 'single') fail('Unsupported carrier topology; Bundle/History need an explicit adapter.');
        if (!pending || e.layer !== pending.layer || role(e.site) !== pending.role) fail('Carrier write has missing or mismatched stats.');
        const key = semanticKey(pending);
        if (seenWrites.has(key)) fail('Duplicate semantic write.');
        seenWrites.add(key); event.kind = 'CarrierWrite'; event.sample = pending;
        previousOrder = pending.layer * 2 + (pending.role === 'ffn_write' ? 1 : 0);
        largestLayer = Math.max(largestLayer, pending.layer);
        const roles = program.get(pending.layer) ?? new Set<Role>(); roles.add(pending.role); program.set(pending.layer, roles);
        if (readout) {
          const targets = array(readout.tokens, 'lens tokens', 128).map(v => standing(v, true));
          const top = array(readout.top, 'top tokens', 128).map(v => standing(v, false));
          const tokenOrder = JSON.stringify(targets.map(t => t.token_id));
          if (armedTokens !== undefined && armedTokens !== tokenOrder) fail('Lens token identity/order changed during the run.');
          armedTokens = tokenOrder;
          validateStandings(targets, top);
          rows.push({ site: key, targets, top, sequence });
        }
        pending = undefined; readout = undefined;
      } else if (e.kind === 'attention_done' || e.kind === 'ffn_done') {
        if (!entered || pending || readout || logitsPositions.has(position)) fail('Boundary interrupts an unfinished write.');
        const layer = integer(e.layer, 'boundary layer', 0, 255);
        const order = layer * 2 + (e.kind === 'ffn_done' ? 1 : 0);
        if (order <= boundaryOrder || order < previousOrder || layer !== Math.floor(previousOrder / 2)) fail('Malformed structural boundary identity/order.');
        boundaryOrder = order;
        // A boundary is not a write, including FfnDone on a mixer-only layer.
      } else if (e.kind === 'logits') {
        if (!entered || pending || readout || logitsPositions.has(position)) fail('Invalid terminal head boundary.');
        vocabularies.add(integer(e.vocab, 'vocabulary size', 1, 10_000_000)); logitsPositions.add(position);
      } else fail(`Unsupported event kind ${String(e.kind)}.`);
    }
    events.push(event);
  }
  if (vocabularies.size > 1) fail('Vocabulary identity changed during the run.');
  const vocab = [...vocabularies][0];
  if (rows.length && !vocab) fail('Lens has no recorded vocabulary boundary.');
  for (const row of rows) for (const t of [...row.targets, ...row.top]) {
    if (t.token_id >= vocab || t.rank > vocab) fail('Lens token/rank exceeds recorded vocabulary.');
  }
  if (complete && (pending || readout || positions.length < promptTokens.length || (vocab && logitsPositions.size !== positions.length))) fail('Completion contradicts the recorded traversal.');
  if (BigInt(passes) < BigInt(readouts) || (!failure && BigInt(passes) !== BigInt(readouts))) fail('Head-pass receipt contradicts readouts.');
  if (!seenWrites.size) fail('No complete carrier writes to display.');
  if (complete) events.push({ run_id: runId, sequence: String(count), timestamp_ns: events.at(-1)!.timestamp_ns, kind: 'RunCompleted', reason: 'Receipt declares complete' });
  const warnings: string[] = [];
  if (!complete) warnings.push('INCOMPLETE RECORD: receipt declares a prefix. Only fully recorded writes are displayed.');
  if (failure) warnings.push(`LENS FAILED: ${failure}. Later readouts are unavailable.`);
  if (liveDropped !== '0') warnings.push(`${liveDropped} events were dropped by the live tap. This file’s event log is intact; live loss is separate.`);
  const record: Recording = {
    schema: 'larql.observatory.adapted-v3.v1', id: runId, title: model, story: 'VINDEX3 RECORD',
    prompt: 'Prompt text not recorded · token IDs below', tokens: positions.map(position => position < promptTokens.length ? `#${promptTokens[position]}` : `Position ${position} · token ID not recorded`),
    provenance: 'executor', capture: 'standard', model, layers: largestLayer + 1,
    bases: [{ id: basisIdentity.id, provider: basisIdentity.provider, hash: `sha256:${basisIdentity.hash_hex}`, hidden: basisIdentity.hidden, dimensions: 3, source: 'recorded-carrier-post-add-pre-layer-scale', label: basisIdentity.id, axes: ['coordinate 1', 'coordinate 2', 'coordinate 3'] }],
    answers: (probes ?? []).map(id => `#${id}`), attention: 'unavailable', execution,
    program: Array.from({ length: largestLayer + 1 }, (_, layer) => ({ layer, roles: [...(program.get(layer) ?? [])].sort((a, b) => (a === b ? 0 : a === 'attention_write' ? -1 : 1)) })),
    standard: { norm_method: 'l2-v1', probe_method: 'dot-raw-v1', coverage: complete ? 'complete' : 'incomplete', observed_positions: positions, identity: { model, component, lowering: fingerprint, session: runId }, timing_intrusive: true },
    vindex3: { schema: V3_SCHEMA, component, source_sha256: digest, log_sha256: logHash, provenance_fingerprint: fingerprint, prompt_tokens: promptTokens, started_unix_ms: started, events: count, complete, live_dropped: liveDropped, head_passes: passes, lens_failure: failure, lens_method: rows.length ? HEAD_METHOD : undefined, warnings }, events,
  };
  reduceEvents(events);
  const lenses: LensRecord | undefined = rows.length ? {
    schema: 'larql.observatory.lenses.v1', run_id: runId, source_sha256: digest, provenance: 'executor', provider: 'VINDEX3 recorded readouts / OBSERVATORY-V3-BRIDGE-1',
    vocabulary: { method: HEAD_METHOD, basis: 'recorded layer scale → prepared final norm / output head / scaling / softcap → full log-softmax', vocab_size: vocab, value_kind: 'logprob', rows },
  } : undefined;
  freeze(record); if (lenses) freeze(lenses);
  return { record, lenses, source: { header, events: wire, receipt }, bytes, digest, format: 'jsonl' };
}

function standing(value: unknown, ranked: boolean): Prediction {
  const r = object(value, 'token standing');
  const id = integer(r.id, 'lens token id', 0, 4294967295), logprob = number(r.logprob, 'log-probability');
  if (logprob > 0) fail('Log-probability is positive.');
  // This conversion is presentation of a recorded log-probability, never a new softmax.
  return { token_id: id, token: `#${id}`, logprob, probability: Math.exp(logprob), rank: ranked ? integer(r.rank, 'token rank', 1) : 0 };
}
function validateStandings(targets: Prediction[], top: Prediction[]) {
  for (const rows of [targets, top]) {
    if (new Set(rows.map(t => t.token_id)).size !== rows.length || rows.reduce((sum, t) => sum + t.probability, 0) > 1.000001) fail('Duplicate token identity or invalid retained probability mass.');
  }
  for (let i = 0; i < top.length; i++) {
    if (i && (top[i].logprob! > top[i - 1].logprob! || (top[i].logprob === top[i - 1].logprob && top[i].token_id < top[i - 1].token_id))) fail('Top readouts are not in canonical order.');
    // Target ranks use competition ranking (1 + count strictly greater); ties share rank.
    top[i].rank = i && top[i].logprob === top[i - 1].logprob ? top[i - 1].rank : i + 1;
  }
  for (const t of targets) {
    const match = top.find(p => p.token_id === t.token_id);
    if (match && (t.rank !== match.rank || t.logprob !== match.logprob)) fail('Target/top readouts disagree.');
    const greater = top.filter(p => p.logprob! > t.logprob!).length;
    if (t.rank < greater + 1 || (!match && top.length && t.logprob! > top.at(-1)!.logprob!)) fail('Target rank contradicts retained top readouts.');
  }
}

/** Views can select evidence but cannot mutate its coordinates or values. */
function freeze(value: object) {
  Object.freeze(value);
  for (const child of Object.values(value)) if (child && typeof child === 'object') freeze(child);
}
