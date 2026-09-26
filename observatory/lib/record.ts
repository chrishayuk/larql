/** UI fixture/replay format. This is NOT the frozen executor ABI. */
export type Role = "embedding" | "attention_write" | "ffn_write";
export type Scene = "Context" | "Map" | "Lenses" | "Trace" | "Graph" | "Matrix" | "Compare";
export interface Basis {
  id: string; hash: string; label: string; axes: [string, string, string];
  source: string; dimensions: 3; provider?: string; hidden?: number;
}
export interface Sample {
  position: number; layer: number; role: Role; topology: "residual";
  carrier: string; before?: number; norm: number; delta: number;
  layer_scale?: number; operation_id?: string;
  projections: Record<string, [number, number, number]>;
  logits: Record<string, number>;
  write_logits?: Record<string, number>;
  sources?: { position: number; weight: number }[];
  duration_ns?: string;
}
export interface Event {
  run_id: string; sequence: string; timestamp_ns: string;
  kind: "RunStarted" | "Tokenized" | "Observation" | "CarrierWrite" | "TokenProduced" | "RunCompleted" | "EventDropped" | "RunRefused" | "SiteCompleted";
  sample?: Sample; output?: string; dropped?: number; reason?: string;
}
export interface Recording {
  schema: "larql.observatory.fixture.v1" | "larql.observatory.adapted-standard.v1" | "larql.observatory.adapted-v3.v1";
  id: string; title: string; story: string; prompt: string; tokens: string[];
  provenance: "synthetic" | "executor"; capture: "standard" | "rich-fixture";
  standard?: { norm_method: string; probe_method: string; coverage: "complete" | "incomplete"; observed_positions: number[]; identity: Record<string, string>; timing_intrusive: boolean; device_readbacks?: number };
  vindex3?: {
    schema: string; component: string; source_sha256: string; log_sha256: string;
    provenance_fingerprint: string; prompt_tokens: number[]; started_unix_ms: string;
    events: number; complete: boolean; live_dropped: string; head_passes: string;
    lens_failure: string | null; lens_method?: string; warnings: string[];
  };
  execution?: { lowering: string[]; arithmetic_arm: string; kquant_execution: string;
    realizations: { representation: string; codec: string | null; form: string; operands: number }[] };
  probeSource?: string;
  model: string; layers: number; bases: Basis[]; answers: string[];
  program: { layer: number; roles: Role[] }[];
  attention: "unavailable" | "synthetic-head-mean";
  readout?: { id: string; hash: string; method: string; rows: Record<string, number[]> };
  events: Event[];
}
export interface RunState {
  events: Event[]; samples: (Sample & { sequence: string })[];
  status: "empty" | "running" | "completed" | "refused"; output: string;
  dropped: number; gaps: { from: string; to: string }[];
}
export const labelLayer = (layer: number) => layer < 0 ? "EMB" : `L${String(layer).padStart(2, "0")}`;
export const labelRole = (role: Role) => ({ embedding: "Embedding", attention_write: "Attention write", ffn_write: "FFN write" })[role];
export const semanticKey = (s: Sample) => `${s.position}:${s.layer}:${s.role}:${s.carrier}:${s.topology}`;
const finite = (n: unknown): n is number => typeof n === "number" && Number.isFinite(n);
const integer = (n: unknown): n is number => finite(n) && Number.isSafeInteger(n);
const counter = (s: unknown): s is string => typeof s === "string" && /^(0|[1-9][0-9]{0,19})$/.test(s);
function fail(message: string): never { throw new Error(message); }

/** Strictly admits this UI's synthetic recordings; live formats need an explicit adapter. */
export function parseRecording(value: unknown): Recording {
  if (!value || typeof value !== "object") fail("This file is not an Observatory recording.");
  const r = value as Recording;
  if (r.schema !== "larql.observatory.fixture.v1" || r.provenance !== "synthetic")
    fail("Unsupported recording schema. This build accepts labelled synthetic fixture records only.");
  if (![r.id, r.title, r.story, r.prompt, r.model].every(v => typeof v === "string" && v.length > 0 && v.length < 8192)) fail("Invalid recording identity.");
  if (!integer(r.layers) || r.layers < 1 || r.layers > 256) fail("Invalid layer count.");
  if (!Array.isArray(r.program) || r.program.length !== r.layers || r.program.some((p, i) => p.layer !== i || !Array.isArray(p.roles) || !p.roles.length || new Set(p.roles).size !== p.roles.length || p.roles.some(role => !["attention_write", "ffn_write"].includes(role)))) fail("Invalid declared fixture program.");
  if (!Array.isArray(r.tokens) || !r.tokens.length || r.tokens.length > 256 || !r.tokens.every(t => typeof t === "string" && t.length < 2048)) fail("Invalid tokens.");
  if (!Array.isArray(r.answers) || !r.answers.length || r.answers.length > 32 || !r.answers.every(t => typeof t === "string" && t.length < 256)) fail("Invalid selected tokens.");
  if (!["standard", "rich-fixture"].includes(r.capture) || !["unavailable", "synthetic-head-mean"].includes(r.attention)) fail("Unknown capture capabilities.");
  if (r.readout && (r.readout.id !== "fixture-linear-reader-v1" || r.readout.method !== "fixed-linear-rows-no-normalization" || !/^sha256:[a-f0-9]{64}$/.test(r.readout.hash) || !r.readout.rows || !r.answers.every(a => Array.isArray(r.readout?.rows[a]) && r.readout.rows[a].length === 6 && r.readout.rows[a].every(finite)))) fail("Invalid readout descriptor.");
  if (!Array.isArray(r.bases) || !r.bases.length || r.bases.length > 8) fail("Missing projection identity.");
  const bases = new Set<string>();
  for (const b of r.bases) {
    if (!b || typeof b.id !== "string" || bases.has(b.id) || !/^sha256:[a-f0-9]{64}$/.test(b.hash) || b.dimensions !== 3 || !Array.isArray(b.axes) || b.axes.length !== 3 || !b.axes.every(a => typeof a === "string") || typeof b.label !== "string" || typeof b.source !== "string") fail("Invalid projection descriptor.");
    bases.add(b.id);
  }
  if (!Array.isArray(r.events) || !r.events.length || r.events.length > 100_000) fail("Invalid event log size.");
  const seen = new Map<string, string>();
  let previous = -1n;
  let time = -1n;
  let terminal = false;
  for (const e of r.events) {
    if (!e || e.run_id !== r.id || !counter(e.sequence) || !counter(e.timestamp_ns)) fail("Invalid event identity.");
    const bytes = JSON.stringify(e);
    if (seen.has(e.sequence)) { if (seen.get(e.sequence) !== bytes) fail("Conflicting duplicate event."); continue; }
    if (terminal || BigInt(e.sequence) <= previous || BigInt(e.timestamp_ns) < time) fail("Events are not in canonical order.");
    previous = BigInt(e.sequence); time = BigInt(e.timestamp_ns); seen.set(e.sequence, bytes);
    if (!["RunStarted", "Tokenized", "Observation", "CarrierWrite", "TokenProduced", "RunCompleted", "EventDropped"].includes(e.kind)) fail("Unknown required event kind.");
    if (e.kind === "RunCompleted") terminal = true;
    if (e.kind === "TokenProduced" && (typeof e.output !== "string" || e.output.length > 8192)) fail("Invalid output token.");
    if (e.kind === "EventDropped" && (!integer(e.dropped) || e.dropped < 1)) fail("Invalid loss report.");
    if (e.sample && e.kind !== "Observation" && e.kind !== "CarrierWrite") fail("Sample attached to a non-observation event.");
    if (e.kind === "Observation" || e.kind === "CarrierWrite") {
      const s = e.sample;
      if (!s || !integer(s.position) || s.position < 0 || s.position >= r.tokens.length || !integer(s.layer) || s.layer < -1 || s.layer >= r.layers || !["embedding", "attention_write", "ffn_write"].includes(s.role) || s.topology !== "residual" || s.carrier !== "main") fail("Invalid semantic site.");
      if ((s.role === "embedding") !== (s.layer === -1)) fail("Embedding has no layer.");
      if (s.layer >= 0 && !r.program[s.layer].roles.includes(s.role)) fail("Observation does not belong to the declared program.");
      if (![s.before, s.norm, s.delta].every(v => finite(v) && v >= 0) || !counter(s.duration_ns)) fail("Invalid metrics.");
      if (!s.projections || !s.logits || typeof s.logits !== "object") fail("Missing recorded readouts.");
      for (const b of r.bases) {
        const p = s.projections[b.id];
        if (!Array.isArray(p) || p.length !== 3 || !p.every(finite)) fail("Invalid recorded projection.");
      }
      if (!r.answers.every(a => finite(s.logits[a]))) fail("Invalid selected logits.");
      if (s.write_logits && (!r.readout || !r.answers.every(a => finite(s.write_logits?.[a])))) fail("Invalid directional write evidence.");
      if (s.sources) {
        if (r.attention !== "synthetic-head-mean" || s.role !== "attention_write" || !Array.isArray(s.sources) || s.sources.length > r.tokens.length) fail("Unsupported attention-source evidence.");
        const positions = new Set<number>();
        let sum = 0;
        for (const a of s.sources) {
          if (!integer(a.position) || a.position < 0 || a.position > s.position || positions.has(a.position) || !finite(a.weight) || a.weight < 0 || a.weight > 1) fail("Invalid attention source.");
          positions.add(a.position); sum += a.weight;
        }
        if (Math.abs(sum - 1) > 0.00001) fail("Attention-source mass does not sum to one.");
      }
    }
  }
  if (r.events[0].kind !== "RunStarted") fail("Recording has no start manifest.");
  return r;
}

/** One reducer for incremental adapter delivery and recorded-prefix replay. */
export function reduceEvents(events: readonly Event[]): RunState {
  const state: RunState = { events: [], samples: [], status: "empty", output: "", dropped: 0, gaps: [] };
  const seen = new Map<string, Event>();
  const sites = new Set<string>();
  let previous = -1n;
  for (const e of events) {
    const id = `${e.run_id}:${e.sequence}`;
    const duplicate = seen.get(id);
    if (duplicate) { if (JSON.stringify(duplicate) !== JSON.stringify(e)) fail("Conflicting duplicate event."); continue; }
    const seq = BigInt(e.sequence);
    if (previous >= 0n && seq <= previous) fail("Out-of-order event; adapter must order delivery.");
    if (seq > previous + 1n) state.gaps.push({ from: String(previous + 1n), to: String(seq - 1n) });
    previous = seq; seen.set(id, e); state.events.push(e);
    if (e.kind === "RunStarted") state.status = "running";
    if (e.sample) {
      const key = semanticKey(e.sample);
      if (sites.has(key)) fail("Duplicate semantic observation.");
      sites.add(key); state.samples.push({ ...e.sample, sequence: e.sequence });
    }
    if (e.kind === "TokenProduced") state.output += e.output;
    if (e.kind === "EventDropped") state.dropped += e.dropped ?? 0;
    if (e.kind === "RunCompleted") state.status = "completed";
    if (e.kind === "RunRefused") state.status = "refused";
  }
  return state;
}

export function samplesAt(state: RunState, position: number, role: Role = "ffn_write") {
  return state.samples.filter(s => s.position === position && (s.role === role || s.role === "embedding"));
}
export function stepIndex(record: Recording, cursor: number, position: number, direction: number, role?: Role) {
  const indices = record.events.map((e, i) => ({ e, i })).filter(({ e }) => e.sample?.position === position && (!role || e.sample.role === role || e.sample.role === "embedding"));
  return (direction > 0 ? indices.find(x => x.i > cursor) : indices.findLast(x => x.i < cursor))?.i;
}
export function compatibleBasis(a: Basis, b: Basis) {
  return a.id === b.id && a.hash === b.hash && a.source === b.source && a.dimensions === b.dimensions && a.provider === b.provider && a.hidden === b.hidden;
}
export function compareSamples(base: Sample[], target: Sample[], basis: Basis, other: Basis) {
  if (!compatibleBasis(basis, other)) fail("Cannot overlay trajectories: projection basis differs.");
  const lookup = new Map(target.map(s => [semanticKey(s), s]));
  return base.map(a => {
    const b = lookup.get(semanticKey(a));
    return { base: a, target: b, distance: b ? Math.hypot(...a.projections[basis.id].map((x, i) => b.projections[basis.id][i] - x)) : null };
  });
}
