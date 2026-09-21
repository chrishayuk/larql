/** Draft runner-to-UI bridge. Mirrors WriteStats, not a new executor ABI. */
import { parseRecording, reduceEvents } from "./record.ts";
import type { Event, Recording, Role, Sample } from "./record.ts";

export const STANDARD_SCHEMA = "larql.observatory.standard.v1";
type ObjectValue = Record<string, unknown>;
function object(value: unknown, name: string): ObjectValue {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${name} must be an object.`);
  return value as ObjectValue;
}
function text(value: unknown, name: string, max = 8192): string {
  if (typeof value !== "string" || !value.length || value.length > max) throw new Error(`Invalid ${name}.`);
  return value;
}
function number(value: unknown, name: string, minimum = -Infinity): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum) throw new Error(`Invalid ${name}.`);
  return value;
}
function integer(value: unknown, name: string, minimum = 0, maximum = Number.MAX_SAFE_INTEGER): number {
  const n = number(value, name, minimum);
  if (!Number.isSafeInteger(n) || n > maximum) throw new Error(`Invalid ${name}.`);
  return n;
}
function counter(value: unknown, name: string): string {
  if (typeof value !== "string" || !/^(0|[1-9][0-9]{0,19})$/.test(value) || BigInt(value) > 18446744073709551615n) throw new Error(`Invalid ${name}; use a decimal u64 string.`);
  return value;
}
function list(value: unknown, name: string, max: number): unknown[] {
  if (!Array.isArray(value) || value.length > max) throw new Error(`Invalid ${name}.`);
  return value;
}
function hash(value: unknown, name: string): string {
  const s = text(value, name);
  if (!/^[a-f0-9]{64}$/.test(s)) throw new Error(`Invalid ${name}; expected lowercase SHA-256 hex.`);
  return s;
}
function boolean(value: unknown, name: string): boolean {
  if (typeof value !== "boolean") throw new Error(`Invalid ${name}.`);
  return value;
}
function role(value: unknown): Role {
  if (value === "Attention") return "attention_write";
  if (value === "Ffn") return "ffn_write";
  throw new Error("Unsupported SublayerSite; expected Attention or Ffn.");
}

/** Never derives missing values, remaps model geometry, or manufactures timestamps. */
export function adaptStandard(value: unknown): Recording {
  const r = object(value, "record");
  if (r.schema !== STANDARD_SCHEMA) throw new Error("Unsupported Standard record schema.");
  if (r.provenance !== "executor" && r.provenance !== "synthetic") throw new Error("Missing source provenance.");
  if (r.capture !== "standard" || r.topology !== "Single") throw new Error("This adapter accepts Standard Single carrier stats only; Bundle/History require their own topology adapter.");
  if (r.norm_method !== "l2-v1" || r.probe_method !== "dot-raw-v1") throw new Error("Unsupported norm/probe method; raw probes are not normalized logits.");
  const id = text(r.run_id, "run id");
  const model = text(r.model, "model");
  const authority = object(r.identity, "execution identity");
  const identity = Object.fromEntries(["container", "plan", "lowering", "tokenizer", "session"].map(k => [k, text(authority[k], `${k} authority`)]));
  const layers = integer(r.layers, "layer count", 1, 256);
  const program = list(r.program, "program", layers).map((p, i) => {
    const node = object(p, "program layer");
    if (node.layer !== i) throw new Error("Program layers must be ordered and contiguous.");
    const sites = list(node.sites, "program sites", 2).map(s => {
      const site = object(s, "program site");
      return { role: role(site.site), operation_id: text(site.operation_id, "operation id") };
    });
    if (!sites.length || new Set(sites.map(s => s.role)).size !== sites.length) throw new Error("Duplicate or empty declared sites.");
    return { layer: i, sites };
  });
  if (program.length !== layers) throw new Error("Program layer count differs from manifest.");
  const operations = program.flatMap(p => p.sites.map(s => s.operation_id));
  if (new Set(operations).size !== operations.length) throw new Error("Operation IDs must be unique.");
  const tokens = list(r.tokens, "tokens", 256).map((t, i) => {
    const token = object(t, "token");
    if (token.position !== i) throw new Error("Token positions must be contiguous.");
    const tokenId = integer(token.token_id, "token id", 0, 4294967295);
    return token.label === undefined ? `#${tokenId}` : text(token.label, "token label", 2048);
  });
  if (!tokens.length) throw new Error("Token manifest is empty.");
  const observed = list(r.observed_positions, "observed positions", tokens.length).map(p => integer(p, "observed position", 0, tokens.length - 1));
  if (!observed.length || new Set(observed).size !== observed.length) throw new Error("Invalid observed position scope.");
  const probes = list(r.probe_tokens, "probe tokens", 32).map(t => {
    const token = object(t, "probe token"), id = integer(token.token_id, "probe token id", 0, 4294967295);
    return { id, label: `${token.label === undefined ? "token" : text(token.label, "probe label", 256)} [${id}]` };
  });
  if (new Set(probes.map(p => p.id)).size !== probes.length) throw new Error("Duplicate probe token ID.");
  const b = object(r.basis, "basis");
  if (b.dims !== 3) throw new Error("Map currently requires a recorded 3D basis; coordinates will not be padded or refitted.");
  const basis = { id: text(b.id, "basis id"), hash: `sha256:${hash(b.hash_hex, "basis hash")}`, provider: text(b.provider, "basis provider"), hidden: integer(b.hidden, "source width", 1), dimensions: 3 as const, source: text(b.source, "basis source"), label: text(b.id, "basis id"), axes: ["coordinate 1", "coordinate 2", "coordinate 3"] as [string, string, string] };
  const capture = object(r.runtime, "capture runtime");
  let execution: Recording["execution"];
  if (r.run_provenance !== undefined) {
    const p = object(r.run_provenance, "run provenance");
    const pb = object(p.basis, "provenance basis");
    if (p.norm_method !== r.norm_method || p.probe_method !== r.probe_method ||
      ["provider", "id", "hash_hex", "dims", "hidden"].some(k => pb[k] !== b[k])) throw new Error("Run provenance disagrees with capture identity.");
    if (JSON.stringify(p.probe_tokens ?? []) !== JSON.stringify(probes.map(p => p.id))) throw new Error("Run provenance disagrees with probe order.");
    const e = object(p.execution, "execution provenance");
    execution = {
      lowering: list(e.lowering, "lowering providers", 64).map(v => text(v, "lowering provider")),
      arithmetic_arm: text(e.arithmetic_arm, "arithmetic arm"),
      kquant_execution: text(e.kquant_execution, "K-quant execution"),
      realizations: list(e.realizations, "realization classes", 4096).map(v => {
        const c = object(v, "realization class");
        return { representation: text(c.representation, "representation"), codec: c.codec === null ? null : text(c.codec, "codec"), form: text(c.form, "physical form"), operands: integer(c.operands, "operand count", 1) };
      }),
    };
    if (!execution.lowering.length || !execution.realizations.length) throw new Error("Execution provenance must name providers and realizations.");
  }
  const standard = { norm_method: "l2-v1", probe_method: "dot-raw-v1", coverage: r.coverage as "complete" | "incomplete", observed_positions: observed, identity, timing_intrusive: boolean(capture.timing_intrusive, "timing intrusion"), device_readbacks: integer(capture.device_readbacks, "device readbacks") };
  if (!["complete", "incomplete"].includes(standard.coverage)) throw new Error("Record must declare coverage.");
  const seen = new Map<string, string>();
  let prior = -1n, time = -1n, terminal = false;
  const events: Event[] = [];
  const positionProgress = new Map<number, number>();
  for (const raw of list(r.events, "events", 100_000)) {
    const e = object(raw, "event"), sequence = counter(e.sequence, "sequence"), timestamp_ns = counter(e.timestamp_ns, "timestamp");
    if (e.run_id !== id) throw new Error("Event run identity differs from manifest.");
    const bytes = JSON.stringify(e);
    if (seen.has(sequence)) { if (seen.get(sequence) !== bytes) throw new Error("Conflicting duplicate event."); continue; }
    if (terminal || BigInt(sequence) <= prior || BigInt(timestamp_ns) < time) throw new Error("Events are not in canonical order.");
    seen.set(sequence, bytes); prior = BigInt(sequence); time = BigInt(timestamp_ns);
    const event: Event = { run_id: id, sequence, timestamp_ns, kind: e.kind as Event["kind"] };
    if (events.length === 0 && (e.kind !== "RunStarted" || sequence !== "0")) throw new Error("Record must start at RunStarted / sequence 0.");
    if (events.length > 0 && e.kind === "RunStarted") throw new Error("Repeated start manifest.");
    if (e.kind === "CarrierWrite") {
      const row = object(e.stats, "WriteStats");
      const position = integer(row.position, "position", 0, tokens.length - 1), layer = integer(row.layer, "layer", 0, layers - 1), site = role(row.site);
      const op = program[layer].sites.find(s => s.role === site);
      if (!op || !observed.includes(position)) throw new Error("Write is outside the declared program/capture scope.");
      const projection = list(row.projection, "projection", 3).map(v => number(v, "coordinate"));
      const probe = list(row.probe, "probe", probes.length).map(v => number(v, "raw probe"));
      if (projection.length !== 3 || probe.length !== probes.length) throw new Error("Projection/probe dimensionality mismatch.");
      const order = operations.indexOf(op.operation_id);
      if (order <= (positionProgress.get(position) ?? -1)) throw new Error("Duplicate semantic write or noncanonical per-position program order.");
      positionProgress.set(position, order);
      const sample: Sample = { position, layer, role: site, topology: "residual", carrier: "main", operation_id: op.operation_id, norm: number(row.norm, "norm", 0), delta: number(row.delta_norm, "delta norm", 0), projections: { [basis.id]: projection as [number, number, number] }, logits: Object.fromEntries(probes.map((p, i) => [p.label, probe[i]])) };
      if (row.layer_scale !== null && row.layer_scale !== undefined) sample.layer_scale = number(row.layer_scale, "layer scale");
      // StatsObserver exposes post-add/pre-layer-scale after, not the scaled boundary.
      // It supplies neither entering-carrier norm, directional write probe, nor timing.
      event.sample = sample;
    } else if (e.kind === "TokenProduced") {
      event.output = text(e.text, "output text");
      integer(e.token_id, "produced token id", 0, 4294967295);
      integer(e.predicting_position, "predicting position", 0, tokens.length - 1);
    } else if (e.kind === "EventDropped") {
      event.dropped = integer(e.count, "dropped count", 1);
    } else if (e.kind === "RunCompleted" || e.kind === "RunRefused") {
      terminal = true;
      event.reason = text(e.reason, "terminal reason");
    } else if (e.kind === "SiteCompleted") {
      // FfnDone is a boundary on mixer-only layers too. Never turn it into a write.
      integer(e.position, "boundary position", 0, tokens.length - 1);
      integer(e.layer, "boundary layer", 0, layers - 1);
      if (e.site !== "AttentionDone" && e.site !== "FfnDone") throw new Error("Unsupported structural boundary.");
    } else if (e.kind !== "RunStarted" && e.kind !== "Tokenized") throw new Error("Unsupported Standard event kind.");
    events.push(event);
  }
  if (!events.length) throw new Error("Empty event record.");
  const state = reduceEvents(events); // Reject duplicate semantic writes before mounting the UI.
  if (standard.coverage === "complete") {
    if (state.status !== "completed" || state.gaps.length || state.dropped) throw new Error("Complete coverage contradicts terminal/loss evidence.");
    const expected = observed.length * operations.length;
    if (state.samples.length !== expected) throw new Error(`Complete coverage needs ${expected} declared writes; found ${state.samples.length}.`);
  }
  return { schema: "larql.observatory.adapted-standard.v1", id, title: model, story: "STANDARD CAPTURE", prompt: typeof r.prompt === "string" ? r.prompt : "Prompt not exported", tokens, provenance: r.provenance, capture: "standard", model, layers, bases: [basis], answers: probes.map(p => p.label), program: program.map(p => ({ layer: p.layer, roles: p.sites.map(s => s.role) })), attention: "unavailable", standard, execution, probeSource: r.probe_source === undefined ? undefined : text(r.probe_source, "probe source"), events };
}

export function openRecording(value: unknown): { record: Recording; source: unknown } {
  const schema = object(value, "record").schema;
  const record = schema === STANDARD_SCHEMA ? adaptStandard(value) : parseRecording(value);
  reduceEvents(record.events);
  return { record, source: value };
}
