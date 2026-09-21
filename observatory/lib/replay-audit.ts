/** Checks transport-to-view fidelity. Does not establish execution or output parity. */
import { adaptStandard } from "./standard.ts";
import { reduceEvents } from "./record.ts";
import type { Recording } from "./record.ts";

export interface ReplayAudit {
  schema: "larql.observatory.replay-audit.v1";
  run_id: string;
  provenance: Recording["provenance"];
  coverage: "complete" | "incomplete";
  writes_checked: number;
  events_checked: number;
  duplicate_deliveries: number;
  replay_checkpoints: number;
  source_values: "exact";
  save_reopen: "exact";
  execution_parity: "not_checked";
}

function equal(actual: unknown, expected: unknown, field: string): void {
  if (Object.is(actual, expected)) return;
  if (actual && expected && typeof actual === "object" && typeof expected === "object") {
    const a = Object.entries(actual), b = Object.entries(expected);
    if (a.length === b.length && a.every(([key]) => Object.hasOwn(expected, key))) {
      for (const [key, value] of a) equal(value, (expected as Record<string, unknown>)[key], `${field}.${key}`);
      return;
    }
  }
  throw new Error(`Replay fidelity mismatch: ${field}.`);
}

/** Optional candidate allows regression tests to corrupt the actual view independently. */
export function auditStandard(source: unknown, candidate?: Recording): ReplayAudit {
  const validated = adaptStandard(source);
  const record = candidate ?? validated;
  // Source has passed the adapter's shape validation. The oracle below reads the
  // original rows directly, rather than comparing two copies of adapter output.
  const raw = source as {
    basis: { id: string };
    program: { layer: number; sites: { site: string; operation_id: string }[] }[];
    probe_tokens: { token_id: number; label?: string }[];
    events: { sequence: string; timestamp_ns: string; kind: string; run_id: string;
      text?: string; stats?: { layer: number; position: number; site: string;
        norm: number; delta_norm: number; layer_scale?: number | null; projection: number[]; probe: number[] } }[];
  };
  const unique = [...new Map(raw.events.map(e => [e.sequence, e])).values()];
  equal(record.events.length, unique.length, "event count");
  const state = reduceEvents(record.events);
  let writes = 0;
  for (let i = 0; i < unique.length; i++) {
    const event = unique[i], view = record.events[i];
    for (const key of ["run_id", "sequence", "timestamp_ns", "kind"] as const) equal(view[key], event[key], `event ${i}/${key}`);
    if (event.kind !== "CarrierWrite") {
      equal(view.sample, undefined, `event ${i}/no invented write`);
      continue;
    }
    const row = event.stats!;
    const expected = {
      position: row.position, layer: row.layer,
      role: row.site === "Attention" ? "attention_write" : "ffn_write",
      topology: "residual", carrier: "main",
      operation_id: raw.program[row.layer].sites.find(s => s.site === row.site)!.operation_id,
      norm: row.norm, delta: row.delta_norm,
      projections: { [raw.basis.id]: row.projection },
      logits: Object.fromEntries(raw.probe_tokens.map((p, j) => [`${p.label ?? "token"} [${p.token_id}]`, row.probe[j]])),
      ...(row.layer_scale == null ? {} : { layer_scale: row.layer_scale }),
    };
    equal(view.sample, expected, `write ${event.sequence}`);
    equal(state.samples[writes++], { ...expected, sequence: event.sequence }, `reduced write ${event.sequence}`);
  }
  equal(state.samples.length, writes, "exactly-once write count");

  // Use the same JSON save/open and prefix reducer as the browser, without a
  // model or runner. Check all boundaries on short logs, evenly spaced ones on
  // larger logs. Bounded work avoids a quadratic audit for the 100k-event limit.
  const reopened = adaptStandard(JSON.parse(JSON.stringify(source)));
  equal(reopened, record, "save/reopen record");
  const checkpoints = new Set([0, unique.length]);
  const count = Math.min(32, unique.length);
  for (let i = 1; i < count; i++) checkpoints.add(Math.floor(i * unique.length / count));
  for (const end of checkpoints) {
    const replay = reduceEvents(reopened.events.slice(0, end));
    const prefix = unique.slice(0, end);
    equal(replay.events.length, end, `prefix ${end}/events`);
    equal(replay.samples, state.samples.filter(s => BigInt(s.sequence) <= BigInt(prefix.at(-1)?.sequence ?? "-1")), `prefix ${end}/samples`);
    equal(replay.output, prefix.filter(e => e.kind === "TokenProduced").map(e => e.text).join(""), `prefix ${end}/output`);
    equal(replay, reduceEvents(record.events.slice(0, end)), `prefix ${end}/save/reopen`);
  }
  return {
    schema: "larql.observatory.replay-audit.v1", run_id: record.id,
    provenance: record.provenance, coverage: record.standard!.coverage,
    writes_checked: writes, events_checked: unique.length,
    duplicate_deliveries: raw.events.length - unique.length,
    replay_checkpoints: checkpoints.size, source_values: "exact", save_reopen: "exact",
    execution_parity: "not_checked",
  };
}
