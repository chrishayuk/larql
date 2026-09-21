import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { parseRecording, reduceEvents, samplesAt, compareSamples, stepIndex } from "../lib/record.ts";

const read = async name => parseRecording(JSON.parse(await readFile(new URL(`../public/fixtures/${name}.json`, import.meta.url), "utf8")));
const answer = await read("answer");

test("all four stories and counterfactual target are explicitly synthetic complete records", async () => {
  for (const name of ["answer", "writes", "address", "counterfactual", "counterfactual-target"]) {
    const r = await read(name), state = reduceEvents(r.events);
    assert.equal(r.provenance, "synthetic"); assert.equal(state.status, "completed");
    assert.equal(state.output, r.answers[0]); assert.equal(state.gaps.length, 0);
    assert.equal(state.dropped, 0); assert.ok(state.samples.length > 100);
  }
});
test("JSON round trip and prefix replay preserve coordinates and semantic selection exactly", () => {
  const replay = parseRecording(JSON.parse(JSON.stringify(answer)));
  for (const cursor of [2, 51, 200, answer.events.length - 1]) {
    const original = reduceEvents(answer.events.slice(0, cursor + 1));
    const restored = reduceEvents(replay.events.slice(0, cursor + 1));
    assert.deepEqual(original, restored);
    assert.deepEqual(samplesAt(original, 4), samplesAt(restored, 4));
  }
});
test("reconnect duplicates are idempotent; conflicting identities refuse", () => {
  const events = answer.events.slice(0, 10);
  assert.deepEqual(reduceEvents([...events, events.at(-1)]), reduceEvents(events));
  assert.throws(() => reduceEvents([...events, { ...events.at(-1), timestamp_ns: "99999999" }]), /Conflicting duplicate/);
});
test("permanent end-of-run loss remains visible without a subsequent observation", () => {
  const r = structuredClone(answer);
  const sequence = Number(r.events.at(-1).sequence);
  r.events.splice(-2, 1);
  r.events.splice(-1, 0, { run_id: r.id, sequence: String(sequence - 1), timestamp_ns: "5400000000", kind: "EventDropped", dropped: 1 });
  const state = reduceEvents(parseRecording(r).events);
  assert.equal(state.status, "completed"); assert.equal(state.dropped, 1); assert.equal(state.output, "");
  const gap = reduceEvents(answer.events.filter((_, i) => i !== 20));
  assert.deepEqual(gap.gaps, [{ from: "20", to: "20" }]);
});
test("Standard stats do not imply attention sources, rank, or full-vocabulary probability", () => {
  assert.equal(answer.capture, "standard"); assert.equal(answer.attention, "unavailable");
  for (const e of answer.events) if (e.sample) {
    assert.equal(e.sample.sources, undefined); assert.equal(e.sample.rank, undefined);
    assert.equal(e.sample.probability, undefined);
  }
});
test("readouts and directional writes share the same authored carrier values", () => {
  const state = reduceEvents(answer.events);
  const prior = new Map();
  for (const s of state.samples) {
    const before = prior.get(s.position);
    if (before) for (const token of answer.answers) {
      assert.ok(Math.abs(s.logits[token] - before.logits[token] - s.write_logits[token]) < 2e-7);
    }
    prior.set(s.position, s);
  }
  assert.equal(answer.readout.method, "fixed-linear-rows-no-normalization");
  assert.equal(answer.events.some(e => e.kind === "Intervention"), false);
});
test("source weights are separately labelled synthetic and preserve masked source mass", async () => {
  const r = await read("writes");
  const samples = reduceEvents(r.events).samples.filter(s => s.sources);
  assert.ok(samples.length); assert.equal(r.attention, "synthetic-head-mean");
  for (const s of samples) {
    assert.ok(s.sources.every(a => a.position <= s.position));
    assert.ok(Math.abs(s.sources.reduce((v, a) => v + a.weight, 0) - 1) < 1e-5);
  }
  const bad = structuredClone(r); bad.events.find(e => e.sample?.sources).sample.sources[0].weight = .5;
  assert.throws(() => parseRecording(bad), /mass/);
});
test("comparison aligns semantic sites even when target order and wall-clock timing differ", async () => {
  const a = await read("counterfactual"), b = await read("counterfactual-target");
  const left = samplesAt(reduceEvents(a.events), 2), right = samplesAt(reduceEvents(b.events), 2).reverse();
  const joined = compareSamples(left, right, a.bases[0], b.bases[0]);
  assert.ok(joined.every(pair => pair.base.layer === pair.target.layer));
  assert.ok(joined.filter(pair => pair.base.layer < 20).every(pair => pair.distance === 0));
  assert.ok(joined.filter(pair => pair.base.layer > 22).some(pair => pair.distance > 0));
  assert.throws(() => compareSamples(left, right, a.bases[0], { ...b.bases[0], hash: "sha256:different" }), /basis differs/);
  assert.equal(compareSamples(left, [], a.bases[0], b.bases[0])[0].distance, null);
});
test("a declared mixer-only program has no invented FFN write", () => {
  const r = structuredClone(answer);
  r.program.forEach(p => p.roles = ["attention_write"]);
  r.events = r.events.filter(e => e.sample?.role !== "ffn_write").map((e, i) => ({ ...e, sequence: String(i) }));
  const state = reduceEvents(parseRecording(r).events);
  assert.equal(state.samples.filter(s => s.role === "ffn_write").length, 0);
  assert.equal(samplesAt(state, 4, "attention_write").length, 33);
  assert.equal(samplesAt(state, 4).length, 1); // embedding only; never synthesize FFN.
});
test("site/layer stepping respects the selected token and goes backward as well as forward", () => {
  const first = stepIndex(answer, 1, 4, 1);
  assert.equal(answer.events[first].sample.role, "embedding");
  const attention = stepIndex(answer, first, 4, 1);
  assert.equal(answer.events[attention].sample.role, "attention_write");
  const boundary = stepIndex(answer, first, 4, 1, "ffn_write");
  assert.equal(answer.events[boundary].sample.role, "ffn_write");
  assert.equal(stepIndex(answer, attention, 4, -1), first);
});
test("malformed evidence and incompatible schemas refuse before rendering", () => {
  const r = structuredClone(answer); r.events.find(e => e.sample).sample.norm = NaN;
  assert.throws(() => parseRecording(r), /metrics/);
  assert.throws(() => parseRecording({ ...answer, schema: "vindex3.observation.v1" }), /Unsupported recording/);
  const unlabelled = structuredClone(answer); unlabelled.provenance = "observed";
  assert.throws(() => parseRecording(unlabelled), /Unsupported recording/);
  const badSite = structuredClone(answer); badSite.events.find(e => e.sample).sample.position = 999;
  assert.throws(() => parseRecording(badSite), /semantic site/);
});
