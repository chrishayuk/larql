# V3-STREAM-1 — can a run's observations survive the runner boundary without losing their identity?

Pre-registered 2026-09-20, before any implementation. Properties and forecasts below are
FROZEN. Nothing here claims a result.

Programme: OBSERVE. Rung 2 of the runner-side substrate, directly above V3-OBS-1
(`v3-obs-1-carrier-observation.md`, PR #484). The peer contract
(`vindex3-observation-contract.md`) owns event meaning, envelope identity, loss
accounting and transport; this rung freezes the **runner-side record** those need and
nothing about transport or UI.

---

## The question

V3-OBS-1 established that the executor emits truthful observations and can say under
what execution they became true (`RunProvenance`). Both exist only for the duration of
a callback: events are owned values handed to an observer, values are borrowed slices
that expire when the callback returns, and provenance is computed from a prepared image
that the session drops. Nothing persists, nothing is ordered across a run, and nothing
can be replayed.

V3-STREAM-1 asks whether one runner-side consumer can turn a run into a **lossless,
sequenced, provenance-bearing record** that replays to the same events, while a
separate **live tap** may lose events without ever stalling the executor or
contaminating the record.

---

## What already exists (read before implementing)

| Fact | Where |
|---|---|
| Tokenwise observed stepping: `Vindex3Session::step_observed(token, &mut dyn StepObserver)` | `crates/larql-inference/src/vindex3/session.rs` |
| Structural events (`StepEvent`, non-exhaustive, owned) and borrowed taps (`carrier_write`, `entering_carrier`) | `larql-vindex … exec/observe.rs` |
| Stats rows computed at the tap (`StatsObserver`, `WriteStats`, `BasisIdentity`) | `larql-vindex … exec/observe_stats.rs` |
| `ExecutionProvenance::of(&PreparedOperands)`, `RunProvenance`, canonical fingerprints, JSON | `larql-vindex … exec/provenance.rs` |
| The peer contract's envelope: `run_id`, run-scoped `sequence`, run-relative `timestamp_ns`, schema; loss ledger outside the queue; receipt hashes the log, the log never includes its own digest | the Observatory session's observation-contract draft (`vindex3-observation-contract.md`, in that session's working tree and not yet committed) §4, §9, §12 |
| `larql-inference` already depends on `serde_json`; `tempfile` is a dev-dependency | `crates/larql-inference/Cargo.toml` |

The batch prefill (`execute_streaming`) fires plane events, not step events, and is
out of scope: a recorded run is a tokenwise run and says so in its identity.

---

## Contract properties (frozen as properties, not names)

**R1 — one consumer, one sequence.** The recorder is a single `StepObserver` that sees
every structural event and every tap in execution order and assigns each recorded
event one run-scoped sequence number, strictly increasing by one from zero. Sequence is
assigned **before** any fan-out, so a dropped live event still has an identity.

**R2 — lossless by construction.** The recorder appends into memory it owns and never
into a bounded queue; it has no path that discards an event. An event variant it does
not know how to spell (the enum is non-exhaustive) is recorded as an `Unknown` event
carrying the executor's own debug spelling, never skipped.

**R3 — values are recorded at the tap, not reconstructed later.** The borrowed carrier
vectors expire with the callback, so the record carries the stats rows the
`StatsObserver` computes at the tap (norms, projection, probe), each as an event with
its own sequence number. No tensor is persisted by this rung.

**R4 — identity and provenance are on the record, not on the executor.** The record
header carries a caller-supplied `run_id`, the container's self-declared model name,
the component, the prompt token ids, the schema name, and the `RunProvenance` from the
session's own prepared image. The executor allocates none of these.

**R5 — timestamps are run-relative and monotonic.** Each event carries nanoseconds
since the recorder was armed, from a monotonic clock, non-decreasing across the record.
Wall-clock start time is header metadata, never an ordering key.

**R6 — the live tap is a second consumer with its own policy.** It receives the same
sequenced events through a bounded, non-blocking channel. When the channel is full the
event is dropped and counted in a ledger that lives outside the channel; the recorder
is unaffected. The tap can never block the executor.

**R7 — the receipt describes the log without being in it.** The receipt carries the
event count, the last sequence, the provenance fingerprint, a SHA-256 over the exact
bytes of the event lines, and whether the run completed. It is written after the event
lines and is not covered by its own hash.

**R8 — replay is the same events.** Reading a written record yields a record equal to
the one written, header, events and receipt included, with every float bit-exact.

---

## Frozen acceptance properties

**S1 — lossless.** For a run of N tokens on the golden plan, the recorder holds exactly
the events a plain `RecordingObserver` sees plus one stats event per carrier write plus
one entering-carrier event per position; no run length changes that equality.

**S2 — replay identity.** `read(write(record)) == record`, including every `f32`/`f64`
in every stats row, checked by bits.

**S3 — sequencing.** Sequences are `0..n` with no gap; timestamps are non-decreasing.

**S4 — live loss is bounded, counted and isolated.** With a tap of capacity `c` and no
consumer, a run producing `n > c` events completes; the tap holds exactly `c`, the drop
ledger reports exactly `n − c` with the first and last dropped sequence, and the
recorder holds all `n`. Repeated with a draining consumer: zero drops.

**S5 — observation parity through the recorder.** Logits at every position with the
recorder, stats observer and live tap attached are bit-identical to the unobserved
run. V3-OBS-1's P1, re-established for this consumer.

**S6 — receipt integrity.** The receipt's hash equals a recomputation over the written
event lines; flipping one byte of one event line is detected; the receipt line is not
part of the hashed bytes.

**S7 — provenance rides along.** The record's provenance equals
`RunProvenance::new(ExecutionProvenance::of(&image), Some(&stats))` for the session's
image, and its fingerprint is in the receipt.

**S8 — real subject.** S1–S7 on Granite 4.2 3B (production CPU, `.s6` container), 8
tokens, with the record written to disk and read back; report the record size and the
event count. Not a performance claim.

---

## Forecasts

- **F1**: S1's equality holds with N = 5 on the golden plan (2 layers): per position
  1 embedded + 1 entering + 2 layers × (write event + stats event + boundary) × 2 sites
  + 1 logits = 1 + 1 + 2 × 2 × 3 + 1 = 15 events; 75 for five positions.
- **F2**: on Granite, 8 tokens: per position 1 + 1 + 40 × 2 × 3 + 1 = 243 events; 1944
  for the run; the record is under 2 MB as JSON lines with a 3-dimensional projection.
- **F3**: S4 with `c = 8` and the golden run of 75 events: tap 8, dropped 67, first
  dropped sequence 8, last 74.
- **F4**: S5 holds on both CPU backends on the golden plan and on Granite production.
- **F5**: no change to `larql-vindex` is needed. If one turns out to be, it is a finding
  about the OBS contract and is recorded as such, not folded silently into this rung.

---

## Amendments (recorded before any result, from the witnesses themselves)

- **2026-09-20, envelope versus payload.** The first schema carried `position` on the
  event envelope and again inside the `embedded` and `entering_carrier` payloads; the
  flattened serialisation then wrote one key twice and the record could not be read
  back at all. The payloads no longer carry a position: the envelope's is the position.
  A wire schema must have one spelling of every fact.
- **2026-09-20, float parsing.** The writer emits the shortest representation that
  round-trips, but `serde_json`'s default parser is not correctly rounded, and one
  carrier norm came back one ulp off. S2 demands bit-exact floats, so the crate enables
  the parser's `float_roundtrip` feature rather than weaken the property. Any other
  consumer of the record needs a correctly-rounded parser to make the same claim.

## Out of scope

Transport of any kind (WebSocket, SSE, relay), the server run registry, export policy
and redaction, payload references, backfill, browser state, interventions, per-head
capture, Metal instrumentation, batch-prefill recording, the Observatory UI.

---

## Results (recorded 2026-09-20, after the forecasts above were frozen)

Implementation: `crates/larql-inference/src/vindex3/record.rs` (`RunIdentity`,
`EventKind`, `RecordedEvent`, `RunRecorder`, `LiveTap`, `DropLedger`, `Receipt`,
`RunRecord` with `write_jsonl` / `read_jsonl`); witnesses in
`crates/larql-inference/src/vindex3/tests/record.rs`. No change to `larql-vindex` (F5
held). Gates: fmt, clippy, larql-inference lib 1542 passed.

| Property | Result |
|---|---|
| S1 lossless | PASS — 75 events on the golden run = plain events + one stats event per write + one entering event per position; F1's 15 per position held |
| S2 replay identity | PASS — after the two amendments above; every float by bits |
| S3 sequencing | PASS — gapless `0..n`, timestamps and positions non-decreasing, position stamped per step |
| S4 live loss | PASS — tap 8 on 75 events: delivered 8, dropped 67, first 8, last 74 (F3 held); a never-full tap drops nothing and delivers every event equal to the record; a disconnected consumer counts as loss, not a stall |
| S5 parity through the recorder | PASS on the reference and production backends |
| S6 receipt integrity | PASS — hash recomputes; a one-key tamper, a missing receipt, a missing header, a removed event line and garbage are each refused with their own error |
| S7 provenance | PASS — equals `RunProvenance::new(ExecutionProvenance::of(&image), Some(&stats))`, fingerprint on the record and the receipt |
| S8 Granite 4.2 3B (production CPU, 8 tokens, release) | PASS — 1944 events, 243 per token (F2 held); record 294,289 bytes (F2's bound held); replay equal; parity held; live tap of 8 delivered 8 and dropped 1936, all counted |

## Verdict rule

V3-STREAM-1 is complete only when S1–S7 PASS on the golden plan and S8 PASS on Granite.
It unlocks the transport rung (the peer contract's §10) and the Observatory's replay.
