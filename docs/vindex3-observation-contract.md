# VINDEX3 observation contract — Observatory draft

Status: design draft, 2026-09-19; no frozen wire ABI, shipped endpoint, or
completed acceptance claim. Companion:
[LARQL Observatory product specification](observatory.md).

This document owns event meaning, identity, capture boundaries, transport,
recording, and evidence completeness. The product document owns visual form
and workflow. Semantic guarantees below are requirements; example field names,
storage choices, and API placement remain proposals until implementation gates
close. This is not an extension of the VINDEX3 container ABI.

The concurrent [V3-OBS-1 preregistration](v3-obs-1-carrier-observation.md)
freezes the first carrier-tap rung. This draft does not alter those properties
or its verdict rule. Per-step executor coordinates map to the run-scoped
envelope below in the adapter; the executor does not allocate run identities.
The anticipated stats/projection rung must reconcile its concrete schema
with this draft before any wire contract is frozen.


UI/executor ownership update: the UI is fixture-first and owns no executor
changes. Standard maps to V3-OBS-1's stats observer: carrier norm, applied-write
norm, fixed projection and predeclared selected-token logits from extracted head
rows. Rich requires a separate per-head rung. The runner adapter owns run ID,
run sequence and monotonic timestamps; structural executor events remain owned
and tensor callbacks borrowed. Live delivery may be lossy/nonblocking; a replay
record must be lossless within its admitted scope or explicitly fail completeness.
The amended V3-OBS-1 freeze names Granite as the primary witness and OLMo2-1B as secondary; Gemma 3 4B and Granite 4.2 3B
are initial UI test beds. None of those real-model gates are asserted by fixtures.

The frontend's `larql.observatory.fixture.v1` file format is an explicitly
synthetic UI adapter format, not this draft's wire ABI. MODEL, EXECUTION and
ALTERNATIVES are lenses over shared object/site identities. Future browse/walk,
representation, expert, WASM and parity events require runner capabilities and
source evidence; their names are extension points, not executor requirements
of V3-OBS-1. Preserve a small initial event vocabulary.

## 1. Execution authority and current seams

The [VINDEX3 runtime](vindex3-runtime.md) executes the container's program.
Observation subscribes to that execution. A transport or frontend must not
recreate its semantics from architecture names or issue another forward pass
to obtain missing observations.

Source inspection at this revision establishes:

| Existing seam | What exists | Observatory gap |
|---|---|---|
| [StepObserver / StepEvent](../crates/larql-vindex/src/format/vindex3/opplan/exec/observe.rs) | Embedded position, attention/FFN completion, logits vocabulary size; borrowed operand inputs and topology-specific records | General carrier/write values, stable operation identity, site entry, metrics/projections, and transport envelopes |
| [DecodeSession](../crates/larql-vindex/src/format/vindex3/opplan/exec/decode.rs) | `step` calls `step_observed` with a no-op observer | Bounded observation work and richer taps need witnesses on each supported path |
| [Observation tests](../crates/larql-vindex/src/format/vindex3/opplan/exec/tests/observe.rs) | Fixture logits agree between observed/plain steps; structural events follow layer order | Real-model, capture-profile, continuation, and backend-specific acceptance |
| [Vindex3Session](../crates/larql-inference/src/vindex3/session.rs) | Exposes observed stepping; tokenwise prefill uses ordinary steps | Run/session adapter carrying observation through actual prompt ingestion and generation |
| [Vindex3Runtime](../crates/larql-inference/src/vindex3/runtime.rs) | `prefill_into` preserves KV; `execute_streaming` exposes plane events for analysis without continuation state | Observed batch prefill must preserve the actual generation KV handoff |
| [Executor plane events](../crates/larql-vindex/src/format/vindex3/opplan/exec/mod.rs) | Embedded/layer traces and topology-specific events | One semantic vocabulary across batch and step execution; no second traversal |
| [V3 serving](../crates/larql-server/src/vindex3.rs) | Canonical prefill plus generation and token callback | Run registry, observation adapter, bounded recording and fan-out |
| [Existing WebSocket route](../crates/larql-server/src/routes/stream.rs) | `/v1/stream` describe/infer/generate protocol | Not the proposed observation protocol; preserve existing client behavior |

`AttentionDone` includes the residual add; it is not an attention write.
`InputSite::Attention` and `InputSite::Ffn` are normalized operator inputs,
not carrier boundaries. `FfnOutput` is before possible post-norm/scaling;
it must not be mislabeled as the applied carrier delta. `StepEvent::Logits`
carries a vocabulary size, not logit values; the session returns the logits.

The existing `execute_streaming` analysis entry point cannot be run before
generation to manufacture its visualization: it does not retain generation
state and would create the forbidden extra pass. A first tokenwise runner
may be valid if it uses one canonical session for prompt and generation,
declares its execution profile, and compares observation on/off within that
profile. It must not claim the timing of the batch-prefill serving path.

## 2. Ownership and one-way dependencies

```text
canonical VINDEX3 executor
  borrowed semantic observations
          |
          v
runner adapter: capture policy, identity, reductions, projection
          |
   bounded nonblocking admission
          |
          v
recorder + independent subscriber fan-out
          |
          +-- local WebSocket -----+
          +-- outbound WSS relay --+--> one client reducer --> all views
          +-- recorded log --------+
```

Executor taps belong beside canonical operations in `larql-vindex`.
Run/tokenizer/sampling orchestration belongs above them in `larql-inference`;
serving/relay handling belongs in the server/runner layer. The CLI stays a
thin dispatcher. Reusable CPU arithmetic follows the substrate rules in
`larql-compute`; Metal kernels remain in their peer crate. No executor
dependency on browser, WebSocket, HAUSE, or server types is introduced.

A dependency-light observation schema module/crate is a possible sharing
boundary, not a decision made by this draft. Do not place runtime observation
types in `larql-vindex-spec` merely because it is already a contract leaf;
that crate has an on-disk manifest responsibility.

Borrowed values expire with the callback. The adapter reduces them or copies
only explicitly admitted payloads within fixed budgets; it cannot hand a
borrowed slice to a background worker after its lifetime. No observer may
mutate activations, KV, sampler state, operands, or execution ordering.

## 3. Negotiation and capture manifest

A session handshake declares protocol version, runner identity, permitted
model handles, command grants, export scope, recording location, and supported
capture capabilities. A run freezes its effective manifest before execution:

* Model/container/component and plan identities; runtime build, backend,
  realization/precision and KV policy; tokenizer/template identity.
* Effective prompt token sequence locally, generation parameters and seed,
  cache/prefix reuse, and initial continuation-state provenance.
* Capture profile, positions/sites, metrics, payload quota, projection/readout
  descriptors, timing method and instrumentation disclosure.
* Queue/retention limits, export policy, schema version, and coverage policy.

Unsupported required capabilities refuse the run before execution. Optional
fields are explicitly unavailable. Do not silently promote Standard to a
device-readback profile. Reused prompt state has no newly executed prompt
sites; mark that coverage or disable reuse before the run when a full prefill
observation is required. Never replay the cached prefix solely for a picture.

## 4. Identity and ordering

Each canonical event has `run_id`, run-scoped monotonically increasing
`sequence`, run-relative monotonic `timestamp_ns`, schema version, and kind.
Assign sequence before bounded admission, so omitted events retain detectable
identities. Timestamps describe execution/capture time, not network arrival.
Use a documented clock origin; wall-clock start time is separate metadata.

Wire sequences and nanosecond counters use decimal strings (or an equivalently
lossless negotiated encoding), not JavaScript numbers with unsafe integer
precision. Duplicate identity with identical content is idempotent; duplicate
identity with different content is corruption and cannot overwrite evidence.

Sequence orders one run. It is not a cross-run alignment key. The semantic
key comprises component, phase, absolute input position, layer when applicable,
operation/site role, carrier/history identity, and invocation ordinal when a
site repeats. Stable operation IDs are derived from the admitted plan; they
are not worker addresses or callback order. Cross-plan mapping is explicit.

Tokenization records input IDs and absolute positions. Token production
records output ordinal, sampled ID, and the position/logit event that predicted
it. The final prompt position predicts the first generated token; the final
emitted token need not itself be stepped when generation stops. Do not invent
a depth trajectory for an unexecuted token.

Batched prefill events identify their actual position range. Row observations
can expand to per-position identities without assigning fictitious per-position
execution times. Embedding/output have no fake layer number. A skipped semantic
site is not a dropped event; its absence follows plan/capability metadata.

## 5. Event vocabulary

| Kind | Meaning |
|---|---|
| `run_started` | Run and effective capture manifest admitted |
| `tokenized` | Actual input positions/IDs and permitted labels |
| `site_entered` | Canonical semantic operation begins, where supported |
| `observation` | Values or explicitly derived measurements at a named site |
| `site_left` | Semantic operation completes; timing method identifies what was measured |
| `token_produced` | Actual sampler output linked to predicting position |
| `event_dropped` | Declared loss summary with origin, sequence coverage, and count |
| `run_completed` | Successful generation terminal state and stop reason |
| `run_refused` | Canonical refusal, including admission failure |
| `run_failed` / `run_cancelled` | Execution error or explicitly authorized cancellation |

These names are proposed. Entry/completion cannot be backdated from a coarse
completion callback to manufacture an operation duration. Map canonical
refusal semantics from [larql-execution](../crates/larql-execution/src/lib.rs);
transport disconnection alone is not a model refusal or cancellation.

## 6. Observation envelope

Illustrative shape, not a fixture or a claim of captured values:

```json
{
  "schema": "vindex3.observation.v1",
  "kind": "observation",
  "run_id": "example-run",
  "sequence": "184",
  "timestamp_ns": "19281733",
  "phase": "prefill",
  "position": 4,
  "token_id": 9182,
  "site": {
    "component": "decoder",
    "operation_id": "plan-local-operation-id",
    "layer": 24,
    "role": "carrier_after_attention",
    "topology": "residual",
    "carrier_id": "main",
    "invocation": 0
  },
  "observation": {
    "source": "observed",
    "metrics": {
      "norm": {"value": 31284.2, "method": "l2-v1"},
      "delta_norm": {
        "value": 8921.3,
        "method": "l2-difference-v1",
        "reference_sequence": "180"
      }
    }
  },
  "projection": {
    "evidence": "projected",
    "descriptor_ref": "projection-1",
    "basis_id": "example-fixed-basis",
    "basis_hash": "sha256:<actual-basis-digest>",
    "values": [0.61, -0.22, 1.07]
  },
  "logits": {"availability": "not_captured"},
  "runtime": {
    "duration_ns": "823000",
    "capture_cost_ns": "11000",
    "timing_method": "host-monotonic-inclusive",
    "device_readbacks": 0
  },
  "payload_ref": null
}
```

Keep identity, semantic site, observation, projection, runtime, and payload
reference separate. METRICS exports omit `token_id`; redaction applies to all
event kinds and descriptors, not just this example. Optional unavailable
fields carry reason codes; malformed/nonfinite values cannot silently become
zero or JSON null. Preserve dtype and exact round-trippable numeric values
for coordinates; the eventual codec must pass the exact replay witness.

Evidence descriptors distinguish measured input, deterministic derivation,
projection, attribution, and intervention-supported claim. A norm is a
reduction of an observed vector and names its formula. A causal claim additionally
references intervention/control evidence; an ordinary observation event cannot
grant that status by naming a field `causal`.

## 7. Semantic roles and topology

Distinguish embedding, carrier before/after sublayer, normalized operator
input, branch output before post-processing, applied attention/FFN write,
block output, and final-head input/logits. Actual roles depend on the plan.
Capture the point that exists; do not reconstruct a branch write by subtracting
carrier states unless labelled derived and justified for that topology.

`FfnDone` is a layer boundary even on a mixer-only layer with no FFN write.
Where a `LayerScalar` applies, the post-add FFN carrier and scaled layer output
are distinct states. Preserve the post-add value and scale per V3-OBS-1;
do not relabel the unscaled value as the next layer's input. Neither uniform
two-write rows nor unconditional attention/FFN pairs describe every plan.

Residual, hyper-connected bundle/reduction, and prefix/history state have
different shapes and identities. Hyper-connection records already expose a
split, reduced vector, branch output, and outgoing bundle. Attention-residual
records expose candidate/snapshot counts, probabilities, mixed vectors,
prefix before/after, and separate boundary events. Preserve these distinctions;
flattening them into one residual discards execution evidence.

Record topology version, carrier shape, vector role, and any reduction method.
The current attention-residual observer intentionally omits layer-0 attention
reduction; a frontend must not fill in that missing site as though it executed.
Topology-specific counts and boundary ordering may be scientifically material
even when numerical outputs agree.

## 8. Projection and readout descriptors

A projection descriptor freezes provider/version, basis ID/hash, dimensionality,
source space and role, preprocessing/centering/scaling, axes, artifact identity,
and training/source run if applicable. A fitted basis is sealed before the run.
Changing a basis creates a distinct derived view with provenance, never an
unannounced refit of recorded coordinates.

Replay uses stored coordinates without loading model weights or rerunning a
provider. Overlay requires matching ID, digest, preprocessing, and compatible
source space. Matching display names alone is insufficient. Different basis
IDs are refused in v1 even if someone asserts equivalence.

A readout descriptor separately defines intermediate logit-lens/reader method,
normalization, vocabulary and selected-token set, probe precision, cost, and
whether the result is derived. Final execution logits name the real output
head and distinguish raw from sampled/temperature-adjusted distributions.

Do not compute layer probabilities by normalizing selected tokens and label
them full-vocabulary probabilities. Exact ranks and entropy require adequate
vocabulary evidence. A selected-token logit difference is available without
the full softmax denominator; other missing measures remain unavailable.

## 9. Bounded work and loss accounting

The observer never awaits disk, network, subscriber acknowledgement, or UI
rendering. Bounded capture work can still cost time; record it. Admission
uses bounded storage and a nonblocking failure path. Large payloads and
additional probes need independent quotas so one full vector cannot bypass
the event budget. Standard mode has no unbounded allocation or tensor dump.

A recorder's target is lossless evidence within the declared capture scope.
If it cannot maintain that guarantee, it must explicitly fail completeness
and refuse any complete-record adjudication, while model execution continues.
A lossy live viewer and a complete recorder are separate capability claims;
sharing the schema does not equate their retention policies.

Separate at least these failure domains:

1. Capture admission loss: the observation could not enter the recorder queue.
2. Recording failure: admitted evidence could not be persisted.
3. Subscriber/relay retention loss: canonical evidence may still exist locally
   but a subscriber fell behind.
4. Browser-local retention loss: a paused/tab-limited client evicted evidence.

Loss counters cannot live only in the queue that overflows. Maintain an
independent bounded loss ledger and terminal summary. Coalesce adjacent ranges;
if the range budget is exhausted, retain an exact total and a labelled coarse
coverage interval rather than allocate without bound or claim exact positions.
Record scope, reason, count, and whether range attribution is exact. Loss-report
messages themselves must not recursively create unbounded reporting work.

Once admitted, loss summaries enter the event log; terminal receipt metadata
includes all counters even if no later observation can carry a notification.
Test a full queue immediately before completion. Reserve terminal/loss summary
storage outside ordinary observation admission and finalize on the control
side after inference returns. If storage fails or the process crashes, the
record is incomplete/unsealed; do not promise an immutable complete receipt
that was never written. Metadata remains queryable independently of a saturated
subscriber queue where the runner survives.

The immutable capture receipt describes the canonical retained log. Subscriber
delivery reports describe transport gaps; they do not rewrite the capture
receipt when a viewer disconnects. Backfill repairs delivery gaps and updates
the viewer's coverage, but cannot undo permanent capture loss.

## 10. Transport and reconnection

Conceptual transport actions are publish event, publish receipt, fetch permitted
payload, and receive authorized command. The executor only sees an observer.
Existing inference SSE/WebSocket clients retain their protocols.

For live viewing, negotiate version and capability, subscribe by run and
`after_sequence`, and return retained events plus a retention watermark and
gap report. Ingress deduplicates by run/sequence and orders retained evidence;
it never pretends a sequence gap is complete. Delivery can be at least once;
the reducer provides exactly-once identity, not exactly-once network delivery.

In hosted mode, the runner initiates outbound WSS to the relay with an expiring
publisher capability. The browser obtains a distinct viewer/operator grant.
Capabilities are session/run scoped and revocable; validate issuer, audience,
expiry, and command scope. Short human-readable run labels are not bearer
secrets. Keep secrets out of shareable run URLs and recorded events.

Local mode binds loopback by default, checks WebSocket origin and session
authorization, and serves bundled assets. Hosted operation does not rely on
a secure webpage opening an insecure socket to a private machine. Private
hosting uses the same runner and browser roles.

Disconnecting a browser or relay never cancels inference. An operator cancellation
is an explicit command. Commands carry idempotency IDs: reconnect/retry must
not start a duplicate run or future intervention. Acknowledgement identifies
accepted/refused action and the authoritative run ID. Limits apply per subscriber
and per run so slow viewers cannot hold the recording pipeline hostage.

## 11. Export policy and payloads

Runner-side allowlists enforce METRICS, OBSERVATIONS, and FULL as defined by
the product spec. Data scope and operator authority are independent. Neither
a relay nor website can widen either. Apply policy before data leaves the
runner, including receipts, descriptors, error strings, payload metadata, and
diagnostic logs intended for transport.

Token IDs can reconstruct prompt text; ordinary prompt hashes can disclose
guessable prompts. METRICS excludes both rather than calling them anonymous.
Projection coordinates themselves may be sensitive and are exported only under
the authorized policy. Top-k labels obey the same token/text restrictions.

Payload references identify content, shape/dtype, availability, and access
requirements without exposing local filesystem paths. `local://` is a logical
runner-owned reference, not browser permission to read a file. Fetch requests
resolve only registered payload IDs under the runner's policy and byte budget.
FULL permits specified data, not arbitrary filesystem reads or all tensors.

A filtered export has its own manifest, content hash, coverage, and receipt
identity. It is not byte-identical to the local record and must not reuse that
record's digest. Sequence holes from redaction are declared as policy omissions,
not reported as queue overflow. Bind exports to parent identity only where
that identity is authorized to leave the runner.

## 12. Recording and immutable receipt

An append-only log stores validated envelopes plus manifests and descriptors;
large payloads are separately content-addressed where retained. The first
implementation may use a lossless JSON event log, but framing, canonical hash
encoding, fsync policy, and export container are not frozen by this draft.

The sealed receipt covers artifact/model authority, component/plan, tokenizer,
prompt tokens, runtime/backend/precision, sampling/seed, initial state/reuse,
capture/projection/readout configuration, log byte digest and sequence coverage,
loss totals and ranges, payload index/digests, output IDs/text digest where
permitted, stop reason, and terminal state. Specify exactly which bytes each
hash covers before implementation; do not hash a UI-formatted reconstruction.

A receipt references the finalized log digest; the log does not include its
own final digest. This avoids a circular content hash. A content digest verifies
integrity, not runner authenticity; signing is a separate declared capability.
Do not invent an authority digest if no authority artifact exists.

During execution, metadata is provisional. Finalization seals it once. Later
annotations and derived comparisons are separate records linked to that receipt.
A truncated log may replay its valid prefix with a persistent incomplete label.
A terminal event alone does not prove the full log was persisted successfully.

## 13. Shared client state and semantic comparison

Live sockets, fetched backfill, and recorded logs enter one validated reducer.
It maintains event identity, coverage, immutable observations, semantic indices,
and terminal evidence. Selection and presentation cursor are separate from
ingestion. Renderer frame batching never discards run-store evidence silently.

Replay seeks by rebuilding from the prefix or a verified client-state checkpoint,
including omission/loss markers. The same selection must yield the same values,
coordinates, labels, and availability as live viewing. A browser state checkpoint
is not a KV checkpoint from which model execution can resume.

Comparison joins explicitly mapped positions and compatible semantic sites,
not event order. Report unmatched rows. For full-state vectors `a` (base) and
`b` (target), the initial metric definitions are:

```text
cosine            = dot(a,b) / (||a||2 * ||b||2)
relative RMS      = RMS(b-a) / RMS(a)
norm ratio        = ||b||2 / ||a||2
projection distance = ||project(b) - project(a)||2
selected-logit Δ  = logit_target - logit_base
rank Δ            = rank_target - rank_base  (rank 1 is best)
```

Zero denominators are undefined with a reason, not hidden epsilon choices.
If another normalization is offered it has a distinct method/version. Cosine
and RMS need actual compatible vectors or a declared sufficient statistic;
projection distance cannot be relabelled full-state distance. Missing vectors
may leave most comparison metrics unavailable under Standard capture. Extra
capture or a runner-local comparison is an explicit capability, never automatic
rerunning of a historical prompt.

Threshold-based divergence summaries record metric, threshold, coverage,
alignment, and normalization. Interpolation between compared endpoints is
presentation state only and emits no fake observation event.

## 14. Implementation gates and unresolved decisions

Implement and retain evidence for the product's real-model acceptance table.
Contract tests additionally cover duplicate/conflicting events, integer precision,
serialization round trips, final-event overflow, loss-ledger saturation, unknown
schema versions, truncated recordings, policy redaction, command retry, unequal
tokenization, missing sites, and topology-specific absence/order. Unknown optional
fields may be ignored under version rules; unknown required semantics cannot
be rendered as known evidence.

The next engineering decisions are concrete:

* Which real artifact/backend closes the first parity and replay witness?
* Which exact canonical prefill and step taps expose carrier and applied writes
  without additional execution or an undisclosed backend fallback?
* What is Standard's bounded projection/readout budget on each backend, and
  which probes require Instrumented capture?
* Where do shared schema types live, and how are browser types validated against
  the Rust representation?
* What fixed projection artifact can be shipped honestly before sealed ADDRESS
  readers are available?
* What are the event/payload limits, loss-range budget, retention policy, durable
  framing, and receipt hash encoding?
* Which pinned HAUSE revision and frontend packaging support both bundled offline
  assets and hosted deployment?

These choices do not require freezing executor internals to begin frontend
composition. They do require closure before claiming the Observatory's real
execution, completeness, timing, or scientific guarantees.

## UI Standard-record bridge (draft implementation)

The frontend now has a [Standard-record adapter](../observatory/STANDARD-ADAPTER.md)
mirroring the executor's `WriteStats` fields. Its `larql.observatory.standard.v1`
file envelope is a proposed runner bridge, not this contract's frozen wire ABI.
The current implementation imports recorded Single-carrier stats through the
same UI reducer as fixtures. It neither provides live transport nor proves
real-model parity. The runner adds identity, timing and program metadata.

The executor's `dot-raw-v1` probe is carrier · pre-extracted head row without
final normalization or softmax. Label it a **raw probe**, and differences as
**probe differences**, not normalized logits or log-odds. Missing entering-state
norms, timing, embedding samples, directional-write probes and attention sources
are unavailable. The amended V3-OBS-1 witness order (Granite primary, OLMo2
secondary, Gemma 3 4B next) supersedes the earlier OLMo2-first handoff.
