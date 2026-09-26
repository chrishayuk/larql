# LARQL Observatory — product specification v0.1

Status: proposed, 2026-09-19. This specifies the product; it does not claim
that an Observatory implementation or acceptance witness has shipped.
The companion [VINDEX3 observation contract](vindex3-observation-contract.md)
is a separate draft. Its wire names and executor integration remain open.

> Run a prompt and watch the computation move through the model live.

The Observatory is a HAUSE-native visual environment for exploring the model
as a computational database: its structure, live traversal, representations,
alternative execution paths, and programmable state transitions. The MVP centers on one experience:
**Prompt → Run → Watch**. Watch a carrier trajectory move through depth while
coordinated views expose attention, FFN, residual change, and computational
structure. Inspect a point and replay the same execution without rerunning it.

The longer-term proposition remains **run a model anywhere, observe it
anywhere**. Compare, hosted multi-user operation, rich receipt presentation,
and interventions follow the live instrument; they do not gate this MVP.

VINDEX3 runs the model. LARQL makes it operable. HAUSE makes computation
visible. The interface answers **what did the model just do?**; it must not
present its pictures as direct access to model reasoning.

## 1. One model world, three visual planes

| Plane | Question | Evidence |
|---|---|---|
| MODEL | What exists? | Admitted objects, operations, representations, placement and dependencies; no inference required |
| EXECUTION | What is happening? | Observed traversal, carrier frontier, writes, routing, queries and output formation |
| ALTERNATIVES | What else could happen? | Real admitted plans or recorded runs; never imaginary executable branches |

Graph and Map refer to the same semantic sites and states using different
geometries. The carrier is the current traversal/query state. Static model
structure remains available before and during execution. The fixture UI may
show its declared synthetic operation graph; it must not call this an extracted
VINDEX3 graph or fabricate France/Paris knowledge edges.

Runner capability discovery will gate BROWSE, WALK, EXECUTE, OBSERVE,
REPRESENT, COMPARE, PROGRAM, PROFILE and HEAD_CAPTURE. Each absent capability
has an explicit unavailable state. Local/remote hosting uses the same capability
surface. Future visual equivalents of DESCRIBE, WALK, SHOW and NEIGHBORS must
consume authoritative query results with provenance. No generated answers are
substituted for model-graph facts.

Attention/mixers use connection/routing forms; FFN uses transformation or,
when evidenced, address/retrieval forms. FFN-as-addressed-retrieval is a research
hypothesis, not a universal semantic label. Representation selection, expert
placement, virtual expert expansion, WASM writes and matmul elision are future
observed facts; the UI must not derive them from model family names. Operation
contracts distinguish exact, bit-identical, numerically equivalent,
behaviorally equivalent, approximate and unverified, with witness references.
Deterministic execution alone establishes no equivalence.

### Execution authority

One execution supplies every view. No secondary analysis model, Python-hook
reconstruction, visualization-specific forward pass, or automatic replay
of inference to fill absent observations. Derived readouts may consume
captured states, with their method and cost declared. Replay consumes a
record, with no model execution.

Passive observation preserves values, output, operation order, and model
state. Timing is a separate claim: reductions, probes, recording, and device
readbacks have costs. A capture label alone never establishes either parity
or representative performance.

The browser is not the runtime. Local, self-hosted, and hosted installations
use the same event semantics, coordinated views, and recording format.
Deployment cannot change the scientific meaning of a run.

## 2. Screen-by-screen MVP

These are states of one workbench, not a sequence of management pages.

### A. Prompt — the opening state

```text
LARQL / OBSERVATORY                                  NO ACTIVE RUN

                    PROMPT A MODEL

             The capital of France is

             model.vindex3 · LOCAL
             Standard capture · Fixed basis          RUN →

             Capture / projection settings           Open recording
```

The prompt owns the screen. Model selection is quiet beneath it; only available,
admitted artifacts appear. Hardware labels come from the runner. Settings are
disclosed inline, not a prerequisite wizard. Run starts execution with the
effective capture manifest. If there is no runner/model, show that prerequisite
in place. A refused start preserves the prompt and gives the actual reason.

### B. Watch — the live default

```text
LARQL / OBSERVATORY        model · run · capture        LIVE · Pause UI
                          Stream  Map  Trace  Graph
The  capital  of  France  is                         selected position
┌─────────────┬────────────────────────────────┬─────────────────────┐
│ STREAM      │ MAP                            │ L24 · AttentionWrite│
│ Embedding ✓ │                                │                     │
│ L00       ✓ │          carrier trajectory    │ ATTENTION SOURCES   │
│ ...         │                                │ token weights       │
│ L23       ✓ │               ● selected       │ total write norm    │
│ L24       ● │            ╱                   │                     │
│ L25         │          ●                     │ FFN / CARRIER       │
│ ...         │       ╱                        │ write / change norms│
│ Output      │     ●                          │                     │
│             │ basis identity / axes          │ OUTPUT MOVEMENT     │
│ attn / FFN  │                                │ selected readout    │
├─────────────┴────────────────────────────────┴─────────────────────┤
│ selected-answer trajectory · generated text                        │
│ prefill / decode · execution cursor · Go Live                      │
└───────────────────────────────────────────────────────────────────┘
```

At desktop width reserve roughly 140–180 px for the layer rail, 300–360 px
for the inspector, and the remaining width for Map. These are composition
targets, not fixed viewport requirements. On smaller widths move the rail
below and disclose the inspector on selection. Map remains the hero.

Tokens appear from tokenization; the rail tracks actual execution; carrier
points extend the trajectory as captured values arrive. Attention/FFN bars
display named measured magnitudes with units and scale, never decorative
activity. Embedding and output are explicit endpoints when observed.

Live follows the final prompt position initially, then the state predicting
the current output token. Clicking a token pins that absolute position until
Follow current is restored. A point selected for an applied write highlights
the corresponding carrier-after point with both roles named; this linkage does
not pretend the write vector is the carrier vector.

No artificial delay makes a fast model seem animated. Live state may fill
immediately and become available for scrub/replay. Pause UI stops cursor
motion while ingestion continues; Go Live catches up without rerunning.

### C. Inspect — selection holds the room

Clicking a layer/site or map point pins its position and semantic site. The
rail, map, inspector, Trace, answer strip, and Graph share that selection.
The inspector stacks attention source distribution and total write magnitude,
FFN write and carrier change, selected-output movement, then compact source
identity/timing. Site evidence is primary; receipt metadata is a disclosure.

Choose Trace to unfold attention → carrier ← FFN for the selected layer, with
the numerical table directly available below. Trace columns always include
attention change, FFN change, and carrier change. These are different vectors;
their norms are not additive. Zero, unavailable, not-yet-executed, and no-FFN
layers have different presentations.

### D. Graph — explanation on demand

Graph replaces the central field, keeping rail, token selection, inspector,
and execution cursor. It opens centered on the selected site/output. Reveal
one predecessor/successor neighborhood at a time; permit returning to Map
without losing selection. A legend distinguishes structural dependencies,
attention weights, and derived output readouts. MVP requires this simple
graph, not source attribution or a circuit tracer.

### E. Complete and replay — the same room

Completion leaves the final output and captured trajectory visible. Replay,
step, and scrub operate on the recorded stream through the same views. An
opened recording enters this state directly with `RECORDED` replacing `LIVE`.
No run browser or experiment registry is required. Saving/opening a record
and a compact completeness/source disclosure are sufficient for MVP.

Failure leaves the valid recorded prefix inspectable and labels its terminal
status. Missing evidence stays visible as gaps. Retained evidence remains
inspectable after disconnect; connection loss alone does not claim execution
has stopped.

## 3. Built with HAUSE

Use the actual HAUSE tokens and forms, pinned to a reviewed library revision,
following its [installation and composition surface](https://hause.design/use).
Typography imitation alone does not satisfy this requirement.

The following is the Observatory's application of the
[HAUSE form vocabulary](https://hause.design/forms):

| Mode | Forms | Observatory use |
|---|---|---|
| READ | Statement, Observation, Evidence, Claim, Refusal | Findings, receipts, unavailable evidence, incompatible comparison |
| OPERATE | Comparison, Decomposition, Lens, Provenance | Compare runs, inspect a layer, change evidence depth, inspect identity |
| WATCH | Procession, Transformation, Unfolding | Follow execution depth, stage a comparison, reveal a decomposition |

HAUSE [Comparison](https://hause.design/forms/comparison) provides the
interaction precedent, not a ready-made trajectory renderer. Its published
props describe labelled blocks; integrating scientific trajectories needs
a real composition or extension, not an assumption that arbitrary plots
already fit that API. Preserve keyboard operation and a text/evidence
fallback. Develop Trajectory, Trace, Fork, or Divergence forms only where
this product demonstrates a reusable need, following HAUSE's
[documented admission practice](https://hause.design/how-hause-grew).

The main field is dark or neutral, spatial, and restrained. Use fine lines,
subtle perspective, crisp labels, and a clear selected point. No arbitrary
rotation, particles, decorative gauges, saturated heatmap spectacle, tile
sprawl, or dashboard-card shell. A light reading environment may serve the
record. Motion directs attention to evidence; reduced motion preserves all
content and controls.

## 4. Persistent context and coordinated selection

Primary scenes are `STREAM · MAP · TRACE · GRAPH`, with `COMPARE` reserved
for the next increment. Stream is the layer procession, Map the carrier
trajectory, Trace the write decomposition, and Graph the structural explanation.
Prompt is the opening state and stays editable for the next run. Run, model,
events, and performance details are disclosures; a full receipt panel is later.

Keep active model, run ID, execution state, capture mode, live/replay state,
and projection identity visible without competing with the computation.
Execution state, connection state, and playback state are distinct: a
disconnected browser does not mean a failed model; a paused view does not
mean paused inference.

Shared selection contains run, phase, absolute token position, semantic
site, and carrier identity. Selecting `France × L24` updates the matrix,
map, trace, answer strip, graph, and inspector together. Layer-level
selection retains the selected role or explicitly chooses a boundary;
it must not silently substitute an attention write for a carrier state.

Desktop layout gives most space to the primary visual, a compact inspector
beside it, and an execution/token/depth timeline below. Tablet is useful;
phone prioritizes read-only evidence and summaries. Selection must be
available through an accessible evidence table as well as the canvas.

## 5. Prompt and capture

The prompt workspace provides model selection, prompt text, generation
settings, capture mode, projection selection, and Run. The runner advertises
supported combinations. Changing capture or basis on an existing record
does not silently execute the model again.

| Capture | Intended observations | Disclosure |
|---|---|---|
| Standard | V3-OBS-1 stats observer: carrier norm, applied-write/delta norm, fixed projection, and predeclared selected-token logits from extracted output-head rows; adapter adds permitted structural/token metadata | No full top-k, per-head capture, tensor persistence or Metal readback |
| Rich | Separate per-head executor rung, enabling truthful source attention summaries and later head drill-down | Not V3-OBS-1; fixture source weights must say synthetic |
| Instrumented | Optional head, kernel, or device-level observations | Prominent timing intrusion and readback disclosure |

These are capability profiles, not promises that every backend exposes
every quantity. A field has a value or a reason: unsupported, not captured,
redacted, dropped, or not applicable. Missing is never zero.

Instrumented mode displays:

> Instrumented capture may alter runtime timing.
> Model output semantics remain parity-gated.

The receipt identifies the actual parity evidence; until a backend/capture
combination passes its gate, the UI must not assert that parity is verified.
Standard capture must not introduce hidden per-layer full-vocabulary probes
or device synchronization under the label of cheap observation.

## 6. Watching execution

The persistent strip follows actual semantic sites, beginning with embedding
and ending at output. Completed sites remain faintly present; current work
is highlighted. Prefill and decode are separate. Generated token text and
ordinal are shown alongside the absolute position of the state that
predicted that token.

An execution event enters the run store once, even after reconnect. Rendering
may batch many events into a frame; it must preserve all retained evidence
for inspection. Pause UI freezes the presentation cursor while ingestion
continues. Go Live catches up immediately. Capture pause, if implemented,
is a separate authorized action and records its missing interval.

## 7. Token × depth matrix

The matrix is an optional dense evidence view after the live MVP. It must not
replace Map as the opening live scene or delay the nine required capabilities.

Columns are actual token positions with tokenizer-derived labels when
permitted. Rows are semantic depth/sites, including embedding and output.
Attention, FFN, and carrier boundary observations remain distinguishable.
Topology may require separate carriers or history-aware rows.

Selectable metrics: carrier/residual change, attention write, FFN write,
block/carrier change, selected-answer logit, entropy, projection displacement,
comparison delta, and execution time. Only supported metrics are enabled.
Each metric declares units, method, normalization, and evidence class.
Timing for a batched operation is not divided into invented per-token
durations. Heatmap legends show the scale, including clipping; unavailable
and dropped cells use distinct marks, not low-intensity colors.

Clicking a cell selects it everywhere. Dense runs may virtualize rows and
columns; this is a display optimization, not event loss.

## 8. MAP and projection identity

The map plots observed states in a declared fixed basis. The states are
observed; their coordinates are projected. Every point identifies its source
event/site, position, carrier, and vector role. A tooltip shows norm, change
metric, coordinates, evidence class, and basis identity.

Projection metadata includes provider, basis ID, content hash, dimensionality,
source space, preprocessing, axis labels, and source/training run when
applicable. The active basis is always visible. Camera movement changes
presentation only. It cannot change stored coordinates or silently rescale
two runs independently.

Supported provider concepts may include fixed random projection, PCA,
answer-token directions, relation/entity/binding readers, aligned readers,
SAE, CLT, operator sketches, and custom providers. These are extension points,
not v1 commitments. V1 requires a selectable registered fixed basis and
reproducible stored coordinates.

ADDRESS is an authored configuration: relation, entity, and binding labels
belong to its sealed reader metadata, not to VINDEX3's executor vocabulary.
Do not assign them to an arbitrary projection. A hyper-connected reduced
vector is prominently labelled `HC reduced projection`; a prefix/history
state is not drawn as an ordinary residual without naming the reduction.

## 9. TRACE and answer trajectory

Selecting a layer unfolds its observed attention write, carrier update, and
FFN write, then presents numerical evidence. An always-available table
provides the dense view: layer/site, carrier change, attention change, FFN
change, and selected-output movement. Controls select position, output token,
absolute/comparison values, and explicit normalization.

The answer strip supports logit, probability, rank, and two-token log-odds.
There are two different sources:

* Final head logits are actual execution output, before declared sampling
  transforms.
* Intermediate layer readouts are derived probes of captured states. They
  identify the reader, normalization, vocabulary coverage, and cost. They
  are not predictions emitted by a partially executed model.

Exact rank, full-vocabulary probability, and entropy require enough evidence
to compute them. Selected logits alone support logit differences, not a
full softmax probability or exact vocabulary rank. Top-k-only coverage can
support bounds, labelled as such. A missing probe leaves a gap.

`Paris became top-1 at L24` may appear only with the probe identity and
sufficient preceding coverage; otherwise say `first observed top-1 at L24`
and expose the coverage limit. The sample prompts do not predetermine the
answer or the layer where it becomes readable.

## 10. GRAPH and evidence language

Graph starts from the selected output or site. `FOLLOW THE COMPUTATION →`
reveals a bounded neighborhood, rather than a whole force-directed graph.
The initial vocabulary is tokens, sites, attention/FFN writes, and output
candidates. Heads, experts, features, and provider-specific nodes are later
capabilities. No inferred token-to-write association is fabricated from
layer completion alone.

Every graph carries a legend. Distinguish declared computational dependency,
observed association, attributed contribution, and intervention-supported
effect. Dashed attribution and double causal lines may encode these; thickness
is meaningful only with a named quantitative measure. Color is supplemental.
An execution dependency is not proof that an input caused the selected answer.

Across views, use explicit states: Observed, Derived, Projected, Compared,
Attributed, Intervened, Causally supported, Unavailable, Dropped. Evidence
class and availability are separate properties; a projection may also be
unavailable. Per-head drill-down has a reserved inspector location and an
honest unavailable state; it is not required for v1.

## 11. COMPARE — next workflow; fixture D exercises it now

Choose a previous run, duplicate the prompt, modify it and run again, or open
a record. Base and target retain immutable identities. Align by semantic
site and declared token-position correspondence, never timestamps or event
sequence. Different tokenizations and missing sites produce explicit unmatched
entries. Cross-model comparison requires compatible spaces and an explicit
mapping, not matching layer numbers alone.

The composition stages the existing trajectory and a second in the same
field, letting shared and divergent regions carry attention. Offer direct
inspection and a reader-controlled comparison. A continuous base/target
morph is **display interpolation**, not a third observed execution: label
it, preserve measured endpoints, and do not attach measured logits or causal
claims to the interpolated path. Prefer endpoint overlays for measurement.

Projection overlays require matching basis IDs **and** content identity,
source-space compatibility, and preprocessing. Refuse incompatible overlays
with both identities shown; do not silently align or normalize them.

Per-site comparison may show cosine, relative RMS, norm ratio, projection
distance, selected-logit delta, rank delta, and top-token agreement. Full-state
cosine/RMS needs compatible vector evidence; projected coordinates alone
cannot supply it. The runner may compute a comparison from retained local
payloads and publish derived results within the session's allowed scope.

`First strong divergence` and `Largest divergence` are descriptive summaries
with metric, threshold, normalization, and coverage disclosed. Missing sites
prevent a definitive earliest-site claim. No summary implies statistical or
causal validation.

## 12. Inspector and runtime evidence

The inspector exposes the selected semantic site, absolute position/token,
vector/carrier role, values and units, output readout method, timing/capture
cost, operation identity, plan identity, and source event. Payload availability
is explicit. Later actions—Zero, Replace, Patch, Scale, Project out, Fork—stay
disabled until their execution and authorization contracts exist.

The performance disclosure can show operation time, requested/resident/loaded
operand bytes, routing, storage activity, observer cost, and device readbacks.
Only counters actually supported by the runtime are displayed. Page faults
are not automatically NVMe bytes; asynchronous submit time is not device
execution time. Metric scope and measurement method are inspectable.

Instrumentation-affected timing is prominently marked and cannot support an
unqualified performance comparison. Performance claims follow the repository's
[run hygiene](kv-attention-scaling.md), including warmed, mutually agreeing
baseline/candidate/baseline brackets and peer exclusivity.

## 13. Local and hosted deployment

Conceptual CLI surfaces, **not existing commands**:

```text
larql observe model.vindex3
larql observe --connect <observatory-session> model.vindex3
```

Local mode serves bundled Observatory assets and a loopback runner endpoint,
works without accounts or a cloud service, and stores records locally. Bundle
fonts and required assets so the local experience has no hidden web dependency.

Hosted mode has the runner establish an outbound authenticated WSS connection
to a relay. Browser viewers subscribe separately. No inbound inference port,
port forwarding, or hosted-page-to-localhost dependency is required. Self-hosted
instances use the same protocol. An expiring session capability grants access;
a short display code such as `8F21` is not itself an authentication secret.

Transport adapters conceptually cover in-process, local WebSocket, remote
WebSocket, and recorded logs. The executor emits to an observer; it does not
know about any of these transports. Local and hosted protocol symmetry is an architectural requirement; the
outbound relay and remote multi-user control do not gate this UI MVP. A public multi-tenant
service, accounts, and distributed execution management are separate work.

## 14. Privacy and capabilities

| Runner-authorized scope | Exported content |
|---|---|
| METRICS | Semantic sites, allowed norms/projections, timing and counters; no prompt, token IDs/text, logits, or tensors |
| OBSERVATIONS | Metrics plus allowed token IDs, selected logits/top-k, and explicitly approved token labels |
| FULL | Explicitly allowed prompt, activation payloads, and captures |

Remote sessions default to METRICS. These names are disclosure policies, not
anonymity guarantees: projections and token IDs may reveal information.
Prompt/output text and their hashes, tokenizer labels, receipt fields, model
paths, payload references, and error strings must obey the same export policy.
A hosted UI cannot elevate runner authorization.

Viewer and operator capabilities are separate. A viewer watches and replays;
an operator may request a run or permitted payload. UI playback and comparing
already-authorized records require no execution permission. Starting a new
comparison run does. Later intervention commands require a separate explicit
grant. A command permission does not grant additional data export.

Full vectors stay on the runner unless export is allowed. The inspector can
say `FULL VECTOR REMAINS ON RUNNER`. A local payload reference is opaque;
the hosted browser must not treat it as a fetchable local file path.

## 15. Recording and replay now; history and receipt presentation later

Live transport and a recorded event log feed the same reducer and renderer.
Replay provides Play, Pause, Step site/layer/token, scrub, 0.25×, 1×, 4×, and
Instant. Timing-preserving playback is optional. Exact recorded coordinates
are reused; replay does not refit or recompute a projection. Stored evidence
remains inspectable even when animation skips frames.

When run history is added, it lists timestamp, permitted prompt label, model, and run
status. Selecting a record opens replay. An uploaded record remains available
after the runner disconnects. A local-only record says `RECORD HELD BY RUNNER`
when unavailable; metadata alone must not impersonate a complete replay.

Canonical links are `/lab`, `/lab/run/<run_id>`,
`/lab/run/<run_id>?view=map`, and `/lab/compare/<run_a>/<run_b>`.
Links identify records; possession of a URL alone does not grant capabilities.

The final receipt seals model authority, artifact, operation plan, tokenizer,
prompt tokens/hash, generation settings/seed, backend/runtime identity,
effective capture policy, projection/readout identities, event-log hash and
coverage, loss/retention information, output/hash, and terminal status.
Unavailable authorities are declared, not fabricated. A running manifest is
provisional; an interrupted run cannot masquerade as a finalized receipt.
Remote redacted records have distinct export identity and coverage.

If events are lost, show, for example:

> 12 observation events dropped. Run execution is valid.
> Visualization is incomplete.

Only assert execution validity if independently established. Persist evidence
loss in the receipt. Transport loss, capture loss, redaction, and an intentionally
uncollected metric are different. Backfill can repair transport loss; it
cannot recover observations never retained. Complete-observation experiments
may refuse adjudication on an incomplete record.

## 16. Keyboard and accessibility

| Key | Action |
|---|---|
| Space | Play/pause presentation or replay |
| Left / Right | Previous/next semantic site |
| Shift + Left / Right | Previous/next layer |
| J / K | Previous/next token |
| C | Open compare for selected run |
| Escape | Clear selection |

No shortcuts intercept prompt inputs, editable fields, sliders, or other
controls that own those keys. Provide focus indicators, labelled controls,
textual metric/evidence access, and motion-independent progress. Selection
does not depend solely on hover, color, or a three-dimensional canvas.

## 17. MVP scope and fixture delivery

The required instrument has nine capabilities:

1. Prompt → Run → Watch entry.
2. Live semantic layer/site progression.
3. Carrier trajectory with explicit fixed projection identity.
4. Per-site attention / FFN / carrier metrics.
5. Predeclared selected-token logit trajectory.
6. Attention-source pane, explicitly unavailable under Standard until the
   separate per-head rung provides truthful source evidence.
7. Coordinated token/layer/site inspector.
8. Bounded structural graph around selection.
9. Lossless-record replay through the same views.

The fixture implementation is in [observatory/](../observatory/README.md).
It is an operable UI prototype, not completion of the live-run acceptance gate.
All four stories are explicitly synthetic authored recordings: A, answer
formation; B, attention versus FFN; C, ADDRESS visual needs; D, counterfactual
separation. C does not claim to reproduce the sealed ADDRESS-BUILD experiment.
B's attention distribution is synthetic, not a new Standard capture capability.
The record schema is a frontend fixture format, not an executor ABI freeze.

Gemma 3 4B is the user's primary UI test bed; Granite 4.2 3B is another initial
subject. The amended V3-OBS-1 freeze names Granite as the primary witness, OLMo2 as a secondary subject, and Gemma 3 4B next.
Granite's 40-layer program must come from the admitted plan at integration;
fixture layer counts do not assert model architecture facts. Do not block
frontend work on Gemma admission, per-head capture or interventions.

Token × depth matrix and fixture comparison are useful supporting views.
Live two-run comparison is the next major workflow. Model browsing,
representations, physical placement and alternative plans grow through runner
capabilities over the same object/site identities. A fixture Model plane can
inspect declared fixture operations without playing the recording.

Interventions, full circuit tracing, remote multi-user control, experiment
registries, statistical adjudication, Gemma Scope, and per-head UI are later.
Rich receipt presentation is later; minimal source, basis, coverage and loss
metadata remain necessary for honest replay now.

## 18. Acceptance witnesses

UI fixture acceptance and live executor acceptance are separate.

| UI witness | Requirement |
|---|---|
| Four stories | All synthetic evidence labelled; no fixture is presented as live inference |
| Shared selection | Map, layer rail, Trace, inspector, answer plot and graph agree on token/site |
| Replay | Identical recorded coordinates and values through one reducer, with no execution |
| Capability absence | Standard attention sources unavailable; missing FFN writes stay missing |
| Graph | Declared structure differs from measured weights, derived readouts and causal claims |
| Counterfactual | Align semantic sites and positions, refuse mismatched bases, report unmatched sites |
| Loss | Duplicate delivery deduplicates; conflicting identity refuses; incomplete records remain visibly incomplete |
| Controls | Scrub and step both ways; keyboard shortcuts leave editable controls alone |

Before calling the live Observatory shipped, a real admitted model must execute
arbitrary prompts through the canonical path, satisfy observed/unobserved parity,
expose each emitted site exactly once, record reproducible coordinates, and replay
without model execution. A blocked/stalled browser must not block execution.
Force end-of-run queue overflow and preserve its loss metadata. The live stream
may be lossy/nonblocking; the replay recorder must be lossless within its admitted
scope or explicitly fail completeness. Matching output cannot repair missing
observations.

Full-vocabulary rank, probability and top-k are not required live. Selected-token
readouts must identify their method; the eventual final output token may not be
in the predeclared set. In that case show its earlier trajectory as unavailable
rather than run another traversal or silently change capture retrospectively.

First live demonstration: France/Germany prompts, actual outputs, carrier
trajectories and available selected logits. Later compare the runs. Paris/Berlin
are intended illustrations, never fabricated results. The authored scientific
ADDRESS study requires the original sealed readers and real experimental
receipts before any early-relation/late-binding conclusion is displayed.

## 19. Ownership and next integration

The executor session owns the frozen
[V3-OBS-1 carrier observation rung](v3-obs-1-carrier-observation.md): the
canonical CPU `leave_site` tap, parity/reconstruction properties, witness map,
and capture-cost protocol. This UI session owns presentation, fixture adapters,
selection, replay and capability disclosures. It does not alter executor internals.

Standard maps to that rung's stats observer. Rich requires the separate per-head
rung. Structural events are owned; tensor observations arrive through borrowed
callbacks. The runner adapter adds run identity, run sequence and monotonic time;
the executor does not own the web envelope. `FfnDone` does not imply an FFN write,
and Bundle/History must retain topology identity. Live delivery and lossless
recording are separate policies over the same tap.

The product and [protocol draft](vindex3-observation-contract.md) remain the
consumer authorities. The frontend fixture schema stays explicitly separate
until the runner adapter is agreed. The first integration replaces the fixture
source with admitted Standard observations while preserving the views. No new
scientific claim is created merely by changing the data source.

## 20. Visual lineage: the-mechanism

The [the-mechanism visuals](https://github.com/chrishayuk/the-mechanism/tree/main/visuals)
provide a vocabulary for this instrument, not evidence about every model.
Its seeded diagrams and its measured model experiments must remain distinguishable.

| Primitive | Observatory meaning | Evidence required |
|---|---|---|
| Spot | State in a declared projection | Coordinates and basis identity |
| Worldline | Carrier states through the executed program | Observed semantic sites |
| Address / route | Selection of stored structure or candidates | Provider-labelled scores or runtime routing |
| Edge / read | Inspectable stored relationships | Model object identity and provenance |
| Write | Applied contribution to the carrier | Captured write statistics or declared derived readout |
| Resolve | Changing selected candidate readouts | Same declared reader through depth |
| Walk / compute | Traversal or computation actually performed | Runtime operation identity; no classification from appearance |
| Patch | A changed stored object or execution | Immutable parent and intervention identity |

The [spot visual](https://github.com/chrishayuk/the-mechanism/blob/main/visuals/v1_spot_in_space.py)
projects its actual seeded vectors. Following that discipline, Observatory fixture
norms, coordinates, linear readouts and directional writes derive from the same
authored carrier values. The fixture reader rows and basis hashes are recorded.
These are synthetic UI witnesses, not Gemma measurements or a sealed ADDRESS reader.

[trace.py](https://github.com/chrishayuk/the-mechanism/blob/main/trace.py)
distinguishes answer readability, directional FFN writes and ablation. Trace and
Graph therefore expose separate Readable, Written and Causal evidence. A readable
answer is not proof that a site supplied it; a directional write is not proof of
necessity. Without an intervention/control witness, Causal says **Not tested**.

The conveyor's authored fact band is not a universal layer schedule. Likewise,
[ladder.py](https://github.com/chrishayuk/the-mechanism/blob/main/ladder.py)
studies a bounded toy linear reader: its results do not license labelling a real
operation as irreducible computation. Model memory edges, lookup replacements,
candidate routing and matrix multiplication avoided remain unavailable until
an appropriate runner or research provider supplies their evidence.

## 21. Extraction workspace — model before execution

The `/extract` workspace makes source admission and container structure visible
before a run. It follows the current VINDEX3 roadmap's **coexistence** rule:
legacy `extract-index` still produces VINDEX2; VINDEX3 explicitly uses
`plan → encode → inspect`. Factory recipes/build records remain a separate
legacy extraction pipeline, not a VINDEX3 mode silently substituted by the UI.

Source, Admission, Model, Representations, Verify and Factory are coordinated
views over actual planner/container/build documents. No fixture completion
animation represents an extraction job. Model objects and interface edges come
from the SystemGraph, never model-family assumptions. A planner's placed graph
is labelled planned; a container inspection is labelled reconstructed.

Runner actions use the existing capability discovery and plan/read routes.
Hostnames confer no capability. Text-generation admission requires all three
of understood semantics, available operands, and supported execution. Whole-model
admission is displayed separately. Findings carry source/planner identity,
semantic class, carriage and declared/resolved values. Unknown schemas refuse.

The current encode route is not mounted; the UI provides a reviewable CLI
handoff and imports its inspection result. It does not implement an executor,
launch a publish/build job, or claim progress absent runner events. Container
inspection without a recorded rehash flag does not establish payload integrity;
structural coherence does not establish execution completeness or output parity.
Factory build outcomes expose their actual scope (checksum VERIFY, not numeric
reconstruction or logit parity). No artifact publication is launched by this UI.

See the [UI extraction workflow](../observatory/README.md#extraction-workspace)
for connection and implementation details.

## 22. Research lenses and the first measured record (2026-09-20)

OBS records what happened. Lenses interpret distinct aspects of that record;
they do not add interpretive semantics to the executor tap. Anatomist's
[Context Map](https://github.com/chrishayuk/chuk-kv-anatomist/tree/main/src/components/context-map)
is the interaction reference: select a token, scrub depth, inspect a mechanism,
then test it. MODEL / RUN / WHAT IF remain the conceptual planes; Context is
the first research scene within a run.

The implementation sequence is **Context → Logit Lens → DLA Heatmap → Head
Content Projection → Layer/Run Comparison → Intervention**, followed by
Relations, Routing/Experts and REPRESENT. All preserve the same selected
run/token/layer/site, extending selection with head/expert/span only when the
corresponding evidence exists. State vocabulary, direct-write vocabulary and
future causal effects are distinct lenses, never interchangeable readouts.

The initial Context view uses Standard carrier norms, applied-write norms and
predeclared raw head-row probes. Its color scale is fixed across the recorded
run for the selected boundary; missing values are unavailable, not zeros.
No probability, entropy, specificity, semantic field, head attribution or
causal effect is derived from these summaries. Full logit lens needs recorded
state payloads and an explicitly identified normalization/head operation;
DLA/head content need the separate per-head capture rung.

The first real-record integration is Granite 4.2 3B: `The capital of France is`,
five positions, 40 layers, 400 writes, 804 events, greedy output ` Paris`.
All captured scalar values survive adaptation/reopen; the replay audit checks
33 prefix boundaries. A separate control session produces bit-identical finite
vocabulary logits at every prompt position under the same prepared production
CPU image. The record, parity witness and replay audit are separate hash-bound
[evidence objects](../observatory/public/recordings/granite-standard.json).
This establishes recorded integration, not live-stream behavior or HF parity.

Run-wide execution provenance is persistently accessible: lowering, stored
representation, codec, pinned physical form, arithmetic arm, K-quant mode,
fingerprint and basis. Aggregated realization classes must not be assigned to
an individual selected operation without an operand-to-operation record. This
Granite image includes BF16→Q8 requantization; its parity control is not a claim
of equivalence to all-BF16 or reference-backend execution.

### Anatomist parity pass / UI analysis record

The UI now implements connected Logits, Heads, Content, Compare, KV anatomy and
recorded Experiment lenses, plus richer Context overlays. Analyses are separate
source-hash-bound sidecars and cannot upgrade a synthetic record to measured
provenance. See [the UI lens contract and parity status](../observatory/LENSES.md).
The current real Granite Standard capture remains stats/raw-probe only; rich
fixture walkthroughs do not count as real DLA, normalized readout or intervention
witnesses. This pass changes no frozen executor contract.


### Measured Granite vocabulary integration

The measured Granite run now has a normalized vocabulary sidecar at all 400
carrier-write sites. These are post-run head-only readouts of states captured
from the same canonical traversal. Both observation/output parity and final
carrier/readout parity pass bit-for-bit at all five prompt positions. The web
player opens this real Logits lens automatically. Per-head and intervention
capabilities remain separate outstanding rungs. See the UI lens contract for
capture costs, source binding, replay and the witness details.
