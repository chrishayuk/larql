# LARQL Observatory UI

A HAUSE-native research workbench for the model as a computational database.
This directory owns UI, synthetic studies, measured Standard records, selection
and replay. The recording runner consumes the frozen
[V3-OBS-1](../docs/v3-obs-1-carrier-observation.md) tap without changing it.

## Native VINDEX3 recordings

**OBSERVATORY-V3-BRIDGE-1** opens the original `vindex3 observe --record run.jsonl`
file directly. Choose **Open run.jsonl**, or **Open Paris recording** for the
unchanged Gemma golden. Receipt hashes, provenance and site identities are
validated; embedded answer readouts drive Lenses without a sidecar or inference.
See [the bridge contract, limits and golden replay](V3-BRIDGE.md). Live execution
remains disconnected. Canonical imports are retained in this browser tab for
refresh when session storage is available.

## Run

```sh
cd observatory
npm ci
npm run dev
```

Open the Local URL printed by the development server. The app starts at
Prompt → Run fixture → Watch, or **Open measured run** for the recorded Granite
study. The browser does not execute a model. An arbitrary
prompt with no matching fixture is refused instead of generating invented data.

Live test beds: **Gemma 3 4B** (primary user model) and **Granite 4.2 3B**.
The amended executor freeze names Granite as the primary witness; OLMo2-1B is a secondary single-stream subject. The
fixtures are authored 32-layer structures and do not assert any real model's
layer count, tokenization or measurements.

## What works

- Four synthetic stories: answer formation, attention/FFN transition, authored
  ADDRESS, and a base/target counterfactual (two records).
- Prompt-led arrival; the Map owns the live field and the layer rail sits below.
- Context token view with a layer scrubber, fixed magnitude scales and shared selection.
- Measured Granite recording: 400 carrier writes, 804 events, output ` Paris`,
  independent same-backend observed/unobserved bit-parity witness.
- Run-wide pinned realizations, lowering, arithmetic arm and basis in a persistent
  expandable provenance strip; no per-operation realization is inferred.
- Map, Trace, Graph, matrix, inspector, selected-token readouts/differences and rail
  share a selected position/site. Fixed 2D/oblique-3D views use recorded bases.
- Model plane shows the fixture's declared operation graph without playback;
  Execution overlays visited sites; Alternatives admits only a recorded pair.
- Play/pause, site/layer stepping, scrub, 0.25×/1×/4×/Instant, JSON record
  download/open, missing-evidence reporting, and keyboard controls.
- Actual HAUSE tokens and Evidence/Refusal forms pinned to revision
  `d8ff653e2dfed6267305ca9568225b0973868918`.

Standard maps to the frozen stats observer: carrier norm, applied-write norm,
fixed projection and predeclared raw selected-token head-row probes. **Attention source
capture is unavailable under Standard.** Story B has explicitly synthetic
head-mean weights to exercise the eventual Rich presentation; Rich requires a
separate per-head executor rung. No full-vocabulary probability or rank is
inferred from selected logits.

The graph distinguishes program dependencies from derived readouts; weights
are not causal contribution. Matrix-multiply elimination, expert placement,
WASM, graph walks, interventions and equivalence witnesses are not simulated
as available runner capabilities.

## Adapter boundary

`lib/record.ts` is a strict **frontend fixture format**,
`larql.observatory.fixture.v1`, not an executor ABI. `parseRecording` validates
input; `reduceEvents` is the shared event-to-state reducer for playback and
future adapted streams. The executor does not emit the web envelope: the
`observatory_record` runner adds run identity, sequence and monotonic timestamps.

`program` declares the fixture's real sites, independently of its event log;
no view should invent an FFN write from a layer boundary. Unsupported topology
is refused by this initial synthetic adapter. Bundle/history UI and live
VINDEX3 adapters remain follow-on work and must preserve topology identity.

Live delivery may be lossy and nonblocking. The replay record must be lossless
within its declared scope or fail completeness explicitly. Local JSON import
has a 10 MB limit and accepts canonical VINDEX3 JSONL, fixture records and the draft Standard bridge described in [STANDARD-ADAPTER.md](STANDARD-ADAPTER.md). It never uploads
records. Recorded vectors/projections are not re-executed or refitted on replay.

The four fixtures are reproducibly authored by:

```sh
python3 scripts/make-fixtures.py
```

No fixture is a parity witness or a reproduction of sealed ADDRESS-BUILD.
Coordinates, norms, selected-token values and directional writes derive from
the same authored six-dimensional carrier. A frozen fixture linear reader
defines both state readouts and write projections; no model is measured.
Trace and Graph distinguish READABLE, WRITTEN and CAUSAL; the last is always
not tested because there is no intervention witness. Attention weights in B are authored synthetic distributions.

## Checks

```sh
npm test
npm run typecheck
npm run lint
npm run build
```

Tests cover round-trip coordinates and prefix replay, reconnect duplicates,
conflicting identities, end-of-run loss, masked synthetic source mass,
semantic alignment/basis mismatch, mixer-only programs, site/layer stepping,
and invalid record rejection. Typecheck covers app/components/adapter; the
Sites scaffold's Worker is built by vinext. In-app browser QA was unavailable
in the implementation session; no visual browser pass is claimed.

See [product specification](../docs/observatory.md) and
[protocol draft](../docs/vindex3-observation-contract.md) for the live contract,
future model/representation/placement capabilities, and acceptance gates.

## Visual lineage

[the-mechanism](https://github.com/chrishayuk/the-mechanism) informs the visual
vocabulary: spot, edge, address, route, write, worldline, resolve, walk,
compute, patch and read. Its seeded-vector graphics and separation of
readability, directional writes and ablation motivate evidence-bound views.
The Observatory does not inherit its experiment verdicts or Python execution
paths. Provider-labelled addresses/relations and knowledge edges appear only
when their evidence exists. The current Model plane contains declared fixture
operations, not an invented entity knowledge graph.

## Standard record integration

“Open recording” now dispatches to an explicit Standard adapter. It mirrors
`WriteStats` without treating raw probes as normalized logits, inventing
embedding samples, or applying the layer scale to recorded pre-scale values.
Missing entering-carrier norms, directional writes, source weights and site
timing stay unavailable. Runner declarations are labelled as such: importing
a file does not validate its model identity or prove output parity.

The original input is retained for download and exact replay. The Standard
bridge is a draft UI handoff, not a frozen executor serializer. The current
executor exports no matching runner file/endpoint in this checkout. Adapter
checks use authored contract cases, not a real model witness. Live transport
and real observed/unobserved parity remain pending the runner handoff.

See [the bridge and handoff checklist](STANDARD-ADAPTER.md).

## Extraction workspace

Open `/extract`, or select **Extract** in the header. This extends the Model
plane with source planning and container inspection, before any inference:

- **Source:** Gemma 3 4B / Granite 4.2 3B shortcuts, up to eight local or HF
  artifacts, whole-model versus text-generation scope, output path.
- **Admission:** actual SystemPlan schema 6, capability closure, attributed
  findings, declared/resolved values, carriage and source revision. An admitted
  text capability requires admissible + available + supported.
- **Model:** component/object browser and declared cross-component interfaces
  from a plan or imported `vindex3 inspect --json` result. Planned structure and
  reconstructed container structure are labelled separately.
- **Representations:** physical directory, encoding, byte counts, full hashes,
  profiles and authority. Optional read-only facts from the bound runner are
  labelled independently from the extraction source.
- **Verify:** inspection defects and CLI verification handoff. The inspection
  JSON does not state whether rehashing was requested; zero defects never
  silently becomes a payload-integrity, execution-completeness or parity pass.
- **Factory:** import the existing Factory BuildRecord and inspect terminal
  failure/passed status, output sizes and release flags. No fabricated progress.

The browser calls the existing `GET /v1/capabilities`, `POST /v1/plan` with
`{sources}`, and capability-gated `GET /v1/{components,representations,provenance,authority}`.
Connection is direct and requires the runner's CORS/network policy to allow it.
Bearer credentials stay in component memory and go only to the entered runner;
there is no hosted proxy, persisted token, or automatic source upload.

The encode route is reserved but not mounted in the current server, and has no
request contract for this UI to implement. Encoding therefore remains an
explicit, safely quoted CLI command gated by the imported/returned plan verdict;
remote HF revisions are pinned from that plan when available. The CLI rechecks
admission. Import the resulting inspection to continue. There is no job dispatch,
server filesystem browser, model load, inference, publication, or cancellation
of server work hidden behind these controls. “Stop waiting” aborts the browser
request only. Late responses cannot overwrite a newer source/connection/import.

Roadmap authorities: `ROADMAP.md` VINDEX3 coexistence, `docs/vindex3-remote-source.md`,
`docs/vindex-factory.md`; implementation authority is the current Rust route ledger
and serializers. Legacy `extract-index` remains VINDEX2; Factory's current VERIFY
is checksum integrity only. No executor or extraction runtime code is changed.

## Fly deployment

The UI can also run as a standalone Node server. The Fly build selects
`OBSERVATORY_TARGET=fly`, excludes the Cloudflare/Sites runtime plugins, and
emits vinext's standalone server with its runtime dependencies and public assets.
The default build still targets Sites. No model or runner credentials enter the
Docker image; its build context is this UI directory only.

```sh
cd observatory
fly deploy --remote-only --ha=false
```

`fly.toml` names `larql-observatory` in `lhr`, with one shared CPU / 512 MB
Machine, HTTPS, automatic start/stop, and an HTTP health check. The multi-stage
Dockerfile builds with Node 22 and runs as the unprivileged `node` user. HAUSE
remains pinned to the lockfile's Git revision; the builder rewrites GitHub SSH
fetches to HTTPS rather than requiring an SSH key.

For a local production check:

```sh
OBSERVATORY_TARGET=fly npm run build
cd dist/standalone
PORT=8080 node server.js
```

The Fly URL serves the UI. Live inference/extraction remains on the separately
connected VINDEX3 runner. This deployment does not proxy runner credentials or
upload locally imported records.

## Real-record witness

The historical `observatory_record` exporter commands below and in the adapter
documents describe separate local capture tooling. Those Rust changes are not
part of the frontend/bridge; use the merged `larql vindex3 observe` CLI for
new canonical recordings. The existing Granite evidence remains inspectable.

See [STANDARD-ADAPTER.md](STANDARD-ADAPTER.md) for capture and audit commands.
The checked-in [Granite record](public/recordings/granite-standard.json),
[parity witness](public/recordings/granite-standard.parity.json) and
[replay audit](public/recordings/granite-standard.replay-audit.json) are separate
evidence objects bound by the record's SHA-256. Tests validate those bindings;
they do not rerun inference. This closes the first recorded-data integration,
not live transport, nonblocking queues or HF parity. Supplemental per-head capture is documented separately below.

## Research lenses

[Anatomist](https://github.com/chrishayuk/chuk-kv-anatomist) supplies the interaction
reference: token Context Map, depth scrubber, then mechanism drill-down. The first
Context view shows carrier norm, applied-write norm and selected raw head-row
probe. It uses recorded values with a fixed run-wide scale for each boundary.
It does not infer logit probabilities, entropy, specificity or semantic spans.

Next after the measured normalized Logit Lens: DLA layer/head heatmap → head content
projection → layer/run comparison → intervention. Relations, routing/experts and
REPRESENT follow. These are lenses over the shared selected run/token/site, not
changes to OBS. Each needs a separately declared input/evidence contract; raw
Standard probes cannot satisfy a full-vocabulary logit lens or DLA.

## Anatomist research workflow

The **Lenses** scene connects Logits → Heads → Content → Experiment while
preserving selection. Context adds entropy/specificity/query-attention modes,
token preview and annotated spans when evidence is present. Compare accepts
arbitrary depths and imported runs; KV anatomy separates accounting from scoped
content estimates. See [LENSES.md](LENSES.md) for usage, the source-bound analysis
interchange and the precise parity status. Rich authored walkthroughs are
synthetic. The measured Granite run now adds a separate normalized vocabulary
lens at all 400 writes, with final readout parity checked at all five positions.
Real Granite now also carries 8,000 per-head observations; `/heads` opens the measured heatmap and content drill-down. Intervention integration remains outstanding.


Generate a real normalized lens alongside Standard capture with:

```bash
target/release/examples/observatory_record --logit-lens \
  ~/chris-models/granite-4.2-3b.s6.vindex3 \
  /tmp/granite-with-lens.json 'The capital of France is' ' Paris' ' Berlin'
```

This writes the source record, parity witness, source-bound `.lenses.json`, and
local `.carriers.f32` snapshots. The measured-run button opens the Residual Atlas:
eight fixed token directions and all 400 captured carrier states, with depth
selection shared by Trace, Context and Lenses. The normalized Logits lens remains
available in Lenses. Context also supports real entropy, specificity and top-20
vocabulary inspection. Standard raw probes remain separately labelled.

The Atlas supports pan, zoom, following the current state, selecting a token
landmark, and inspecting each attention/FFN write. Its neighbor graph uses
full-dimensional cosine among the retained token directions; it does not claim
operational adjacency or causal routes. Projected separation, full-space cosine,
and vocabulary probability are shown separately. See [LENSES.md](LENSES.md) for
the offline export command and fixed-basis contract. Exporting the Atlas uses
saved carrier snapshots and executes no inference.
