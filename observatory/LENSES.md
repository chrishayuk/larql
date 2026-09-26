# Observatory analysis lenses

OBS records execution. A lens is a separate, source-bound interpretation of that
record. The browser never runs a second model, derives full probabilities from
Standard probes, or fabricates missing attention/intervention results.

## Canonical VINDEX3 readouts

Opening `larql.run-record.v1` JSONL attaches its `head-v1` readouts automatically.
These retain log-probabilities and target ranks; probability is displayed as
`exp(logprob)`. Raw logits and full-vocabulary entropy are absent and remain
unavailable. The adapted vocabulary uses `value_kind: "logprob"`, optional
`logprob` instead of `logit`, optional entropy, and the matching write sequence
for replay visibility. Competition ranks can tie. Canonical readouts cannot be
replaced by a sidecar. See [OBSERVATORY-V3-BRIDGE-1](V3-BRIDGE.md).

## Using the instrument

Open any of the four authored stories and choose **Lenses**. The same token,
layer, boundary and selected head carry through the investigation:

1. **Logits**: choose a tracked token and probability/rank/logit; select a depth
   to inspect retained vocabulary predictions and full-vocabulary entropy.
2. **Heads**: inspect signed layer × head DLA and the absolute-ranked head strip.
   Click a cell to select its actual attention site and open **Content**.
3. **Content**: token-direction coefficients, head source weights, signal
   dimensionality (95%, 99%, 99.9%) and orthogonality diagnostics.
4. **Experiment**: inspect a recorded fork's before/after distributions,
   probability differences and, when supplied, full-vocabulary KL. The fixture
   example is L23/H3. This is not a remote intervention command.
5. **KV anatomy**: K/V accounting and a separately scoped content-size estimate.
   This is not a claim that a byte slice can be removed from the cache.
6. **Compare**: select any two retained boundaries or import another recording
   and its analysis. Sites align by token position and semantic role, not time.

**Context** adds token hover/focus previews, retained top-k selection, entropy,
normalized-entropy specificity, query-head attention overlays, and provider
annotated spans. A depth scrubber is available across scenes. Standard captures
retain norm/write/probe colour modes without pretending to support richer modes.
The global matrix now uses a fixed run-wide write scale instead of clipping
real-model magnitudes to the fixture's 0.6 scale.

## Analysis interchange v1 (UI/runner draft, not executor ABI)

Use **Lenses → Open analysis** after opening a recording. Files are limited to
10 MB. The schema is `larql.observatory.lenses.v1`; its typed definition and
validation live in [lib/lenses.ts](lib/lenses.ts). A generated example is
[public/fixtures/answer.lenses.json](public/fixtures/answer.lenses.json).

Required envelope:

```json
{
  "schema": "larql.observatory.lenses.v1",
  "run_id": "same-as-source-record",
  "source_sha256": "64 lowercase hex characters",
  "provenance": "executor",
  "provider": "named-analysis-provider/version"
}
```

Add at least one of the optional sections below. `source_sha256` hashes the exact
UTF-8 source record bytes, including whitespace. Save original record preserves
those bytes. Save analysis exports the sidecar separately. A source hash binds
identity; it does not authenticate or scientifically validate a provider.
Synthetic analysis cannot attach to an executor record.

A site key is `position:layer:role:carrier:topology`, e.g.
`4:23:attention_write:main:residual`. Every site must occur in the source record.
Duplicate vocabulary/head entries refuse. This draft supports the UI's existing
Single carrier adapter; it does not reinterpret Bundle or History topology.

- `vocabulary`: `method`, `basis`, `vocab_size`, `rows`. Each row has `site`,
  `top`, `targets`, `entropy` (full vocabulary, nats). Each prediction has
  `token_id`, `token`, `probability`, `logit`, `rank` (1-based). Top rows must be
  contiguous and sorted. Retained mass may be less than one and is never
  re-normalized. Method must describe final norm, scaling/soft-capping, and the
  exact carrier boundary; Standard's post-add/pre-layer-scale state matters.
- `heads`: `method`, `basis`, `rows`. Each row has `site`, `head`, `target`,
  `dla`, `sources` (`position`, `weight`), optional `content`. Only observed
  attention sites are valid. Missing source mass may reflect sinks or partial
  capture; it is not redistributed. DLA is attribution, not causality.
- Head `content`: `method`, `basis`, `vector_norm`, `projections` (`token`,
  `token_id`, signed `coefficient`). Optional `dimensionality` declares a method,
  total `dimensions`, and ordered `d95`, `d99`, `d999`. Optional `orthogonality`
  declares a method and `pairs` (`label`, `cosine`). Nonorthogonal token directions
  must not be treated as an orthonormal energy decomposition.
- `spans`: inclusive `start`, `end`, `label`, `method`. Displayed as provider
  annotations, not automatically discovered universal semantics.
- `kv`: source `sequence`, accounting `method`, `total_bytes`, `key_bytes`,
  `value_bytes` (K + V must equal total). Optional `content` has `bytes`, `method`,
  `scope`, `witness`. It compares representations, not physical memory slices.
- `experiments`: `site`, `head`, `kind` (`ablate`, `patch`, `inject`), `parent`,
  distinct `fork`, `method`, `witness`, `target`, `before`, `after` prediction
  lists. Optional `kl` declares `direction: before-to-after`,
  `scope: full-vocabulary`, and `nats`. Missing token entries are unknown, never
  zero. KL cannot be calculated from retained top-k. These are provider-reported
  results: inspection does not execute a fork or verify its controls.

Replay reveals a site analysis only when that site's source observation is
visible. KV accounting waits for its declared sequence. Imported comparison
records are complete independent records; they are not replay-clock aligned.
Raw cross-run probe differences require matching reader identity, not just an
identical displayed token label. Projection overlay continues to require the
same declared basis.

## Parity status / September 2026

| Analytical primitive | UI | Measured Granite Standard record |
| --- | --- | --- |
| Context token map, hover/focus, depth scrub | Implemented | Norms, applied writes, raw probes |
| Two-boundary / two-run comparison | Implemented, record import | Scalar stats and compatible raw probes |
| Logit lens, entropy, specificity, top-k | Implemented, sidecar import | Measured at all 400 carrier writes; 100,352-token vocabulary |
| DLA heatmap and ranked heads | Implemented, sidecar import | Needs separate per-head rung |
| Head content, dimensionality, orthogonality | Implemented, sidecar import | Needs head-analysis provider |
| Query attention and annotated spans | Implemented, sidecar import | Needs supporting provider evidence |
| KV accounting/content-size comparison | Implemented, sidecar import | Needs accounting and scoped witness |
| Recorded intervention distributions/KL | Implemented, sidecar import | Needs canonical fork evidence |
| Launch ablate/patch/inject from browser | Not implemented | No intervention runner |
| Knowledge-store/window browsing | Not implemented | No store capability |
| Boundary residual loading | Not implemented | No payload capability |

The richer stories are explicitly synthetic, six-token toy computations. They
exercise UI interaction and parser behavior; they are not acceptance witnesses
for real interpretability. The current real Granite record includes a separate measured vocabulary lens.
The previous Standard-only record is preserved in `public/recordings/archive/`.
No per-head tap or frozen Standard contract was changed. The prepared image
exposes a head-only readout method using its existing norm/head implementation.
Gemma 3 4B remains a target test bed, not a demonstrated capture in this checkout.

Regenerate analysis fixtures with `python3 scripts/make-lens-fixtures.py` after
changing base fixtures; source hashes must be regenerated too. `npm test` covers
source binding, invalid science fields, replay visibility, partial evidence,
workflow identity, and the deterministic toy calculations.


## Real Granite vocabulary witness

`observatory_record --logit-lens CONTAINER OUTPUT.json PROMPT [PROBE_TEXT ...]`
adds an explicitly intrusive, bounded (64 MiB maximum) owned-carrier capture to
the same observed traversal. After execution and an independent parity control,
it applies the recorded layer scale, prepared final norm, pinned output head,
multiplier and softcap to each captured state. It computes a full-vocabulary
f64 softmax, full ranks and entropy, retaining top-20 plus the declared targets
and actual output token. No transformer layers are re-executed for this analysis.

The exporter refuses publication unless the last captured carrier readout is
bit-identical to canonical output logits at **every** prompt position. The
`readout_witness` in the lens sidecar records this check; its canonical hash also
matches the separate observed/unobserved parity witness. Intermediate readouts
remain projected vocabulary interpretations, not proof of causal knowledge.

The September 20 Granite capture contains 400 sites over five positions and
40 layers. The selected token ` Paris` first becomes top-1 at the final prompt
position after FFN L35 (zero-based); final probability is 0.9061932565969187.
Early and intermediate confidence need not increase monotonically.

The exporter also saves local `*.carriers.f32` snapshots with a SHA-256 identity
in the analysis sidecar. These full vectors are **not** uploaded to Fly. The web
player consumes the published statistics and vocabulary results only.
Per-head DLA, attention sources, KV content estimates and intervention evidence
remain unavailable on this real run.

## Residual Atlas / measured token geography

Open the measured Granite run to enter **Atlas**. Its eight fixed places are
stored output-token directions (Paris, Berlin, France, London, Rome, city,
country, capital). The traveller is the recorded carrier after its declared
layer scale and prepared final normalization, normalized to unit L2 length.
Both states and landmarks use the same fixed two-dimensional orthonormal plane:
Paris direction, then the orthogonal component of Berlin (Gram–Schmidt).
Axes, full basis values and hashes are declared in `token_map.basis`.

Scrub depth or play the record to follow the traveller; select a stop to inspect
its real write. Pan by dragging, zoom with +/−, or follow the traveller. Labels
thin out when crowded and reappear on focus/hover or zoom. Selecting an anchor shows full-space cosine and projected separation. The separate
Answer Compass shows the selected state’s top five vocabulary candidates, probability,
rank, logit and change from the previous contiguous write. Geometry is not probability. The
route stops at the selected write and distinguishes attention and FFN stages.
Missing boundaries are not connected by an invented line.

The similarity graph is the union of each landmark's two nearest neighbours
by **full-dimensional stored-row cosine**, over these eight selected landmarks
only. It is a declared derived graph, not execution connectivity, learned
relations, or causal evidence. The neighbor list is navigable independently
of screen distance. A change in camera never changes adjacency or coordinates.

No semantic districts, FFN feature labels, source-token transport bridges,
recurrent routes, or causal road closures are claimed for this capture.
Those are extensions that require appropriate evidence providers. This is the
first token-direction geography for the Atlas, not an operational-distance map.

The offline exporter reuses the saved carriers without running inference:

```bash
target/release/examples/observatory_record --token-map \
  MODEL.vindex3 SOURCE.json SOURCE.lenses.json SOURCE.carriers.f32 \
  WITH-ATLAS.lenses.json ' Paris' ' Berlin' ' France' ' London' \
  ' Rome' ' city' ' country' ' capital'
```

It verifies the source, carrier payload, tokenizer, plan, container payloads and
prepared realization identity before applying the final norm. Only selected
stored head rows are widened. Full vectors remain local. The output extends
the same analysis sidecar; Standard events, normalized readout results and
parity witnesses are unchanged.

### Position / Direction / Destination

The Atlas keeps one persistent map. Map-layer toggles control Answer Readout, recorded remainder, and destination; hiding the readout preserves the map camera and width.
Token-direction anchors are geometric references, not destinations. There is no
carrier-to-token destination line. The Answer Readout ring encodes the selected
token's full-vocabulary probability; the ranking retains its original probability
mass. Its ring is not an angular geometry. Numeric token-axis bearing is available
only for captured landmark directions and is labelled relative to the fixed +X axis.
Missing token measurements and missing previous-write evidence remain unavailable.

Destination means the actual measured terminal carrier for the selected position.
The endpoint label uses that state's coordinates and vocabulary readout, never the
token vector. For a complete recording, an explicitly labelled retrospective overlay shows the
measured remainder as dotted straight segments and the endpoint as a hollow reticle
even when an earlier write is selected. The current readout still uses only that
selected write. Turning off the retrospective layers hides future information from
the map. Incomplete records never gain a future overlay. For non-final prompt positions,
the label is the terminal state's readout, not a separately generated output token.
No basin, commitment, convergence, or destination-prediction claim is made from this
single recording. Such views require separately identified multi-run evidence.

### Recorded route treatment

Solid piecewise segments connect traversed carrier observations. Dotted segments
are the completed recording's measured remainder, never predictions or interpolated
observations. Missing write boundaries are never connected. A filled current-position
marker reaches the hollow terminal reticle. TERMINAL STATE is the primary endpoint
label; token and probability are secondary readout text. The write timeline permits
explicit navigation to any recorded boundary while preserving global selection.

The map dominates the Atlas workspace; detailed operation inspection remains in
Trace. Answer Readout uses a probability ring and ranked bars, not angular token
placement. Camera, landmark, answer selection and readout mounting survive overlay
toggles. The fixed model token-direction basis is unchanged. Permanent metadata shows
carrier dimensionality, basis identity and current off-plane squared-norm fraction
(1 − x² − y² for these unit-normalized states and orthonormal axes). This is not
route variance retained, which is explicitly unavailable. No PCA, terrain, clusters,
operational roads or new semantic labels are inferred by this visual pass.

### Route framing and readout context

Complete recordings open with the selected position's full recorded trajectory
framed, instead of fitting distant token anchors. This is a camera transform only;
the basis and coordinates stay unchanged. The frame remains fixed while scrubbing
and toggling overlays. Choosing another position frames that recorded route; manual
pan/zoom remains available, and All anchors restores the vocabulary-wide view.

The readout states the source token and position explicitly. For the real Granite
record, position 2 (“of”) ends with “the” at 20.78%; position 4 (“is”) ends with
“Paris” at 90.62%. These are different recorded states. The final-position shortcut
seeks the same layer/write for position 4, rather than relabelling the current state.


### Real Granite heads

`/heads` opens a new canonical Granite recording with supplemental intrusive head
capture. It covers 40 query heads × 40 layers × 5 positions. Click a heatmap cell
or ranked head to select its attention write and open Content. The inspector shows
that head's measured source weights; otherwise it shows a labelled arithmetic mean
of the captured heads at the selected attention site. Replay only exposes heads
whose source write is visible.

The heatmap targets the first predeclared token (Paris). Content shows eight
predeclared token directions, not full-vocabulary top-k. The scores are derived
raw stored-weight projections, without final normalization. They cannot be read as
probabilities, additive normalized-logit changes or ablation effects. Actual source
weights and weighted-V vectors come from the canonical attention loop. Full-logit
parity and reconstruction of all actual attention writes are separately checked.
See [head capture](../docs/v3-observatory-head-capture.md) for the exact contract.
