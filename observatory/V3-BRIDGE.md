# OBSERVATORY-V3-BRIDGE-1

The Observatory reads `larql.run-record.v1` directly from the unchanged JSONL
written by `larql vindex3 observe`. No model is rerun and no Observatory export
is required. Live execution is deliberately outside this milestone.

## Open a recording

Choose **Open recording / Open run.jsonl**. JSONL and the existing Standard and
synthetic JSON formats enter through `lib/ingestion.ts` (`openRecordingText`).
Comparison import uses the same entry point. A canonical file's embedded
`head-v1` readouts attach automatically; open **Lenses → Logits** to inspect
probability, rank or recorded log-probability through depth. The inspector
shows the exact source precision for norms, write magnitude and coordinates.

Files remain in the browser. Successfully opened canonical bytes are retained
in this tab's session storage for refresh; if browser quota/privacy settings
prevent this, the UI says to save and reopen instead. **Save original record**
returns the exact input bytes as `.jsonl`. No event is reserialized for export. Canonical readouts cannot be replaced by an
analysis sidecar; the JSONL remains their authority.

## One ingestion boundary

`JSONL bytes → framing / integrity / contract validation → immutable run → views`

`lib/vindex3-record.ts` mirrors the runner contract in
`crates/larql-inference/src/vindex3/record.rs`, not the internal model operators.
It has no model-family dispatch, tokenizer, inference, norm/head calculation,
projection-basis generation, coordinate fitting or softmax implementation.
A future framed transport should supply this same logical evidence boundary;
no live transport or execution endpoint is implemented here.

- Header schema, basis identity/dimensions, execution provenance, methods,
  numeric domains, event position/site/order and receipt are validated.
- Event-line SHA-256 is checked over the original lines, matching Rust's newline
  convention. The RunProvenance fingerprint is checked in Rust struct field
  order, including realization classes, arithmetic arm and basis.
- Rust u64 counters/timestamps never pass through an imprecise JavaScript
  number. Original bytes, including whitespace, are retained independently.
- `carrier_stats` and optional `readout` join only their matching explicit
  `carrier_write`. A structural `ffn_done` never manufactures an FFN write.
  Each wire event keeps its original sequence/time; only the receipt maps to
  an additional terminal UI event. No wall-clock/site duration is invented.
- Readouts become visible with their write. The shared prefix reducer supports
  backward/forward scrubbing; immutable evidence prevents view controls from
  altering values. The camera transforms recorded coordinates using a fixed
  full-record domain. It does not refit or change the underlying trajectory.
- Probability is `exp(recorded logprob)` for presentation. Top-k mass is never
  renormalized. Target ranks retain the executor's competition-ranking rule;
  tied tokens may share a rank. Raw logits and entropy remain unavailable.

## Refusal and degraded evidence

Unknown schema/event/lens method, missing basis, malformed site identity,
unsupported topology, shape mismatch, invalid numbers, sequence disorder,
corrupt hashes or contradictory receipts refuse before mounting the new run.
The initial geometry adapter admits Single carriers with three recorded
coordinates. Bundle/History or other basis dimensions refuse explicitly.
Limits: 10 MB, 100,000 events, 256 positions/layers, 128 retained lens tokens.

A missing receipt is an unsealed stream and refuses. A valid receipt with
`complete:false` opens as an explicitly incomplete prefix; pending writes are
not plotted. `lens_failure` is a visible degraded state. `live_dropped` is shown
separately: lost live deliveries do not turn the intact lossless file into a
lossy record. Importing a complete file does not establish observed/unobserved
parity, independent model identity, or authenticity of its producer.

## Facts the current wire format does not carry

The visible provenance panel names model identity, component, run, schema,
prompt token IDs, basis ID/provider/hash, methods, head-pass count, completeness,
live loss and source/provenance hashes. The current record does **not** include:

- Container content hash, tokenizer identity, prompt text or token spellings.
- Token IDs for extra generated positions, or the final sampled output.
- Independent full program inventory, per-operation realization assignment,
  head attention sources, entropy or raw full-vocabulary logits.

Those facts are labelled absent. Tokens use `#ID` and extra positions use an
explicit unknown-token label. Model structure is labelled **observed sites**,
not an independently extracted operation graph. Completion is the receipt's
claim, checked for structural contradictions; the UI cannot certify an
inventory the recording does not contain. The final retained head candidate
is not relabelled as generated output.

## Paris golden

`public/recordings/gemma-paris-v3.jsonl` is a byte-for-byte copy of the existing
Gemma Paris CLI capture, run ID `gemma3-4b-france-lens`. Its SHA-256 is:

`12cdf4d9d06f5675233d71556c2158702cff88dd89efe42af72ab9dfb345dbd4`

Capture-study context identifies prompt “The capital of France is” and token
9079 as “ Paris”; these strings are not fields in the source recording and are
not silently inserted into its evidence. The file contains 1,446 wire events,
408 carrier writes across six positions and 34 layers, and 204 FFN lens readouts.
At final prompt position 5, token 9079 has rank 14,055 at FFN layer 0, rank 1
at layer 24, log p −0.0010202578566627096 at layer 26, and log p
−0.22265967015508892 at the final layer 33. Confidence is not monotonic.

`tests/golden/gemma-paris-v3.expected.json` freezes source stats, structural
writes, both tracked tokens' lens values at representative boundaries, and the
terminal logits boundary/receipt. Tests use these fixed values, compare every
write to its source, render the actual inspector, scrub in both directions,
reopen the same bytes, exercise camera controls, check readout visibility and
refuse corrupted or incompatible records. They do not rerun the model or infer
that record integrity proves execution parity.

Run `npm test`, `npm run typecheck`, `npm run lint`, and `npm run build` from
this directory. The golden and bridge tests are in `tests/vindex3-record.test.mjs`.

## Validation result

The bridge's frozen replay checks and server rendering of the actual inspector
passed, together with the existing app tests, TypeScript, lint and production
build. Local HTTP checks served the app and golden file successfully; the served
file matched the frozen source hash. Browser automation was unavailable in this
session, so interactive click/refresh and visual QA are not claimed. Session
restore reuses the same byte ingestion function covered by the reopen tests.

The [browser acceptance kit](qa/v3-bridge-qa-1/ACCEPTANCE.md) supplies the
unchanged Paris recording, clearly labelled degraded/refused fixtures, and
exact replay checkpoints. Manual acceptance remains pending; complete that
gate before adding live execution.
