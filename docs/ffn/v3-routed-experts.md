# V3 routed expert operation providers

VINDEX3 can place selected, unweighted expert transforms on CPU workers while
keeping normalization, routing, selection, attention, KV and deterministic
weighted reduction in the coordinator. Workers use the same production expert
transform as local V3 execution. Expert down bias is applied **before** weighting.

## Current scope

This first rung supports single-stream softmax stacks with packed MXFP4 routed
experts on every layer, CPU production lowering to F32, and no shared or latent
expert branches. GPT-OSS is the real-model target. K3's hybrid attention and
other expert representations, Metal coordinators/workers, discovery, resharding,
stream transport and carrier quantization remain separate work. Unsupported
plans refuse before loading the worker payload. No legacy remote MoE backend
is involved.

`--v3-ffn-shards` selects the dense or routed provider from the bound plan.
Routed placement accepts only `--v3-ffn-wire binary` (the default).
`--v3-profile PATH` records routed-operation and per-shard diagnostics beside
the existing per-position timings. See the [exact routed profiling protocol](../../bench/v3-routed-experts/PROFILE-1.md)
for byte accounting, parallel timing interpretation and measurement gates.

## GPT-OSS 20B example

Ranges on the CLI are inclusive. Protocol and execution slices use exclusive
ends. Each worker below owns half of the 32 experts in every one of 24 layers:

```bash
larql-server gpt-oss-20b.vindex3 --ffn-only --layers 0-23 --experts 0-15 --port 9181
larql-server gpt-oss-20b.vindex3 --ffn-only --layers 0-23 --experts 16-31 --port 9182
larql run gpt-oss-20b.vindex3 "The capital of France is" --emit-ids \
  --v3-ffn-shards http://localhost:9181,http://localhost:9182
```

Workers may own different layer ranges and expert ranges. Every routed layer
must have exactly one owner for every expert; gaps and overlaps refuse before
the coordinator loads its local payload. Shard URL order does not set numerical
reduction order. Selected experts are grouped by owner, groups dispatch in
parallel, and returned rows are accumulated in production selection order.

CPU workers widen their **owned** MXFP4 rows to F32 at preparation, so capacity
planning must use widened weight sizes. The coordinator loads no expert bank.
A preparation ledger declares each owned code, scale and bias byte range;
range reads validate the full tensor size and read only the declared window.
Worker residency counts expert matrices and biases, with no attention or head.
The current selection budget may conservatively price whole source streams;
physical preparation reads and resident matrices are bounded to owned rows.

## Authority and exact wire

- `GET /v1/vindex3/experts` describes artifact/plan identity, CPU numerical
  provider revision, model dimensions, half-open ownership, representation
  identities and the canonical serialized byte-range ledger.
- `POST /v1/vindex3/experts/open` validates that binding and returns version 1
  plus a 128-bit worker-incarnation handle. The handle is a correlation value;
  the existing HTTP bearer middleware remains the authentication boundary.
- `POST /v1/vindex3/experts/binary` sends exact little-endian F32 carriers on
  persistent HTTP connections. Bearer configuration uses the existing
  `--v3-shard-token-env` option. Redirects are refused.

Both frames have a 40-byte header: four-byte magic (`VEX1` request, `VEY1`
response), handle (16), sequence (8), layer (4), hidden width (4), count (4).
Requests append `count` U32 expert IDs and **one** hidden-width F32 input row.
Responses append `count` entries, each a U32 ID and an unweighted F32 output row.
There are no routing weights on the wire. Body sizes are:

```text
request:  40 + 4 × selected-on-worker + 4 × hidden
response: 40 + selected-on-worker × (4 + 4 × hidden)
```

Length, count, dimensions, finite values, ownership and response correlation
are checked. Missing, duplicated or unsolicited output IDs refuse. A stale
handle after worker restart refuses; there is no automatic rebind, retry,
local fallback or zero contribution. Workers are stateless, so repeated valid
requests compute the same transform.

Any remote step failure invalidates that coordinator session. Its last
committed position stays unchanged; recovery creates a fresh session and
replays the committed prefix. Separate sessions retain separate KV state.

## Validation

Tests cover three topologies, exact owned-byte reads (including biases),
bitwise continuation logits beyond a fixture's sliding window, interleaved
sessions, shuffled response order, floating-point order sensitivity, admission
failures, malformed frames, corrupt server replies and failed-step recovery.
Existing local routed forward oracles also gate the extracted transform.

The [CLI correctness driver](../../bench/v3-routed-experts/check_cli.py) records
binary hashes, artifact metadata hashes, bindings and emitted token IDs. It
runs prose, code and a 151-token prompt that crosses GPT-OSS's 128-token window.
These checks establish only the recorded correctness scope; they are not a
performance benchmark or a K3 validation.

GPT-OSS 20B passes the [recorded real-model CLI checks](../../bench/v3-routed-experts/README.md):
three prompts, including 151 input tokens, with 32 generated tokens each across
local, one-worker, two-way expert and mixed layer/expert layouts. Every candidate
matched the saved pre-refactor IDs. This is loopback CPU correctness evidence.
