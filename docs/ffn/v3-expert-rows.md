# V3 multi-row expert operation: exact, versioned, single client

Status: **correctness rung only.** One operation can carry several rows to an
expert worker and return every row's unweighted expert outputs, bit for bit
what the [single-row operation](v3-routed-experts.md) returns for each row
alone. Nothing calls it during decode yet: `larql run --v3-ffn-shards` still
sends one row per layer per position. No throughput, latency or memory-traffic
claim is made here.

Not in this rung: speculative verification, a multi-position coordinator
session, cross-client batching or scheduling, a multi-row expert kernel, any
new carrier precision or transport.

## The operation

A caller hands the coordinator's expert provider a layer and R rows. Each row
has a caller-assigned id, one hidden-width input and its own selected expert
ids. The provider returns one unweighted output per `(row, expert)` pair.

- **Row ids belong to the caller.** A worker echoes them and attaches no
  meaning to them. It does not know whether rows are positions of one session
  or anything else, and it stays stateless.
- **No routing weights cross the wire.** Routing, selection and weighted
  reduction stay in the coordinator, as for the single-row operation.
- **Reduction is per row and unchanged.** Each row's outputs are accumulated
  in that row's production selection order. Arrival order never matters.
- **The worker groups by expert.** Every row that selected an expert is run
  through it before the next expert starts. Each pair goes through the same
  transform call as the single-row path.

## Wire frames

A new frame pair with its own endpoint, content type and magics. It is not an
extension of `VEX1`/`VEY1`, which are unchanged.

| | Single row (unchanged) | Multi-row |
|---|---|---|
| Path | `/v1/vindex3/experts/binary` | `/v1/vindex3/experts/rows` |
| Content type | `application/vnd.larql.v3-experts.f32` | `application/vnd.larql.v3-expert-rows.f32` |
| Magics | `VEX1` / `VEY1` | `VXR1` / `VXS1` |
| Header | 40 bytes | 44 bytes |

All integers and F32 bits are little-endian, with no padding.

```text
header (both directions)
  magic[4]  handle[16]  sequence u64  layer u32  hidden u32  rows u32  pairs u32

request  VXR1, then per row:
  row u32   count u32   count × expert u32   hidden × f32

response VXS1, then per pair:
  row u32   expert u32   hidden × f32
```

```text
request bytes:   44 + rows × (8 + 4 × hidden) + 4 × pairs
response bytes:  44 + pairs × (8 + 4 × hidden)
```

A frame is refused unless its length is exact, `hidden` matches the bound
worker, `rows` is between 1 and the frame limit (256), every row selects
between 1 and the layer's top-k experts, row ids are distinct, experts are
distinct within a row, `pairs` equals the sum of the row counts, and every
value is finite. Binding, open and the worker-incarnation handle are the
single-row operation's: a stale handle refuses.

### Versions never blur

- A worker that predates the frame has no such route. The client reports
  `expert worker refused the multi-row frame: HTTP 404` and stops. It does
  not retry on the single-row endpoint.
- A single-row frame sent to the multi-row endpoint, or the reverse, fails
  that endpoint's content-type or magic check with HTTP 400.
- The transport trait has no default for the multi-row method, so a transport
  cannot silently turn one batch into one request per row.

## Whole-call failure

A multi-row call returns every requested pair or an error. Four layers
enforce that, each before anything is handed upward:

1. **Worker.** Every row is validated before any transform runs; one bad row
   refuses the frame.
2. **Coordinator, before dispatch: everything it can decide alone.** For
   the whole call, before any worker is contacted, the coordinator checks:
   distinct row ids; at least one expert per row, each at most once; no more
   experts per row than the layer's top-k; every input the bound width and
   finite; every selected expert owned by an admitted worker; the layer
   owned; and no more rows than one frame may hold (256, which bounds every
   worker's frame). **A call refused by any of these sends zero requests.**
   Judging the whole call matters: a fault confined to one worker's share
   would otherwise be found only after the other workers had been asked, and
   two rows sharing an id could land in separate, individually well-formed
   frames and come back as one row built from two inputs.

   **What cannot be decided here is a worker's own state:** a stale handle
   after a restart, a worker that is unreachable or fails, a worker whose
   request-size limit is lower than the frame. Those are discovered only by
   asking, after other workers may already have answered. They still fail
   the whole call; they do not guarantee zero requests.
3. **Client and coordinator, on response.** A response must answer every requested
   `(row, expert)` exactly once and nothing else, from a worker that owns the
   expert, with the request's handle, sequence, layer and row count. When a
   layer's rows span several workers, every worker must succeed: one failure
   discards the others' answers.
4. **Reduction.** Rows are split and reduced only after the complete response
   passed; a missing, duplicated, unsolicited, short or non-finite output
   fails the call.

The call returns `Result<Vec<_>>`: an error carries no rows, so a caller has
nothing partial to commit. Scope is one coordinator's failure domain. Rows
from independent clients must not share a frame until cross-client failure
isolation is designed.

## Interfaces

| Seam | Addition |
|---|---|
| `RoutedExpertProvider` | `apply_rows(layer, &[ExpertRow])`. Its default is `apply` once per row, so a one-row call on any existing provider is exactly `apply` |
| `PlanBackend` | `routed_ffn_placed_rows(&[RoutedFfnCall], layer, provider)`: route each row, one provider call, reduce each row. CPU production implements it; other backends refuse by name |
| `PreparedRoutedExperts` (worker) | `apply_rows(backend, layer, rows, accounting)` |
| `ExpertTransport` | `forward_rows(shard, layer, rows)`, required |
| `admit_expert_provider` | The admitted provider, obtainable without binding it into single-row coordinator operands |

## What the counters mean

| Counter | Where | Shows |
|---|---|---|
| One `expert_rows_fanout` record per call | Coordinator profile | The operation: workers contacted, `rows`, `pairs`, and `complete` only if every worker's answer was accepted |
| One `expert_rows_shard` record per contacted worker | Coordinator profile | That worker's exchange, with the transport's own request records nested under `transport` |
| One `expert-rows-f32-v1` record per request | Nested in its shard record | A network request, with `rows` and `pairs` carried |
| `pairs` | Worker accounting | `(row, expert)` outputs served |
| `distinct_experts` | Worker accounting | Logical grouping: experts shared by rows |
| `transform_calls` | Worker accounting | Expert kernel invocations |

Requests are made on per-worker dispatch threads, and profile records are
per thread. The coordinator captures each dispatch thread's records and
re-records them on the calling thread, so a caller's profile holds every
request its call caused and no other call's. A call refused before fan-out
records nothing. With profiling off, none of this runs.

Worker accounting is returned in `x-larql-expert-rows-accounting` only when
the request carries the profile header; it is never part of the numerical
frame.

**`transform_calls` equals `pairs`.** The expert kernel takes one row at a
time, so grouping changes the order experts are visited in and not how often
their weights are read. `distinct_experts < pairs` shows that reuse is
logically available. It is not evidence that any weight read was avoided, and
nothing here measures weight bytes read from memory. Physical reuse needs a
kernel that makes one pass over an expert's weights for several rows, and
memory-level evidence; both are a later rung.

## Evidence

Fixture and stub evidence, all bitwise. There is no real-model multi-row
evidence, because no decode session calls the operation; the real-model
evidence is that the existing single-row path still reproduces the GPT-OSS
20B control on this branch. Physical weight reuse has not been established.

- **Wire:** exact carrier round trip; every truncation, extension, header
  corruption and non-finite value refused; single-row and multi-row frames
  never decode as each other.
- **Backend:** one row and five rows through `routed_ffn_placed_rows` equal
  each row through the single-row placed path, which equals local execution.
  The batch shares experts and routes rows differently. A provider that only
  has the single-row operation gives the same bits through the default.
- **Worker:** `apply_rows` equals `apply` per pair, outputs are grouped by
  expert, and the accounting is as described above.
- **Coordinator:** real workers in-process behind a stub transport, driven
  through the admitted provider. One row and three rows equal single-row
  dispatch per pair; each contacted worker receives one frame holding only
  the experts it owns; a multi-row call never uses the single-row route; a
  wrong answer from one worker rejects the call after every worker was asked.
  Admission through `admit_expert_provider` refuses the same mutated
  bindings coordinator preparation refuses, without reading local payload.
- **HTTP, real workers:** one-worker and two-worker topologies; one row and
  three rows equal the single-row frames per pair; one request per contacted
  worker; no single-row request is made by a multi-row call.
- **Failure:** dropped, duplicated, substituted, unsolicited, non-finite and
  short outputs, a failing worker, a foreign expert, stale handle, wrong
  layer, wrong endpoint and truncated frame all fail the whole call.

## Open interface decisions

1. **Who calls it.** A multi-position coordinator session is the first real
   caller and does not exist. It owns causal attention within a block,
   acceptance and rollback; none of that is designed here.
2. **Multi-row kernel.** `PlanBackend::expert_transform` is one row. A rows
   form is where physical reuse would be implemented and measured.
3. **Frame limit.** 256 rows per frame is a decoder allocation bound, not a
   tuned value.
4. **Input duplication across workers.** A row that selects experts on two
   workers sends its input to both, as the single-row operation does.
5. **Cross-client failure isolation.** Deferred; see whole-call failure.
