# V3-FFN-WIRE-1: bind once, exact binary carriers

`larql run --v3-ffn-shards URL,...` uses the binary f32 wire by default.
`--v3-ffn-wire json` selects the original full-binding JSON control explicitly.
The worker command is unchanged. Both wires execute the same prepared dense
FFN operands through CPU production lowering; attention and KV remain local.

```bash
larql run model.vindex3 "Hello" --v3-ffn-shards http://localhost:9181,http://localhost:9182 \
  --v3-ffn-wire binary --v3-profile binary.jsonl
larql run model.vindex3 "Hello" --v3-ffn-shards http://localhost:9181,http://localhost:9182 \
  --v3-ffn-wire json --v3-profile json.jsonl
```

## Authority and lifetime

1. `GET /v1/vindex3/ffn` discovers the complete binding.
2. `POST /v1/vindex3/ffn/open` submits that binding as JSON. The worker compares
   it with the binding of its immutable prepared image and returns
   `{version: 1, handle: [16 bytes], binding: ...}`. The client checks the version
   and echoed binding; the coordinator admits artifact, plan, lowering,
   representations, dimensions and complete layer coverage before execution.
3. Hot requests use `POST /v1/vindex3/ffn/binary`, containing only the handle,
   sequence, layer, width and exact f32 carrier. Responses echo the identifiers.

The handle is a random 128-bit **worker incarnation** identifier. It is created
at preparation and shared by clients opening that immutable worker. Reopening
even the same artifact creates a different handle. There is no per-client
session table, remote KV, growing handle cache or expiry timer. The bound
execution object owns an `Arc` of the prepared runtime, so no request can supply
a replacement plan, operands or backend after the initial authority checks.
Range, shape, finite values and numerical provider identity remain checked at
the operation boundary.

The handle is not an authentication credential; all routes use the existing
server authentication and body limits. An unknown handle, failed open, missing
binary endpoint or malformed reply refuses execution. There is no JSON
fallback, implicit reopen, or application retry. A failed FFN still invalidates
the coordinator continuation and requires a fresh session with replay.

Sequence numbers are client-side correlation IDs, monotonically allocated per
worker connection pool. Exhaustion refuses rather than wrapping. They are not
server-side replay protection: a duplicate request is a legal repeat of the
same stateless transform. Different clients may independently start at sequence
one. The HTTP response belongs to its request, and the client additionally checks
handle, sequence, layer, width, framing and finiteness before accepting the row.

## Binary ABI v1

Content type: `application/vnd.larql.v3-ffn.f32`. No padding. All integers and
IEEE-754 f32 bits are little-endian.

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 4 | `VFF1` request / `VFR1` response; direction and version |
| 4 | 16 | Worker incarnation handle |
| 20 | 8 | Sequence, u64 |
| 28 | 4 | Layer index, u32 |
| 32 | 4 | Hidden width, u32 |
| 36 | `4 × hidden` | Carrier input or unweighted dense FFN delta |

Exact expected length is required, including rejection of trailing bytes.
Dimensions are checked against the bound image before carrier allocation. The
client bounds response reads to expected length plus one byte, so an oversized
reply cannot cause an unbounded response allocation. Finite f32 bits, including
signed zero and subnormals, round-trip unchanged. NaN and infinity refuse.

A token crossing 28 layers at width 1024 uses **115,696 body bytes per direction**;
34 layers at width 2560 use **349,384 bytes per direction**. This includes every
36-byte header. Cold open/discovery traffic and HTTP/TLS/TCP headers are outside
these counts. No Q8, prediction, innovation or lossy carrier coding is involved.

## Connections, timing and validation

A single blocking HTTP client maintains persistent connections, with one idle
connection retained per worker and no pool idle timeout. Both JSON and binary
controls use this policy. This version is sequential request/response, not a
new multiplexed streaming or batch execution protocol.

`--v3-profile` records actual binary body lengths and nested worker timings using
the same [profile schema](v3-dense-profile.md). Requests opt into the timing
header; the numerical response body never includes profiling data. On the
binary worker, `execute_ns` and `ffn_ns` cover the bound transform; expensive
descriptor checks were completed during preparation/open. Cold setup is not
included in per-token measurements.

Gates cover fixed byte layout, signed-zero/subnormal bit preservation, truncated
and oversized frames, non-finite values, wrong content type/version, changed
open bindings, sequence/layer/handle mismatch, stale incarnations, sequence
exhaustion, deterministic duplicate requests and local/JSON/binary bitwise
fixture logits through the sliding-window boundary. These correctness gates
are separate from real-model timing diagnostics.
