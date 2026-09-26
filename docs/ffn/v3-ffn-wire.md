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
server authentication. HTTP bodies retain the existing body limits; stream
messages have explicit size limits described below. An unknown handle, failed open, missing
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

## Experimental persistent stream

`--v3-ffn-wire stream` selects WIRE-2. Binary HTTP remains the default.
After the same GET/open binding exchange, the client upgrades
`/v1/vindex3/ffn/stream` with WebSocket subprotocol `larql.v3-ffn.f32.v1`.
HTTP roots select WS and HTTPS roots select WSS with certificate verification;
the upgrade sends the same bearer credential and passes the existing route auth.
This is a new WebSocket transport on the HTTP listener, not the legacy MoE gRPC
service: the latter's routing/reduction messages have different semantics.

There is one connection per worker, serialized by a client mutex, and at most
one outstanding operation per connection. The first text message is
`{"profile":false}` (or `true`); this mode is fixed for that connection.
Each subsequent request is exactly one binary VFF1 message, and each numerical
reply is exactly one binary VFR1 message. The 36-byte ABI and f32 arithmetic are
unchanged. Server compute still uses the same per-operation blocking dispatch.
No batching, prediction, Q8 carrier, remote KV or automatic reconnection is added.

The connection pins the prepared worker incarnation present at upgrade. Reload
cannot substitute a different model beneath it; new connections must reopen the
new incarnation. Stream sequences must strictly increase from a nonzero value.
A duplicate is a protocol error on this connection (HTTP stateless replay remains
legal). Any unexpected message type, stale handle, bad sequence, invalid frame,
failed transform, disconnect or I/O timeout ends the stream. A client only
restores its socket after a fully checked reply, so a late response cannot be
accepted by a subsequent call after failure. Create a fresh coordinator and
replay the continuation after a stream failure.

Both endpoints bound incoming WebSocket frames/messages to
`max(36 + 4 * hidden, 1024)` bytes. Control messages additionally have a 1024-byte
limit. The server has a 60-second receive/send timeout (including idle streams);
the client has 60-second socket I/O timeouts. DNS lookup and waiting for the
per-worker mutex are outside socket timeouts. Only this sequential protocol is
supported; unsolicited text, ping/pong and close messages end operation service.
TLS support uses the system native-TLS backend; current automated and model-backed
transport checks exercise loopback WS, not a deployed WSS endpoint.

When profiling, a correlated text message `{sequence, timing}` precedes each
binary reply. `request_bytes` and `response_bytes` count numerical payloads,
just as HTTP body counts do. `telemetry_bytes` records the text payload;
`websocket_overhead_bytes` records WS headers/masking for numerical and timing
messages. `stream_setup_bytes` records the initial options message and its WS
framing on the first operation. Cold HTTP binding/upgrade, TCP and TLS overhead
are excluded. Without profiling, neither timing message nor timing work occurs.

Stream worker `decode_ns` starts after a complete WS message arrives; HTTP worker
`decode_ns` includes body receipt. Consequently the transport remainder also
includes inbound WS message assembly and must not be called pure network RTT.
Both remainders include response sending and scheduling outside the measured
handler. The diagnostic text encode/send is also outside stream handler time.

Validation includes repeated bitwise finite-carrier round trips, permanently
refused reuse after corrupted/miscorrelated/oversized replies or peer close,
worker rejection of stale handles, repeated sequences, wrong layers and invalid
carriers, and local/JSON/HTTP/stream bitwise continuation across the fixture's
sliding window. Real-model measurements use `run_loopback.py --stream-comparison`
and keep drift-rejected brackets separate.
