# ADR-0019 — Real HTTP/3 for Shard Fan-Out

**Status:** Accepted — Phases 1+2+3 shipped 2026-05-16 in
`larql-router-protocol::transport::h3`. Round-trip integration test
proves the transport works end-to-end. Phase 4 (CLI wiring + bench)
is the remaining work, spanning `larql-router` and `larql-server`.
**Depends on:** ADR-0010 (QUIC for the grid Join stream),
ADR-0018 (MoE expert routing — the use case that motivates this).
**Affects:**
- `crates/larql-router-protocol/` (transport scaffolding,
  feature-gated h3 deps),
- `crates/larql-router/src/{cli_helpers, dispatch, http}.rs` (client
  side: opt-in shard transport),
- `crates/larql-server/src/{bootstrap, routes}.rs` (server side:
  axum-style h3 listener).

---

## Context

The router fan-out path today uses **HTTP/2 over TCP** via `reqwest`:

```
router  ── POST /v1/walk-ffn ──▶ shard
        (reqwest HTTP/2 client, TCP)
```

This works fine for dense routing (one HTTP request per layer hop)
and for static `--shards` deployments. The performance ceiling shows
up under two specific load shapes:

1. **MoE per-token expert fan-out (ADR-0018).** A K2.6 / DeepSeek-V3
   top-K dispatch issues 6-8 expert sub-requests per layer × 60-80
   layers = 360-640 sub-requests per token. Many of these go to the
   same shard host (when one host owns multiple expert-shards across
   layers, or when several token-time fan-outs land on the same
   shard concurrently). Under HTTP/2 multiplexing, **TCP-level
   head-of-line blocking** on a single TCP connection stalls every
   stream sharing the connection whenever a packet is lost. The
   stream-multiplexing benefit of HTTP/2 is partially defeated by
   the TCP transport.

2. **Cross-region grid deployments.** Higher RTT amplifies the
   probability of packet loss on a long-lived TCP connection. The
   tail latency of a multi-stream fan-out is governed by the
   slowest stream, which under TCP HoL can be the whole connection.

QUIC fixes both: each stream has independent flow control + loss
recovery, so packet loss on stream A doesn't stall stream B.

ADR-0010 introduced QUIC for the **grid Join stream** but
deliberately wraps **HTTP/2 over a single QUIC bi-stream** — fine
for the one-stream Join call, but it doesn't give per-stream
independence for parallel walk-ffn requests because all walk-ffn
traffic goes over reqwest's HTTP/2-over-TCP path, not over QUIC.

Now that MoE expert routing has shipped (ADR-0018) and per-token
fan-out is a real workload, the TCP HoL cost is no longer
theoretical.

---

## Decision

Add **real HTTP/3** (via the `h3` + `h3-quinn` crates) as an
**opt-in** shard transport. When enabled on both router and server,
each `POST /v1/walk-ffn` call runs as an independent QUIC stream
over a shared QUIC connection per shard host, with per-stream loss
recovery and flow control. When disabled, the existing
HTTP/2-over-TCP path runs unchanged.

### Three independent components

```
┌─────────────────────────────────────────────────────────────────┐
│ larql-router-protocol                                           │
│  - shared h3+quinn transport scaffolding (TLS, endpoint setup)  │
│  - cert pinning reuse from ADR-0010                             │
│  - feature flag: `http3` (default off)                          │
└─────────────────────────────────────────────────────────────────┘
        ▲                                          ▲
        │                                          │
┌───────┴──────────────┐               ┌───────────┴────────────┐
│ larql-router         │               │ larql-server           │
│  - client: opt-in    │               │  - server: opt-in      │
│    `--http3-shards`  │   POST h3 ───▶│    `--http3-port`      │
│    h3 client per     │               │  - h3 listener on UDP  │
│    shard host        │               │    binds new port,     │
│  - falls back to     │               │    routes to existing  │
│    HTTP/2 when off   │               │    walk-ffn handler    │
└──────────────────────┘               └────────────────────────┘
```

### Wire surface (unchanged at the HTTP layer)

The HTTP semantics — `POST /v1/walk-ffn`, JSON bodies, response
shape — are identical to today. Only the transport changes. A
client speaking HTTP/3 against a server speaking HTTP/3 gets:

- One UDP-based QUIC connection per shard host (reused across
  requests, like a TCP connection today).
- Per-token fan-out runs N concurrent QUIC streams over the same
  connection.
- TCP HoL gone; per-stream loss recovery + congestion control.

### CLI surface

**Router** — new flag `--http3-shards <BOOL>` (default `false`).
When set, the dispatch layer wraps shard URLs in an h3 client. The
shard URL scheme stays `http://` for now; the transport choice is
operator-driven, not URL-driven. Mixed-mode (some shards HTTP/2,
some HTTP/3) is **not supported** in v1 — the router uses one
transport per process.

**Server** — new flag `--http3-port <PORT>` (default disabled).
When set, larql-server spawns a separate UDP listener on the
specified port using `h3` + `h3-quinn` + a thin axum-compatible
adapter. The HTTP/2 listener on `--port` keeps running, so a
router that hasn't opted in can still reach the shard over TCP.

### TLS reuse

The h3 path uses the **same TLS setup** as ADR-0010's QUIC
listener:

- Server uses `--quic-cert` / `--quic-key` (or the auto-generated
  self-signed cert).
- Client pins the SHA-256 fingerprint via `--quic-cert-fingerprint`
  on `--join`. The same fingerprint can be used for the h3
  connection.

This avoids forcing operators to provision a second cert.

### Feature gating

`http3` is a Cargo feature on `larql-router-protocol`, with
matching features on `larql-router` and `larql-server`. When the
feature is off:

- No `h3` / `h3-quinn` symbols are compiled in.
- The CLI flags `--http3-shards` and `--http3-port` are absent
  (gated by `#[cfg(feature = "http3")]`).
- The binary is identical to today's, modulo a no-op feature flag.

This mirrors how ADR-0010's `quic` feature works.

### `reqwest` http3 — considered, rejected for now

reqwest 0.12 has an unstable `http3` feature. It requires:

- `RUSTFLAGS='--cfg reqwest_unstable'` at every build invocation.
- The underlying h3 + h3-quinn versions are pinned to reqwest's
  internal choices.

Rejected because:

1. The `--cfg reqwest_unstable` flag must be set at every
   build/test/CI invocation; it's a workspace-level commitment.
2. Pinning to reqwest's internal h3 version makes it hard to
   coordinate the server-side h3 (which uses h3 directly).
3. Using `h3` + `h3-quinn` directly is more verbose but gives
   stable API control on both sides.

The router's client side will use `h3-client` (a thin wrapper
in `larql-router-protocol::transport::h3_client`) rather than
reqwest with http3 enabled.

---

## Implementation Plan

### Phase 1 — Scaffolding (`larql-router-protocol`, this session)

- Add `h3` and `h3-quinn` deps under the `http3` feature flag.
- Add `src/transport/h3.rs` with:
  - `H3Client` struct wrapping a `quinn::Endpoint`.
  - `H3Client::request(url, body) -> Future<Response>` —
    one-request-one-stream interface.
  - Cert pinning reuse from `transport::quic`.
- No CLI integration yet; just the building blocks.

### Phase 2 — Server (`larql-server`, next session)

- Add `--http3-port` flag.
- Spawn a `tokio::spawn` listener on UDP, accept QUIC connections,
  delegate to a thin h3 adapter that forwards request bodies to the
  same `axum::Router` the HTTP/2 listener uses.
- TLS shares `--quic-cert` / `--quic-key`.
- Smoke test: curl-quic against `/v1/health`.

### Phase 3 — Router client (`larql-router`, next session)

- Add `--http3-shards` flag.
- When set, `cli_helpers::build_shard_client` returns an `H3Client`
  wrapper instead of `reqwest::Client`.
- `dispatch::*` paths use a transport-trait abstraction so the call
  site doesn't care which transport is in use.
- Cert pinning via `--shard-cert-fingerprint` (new flag, mirrors
  `--quic-cert-fingerprint` on the grid Join side).

### Phase 4 — Benchmarks (next-next session)

- Add `bench-http3` make target.
- Two-host LAN benchmark: 60-layer MoE fan-out, top-8 experts per
  layer, 10 minutes of sustained load.
- Compare wire RTT, p50/p99 latency, throughput, and packet-loss
  recovery time vs the HTTP/2/TCP baseline.
- Acceptance criterion: ≥20% p99 reduction on a 1% packet-loss
  injected link (matches published HTTP/2 vs HTTP/3 results).

---

## Alternatives Considered

### Route walk-ffn over the existing QUIC Join channel

Multiplex walk-ffn calls on the same QUIC bi-stream that ADR-0010
uses for the grid Join control stream. Avoids the second listener.

Rejected because:

- Couples request-level lifecycle to grid-registration lifecycle.
  If a server's Join stream restarts, in-flight walk-ffn requests
  fail mid-flight.
- The Join stream uses HTTP/2 over a single QUIC bi-stream, so
  walk-ffn calls would multiplex within HTTP/2 — same HoL we're
  trying to escape.
- Two QUIC connections per shard (one for Join, one for walk-ffn)
  uses ~2× UDP file descriptors but isolates failure modes.

### Use reqwest's `http3` feature

See "considered, rejected" above. Workspace-level `RUSTFLAGS`
commitment is the showstopper.

### Use raw QUIC streams without HTTP framing

Skip the h3 framing layer entirely; stream JSON bodies directly
over QUIC streams. Saves a small amount of per-stream overhead.

Rejected because:

- Loses HTTP status codes, headers, content-type — operators
  expect these for debugging.
- Existing `axum::Router` on the server side wants standard HTTP.
  Stripping framing means duplicating the routing layer.
- The h3 framing overhead is single-digit-bytes per request,
  irrelevant against the JSON body size.

### Wait for stable `reqwest::http3`

Status as of 2026-05-16: still unstable, no clear stabilization
date. Rejected — we'd be perpetually waiting.

---

## Consequences

### Positive

- MoE per-token fan-out gets per-stream independence. K2.6's
  640-pairs-per-token fan-out under 1% packet loss should see a
  measurable p99 improvement.
- Cross-region grid deployments become more viable — RTT × loss
  amplification is the wire-latency killer; QUIC eliminates it.
- The h3 path is **opt-in**, so existing HTTP/2/TCP deployments
  are unaffected.
- TLS infrastructure reuses ADR-0010; no second cert provisioning.

### Negative

- A second listener on each shard (UDP, separate port). Operators
  need to open a UDP firewall rule, plus ensure UDP isn't dropped
  by middleboxes (corp networks sometimes do).
- More code paths in the dispatch layer (transport abstraction).
  Mixed-mode is not supported in v1 to keep the dispatch logic
  simple.
- New deps: `h3`, `h3-quinn`, possibly `bytes`-related transitives.
  Feature-gated, so the default build is unchanged.
- HTTP/3 client-side implementation is more verbose than reqwest
  HTTP/2. `larql-router-protocol::transport::h3_client` carries
  that complexity.

### Neutral

- Wire semantics (HTTP verbs, paths, JSON shape) unchanged.
- The `--http3-shards` opt-in is a per-router choice; operators
  who want HTTP/2 today aren't forced to switch.

---

## Open Questions

1. **Connection migration.** QUIC supports connection migration
   (a client moving between networks while keeping the connection
   alive). Useful for mobile / edge clients; irrelevant for the
   router-server LAN/WAN case. Out of scope for v1.

2. **0-RTT resumption.** QUIC supports 0-RTT for repeat
   connections. Could shave first-request latency. Defer to a
   future ADR if a real workload shows the win — most shard
   connections are long-lived (minutes to hours), so the
   first-request cost is amortized.

3. **Backpressure / flow control.** h3 inherits QUIC's
   per-stream flow control. Today's HTTP/2/TCP path has TCP-level
   backpressure. The h3 path gives finer-grained control but the
   default `quinn` config should be fine for the LAN case.

4. **Mixed-mode support.** Some shards HTTP/2, some HTTP/3. Not
   in v1; operators choose one transport per router. Future ADR
   if a real migration scenario requires it.

5. **Server-side adapter to `axum::Router`.** Cleanest pattern is
   probably: h3 accepts a request, decode body to `axum::Request`,
   `axum::Router::call`, encode `axum::Response` back to h3. Some
   middleware (extractors that need TCP-specific connection info)
   may need adjustment. Detail for Phase 2.

---

## Implementation pointers (forward)

| File | Role (after impl) |
|---|---|
| `crates/larql-router-protocol/Cargo.toml` | `[features] http3 = ["dep:h3", "dep:h3-quinn", "quic"]` |
| `crates/larql-router-protocol/src/transport/h3.rs` | `H3Client` + `H3Server` scaffolding |
| `crates/larql-router/Cargo.toml` | `[features] http3 = ["larql-router-protocol/http3"]` |
| `crates/larql-router/src/main.rs` | `--http3-shards` flag |
| `crates/larql-router/src/cli_helpers.rs` | `build_shard_client` returns transport-trait |
| `crates/larql-router/src/dispatch.rs` | use trait dispatch instead of direct `reqwest::Client` calls |
| `crates/larql-server/Cargo.toml` | `[features] http3 = ["larql-router-protocol/http3"]` |
| `crates/larql-server/src/bootstrap.rs` | spawn h3 listener on `--http3-port` |
| `crates/larql-server/src/routes.rs` | h3 → axum adapter |

### Test coverage strategy

- Unit tests for the h3 client/server adapter logic (request body
  passthrough, status code handling, cert pinning).
- Integration test: in-process h3 server, in-process h3 client,
  send a request, verify round-trip.
- Bench: `bench-http3` make target (Phase 4).

---

## Status notes

- **Phase 1** (scaffolding in `larql-router-protocol`): ✅ shipped
  2026-05-16. `transport::h3::{client_endpoint, server_endpoint,
  H3Error}` + ALPN constant + 5 unit tests.
- **Phase 2** (server-side h3 via axum::Router): ✅ shipped
  2026-05-16. `transport::h3::serve_axum` wraps `h3-axum 0.2.0`'s
  `serve_h3_with_axum` for each accepted connection. Reuses
  whichever `axum::Router` the dense HTTP/2 listener uses, so
  handlers are shared.
- **Phase 3** (router-side h3 client): ✅ shipped 2026-05-16.
  `H3Client::post_json(addr, server_name, path, body) ->
  Result<H3Response, H3Error>`. Each call opens a fresh QUIC
  connection (no pooling in v1 — see below). Three integration
  tests prove the round-trip works:
  - `h3_post_json_round_trips` — single request, body echo, server
    saw exactly one call.
  - `h3_two_concurrent_requests_round_trip` — `tokio::join!`-d
    pair of POSTs against the same client succeed.
  - `h3_unknown_route_returns_404` — axum routing through the h3
    adapter surfaces real HTTP status codes.
- **Phase 4** (CLI wiring + benchmarks): pending.
  - `larql-router` needs `--http3-shards` flag + a transport-trait
    abstraction so dispatch can pick between reqwest HTTP/2 and
    `H3Client`.
  - `larql-server` needs `--http3-port` flag + bootstrap to spawn
    `serve_axum` on a UDP listener with the same `axum::Router`.
  - Benchmark: `bench-http3` make target running a two-host LAN
    fan-out, 1% packet-loss injected link, p99 latency vs
    HTTP/2-over-TCP baseline.

### v1 limitations (documented; addressable later)

1. **No connection pooling.** `H3Client::post_json` opens, uses,
   and tears down a QUIC connection per call. Acceptable for
   correctness; the connect cost (~1 RTT after first handshake)
   will show up as overhead in benchmarks. A `HashMap<server_addr,
   Connection>` cache with idle-timeout eviction is the natural
   next step.
2. **Single body buffer cap of 64 MiB.** Mirrors the existing axum
   body limit on the dense path. Adequate for FFN walks; will need
   raising if anyone sends very-large JSON bodies.
3. **No retries or circuit-breaker.** A connect failure or stream
   error surfaces as `H3Error::Connect` / `H3Error::Request`.
   Caller decides; integration with backpressure / circuit-breaker
   (Self-healing P1) is a separate ADR.
