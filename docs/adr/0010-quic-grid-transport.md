# ADR-0010 — QUIC Transport for Grid gRPC Streams

**Status:** Accepted — shipped 2026-05-15 as GT7. Opt-in via `--features quic` + `--quic-port`; HTTP/2 over a single QUIC bi-stream, TLS 1.3, SHA-256 fingerprint pinning on the client side. Real HTTP/3 (per-stream independence) tracked as post-GT7 in ROADMAP.  
**Depends on:** ADR-0004 (Self-Assembling Grid)

---

## Context

The larql grid uses TCP-based gRPC for the `GridService.Join` bidirectional
stream and for `ExpertService` calls. TCP is the right default for LAN where
latency is 0.1–1 ms and packet loss is negligible.

For cross-region and internet deployments the TCP model has three structural
problems:

1. **Head-of-line blocking.** TCP delivers bytes in order. If a single packet
   is lost, all multiplexed gRPC streams on the same connection stall until
   the retransmit arrives. For parallel expert fan-out (8 experts × 30 layers
   per token), a single lost packet serialises all 8 in-flight responses.

2. **1-RTT reconnect overhead.** A TCP connection requires 1 RTT (SYN/SYN-ACK)
   before the first byte. With exponential backoff after grid-server restart,
   each reconnect costs at minimum 1 RTT. On a 50 ms cross-region link that
   is 50 ms dead time per reconnect before the grid resumes inference.

3. **Congestion on lossy paths.** TCP CUBIC backs off aggressively on any loss.
   Mobile/satellite paths with 1–5% loss cause TCP to throttle to well below
   link capacity. QUIC's BBRv2-derived congestion control handles loss better
   and distinguishes loss from true congestion.

QUIC (RFC 9000) solves all three:
- **Per-stream independence**: each gRPC stream is a QUIC stream; packet loss
  in one stream does not stall others.
- **0-RTT reconnect**: QUIC stores session tickets; reconnecting to a known
  server costs 0 RTT for the first byte.
- **Better loss tolerance**: QUIC's congestion control (default BBRv2 in quinn)
  handles lossy paths without excessive backoff.

---

## Decision

Add QUIC as an optional, feature-gated transport for the grid gRPC stream.
TCP remains the default and is always available. QUIC is an additive
transport that is opt-in at deployment time.

```
larql serve /path/to/vindex --join quic://router:50053 --layers 0-14
larql-router --grid-port 50052 --quic-port 50053
```

The expert HTTP/2 services (`ExpertService`) remain on TCP/gRPC; QUIC is
applied only to the long-lived `GridService.Join` control stream and, in a
second phase, to the expert streaming path.

The `quinn` crate (v0.11) is used as the QUIC implementation. It is mature,
maintained by the Rust community, and used in production at Cloudflare and
Mozilla. `s2n-quic` (AWS) is an alternative but requires C dependencies;
`quinn` is pure Rust and aligns with the existing async-Rust/tokio stack.

---

## Architecture

```
larql-server                          larql-router
──────────────                        ──────────────────────
announce.rs                           main.rs
  parse --join quic://router:50053      spawn QuicGridEndpoint on --quic-port
  connect via quinn::Endpoint           QuicGridEndpoint: accepts QUIC streams
  open QUIC stream to router            each stream → GridService.join() handler
  wrap stream as tonic Channel          (existing grid/service.rs logic unchanged)
  send AnnounceMsg, Heartbeat, etc.

transport/quic.rs (new, shared)
  quinn::Endpoint setup (client + server)
  TLS cert/key handling (self-signed or ACME)
  Stream wrapper → implements tokio::io::AsyncRead + AsyncWrite
  → usable as a tonic transport Channel
```

tonic does not expose a QUIC transport natively. The approach: implement
`tonic::transport::channel::Endpoint` by wrapping a quinn `SendStream` +
`RecvStream` pair as a `tokio::io::AsyncRead + AsyncWrite` duplex, then
pass it as a `tonic::transport::Channel` via `Channel::from_shared` +
custom connector. This is the same pattern used by `quinn`'s h3 integration.

---

## Feature Gate

```toml
# larql-server/Cargo.toml
[features]
quic = ["dep:quinn", "dep:rustls"]

[dependencies]
quinn = { version = "0.11", optional = true }
rustls = { version = "0.23", optional = true, features = ["ring"] }
```

```toml
# larql-router/Cargo.toml
[features]
quic = ["dep:quinn", "dep:rustls"]
```

Build with QUIC: `cargo build --release --features quic`

---

## TLS

QUIC requires TLS 1.3. For grid-internal traffic (LAN / private network),
a self-signed certificate is generated at router startup and fingerprint is
distributed via the `--grid-key` mechanism (shared secret). Servers pin the
router's certificate fingerprint instead of using a CA chain.

For internet-facing routers, an ACME-provisioned certificate (via
`rustls-acme`) is the recommended path. Not implemented in phase 1; users
can supply `--quic-cert` / `--quic-key` paths.

---

## CLI

```
larql-router
  --grid-port PORT     existing: TCP gRPC grid port
  --quic-port PORT     new: QUIC grid port (requires --features quic)
  --quic-cert PATH     new: TLS cert PEM (default: self-signed)
  --quic-key  PATH     new: TLS key PEM (default: self-signed)

larql serve
  --join grpc://host:PORT    existing: TCP gRPC (unchanged)
  --join quic://host:PORT    new: QUIC (requires --features quic)
```

---

## Implementation Files

| File | Change |
|---|---|
| `crates/larql-router-protocol/src/transport/` | NEW directory |
| `crates/larql-router-protocol/src/transport/quic.rs` | QUIC client/server endpoint setup, stream wrapper |
| `crates/larql-router-protocol/src/transport/mod.rs` | Re-export; feature-gated |
| `crates/larql-router-protocol/Cargo.toml` | Add `quinn` optional dep + `quic` feature |
| `crates/larql-router/src/main.rs` | Spawn `QuicGridEndpoint` when `--quic-port` given |
| `crates/larql-server/src/announce.rs` | Parse `quic://` scheme; use QUIC transport |
| `crates/larql-server/src/bootstrap.rs` | Accept `--quic-port`; generate self-signed cert |
| `crates/larql-server/Cargo.toml` | Add `quinn` optional dep + `quic` feature |
| `crates/larql-router/Cargo.toml` | Add `quinn` optional dep + `quic` feature |

---

## Trade-offs

- **UDP blocking**: some corporate firewalls block UDP 443/non-80. TCP
  fallback is always available; QUIC is an opt-in with no downgrade path
  needed (the client explicitly chooses `quic://`).
- **Implementation complexity**: wrapping quinn as a tonic transport is
  ~300 LOC of non-trivial async code. The pattern is well-documented in
  the quinn ecosystem but requires careful backpressure handling.
- **TLS overhead**: QUIC requires TLS 1.3. For grid-internal traffic on a
  trusted LAN, this adds ~1 ms of CPU overhead on connect (negligible).
  0-RTT session resumption eliminates this on reconnect.
- **quinn version stability**: quinn 0.11 tracks the stable QUIC RFC. Breaking
  API changes are infrequent; pin to minor version.

---

## Rollout

Phase 1 (this ADR): QUIC for `GridService.Join` control stream only.
Phase 2 (future ADR): QUIC for `ExpertService` streaming dispatch — the
path where per-stream independence matters most (8 parallel expert streams).
