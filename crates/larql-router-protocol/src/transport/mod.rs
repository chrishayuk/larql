//! Optional transports for the grid gRPC stream. TCP is the default and
//! always available; this module gates additional transports (QUIC today,
//! potentially WireGuard or h2c later) behind the `quic` feature so a
//! non-QUIC build doesn't pull in TLS, certificate handling, or UDP
//! sockets.
//!
//! ADR-0010 covers the QUIC rollout in detail. The short version:
//!   * QUIC is layered under HTTP/2 inside tonic — a single bidirectional
//!     QUIC stream carries the gRPC HTTP/2 framing, the way TCP normally
//!     would. This is *not* HTTP/3 but it still buys 0-RTT reconnect,
//!     TLS 1.3 by default, and BBRv2-class congestion control.
//!   * Per-stream independence is not a benefit here because each
//!     `GridService.Join` call already uses a single bidirectional gRPC
//!     stream; HoL on that one stream is unavoidable.
//!
//! ADR-0019 adds an `http3` feature on top of `quic` — real HTTP/3
//! framing on the shard fan-out path so MoE per-token expert
//! dispatch (ADR-0018) can issue parallel sub-requests as
//! independent QUIC streams.

pub mod quic;

#[cfg(feature = "http3")]
pub mod h3;
