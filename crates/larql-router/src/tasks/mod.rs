//! Long-lived background tasks spawned at router startup.
//!
//! Both submodules follow the same shape: a public `spawn(state,
//! config)` entry point that registers a tokio task running for the
//! process lifetime. They share read/write access to [`GridState`]
//! through a shared `Arc<RwLock<GridState>>`.
//!
//! - [`rebalancer`] — 30 s tick that drives hot-shard elevation,
//!   under/over-replication checks, stale-heartbeat eviction, and
//!   per-layer latency imbalance detection.
//! - [`rtt_probe`] — optional active-probe loop that periodically
//!   hits each serving server's `/v1/health` and records the
//!   round-trip as `rtt_ms` for the routing tie-breaker.

pub mod rebalancer;
pub mod rtt_probe;
