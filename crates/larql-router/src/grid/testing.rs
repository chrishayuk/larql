//! Shared `#[cfg(test)]` helpers for the grid module suite.
//!
//! Each grid submodule has its own `mod tests` block, and the
//! [`entry`] constructor for a default-populated [`ServerEntry`] is
//! reused across all of them. Hoisting it here avoids four copies of
//! the same struct literal drifting out of sync as the entry fields
//! evolve.

#![cfg(test)]

use std::collections::HashMap;
use std::time::Instant;

use super::ServerEntry;

/// Build a [`ServerEntry`] with sensible test defaults. Caller can
/// mutate the returned value before registering it with the grid.
pub(crate) fn entry(
    server_id: &str,
    listen_url: &str,
    model_id: &str,
    layer_start: u32,
    layer_end: u32,
) -> ServerEntry {
    ServerEntry {
        server_id: server_id.into(),
        listen_url: listen_url.into(),
        model_id: model_id.into(),
        layer_start,
        layer_end,
        vindex_hash: format!("hash-{server_id}"),
        cpu_pct: 0.0,
        ram_used: 1024,
        requests_in_flight: 0,
        last_seen: Instant::now(),
        layer_latencies: HashMap::new(),
        req_per_sec: 0.0,
        rtt_ms: None,
        // ADR-0018: dense by default. Test cases that need MoE
        // shards mutate these fields after construction.
        expert_start: 0,
        expert_end: 0,
    }
}

/// ADR-0018 / ADR-0021 — like [`entry`] but with an expert range.
/// Useful for MoE-shaped tests of `route_expert` /
/// `route_expert_with_rank`.
pub(crate) fn entry_with_experts(
    server_id: &str,
    listen_url: &str,
    model_id: &str,
    layer_start: u32,
    layer_end: u32,
    expert_start: u32,
    expert_end: u32,
) -> ServerEntry {
    let mut e = entry(server_id, listen_url, model_id, layer_start, layer_end);
    e.expert_start = expert_start;
    e.expert_end = expert_end;
    e
}
