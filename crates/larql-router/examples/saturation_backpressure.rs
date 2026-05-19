//! ADR-0020 — backpressure / saturation tier in `route()`.
//!
//! Shows the routing-layer behaviour an operator wires up via
//! `--saturation-ceiling N`: replicas whose `requests_in_flight ≥ N`
//! are filtered out of `route()` before the GT3/RTT/in-flight
//! comparator runs. When every owning replica is over the ceiling,
//! `route()` returns `None` and the HTTP dispatcher (in `http.rs`)
//! returns `503 Service Unavailable` with `Retry-After: 0.5` and
//! bumps `larql_router_route_saturation_total` — load-shedding, not
//! forwarding to the least-bad replica.
//!
//! Run with:
//!
//!   cargo run -p larql-router --example saturation_backpressure

use std::collections::HashMap;
use std::time::Instant;

use larql_router::grid::{GridState, ServerEntry};

fn server(server_id: &str, listen_url: &str, layer_start: u32, layer_end: u32) -> ServerEntry {
    ServerEntry {
        server_id: server_id.into(),
        listen_url: listen_url.into(),
        model_id: "m".into(),
        layer_start,
        layer_end,
        vindex_hash: "h".into(),
        cpu_pct: 0.0,
        ram_used: 0,
        requests_in_flight: 0,
        last_seen: Instant::now(),
        layer_latencies: HashMap::new(),
        req_per_sec: 0.0,
        rtt_ms: None,
        expert_start: 0,
        expert_end: 0,
    }
}

fn show(grid: &GridState, label: &str, layer: u32) {
    let routed = grid.route(Some("m"), layer);
    let owners = grid.has_owners_for(Some("m"), layer);
    println!(
        "  {label:<28} → route()={routed:<32?} has_owners_for={owners} {}",
        match (&routed, owners) {
            (Some(_), _) => "→ 200 dispatch",
            (None, true) => "→ 503 (Retry-After: 0.5)",
            (None, false) => "→ 400 (no owning shard)",
        }
    );
}

fn main() {
    let mut grid = GridState::default();
    grid.register(server("a", "http://shard-a:9181", 0, 14));
    grid.register(server("b", "http://shard-b:9181", 0, 14));

    println!("== No saturation ceiling (filter disabled) ==");
    grid.update_heartbeat("a", 50.0, 0, 100, Vec::new(), 0.0);
    grid.update_heartbeat("b", 50.0, 0, 100, Vec::new(), 0.0);
    show(&grid, "both replicas in_flight=100", 5);

    println!("\n== Ceiling=32, neither replica at the ceiling ==");
    grid.set_saturation_ceiling(Some(32));
    grid.update_heartbeat("a", 50.0, 0, 4, Vec::new(), 0.0);
    grid.update_heartbeat("b", 50.0, 0, 4, Vec::new(), 0.0);
    show(&grid, "in_flight=4 / 4", 5);

    println!("\n== Ceiling=32, one replica saturated ==");
    grid.update_heartbeat("a", 50.0, 0, 64, Vec::new(), 0.0);
    grid.update_heartbeat("b", 50.0, 0, 4, Vec::new(), 0.0);
    show(&grid, "a=64 (over), b=4", 5);

    println!("\n== Ceiling=32, every owning replica saturated ==");
    grid.update_heartbeat("a", 50.0, 0, 64, Vec::new(), 0.0);
    grid.update_heartbeat("b", 50.0, 0, 64, Vec::new(), 0.0);
    show(&grid, "a=64, b=64 (both over)", 5);
    println!("  → dispatcher emits 503 + Retry-After, bumps route_saturation_total");

    println!("\n== Same ceiling, but the layer has no owners ==");
    show(&grid, "layer 99 (not registered)", 99);
    println!("  → dispatcher emits 400 — has_owners_for distinguishes 400 from 503");

    println!("\n== Ceiling cleared — pre-ADR-0020 behaviour restored ==");
    grid.set_saturation_ceiling(None);
    show(&grid, "a=64, b=64, ceiling=None", 5);
    println!("  → least-bad replica still routed; no load-shedding (legacy behavior)");
}
