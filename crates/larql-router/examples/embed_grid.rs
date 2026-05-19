//! Programmatic GridState wiring.
//!
//! Demonstrates the router as a library: build a GridState by hand,
//! register a couple of serving servers + a Mode B spare, query
//! routes, inspect coverage gaps, and exercise the rebalancer's
//! state-shaping methods directly. No gRPC server, no HTTP listener,
//! no tokio loop running — just the in-memory data structures.
//!
//! Run with `cargo run -p larql-router --example embed_grid`.

use std::collections::HashMap;
use std::time::Instant;

use larql_router::grid::{GridState, ServerEntry};

fn server(
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

fn main() {
    let mut grid = GridState::default();

    // Register two serving shards of gemma-3-4b: layers 0-14 on shard-a,
    // layers 15-29 on shard-b. (32-layer model — 30-31 left uncovered to
    // showcase coverage_gaps.)
    grid.register(server("a", "http://shard-a:9181", "gemma3:4b", 0, 14));
    grid.register(server("b", "http://shard-b:9182", "gemma3:4b", 15, 29));

    // Routing a covered layer returns the owning shard's URL.
    println!("== Routing ==");
    let route_5 = grid.route(Some("gemma3:4b"), 5).unwrap();
    println!("  layer  5 -> {route_5}");
    let route_20 = grid.route(Some("gemma3:4b"), 20).unwrap();
    println!("  layer 20 -> {route_20}");
    println!(
        "  layer 30 -> {:?}  (no shard)",
        grid.route(Some("gemma3:4b"), 30)
    );

    // route_all batches the lookup; returns Err(first uncovered layer).
    let plan = grid.route_all(Some("gemma3:4b"), &[0, 5, 14, 15, 29]);
    println!("\n== Batched route_all ==");
    println!("  contiguous coverage: {plan:?}");
    let partial = grid.route_all(Some("gemma3:4b"), &[0, 5, 30, 31]);
    println!("  partial coverage:    {partial:?}");

    // Hot-shard elevation. Once a range crosses the configured per-shard
    // req/sec threshold the rebalancer raises its effective target by 1
    // so the under-replication tick will pull a spare.
    grid.set_target_replicas(1);
    // ADR-0018: dense shards pass 0/0 for the expert range.
    grid.mark_elevated("gemma3:4b", 0, 14, 0, 0);
    println!("\n== Hot-shard elevation ==");
    println!(
        "  effective_target_for(0-14) = {}",
        grid.effective_target_for("gemma3:4b", 0, 14, 0, 0)
    );
    println!(
        "  under_replicated_ranges    = {:?}",
        grid.under_replicated_ranges()
    );
    grid.demote_elevated("gemma3:4b", 0, 14, 0, 0);

    // Coverage gaps + over/under-replication ledger.
    println!("\n== Coverage + replication ==");
    println!("  coverage_gaps              = {:?}", grid.coverage_gaps());
    println!(
        "  under_replicated_ranges    = {:?}",
        grid.under_replicated_ranges()
    );
    println!(
        "  over_replicated_ranges     = {:?}",
        grid.over_replicated_ranges()
    );

    // status_response is what the gRPC `status` RPC returns; the
    // `larql-router status` CLI sub-command formats it for humans.
    let snap = grid.status_response();
    println!("\n== status_response ==");
    println!("  servers reported: {}", snap.servers.len());
    println!(
        "  shards in model:  {}",
        snap.models
            .iter()
            .find(|m| m.model_id == "gemma3:4b")
            .map(|m| m.shards.len())
            .unwrap_or(0)
    );

    // ── ADR-0018: MoE expert routing ────────────────────────────────────────
    //
    // Demo with a Mixtral-8x7B-shaped layer: 1 layer (call it layer 100),
    // 8 experts split into 2 expert-shards (experts 0-3 + experts 4-7).
    // Two physical hosts; each owns one expert-shard.
    let mut moe_lo = server("moe-lo", "http://moe-lo:9101", "mixtral:8x7b", 100, 100);
    moe_lo.expert_start = 0;
    moe_lo.expert_end = 3;
    let mut moe_hi = server("moe-hi", "http://moe-hi:9102", "mixtral:8x7b", 100, 100);
    moe_hi.expert_start = 4;
    moe_hi.expert_end = 7;
    grid.register(moe_lo);
    grid.register(moe_hi);

    println!("\n== MoE routing ==");
    println!(
        "  route_expert(layer=100, expert=0) -> {:?}",
        grid.route_expert(Some("mixtral:8x7b"), 100, 0)
    );
    println!(
        "  route_expert(layer=100, expert=5) -> {:?}",
        grid.route_expert(Some("mixtral:8x7b"), 100, 5)
    );
    println!(
        "  route_expert(layer=100, expert=99 — out of range) -> {:?}",
        grid.route_expert(Some("mixtral:8x7b"), 100, 99)
    );
    // Batched form — top-K fan-out for a single token.
    let top_k_pairs = vec![(100usize, 0u32), (100, 3), (100, 5), (100, 7)];
    println!(
        "  route_all_experts(top-4) = {:?}",
        grid.route_all_experts(Some("mixtral:8x7b"), &top_k_pairs)
    );

    // Dense routing still works unchanged — the gemma3:4b shards
    // registered earlier respond to `route()` the way they always did.
    println!("\n== Dense routing still works after MoE registration ==");
    println!(
        "  route(gemma3:4b, layer=5) -> {:?}",
        grid.route(Some("gemma3:4b"), 5)
    );
}
