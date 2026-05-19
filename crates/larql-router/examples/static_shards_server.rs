//! Minimal HTTP router with a hard-coded static shard map — no CLI,
//! no grid, no rebalancer. Shows the smallest possible deployment
//! path: `parse_shards` → `AppState` → `build_router` → `axum::serve`.
//!
//! Hit it after startup:
//!   curl http://127.0.0.1:9090/v1/health
//!   curl -X POST http://127.0.0.1:9090/v1/walk-ffn \
//!        -H 'Content-Type: application/json' \
//!        -d '{"layer": 0}'
//!
//! The walk-ffn POST will 502 because the example doesn't actually
//! stand up backend shards at the configured URLs — the point is to
//! show how the request-routing wiring is assembled, not to serve
//! real inference. To exercise the full path, run real
//! `larql-server` instances at the URLs in `SHARD_SPEC`.
//!
//! Run with `cargo run -p larql-router --example static_shards_server`.

use std::net::SocketAddr;
use std::sync::Arc;

use larql_router::cli_helpers::build_shard_client;
use larql_router::http::{build_router, AppState};
use larql_router::shards::parse_shards;

/// Two-shard map covering layers 0-29 (a 30-layer model split 50/50).
const SHARD_SPEC: &str = "0-14=http://localhost:9181,15-29=http://localhost:9182";

const LISTEN_ADDR: &str = "127.0.0.1:9090";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse `START-END=URL,...` into a Vec<Shard>.
    let shards = parse_shards(SHARD_SPEC)?;
    println!("Loaded {} static shards from spec:", shards.len());
    for s in &shards {
        // `layer_end` is half-open in the struct; subtract 1 to print
        // the inclusive end the user typed in the spec.
        println!(
            "  layers {}-{} -> {}",
            s.layer_start,
            s.layer_end - 1,
            s.url
        );
    }

    // Shared reqwest::Client with a connection pool reused across
    // every fan-out call. 120 s timeout matches the default
    // `--timeout-secs` in the CLI binary.
    let client = build_shard_client(120)?;

    let state = Arc::new(AppState {
        static_shards: shards,
        grid: None, // No self-assembling grid in this example.
        client,
        // Metrics are optional; the example skips them for brevity.
        // Production builds carry `Some(RouterMetrics::new())` here.
        metrics: None,
        #[cfg(feature = "http3")]
        h3_client: None,
        hedge_after: None,
    });

    let app = build_router(state);
    let addr: SocketAddr = LISTEN_ADDR.parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("\nlarql-router listening on http://{addr}");
    println!("  GET  /v1/health");
    println!("  POST /v1/walk-ffn   (proxies to the static shard map)");
    println!("\nCtrl+C to stop.");

    axum::serve(listener, app).await?;
    Ok(())
}
