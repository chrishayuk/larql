//! Programmatic admin RPC client.
//!
//! Calls `admin_status`, `admin_drain`, and `admin_assign` against a
//! running router's gRPC port. These are the same code paths the
//! `larql-router status / drain / assign` subcommands use; calling
//! them as a library lets you wrap the router from your own ops
//! tooling (a dashboard, a Slack bot, a custom autoscaler, …).
//!
//! Prereq: a `larql-router` instance listening on `--grid-port 50052`.
//! Either run one yourself or change `ROUTER_URL` below.
//!
//! Run with `cargo run -p larql-router --example admin_client`.

use larql_router::admin::{admin_assign, admin_drain, admin_status, AdminError};

const ROUTER_URL: &str = "http://127.0.0.1:50052";

/// Replace these with real values from your `admin_status` output.
const TARGET_SERVER_ID: &str = "srv-1700000000-1";
const TARGET_MODEL_ID: &str = "gemma3:4b";
const TARGET_LAYERS: &str = "0-14";

#[tokio::main]
async fn main() -> Result<(), AdminError> {
    // 1. Snapshot the grid. Same rendering the CLI uses.
    println!("== admin_status ==");
    match admin_status(ROUTER_URL).await {
        Ok(lines) => {
            for line in lines {
                println!("  {line}");
            }
        }
        Err(e) => {
            eprintln!("  status failed: {e}");
            eprintln!("\nIs there a router on {ROUTER_URL}?");
            eprintln!("  larql-router --grid-port 50052");
            return Ok(());
        }
    }

    // 2. Drain a specific server. (Real callers would parse the
    //    status response for a server_id; we hard-code one here.)
    println!("\n== admin_drain ==");
    let ack = admin_drain(ROUTER_URL, TARGET_SERVER_ID, "operator-driven drain").await?;
    println!("  ok      = {}", ack.ok);
    println!("  message = {:?}", ack.message);

    // 3. Assign a range explicitly. Pass `Some(target_server_id)` to
    //    pin the destination, or `None` to let the router pick any
    //    spare. Pass `origin_url=Some(...)` for external origins
    //    (S3, mirror, etc.), or `None` to let the router resolve
    //    from a live replica.
    println!("\n== admin_assign ==");
    let ack = admin_assign(
        ROUTER_URL,
        TARGET_MODEL_ID,
        TARGET_LAYERS,
        /* target_server_id */ None,
        /* origin_url       */ Some("http://origin-store.local/gemma3-4b-0-14.vindex"),
        /* origin_hash      */ "deadbeef",
    )
    .await?;
    println!("  ok      = {}", ack.ok);
    println!("  message = {:?}", ack.message);

    Ok(())
}
