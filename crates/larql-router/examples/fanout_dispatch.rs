//! The fan-out dispatch building blocks in isolation.
//!
//! A real `POST /v1/walk-ffn` request spanning multiple shards is
//! handled in three steps:
//!
//!   1. **Resolve** each layer to its owning shard URL
//!      (`AppState::resolve_all` / `resolve_static_only`).
//!   2. **Group** the resolved layer→url map back into url→layers
//!      so each shard gets one sub-request
//!      (`group_layers_by_url`).
//!   3. **Build** each shard's sub-request body from the original
//!      JSON template (`build_subrequest_body`).
//!   4. **Merge** all of the shards' responses into a single
//!      `{results: [...], latency_ms: max}` envelope
//!      (`merge_shard_responses`).
//!
//! This example exercises 1-4 against a synthetic shard map and
//! synthetic shard responses — no network. Useful when you're
//! building your own dispatcher and want to verify the JSON shape.
//!
//! Run with `cargo run -p larql-router --example fanout_dispatch`.

use larql_router::dispatch::{
    build_subrequest_body, group_layers_by_url, merge_shard_responses, resolve_static_only,
};
use larql_router::shards::parse_shards;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Static shard map: layers 0-2 on shard-a, 3-5 on shard-b.
    let shards = parse_shards("0-2=http://shard-a,3-5=http://shard-b")?;
    println!("== Step 1: parse_shards + resolve ==");

    // Caller wants the FFN walk for layers 0, 2, 3, 5 (skipping 1 and 4).
    let layers = vec![0, 2, 3, 5];
    let layer_urls = resolve_static_only(&shards, &layers)
        .map_err(|layer| format!("uncovered layer {layer}"))?;
    println!("  layers requested: {layers:?}");
    println!("  resolved (layer -> url):");
    let mut sorted: Vec<_> = layer_urls.iter().collect();
    sorted.sort_by_key(|(k, _)| **k);
    for (l, u) in &sorted {
        println!("    {l} -> {u}");
    }

    // 2. Group the layer→url map so each shard gets one call.
    let by_url = group_layers_by_url(&layer_urls);
    println!("\n== Step 2: group_layers_by_url ==");
    let mut sorted: Vec<_> = by_url.iter().collect();
    sorted.sort_by_key(|(url, _)| (*url).clone());
    for (url, ls) in &sorted {
        let mut ls = (*ls).clone();
        ls.sort();
        println!("    {url} -> {ls:?}");
    }

    // 3. Build the per-shard sub-request bodies from a JSON template.
    let template = json!({
        "model_id": "gemma3:4b",
        "tokens":   [1, 2, 3],
        // `layers` will be replaced per-shard; the dispatcher strips
        // whichever of `layer`/`layers` doesn't match the shard's slice.
        "layers":   layers,
    });
    println!("\n== Step 3: build_subrequest_body (per shard) ==");
    for (url, ls) in &sorted {
        let mut ls = (*ls).clone();
        ls.sort();
        let body = build_subrequest_body(&template, &ls);
        println!("  POST {url}");
        println!("    {body}");
    }

    // 4. Pretend each shard replied; merge the responses.
    let shard_responses = vec![
        json!({
            "results": [
                {"layer": 0, "out": [0.1, 0.2]},
                {"layer": 2, "out": [0.3, 0.4]},
            ],
            "latency_ms": 4.3,
        }),
        json!({
            "results": [
                {"layer": 3, "out": [0.5, 0.6]},
                {"layer": 5, "out": [0.7, 0.8]},
            ],
            "latency_ms": 5.1, // slower shard sets the envelope latency.
        }),
    ];
    let merged = merge_shard_responses(&shard_responses);
    println!("\n== Step 4: merge_shard_responses ==");
    println!("  {}", serde_json::to_string_pretty(&merged)?);

    Ok(())
}
