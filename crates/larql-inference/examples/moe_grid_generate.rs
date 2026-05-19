//! End-to-end demo: generation through a sharded expert grid.
//!
//! Usage (two shards):
//!   larql-server <vindex> --experts 0-63  --port 9191 &
//!   larql-server <vindex> --experts 64-127 --port 9192 &
//!
//!   VINDEX=~/chris-models/gemma-4-26B-A4B-it.vindex \
//!   SHARDS="0-63:http://localhost:9191,64-127:http://localhost:9192" \
//!   PROMPT="The capital of France is" \
//!   MAX_TOKENS=8 \
//!   cargo run --release --example moe_grid_generate
//!
//! Single-server shortcut (all experts):
//!   SHARDS="0-127:http://localhost:9191" ...

extern crate blas_src;

use larql_inference::{
    encode_prompt, layer_graph::grid::generate_with_remote_moe, EosConfig, RemoteMoeBackend,
    ShardConfig,
};
use larql_vindex::{load_vindex_tokenizer, SilentLoadCallbacks, VectorIndex};
use std::sync::Arc;

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

fn main() -> Result<(), BoxErr> {
    let vindex_path = std::env::var("VINDEX")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_default();
            std::path::PathBuf::from(home).join("chris-models/gemma-4-26B-A4B-it.vindex")
        });

    let shards_spec =
        std::env::var("SHARDS").unwrap_or_else(|_| "0-127:http://localhost:9191".into());
    let prompt = std::env::var("PROMPT").unwrap_or_else(|_| "The capital of France is".into());
    let max_tokens: usize = std::env::var("MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    println!("vindex : {}", vindex_path.display());
    println!("shards : {shards_spec}");
    println!("prompt : \"{prompt}\"");
    println!("tokens : {max_tokens}");
    println!();

    // ── Parse shard spec "START-END:URL,..." ─────────────────────────────────
    let shard_configs: Vec<ShardConfig> = shards_spec
        .split(',')
        .map(|piece| {
            // Find the colon that separates range from URL (URL contains colons too).
            let dash = piece.find('-').unwrap_or(0);
            let colon = piece[dash..]
                .find(':')
                .map(|c| c + dash)
                .unwrap_or(piece.len());
            let range_str = &piece[..colon];
            let url_str = piece[colon + 1..].to_string();
            let (start, end) = parse_range(range_str);
            ShardConfig::new(start, end, url_str)
        })
        .collect();

    println!("Connecting to {} shard(s)…", shard_configs.len());
    let remote = Arc::new(RemoteMoeBackend::connect(shard_configs)?);
    println!("Connected.\n");

    // ── Load vindex + model weights ───────────────────────────────────────────
    print!("Loading vindex… ");
    std::io::Write::flush(&mut std::io::stdout()).ok();
    let t0 = std::time::Instant::now();
    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    index.load_attn_kquant(&vindex_path).ok();
    index.load_interleaved_kquant(&vindex_path).ok();

    let cfg = larql_vindex::load_vindex_config(&vindex_path)?;
    let weights = larql_vindex::load_model_weights_kquant(&vindex_path, &mut cb)?;
    let tokenizer = load_vindex_tokenizer(&vindex_path)?;
    println!(
        "done ({:.1}s)  model={} layers={} hidden={}",
        t0.elapsed().as_secs_f64(),
        cfg.model,
        cfg.num_layers,
        cfg.hidden_size
    );

    // ── Backend (Metal or CPU) ────────────────────────────────────────────────
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let backend = larql_compute_metal::MetalBackend::new().ok_or("Metal not available")?;
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    let backend = larql_compute::CpuBackend;

    // ── Tokenize ──────────────────────────────────────────────────────────────
    let arch = &*weights.arch;
    let prompt_ids = encode_prompt(&tokenizer, arch, &prompt)?;
    println!("Prompt tokens: {}", prompt_ids.len());
    println!();

    // ── Generate ─────────────────────────────────────────────────────────────
    print!("{prompt}");
    std::io::Write::flush(&mut std::io::stdout()).ok();

    let eos = EosConfig::from_vindex_dir(&vindex_path);
    let result = generate_with_remote_moe(
        &weights, &tokenizer, prompt_ids, max_tokens, &index, &remote, &backend, &eos,
    )?;

    for (tok, ms) in result.tokens.iter().zip(result.decode_ms.iter()) {
        print!("{tok}");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        eprintln!("  [{ms:.0}ms]");
    }
    // Print remaining tokens that have no latency entry (prefill token).
    for tok in result.tokens.iter().skip(result.decode_ms.len()) {
        print!("{tok}");
    }
    println!();
    println!(
        "\n{} tokens  avg decode {:.0}ms/tok",
        result.tokens.len(),
        result.decode_ms.iter().sum::<f64>() / result.decode_ms.len().max(1) as f64
    );

    Ok(())
}

fn parse_range(s: &str) -> (usize, usize) {
    let parts: Vec<&str> = s.splitn(2, '-').collect();
    let start = parts
        .first()
        .and_then(|p| p.trim().parse().ok())
        .unwrap_or(0);
    let end = parts
        .get(1)
        .and_then(|p| p.trim().parse().ok())
        .unwrap_or(start);
    (start, end)
}
