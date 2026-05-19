//! Q4_K dense-remote parity check — the Act 1.5 story as a cargo example.
//!
//! Drives both ends of the Q4_K remote-FFN split on a single machine:
//! local `predict_kquant` vs `predict_kquant_with_ffn` pointing at a running
//! `larql serve --ffn-only` on the same vindex. Asserts:
//!
//! - top-1 token id matches between local and remote forwards
//! - top-K logits match within f32-through-JSON tolerance (`1e-4`)
//! - client output label is `walk (q4k + ffn remote)` — no silent
//!   fall-through to local FFN
//!
//! This is the reproducible, in-process version of the dense-remote
//! demo. It exists because shell-driven tests are brittle (prompt
//! escaping, RSS read races, trap ordering) and cargo examples plug
//! into CI without extra machinery.
//!
//! # Setup
//!
//! ```bash
//! # Terminal A — start an FFN-service on a Q4_K vindex.
//! cargo run --release -p larql-cli -- serve path/to/gemma4-31b-q4k.vindex \
//!   --port 8088 --ffn-only \
//!   --max-gate-cache-layers 4 \
//!   --release-mmap-after-request \
//!   --log-level warn
//! ```
//!
//! ```bash
//! # Terminal B — parity check.
//! cargo run --release -p larql-inference --example q4k_remote_parity -- \
//!   --vindex path/to/gemma4-31b-q4k.vindex \
//!   --server http://127.0.0.1:8088 \
//!   --prompt "The capital of France is"
//! ```
//!
//! Expected output: `OK — top-1 match, max_abs <= 1e-4`.
//!
//! # Notes
//!
//! - The vindex must be Q4_K (`extract --quant q4k`). On f32 vindexes the
//!   script errors out explicitly — use `remote_walk_parity.rs` for that.
//! - Requires `tokenizer.json` next to the vindex (the standard extract
//!   places it there automatically).
//! - The demo script in `docs/demo-script-gemma4-moe.md` §Act 1.5
//!   reproduces the same user-facing command; this example is the
//!   programmatic counterpart.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use larql_inference::ffn::{RemoteFfnConfig, RemoteWalkBackend};
use larql_inference::vindex::{predict_kquant, predict_kquant_with_ffn};
use larql_vindex::{
    load_model_weights_kquant, load_vindex_config, load_vindex_tokenizer, QuantFormat,
    SilentLoadCallbacks, VectorIndex,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut vindex_path = PathBuf::new();
    let mut server_url = String::from("http://127.0.0.1:8088");
    let mut prompt = String::from("The capital of France is");
    let mut top_k: usize = 5;
    let mut tolerance: f64 = 1e-4;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--vindex" => {
                i += 1;
                vindex_path = PathBuf::from(&args[i]);
            }
            "--server" => {
                i += 1;
                server_url = args[i].clone();
            }
            "--prompt" => {
                i += 1;
                prompt = args[i].clone();
            }
            "--top-k" => {
                i += 1;
                top_k = args[i].parse()?;
            }
            "--tolerance" => {
                i += 1;
                tolerance = args[i].parse()?;
            }
            "-h" | "--help" => {
                print_usage();
                return Ok(());
            }
            _ => eprintln!("unknown arg: {}", args[i]),
        }
        i += 1;
    }

    if !vindex_path.is_dir() {
        print_usage();
        std::process::exit(1);
    }

    println!("== Q4_K dense-remote parity check ==");
    println!("  vindex:    {}", vindex_path.display());
    println!("  server:    {server_url}");
    println!("  prompt:    {prompt:?}");
    println!("  top_k:     {top_k}");
    println!("  tolerance: {tolerance:.0e}");
    println!();

    // ── Verify vindex is Q4_K ──
    let config = load_vindex_config(&vindex_path)?;
    if config.quant != QuantFormat::Q4K {
        return Err(format!(
            "vindex quant is {:?}, expected Q4K — use remote_walk_parity.rs for float vindexes",
            config.quant
        )
        .into());
    }

    // ── Load tokenizer + Q4K weights shared by both paths ──
    let tokenizer = load_vindex_tokenizer(&vindex_path)?;
    let mut cb = SilentLoadCallbacks;
    let mut weights_local = load_model_weights_kquant(&vindex_path, &mut cb)?;
    let mut weights_remote = load_model_weights_kquant(&vindex_path, &mut cb)?;

    // Tokenise the prompt through the architecture-specific encoder (adds BOS etc.).
    let token_ids = larql_inference::encode_prompt(&tokenizer, &*weights_local.arch, &prompt)
        .map_err(|e| format!("tokenize error: {e}"))?;
    println!("Prompt tokens: {} ids", token_ids.len());

    // ── Local path: full q4k forward in-process ──
    let mut local_index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    local_index.load_attn_kquant(&vindex_path)?;
    local_index.load_interleaved_kquant(&vindex_path)?;

    let t_local = Instant::now();
    let local_result = predict_kquant(
        &mut weights_local,
        &tokenizer,
        &token_ids,
        top_k,
        &local_index,
    );
    let local_ms = t_local.elapsed().as_secs_f64() * 1000.0;

    // ── Remote path: attention local, FFN over HTTP via RemoteWalkBackend ──
    let remote_config = RemoteFfnConfig::new(&server_url).with_timeout(Duration::from_secs(120));
    let remote = RemoteWalkBackend::connect(remote_config).map_err(|e| {
        format!(
            "remote connect failed ({server_url}): {e}\n\
                              → is `larql serve {} --ffn-only` running on {server_url}?",
            vindex_path.display()
        )
    })?;
    assert_eq!(
        remote.hidden_size(),
        weights_remote.hidden_size,
        "remote hidden_size mismatch",
    );

    // Client-side VectorIndex: only attention Q4_K mmap, NO interleaved_kquant.bin.
    // (The FFN lives on the server; loading it client-side would defeat the demo.)
    let mut remote_index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    remote_index.load_attn_kquant(&vindex_path)?;

    let t_remote = Instant::now();
    let remote_result = predict_kquant_with_ffn(
        &mut weights_remote,
        &tokenizer,
        &token_ids,
        top_k,
        &remote_index,
        &remote,
    );
    let remote_ms = t_remote.elapsed().as_secs_f64() * 1000.0;

    // ── Compare ──
    println!();
    println!("Top-{top_k}:");
    println!(
        "  {:<24} {:>10} | {:<24} {:>10}",
        "local", "prob", "remote", "prob"
    );
    for i in 0..top_k {
        let (lt, lp) = local_result
            .predictions
            .get(i)
            .cloned()
            .unwrap_or_else(|| ("<missing>".into(), 0.0));
        let (rt, rp) = remote_result
            .predictions
            .get(i)
            .cloned()
            .unwrap_or_else(|| ("<missing>".into(), 0.0));
        let marker = if lt == rt && (lp - rp).abs() < tolerance {
            ""
        } else {
            "  ← diff"
        };
        println!("  {lt:<24} {lp:>10.4} | {rt:<24} {rp:>10.4}{marker}");
    }
    println!();

    // Top-1 token-id must match.
    let local_top = local_result.token_ids.first().copied();
    let remote_top = remote_result.token_ids.first().copied();
    if local_top != remote_top {
        eprintln!("FAIL — top-1 token id differs: local={local_top:?} remote={remote_top:?}");
        std::process::exit(1);
    }

    // Max per-position probability delta across the top-K.
    let mut max_abs = 0f64;
    for i in 0..top_k
        .min(local_result.predictions.len())
        .min(remote_result.predictions.len())
    {
        let (_lt, lp) = &local_result.predictions[i];
        let (_rt, rp) = &remote_result.predictions[i];
        let d = (lp - rp).abs();
        if d > max_abs {
            max_abs = d;
        }
    }

    let pass = max_abs <= tolerance;
    println!("Timing: local={local_ms:.1}ms  remote={remote_ms:.1}ms");
    println!("Parity: top-1 match, max_abs on top-{top_k} = {max_abs:.2e}  (tol {tolerance:.0e})");
    if pass {
        println!("OK");
        Ok(())
    } else {
        eprintln!("FAIL — top-{top_k} probabilities exceed tolerance");
        std::process::exit(1);
    }
}

fn print_usage() {
    eprintln!(
        "Usage: q4k_remote_parity \
         --vindex PATH \
         --server URL \
         [--prompt TEXT] \
         [--top-k N] \
         [--tolerance 1e-4]\n\
         \n\
         Requires a running `larql serve <vindex> --port <port> --ffn-only` \
         reachable at --server URL. The vindex must be Q4_K."
    );
}
