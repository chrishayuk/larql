//! I/O-bound runtime for the remote-FFN bench. Lives in its own file so the
//! coverage policy can exclude it cleanly — the inner functions wrap
//! `LayerShardedBackend::connect`, vindex loading, and
//! `generate_with_remote_ffn`, none of which we unit-test (no live FFN
//! server in CI).
//!
//! Pure post-processing lives in `remote_ffn.rs` and is gated to 90%+.

use super::args::BenchArgs;
use super::remote_ffn::{
    combine_concurrent_rows, compute_wire_bytes_per_tok, format_ffn_backend_label,
    summarize_ffn_result,
};
use super::row::BenchRow;

/// Run `args.concurrent` parallel FFN clients against the same shard and
/// aggregate them into one row. With `concurrent == 1` this is a
/// pass-through to `run_remote_ffn_bench`.
pub(super) fn run_concurrent_ffn(
    vindex_path: &std::path::Path,
    args: &BenchArgs,
    ffn_url: &str,
    pref: larql_inference::WirePreference,
) -> Result<BenchRow, Box<dyn std::error::Error>> {
    let n = args.concurrent.max(1);
    if n == 1 {
        return run_remote_ffn_bench(vindex_path, args, ffn_url, pref);
    }

    let mut handles = Vec::with_capacity(n);
    for _ in 0..n {
        let vp = vindex_path.to_path_buf();
        let a = args.clone();
        let url = ffn_url.to_string();
        handles.push(std::thread::spawn(move || {
            run_remote_ffn_bench(&vp, &a, &url, pref).map_err(|e| e.to_string())
        }));
    }

    let mut rows: Vec<BenchRow> = Vec::with_capacity(n);
    for h in handles {
        match h.join() {
            Ok(Ok(row)) => rows.push(row),
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => {
                return Err("concurrent FFN bench worker panicked — see stderr for details".into());
            }
        }
    }
    Ok(combine_concurrent_rows(rows, n))
}

/// Bench the remote-FFN path: attention runs locally on Metal, FFN is a
/// round-trip to `ffn_url` via `LayerShardedBackend`.
pub(super) fn run_remote_ffn_bench(
    vindex_path: &std::path::Path,
    args: &BenchArgs,
    ffn_url: &str,
    wire_pref: larql_inference::WirePreference,
) -> Result<BenchRow, Box<dyn std::error::Error>> {
    use larql_inference::{
        generate_with_remote_ffn, generate_with_remote_ffn_batch, LayerShardedBackend,
    };
    use std::time::Duration;

    if args.verbose {
        eprintln!("[bench] loading vindex for remote-ffn…");
    }

    let timeout = Duration::from_secs(args.ffn_timeout_secs);
    let backend = larql_compute::default_backend();

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_kquant(vindex_path, &mut cb)
        .map_err(|e| format!("failed to load client weights: {e}"))?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex_path)
        .map_err(|e| format!("failed to load tokenizer: {e}"))?;
    let mut index = larql_vindex::VectorIndex::load_vindex(vindex_path, &mut cb)
        .map_err(|e| format!("failed to load vindex: {e}"))?;
    index.load_attn_kquant(vindex_path)?;
    index.load_interleaved_kquant(vindex_path)?;
    let _ = index.load_lm_head_kquant(vindex_path);

    eprintln!("Connecting to remote FFN at {ffn_url}…");
    let remote = LayerShardedBackend::connect_with_wire(ffn_url, timeout, wire_pref)
        .map_err(|e| format!("failed to connect to remote FFN: {e}"))?;
    eprintln!("  Attention:  {} (local)", backend.name());
    eprintln!("  FFN:        remote  ({})", ffn_url);

    let wrapped_prompt =
        larql_inference::chat::render_user_prompt(vindex_path, weights.arch.family(), &args.prompt)
            .unwrap_or_else(|_| args.prompt.clone());
    let prompt_ids = larql_inference::encode_prompt(&tokenizer, &*weights.arch, &wrapped_prompt)
        .map_err(|e| format!("tokenise: {e}"))?;

    let eos = larql_inference::layer_graph::generate::eos::EosConfig::from_vindex_dir(vindex_path);
    let max_tokens = args.warmup + args.tokens;

    let is_batch = args.ffn_dispatch.trim() == "batch";

    if args.verbose {
        eprintln!("[bench] remote-ffn warmup ({} tokens)…", args.warmup.max(1));
    }
    if is_batch {
        let _ = generate_with_remote_ffn_batch(
            &weights,
            &tokenizer,
            prompt_ids.clone(),
            args.warmup.max(1),
            &index,
            &*backend,
            &remote,
            &eos,
            1,
        );
    } else {
        let _ = generate_with_remote_ffn(
            &weights,
            &tokenizer,
            prompt_ids.clone(),
            args.warmup.max(1),
            &index,
            &*backend,
            &remote,
            &eos,
        );
    }

    remote.reset_wire_counters();

    let _t_wall = std::time::Instant::now();
    let result = if is_batch {
        generate_with_remote_ffn_batch(
            &weights,
            &tokenizer,
            prompt_ids.clone(),
            max_tokens,
            &index,
            &*backend,
            &remote,
            &eos,
            1,
        )
        .map_err(|e| format!("remote-ffn generate failed (batch): {e}"))?
    } else {
        generate_with_remote_ffn(
            &weights,
            &tokenizer,
            prompt_ids.clone(),
            max_tokens,
            &index,
            &*backend,
            &remote,
            &eos,
        )
        .map_err(|e| format!("remote-ffn generate failed: {e}"))?
    };

    let summary = summarize_ffn_result(
        &result.decode_ms,
        &result.ffn_rtt_ms,
        args.warmup,
        args.tokens,
    );
    let wire_bytes_per_tok = compute_wire_bytes_per_tok(
        remote.wire_bytes_sent() + remote.wire_bytes_recv(),
        summary.n_steps,
    );

    let _ = weights; // keep alive through the bench

    Ok(BenchRow {
        backend: format_ffn_backend_label(is_batch, wire_pref, ffn_url),
        prefill_ms: 0.0,
        avg_decode_ms: summary.avg_decode_ms,
        p50_ms: summary.p50_ms,
        p99_ms: summary.p99_ms,
        tok_per_s: summary.tok_per_s,
        stages: None,
        ffn_rtt_ms: summary.ffn_rtt_ms,
        attn_ms: summary.attn_ms,
        wire_bytes_per_tok,
        shard_efficiency: None,
        n_steps: summary.n_steps,
        note: summary.note,
    })
}
