//! I/O-bound runtime for the remote-MoE bench. Lives in its own file so the
//! coverage policy can exclude it cleanly — the inner functions wrap
//! `RemoteMoeBackend::connect`, vindex loading, and `generate_with_remote_moe`,
//! none of which we unit-test (no live shards in CI).
//!
//! Pure post-processing lives in `remote_moe.rs` and is gated to 90%+.

use super::args::BenchArgs;
use super::remote_ffn::combine_concurrent_rows;
use super::remote_moe::{format_moe_backend_label, parse_shard_segments, summarize_moe_result};
use super::row::BenchRow;

/// Run `args.concurrent` parallel MoE clients against the same shard map
/// and aggregate them into one row. `concurrent == 1` short-circuits to
/// `run_remote_moe_bench`.
pub(super) fn run_concurrent_moe(
    vindex_path: &std::path::Path,
    args: &BenchArgs,
    shards_str: &str,
) -> Result<BenchRow, Box<dyn std::error::Error>> {
    let n = args.concurrent.max(1);
    if n == 1 {
        return run_remote_moe_bench(vindex_path, args, shards_str);
    }

    let mut handles = Vec::with_capacity(n);
    for _ in 0..n {
        let vp = vindex_path.to_path_buf();
        let a = args.clone();
        let shards = shards_str.to_string();
        handles.push(std::thread::spawn(move || {
            run_remote_moe_bench(&vp, &a, &shards).map_err(|e| e.to_string())
        }));
    }
    let mut rows: Vec<BenchRow> = Vec::with_capacity(n);
    for h in handles {
        match h.join() {
            Ok(Ok(row)) => rows.push(row),
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => {
                return Err("concurrent MoE bench worker panicked — see stderr for details".into());
            }
        }
    }
    Ok(combine_concurrent_rows(rows, n))
}

/// Bench the remote MoE expert path. Attention + router run locally; expert
/// blocks are dispatched to remote shards via `RemoteMoeBackend`.
pub(super) fn run_remote_moe_bench(
    vindex_path: &std::path::Path,
    args: &BenchArgs,
    shards_str: &str,
) -> Result<BenchRow, Box<dyn std::error::Error>> {
    use larql_inference::ffn::moe_remote::RemoteMoeBackend;
    use larql_inference::{generate_with_remote_moe, generate_with_remote_moe_batch};

    let configs = parse_shard_segments(shards_str)?;
    let num_shards = configs.len();
    let backend = larql_compute::default_backend();
    eprintln!("Connecting to {} MoE shard(s)…", num_shards);
    let remote = RemoteMoeBackend::connect(configs)
        .map_err(|e| format!("failed to connect to MoE shards: {e}"))?;
    eprintln!("  Attention:  {} (local)", backend.name());
    eprintln!("  Router:     local");
    eprintln!(
        "  Experts:    remote  (sharded across {} endpoint{})",
        num_shards,
        if num_shards == 1 { "" } else { "s" }
    );

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

    let wrapped_prompt =
        larql_inference::chat::render_user_prompt(vindex_path, weights.arch.family(), &args.prompt)
            .unwrap_or_else(|_| args.prompt.clone());
    let prompt_ids = larql_inference::encode_prompt(&tokenizer, &*weights.arch, &wrapped_prompt)
        .map_err(|e| format!("tokenise: {e}"))?;

    let eos = larql_inference::layer_graph::generate::eos::EosConfig::from_vindex_dir(vindex_path);
    let max_tokens = args.warmup + args.tokens;
    let is_batch = args.moe_dispatch.trim() == "batch";
    let iters = args.moe_predispatch_iters.max(1);

    let run_once =
        |n: usize| -> Result<larql_inference::layer_graph::grid::GridGenerateResult, String> {
            if is_batch {
                generate_with_remote_moe_batch(
                    &weights,
                    &tokenizer,
                    prompt_ids.clone(),
                    n,
                    &index,
                    &remote,
                    &*backend,
                    &eos,
                    iters,
                )
                .map_err(|e| e.to_string())
            } else {
                generate_with_remote_moe(
                    &weights,
                    &tokenizer,
                    prompt_ids.clone(),
                    n,
                    &index,
                    &remote,
                    &*backend,
                    &eos,
                )
                .map_err(|e| e.to_string())
            }
        };

    let _ = run_once(args.warmup.max(1));

    let result = run_once(max_tokens).map_err(|e| format!("moe bench generate failed: {e}"))?;

    let summary = summarize_moe_result(
        &result.decode_ms,
        &result.ffn_rtt_ms,
        args.warmup,
        args.tokens,
    );

    Ok(BenchRow {
        backend: format_moe_backend_label(is_batch, num_shards),
        prefill_ms: 0.0,
        avg_decode_ms: summary.avg_decode_ms,
        p50_ms: summary.p50_ms,
        p99_ms: summary.p99_ms,
        tok_per_s: summary.tok_per_s,
        stages: None,
        ffn_rtt_ms: summary.ffn_rtt_ms,
        attn_ms: summary.attn_ms,
        wire_bytes_per_tok: None,
        shard_efficiency: None,
        n_steps: summary.n_steps,
        note: summary.note,
    })
}
