//! I/O-bound runtime for the local Metal/CPU bench. Excluded from the
//! per-file coverage gate — every call here hits real vindex mmaps, real
//! model weights, and (when `metal`) the Metal pipeline. Pure helpers live
//! in `local.rs`.

use std::time::Instant;

use super::args::BenchArgs;
use super::local::{
    append_cpu_fallback_note, append_repeat_note, backend_name_for, format_early_stop_note,
    format_q4k_cache_log, generation_fingerprint,
};
use super::row::{compute_percentiles, BenchRow};

/// Run the larql generate loop with the selected backend, once per
/// `--repeat`, returning one row per repeat.
///
/// Warmup runs are discarded; the measured window is `args.tokens` steps
/// AFTER warmup. Repeats share one process, so the model is opened and the
/// page cache warmed once for the whole series rather than once per row.
pub(super) fn run_larql(
    vindex_path: &std::path::Path,
    args: &BenchArgs,
    metal: bool,
) -> Result<Vec<BenchRow>, Box<dyn std::error::Error>> {
    use larql_inference::layer_graph::generate::generate;
    use larql_inference::layer_graph::CachedLayerGraph;

    if args.verbose {
        eprintln!(
            "[bench] loading vindex for {}…",
            if metal { "metal" } else { "cpu" }
        );
    }

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let mut index = larql_vindex::VectorIndex::load_vindex(vindex_path, &mut cb)?;
    index.load_attn_kquant(vindex_path)?;
    index.load_interleaved_kquant(vindex_path)?;
    // The k-quant lm_head view. Without it `lm_head_topk` finds no
    // `lm_head_kquant_view()`, falls through to `backend_lm_head_topk`, and
    // runs an f32 gemv over the dequantised `weights.lm_head` — for GPT-OSS
    // that is a 2.3 GB matrix, ~8 ms/token against ~1.5 ms for the Q4_K
    // matvec. Every other inference entry point loads it (`run_cmd`,
    // `walk_cmd`, `diag_cmd`, the server, and bench's own remote-FFN and
    // remote-MoE runtimes); this one did not, so the local bench was
    // measuring a slower lm_head than the path it claims to benchmark.
    let _ = index.load_lm_head_kquant(vindex_path);

    let cfg = larql_vindex::load_vindex_config(vindex_path)?;
    if cfg.quant != larql_vindex::QuantFormat::Q4K {
        return Err(format!(
            "larql bench currently requires a Q4K vindex (got {:?})",
            cfg.quant,
        )
        .into());
    }
    let mut weights = larql_vindex::load_model_weights_kquant(vindex_path, &mut cb)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex_path)?;
    let wrapped_prompt = larql_inference::chat::render_user_prompt(
        vindex_path,
        weights.arch.family(),
        args.prompt.as_str(),
    )
    .unwrap_or_else(|_| args.prompt.to_string());
    let token_ids: Vec<u32> =
        larql_inference::encode_prompt(&tokenizer, &*weights.arch, &wrapped_prompt)
            .map_err(|e| format!("tokenize: {e}"))?;

    let backend: Box<dyn larql_compute::ComputeBackend> =
        crate::backend_select::backend_for_metal_flag(metal)?;

    let cached_layers = CachedLayerGraph::from_residuals(Vec::new());

    // Composed bench: the routed container replaces the expert-bank
    // authority and nothing else, exactly as `larql run --routed-from`.
    // Opened before any timer so composition refusals cannot read as a
    // slow prefill.
    let routed = args
        .routed_from
        .as_deref()
        .map(|dir| {
            larql_inference::ffn::ContainerRoutedBackend::open(
                std::path::Path::new(dir),
                &weights,
                true,
            )
        })
        .transpose()
        .map_err(|e| format!("--routed-from refused: {e}"))?;
    let num_layers = weights.num_layers;
    let generate_n = |weights: &mut larql_models::ModelWeights, max_tokens: usize| match &routed {
        Some(routed) => larql_inference::layer_graph::generate_routed(
            weights,
            &tokenizer,
            &token_ids,
            max_tokens,
            &index,
            &*backend,
            &cached_layers,
            0..num_layers,
            routed,
        ),
        None => generate(
            weights,
            &tokenizer,
            &token_ids,
            max_tokens,
            &index,
            &*backend,
            &cached_layers,
            0..num_layers,
        ),
    };

    // Pre-warm: one generate call to allocate the KV cache and populate the
    // Metal buffer caches. The prefill timer would otherwise include this
    // one-time allocation cost.
    if metal {
        let warm = generate_n(&mut weights, 1);
        if let Some(e) = &warm.error {
            // A pre-warm that failed is a row-shaped lie waiting to happen:
            // say so, rather than let the timed run inherit the state.
            eprintln!("[bench] pre-warm generate failed: {e}");
        }
    }

    // NOTE: `--profile` enables engine-side stage timers
    // (`EngineProfiler`) only — cheap, just per-step `Instant::now()`
    // records. The kernel-side per-stage GPU-timestamp breakdown
    // (`LARQL_PROFILE_SPLIT=1`) is intentionally NOT coupled here.
    // Measured 2026-05-18: setting `LARQL_PROFILE_SPLIT=1`
    // automatically with `--profile` added ~20 ms CPU per token
    // (102 GPU-timestamp queries) and turned the dispatch hot path
    // from 11 ms/step into 30 ms/step — a 2.7× distortion that
    // masked the actual W10 deltas. Users who specifically want the
    // GPU-stage breakdown set `LARQL_PROFILE_SPLIT=1` explicitly.
    let max_tokens = args.warmup + args.tokens;
    let repeats = args.repeat.max(1);
    let mut rows = Vec::with_capacity(repeats);
    for repeat_idx in 0..repeats {
        let t0 = Instant::now();
        let result = generate_n(&mut weights, max_tokens);
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if let Some(e) = &result.error {
            eprintln!("[bench] generate failed: {e}");
        }

        if args.verbose {
            let text: String = result.tokens.iter().map(|(t, _)| t.as_str()).collect();
            eprintln!(
                "[bench] {} prompt ids ({}): {:?}",
                backend_name_for(metal),
                token_ids.len(),
                token_ids
            );
            eprintln!("[bench] {} generated: {text:?}", backend_name_for(metal));
            let (slots, bytes) = index.kquant_ffn_cache_stats();
            eprintln!(
                "{}",
                format_q4k_cache_log(backend_name_for(metal), slots, bytes)
            );
        }

        let n_warm = args.warmup.min(result.decode_ms.len());
        let measured = &result.decode_ms[n_warm..];
        let measured_n = measured.len();
        let (prefill_ms, avg_decode_ms, p50_ms, p99_ms, tok_per_s) = if measured_n == 0 {
            (result.prefill_ms, 0.0, 0.0, 0.0, 0.0)
        } else {
            let (avg, p50, p99) = compute_percentiles(measured);
            (result.prefill_ms, avg, p50, p99, 1000.0 / avg)
        };

        let backend_name = backend_name_for(metal);
        let mut note = format_early_stop_note(measured_n, args.tokens, wall_ms);
        if !metal {
            let cached = larql_inference::vindex::supports_cached_decode(&weights);
            note = append_cpu_fallback_note(note, cached);
        }
        note = append_repeat_note(
            note,
            repeat_idx,
            repeats,
            &generation_fingerprint(&result.tokens),
        );
        let stages = Some(result.stage_timings.avg_per_step(result.decode_ms.len()));

        rows.push(BenchRow {
            backend: backend_name.to_string(),
            prefill_ms,
            avg_decode_ms,
            p50_ms,
            p99_ms,
            tok_per_s,
            stages,
            ffn_rtt_ms: None,
            attn_ms: None,
            wire_bytes_per_tok: None,
            shard_efficiency: None,
            n_steps: measured_n,
            note,
        });
    }

    Ok(rows)
}
