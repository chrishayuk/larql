//! I/O-bound runtime for the local Metal/CPU bench. Excluded from the
//! per-file coverage gate — every call here hits real vindex mmaps, real
//! model weights, and (when `metal`) the Metal pipeline. Pure helpers live
//! in `local.rs`.

use std::time::Instant;

use super::args::BenchArgs;
use super::local::{
    append_cpu_fallback_note, backend_name_for, format_early_stop_note, format_q4k_cache_log,
};
use super::row::{compute_percentiles, BenchRow};

/// Run the larql generate loop once with the selected backend.
///
/// Warmup runs are discarded; the measured window is `args.tokens` steps
/// AFTER warmup.
pub(super) fn run_larql(
    vindex_path: &std::path::Path,
    args: &BenchArgs,
    metal: bool,
) -> Result<BenchRow, Box<dyn std::error::Error>> {
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

    let backend: Box<dyn larql_compute::ComputeBackend> = if metal {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let b = larql_compute_metal::MetalBackend::new().ok_or(
                "Metal backend unavailable — rebuild with `--features metal` on an M-series Mac",
            )?;
            Box::new(b)
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            return Err("Metal backend requires the `metal` feature on macOS".into());
        }
    } else {
        Box::new(larql_compute::CpuBackend)
    };

    let cached_layers = CachedLayerGraph::from_residuals(Vec::new());

    // Pre-warm: one generate call to allocate the KV cache and populate the
    // Metal buffer caches. The prefill timer would otherwise include this
    // one-time allocation cost.
    if metal {
        let num_layers = weights.num_layers;
        let _ = generate(
            &mut weights,
            &tokenizer,
            &token_ids,
            1,
            &index,
            &*backend,
            &cached_layers,
            0..num_layers,
        );
    }

    if args.profile {
        std::env::set_var("LARQL_PROFILE_SPLIT", "1");
    }
    let max_tokens = args.warmup + args.tokens;
    let num_layers = weights.num_layers;
    let t0 = Instant::now();
    let result = generate(
        &mut weights,
        &tokenizer,
        &token_ids,
        max_tokens,
        &index,
        &*backend,
        &cached_layers,
        0..num_layers,
    );
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    if args.verbose {
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
    let stages = Some(result.stage_timings.avg_per_step(result.decode_ms.len()));

    Ok(BenchRow {
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
    })
}
