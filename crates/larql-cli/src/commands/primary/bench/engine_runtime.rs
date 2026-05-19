//! I/O-bound runtime for the KV-engine bench. Wraps the engine's
//! `prefill` / `decode_step` API. Excluded from the per-file coverage gate
//! because each call hits real weights / Metal pipeline; pure helpers live
//! in `engine.rs`.

use std::time::Instant;

use larql_kv::EngineKind;

use super::args::BenchArgs;
use super::engine::{
    argmax_token, format_engine_label, format_kv_memory_note, summarize_engine_result,
};
use super::row::BenchRow;

/// Run the CPU KV-engine bench path for a single engine kind.
///
/// Runs prefill on `token_ids` then decodes `args.tokens` steps with greedy
/// argmax. Reports prefill time, avg decode time, and engine memory.
pub(super) fn run_engine(
    weights: &larql_inference::ModelWeights,
    token_ids: &[u32],
    kv_ref_bytes: usize,
    kind: EngineKind,
    backend: Box<dyn larql_inference::EngineBackend>,
    args: &BenchArgs,
) -> Result<BenchRow, Box<dyn std::error::Error>> {
    use larql_inference::ffn::WeightFfn;
    use larql_inference::forward::hidden_to_raw_logits;

    let mut engine = kind.build_with_profiling(backend, args.profile);
    let info = engine.info();
    let label = format_engine_label(
        &info.name,
        &info.backend,
        &info.config,
        /* q4k */ false,
    );

    if args.verbose {
        eprintln!("[bench] {}", info.summary());
    }

    // Default FFN: local dense compute from weights. The four shipped engines
    // currently ignore this parameter, but the trait carries it for engines
    // that want to route FFN (e.g. remote grid).
    let ffn = WeightFfn { weights };

    // Prefill.
    let t_pre = Instant::now();
    let mut hidden = engine
        .prefill(weights, &ffn, token_ids)
        .ok_or("engine prefill failed")?;
    let prefill_ms = t_pre.elapsed().as_secs_f64() * 1000.0;

    // Decode loop: greedy argmax over vocab.
    let max_steps = args.warmup + args.tokens;
    let mut decode_ms_all: Vec<f64> = Vec::with_capacity(max_steps);
    let mut last_token = {
        let logits = hidden_to_raw_logits(weights, &hidden);
        argmax_token(&logits)
    };

    for _ in 0..max_steps {
        let t = Instant::now();
        hidden = engine
            .decode_step(weights, &ffn, last_token)
            .ok_or("engine decode_step failed")?;
        decode_ms_all.push(t.elapsed().as_secs_f64() * 1000.0);
        last_token = argmax_token(&hidden_to_raw_logits(weights, &hidden));
    }

    let summary = summarize_engine_result(&decode_ms_all, args.warmup);
    let note = format_kv_memory_note(engine.memory_bytes(), engine.cold_bytes(), kv_ref_bytes);

    if args.verbose {
        eprintln!(
            "[bench] {} post-decode: {}",
            info.name,
            engine.info().description
        );
    }
    if args.profile {
        if let Some(s) = engine.stage_summary() {
            s.print();
        }
    }

    Ok(BenchRow {
        backend: label,
        prefill_ms,
        avg_decode_ms: summary.avg_decode_ms,
        p50_ms: summary.p50_ms,
        p99_ms: summary.p99_ms,
        tok_per_s: summary.tok_per_s,
        stages: None,
        ffn_rtt_ms: None,
        attn_ms: None,
        wire_bytes_per_tok: None,
        shard_efficiency: None,
        n_steps: summary.n_steps,
        note,
    })
}

/// Q4K engine bench: uses `prefill_quant`/`decode_step_quant` which route through
/// the Metal pipeline for UnlimitedContext and WalkFfn Q4K FFN for MarkovRS.
pub(super) fn run_engine_q4k(
    weights: &mut larql_inference::ModelWeights,
    index: &larql_vindex::VectorIndex,
    token_ids: &[u32],
    kv_ref_bytes: usize,
    kind: EngineKind,
    backend: Box<dyn larql_inference::EngineBackend>,
    args: &BenchArgs,
) -> Result<BenchRow, Box<dyn std::error::Error>> {
    let want_metal_q4k = args.backends.contains("metal");
    let backend_for_q4k: Box<dyn larql_inference::ComputeBackend> = if want_metal_q4k {
        // Use the Metal-aware factory. `larql_inference::default_backend()`
        // (= `larql_compute::default_backend()`) lost its Metal detection
        // after the `larql-compute-metal` extraction and always returns
        // CpuBackend now. Engines that look at
        // `backend.supports_quant(Q4_K)` to decide whether to route
        // through `fused_decode_step` / Metal's fused `decode_token`
        // would get a CpuBackend that advertises Q4_K support and then
        // `backend.decode_token()` returns None (CPU doesn't implement
        // the fused decode kernel), silently falling back to the slow
        // CPU path.
        larql_inference::default_compute_backend()
    } else {
        larql_inference::cpu_backend()
    };
    let mut engine = kind.build_with_profiling(backend, args.profile);
    let info = engine.info();
    let label = format_engine_label(&info.name, &info.backend, &info.config, /* q4k */ true);

    if args.verbose {
        eprintln!("[bench] Q4K engine: {}", info.summary());
    }

    use larql_inference::layer_graph::generate::lm_head_topk;
    let be = backend_for_q4k.as_ref();

    macro_rules! pick_next {
        ($h:expr) => {{
            let h_1d = ndarray::Array1::from_iter($h.iter().copied());
            lm_head_topk(index, weights, &h_1d, 1, be)
                .first()
                .map(|(t, _)| *t)
                .unwrap_or_else(|| {
                    argmax_token(&larql_inference::forward::hidden_to_raw_logits(weights, $h))
                })
        }};
    }

    // Legacy engines dispatch FFN internally from `weights` and ignore
    // this parameter. Migrated engines on the `*_via_executor` path
    // honor it. `NullFfn` works for both without conflicting with the
    // `&mut weights` borrow.
    let ffn = larql_inference::ffn::NullFfn;

    // Optional executor wrap: route through the new `LayerExecutor`
    // surface. For migrated engines this exercises the per-layer walk
    // path + FFN honoring; for unmigrated engines the trait's default
    // impl transparently falls through to the legacy path.
    let executor = if args.via_executor {
        Some(larql_inference::layer_executor::LocalWalkExecutor::new(be))
    } else {
        None
    };

    let t_pre = Instant::now();
    let mut hidden = match executor.as_ref() {
        Some(exec) => engine
            .prefill_quant_via_executor(weights, exec, &ffn, index, token_ids)
            .ok_or("Q4K engine prefill (via executor) failed")?,
        None => engine
            .prefill_quant(weights, &ffn, index, token_ids, be)
            .ok_or("Q4K engine prefill failed")?,
    };
    let prefill_ms = t_pre.elapsed().as_secs_f64() * 1000.0;

    let max_steps = args.warmup + args.tokens;
    let mut decode_ms_all: Vec<f64> = Vec::with_capacity(max_steps);
    let mut last_token = pick_next!(&hidden);

    for _ in 0..max_steps {
        let t = Instant::now();
        hidden = match executor.as_ref() {
            Some(exec) => engine
                .decode_step_quant_via_executor(weights, exec, &ffn, index, last_token)
                .ok_or("Q4K engine decode_step (via executor) failed")?,
            None => engine
                .decode_step_quant(weights, &ffn, index, last_token, be)
                .ok_or("Q4K engine decode_step failed")?,
        };
        decode_ms_all.push(t.elapsed().as_secs_f64() * 1000.0);
        last_token = pick_next!(&hidden);
    }

    let summary = summarize_engine_result(&decode_ms_all, args.warmup);
    let note = format_kv_memory_note(engine.memory_bytes(), engine.cold_bytes(), kv_ref_bytes);

    if args.profile {
        if let Some(s) = engine.stage_summary() {
            s.print();
        }
    }

    Ok(BenchRow {
        backend: label,
        prefill_ms,
        avg_decode_ms: summary.avg_decode_ms,
        p50_ms: summary.p50_ms,
        p99_ms: summary.p99_ms,
        tok_per_s: summary.tok_per_s,
        stages: None,
        ffn_rtt_ms: None,
        attn_ms: None,
        wire_bytes_per_tok: None,
        shard_efficiency: None,
        n_steps: summary.n_steps,
        note,
    })
}
