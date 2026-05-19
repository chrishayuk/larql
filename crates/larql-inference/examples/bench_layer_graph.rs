//! Production pipeline benchmark: cache L0-12 + mmap walk L13-33 via LayerGraph.
//!
//! Tests:
//!   1. Dense baseline (predict)
//!   2. LayerGraph dense (verify match)
//!   3. Cache+Dense (skip L0-12)
//!   4. Cache+Walk (skip L0-12, mmap FFN L13-33)
//!   5. Walk only (mmap FFN all layers)
//!
//! Usage:
//!   cargo run --release -p larql-inference --example bench_layer_graph -- \
//!     --vindex output/gemma3-4b-v2.vindex

use std::time::Instant;

use larql_inference::vindex::WalkFfn;
use larql_inference::{
    build_adaptive_graph, default_backend, predict, predict_honest, predict_pipeline,
    predict_split_cached, predict_split_pass, predict_with_graph, predict_with_graph_vindex_logits,
    AttentionCache, CachedLayerGraph, InferenceModel, PipelinedLayerGraph, WalkLayerGraph,
    WeightFfn,
};
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

fn bench(
    weights: &larql_inference::ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    graph: &dyn larql_inference::LayerGraph,
    n: usize,
) -> (String, f64, f64) {
    let _ = predict_with_graph(weights, tokenizer, token_ids, 5, graph);
    let t0 = Instant::now();
    for _ in 0..n {
        let _ = predict_with_graph(weights, tokenizer, token_ids, 5, graph);
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let r = predict_with_graph(weights, tokenizer, token_ids, 5, graph);
    let (tok, prob) = r
        .predictions
        .first()
        .map(|(t, p)| (t.clone(), *p))
        .unwrap_or_default();
    (tok, prob, ms)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut vindex_path = std::path::PathBuf::from("output/gemma3-4b-v2.vindex");
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--vindex" {
            i += 1;
            vindex_path = std::path::PathBuf::from(&args[i]);
        }
        i += 1;
    }

    let model = InferenceModel::load("google/gemma-3-4b-it")?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let num_layers = weights.num_layers;
    check_norms(weights);

    let mut cb = SilentLoadCallbacks;
    eprint!("Loading vindex... ");
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    eprint!("down_features... ");
    index.load_down_features(&vindex_path)?;
    eprint!("up_features... ");
    index.load_up_features(&vindex_path)?;
    eprint!("lm_head... ");
    index.load_lm_head(&vindex_path)?;
    if let Ok(()) = index.load_lm_head_kquant(&vindex_path) {
        print!("lm_head_q4 ")
    }
    if let Ok(()) = index.load_attn_q4(&vindex_path) {
        print!("attn_q4 ")
    }
    if let Ok(()) = index.load_attn_kquant(&vindex_path) {
        print!("attn_kquant ")
    }
    if let Ok(()) = index.load_attn_q8(&vindex_path) {
        print!("attn_q8 ")
    }
    if let Ok(()) = index.load_interleaved(&vindex_path) {
        print!("interleaved ")
    }
    if let Ok(()) = index.load_interleaved_q4(&vindex_path) {
        print!("Q4 ")
    }
    if let Ok(()) = index.load_interleaved_kquant(&vindex_path) {
        print!("Q4K_FFN ")
    }
    println!("lm_head (vocab={})\n", index.vocab_size);

    let dense_ffn = WeightFfn { weights };
    let walk_ffn_cpu = WalkFfn::new(weights, &index, 8092);
    let gpu_be = default_backend();
    let walk_ffn_gpu = WalkFfn::new_with_backend(weights, &index, 8092, &*gpu_be);

    let n = 3;

    let prompt = "The capital of France is";
    let encoding = tokenizer.encode(prompt, true).map_err(|e| format!("{e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    println!("=== Production Pipeline Benchmark ===");
    println!("Prompt: \"{prompt}\" ({} tokens)", token_ids.len());
    println!("Backend: {}\n", gpu_be.name());

    // 1. Dense baseline (no LayerGraph)
    let _ = predict(weights, tokenizer, &token_ids, 5);
    let t0 = Instant::now();
    for _ in 0..n {
        let _ = predict(weights, tokenizer, &token_ids, 5);
    }
    let dense_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let dense_r = predict(weights, tokenizer, &token_ids, 5);
    let (dense_tok, dense_prob) = dense_r
        .predictions
        .first()
        .map(|(t, p)| (t.clone(), *p))
        .unwrap_or_default();

    // 2. Cache+Walk (CPU) — FFN through CPU BLAS
    let walk_cpu_graph = WalkLayerGraph {
        ffn: &walk_ffn_cpu,
        backend: None,
    };
    let cached_layers: Vec<usize> = (0..=12).collect();
    let cache = CachedLayerGraph::build(weights, &token_ids, &cached_layers, &dense_ffn);
    let cw_cpu = build_adaptive_graph(&cache, &walk_cpu_graph, num_layers, &(0..=12));
    let (cw_cpu_tok, _, cw_cpu_ms) = bench(weights, tokenizer, &token_ids, &cw_cpu, n);

    // 3. Cache+Walk (Metal Q4 FFN, CPU attention)
    let walk_gpu_graph = WalkLayerGraph {
        ffn: &walk_ffn_gpu,
        backend: None,
    };
    let cw_gpu = build_adaptive_graph(&cache, &walk_gpu_graph, num_layers, &(0..=12));
    let (cw_gpu_tok, _, cw_gpu_ms) = bench(weights, tokenizer, &token_ids, &cw_gpu, n);

    // 4. Full pipeline (CPU): Cache+Walk(CPU)+Vindex logits
    let _ = predict_with_graph_vindex_logits(weights, tokenizer, &token_ids, 5, &cw_cpu, &index);
    let t0 = Instant::now();
    for _ in 0..n {
        let _ =
            predict_with_graph_vindex_logits(weights, tokenizer, &token_ids, 5, &cw_cpu, &index);
    }
    let full_cpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let full_cpu_r =
        predict_with_graph_vindex_logits(weights, tokenizer, &token_ids, 5, &cw_cpu, &index);
    let (full_cpu_tok, _) = full_cpu_r
        .predictions
        .first()
        .map(|(t, p)| (t.clone(), *p))
        .unwrap_or_default();

    // 5. Full pipeline (Metal Q4 FFN, CPU attention, vindex logits)
    let _ = predict_with_graph_vindex_logits(weights, tokenizer, &token_ids, 5, &cw_gpu, &index);
    let t0 = Instant::now();
    for _ in 0..n {
        let _ =
            predict_with_graph_vindex_logits(weights, tokenizer, &token_ids, 5, &cw_gpu, &index);
    }
    let full_gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let full_gpu_r =
        predict_with_graph_vindex_logits(weights, tokenizer, &token_ids, 5, &cw_gpu, &index);
    let (full_gpu_tok, full_gpu_prob) = full_gpu_r
        .predictions
        .first()
        .map(|(t, p)| (t.clone(), *p))
        .unwrap_or_default();

    println!(
        "  Dense (baseline):    {dense_tok:>10} ({:.2}%)  {dense_ms:>6.0}ms  ({:.1} tok/s)",
        dense_prob * 100.0,
        1000.0 / dense_ms
    );
    println!(
        "  Cache+Walk (CPU):    {cw_cpu_tok:>10}           {cw_cpu_ms:>6.0}ms  ({:.1} tok/s)",
        1000.0 / cw_cpu_ms
    );
    println!(
        "  Cache+Walk (GPU):    {cw_gpu_tok:>10}           {cw_gpu_ms:>6.0}ms  ({:.1} tok/s)",
        1000.0 / cw_gpu_ms
    );
    println!(
        "  Full pipe (CPU):     {full_cpu_tok:>10}           {full_cpu_ms:>6.0}ms  ({:.1} tok/s)",
        1000.0 / full_cpu_ms
    );
    println!(
        "  Full pipe (GPU):     {full_gpu_tok:>10} ({:.2}%)  {full_gpu_ms:>6.0}ms  ({:.1} tok/s)",
        full_gpu_prob * 100.0,
        1000.0 / full_gpu_ms
    );

    // 6. Pipelined: Cache + Q4 Metal FFN (per-layer dispatch via PipelinedLayerGraph)
    let pipelined = PipelinedLayerGraph {
        index: &index,
        backend: &*gpu_be,
        layer_range: 13..num_layers,
    };
    let pipelined_graph = build_adaptive_graph(&cache, &pipelined, num_layers, &(0..=12));
    let _ = predict_pipeline(
        weights,
        tokenizer,
        &token_ids,
        5,
        &pipelined_graph,
        Some(&index),
    );
    let t0 = Instant::now();
    for _ in 0..n {
        let _ = predict_pipeline(
            weights,
            tokenizer,
            &token_ids,
            5,
            &pipelined_graph,
            Some(&index),
        );
    }
    let pipelined_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let pipelined_r = predict_pipeline(
        weights,
        tokenizer,
        &token_ids,
        5,
        &pipelined_graph,
        Some(&index),
    );
    let (pipelined_tok, pipelined_prob) = pipelined_r
        .predictions
        .first()
        .map(|(t, p)| (t.clone(), *p))
        .unwrap_or_default();

    println!(
        "  Pipelined (Q4+KNN): {pipelined_tok:>10} ({:.2}%)  {pipelined_ms:>6.0}ms  ({:.1} tok/s)",
        pipelined_prob * 100.0,
        1000.0 / pipelined_ms
    );
    println!();
    // 7. Split-pass: attention CPU + batched Metal Q4 FFN + vindex logits
    let _ = predict_split_pass(
        weights,
        tokenizer,
        &token_ids,
        5,
        &index,
        &*gpu_be,
        &cache,
        13..num_layers,
    );
    let t0 = Instant::now();
    for _ in 0..n {
        let _ = predict_split_pass(
            weights,
            tokenizer,
            &token_ids,
            5,
            &index,
            &*gpu_be,
            &cache,
            13..num_layers,
        );
    }
    let split_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let split_r = predict_split_pass(
        weights,
        tokenizer,
        &token_ids,
        5,
        &index,
        &*gpu_be,
        &cache,
        13..num_layers,
    );
    let (split_tok, split_prob) = split_r
        .predictions
        .first()
        .map(|(t, p)| (t.clone(), *p))
        .unwrap_or_default();

    println!(
        "  Split pass (Q4+KNN): {split_tok:>10} ({:.2}%)  {split_ms:>6.0}ms  ({:.1} tok/s)",
        split_prob * 100.0,
        1000.0 / split_ms
    );
    println!();
    // 8. Split cached: exact attention cache + batched Metal Q4 FFN + vindex logits
    // Build attention cache from one exact run (one-time cost)
    let t0 = Instant::now();
    let attn_cache = AttentionCache::build(weights, &token_ids, &cache, &dense_ffn, 13..num_layers);
    let cache_build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let _ = predict_split_cached(
        weights,
        tokenizer,
        5,
        &index,
        &*gpu_be,
        &attn_cache,
        13..num_layers,
    );
    let t0 = Instant::now();
    for _ in 0..n {
        let _ = predict_split_cached(
            weights,
            tokenizer,
            5,
            &index,
            &*gpu_be,
            &attn_cache,
            13..num_layers,
        );
    }
    let cached_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let cached_r = predict_split_cached(
        weights,
        tokenizer,
        5,
        &index,
        &*gpu_be,
        &attn_cache,
        13..num_layers,
    );
    let (cached_tok, cached_prob) = cached_r
        .predictions
        .first()
        .map(|(t, p)| (t.clone(), *p))
        .unwrap_or_default();

    println!("  Split cached (Q4):   {cached_tok:>10} ({:.2}%)  {cached_ms:>6.0}ms  ({:.1} tok/s)  [cache build: {cache_build_ms:.0}ms]", cached_prob * 100.0, 1000.0/cached_ms);
    println!();
    // 9. Honest: cache L0-12, compute L13-33 (interleaved attn+FFN), GPU Q4 logits
    let _ = predict_honest(
        weights,
        tokenizer,
        &token_ids,
        5,
        &index,
        &*gpu_be,
        &cache,
        13..num_layers,
    );
    let t0 = Instant::now();
    for _ in 0..n {
        let _ = predict_honest(
            weights,
            tokenizer,
            &token_ids,
            5,
            &index,
            &*gpu_be,
            &cache,
            13..num_layers,
        );
    }
    let honest_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let honest_r = predict_honest(
        weights,
        tokenizer,
        &token_ids,
        5,
        &index,
        &*gpu_be,
        &cache,
        13..num_layers,
    );
    let (honest_tok, honest_prob) = honest_r
        .predictions
        .first()
        .map(|(t, p)| (t.clone(), *p))
        .unwrap_or_default();

    println!();
    println!("  ═══ HONEST PRODUCTION PATH ═══");
    println!(
        "  Honest (Q4+cache13):  {honest_tok:>10} ({:.2}%)  {honest_ms:>6.0}ms  ({:.1} tok/s)",
        honest_prob * 100.0,
        1000.0 / honest_ms
    );
    println!();
    println!(
        "  Honest vs Dense:     {:.1}x ({:.0}ms saved)",
        dense_ms / honest_ms,
        dense_ms - honest_ms
    );
    println!(
        "  Honest vs Ollama:    {:.1}x (Ollama ~10ms = 98 tok/s)",
        10.0 / honest_ms
    );

    // Prefill → Decode with KV cache
    {
        use larql_inference::layer_graph::predict::{finalize_logits, prefill_with_kv};

        // Step 1: Prefill (populates KV cache on Metal)
        gpu_be.reset_kv_cache();
        let t0 = std::time::Instant::now();
        let h_prefill = prefill_with_kv(weights, &token_ids, &index, &*gpu_be, 0..num_layers);
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Step 2: Use prefill output for logits (the hidden state is already computed)
        // This measures the logits-only path (norm + vindex KNN) after prefill.
        let _empty_cache = CachedLayerGraph::from_residuals(vec![]);
        // Measure logits from prefill output (norm + vindex KNN only — 0 layers)
        let norm_offset = weights.arch.norm_weight_offset();
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            let _ = finalize_logits(
                weights,
                tokenizer,
                &h_prefill,
                5,
                &index,
                &*gpu_be,
                norm_offset,
            );
        }
        let logits_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let decode_r = finalize_logits(
            weights,
            tokenizer,
            &h_prefill,
            5,
            &index,
            &*gpu_be,
            norm_offset,
        );
        let (decode_tok, decode_prob) = decode_r
            .predictions
            .first()
            .map(|(t, p)| (t.clone(), *p))
            .unwrap_or_default();

        println!("\n  ═══ PREFILL → DECODE (KV cache) ═══");
        println!(
            "  Prefill ({} tokens):                {prefill_ms:>6.0}ms",
            token_ids.len()
        );
        println!(
            "  Logits (from prefill): {decode_tok:>10} ({:.2}%)  {logits_ms:>6.1}ms",
            decode_prob * 100.0
        );
        println!("  Ollama:              prefill ~15ms, decode 10ms (99 tok/s)");
    }

    println!("=== Done ===");
    Ok(())
}

// Diagnostic: check norm weights are loaded
fn check_norms(weights: &larql_inference::ModelWeights) {
    let arch = &*weights.arch;
    for layer in [0, 13, 33] {
        let key = arch.input_layernorm_key(layer);
        let has = weights.vectors.contains_key(&key);
        let len = weights.vectors.get(&key).map(|v| v.len()).unwrap_or(0);
        eprintln!("  norm L{layer}: {key} → has={has} len={len}");
    }
}
