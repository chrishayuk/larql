//! Benchmark: hybrid pipeline (GPU attention + vindex walk FFN) vs full GPU decode.
//!
//! Requires: model weights + vindex with down_features.bin + attn_weights_q4k.bin
//!
//! Run:
//!   cargo run --release --features metal -p larql-inference --example bench_hybrid -- \
//!     --model google/gemma-3-4b-it --vindex output/gemma3-4b-v2.vindex

use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("google/gemma-3-4b-it");
    let vindex_path = args
        .iter()
        .position(|a| a == "--vindex")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("output/gemma3-4b-v2.vindex");

    println!("=== Hybrid Pipeline Benchmark ===\n");
    println!("Model:  {model_path}");
    println!("Vindex: {vindex_path}\n");

    // Load model
    eprintln!("Loading model...");
    let model = larql_inference::InferenceModel::load(model_path).expect("Failed to load model");
    let weights = model.weights();
    eprintln!(
        "  {} layers, hidden={}",
        weights.num_layers, weights.hidden_size
    );

    // Load vindex + all walk/attn data
    eprintln!("Loading vindex...");
    let vindex_dir = std::path::PathBuf::from(vindex_path);
    let mut index =
        larql_vindex::VectorIndex::load_vindex(&vindex_dir, &mut larql_vindex::SilentLoadCallbacks)
            .expect("Failed to load vindex");

    // Load optional data files
    let _ = index.load_down_features(&vindex_dir);
    let _ = index.load_attn_kquant(&vindex_dir);
    let _ = index.load_interleaved_kquant(&vindex_dir);
    let _ = index.load_interleaved_q4(&vindex_dir);
    let _ = index.load_lm_head(&vindex_dir);

    let gate_index: &dyn larql_vindex::GateIndex = &index;
    eprintln!("  down_features: {}", gate_index.has_down_features());
    eprintln!("  attn Q4K: {}", index.attn_kquant_layer_data(0).is_some());
    eprintln!("  interleaved Q4K: {}", gate_index.has_interleaved_kquant());
    eprintln!(
        "  interleaved Q4: {}",
        gate_index.interleaved_q4_mmap_ref().is_some()
    );
    eprintln!("  lm_head: {}", index.has_lm_head());

    // Backend
    let backend = larql_inference::default_backend();
    eprintln!("  backend: {} ({})", backend.name(), backend.device_info());

    // Cached layers (none cached — all layers computed)
    let cached = larql_inference::CachedLayerGraph::from_residuals(vec![]);
    let layer_range = 0..weights.num_layers;

    // Tokenize
    let prompt = "The capital of France is";
    let encoding = model.tokenizer().encode(prompt, true).unwrap();
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("  prompt: \"{prompt}\" ({} tokens)\n", token_ids.len());

    let iters = 3;

    // ── 1. Hybrid: GPU attention + walk FFN ──
    println!("--- Hybrid (GPU attention + walk FFN) ---\n");
    {
        // Warm up
        let _ = larql_inference::predict_hybrid(
            weights,
            model.tokenizer(),
            &token_ids,
            5,
            &index,
            &*backend,
            &cached,
            layer_range.clone(),
        );

        let t = Instant::now();
        let mut result = None;
        for _ in 0..iters {
            backend.reset_kv_cache();
            result = Some(larql_inference::predict_hybrid(
                weights,
                model.tokenizer(),
                &token_ids,
                5,
                &index,
                &*backend,
                &cached,
                layer_range.clone(),
            ));
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let r = result.unwrap();
        let (tok, prob) = r
            .predictions
            .first()
            .map(|(t, p)| (t.as_str(), *p))
            .unwrap_or(("?", 0.0));
        println!("  Time:   {ms:.1}ms");
        println!("  tok/s:  {:.0}", 1000.0 / ms);
        println!("  Top-1:  {tok} ({:.1}%)\n", prob * 100.0);
    }

    // ── 2. predict_honest: full GPU decode_token ──
    println!("--- predict_honest (full GPU decode) ---\n");
    {
        let _ = larql_inference::predict_honest(
            weights,
            model.tokenizer(),
            &token_ids,
            5,
            &index,
            &*backend,
            &cached,
            layer_range.clone(),
        );

        let t = Instant::now();
        let mut result = None;
        for _ in 0..iters {
            backend.reset_kv_cache();
            result = Some(larql_inference::predict_honest(
                weights,
                model.tokenizer(),
                &token_ids,
                5,
                &index,
                &*backend,
                &cached,
                layer_range.clone(),
            ));
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let r = result.unwrap();
        let (tok, prob) = r
            .predictions
            .first()
            .map(|(t, p)| (t.as_str(), *p))
            .unwrap_or(("?", 0.0));
        println!("  Time:   {ms:.1}ms");
        println!("  tok/s:  {:.0}", 1000.0 / ms);
        println!("  Top-1:  {tok} ({:.1}%)\n", prob * 100.0);
    }

    // ── 3. CPU walk (predict_with_graph + WalkFfn) ──
    println!("--- CPU walk (BLAS attention + walk FFN) ---\n");
    {
        let walk_ffn = larql_inference::vindex::WalkFfn::new(weights, &index, 8192);
        let walk_graph = larql_inference::WalkLayerGraph {
            ffn: &walk_ffn,
            backend: None,
        };

        let t = Instant::now();
        let result = larql_inference::predict_with_graph(
            weights,
            model.tokenizer(),
            &token_ids,
            5,
            &walk_graph,
        );
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let (tok, prob) = result
            .predictions
            .first()
            .map(|(t, p)| (t.as_str(), *p))
            .unwrap_or(("?", 0.0));
        println!("  Time:   {ms:.1}ms");
        println!("  tok/s:  {:.0}", 1000.0 / ms);
        println!("  Top-1:  {tok} ({:.1}%)\n", prob * 100.0);
    }

    // ── Summary ──
    println!("--- Summary ---\n");
    println!("  Pipeline               ms/tok    tok/s");
    println!("  ─────────────────────── ─────── ───────");
    println!("  Hybrid (GPU attn+walk)  measure  above");
    println!("  predict_honest (GPU)    measure  above");
    println!("  CPU walk (BLAS+walk)    measure  above");
    println!("  Ollama reference        ~10ms    ~100");

    println!("\n=== Done ===");
}
