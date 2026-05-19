//! Bottleneck analysis: WalkFfn step-by-step timing per layer.
//!
//! Measures each phase of the walk FFN independently:
//!   1. Gate KNN (find top-K features)
//!   2. Up dot products (per-feature dot against input)
//!   3. GEGLU activation
//!   4. Down accumulation (scaled_add per feature)
//!   5. Norm (pre-FFN)
//!
//! Run:
//!   cargo run --release -p larql-inference --example profile_walk_ffn -- \
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

    println!("=== WalkFfn Bottleneck Analysis ===\n");

    // Load
    eprintln!("Loading model + vindex...");
    let model = larql_inference::InferenceModel::load(model_path).unwrap();
    let weights = model.weights();
    let vindex_dir = std::path::PathBuf::from(vindex_path);
    let mut index =
        larql_vindex::VectorIndex::load_vindex(&vindex_dir, &mut larql_vindex::SilentLoadCallbacks)
            .unwrap();
    let _ = index.load_down_features(&vindex_dir);
    let _ = index.load_up_features(&vindex_dir);
    let _ = index.load_gate_vectors_q4(&vindex_dir);

    let gate_index: &dyn larql_vindex::GateIndex = &index;
    let hidden = weights.hidden_size;
    let intermediate = gate_index.num_features(14);
    let arch = &*weights.arch;
    let use_gelu = matches!(
        arch.activation(),
        larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
    );
    let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;

    println!("  hidden={hidden}, intermediate={intermediate}");
    println!("  activation={:?}, gated={is_gated}", arch.activation());
    println!("  down_features: {}", gate_index.has_down_features());
    println!(
        "  up_features: {}",
        gate_index.up_layer_matrix(14).is_some()
    );

    // Get a realistic hidden state by running forward to L14
    eprintln!("Running forward to L14 for realistic hidden state...");
    let prompt = "The capital of France is";
    let encoding = model.tokenizer().encode(prompt, true).unwrap();
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let mut h = larql_inference::forward::embed_tokens_pub(weights, &token_ids);
    for layer in 0..14 {
        let (h_post_attn, _, _) =
            larql_inference::attention::run_attention_block_gpu(weights, &h, layer, false, None)
                .unwrap();
        let dense_ffn = larql_inference::WeightFfn { weights };
        let (h_out, _) =
            larql_inference::forward::run_ffn(weights, &h_post_attn, layer, &dense_ffn, false);
        h = h_out;
    }
    // Use last position
    let x = h
        .slice(ndarray::s![h.shape()[0] - 1..h.shape()[0], ..])
        .to_owned();
    eprintln!("  hidden state shape: {:?}\n", x.shape());

    let norm_offset = arch.norm_weight_offset();
    let x_normed = larql_inference::forward::apply_norm(
        weights,
        &x,
        &arch.post_attention_layernorm_key(14),
        norm_offset,
    );

    let test_layers = [14, 18, 22, 26, 30];
    let top_k_values = [50, 200, 500, 2000, 8192];
    eprintln!("  gate_q4: {}", index.gate_q4_data(14).is_some());

    let backend = larql_inference::default_backend();
    eprintln!(
        "  backend: {} (supports Q4_K = {})\n",
        backend.name(),
        backend.supports_quant(::larql_compute::QuantFormat::Q4_K)
    );

    let iters = 20;

    // ── Q4 vs f32 gate KNN comparison ──
    println!("=== Gate KNN: f32 vs Q4 (L14, K=200) ===\n");
    {
        let layer = 14;
        let k = 200;
        let x_row = x_normed.row(0).to_owned();

        // f32 BLAS
        let t = Instant::now();
        for _ in 0..iters {
            let _ = gate_index.gate_knn(layer, &x_row, k);
        }
        let f32_us = t.elapsed().as_micros() as f64 / iters as f64;

        // Q4 via backend
        let t = Instant::now();
        let mut q4_hits = None;
        for _ in 0..iters {
            q4_hits = gate_index.gate_knn_q4(layer, &x_row, k, &*backend);
        }
        let q4_us = t.elapsed().as_micros() as f64 / iters as f64;

        println!("  f32 BLAS gate KNN:  {:>7.0}µs", f32_us);
        if q4_hits.is_some() {
            println!(
                "  Q4 gate KNN:        {:>7.0}µs  ({:.1}x faster)",
                q4_us,
                f32_us / q4_us
            );
        } else {
            println!("  Q4 gate KNN:        not available (no Q4 gate data or backend)");
        }
        println!();
    }

    // ── Per-step breakdown at K=200, L14 ──
    println!("=== Step-by-step breakdown (L14, K=200, f32 gate) ===\n");
    {
        let layer = 14;
        let k = 200;
        let x_row = x_normed.row(0).to_owned();

        // 1. Gate KNN
        let t = Instant::now();
        let mut hits = vec![];
        for _ in 0..iters {
            hits = gate_index.gate_knn(layer, &x_row, k);
        }
        let gate_us = t.elapsed().as_micros() as f64 / iters as f64;

        // 2. Up dot products
        let up_view = gate_index.up_layer_matrix(layer);
        let t = Instant::now();
        let mut up_scores = vec![0.0f32; k];
        for _ in 0..iters {
            if let Some(ref uv) = up_view {
                for (i, &(feat, _)) in hits.iter().enumerate().take(k) {
                    up_scores[i] = uv.row(feat).dot(&x_row);
                }
            }
        }
        let up_us = t.elapsed().as_micros() as f64 / iters as f64;

        // 3. GEGLU activation
        let t = Instant::now();
        let mut activations = vec![0.0f32; k];
        for _ in 0..iters {
            for (i, &(_, gate_score)) in hits.iter().enumerate().take(k) {
                let activated_gate = if use_gelu {
                    larql_inference::ffn::gelu_tanh(gate_score)
                } else {
                    gate_score * larql_inference::ffn::sigmoid(gate_score)
                };
                activations[i] = if is_gated {
                    activated_gate * up_scores[i]
                } else {
                    activated_gate
                };
            }
        }
        let act_us = t.elapsed().as_micros() as f64 / iters as f64;

        // 4. Down accumulation
        let down_view = gate_index.down_layer_matrix(layer);
        let t = Instant::now();
        for _ in 0..iters {
            let mut out = ndarray::Array1::<f32>::zeros(hidden);
            if let Some(ref dv) = down_view {
                for (i, &(feat, _)) in hits.iter().enumerate().take(k) {
                    if activations[i].abs() > 1e-10 {
                        out.scaled_add(activations[i], &dv.row(feat));
                    }
                }
            }
        }
        let down_us = t.elapsed().as_micros() as f64 / iters as f64;

        // 5. Pre-FFN norm
        let t = Instant::now();
        for _ in 0..iters {
            let _ = larql_inference::forward::apply_norm(
                weights,
                &x,
                &arch.post_attention_layernorm_key(layer),
                norm_offset,
            );
        }
        let norm_us = t.elapsed().as_micros() as f64 / iters as f64;

        let total = gate_us + up_us + act_us + down_us + norm_us;
        println!("  Step              µs       %");
        println!("  ──────────────── ────── ─────");
        println!(
            "  Gate KNN (K={k})  {:>6.0}  {:>4.1}%",
            gate_us,
            gate_us / total * 100.0
        );
        println!(
            "  Up dots ({k} dots){:>7.0}  {:>4.1}%",
            up_us,
            up_us / total * 100.0
        );
        println!(
            "  GEGLU activation {:>6.0}  {:>4.1}%",
            act_us,
            act_us / total * 100.0
        );
        println!(
            "  Down accum ({k}×h){:>6.0}  {:>4.1}%",
            down_us,
            down_us / total * 100.0
        );
        println!(
            "  Pre-FFN norm     {:>6.0}  {:>4.1}%",
            norm_us,
            norm_us / total * 100.0
        );
        println!("  ──────────────── ──────");
        println!("  Total            {:>6.0}µs", total);
        println!(
            "  Non-zero feats:  {}/{k}",
            activations.iter().filter(|a| a.abs() > 1e-10).count()
        );
    }

    // ── K scaling ──
    println!("\n=== K scaling (L14) ===\n");
    println!("  K       Gate     Up      Act     Down    Total    vs dense");
    println!("  ───── ─────── ─────── ─────── ─────── ──────── ──────────");
    for &k in &top_k_values {
        let layer = 14;
        let x_row = x_normed.row(0).to_owned();

        let t = Instant::now();
        for _ in 0..iters {
            let hits = gate_index.gate_knn(layer, &x_row, k);
            let _ = hits;
        }
        let gate_us = t.elapsed().as_micros() as f64 / iters as f64;

        let hits = gate_index.gate_knn(layer, &x_row, k);
        let up_view = gate_index.up_layer_matrix(layer);
        let down_view = gate_index.down_layer_matrix(layer);

        let t = Instant::now();
        for _ in 0..iters {
            if let Some(ref uv) = up_view {
                for &(feat, _) in hits.iter().take(k) {
                    let _ = uv.row(feat).dot(&x_row);
                }
            }
        }
        let up_us = t.elapsed().as_micros() as f64 / iters as f64;

        let t = Instant::now();
        for _ in 0..iters {
            let mut out = ndarray::Array1::<f32>::zeros(hidden);
            if let Some(ref dv) = down_view {
                for &(feat, gate_score) in hits.iter().take(k) {
                    let act = if use_gelu {
                        larql_inference::ffn::gelu_tanh(gate_score)
                    } else {
                        gate_score * larql_inference::ffn::sigmoid(gate_score)
                    };
                    if act.abs() > 1e-10 {
                        out.scaled_add(act, &dv.row(feat));
                    }
                }
            }
        }
        let down_us = t.elapsed().as_micros() as f64 / iters as f64;

        let total = gate_us + up_us + down_us;
        // Dense FFN: gate+up+down = ~9ms (from bench_components)
        println!(
            "  {k:>5}  {gate_us:>6.0}  {up_us:>6.0}  {:>6}  {down_us:>6.0}  {total:>7.0}   {:.2}x",
            "-",
            total / 9000.0
        );
    }

    // ── Layer variation ──
    println!("\n=== Layer variation (K=200) ===\n");
    println!("  Layer   Gate     Up      Down    Total");
    println!("  ───── ─────── ─────── ─────── ───────");
    for &layer in &test_layers {
        let x_row = x_normed.row(0).to_owned();
        let k = 200;

        let t = Instant::now();
        for _ in 0..iters {
            let _ = gate_index.gate_knn(layer, &x_row, k);
        }
        let gate_us = t.elapsed().as_micros() as f64 / iters as f64;

        let hits = gate_index.gate_knn(layer, &x_row, k);

        let t = Instant::now();
        if let Some(uv) = gate_index.up_layer_matrix(layer) {
            for _ in 0..iters {
                for &(feat, _) in hits.iter().take(k) {
                    let _ = uv.row(feat).dot(&x_row);
                }
            }
        }
        let up_us = t.elapsed().as_micros() as f64 / iters as f64;

        let t = Instant::now();
        if let Some(dv) = gate_index.down_layer_matrix(layer) {
            for _ in 0..iters {
                let mut out = ndarray::Array1::<f32>::zeros(hidden);
                for &(feat, gs) in hits.iter().take(k) {
                    let act = if use_gelu {
                        larql_inference::ffn::gelu_tanh(gs)
                    } else {
                        gs * larql_inference::ffn::sigmoid(gs)
                    };
                    if act.abs() > 1e-10 {
                        out.scaled_add(act, &dv.row(feat));
                    }
                }
            }
        }
        let down_us = t.elapsed().as_micros() as f64 / iters as f64;

        println!(
            "  L{layer:>2}   {gate_us:>6.0}  {up_us:>6.0}  {down_us:>6.0}  {:>6.0}",
            gate_us + up_us + down_us
        );
    }

    println!("\n=== Done ===");
}
