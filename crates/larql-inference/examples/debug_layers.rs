//! Debug: test decode_token output at 1, 2, 5, 10, 34 layers

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = larql_inference::InferenceModel::load("google/gemma-3-4b-it")?;
    let weights = model.weights();
    let vd = std::path::PathBuf::from("output/gemma3-4b-v2.vindex");
    let mut index =
        larql_vindex::VectorIndex::load_vindex(&vd, &mut larql_vindex::SilentLoadCallbacks)?;
    let _ = index.load_attn_kquant(&vd);
    let _ = index.load_interleaved_kquant(&vd);
    let backend = larql_inference::default_backend();
    let gate_index: &dyn larql_vindex::GateIndex = &index;

    let q4_ffn_mmap = gate_index.interleaved_kquant_mmap_ref().unwrap();
    let intermediate = gate_index.num_features(0);
    let hidden = weights.hidden_size;
    let q4_ffn_per_matrix = (intermediate * hidden).div_ceil(256) * 148;
    let ffn_format = larql_compute::QuantFormat::Q4_K;

    let encoding = model
        .tokenizer()
        .encode("The capital of France is", true)
        .unwrap();
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    let h = larql_inference::forward::embed_tokens_pub(weights, &ids);
    let x: Vec<f32> = h.row(0).to_vec();

    // `q_dim` / `kv_dim` / `rope` used to be passed to `decode_token`;
    // the post-refactor 4-arg API doesn't need them.

    println!("=== Layer-by-Layer Decode Debug ===\n");
    println!("  Layers  nonzero  max_abs   norm");
    println!("  ─────── ──────── ──────── ──────");

    for n_layers in [1, 2, 5, 10, 20, 34] {
        let layers = larql_inference::layer_graph::pipeline_layer::build_pipeline_layers(
            weights,
            &index,
            0..n_layers,
            q4_ffn_mmap,
            q4_ffn_per_matrix,
            ffn_format,
        );

        backend.reset_kv_cache();
        let result = backend.decode_token(&layers, &x, hidden, intermediate);

        if let Some(ref h) = result {
            let nonzero = h.iter().filter(|v| v.abs() > 1e-10).count();
            let max = h.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            let norm: f32 = h.iter().map(|v| v * v).sum::<f32>().sqrt();
            println!(
                "  {:>5}   {:>6}   {:>7.4}   {:>7.2}",
                n_layers, nonzero, max, norm
            );
        } else {
            println!("  {:>5}   None", n_layers);
        }
    }

    // Also try the CPU forward pass for comparison
    println!("\n  CPU dense forward (34 layers):");
    let dense_ffn = larql_inference::WeightFfn { weights };
    let mut h_cpu = h.clone();
    for layer in 0..weights.num_layers {
        let (h_pa, _, _) = larql_inference::attention::run_attention_block_gpu(
            weights, &h_cpu, layer, false, None,
        )
        .unwrap();
        let (h_out, _) =
            larql_inference::forward::run_ffn(weights, &h_pa, layer, &dense_ffn, false);
        h_cpu = h_out;
    }
    let cpu_row = h_cpu.row(0);
    let nonzero = cpu_row.iter().filter(|v| v.abs() > 1e-10).count();
    let max = cpu_row.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let norm: f32 = cpu_row.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!(
        "  {:>5}   {:>6}   {:>7.4}   {:>7.2}",
        34, nonzero, max, norm
    );

    println!("\n=== Done ===");
    Ok(())
}
