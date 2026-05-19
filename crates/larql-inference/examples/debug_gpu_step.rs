//! Debug: step-by-step GPU pipeline output

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = larql_inference::InferenceModel::load("google/gemma-3-4b-it")?;
    let weights = model.weights();
    let vd = std::path::PathBuf::from("output/gemma3-4b-v2.vindex");
    let mut index =
        larql_vindex::VectorIndex::load_vindex(&vd, &mut larql_vindex::SilentLoadCallbacks)?;
    let _ = index.load_attn_kquant(&vd);
    let _ = index.load_interleaved_kquant(&vd);

    let backend = larql_compute::default_backend();
    let gate_index: &dyn larql_vindex::GateIndex = &index;
    let q4_ffn_mmap = gate_index.interleaved_kquant_mmap_ref().unwrap();
    let intermediate = gate_index.num_features(0);
    let hidden = weights.hidden_size;
    let q4_ffn_per_matrix = (intermediate * hidden).div_ceil(256) * 148;
    let ffn_format = larql_compute::QuantFormat::Q4_K;

    println!("=== GPU Pipeline Step Debug ===\n");

    // Build layer 0
    let layers = larql_inference::layer_graph::pipeline_layer::build_pipeline_layers(
        weights,
        &index,
        0..1,
        q4_ffn_mmap,
        q4_ffn_per_matrix,
        ffn_format,
    );
    let layer = &layers[0];
    println!(
        "Layer 0 formats: wq={:?}, wk={:?}, wv={:?}, wo={:?}",
        layer.wq.format, layer.wk.format, layer.wv.format, layer.wo.format
    );
    println!(
        "Layer 0 dims: hd={}, nq={}, nkv={}",
        layer.head_dim, layer.num_q_heads, layer.num_kv_heads
    );

    // Embedding
    let encoding = model.tokenizer().encode("Hello", true).unwrap();
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    let h = larql_inference::forward::embed_tokens_pub(weights, &ids);
    let x: Vec<f32> = h.row(0).to_vec();
    let x_nonzero = x.iter().filter(|v| v.abs() > 1e-10).count();
    let x_max = x.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    println!(
        "\nInput x: len={}, nonzero={}, max={:.4}",
        x.len(),
        x_nonzero,
        x_max
    );

    // Test standalone q4k_matvec with Q proj weights
    println!("\n=== Standalone Q4K matvec tests ===");
    let q_dim = weights.num_q_heads * weights.head_dim;
    let kv_dim = weights.num_kv_heads * weights.head_dim;

    let q_result = backend.q4k_matvec(layer.wq.data, &x, q_dim, hidden);
    if let Some(ref r) = q_result {
        println!(
            "  Q proj: nonzero={}/{}, max={:.4}",
            r.iter().filter(|v: &&f32| v.abs() > 1e-10).count(),
            r.len(),
            r.iter().cloned().fold(0.0f32, f32::max)
        );
    } else {
        println!("  Q proj: None");
    }

    let k_result = backend.q4k_matvec(layer.wk.data, &x, kv_dim, hidden);
    if let Some(ref r) = k_result {
        println!(
            "  K proj: nonzero={}/{}, max={:.4}",
            r.iter().filter(|v: &&f32| v.abs() > 1e-10).count(),
            r.len(),
            r.iter().cloned().fold(0.0f32, f32::max)
        );
    } else {
        println!("  K proj: None");
    }

    // V is Q6_K — use q6k_matvec
    let v_result = backend.q6k_matvec(layer.wv.data, &x, kv_dim, hidden);
    if let Some(ref r) = v_result {
        println!(
            "  V proj (Q6K): nonzero={}/{}, max={:.4}",
            r.iter().filter(|v: &&f32| v.abs() > 1e-10).count(),
            r.len(),
            r.iter().cloned().fold(0.0f32, f32::max)
        );
    } else {
        println!("  V proj: None");
    }

    // Now test decode_token
    println!("\n=== decode_token test ===");
    backend.reset_kv_cache();
    let result = backend.decode_token(&layers, &x, hidden, intermediate);
    if let Some(ref r) = result {
        let nz = r.iter().filter(|v: &&f32| v.abs() > 1e-10).count();
        let max = r.iter().cloned().fold(0.0f32, f32::max);
        println!("  decode_token: nonzero={}/{}, max={:.4}", nz, r.len(), max);
    } else {
        println!("  decode_token: None");
    }

    // Compare: CPU norm → CPU Q proj
    println!("\n=== CPU reference ===");
    let norm_key = weights.arch.input_layernorm_key(0);
    let norm_offset = weights.arch.norm_weight_offset();
    let h_norm = larql_inference::forward::apply_norm(weights, &h, &norm_key, norm_offset);
    let h_norm_row = h_norm.row(0);
    let norm_nz = h_norm_row.iter().filter(|v| v.abs() > 1e-10).count();
    let norm_max = h_norm_row.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    println!(
        "  CPU norm: nonzero={}/{}, max={:.4}",
        norm_nz,
        h_norm_row.len(),
        norm_max
    );

    // CPU Q proj
    let wq = weights.tensors.get(&weights.arch.attn_q_key(0)).unwrap();
    let cpu_q = h_norm.dot(&wq.t());
    let cpu_q_row = cpu_q.row(0);
    let cpu_nz = cpu_q_row.iter().filter(|v| v.abs() > 1e-10).count();
    let cpu_max = cpu_q_row.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    println!(
        "  CPU Q proj: nonzero={}/{}, max={:.4}",
        cpu_nz,
        cpu_q_row.len(),
        cpu_max
    );

    println!("\n=== Done ===");
    Ok(())
}
