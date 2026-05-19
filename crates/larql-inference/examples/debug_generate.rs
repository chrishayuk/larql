//! Debug: why does generate() fall back to CPU?

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = larql_inference::InferenceModel::load("google/gemma-3-4b-it")?;
    let weights = model.weights();
    let vd = std::path::PathBuf::from("output/gemma3-4b-v2.vindex");
    let mut index =
        larql_vindex::VectorIndex::load_vindex(&vd, &mut larql_vindex::SilentLoadCallbacks)?;
    let _ = index.load_attn_kquant(&vd);
    let _ = index.load_interleaved_kquant(&vd);
    let _ = index.load_interleaved_q4(&vd);
    let _ = index.load_lm_head(&vd);
    let _ = index.load_down_features(&vd);

    let backend = larql_inference::default_backend();
    let gate_index: &dyn larql_vindex::GateIndex = &index;

    println!("=== Debug Generate Pipeline ===\n");
    println!(
        "Backend: {} (supports Q4_K = {})",
        backend.name(),
        backend.supports_quant(::larql_compute::QuantFormat::Q4_K)
    );
    println!(
        "has_q4k attn L0: {}",
        index.attn_kquant_layer_data(0).is_some()
    );
    println!("has_q8 attn L0: {}", index.attn_q8_layer_data(0).is_some());
    println!("interleaved Q4K: {}", gate_index.has_interleaved_kquant());
    println!(
        "interleaved Q4: {}",
        gate_index.interleaved_q4_mmap_ref().is_some()
    );
    println!("has_lm_head: {}", index.has_lm_head());
    println!("down_features: {}", gate_index.has_down_features());

    // Check what predict_honest does
    let (q4_ffn, ffn_is_q4k) = if let Some(mmap) = gate_index.interleaved_kquant_mmap_ref() {
        (Some(mmap), true)
    } else {
        (gate_index.interleaved_q4_mmap_ref(), false)
    };
    println!(
        "\nFFN data: q4k={ffn_is_q4k}, has_data={}",
        q4_ffn.is_some()
    );

    let has_q4k_attn = index.attn_kquant_layer_data(0).is_some();
    let has_q8_attn = index.attn_q8_layer_data(0).is_some();
    println!("Attn data: q4k={has_q4k_attn}, q8={has_q8_attn}");

    if let Some(q4_ffn_mmap) = q4_ffn {
        let intermediate = gate_index.num_features(0);
        let hidden = weights.hidden_size;
        println!("intermediate={intermediate}, hidden={hidden}");

        let q4_ffn_per_matrix = if ffn_is_q4k {
            (intermediate * hidden).div_ceil(256) * 148
        } else {
            intermediate * hidden / 32 * 18
        };
        let q4_ffn_per_layer = q4_ffn_per_matrix * 3;
        println!("q4_ffn_per_matrix={q4_ffn_per_matrix}, per_layer={q4_ffn_per_layer}");
        println!("q4_ffn_mmap total bytes: {}", q4_ffn_mmap.len());
        println!("expected for 34 layers: {}", q4_ffn_per_layer * 34);
        println!(
            "mmap >= expected: {}",
            q4_ffn_mmap.len() >= q4_ffn_per_layer * 34
        );

        // Try building one layer
        let ffn_format = if ffn_is_q4k {
            larql_compute::QuantFormat::Q4_K
        } else {
            larql_compute::QuantFormat::Q4_0
        };
        let layers = larql_inference::layer_graph::pipeline_layer::build_pipeline_layers(
            weights,
            &index,
            0..1,
            q4_ffn_mmap,
            q4_ffn_per_matrix,
            ffn_format,
        );
        println!(
            "\nBuilt layer 0: head_dim={}, num_q={}, num_kv={}, rope_base={:.0}",
            layers[0].head_dim, layers[0].num_q_heads, layers[0].num_kv_heads, layers[0].rope_base
        );
        println!("wq data len: {}", layers[0].wq.data.len());
        println!("wk data len: {}", layers[0].wk.data.len());
        println!("gate data len: {}", layers[0].gate.data.len());
        println!("input_norm len: {}", layers[0].input_norm.len());

        // Try decode_token with 1 layer
        let encoding = model.tokenizer().encode("Hello", true).unwrap();
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let h = larql_inference::forward::embed_tokens_pub(weights, &token_ids);
        let x: Vec<f32> = h.row(0).to_vec();

        // `q_dim` / `kv_dim` used to be passed to the pre-refactor
        // `decode_token` arg-list; the new 4-arg API no longer needs them.

        println!("\nTrying decode_token with 1 layer...");
        let result = backend.decode_token(&layers, &x, hidden, intermediate);
        println!(
            "decode_token result: {}",
            if result.is_some() { "Some" } else { "None" }
        );

        // Try with all 34 layers
        println!("\nBuilding all 34 layers...");
        let all_layers = larql_inference::layer_graph::pipeline_layer::build_pipeline_layers(
            weights,
            &index,
            0..weights.num_layers,
            q4_ffn_mmap,
            q4_ffn_per_matrix,
            ffn_format,
        );
        println!("Built {} layers", all_layers.len());

        println!(
            "Trying decode_token with all {} layers...",
            all_layers.len()
        );
        backend.reset_kv_cache();
        let result = backend.decode_token(&all_layers, &x, hidden, intermediate);
        println!(
            "decode_token result: {}",
            if result.is_some() { "Some" } else { "None" }
        );
        if let Some(ref h) = result {
            let nonzero = h.iter().filter(|v| v.abs() > 1e-10).count();
            let max = h.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            println!("  len={}, nonzero={}, max_abs={:.4}", h.len(), nonzero, max);
        }

        // Try prefill
        println!(
            "\nTrying prefill_kquant with all layers, seq={}...",
            token_ids.len()
        );
        backend.reset_kv_cache();
        let x_all: Vec<f32> = h.as_slice().unwrap_or(&[]).to_vec();
        let softcap = weights.arch.attn_logit_softcapping().unwrap_or(0.0);
        let qk_norm = weights.arch.attn_q_norm_key(0).is_some();
        let prefill_result = backend.prefill_kquant(
            &all_layers,
            &x_all,
            hidden,
            intermediate,
            token_ids.len(),
            qk_norm,
            softcap,
        );
        println!(
            "prefill_kquant result: {}",
            if prefill_result.is_some() {
                "Some"
            } else {
                "None"
            }
        );
        if let Some(ref h) = prefill_result {
            let nonzero = h.iter().filter(|v| v.abs() > 1e-10).count();
            let max = h.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            println!("  len={}, nonzero={}, max_abs={:.4}", h.len(), nonzero, max);
        }
    }

    println!("\n=== Done ===");
    Ok(())
}
