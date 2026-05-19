//! Debug: check Q4_K weight data integrity

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = larql_inference::InferenceModel::load("google/gemma-3-4b-it")?;
    let weights = model.weights();
    let vd = std::path::PathBuf::from("output/gemma3-4b-v2.vindex");
    let mut index =
        larql_vindex::VectorIndex::load_vindex(&vd, &mut larql_vindex::SilentLoadCallbacks)?;
    let _ = index.load_attn_kquant(&vd);
    let _ = index.load_interleaved_kquant(&vd);
    let backend = larql_inference::default_backend();

    println!("=== Q4_K Data Integrity Check ===\n");

    // Check attn Q4K data for layer 0
    if let Some([q, k, v, o]) = index.attn_kquant_layer_data(0) {
        println!("Layer 0 attention Q4_K:");
        println!("  Q: {} bytes, format={}", q.0.len(), q.1);
        println!("  K: {} bytes, format={}", k.0.len(), k.1);
        println!("  V: {} bytes, format={}", v.0.len(), v.1);
        println!("  O: {} bytes, format={}", o.0.len(), o.1);

        // Check first few bytes aren't all zeros
        let q_nonzero = q.0.iter().take(1000).filter(|&&b| b != 0).count();
        let k_nonzero = k.0.iter().take(1000).filter(|&&b| b != 0).count();
        println!("  Q first 1000 bytes: {}/1000 nonzero", q_nonzero);
        println!("  K first 1000 bytes: {}/1000 nonzero", k_nonzero);

        // Expected sizes
        let hidden = weights.hidden_size; // 2560
        let head_dim = weights.head_dim; // 256
        let nq = weights.num_q_heads; // 8
        let nkv = weights.num_kv_heads; // 4
        let q_dim = nq * head_dim; // 2048
        let kv_dim = nkv * head_dim; // 1024

        // Q4_K: 148 bytes per 256 values, rows × cols
        let expected_q = (q_dim * hidden).div_ceil(256) * 148;
        let expected_k = (kv_dim * hidden).div_ceil(256) * 148;
        let _expected_o = (hidden * q_dim).div_ceil(256) * 148;
        println!(
            "\n  Expected Q bytes: {} (q_dim={} × hidden={})",
            expected_q, q_dim, hidden
        );
        println!("  Actual Q bytes:   {}", q.0.len());
        println!("  Match: {}\n", q.0.len() == expected_q);
        println!("  Expected K bytes: {}", expected_k);
        println!("  Actual K bytes:   {}", k.0.len());
        println!("  Match: {}\n", k.0.len() == expected_k);
    } else {
        println!("No Q4K attention data!");
    }

    // Try a simple Q4K matvec to verify the data works
    println!("=== Q4K matvec test ===\n");
    if let Some([q_data, _, _, _]) = index.attn_kquant_layer_data(0) {
        let hidden = weights.hidden_size;
        let q_dim = weights.num_q_heads * weights.head_dim;
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.001).collect();

        // Try via compute backend
        let result = backend.q4k_matvec(q_data.0, &x, q_dim, hidden);
        if let Some(ref r) = result {
            let nonzero = r.iter().filter(|v| v.abs() > 1e-10).count();
            let max = r.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            println!(
                "  q4k_matvec result: len={}, nonzero={}, max={:.4}",
                r.len(),
                nonzero,
                max
            );
        } else {
            println!("  q4k_matvec returned None!");
        }
    }

    // Check interleaved Q4K FFN data
    let gate_index: &dyn larql_vindex::GateIndex = &index;
    if let Some(mmap) = gate_index.interleaved_kquant_mmap_ref() {
        let intermediate = gate_index.num_features(0);
        let hidden = weights.hidden_size;
        let per_matrix = (intermediate * hidden).div_ceil(256) * 148;

        println!("\n=== Interleaved Q4K FFN ===\n");
        println!("  Total mmap: {} bytes", mmap.len());
        println!("  Per matrix: {} bytes", per_matrix);
        println!("  Per layer: {} bytes", per_matrix * 3);
        println!("  Expected 34L: {} bytes", per_matrix * 3 * 34);

        // Check gate data nonzero
        let gate_data = &mmap[0..per_matrix];
        let gate_nz = gate_data.iter().take(1000).filter(|&&b| b != 0).count();
        println!("  Gate L0 first 1000 bytes: {}/1000 nonzero", gate_nz);

        // Try gate matvec
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.001).collect();
        let result = backend.q4k_matvec(gate_data, &x, intermediate, hidden);
        if let Some(ref r) = result {
            let nonzero = r.iter().filter(|v| v.abs() > 1e-10).count();
            let max = r.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            println!(
                "  Gate Q4K matvec: len={}, nonzero={}, max={:.4}",
                r.len(),
                nonzero,
                max
            );
        } else {
            println!("  Gate Q4K matvec returned None!");
        }
    }

    println!("\n=== Done ===");
    Ok(())
}
