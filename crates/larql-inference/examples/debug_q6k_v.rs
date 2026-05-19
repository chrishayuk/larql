//! Debug: q6k_matvec with real V projection data

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vd = std::path::PathBuf::from("output/gemma3-4b-v2.vindex");
    let mut index =
        larql_vindex::VectorIndex::load_vindex(&vd, &mut larql_vindex::SilentLoadCallbacks)?;
    let _ = index.load_attn_kquant(&vd);
    let backend = larql_compute::default_backend();

    println!("=== Q6_K V projection debug ===\n");

    if let Some([_q, _k, v, _o]) = index.attn_kquant_layer_data(0) {
        let v_data = v.0;
        let v_format = v.1;
        println!("V data: {} bytes, format={}", v_data.len(), v_format);

        let kv_dim = 1024; // 4 heads × 256 head_dim
        let hidden = 2560;

        // Expected Q6_K size: (kv_dim * hidden) / 256 * 210
        let expected = (kv_dim * hidden) / 256 * 210;
        println!("Expected Q6_K: {} bytes", expected);
        println!("Match: {}\n", v_data.len() == expected);

        // Test with ones input
        let x_ones: Vec<f32> = vec![1.0; hidden];
        let result = backend.q6k_matvec(v_data, &x_ones, kv_dim, hidden);
        match result {
            Some(ref r) => {
                let nz = r.iter().filter(|&&v| v.abs() > 1e-10).count();
                let max_abs = r.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                println!(
                    "q6k_matvec(ones): nonzero={}/{}, max_abs={:.4}",
                    nz,
                    r.len(),
                    max_abs
                );
            }
            None => println!("q6k_matvec(ones): None"),
        }

        // Test with the actual embed
        let model = larql_inference::InferenceModel::load("google/gemma-3-4b-it")?;
        let weights = model.weights();
        let encoding = model.tokenizer().encode("Hello", true).unwrap();
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let h = larql_inference::forward::embed_tokens_pub(weights, &ids);
        let x: Vec<f32> = h.row(0).to_vec();

        let result = backend.q6k_matvec(v_data, &x, kv_dim, hidden);
        match result {
            Some(ref r) => {
                let nz = r.iter().filter(|&&v| v.abs() > 1e-10).count();
                let max_abs = r.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                println!(
                    "q6k_matvec(embed): nonzero={}/{}, max_abs={:.4}",
                    nz,
                    r.len(),
                    max_abs
                );
            }
            None => println!("q6k_matvec(embed): None"),
        }

        // CPU reference: dequantize Q6_K and matmul
        println!("\nCPU Q6_K dequant test:");
        let deq = larql_models::quant::ggml::dequantize(
            v_data,
            larql_models::quant::ggml::TYPE_Q6_K,
            kv_dim * hidden,
        );
        match deq {
            Ok(ref f32_data) => {
                let nz = f32_data.iter().filter(|v| v.abs() > 1e-10).count();
                let max_abs = f32_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                println!(
                    "  Dequantized: {} floats, nonzero={}, max_abs={:.4}",
                    f32_data.len(),
                    nz,
                    max_abs
                );

                // Manual matmul: V[kv_dim, hidden] @ x[hidden] → out[kv_dim]
                let mut out = vec![0.0f32; kv_dim];
                for row in 0..kv_dim {
                    for col in 0..hidden {
                        out[row] += f32_data[row * hidden + col] * x[col];
                    }
                }
                let nz = out.iter().filter(|v| v.abs() > 1e-10).count();
                let max_abs = out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                println!(
                    "  CPU matmul: nonzero={}/{}, max_abs={:.4}",
                    nz, kv_dim, max_abs
                );
            }
            Err(e) => println!("  Dequantize failed: {}", e),
        }
    }

    println!("\n=== Done ===");
    Ok(())
}
