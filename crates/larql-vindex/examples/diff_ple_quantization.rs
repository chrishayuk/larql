//! Measure the round-trip error on Gemma 4 E2B's PLE tensors:
//! dense-loaded BF16→f32 vs the Q4K-vindex dequantised f32.
//!
//! Usage: `cargo run --release -p larql-vindex \
//!          --example diff_ple_quantization -- \
//!          ~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/<rev> \
//!          output/gemma4-e2b-q4k.vindex`

use std::env;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: {} <model_dir> <vindex_dir>",
            args.first()
                .map(String::as_str)
                .unwrap_or("diff_ple_quantization")
        );
        std::process::exit(2);
    }
    let model_dir = PathBuf::from(&args[1]);
    let vindex_dir = PathBuf::from(&args[2]);

    eprintln!("Loading dense model from {}...", model_dir.display());
    let dense = larql_models::load_model_dir(&model_dir).expect("dense load");
    eprintln!("  dense tensors: {}", dense.tensors.len());

    eprintln!("Loading Q4K vindex from {}...", vindex_dir.display());
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let mut q4k = larql_vindex::load_model_weights_kquant(&vindex_dir, &mut cb).expect("q4k load");
    eprintln!("  q4k tensors:   {}", q4k.tensors.len());

    // Also dequantise layer 0's attn/FFN Q4K blocks into q4k.tensors so the
    // same diff loop covers the matmul weights, not just PLE tensors.
    let mut attn_cb = larql_vindex::SilentLoadCallbacks;
    let mut index =
        larql_vindex::VectorIndex::load_vindex(&vindex_dir, &mut attn_cb).expect("vindex load");
    index
        .load_attn_kquant(&vindex_dir)
        .expect("load_attn_kquant");
    index
        .load_interleaved_kquant(&vindex_dir)
        .expect("load_interleaved");
    for layer in [0usize, 10] {
        let hidden = q4k.hidden_size;
        let intermediate = q4k.intermediate_size;
        let (num_q, num_kv, hd, q_key, k_key, v_key, o_key, g_key, u_key, d_key) = {
            let arch = &*q4k.arch;
            (
                arch.num_q_heads_for_layer(layer),
                arch.num_kv_heads_for_layer(layer),
                arch.head_dim_for_layer(layer),
                arch.attn_q_key(layer),
                arch.attn_k_key(layer),
                arch.attn_v_key(layer),
                arch.attn_o_key(layer),
                arch.ffn_gate_key(layer),
                arch.ffn_up_key(layer),
                arch.ffn_down_key(layer),
            )
        };
        let q_dim = num_q * hd;
        let kv_dim = num_kv * hd;
        let attn = index.attn_kquant_layer_data(layer).unwrap();
        let ffn = index.interleaved_kquant_layer_data(layer).unwrap();
        let dequant = |(bytes, fmt): (&[u8], &str), rows: usize, cols: usize| {
            let n = rows * cols;
            let padded = n.div_ceil(256) * 256;
            let floats = match fmt {
                "Q4_K" => larql_models::quant::ggml::dequantize_q4_k(bytes, padded).unwrap(),
                "Q6_K" => larql_models::quant::ggml::dequantize_q6_k(bytes, padded).unwrap(),
                _ => panic!("unexpected fmt {fmt}"),
            };
            ndarray::Array2::from_shape_vec((rows, cols), floats[..n].to_vec()).unwrap()
        };
        q4k.tensors
            .insert(q_key, dequant(attn[0], q_dim, hidden).into_shared());
        q4k.tensors
            .insert(k_key, dequant(attn[1], kv_dim, hidden).into_shared());
        q4k.tensors
            .insert(v_key, dequant(attn[2], kv_dim, hidden).into_shared());
        q4k.tensors
            .insert(o_key, dequant(attn[3], hidden, q_dim).into_shared());
        q4k.tensors
            .insert(g_key, dequant(ffn[0], intermediate, hidden).into_shared());
        q4k.tensors
            .insert(u_key, dequant(ffn[1], intermediate, hidden).into_shared());
        q4k.tensors
            .insert(d_key, dequant(ffn[2], hidden, intermediate).into_shared());
    }

    // Key-set diff: collapse `.<digits>.` to `.N.` so per-layer keys
    // collapse to one pattern. Skip multimodal branches (vision/audio) —
    // Q4K vindex is text-only by design.
    let collapse = |k: &str| -> Option<String> {
        if k.contains("audio_tower")
            || k.contains("vision_tower")
            || k.contains("embed_audio")
            || k.contains("embed_vision")
        {
            return None;
        }
        let parts: Vec<String> = k
            .split('.')
            .map(|p| {
                if p.chars().all(|c| c.is_ascii_digit()) {
                    "N".to_string()
                } else {
                    p.to_string()
                }
            })
            .collect();
        Some(parts.join("."))
    };

    use std::collections::BTreeSet;
    let dense_tensor_pats: BTreeSet<String> =
        dense.tensors.keys().filter_map(|k| collapse(k)).collect();
    let q4k_tensor_pats: BTreeSet<String> =
        q4k.tensors.keys().filter_map(|k| collapse(k)).collect();
    let dense_vec_pats: BTreeSet<String> =
        dense.vectors.keys().filter_map(|k| collapse(k)).collect();
    let q4k_vec_pats: BTreeSet<String> = q4k.vectors.keys().filter_map(|k| collapse(k)).collect();

    println!("\n== TENSOR patterns in DENSE but MISSING from Q4K ==");
    for p in dense_tensor_pats.difference(&q4k_tensor_pats) {
        println!("  {p}");
    }
    println!("\n== TENSOR patterns in Q4K but not in DENSE ==");
    for p in q4k_tensor_pats.difference(&dense_tensor_pats) {
        println!("  {p}");
    }
    println!("\n== VECTOR patterns in DENSE but MISSING from Q4K ==");
    for p in dense_vec_pats.difference(&q4k_vec_pats) {
        println!("  {p}");
    }
    println!("\n== VECTOR patterns in Q4K but not in DENSE ==");
    for p in q4k_vec_pats.difference(&dense_vec_pats) {
        println!("  {p}");
    }

    let targets = [
        "per_layer_model_projection.weight",
        "embed_tokens_per_layer.weight",
        "layers.0.per_layer_input_gate.weight",
        "layers.0.per_layer_projection.weight",
        "layers.17.per_layer_input_gate.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "layers.10.self_attn.q_proj.weight",
    ];

    println!();
    println!(
        "{:55} {:>12} {:>14} {:>14} {:>10}",
        "tensor", "n_elements", "max_abs_err", "mean_abs_err", "cos_sim"
    );
    println!("{}", "-".repeat(110));

    for key in targets {
        let d = dense.tensors.get(key);
        let q = q4k.tensors.get(key);
        match (d, q) {
            (Some(d), Some(q)) => {
                let ds = d.shape();
                let qs = q.shape();
                if ds != qs {
                    println!("{:55} SHAPE MISMATCH dense={:?} q4k={:?}", key, ds, qs);
                    continue;
                }
                // Per-element diff across a sample window to keep cost bounded
                // on the big embed_tokens_per_layer.
                let n_total = d.len();
                let stride = (n_total / 200_000).max(1);
                let mut max_abs = 0.0f32;
                let mut sum_abs = 0.0f64;
                let mut dot = 0.0f64;
                let mut d_norm2 = 0.0f64;
                let mut q_norm2 = 0.0f64;
                let mut count = 0u64;
                let ds_slice = d.as_slice().expect("dense contig");
                let qs_slice = q.as_slice().expect("q4k contig");
                let mut i = 0usize;
                while i < n_total {
                    let a = ds_slice[i];
                    let b = qs_slice[i];
                    let diff = (a - b).abs();
                    max_abs = max_abs.max(diff);
                    sum_abs += diff as f64;
                    dot += (a as f64) * (b as f64);
                    d_norm2 += (a as f64).powi(2);
                    q_norm2 += (b as f64).powi(2);
                    count += 1;
                    i += stride;
                }
                let mean_abs = sum_abs / count as f64;
                let cos = dot / (d_norm2.sqrt() * q_norm2.sqrt() + 1e-12);
                println!(
                    "{:55} {:>12} {:>14.6e} {:>14.6e} {:>10.6}",
                    key, n_total, max_abs, mean_abs, cos
                );
            }
            (Some(_), None) => println!("{:55} MISSING IN Q4K", key),
            (None, Some(_)) => println!("{:55} MISSING IN DENSE", key),
            (None, None) => println!("{:55} missing both sides", key),
        }
    }
}
