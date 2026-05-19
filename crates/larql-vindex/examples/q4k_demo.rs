//! Streaming Q4_K extract showcase.
//!
//! Builds a tiny synthetic safetensors model in a temp directory, runs
//! the streaming vindex extractor twice — once as float (`QuantFormat::None`)
//! and once as Ollama-compatible Q4_K/Q6_K — and prints:
//!
//!   1. Size comparison of the two vindex directories.
//!   2. File layout of the quantised vindex (what's baked, what's
//!      hard-linked, what's the manifest).
//!   3. A dequant round-trip on the Q slot of layer 0 so you can see
//!      the write-side bytes actually decode back to something close
//!      to the source data.
//!
//! This is a pure-synthetic demo — no model download, runs in CI.
//!
//! Run: cargo run --release -p larql-vindex --example q4k_demo

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use larql_vindex::{
    build_vindex_streaming, ExtractLevel, QuantFormat, SilentBuildCallbacks, StorageDtype,
};

fn main() {
    println!("=== larql-vindex: streaming Q4_K demo ===\n");

    let tmp = std::env::temp_dir().join("larql_q4k_demo");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    let model_dir = tmp.join("synth_model");
    let out_f32 = tmp.join("out_f32.vindex");
    let out_q4k = tmp.join("out_q4k.vindex");
    std::fs::create_dir_all(&model_dir).unwrap();

    // ── Synthetic model: small llama, real tensor shapes, filler data ──
    //
    // Dimensions chosen so each attn/FFN tensor spans multiple Q4_K
    // super-blocks (256 f32s), not just one — gives the manifest and
    // the size comparison realistic shape.
    let hidden = 64usize;
    let intermediate = 128usize;
    let num_layers = 4usize;
    let vocab = 32usize;

    println!("Building synthetic llama fixture...");
    println!("  hidden={hidden}  intermediate={intermediate}  layers={num_layers}  vocab={vocab}");
    make_synthetic_model(&model_dir, hidden, intermediate, num_layers, vocab);

    // ── Extract twice: once as f32, once as Q4_K ──

    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(MINIMAL_TOKENIZER).unwrap();

    println!("\nExtracting as f32 ({}):", out_f32.display());
    let t0 = std::time::Instant::now();
    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "demo/q4k",
        &out_f32,
        5,
        ExtractLevel::All,
        StorageDtype::F32,
        QuantFormat::None,
        larql_vindex::WriteWeightsOptions::default(),
        larql_vindex::KquantWriteOptions::default(),
        false,
        &mut cb,
    )
    .expect("f32 extract");
    println!("  took {:.0} ms", t0.elapsed().as_secs_f64() * 1000.0);

    println!(
        "\nExtracting as Q4_K ({}):  (--quant q4k path)",
        out_q4k.display()
    );
    let t0 = std::time::Instant::now();
    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "demo/q4k",
        &out_q4k,
        5,
        ExtractLevel::All,
        StorageDtype::F32,
        QuantFormat::Q4K,
        larql_vindex::WriteWeightsOptions::default(),
        larql_vindex::KquantWriteOptions::default(),
        false,
        &mut cb,
    )
    .expect("q4k extract");
    println!("  took {:.0} ms", t0.elapsed().as_secs_f64() * 1000.0);

    // ── Size comparison ──

    let f32_size = dir_size(&out_f32);
    let q4k_size = dir_size(&out_q4k);
    let ratio = f32_size as f64 / q4k_size as f64;

    println!("\n── Size comparison ──");
    println!("  f32 vindex : {:>10}", fmt_bytes(f32_size));
    println!("  Q4_K vindex: {:>10}", fmt_bytes(q4k_size));
    println!("  ratio      : {ratio:.2}× smaller");

    // ── File layout of the Q4_K vindex ──

    println!("\n── Q4_K vindex layout ──");
    let mut entries: Vec<_> = std::fs::read_dir(&out_q4k)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| {
            (
                e.file_name().into_string().unwrap(),
                e.metadata().map(|m| m.len()).unwrap_or(0),
            )
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, size) in &entries {
        let marker = if name.contains("q4k") {
            " ← Q4_K bytes"
        } else {
            ""
        };
        println!("  {:<38} {:>10}{marker}", name, fmt_bytes(*size));
    }

    // ── Manifest preview ──

    println!("\n── attn_weights_q4k_manifest.json (first 2 entries) ──");
    let attn_manifest =
        std::fs::read_to_string(out_q4k.join("attn_weights_q4k_manifest.json")).unwrap();
    let attn_entries: Vec<serde_json::Value> = serde_json::from_str(&attn_manifest).unwrap();
    for entry in attn_entries.iter().take(2) {
        println!("  {{ key: {},", entry["key"].as_str().unwrap());
        println!(
            "    shape: {:?}, format: {}, offset: {}, length: {} }}",
            entry["shape"].as_array().unwrap(),
            entry["format"].as_str().unwrap(),
            entry["offset"].as_u64().unwrap(),
            entry["length"].as_u64().unwrap()
        );
    }
    println!(
        "  ... {} more entries (4 per layer × {num_layers} layers)",
        attn_entries.len() - 2
    );

    // ── Config dispatch ──

    let cfg = larql_vindex::load_vindex_config(&out_q4k).unwrap();
    println!("\n── index.json dispatch field ──");
    println!("  config.quant = {}", cfg.quant);
    println!("  (loaders branch on this — no filename sniffing required)");

    // ── Dequant round-trip sample ──

    println!("\n── Dequant round-trip (layer 0 Q tensor) ──");
    let mut lcb = larql_vindex::SilentLoadCallbacks;
    let mut index = larql_vindex::VectorIndex::load_vindex(&out_q4k, &mut lcb).unwrap();
    index.load_attn_kquant(&out_q4k).unwrap();
    let slices = index.attn_kquant_layer_data(0).expect("layer 0 slices");
    let (q_bytes, q_format) = slices[0];
    let n_elements = hidden * hidden; // Q shape [hidden, hidden]
                                      // Dequant reads from the raw slab; padded tail beyond n_elements
                                      // is zero and left unchanged.
    let padded = n_elements.div_ceil(256) * 256;
    let dequant = larql_models::quant::ggml::dequantize_q4_k(q_bytes, padded).unwrap();

    let source_sample: Vec<f32> = (0..n_elements).map(|i| (i as f32) * 0.01).collect();
    let max_err = dequant[..n_elements]
        .iter()
        .zip(source_sample.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let mean_err = dequant[..n_elements]
        .iter()
        .zip(source_sample.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / n_elements as f32;

    println!("  format:      {q_format}");
    println!("  n_elements:  {n_elements}  (padded to {padded} for super-blocks)");
    println!("  max error:   {max_err:.5}");
    println!("  mean error:  {mean_err:.5}");
    println!("  first 5 source:  {:?}", &source_sample[..5]);
    println!(
        "  first 5 dequant: {:?}",
        &dequant[..5]
            .iter()
            .map(|x| (x * 10000.0).round() / 10000.0)
            .collect::<Vec<_>>()
    );

    // ── V slot is Q6_K — tighter tolerance ──

    let (v_bytes, v_format) = slices[2];
    let v_dequant = larql_models::quant::ggml::dequantize_q6_k(v_bytes, padded).unwrap();
    let v_max_err = v_dequant[..n_elements]
        .iter()
        .zip(source_sample.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("\n  V slot uses {v_format} (higher precision than Q/K/O):");
    println!("    max error:   {v_max_err:.5}  (about 2-3× tighter than Q4_K)");

    // ── Cleanup ──

    let _ = std::fs::remove_dir_all(&tmp);
    println!("\n=== done ===");
}

// ── Fixture helpers ──

const MINIMAL_TOKENIZER: &[u8] =
    br#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;

fn make_synthetic_model(
    dir: &Path,
    hidden: usize,
    intermediate: usize,
    num_layers: usize,
    vocab: usize,
) {
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": hidden,
        "rope_theta": 10000.0,
        "vocab_size": vocab,
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();
    std::fs::write(dir.join("tokenizer.json"), MINIMAL_TOKENIZER).unwrap();

    let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut metadata: Vec<(String, Vec<usize>)> = Vec::new();

    let push = |tensors: &mut HashMap<String, Vec<f32>>,
                metadata: &mut Vec<(String, Vec<usize>)>,
                name: &str,
                shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        tensors.insert(name.into(), data);
        metadata.push((name.into(), shape));
    };

    push(
        &mut tensors,
        &mut metadata,
        "model.embed_tokens.weight",
        vec![vocab, hidden],
    );
    push(
        &mut tensors,
        &mut metadata,
        "model.norm.weight",
        vec![hidden],
    );
    for layer in 0..num_layers {
        let lp = format!("model.layers.{layer}");
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.q_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.k_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.v_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.o_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.gate_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.up_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.down_proj.weight"),
            vec![hidden, intermediate],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.input_layernorm.weight"),
            vec![hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.post_attention_layernorm.weight"),
            vec![hidden],
        );
    }

    let tensor_bytes: Vec<(String, Vec<u8>, Vec<usize>)> = metadata
        .iter()
        .map(|(name, shape)| {
            let data = &tensors[name];
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            (name.clone(), bytes, shape.clone())
        })
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensor_bytes
        .iter()
        .map(|(name, bytes, shape)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .unwrap(),
            )
        })
        .collect();
    let serialized = safetensors::tensor::serialize(views, None).unwrap();
    std::fs::write(dir.join("model.safetensors"), &serialized).unwrap();
}

fn dir_size(p: &PathBuf) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(p) {
        for e in entries.flatten() {
            if let Ok(md) = e.metadata() {
                total += md.len();
            }
        }
    }
    total
}

fn fmt_bytes(n: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < UNITS.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else {
        format!("{v:.2} {}", UNITS[i])
    }
}
