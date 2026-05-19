//! Q4_K vs f32 per-layer attention retrieval bench.
//!
//! Inference reads per-layer attention weights hundreds of times per
//! token; this bench measures the cost of getting one layer's Q
//! tensor as a usable `Vec<f32>` from each storage format.
//!
//! Two paths, same output shape:
//!
//!   f32   — slice `attn_weights.bin` via the weight manifest,
//!            `decode_floats` (identity for f32) → `Vec<f32>`.
//!   Q4_K  — `attn_kquant_layer_data(layer)[0]` → raw Q4_K bytes,
//!            `dequantize_q4_k` → `Vec<f32>`.
//!
//! Both fixtures extract the same synthetic model to disk once at
//! setup; each iteration re-reads the on-disk data to keep mmap
//! page-cache behaviour realistic.
//!
//! Run: `cargo bench -p larql-vindex --bench q4k_vs_f32`

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

const MINIMAL_TOKENIZER: &[u8] =
    br#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;

fn make_model(dir: &Path, hidden: usize, intermediate: usize, num_layers: usize, vocab: usize) {
    std::fs::create_dir_all(dir).unwrap();
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
    let mut push = |name: &str, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.001).sin()).collect();
        tensors.insert(name.into(), data);
        metadata.push((name.into(), shape));
    };

    push("model.embed_tokens.weight", vec![vocab, hidden]);
    push("model.norm.weight", vec![hidden]);
    for layer in 0..num_layers {
        let lp = format!("model.layers.{layer}");
        push(
            &format!("{lp}.self_attn.q_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &format!("{lp}.self_attn.k_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &format!("{lp}.self_attn.v_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &format!("{lp}.self_attn.o_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &format!("{lp}.mlp.gate_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            &format!("{lp}.mlp.up_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            &format!("{lp}.mlp.down_proj.weight"),
            vec![hidden, intermediate],
        );
        push(&format!("{lp}.input_layernorm.weight"), vec![hidden]);
        push(
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

/// Grab the manifest entry for layer 0's Q tensor in the f32 vindex
/// and return (byte_offset, byte_length, n_elements). Used to slice
/// `attn_weights.bin` at bench time.
fn locate_q_entry_f32(dir: &Path) -> (u64, u64, usize) {
    let manifest_text = std::fs::read_to_string(dir.join("weight_manifest.json")).unwrap();
    let entries: Vec<serde_json::Value> = serde_json::from_str(&manifest_text).unwrap();
    for e in &entries {
        let key = e["key"].as_str().unwrap_or("");
        // Manifest keys are the normalised form (no `model.` prefix);
        // use `ends_with` so this bench works across architectures
        // that prefix differently.
        if key.ends_with("layers.0.self_attn.q_proj.weight") {
            let offset = e["offset"].as_u64().unwrap();
            let length = e["length"].as_u64().unwrap();
            let shape: Vec<usize> = e["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let n: usize = shape.iter().product();
            return (offset, length, n);
        }
    }
    panic!("Q entry not found in f32 manifest");
}

fn bench_q4k_vs_f32(c: &mut Criterion) {
    // Production-scale single layer. `hidden=2048`, `intermediate=4096`
    // is Gemma-like, large enough that both formats do real work each
    // iteration (Q tensor = 16 super-blocks for Q4_K; 16 MB raw f32).
    let hidden = 2048usize;
    let intermediate = 4096usize;
    let num_layers = 1usize;
    let vocab = 256usize;

    let root = std::env::temp_dir().join("larql_bench_q4k_vs_f32");
    let _ = std::fs::remove_dir_all(&root);
    let model_dir = root.join("synth_model");
    make_model(&model_dir, hidden, intermediate, num_layers, vocab);

    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(MINIMAL_TOKENIZER).unwrap();

    // ── Extract once per format ──
    let f32_dir = root.join("out_f32");
    let q4k_dir = root.join("out_q4k");

    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "bench/q4k_vs_f32",
        &f32_dir,
        5,
        larql_vindex::ExtractLevel::All,
        larql_vindex::StorageDtype::F32,
        larql_vindex::QuantFormat::None,
        larql_vindex::WriteWeightsOptions::default(),
        larql_vindex::KquantWriteOptions::default(),
        false,
        &mut cb,
    )
    .unwrap();

    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "bench/q4k_vs_f32",
        &q4k_dir,
        5,
        larql_vindex::ExtractLevel::All,
        larql_vindex::StorageDtype::F32,
        larql_vindex::QuantFormat::Q4K,
        larql_vindex::WriteWeightsOptions::default(),
        larql_vindex::KquantWriteOptions::default(),
        false,
        &mut cb,
    )
    .unwrap();

    // ── Size comparison printed once for context ──
    let f32_attn = std::fs::metadata(f32_dir.join("attn_weights.bin"))
        .unwrap()
        .len();
    // K-quant writer renamed the output to `attn_weights_kquant.bin`
    // (the old `attn_weights_q4k.bin` is kept as a legacy fallback for
    // older vindexes). Use the canonical filename constant from the
    // crate so this bench tracks future renames automatically.
    let q4k_attn =
        std::fs::metadata(q4k_dir.join(larql_vindex::format::filenames::ATTN_WEIGHTS_KQUANT_BIN))
            .unwrap()
            .len();
    eprintln!(
        "\n  attn_weights.bin   {} bytes (f32)\n  attn_weights_kquant.bin {} bytes ({:.2}× smaller)\n",
        f32_attn, q4k_attn,
        f32_attn as f64 / q4k_attn as f64,
    );

    // ── f32 setup: mmap the attn file, locate Q entry ──
    let f32_attn_file = std::fs::File::open(f32_dir.join("attn_weights.bin")).unwrap();
    let f32_attn_mmap = unsafe { memmap2::Mmap::map(&f32_attn_file).unwrap() };
    let (q_offset, q_length, q_elems) = locate_q_entry_f32(&f32_dir);

    // ── Q4_K setup: load via VectorIndex so attn_kquant_layer_data works ──
    let mut lcb = larql_vindex::SilentLoadCallbacks;
    let mut q4k_index = larql_vindex::VectorIndex::load_vindex(&q4k_dir, &mut lcb).unwrap();
    q4k_index.load_attn_kquant(&q4k_dir).unwrap();
    let padded = q_elems.div_ceil(256) * 256;

    let mut group = c.benchmark_group("q4k_vs_f32_per_layer_q");
    group.sample_size(50);

    // f32 path: slice mmap + decode. decode_floats on f32 is a
    // bitwise memcpy but still copies into a fresh Vec<f32> the same
    // size the Q4_K dequant produces, so the two outputs are directly
    // comparable.
    group.bench_with_input(BenchmarkId::from_parameter("f32"), &(), |b, _| {
        b.iter(|| {
            let bytes = &f32_attn_mmap[q_offset as usize..(q_offset + q_length) as usize];
            let floats =
                larql_vindex::config::dtype::decode_floats(bytes, larql_vindex::StorageDtype::F32);
            criterion::black_box(floats);
        });
    });

    // Q4_K path: slice lookup + dequant. `attn_kquant_layer_data[0]` is
    // the Q slot, Q4_K format; `dequantize_q4_k` produces a Vec<f32>
    // the same size as the f32 path's output (minus padding overhead).
    group.bench_with_input(BenchmarkId::from_parameter("q4k"), &(), |b, _| {
        b.iter(|| {
            let slices = q4k_index.attn_kquant_layer_data(0).unwrap();
            let (bytes, _format) = slices[0];
            let floats = larql_models::quant::ggml::dequantize_q4_k(bytes, padded).unwrap();
            criterion::black_box(floats);
        });
    });

    group.finish();
    let _: PathBuf = root;
}

criterion_group!(benches, bench_q4k_vs_f32);
criterion_main!(benches);
