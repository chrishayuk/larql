//! Streaming-extract throughput bench.
//!
//! Compares `build_vindex_streaming` with `QuantFormat::None` (f32
//! write path) vs `QuantFormat::Q4K` (streaming quantise) on a
//! single-layer synthetic safetensors fixture shaped like a real LLM.
//!
//! The headline this bench produces: how long does the one-pass Q4_K
//! extractor take vs the classic f32 extractor on the same data? The
//! ratio tells you what the `--quant q4k` CLI flag is actually doing
//! — quantisation work in the write path vs the f32 baseline, no
//! post-hoc build tools.
//!
//! Synthetic dims: hidden=512, intermediate=1024, 1 layer, vocab=1024.
//! Each extract writes its vindex to a fresh temp dir — setup is
//! amortised across iterations, teardown is deferred to the bench
//! runner's drop.
//!
//! Run: `cargo bench -p larql-vindex --bench extract_throughput`

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use larql_vindex::{
    build_vindex_streaming, ExtractLevel, QuantFormat, SilentBuildCallbacks, StorageDtype,
};

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

fn bench_extract_throughput(c: &mut Criterion) {
    // One-layer production-scale dims. `hidden=512`, `intermediate=1024` is
    // chosen as the sweet spot: small enough to extract in tens of ms (so
    // criterion's outer loop converges), wide enough that Q4_K's per-block
    // overhead is realistic (~4 blocks per Q/K/V/O tensor, ~8 blocks for
    // gate/up/down).
    let hidden = 512usize;
    let intermediate = 1024usize;
    let num_layers = 1usize;
    let vocab = 1024usize;

    let bench_root = std::env::temp_dir().join("larql_bench_extract_throughput");
    let _ = std::fs::remove_dir_all(&bench_root);
    let model_dir = bench_root.join("synth_model");
    make_model(&model_dir, hidden, intermediate, num_layers, vocab);

    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(MINIMAL_TOKENIZER).unwrap();

    let mut group = c.benchmark_group("extract_throughput");
    group.sample_size(20);

    for (tag, quant) in [("f32", QuantFormat::None), ("q4k", QuantFormat::Q4K)] {
        let out_dir = bench_root.join(format!("out_{tag}"));
        group.bench_with_input(BenchmarkId::from_parameter(tag), &quant, |b, &q| {
            b.iter(|| {
                // Clean prior run so build_vindex_streaming has a fresh dir.
                let _ = std::fs::remove_dir_all(&out_dir);
                let mut cb = SilentBuildCallbacks;
                build_vindex_streaming(
                    &model_dir,
                    &tokenizer,
                    "bench/extract",
                    &out_dir,
                    5,
                    ExtractLevel::All,
                    StorageDtype::F32,
                    q,
                    larql_vindex::WriteWeightsOptions::default(),
                    larql_vindex::KquantWriteOptions::default(),
                    false,
                    &mut cb,
                )
                .expect("extract");
            });
        });
    }

    // ── Auto-resume case (round-3): time the resumed run vs the
    //    fresh Q4K case above. Produce a "reference" extract once,
    //    then per-iteration plant a checkpoint that says the gate
    //    phase is already done and rerun.
    let ref_dir = bench_root.join("out_q4k_resume_ref");
    let _ = std::fs::remove_dir_all(&ref_dir);
    {
        let mut cb = SilentBuildCallbacks;
        build_vindex_streaming(
            &model_dir,
            &tokenizer,
            "bench/extract",
            &ref_dir,
            5,
            ExtractLevel::All,
            StorageDtype::F32,
            QuantFormat::Q4K,
            larql_vindex::WriteWeightsOptions::default(),
            larql_vindex::KquantWriteOptions::default(),
            false,
            &mut cb,
        )
        .expect("reference extract for resume bench");
    }
    let ref_idx: serde_json::Value =
        serde_json::from_slice(&std::fs::read(ref_dir.join("index.json")).unwrap()).unwrap();
    let layers = ref_idx["layers"].clone();
    let checkpoint_json = serde_json::json!({
        "version": 1,
        "model_dir": model_dir.display().to_string(),
        "model_name": "bench/extract",
        "num_layers": num_layers,
        "completed": ["gate"],
        "last_update": "2026-04-25T00:00:00Z",
        "gate_layer_infos": layers,
    });
    let checkpoint_text = serde_json::to_string_pretty(&checkpoint_json).unwrap();

    let resume_dir = bench_root.join("out_q4k_resume");
    group.bench_function("q4k_resume_after_gate", |b| {
        b.iter(|| {
            let _ = std::fs::remove_dir_all(&resume_dir);
            std::fs::create_dir_all(&resume_dir).unwrap();
            std::fs::copy(
                ref_dir.join("gate_vectors.bin"),
                resume_dir.join("gate_vectors.bin"),
            )
            .unwrap();
            std::fs::write(
                resume_dir.join(".extract_checkpoint.json"),
                &checkpoint_text,
            )
            .unwrap();
            let mut cb = SilentBuildCallbacks;
            build_vindex_streaming(
                &model_dir,
                &tokenizer,
                "bench/extract",
                &resume_dir,
                5,
                ExtractLevel::All,
                StorageDtype::F32,
                QuantFormat::Q4K,
                larql_vindex::WriteWeightsOptions::default(),
                larql_vindex::KquantWriteOptions::default(),
                false,
                &mut cb,
            )
            .expect("resumed extract");
        });
    });

    group.finish();

    // Leave the fixture in place; criterion's auto-cleanup isn't
    // deterministic and the dir is tiny.
    let _: PathBuf = bench_root;
}

criterion_group!(benches, bench_extract_throughput);
criterion_main!(benches);
