//! Criterion benchmarks for `COMPILE INTO VINDEX` — measures the
//! end-to-end bake on a small synthetic vindex with a non-trivial
//! patch session. The bake does three things: hard-link unchanged
//! weight files, fresh-write `gate_vectors.bin`, and (if there are
//! down vector overrides) copy + seek-write `down_weights.bin`.
//!
//! Run: `cargo bench -p larql-lql --bench compile`
//!
//! The hot path here is `patch_down_weights` (which copies the source
//! `down_weights.bin` and seek-writes the override columns in place).
//! On a real Gemma 4B vindex this takes ~25 seconds because the copy
//! is multi-GB; the synthetic bench shrinks dimensions to make the
//! operation fit comfortably in a criterion sample window while still
//! exercising every code path.

use criterion::{criterion_group, criterion_main, Criterion};
use larql_lql::{parse, Session};
use larql_models::TopKEntry;
use larql_vindex::ndarray::Array2;
use larql_vindex::{ExtractLevel, FeatureMeta, StorageDtype, VectorIndex, VindexConfig};
use std::path::PathBuf;

/// Render a filesystem path for safe inclusion inside an LQL quoted
/// string literal. The LQL lexer decodes `\X` escape sequences, so a
/// raw Windows path like `C:\Users\runner\...` ends up as
/// `C:UsersRUNNER...`. Doubling each backslash leaves the path
/// untouched after the lexer's escape pass on every platform.
fn lql_path(p: impl AsRef<std::path::Path>) -> String {
    p.as_ref().display().to_string().replace('\\', "\\\\")
}

/// Build a synthetic vindex with the SHAPE of a real model (so the byte
/// offsets in `down_weights.bin` are non-trivial) but small enough to
/// compile in a few milliseconds.
fn make_compile_bench_vindex(tag: &str, with_down_weights: bool) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("larql_lql_compile_bench_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let hidden = 64;
    let intermediate = 128;
    let num_layers = 4;
    let vocab_size = 32;

    // Gate vectors per layer.
    let mut gate_layers = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let mut g = Array2::<f32>::zeros((intermediate, hidden));
        for i in 0..intermediate {
            g[[i, i % hidden]] = 1.0;
        }
        gate_layers.push(Some(g));
    }

    // Synthetic feature metas.
    let mut down_meta = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let metas: Vec<Option<FeatureMeta>> = (0..intermediate)
            .map(|i| {
                Some(FeatureMeta {
                    top_token: format!("tok{}_{}", layer, i),
                    top_token_id: (layer * intermediate + i) as u32,
                    c_score: 0.5,
                    top_k: vec![TopKEntry {
                        token: format!("tok{}_{}", layer, i),
                        token_id: (layer * intermediate + i) as u32,
                        logit: 0.5,
                    }],
                })
            })
            .collect();
        down_meta.push(Some(metas));
    }

    let index = VectorIndex::new(gate_layers, down_meta, num_layers, hidden);

    let mut config = VindexConfig {
        version: 2,
        model: "bench/compile".into(),
        family: "llama".into(),
        source: None,
        checksums: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: intermediate,
        vocab_size,
        embed_scale: 1.0,
        extract_level: if with_down_weights {
            ExtractLevel::All
        } else {
            ExtractLevel::Browse
        },
        dtype: StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: 1,
        has_model_weights: with_down_weights,
        model_config: None,
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };
    index.save_vindex(&dir, &mut config).unwrap();

    // Embeddings, tokenizer.
    let embed_bytes = vec![0u8; vocab_size * hidden * 4];
    std::fs::write(dir.join("embeddings.bin"), embed_bytes).unwrap();
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();

    if with_down_weights {
        // Synthetic down_weights.bin: per-layer [hidden, intermediate] f32.
        // The compile bake path will copy this and rewrite override columns.
        let layer_floats = hidden * intermediate;
        let total = num_layers * layer_floats;
        let mut bytes: Vec<u8> = Vec::with_capacity(total * 4);
        for i in 0..total {
            let v = (i as f32) * 0.0001;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(dir.join("down_weights.bin"), &bytes).unwrap();
        // A vindex that declares model weights names its weight files in
        // a manifest; COMPILE refuses one that does not.
        let manifest = serde_json::json!([{
            "key": "bench.down_weights",
            "file": "down_weights.bin",
            "offset": 0,
            "length": bytes.len(),
        }]);
        std::fs::write(dir.join("weight_manifest.json"), manifest.to_string()).unwrap();
    }

    dir
}

/// `COMPILE CURRENT INTO VINDEX` with no patches — pure structural
/// compile (hard-link unchanged files, save gate_vectors.bin from
/// cloned base). This is the lower bound for compile cost.
fn bench_compile_no_patches(c: &mut Criterion) {
    let mut group = c.benchmark_group("compile_into_vindex");
    group.sample_size(20);

    let src_dir = make_compile_bench_vindex("nopatches", false);

    group.bench_function("no_patches_no_weights", |b| {
        let dst = std::env::temp_dir().join("larql_compile_bench_dst_nopatches");
        b.iter(|| {
            let _ = std::fs::remove_dir_all(&dst);
            let mut session = Session::new();
            let use_stmt = parse(&format!(r#"USE "{}";"#, lql_path(&src_dir))).unwrap();
            session.execute(&use_stmt).unwrap();
            let stmt = parse(&format!(
                r#"COMPILE CURRENT INTO VINDEX "{}";"#,
                lql_path(&dst)
            ))
            .unwrap();
            session.execute(&stmt).unwrap();
        });
        let _ = std::fs::remove_dir_all(&dst);
    });

    let _ = std::fs::remove_dir_all(&src_dir);
    group.finish();
}

/// `COMPILE INTO VINDEX` on a vindex that has model weights
/// (`down_weights.bin` present). The first case measures the structural
/// cost with zero overrides; the second injects a single down-vector
/// override so the benchmark exercises the copy + seek-write bake path.
fn bench_compile_with_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("compile_into_vindex");
    group.sample_size(20);

    let src_dir = make_compile_bench_vindex("patched", true);

    group.bench_function("with_weights_no_patches", |b| {
        let dst = std::env::temp_dir().join("larql_compile_bench_dst_patched");
        b.iter(|| {
            let _ = std::fs::remove_dir_all(&dst);
            let mut session = Session::new();
            let use_stmt = parse(&format!(r#"USE "{}";"#, lql_path(&src_dir))).unwrap();
            session.execute(&use_stmt).unwrap();
            let stmt = parse(&format!(
                r#"COMPILE CURRENT INTO VINDEX "{}";"#,
                lql_path(&dst)
            ))
            .unwrap();
            session.execute(&stmt).unwrap();
        });
        let _ = std::fs::remove_dir_all(&dst);
    });

    group.bench_function("with_weights_one_down_override", |b| {
        let dst = std::env::temp_dir().join("larql_compile_bench_dst_one_override");
        b.iter(|| {
            let _ = std::fs::remove_dir_all(&dst);
            let mut session = Session::new();
            let use_stmt = parse(&format!(r#"USE "{}";"#, lql_path(&src_dir))).unwrap();
            session.execute(&use_stmt).unwrap();
            {
                let overlay = session.patched_overlay_mut().expect("vindex backend");
                overlay.insert_feature(
                    0,
                    0,
                    vec![1.0; 64],
                    FeatureMeta {
                        top_token: "patched".into(),
                        top_token_id: 1,
                        c_score: 0.9,
                        top_k: vec![TopKEntry {
                            token: "patched".into(),
                            token_id: 1,
                            logit: 0.9,
                        }],
                    },
                );
                overlay.set_down_vector(0, 0, vec![0.25; 64]);
            }
            let stmt = parse(&format!(
                r#"COMPILE CURRENT INTO VINDEX "{}";"#,
                lql_path(&dst)
            ))
            .unwrap();
            session.execute(&stmt).unwrap();
        });
        let _ = std::fs::remove_dir_all(&dst);
    });

    let _ = std::fs::remove_dir_all(&src_dir);
    group.finish();
}

criterion_group!(
    benches,
    bench_compile_no_patches,
    bench_compile_with_weights,
);
criterion_main!(benches);
