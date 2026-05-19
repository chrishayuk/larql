//! End-to-end FP4/FP8 storage integration test.
//!
//! Loads the real `gemma3-4b-fp4.vindex` produced by the `fp4_convert`
//! example, and compares `fp4_ffn_row_dot` / `fp4_ffn_row_scaled_add`
//! results against the source `gemma3-4b-f16.vindex` baseline (which
//! stores weights in f32 on disk).
//!
//! The test is guarded on fixture presence — it prints a notice and
//! returns without asserting when the fixture isn't on disk, so CI
//! passes without the 15 GB source vindex being checked out. Run
//! locally after `cargo run --release -p larql-vindex --example
//! fp4_convert ...`.

#![cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

use larql_vindex::{SilentLoadCallbacks, VectorIndex};

const SOURCE: &str = "output/gemma3-4b-f16.vindex";
const TARGET: &str = "output/gemma3-4b-fp4.vindex";

fn fixture_paths() -> Option<(PathBuf, PathBuf)> {
    // Paths are relative to the repo root; cargo runs tests with cwd at
    // the crate root, so walk up two levels.
    let repo_root = std::env::current_dir()
        .ok()?
        .parent()?
        .parent()?
        .to_path_buf();
    let src = repo_root.join(SOURCE);
    let tgt = repo_root.join(TARGET);
    if src.is_dir() && tgt.is_dir() {
        Some((src, tgt))
    } else {
        None
    }
}

/// Read one feature vector from a source vindex (f32 on disk) by direct
/// file access — simpler than loading the whole VectorIndex, keeps the
/// test independent of any potential load-time side effects.
fn read_source_feature(
    vindex_dir: &std::path::Path,
    proj_file: &str,
    layer: usize,
    feat: usize,
    hidden: usize,
    per_layer_features: &[usize],
    dtype: &str,
) -> Vec<f32> {
    let bpf = if dtype == "f32" { 4 } else { 2 };
    let cursor: usize = per_layer_features[..layer].iter().sum::<usize>() * hidden * bpf;
    let offset = cursor + feat * hidden * bpf;
    let bytes = std::fs::read(vindex_dir.join(proj_file)).unwrap();
    let slice = &bytes[offset..offset + hidden * bpf];
    match dtype {
        "f32" => {
            let v: &[f32] =
                unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const f32, hidden) };
            v.to_vec()
        }
        "f16" => larql_models::quant::half::decode_f16(slice),
        "bf16" => larql_models::quant::half::decode_bf16(slice),
        _ => panic!("unsupported dtype {dtype}"),
    }
}

#[test]
fn fp4_storage_loads_from_real_vindex() {
    let Some((src_dir, tgt_dir)) = fixture_paths() else {
        eprintln!("skipping: {TARGET} / {SOURCE} not present on disk");
        return;
    };

    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(&tgt_dir, &mut cb).expect("load fp4 vindex");

    assert!(index.has_fp4_storage(), "fp4 storage should be attached");

    // Sanity — source is expected to load too, but we only need it as
    // a raw-bytes oracle, not as a VectorIndex.
    assert!(src_dir.join("gate_vectors.bin").exists());
}

#[test]
fn fp4_row_dot_matches_source_f32_baseline() {
    let Some((src_dir, tgt_dir)) = fixture_paths() else {
        eprintln!("skipping — fixtures not present");
        return;
    };

    // Load target's config to get hidden, per-layer counts, precision tags.
    let tgt_config_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(tgt_dir.join("index.json")).unwrap())
            .unwrap();
    let src_config_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(src_dir.join("index.json")).unwrap())
            .unwrap();
    let hidden = tgt_config_json["hidden_size"].as_u64().unwrap() as usize;
    let per_layer_features: Vec<usize> = tgt_config_json["layers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l["num_features"].as_u64().unwrap() as usize)
        .collect();
    let src_dtype = src_config_json["dtype"]
        .as_str()
        .unwrap_or("f32")
        .to_string();

    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(&tgt_dir, &mut cb).expect("load");

    // Deterministic pseudo-random x vector.
    let x: Vec<f32> = (0..hidden)
        .map(|i| (i as f32 * 0.137).sin() * 2.0 - 0.3)
        .collect();

    // Per-projection expected tolerances (loose upper bounds measured
    // from fp4_verify on Gemma 3 4B). Normalised by |source| × |x|.
    // The (component, source-file, default-tolerance) trio covers all three
    // projections; per-component precision is read from the manifest below
    // and components stored at source dtype (currently gate under all
    // policies — gate KNN still wants the dense f32 matrix) are skipped:
    // `fp4_ffn_row_dot` returns None for non-FP4/FP8 components.
    let projections: [(usize, &str, f64, f64); 3] = [
        (0, "gate_vectors.bin", 0.04, 0.0001), // fp4 tol vs f32 tol (perfect when source-dtype)
        (1, "up_features.bin", 0.04, 0.0001),
        (2, "down_features.bin", 0.01, 0.0001), // FP8 ~10× tighter
    ];

    let sample_layers = [0usize, 12, 33];
    let sample_feats = [0usize, 1000, 8000];

    let mut all_ok = true;
    for (comp, src_file, fp4_tol, _src_tol) in projections.iter() {
        // Read the component's stored precision from the manifest. f16/f32
        // means the converter linked the source dtype through (gate today)
        // and `fp4_ffn_row_dot` will return None — skip and let the legacy
        // KNN path own that case.
        let prec = tgt_config_json["fp4"]["projections"][match *comp {
            0 => "gate",
            1 => "up",
            _ => "down",
        }]["precision"]
            .as_str()
            .unwrap_or("");
        if prec != "fp4" && prec != "fp8" {
            assert!(
                index
                    .fp4_ffn_row_dot(*sample_layers.first().unwrap(), *comp, 0, &x)
                    .is_none(),
                "component {comp} stored as {prec} should return None from fp4_ffn_row_dot"
            );
            continue;
        }
        let tol_frac = *fp4_tol;
        for &layer in &sample_layers {
            for &feat in &sample_feats {
                if feat >= per_layer_features[layer] {
                    continue;
                }
                let src_row = read_source_feature(
                    &src_dir,
                    src_file,
                    layer,
                    feat,
                    hidden,
                    &per_layer_features,
                    &src_dtype,
                );
                let src_dot: f32 = src_row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

                let tgt_dot = index
                    .fp4_ffn_row_dot(layer, *comp, feat, &x)
                    .expect("fp4 dot should return Some");

                // Tolerance: fraction of |src_row| * |x| (scale-relative).
                let src_norm: f32 = src_row.iter().map(|v| v * v).sum::<f32>().sqrt();
                let x_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
                let bound = (src_norm * x_norm) as f64 * tol_frac;
                let err = (src_dot - tgt_dot).abs() as f64;
                if err > bound {
                    eprintln!(
                        "FAIL c{comp} L{layer} f{feat}: src_dot={src_dot:.5e} tgt_dot={tgt_dot:.5e} \
                         err={err:.3e} bound={bound:.3e} (|src|={src_norm:.3} |x|={x_norm:.3})"
                    );
                    all_ok = false;
                }
            }
        }
    }
    assert!(
        all_ok,
        "FP4 row_dot diverged beyond tolerance; see eprintln output"
    );
}

#[test]
fn fp4_row_scaled_add_matches_source_baseline() {
    let Some((src_dir, tgt_dir)) = fixture_paths() else {
        eprintln!("skipping — fixtures not present");
        return;
    };
    let tgt_config_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(tgt_dir.join("index.json")).unwrap())
            .unwrap();
    let src_config_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(src_dir.join("index.json")).unwrap())
            .unwrap();
    let hidden = tgt_config_json["hidden_size"].as_u64().unwrap() as usize;
    let per_layer_features: Vec<usize> = tgt_config_json["layers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l["num_features"].as_u64().unwrap() as usize)
        .collect();
    let src_dtype = src_config_json["dtype"]
        .as_str()
        .unwrap_or("f32")
        .to_string();

    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(&tgt_dir, &mut cb).expect("load");

    // Component = 2 (down), since that's the one the walk kernel hits
    // with scaled_add (writing back to the residual stream).
    let layer = 15;
    let feat = 2500;
    let alpha = 0.375f32;

    let src_row = read_source_feature(
        &src_dir,
        "down_features.bin",
        layer,
        feat,
        hidden,
        &per_layer_features,
        &src_dtype,
    );

    let mut tgt_out = vec![0.0f32; hidden];
    assert!(index.fp4_ffn_row_scaled_add(layer, 2, feat, alpha, &mut tgt_out));

    // Expected: tgt_out[i] == alpha * src_row[i] (within FP8 quant bound).
    let expected: Vec<f32> = src_row.iter().map(|v| alpha * v).collect();
    let block_max = src_row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let bound = alpha.abs() * block_max * 0.15; // E4M3 per-element worst case.
    for i in 0..hidden {
        let err = (expected[i] - tgt_out[i]).abs();
        assert!(
            err <= bound,
            "elem {i}: err {err} > bound {bound} (exp {} got {})",
            expected[i],
            tgt_out[i]
        );
    }
}

#[test]
fn fp4_storage_absent_on_legacy_vindex() {
    // Sanity: legacy F16/F32 vindex has no fp4 field and storage is None.
    let Some((src_dir, _)) = fixture_paths() else {
        eprintln!("skipping — fixtures not present");
        return;
    };
    let mut cb = SilentLoadCallbacks;
    let legacy = VectorIndex::load_vindex(&src_dir, &mut cb).expect("load legacy");
    assert!(
        !legacy.has_fp4_storage(),
        "legacy f16 vindex must not carry fp4 storage"
    );
}
