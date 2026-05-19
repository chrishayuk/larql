//! Synthetic-fixture end-to-end test for FP4 row accessors.
//!
//! Unlike `test_fp4_storage.rs` — which requires the real 15 GB
//! gemma3-4b-fp4.vindex on disk — this test builds a minimal FP4
//! vindex in a tempdir (a handful of layers, small hidden) and runs
//! the full load path: `VectorIndex::load_vindex` → `has_fp4_storage`
//! → `ffn_row_dot` / `ffn_row_scaled_add` / `ffn_row_into`.
//!
//! Purpose: provide always-on coverage for the FP4 walk-kernel entry
//! points that doesn't depend on a developer having converted the
//! reference vindex. Complements the real-fixture integration test.

use larql_vindex::format::filenames::*;
use std::path::Path;

use larql_models::quant::fp4_block::BLOCK_ELEMENTS;
use larql_vindex::format::fp4_storage::{write_fp4_projection, write_fp8_projection};
use larql_vindex::{
    ExtractLevel, FfnRowAccess, Fp4Config, SilentLoadCallbacks, StorageDtype, VectorIndex,
    VindexConfig, VindexLayerInfo,
};

/// Minimal tempdir that cleans up on drop.
///
/// `SystemTime::now().as_nanos()` is *not* unique across concurrently
/// scheduled threads on macOS (the clock source can return the same
/// value from two cores). Fold in a process-global atomic counter so
/// parallel-running tests under `cargo test` never share a tempdir.
struct TempDir(std::path::PathBuf);
impl TempDir {
    fn new(label: &str) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let base = std::env::temp_dir();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = base.join(format!(
            "fp4_synth_{label}_{}_{}_{seq}",
            std::process::id(),
            ts
        ));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Produce a flat `[num_features × hidden]` layer of synthetic f32 data.
fn synth_layer(num_features: usize, hidden: usize, seed: f32) -> Vec<f32> {
    (0..num_features * hidden)
        .map(|i| ((i as f32 + seed * 100.0) * 0.017).sin() * 0.5)
        .collect()
}

/// Build an absolutely minimal FP4 vindex on disk:
///   - 3 layers, small hidden (256 → 1 block/feat)
///   - Option B precision tags (gate/up FP4, down FP8)
///   - Index.json with fp4 manifest
///   - down_meta.bin empty stub
///   - tokenizer.json stub
///
/// Returns (tmp, dir, reference_layers_per_projection).
#[allow(clippy::type_complexity)]
fn build_minimal_vindex() -> (
    TempDir,
    std::path::PathBuf,
    Vec<Vec<f32>>, // gate
    Vec<Vec<f32>>, // up
    Vec<Vec<f32>>, // down
    usize,         // hidden
    Vec<usize>,    // per_layer_features
) {
    let tmp = TempDir::new("vindex");
    let dir = tmp.0.clone();
    let hidden = BLOCK_ELEMENTS; // 256
    let per_layer_features = vec![4usize, 8, 6];

    // Synthetic reference data per projection.
    let gate: Vec<Vec<f32>> = per_layer_features
        .iter()
        .enumerate()
        .map(|(i, &n)| synth_layer(n, hidden, i as f32 + 1.0))
        .collect();
    let up: Vec<Vec<f32>> = per_layer_features
        .iter()
        .enumerate()
        .map(|(i, &n)| synth_layer(n, hidden, i as f32 + 10.0))
        .collect();
    let down: Vec<Vec<f32>> = per_layer_features
        .iter()
        .enumerate()
        .map(|(i, &n)| synth_layer(n, hidden, i as f32 + 100.0))
        .collect();

    let gate_refs: Vec<&[f32]> = gate.iter().map(|v| v.as_slice()).collect();
    let up_refs: Vec<&[f32]> = up.iter().map(|v| v.as_slice()).collect();
    let down_refs: Vec<&[f32]> = down.iter().map(|v| v.as_slice()).collect();

    write_fp4_projection(&dir.join(GATE_VECTORS_FP4_BIN), hidden, &gate_refs).unwrap();
    write_fp4_projection(&dir.join(UP_FEATURES_FP4_BIN), hidden, &up_refs).unwrap();
    write_fp8_projection(&dir.join(DOWN_FEATURES_FP8_BIN), hidden, &down_refs).unwrap();

    // Index.json — uses Default derive + FRU.
    let layers: Vec<VindexLayerInfo> = per_layer_features
        .iter()
        .enumerate()
        .map(|(i, &n)| VindexLayerInfo {
            layer: i,
            num_features: n,
            offset: 0,
            length: (n * hidden * 4) as u64,
            ..Default::default()
        })
        .collect();
    let config = VindexConfig {
        version: 2,
        model: "synthetic-fp4".into(),
        family: "synthetic".into(),
        num_layers: per_layer_features.len(),
        hidden_size: hidden,
        intermediate_size: *per_layer_features.iter().max().unwrap(),
        vocab_size: 16,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: StorageDtype::F32,
        layers,
        down_top_k: 1,
        fp4: Some(Fp4Config::option_b_default()),
        ..Default::default()
    };
    let config_json = serde_json::to_string_pretty(&config).unwrap();
    std::fs::write(dir.join("index.json"), config_json).unwrap();

    // Minimal tokenizer + down_meta stubs so the loader doesn't choke.
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();
    let down_meta: Vec<Option<Vec<Option<larql_vindex::FeatureMeta>>>> = per_layer_features
        .iter()
        .map(|&n| Some(vec![None; n]))
        .collect();
    larql_vindex::down_meta::write_binary(&dir, &down_meta, 1).unwrap();

    // A zeroed embeddings.bin so any opportunistic embed reader doesn't
    // trip on a missing file. Size = vocab × hidden × 4.
    std::fs::write(dir.join("embeddings.bin"), vec![0u8; 16 * hidden * 4]).unwrap();

    // Gate_vectors.bin placeholder for any KNN path that looks at it —
    // written as f32 synthetic data (same as `gate` above, concatenated).
    let mut gate_f32: Vec<u8> = Vec::new();
    for layer in &gate {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                layer.as_ptr() as *const u8,
                layer.len() * std::mem::size_of::<f32>(),
            )
        };
        gate_f32.extend_from_slice(bytes);
    }
    std::fs::write(dir.join("gate_vectors.bin"), gate_f32).unwrap();

    (tmp, dir, gate, up, down, hidden, per_layer_features)
}

fn load_minimal(dir: &Path) -> VectorIndex {
    let mut cb = SilentLoadCallbacks;
    VectorIndex::load_vindex(dir, &mut cb).expect("load minimal fp4 vindex")
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[test]
fn minimal_synthetic_vindex_loads_fp4_storage() {
    let (_tmp, dir, _, _, _, _, _) = build_minimal_vindex();
    let index = load_minimal(&dir);
    assert!(index.has_fp4_storage(), "expected FP4 storage attached");
    assert_eq!(index.num_layers, 3);
    assert_eq!(index.hidden_size, 256);
}

#[test]
fn synthetic_ffn_row_dot_uses_fp4_backend() {
    let (_tmp, dir, gate, up, _, hidden, per_layer_features) = build_minimal_vindex();
    let index = load_minimal(&dir);

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.013).cos()).collect();
    let x_view = ndarray::ArrayView1::from(&x);

    // Exercise gate, up across all layers / first-middle-last features.
    for (component, projection) in [(0usize, &gate), (1, &up)] {
        for (layer, layer_values) in projection.iter().enumerate() {
            let n = per_layer_features[layer];
            for feat in [0usize, n / 2, n - 1] {
                let tgt = index
                    .ffn_row_dot(layer, component, feat, &x)
                    .expect("unified dispatch returned None");

                // Source dot for comparison.
                let src_row = &layer_values[feat * hidden..(feat + 1) * hidden];
                let src_view = ndarray::ArrayView1::from(src_row);
                let src_dot = src_view.dot(&x_view);

                let src_norm: f32 = src_view.iter().map(|v| v * v).sum::<f32>().sqrt();
                let x_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
                // FP4 → ~12% per-element; dot error ≤ ~20% of |src|·|x| loose.
                let bound = 0.20 * src_norm * x_norm;
                assert!(
                    (src_dot - tgt).abs() <= bound,
                    "c{component} L{layer} f{feat}: err {} > bound {bound}",
                    (src_dot - tgt).abs()
                );
            }
        }
    }
}

#[test]
fn synthetic_ffn_row_dot_down_uses_fp8_backend() {
    let (_tmp, dir, _, _, down, hidden, per_layer_features) = build_minimal_vindex();
    let index = load_minimal(&dir);

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.021).sin()).collect();
    let x_view = ndarray::ArrayView1::from(&x);

    for (layer, layer_values) in down.iter().enumerate() {
        let n = per_layer_features[layer];
        for feat in [0usize, n / 2, n - 1] {
            let tgt = index
                .ffn_row_dot(layer, 2, feat, &x)
                .expect("down dispatch returned None");

            let src_row = &layer_values[feat * hidden..(feat + 1) * hidden];
            let src_dot = ndarray::ArrayView1::from(src_row).dot(&x_view);

            let src_norm: f32 = src_row.iter().map(|v| v * v).sum::<f32>().sqrt();
            let x_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
            // FP8 ~3–4% per-element → tighter dot bound than FP4.
            let bound = 0.06 * src_norm * x_norm;
            assert!(
                (src_dot - tgt).abs() <= bound,
                "down L{layer} f{feat}: err {} > bound {bound} (src_dot={src_dot:.3e}, tgt={tgt:.3e})",
                (src_dot - tgt).abs()
            );
        }
    }
}

#[test]
fn synthetic_ffn_row_scaled_add_matches_source() {
    let (_tmp, dir, _, _, down, hidden, per_layer_features) = build_minimal_vindex();
    let index = load_minimal(&dir);

    let alpha = 0.375f32;
    let layer = 1;
    let n = per_layer_features[layer];

    for feat in [0usize, n / 2, n - 1] {
        let mut out = vec![0.0f32; hidden];
        assert!(index.ffn_row_scaled_add(layer, 2, feat, alpha, &mut out));

        let src_row = &down[layer][feat * hidden..(feat + 1) * hidden];
        let block_max = src_row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let bound = alpha.abs() * block_max * 0.20;

        for i in 0..hidden {
            let expected = alpha * src_row[i];
            let err = (expected - out[i]).abs();
            assert!(
                err <= bound.max(1e-4),
                "elem {i}: err {err} > bound {bound} (expected {expected}, got {})",
                out[i]
            );
        }
    }
}

#[test]
fn synthetic_ffn_row_into_decodes_correctly() {
    let (_tmp, dir, gate, _, _, hidden, per_layer_features) = build_minimal_vindex();
    let index = load_minimal(&dir);

    let layer = 2;
    let feat = per_layer_features[layer] - 1;
    let mut out = vec![0.0f32; hidden];
    assert!(index.ffn_row_into(layer, 0, feat, &mut out));

    let src_row = &gate[layer][feat * hidden..(feat + 1) * hidden];
    let block_max = src_row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let bound = block_max / 3.0; // FP4 worst-case per-element

    for i in 0..hidden {
        let err = (src_row[i] - out[i]).abs();
        assert!(err <= bound, "elem {i}: err {err} > bound {bound}");
    }
}

#[test]
fn synthetic_ffn_row_returns_none_on_oob() {
    let (_tmp, dir, _, _, _, hidden, per_layer_features) = build_minimal_vindex();
    let index = load_minimal(&dir);
    let x = vec![0.0f32; hidden];

    // Layer out of range.
    assert!(index.ffn_row_dot(99, 0, 0, &x).is_none());
    // Feature out of range.
    assert!(index
        .ffn_row_dot(0, 0, per_layer_features[0] + 100, &x)
        .is_none());
    // Invalid component.
    assert!(index.ffn_row_dot(0, 9, 0, &x).is_none());
}

/// Exp 26 Q2 regression guard: a VectorIndex loaded from an FP4-only
/// vindex directory must report `num_features > 0` per layer. Before
/// the `fp4_storage` fallback in `VectorIndex::num_features`, this
/// returned 0 because the legacy `gate_vectors.bin` was absent — which
/// in turn caused the walk kernel to short-circuit to
/// `zero_features_dense` and silently run on safetensors weights,
/// hiding FP4 quantisation error entirely.
///
/// This test asserts the fallback works at the VectorIndex level; the
/// walk-kernel-level regression guard (routing picks FP4 not
/// `zero_features_dense`) lives in `walk_ffn/routing_tests.rs`
/// and covers the pure predicate logic.
#[test]
fn synthetic_num_features_never_zero_on_fp4_vindex() {
    let (_tmp, dir, _, _, _, _, per_layer_features) = build_minimal_vindex();
    let index = load_minimal(&dir);

    for (layer, &expected) in per_layer_features.iter().enumerate() {
        let got = larql_vindex::GateLookup::num_features(&index, layer);
        assert_eq!(
            got, expected,
            "layer {layer}: num_features returned {got}, expected {expected} — \
             FP4 fallback regression (see VectorIndex::num_features)"
        );
    }
}

#[test]
fn synthetic_cloned_index_preserves_fp4_storage() {
    // Clone invariants test: after cloning a loaded VectorIndex, the
    // clone must still have FP4 storage attached (Arc share) and must
    // produce the same row_dot results as the source.
    let (_tmp, dir, gate, _, _, hidden, _) = build_minimal_vindex();
    let index = load_minimal(&dir);
    let cloned = index.clone();

    assert!(cloned.has_fp4_storage(), "clone lost FP4 storage");

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.005).sin()).collect();
    let src_dot = index.ffn_row_dot(0, 0, 0, &x).unwrap();
    let cln_dot = cloned.ffn_row_dot(0, 0, 0, &x).unwrap();
    // Same backend, same bytes → identical dot.
    assert_eq!(
        src_dot.to_bits(),
        cln_dot.to_bits(),
        "cloned dispatch diverges from source"
    );

    // Sanity: both are within bound of the source.
    let src_row = &gate[0][0..hidden];
    let src_view = ndarray::ArrayView1::from(src_row);
    let src_norm: f32 = src_view.iter().map(|v| v * v).sum::<f32>().sqrt();
    let x_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let true_dot = src_view.dot(&ndarray::ArrayView1::from(&x));
    assert!((true_dot - src_dot).abs() <= 0.20 * src_norm * x_norm);
}
