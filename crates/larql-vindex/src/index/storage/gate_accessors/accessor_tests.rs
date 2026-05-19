//! Coverage for the read-only accessors: heap + mmap branches of
//! `feature_meta` / `num_features` / `total_*` / `gate_vector` /
//! `gate_vectors_flat` / `loaded_layers` / `down_meta_at` /
//! `gate_vectors_at`, plus `describe_ffn_backend` and `warmup`.

use super::*;
use crate::config::dtype::StorageDtype;
use crate::index::core::VectorIndex;
use crate::index::types::GateLayerSlice;
use larql_models::TopKEntry;
use ndarray::Array2;

fn meta(token: &str) -> FeatureMeta {
    FeatureMeta {
        top_token: token.into(),
        top_token_id: 1,
        c_score: 0.5,
        top_k: vec![TopKEntry {
            token: token.into(),
            token_id: 1,
            logit: 0.5,
        }],
    }
}

/// Build an f16-backed mmap from a flat f32 buffer.
fn f16_mmap_from(floats: &[f32]) -> memmap2::Mmap {
    let bytes = floats.len() * 2;
    let mut anon = memmap2::MmapMut::map_anon(bytes).unwrap();
    let encoded = larql_models::quant::half::encode_f16(floats);
    anon[..bytes].copy_from_slice(&encoded);
    anon.make_read_only().unwrap()
}

// ── feature_meta ──

#[test]
fn feature_meta_returns_none_when_neither_path_populated() {
    let v = VectorIndex::empty(2, 4);
    assert!(v.feature_meta(0, 0).is_none());
}

#[test]
fn feature_meta_uses_heap_path_when_down_meta_populated() {
    let mut v = VectorIndex::empty(2, 4);
    v.metadata.down_meta[0] = Some(vec![Some(meta("Paris")), None]);
    let m = v.feature_meta(0, 0).expect("heap meta present");
    assert_eq!(m.top_token, "Paris");
    // Sibling slot empty → None.
    assert!(v.feature_meta(0, 1).is_none());
}

#[test]
fn feature_meta_returns_none_for_oob_layer() {
    let v = VectorIndex::empty(2, 4);
    assert!(v.feature_meta(99, 0).is_none());
}

// ── num_features ──

#[test]
fn num_features_returns_zero_for_empty_index() {
    let v = VectorIndex::empty(2, 4);
    for layer in 0..2 {
        assert_eq!(v.num_features(layer), 0);
    }
}

#[test]
fn num_features_reads_heap_gate_shape() {
    let mut v = VectorIndex::empty(2, 4);
    v.gate.gate_vectors[0] = Some(Array2::<f32>::zeros((7, 4)));
    v.gate.gate_vectors[1] = Some(Array2::<f32>::zeros((3, 4)));
    assert_eq!(v.num_features(0), 7);
    assert_eq!(v.num_features(1), 3);
}

#[test]
fn num_features_reads_mmap_slices() {
    let floats = vec![1.0_f32; 8];
    let mmap = f16_mmap_from(&floats);
    let slices = vec![
        GateLayerSlice {
            float_offset: 0,
            num_features: 4,
        },
        GateLayerSlice {
            float_offset: 16,
            num_features: 0,
        },
    ];
    let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 2, 4);
    assert_eq!(v.num_features(0), 4);
    // Slice with num_features = 0 falls through to fp4 (None) → 0.
    assert_eq!(v.num_features(1), 0);
}

#[test]
fn num_features_oob_layer_returns_zero() {
    let v = VectorIndex::empty(2, 4);
    assert_eq!(v.num_features(99), 0);
}

// ── num_features Q4K-width fallback ──
//
// Regression guard for the 100× decode regression on Gemma 3 4B Q4K:
// a vindex with only Q4K FFN bytes (no `gate_vectors.bin`, no FP4)
// must surface a non-zero intermediate width via `num_features` so
// the WalkFfn ladder doesn't drop to the dense f32 fallback and
// `kquant_matmul_transb` doesn't return zero-row matrices.

/// Build a synthetic Q4_K interleaved manifest with the given
/// intermediate width on every layer. Byte payloads are zero — content
/// is irrelevant for the width derivation (`num_features` reads byte
/// *length* from the manifest, not the encoded scales).
fn install_synthetic_q4k(v: &mut VectorIndex, intermediate: usize) {
    use crate::index::storage::ffn_store::FFN_COMPONENTS_PER_LAYER;
    use crate::quant::registry::lookup;
    let hidden = v.hidden_size;
    let bytes_per_row = lookup("Q4_K")
        .expect("Q4_K registered")
        .bytes_per_row(hidden)
        .expect("hidden is block-aligned");
    let gate_up_bytes = intermediate * bytes_per_row;
    let down_bytes = hidden * bytes_per_row;
    let per_layer = gate_up_bytes * 2 + down_bytes;
    let total = per_layer * v.num_layers;

    let payload = vec![0u8; total];
    let mut manifest = Vec::with_capacity(v.num_layers * FFN_COMPONENTS_PER_LAYER);
    let mut offset = 0;
    for _ in 0..v.num_layers {
        manifest.push((offset, gate_up_bytes, "Q4_K".to_string()));
        offset += gate_up_bytes;
        manifest.push((offset, gate_up_bytes, "Q4_K".to_string()));
        offset += gate_up_bytes;
        manifest.push((offset, down_bytes, "Q4_K".to_string()));
        offset += down_bytes;
    }

    let mut anon = memmap2::MmapMut::map_anon(total).unwrap();
    anon.copy_from_slice(&payload);
    let mmap = std::sync::Arc::new(anon.make_read_only().unwrap());
    let storage = std::sync::Arc::make_mut(&mut v.storage);
    storage.set_interleaved_kquant(mmap, Some(manifest));
}

#[test]
fn num_features_falls_back_to_q4k_width() {
    let mut v = VectorIndex::empty(2, 256);
    install_synthetic_q4k(&mut v, 768);
    assert_eq!(v.num_features(0), 768);
    assert_eq!(v.num_features(1), 768);
}

#[test]
fn num_features_q4k_fallback_returns_zero_for_oob_layer() {
    let mut v = VectorIndex::empty(2, 256);
    install_synthetic_q4k(&mut v, 768);
    assert_eq!(v.num_features(99), 0);
}

#[test]
fn num_features_legacy_gate_wins_over_q4k_fallback() {
    // Q4K width says 768 but heap gate vectors say 5 — heap wins (it's
    // the legacy KNN feature count, takes priority).
    let mut v = VectorIndex::empty(2, 256);
    v.gate.gate_vectors[0] = Some(Array2::<f32>::zeros((5, 256)));
    install_synthetic_q4k(&mut v, 768);
    assert_eq!(v.num_features(0), 5, "heap gate vectors take priority");
    // Layer without heap gate falls through to Q4K width.
    assert_eq!(v.num_features(1), 768);
}

#[test]
fn q4k_ffn_intermediate_width_returns_some_for_real_manifest() {
    let mut v = VectorIndex::empty(3, 256);
    install_synthetic_q4k(&mut v, 1024);
    for layer in 0..3 {
        assert_eq!(v.kquant_ffn_intermediate_width(layer), Some(1024));
    }
}

#[test]
fn q4k_ffn_intermediate_width_none_when_no_manifest() {
    let v = VectorIndex::empty(2, 256);
    assert!(v.kquant_ffn_intermediate_width(0).is_none());
}

#[test]
fn q4k_ffn_intermediate_width_none_on_unknown_format() {
    // Manifest claims format "QX_K" — not in the registry. `lookup` must
    // return None so the fallback skips this layer rather than guessing
    // a stride. Without this guard a future format tag could silently
    // produce wrong widths.
    use crate::index::storage::ffn_store::FFN_COMPONENTS_PER_LAYER;
    let mut v = VectorIndex::empty(1, 256);
    let total = FFN_COMPONENTS_PER_LAYER * 256; // any nonzero size
    let mut anon = memmap2::MmapMut::map_anon(total).unwrap();
    anon[..total].fill(0);
    let mmap = std::sync::Arc::new(anon.make_read_only().unwrap());
    let manifest = vec![
        (0, 256, "QX_K".to_string()),
        (256, 256, "QX_K".to_string()),
        (512, 256, "QX_K".to_string()),
    ];
    let storage = std::sync::Arc::make_mut(&mut v.storage);
    storage.set_interleaved_kquant(mmap, Some(manifest));
    assert!(v.kquant_ffn_intermediate_width(0).is_none());
}

#[test]
fn q4k_ffn_intermediate_width_none_when_bytes_not_a_whole_row() {
    // Gate byte length isn't a multiple of `bytes_per_row(hidden)`.
    // The fallback must refuse to round — silent rounding would hide a
    // corrupt or stale manifest.
    use crate::index::storage::ffn_store::FFN_COMPONENTS_PER_LAYER;
    use crate::quant::registry::lookup;
    let mut v = VectorIndex::empty(1, 256);
    let bytes_per_row = lookup("Q4_K").unwrap().bytes_per_row(256).unwrap();
    // Truncate by one byte → not a whole row.
    let gate_len = bytes_per_row * 3 - 1;
    let up_len = bytes_per_row * 3;
    let down_len = bytes_per_row * 256;
    let total = gate_len + up_len + down_len;
    let mut anon = memmap2::MmapMut::map_anon(total).unwrap();
    anon[..total].fill(0);
    let mmap = std::sync::Arc::new(anon.make_read_only().unwrap());
    let mut manifest = Vec::with_capacity(FFN_COMPONENTS_PER_LAYER);
    let mut offset = 0;
    manifest.push((offset, gate_len, "Q4_K".to_string()));
    offset += gate_len;
    manifest.push((offset, up_len, "Q4_K".to_string()));
    offset += up_len;
    manifest.push((offset, down_len, "Q4_K".to_string()));
    let storage = std::sync::Arc::make_mut(&mut v.storage);
    storage.set_interleaved_kquant(mmap, Some(manifest));
    assert!(v.kquant_ffn_intermediate_width(0).is_none());
}

#[test]
fn num_features_q4k_fallback_real_gemma_test_fixture() {
    // The synthetic-Q4K test fixture in `larql-inference::test_utils` is
    // the closest thing to a real Gemma 3 4B Q4K vindex shape: no
    // `gate_vectors.bin`, no FP4 storage, only Q4K FFN bytes. Before the
    // Q4K width fallback this returned 0 and WalkFfn fell through to
    // dense f32 matmul — the 100× decode regression. Build the same
    // shape inline (avoids a cross-crate test dep) and assert
    // `num_features` reports the real intermediate width.
    use crate::index::storage::ffn_store::FFN_COMPONENTS_PER_LAYER;
    use crate::quant::registry::lookup;
    let hidden = 256;
    let intermediate = 256;
    let num_layers = 2;
    let mut v = VectorIndex::empty(num_layers, hidden);
    let bytes_per_row = lookup("Q4_K").unwrap().bytes_per_row(hidden).unwrap();
    let gate_up = intermediate * bytes_per_row;
    let down = hidden * bytes_per_row;
    let per_layer = gate_up * 2 + down;
    let total = per_layer * num_layers;
    let mut anon = memmap2::MmapMut::map_anon(total).unwrap();
    anon[..total].fill(0);
    let mmap = std::sync::Arc::new(anon.make_read_only().unwrap());
    let mut manifest = Vec::with_capacity(num_layers * FFN_COMPONENTS_PER_LAYER);
    let mut offset = 0;
    for _ in 0..num_layers {
        manifest.push((offset, gate_up, "Q4_K".to_string()));
        offset += gate_up;
        manifest.push((offset, gate_up, "Q4_K".to_string()));
        offset += gate_up;
        manifest.push((offset, down, "Q4_K".to_string()));
        offset += down;
    }
    let storage = std::sync::Arc::make_mut(&mut v.storage);
    storage.set_interleaved_kquant(mmap, Some(manifest));
    for layer in 0..num_layers {
        assert_eq!(
            v.num_features(layer),
            intermediate,
            "Q4K-only vindex must surface intermediate width — \
             regressing this re-introduces the 100× FFN slowdown"
        );
    }
}

// ── total_gate_vectors / total_down_meta ──

#[test]
fn total_gate_vectors_sums_heap_layers() {
    let mut v = VectorIndex::empty(3, 4);
    v.gate.gate_vectors[0] = Some(Array2::<f32>::zeros((5, 4)));
    v.gate.gate_vectors[2] = Some(Array2::<f32>::zeros((7, 4)));
    assert_eq!(v.total_gate_vectors(), 12);
}

#[test]
fn total_gate_vectors_sums_mmap_slices() {
    let floats = vec![1.0_f32; 16];
    let mmap = f16_mmap_from(&floats);
    let slices = vec![
        GateLayerSlice {
            float_offset: 0,
            num_features: 2,
        },
        GateLayerSlice {
            float_offset: 8,
            num_features: 2,
        },
    ];
    let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 2, 4);
    assert_eq!(v.total_gate_vectors(), 4);
}

#[test]
fn total_down_meta_counts_heap_metas() {
    let mut v = VectorIndex::empty(3, 4);
    v.metadata.down_meta[0] = Some(vec![Some(meta("a")), None, Some(meta("b"))]);
    v.metadata.down_meta[2] = Some(vec![Some(meta("c"))]);
    assert_eq!(v.total_down_meta(), 3);
}

#[test]
fn total_down_meta_zero_when_empty() {
    let v = VectorIndex::empty(2, 4);
    assert_eq!(v.total_down_meta(), 0);
}

// ── loaded_layers ──

#[test]
fn loaded_layers_returns_indices_with_heap_gate() {
    let mut v = VectorIndex::empty(4, 4);
    v.gate.gate_vectors[0] = Some(Array2::<f32>::zeros((2, 4)));
    v.gate.gate_vectors[2] = Some(Array2::<f32>::zeros((2, 4)));
    v.gate.gate_vectors[3] = Some(Array2::<f32>::zeros((2, 4)));
    assert_eq!(v.loaded_layers(), vec![0, 2, 3]);
}

#[test]
fn loaded_layers_filters_zero_feature_mmap_slices() {
    let floats = vec![1.0_f32; 16];
    let mmap = f16_mmap_from(&floats);
    let slices = vec![
        GateLayerSlice {
            float_offset: 0,
            num_features: 2,
        },
        GateLayerSlice {
            float_offset: 0,
            num_features: 0,
        }, // empty layer
        GateLayerSlice {
            float_offset: 8,
            num_features: 2,
        },
    ];
    let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 3, 4);
    assert_eq!(v.loaded_layers(), vec![0, 2]);
}

#[test]
fn loaded_layers_empty_when_nothing_loaded() {
    let v = VectorIndex::empty(3, 4);
    assert!(v.loaded_layers().is_empty());
}

// ── down_meta_at / gate_vectors_at ──

#[test]
fn down_meta_at_returns_layer_slice() {
    let mut v = VectorIndex::empty(2, 4);
    v.metadata.down_meta[1] = Some(vec![Some(meta("x"))]);
    assert!(v.down_meta_at(0).is_none());
    let slice = v.down_meta_at(1).expect("layer 1 present");
    assert_eq!(slice.len(), 1);
}

#[test]
fn gate_vectors_at_returns_matrix_only_in_heap_mode() {
    let mut v = VectorIndex::empty(2, 4);
    v.gate.gate_vectors[0] = Some(Array2::<f32>::zeros((2, 4)));
    assert_eq!(v.gate_vectors_at(0).unwrap().shape(), &[2, 4]);
    assert!(v.gate_vectors_at(1).is_none());
    assert!(v.gate_vectors_at(99).is_none());
}

// ── gate_vector ──

#[test]
fn gate_vector_heap_returns_row() {
    let mut v = VectorIndex::empty(1, 4);
    let mut m = Array2::<f32>::zeros((3, 4));
    for j in 0..4 {
        m[[1, j]] = (j + 10) as f32;
    }
    v.gate.gate_vectors[0] = Some(m);
    let row = v.gate_vector(0, 1).unwrap();
    assert_eq!(row, vec![10.0, 11.0, 12.0, 13.0]);
}

#[test]
fn gate_vector_heap_returns_none_for_oob_feature() {
    let mut v = VectorIndex::empty(1, 4);
    v.gate.gate_vectors[0] = Some(Array2::<f32>::zeros((2, 4)));
    assert!(v.gate_vector(0, 99).is_none());
}

#[test]
fn gate_vector_returns_none_when_nothing_loaded() {
    let v = VectorIndex::empty(2, 4);
    assert!(v.gate_vector(0, 0).is_none());
}

#[test]
fn gate_vector_mmap_returns_decoded_floats() {
    let floats = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 features × 4 hidden
    let mmap = f16_mmap_from(&floats);
    let slices = vec![GateLayerSlice {
        float_offset: 0,
        num_features: 2,
    }];
    let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 1, 4);
    let row = v.gate_vector(0, 1).unwrap();
    // f16 round-trip is lossy for 5..8 but they fit exactly.
    assert!((row[0] - 5.0).abs() < 1e-3);
    assert!((row[3] - 8.0).abs() < 1e-3);
}

#[test]
fn gate_vector_mmap_returns_none_for_oob_feature() {
    let floats = vec![1.0_f32; 8];
    let mmap = f16_mmap_from(&floats);
    let slices = vec![GateLayerSlice {
        float_offset: 0,
        num_features: 2,
    }];
    let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 1, 4);
    assert!(v.gate_vector(0, 99).is_none());
}

// ── gate_vectors_flat ──

#[test]
fn gate_vectors_flat_heap_returns_data_rows_cols() {
    let mut v = VectorIndex::empty(1, 4);
    let mut m = Array2::<f32>::zeros((2, 4));
    for r in 0..2 {
        for j in 0..4 {
            m[[r, j]] = (r * 10 + j) as f32;
        }
    }
    v.gate.gate_vectors[0] = Some(m);
    let (data, rows, cols) = v.gate_vectors_flat(0).unwrap();
    assert_eq!(rows, 2);
    assert_eq!(cols, 4);
    assert_eq!(data, vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]);
}

#[test]
fn gate_vectors_flat_returns_none_for_unloaded_layer() {
    let v = VectorIndex::empty(2, 4);
    assert!(v.gate_vectors_flat(0).is_none());
}

#[test]
fn gate_vectors_flat_mmap_returns_decoded_layer() {
    let floats = vec![1.0_f32; 8];
    let mmap = f16_mmap_from(&floats);
    let slices = vec![GateLayerSlice {
        float_offset: 0,
        num_features: 2,
    }];
    let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 1, 4);
    let (data, rows, cols) = v.gate_vectors_flat(0).unwrap();
    assert_eq!(rows, 2);
    assert_eq!(cols, 4);
    assert_eq!(data.len(), 8);
}

#[test]
fn gate_vectors_flat_mmap_returns_none_when_zero_features() {
    let floats = vec![1.0_f32; 4];
    let mmap = f16_mmap_from(&floats);
    let slices = vec![GateLayerSlice {
        float_offset: 0,
        num_features: 0,
    }];
    let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 1, 4);
    assert!(v.gate_vectors_flat(0).is_none());
}

// ── num_features_at ──

#[test]
fn num_features_at_heap_path_matches_num_features() {
    let mut v = VectorIndex::empty(2, 4);
    v.gate.gate_vectors[0] = Some(Array2::<f32>::zeros((6, 4)));
    assert_eq!(v.num_features_at(0), 6);
    assert_eq!(v.num_features_at(1), 0);
}

#[test]
fn num_features_at_mmap_path_uses_slice_count() {
    let floats = vec![1.0_f32; 16];
    let mmap = f16_mmap_from(&floats);
    let slices = vec![
        GateLayerSlice {
            float_offset: 0,
            num_features: 4,
        },
        GateLayerSlice {
            float_offset: 16,
            num_features: 0,
        },
    ];
    let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 2, 4);
    assert_eq!(v.num_features_at(0), 4);
    assert_eq!(v.num_features_at(1), 0);
    // OOB layer → 0.
    assert_eq!(v.num_features_at(99), 0);
}

// ── describe_ffn_backend ──

#[test]
fn describe_ffn_backend_reports_weights_fallback_when_empty() {
    let v = VectorIndex::empty(1, 4);
    let s = v.describe_ffn_backend();
    assert!(s.contains("weights fallback"), "got: {s}");
}

#[test]
fn describe_ffn_backend_reports_gate_mmap_dtype() {
    let floats = vec![1.0_f32; 4];
    let mmap = f16_mmap_from(&floats);
    let slices = vec![GateLayerSlice {
        float_offset: 0,
        num_features: 1,
    }];
    let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 1, 4);
    let s = v.describe_ffn_backend();
    assert!(s.contains("gate KNN"), "got: {s}");
    assert!(s.contains("F16"), "got: {s}");
}

// ── warmup ──

#[test]
fn warmup_is_noop_for_f32_mmap() {
    // f32 path returns immediately — warmed_gates stays empty.
    let bytes = 16; // 4 floats × 4 bytes
    let anon = memmap2::MmapMut::map_anon(bytes).unwrap();
    let mmap = anon.make_read_only().unwrap();
    let slices = vec![GateLayerSlice {
        float_offset: 0,
        num_features: 1,
    }];
    let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F32, None, 1, 4);
    v.warmup();
    let warmed = v.gate.warmed_gates.read().unwrap();
    assert!(warmed.iter().all(|s| s.is_none()), "f32 path no-ops");
}

#[test]
fn warmup_decodes_f16_into_warmed_gates() {
    let floats = vec![1.0_f32, 2.0, 3.0, 4.0]; // 1 feature × 4 hidden
    let mmap = f16_mmap_from(&floats);
    let slices = vec![GateLayerSlice {
        float_offset: 0,
        num_features: 1,
    }];
    let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 1, 4);
    v.warmup();
    let warmed = v.gate.warmed_gates.read().unwrap();
    let layer0 = warmed[0].as_ref().expect("layer 0 warmed");
    assert_eq!(layer0.len(), 4);
    for (i, want) in [1.0_f32, 2.0, 3.0, 4.0].iter().enumerate() {
        assert!((layer0[i] - want).abs() < 1e-3, "f16 round-trip");
    }
}

#[test]
fn warmup_skips_zero_feature_layers() {
    let floats = vec![1.0_f32, 2.0, 3.0, 4.0];
    let mmap = f16_mmap_from(&floats);
    let slices = vec![
        GateLayerSlice {
            float_offset: 0,
            num_features: 1,
        },
        GateLayerSlice {
            float_offset: 0,
            num_features: 0,
        },
    ];
    let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 2, 4);
    v.warmup();
    let warmed = v.gate.warmed_gates.read().unwrap();
    assert!(warmed[0].is_some());
    assert!(warmed[1].is_none(), "empty layer left None");
}

#[test]
fn warmup_is_idempotent() {
    let floats = vec![1.0_f32; 4];
    let mmap = f16_mmap_from(&floats);
    let slices = vec![GateLayerSlice {
        float_offset: 0,
        num_features: 1,
    }];
    let v = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 1, 4);
    v.warmup();
    v.warmup(); // second call short-circuits per layer
    let warmed = v.gate.warmed_gates.read().unwrap();
    assert!(warmed[0].is_some());
}

#[test]
fn warmup_no_op_without_mmap() {
    // Heap-only index — no gate mmap → early return regardless
    // of dtype. After step 6 the dtype lives on `MmapStorage`,
    // not the substore; an empty storage stays at the F32 default
    // and the warmup early-returns on the dtype check anyway.
    let v = VectorIndex::empty(1, 4);
    v.warmup();
    let warmed = v.gate.warmed_gates.read().unwrap();
    assert!(warmed.iter().all(|s| s.is_none()));
}
