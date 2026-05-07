//! FFN storage — mmap loaders, accessors, prefetchers, and the
//! Q4_K/Q6_K dequant cache. Compute-side codec dispatch (matmul +
//! row-level fused decode) lives in
//! `crate::index::compute::q4k_dispatch`.
//!
//! Files managed (split-by-variant, M7 cleanup 2026-05-01):
//! - `down.rs`             — `down_features.bin` (feature-major f32)
//! - `up.rs`               — `up_features.bin` (feature-major f32)
//! - `interleaved.rs`      — `interleaved.bin` (f32 [gate|up|down])
//! - `interleaved_q4.rs`   — `interleaved_q4.bin` (Q4_0)
//! - `interleaved_q4k.rs`  — `interleaved_q4k.bin` + manifests +
//!                           `down_features_q4k.bin` (Q4_K/Q6_K)
//! - `gate_q4.rs`          — Q4_0 gate-vector mmap (KNN side-channel)
//! - `fp4.rs`              — FP4 / FP8 FFN storage (exp 26)
//! - `q4k_cache.rs`        — bounded LRU dequant cache (`q4k_ffn_cache`)
//!
//! `FfnStore` lives here as the composed substore on `VectorIndex`,
//! along with `ffn_layer_byte_offset` — the prefix-sum every f32 / Q4
//! accessor uses to translate `layer` to a byte offset (correct under
//! variable per-layer feature counts; collapses to `layer * size` for
//! constant dense models).
//!
//! The cache (`q4k_ffn_cache`) is bounded by
//! `set_q4k_ffn_cache_max_layers`; only the CPU per-position fallback
//! populates it (Metal full-K decode streams Q4_K bytes through
//! `compute::q4k_dispatch::q4k_matmul_transb`).

use std::sync::{Arc, Mutex};

use crate::index::core::VectorIndex;

mod down;
mod fp4;
mod gate_q4;
mod interleaved;
mod interleaved_q4;
mod interleaved_q4k;
mod q4k_cache;
mod up;

// ── FfnStore composed-substore ─────────────────────────────────────────

/// Per-layer Q4_K/Q6_K FFN dequant cache: outer index = layer, inner array =
/// `[gate, up, down]`. `Arc` shares the decoded matrix across `VectorIndex`
/// clones; `Mutex` guards LRU eviction.
pub type Q4kFfnCache = Mutex<Vec<[Option<Arc<Vec<f32>>>; 3]>>;

/// Per-(layer, component) lock-free dequant slot. `OnceLock` initialises at
/// most once over the process lifetime; the `Option` distinguishes "decoded
/// successfully but the matrix is empty" from "not yet decoded".
pub type Q4kFfnSlot = std::sync::OnceLock<Option<Arc<Vec<f32>>>>;

/// Per-layer manifest entry for `down_features_q4k.bin` (W2). Carries
/// the padded row width so the row decoder doesn't have to back-derive
/// it from `length / n_features`.
#[derive(Clone, Debug)]
pub struct DownFeaturesQ4kEntry {
    pub offset: usize,
    pub length: usize,
    pub format: String,
    /// Row stride in elements after `pad_rows_to_block`. For production
    /// models this equals `hidden_size`; preserved literally so the
    /// decoder can dequant `padded_width` floats per feature and the
    /// caller takes the first `hidden_size` of them.
    pub padded_width: usize,
}

pub struct FfnStore {
    /// Feature-major down projections (f32 mmap).
    pub down_features_mmap: Option<Arc<memmap2::Mmap>>,
    /// Feature-major Q4_K-encoded down projections — W2 of perf round-4.
    /// When present, lets per-feature down decode skip the
    /// `q4k_ffn_layer` cache (which dequants the whole layer). See
    /// `DOWN_FEATURES_Q4K_BIN` for the rationale.
    pub down_features_q4k_mmap: Option<Arc<memmap2::Mmap>>,
    /// Per-layer entries for `down_features_q4k_mmap`. One entry per
    /// layer (vs three for the interleaved manifest). `padded_width`
    /// is the row stride after `pad_rows_to_block` — usually equal to
    /// `hidden_size`, but on synthetic fixtures with `hidden % 256 != 0`
    /// it's the next 256-multiple. Carrying it in the manifest avoids
    /// rederiving it from `length` at every row decode.
    pub down_features_q4k_manifest: Option<Vec<DownFeaturesQ4kEntry>>,
    /// Feature-major up projections (f32 mmap).
    pub up_features_mmap: Option<Arc<memmap2::Mmap>>,
    /// Interleaved [gate|up|down] FFN data (f32, packed per layer).
    pub interleaved_mmap: Option<Arc<memmap2::Mmap>>,
    /// Q4_0 quantized interleaved FFN.
    pub interleaved_q4_mmap: Option<Arc<memmap2::Mmap>>,
    /// Q4_K / Q6_K quantized interleaved FFN (Ollama-compatible).
    pub interleaved_q4k_mmap: Option<Arc<memmap2::Mmap>>,
    /// Per-matrix (offset, length, format) entries — 3 per layer in
    /// `[gate, up, down]` order.
    pub interleaved_q4k_manifest: Option<Vec<(usize, usize, String)>>,
    /// Per-layer lazy dequant cache for Q4_K/Q6_K FFN tensors.
    /// `q4k_ffn_cache[layer][c]` is the dequantised
    /// `[intermediate × hidden]` matrix for component `c`
    /// (0=gate, 1=up, 2=down). LRU-bounded by
    /// `q4k_ffn_cache_max_layers`.
    pub q4k_ffn_cache: Q4kFfnCache,
    /// LRU of layers held in `q4k_ffn_cache`. Front = newest.
    pub q4k_ffn_cache_lru: Mutex<std::collections::VecDeque<usize>>,
    /// Cap on `q4k_ffn_cache`. 0 = unlimited (default).
    pub q4k_ffn_cache_max_layers: std::sync::atomic::AtomicUsize,
    /// Lock-free per-slot dequant cache for the parallel-batch server path.
    ///
    /// `q4k_ffn_once[layer][c]` is populated at most once per process
    /// lifetime via `OnceLock::get_or_init`.  After the first call for a
    /// given (layer, component) all reads are a single atomic load + Arc
    /// clone — no mutex, no LRU, no contention across rayon workers.
    ///
    /// Memory cost (31B, all 60 layers, all 3 components):
    ///   60 × 3 × (intermediate × hidden × 4 bytes) ≈ 60 × 3 × 462 MB ≈ 83 GB f32.
    /// In practice only the down component (component=2) is fetched from
    /// this cache; gate/up use the NEON Q4K×Q8K kernel directly on mmap
    /// bytes and never populate their slots here.
    pub q4k_ffn_once: Vec<[Q4kFfnSlot; 3]>,
    /// FP4 / FP8 FFN storage (exp 26).
    pub fp4_storage: Option<Arc<crate::index::fp4_storage::Fp4Storage>>,
}

impl FfnStore {
    pub fn empty(num_layers: usize) -> Self {
        Self {
            down_features_mmap: None,
            down_features_q4k_mmap: None,
            down_features_q4k_manifest: None,
            up_features_mmap: None,
            interleaved_mmap: None,
            interleaved_q4_mmap: None,
            interleaved_q4k_mmap: None,
            interleaved_q4k_manifest: None,
            q4k_ffn_cache: Mutex::new((0..num_layers).map(|_| [None, None, None]).collect()),
            q4k_ffn_cache_lru: Mutex::new(std::collections::VecDeque::new()),
            q4k_ffn_cache_max_layers: std::sync::atomic::AtomicUsize::new(0),
            q4k_ffn_once: (0..num_layers)
                .map(|_| std::array::from_fn(|_| std::sync::OnceLock::new()))
                .collect(),
            fp4_storage: None,
        }
    }
}

impl Clone for FfnStore {
    fn clone(&self) -> Self {
        use std::sync::atomic::Ordering;
        let nl = self.q4k_ffn_cache.lock().map(|c| c.len()).unwrap_or(0);
        Self {
            down_features_mmap: self.down_features_mmap.clone(),
            down_features_q4k_mmap: self.down_features_q4k_mmap.clone(),
            down_features_q4k_manifest: self.down_features_q4k_manifest.clone(),
            up_features_mmap: self.up_features_mmap.clone(),
            interleaved_mmap: self.interleaved_mmap.clone(),
            interleaved_q4_mmap: self.interleaved_q4_mmap.clone(),
            interleaved_q4k_mmap: self.interleaved_q4k_mmap.clone(),
            interleaved_q4k_manifest: self.interleaved_q4k_manifest.clone(),
            q4k_ffn_cache: Mutex::new((0..nl).map(|_| [None, None, None]).collect()),
            q4k_ffn_cache_lru: Mutex::new(std::collections::VecDeque::new()),
            q4k_ffn_cache_max_layers: std::sync::atomic::AtomicUsize::new(
                self.q4k_ffn_cache_max_layers.load(Ordering::Relaxed),
            ),
            q4k_ffn_once: (0..nl)
                .map(|_| std::array::from_fn(|_| std::sync::OnceLock::new()))
                .collect(),
            fp4_storage: self.fp4_storage.clone(),
        }
    }
}

impl VectorIndex {
    /// Byte offset where layer `layer` starts in a packed per-layer f32
    /// FFN file. `matrices_per_layer` = 1 for feature-major files
    /// (`down_features.bin`, `up_features.bin`) and 3 for the interleaved
    /// `[gate|up|down]` file. Computed as a prefix sum over
    /// `num_features(l) * hidden_size` rather than `layer * intermediate`
    /// — the latter is wrong when `layers[].num_features` varies (MoE
    /// shards with per-layer expert counts), and the prefix sum collapses
    /// to the same value for constant-feature dense models.
    pub(super) fn ffn_layer_byte_offset(&self, layer: usize, matrices_per_layer: usize) -> usize {
        let mut floats: usize = 0;
        for l in 0..layer {
            floats += self.num_features(l) * self.hidden_size;
        }
        floats * 4 * matrices_per_layer
    }
}

#[cfg(test)]
mod ffn_layer_byte_offset_tests {
    //! `ffn_layer_byte_offset` is the load-bearing prefix-sum that lets
    //! the legacy f32 FFN accessors handle layouts where
    //! `layers[].num_features` varies (MoE shards). Pre-fix it was
    //! `layer * num_features(layer)`, which silently mis-addressed every
    //! layer past the first whenever feature counts weren't constant.

    use crate::index::core::VectorIndex;
    use ndarray::Array2;

    /// Build an in-memory VectorIndex whose `num_features(layer)` reads
    /// from the heap gate-vectors fallback (no mmap needed). Each gate
    /// matrix has shape `[num_features[l], hidden]`.
    fn index_with_layers(num_features: &[usize], hidden: usize) -> VectorIndex {
        let gate_vectors: Vec<Option<Array2<f32>>> = num_features
            .iter()
            .map(|&n| Some(Array2::zeros((n, hidden))))
            .collect();
        let down_meta = vec![None; num_features.len()];
        VectorIndex::new(gate_vectors, down_meta, num_features.len(), hidden)
    }

    #[test]
    fn constant_features_collapses_to_layer_times_size() {
        // Dense path: every layer has the same num_features. The prefix
        // sum equals `layer * num_features * hidden * 4 * mults`, so
        // existing dense vindex files keep their byte layout.
        let v = index_with_layers(&[8, 8, 8, 8], 4);
        for layer in 0..4 {
            for mults in [1, 3] {
                let expected = layer * 8 * 4 * 4 * mults;
                assert_eq!(
                    v.ffn_layer_byte_offset(layer, mults),
                    expected,
                    "layer={layer} mults={mults}"
                );
            }
        }
    }

    #[test]
    fn variable_features_uses_prefix_sum() {
        // MoE path: feature counts differ per layer. Layer L starts at
        // `sum_{l<L} num_features(l) * hidden * 4 * mults` — *not*
        // `L * num_features(L) * hidden * 4 * mults`. Pre-fix code
        // computed the latter and silently mis-addressed L1+.
        let v = index_with_layers(&[10, 20, 30], 4);

        // mults=1 (down_features.bin, up_features.bin):
        // L0 → 0
        // L1 → 10*4*4 = 160
        // L2 → (10+20)*4*4 = 480
        assert_eq!(v.ffn_layer_byte_offset(0, 1), 0);
        assert_eq!(v.ffn_layer_byte_offset(1, 1), 160);
        assert_eq!(v.ffn_layer_byte_offset(2, 1), 480);

        // mults=3 (interleaved.bin, gate+up+down per layer):
        // L0 → 0
        // L1 → 10*4*4*3 = 480
        // L2 → (10+20)*4*4*3 = 1440
        assert_eq!(v.ffn_layer_byte_offset(0, 3), 0);
        assert_eq!(v.ffn_layer_byte_offset(1, 3), 480);
        assert_eq!(v.ffn_layer_byte_offset(2, 3), 1440);
    }

    #[test]
    fn matches_pre_fix_math_for_first_layer() {
        // Layer 0 is always offset 0 regardless of the prefix sum vs
        // `layer * size` formula — the regression only shows up at
        // layer >= 1. This test pins that L0 doesn't shift.
        let v = index_with_layers(&[7, 11, 13], 5);
        assert_eq!(v.ffn_layer_byte_offset(0, 1), 0);
        assert_eq!(v.ffn_layer_byte_offset(0, 3), 0);
    }
}
