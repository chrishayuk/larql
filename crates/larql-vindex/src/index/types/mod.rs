//! Shared types and traits for the vindex index.
//!
//! ## File layout
//!
//! `mod.rs` keeps the **data** — small POD structs, enums, constants,
//! and the on-disk `DownMetaMmap` reader. The **capability traits** are
//! split one-per-sibling so each surface can be read on its own:
//!
//! - [`gate_lookup`] — `GateLookup` (KNN + feature-meta lookup)
//! - [`patch_overrides`] — `PatchOverrides` (overlay-vector hooks)
//! - [`native_ffn`] — `NativeFfnAccess` (f32/f16 mmap rows)
//! - [`quantized_ffn`] — `QuantizedFfnAccess` (Q4_0 / Q4_K / Q6_K rows)
//! - [`fp4_ffn`] — `Fp4FfnAccess` (FP4 / FP8 rows, exp 26)
//! - [`ffn_row`] — `FfnRowAccess` unified dispatch + `GateIndex`
//!   compatibility composition (blanket impls live here)
//!
//! All traits are re-exported below so `crate::index::types::*`
//! continues to bring the full surface into scope unchanged.

use larql_models::TopKEntry;

mod ffn_row;
mod fp4_ffn;
mod gate_lookup;
mod native_ffn;
mod patch_overrides;
mod quantized_ffn;

pub use ffn_row::{FfnRowAccess, GateIndex};
pub use fp4_ffn::Fp4FfnAccess;
pub use gate_lookup::GateLookup;
pub use native_ffn::NativeFfnAccess;
pub use patch_overrides::PatchOverrides;
pub use quantized_ffn::QuantizedFfnAccess;

/// Default `c_score` for a `FeatureMeta` synthesised without an explicit
/// confidence — used by the patch loader when an `Insert` op omits
/// `confidence`, and by the vindexfile builder when a fact is inserted
/// from a `.vindexfile` directive without a probed score. Lifted to a
/// constant so a future tune of the default touches one site instead of
/// drifting independently across the two callers.
pub const DEFAULT_C_SCORE: f32 = 0.9;

/// Metadata for a single FFN feature (from extraction).
#[derive(Clone)]
pub struct FeatureMeta {
    pub top_token: String,
    pub top_token_id: u32,
    pub c_score: f32,
    pub top_k: Vec<TopKEntry>,
}

/// A single step in the walk trace — one feature that fired at one layer.
pub struct WalkHit {
    pub layer: usize,
    pub feature: usize,
    pub gate_score: f32,
    pub meta: FeatureMeta,
}

/// Result of a walk — per-layer feature activations with full metadata.
pub struct WalkTrace {
    pub layers: Vec<(usize, Vec<WalkHit>)>,
}

/// Storage class for the index's primary FFN payload.
///
/// Walk-path equivalence audits and downstream tooling use this to bucket
/// paths by the precision of the data they walk against, without having
/// to re-derive the right grouping from the `has_*` flags. New storage
/// formats should update [`FfnRowAccess::primary_storage_bucket`]'s default
/// impl so consumers automatically pick up the right bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageBucket {
    /// f32 / f16 features. Walk paths land within float-noise of the
    /// dense matmul reference (cos ≥ 0.99999 territory on f16 vindexes).
    Exact,
    /// Q4_0 / Q4_K / Q6_K interleaved or dequant. Walk paths carry
    /// per-block dequant noise (cos ≥ 0.99 territory).
    Quantized,
    /// FP4 / FP8 storage. Walk paths carry per-block FP4 dequant noise
    /// (cos ≥ 0.98 territory).
    Fp4,
}

/// Progress callbacks for index loading.
pub trait IndexLoadCallbacks {
    fn on_file_start(&mut self, _component: &str, _path: &str) {}
    fn on_progress(&mut self, _records: usize) {}
    fn on_file_done(&mut self, _component: &str, _records: usize, _elapsed_ms: f64) {}
}

pub struct SilentLoadCallbacks;
impl IndexLoadCallbacks for SilentLoadCallbacks {}

/// Per-layer gate vector offset info for mmap mode.
#[derive(Clone, Copy)]
pub struct GateLayerSlice {
    pub float_offset: usize,
    pub num_features: usize,
}

/// Per-layer Q4 gate data offset info.
#[derive(Clone)]
pub struct GateQ4Slice {
    pub byte_offset: usize,
    pub byte_len: usize,
    pub num_features: usize,
}

/// Mmap'd down_meta.bin — reads individual feature records on demand.
#[derive(Clone)]
pub struct DownMetaMmap {
    pub(crate) mmap: std::sync::Arc<memmap2::Mmap>,
    pub(crate) layer_offsets: Vec<usize>,
    pub(crate) layer_num_features: Vec<usize>,
    pub(crate) top_k_count: usize,
    pub(crate) tokenizer: std::sync::Arc<tokenizers::Tokenizer>,
}

impl DownMetaMmap {
    fn record_size(&self) -> usize {
        8 + self.top_k_count * 8
    }

    pub fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        if layer >= self.layer_offsets.len() {
            return None;
        }
        let num_features = self.layer_num_features[layer];
        if num_features == 0 || feature >= num_features {
            return None;
        }

        let offset = self.layer_offsets[layer] + feature * self.record_size();
        let rec_size = self.record_size();
        if offset + rec_size > self.mmap.len() {
            return None;
        }

        let b = &self.mmap[offset..offset + rec_size];
        let top_token_id = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let c_score = f32::from_le_bytes([b[4], b[5], b[6], b[7]]);

        if top_token_id == 0 && c_score == 0.0 {
            return None;
        }

        let mut top_k = Vec::new();
        for i in 0..self.top_k_count {
            let o = 8 + i * 8;
            let tid = u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
            let logit = f32::from_le_bytes([b[o + 4], b[o + 5], b[o + 6], b[o + 7]]);
            if tid > 0 || logit != 0.0 {
                let token = self
                    .tokenizer
                    .decode(&[tid], true)
                    .unwrap_or_else(|_| format!("T{tid}"))
                    .trim()
                    .to_string();
                top_k.push(TopKEntry {
                    token,
                    token_id: tid,
                    logit,
                });
            }
        }

        let top_token = self
            .tokenizer
            .decode(&[top_token_id], true)
            .unwrap_or_else(|_| format!("T{top_token_id}"))
            .trim()
            .to_string();

        Some(FeatureMeta {
            top_token,
            top_token_id,
            c_score,
            top_k,
        })
    }

    pub fn num_features(&self, layer: usize) -> usize {
        self.layer_num_features.get(layer).copied().unwrap_or(0)
    }

    pub fn total_features(&self) -> usize {
        self.layer_num_features.iter().sum()
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    /// Top-K entry as a tuple of `(token_id, logit)` for fixture construction.
    type FixtureTopK<'a> = &'a [(u32, f32)];
    /// One feature's record: `(top_token_id, c_score, top_k_slots)`.
    type FixtureRecord<'a> = (u32, f32, FixtureTopK<'a>);
    /// One layer is a slice of records.
    type FixtureLayer<'a> = &'a [FixtureRecord<'a>];

    /// Hand-build a `DownMetaMmap` over a tempfile so the binary
    /// decode logic in `feature_meta` can be exercised end-to-end.
    /// Layout per feature: `[token_id u32][c_score f32][top_k × (id u32, logit f32)]`.
    fn make_mmap_fixture(
        layers: &[FixtureLayer<'_>],
        top_k_count: usize,
    ) -> (DownMetaMmap, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("down_meta.bin");

        let record_size = 8 + top_k_count * 8;
        let mut bytes: Vec<u8> = Vec::new();
        let mut layer_offsets: Vec<usize> = Vec::with_capacity(layers.len());
        let mut layer_num_features: Vec<usize> = Vec::with_capacity(layers.len());

        for layer in layers {
            layer_offsets.push(bytes.len());
            layer_num_features.push(layer.len());
            for (token_id, c_score, top_k) in *layer {
                bytes.extend_from_slice(&token_id.to_le_bytes());
                bytes.extend_from_slice(&c_score.to_le_bytes());
                // Emit exactly `top_k_count` slots, padding short top_k
                // lists with (0, 0.0) and truncating long ones.
                for slot in 0..top_k_count {
                    let (tid, logit) = top_k.get(slot).copied().unwrap_or((0, 0.0));
                    bytes.extend_from_slice(&tid.to_le_bytes());
                    bytes.extend_from_slice(&logit.to_le_bytes());
                }
                debug_assert_eq!(
                    bytes.len() % record_size,
                    layer_offsets.last().copied().unwrap_or(0) % record_size
                );
            }
        }

        std::fs::write(&path, &bytes).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };

        // We never assert on decoded text, only on the binary record
        // shape — a trivial WordLevel tokenizer is enough to satisfy
        // the `feature_meta` `decode` call path. The real tokenizer
        // lives in a real vindex; the binary decode logic doesn't care.
        let tokenizer = empty_tokenizer();

        (
            DownMetaMmap {
                mmap: std::sync::Arc::new(mmap),
                layer_offsets,
                layer_num_features,
                top_k_count,
                tokenizer: std::sync::Arc::new(tokenizer),
            },
            dir,
        )
    }

    #[test]
    fn record_size_includes_header_plus_top_k() {
        let dmm = DownMetaMmap {
            mmap: std::sync::Arc::new(memmap_empty()),
            layer_offsets: Vec::new(),
            layer_num_features: Vec::new(),
            top_k_count: 5,
            tokenizer: std::sync::Arc::new(empty_tokenizer()),
        };
        assert_eq!(dmm.record_size(), 8 + 5 * 8);
    }

    #[test]
    fn num_features_returns_zero_for_unknown_layer() {
        let dmm = DownMetaMmap {
            mmap: std::sync::Arc::new(memmap_empty()),
            layer_offsets: vec![0, 16],
            layer_num_features: vec![1, 2],
            top_k_count: 1,
            tokenizer: std::sync::Arc::new(empty_tokenizer()),
        };
        assert_eq!(dmm.num_features(0), 1);
        assert_eq!(dmm.num_features(1), 2);
        assert_eq!(dmm.num_features(99), 0);
    }

    #[test]
    fn total_features_sums_across_layers() {
        let dmm = DownMetaMmap {
            mmap: std::sync::Arc::new(memmap_empty()),
            layer_offsets: vec![0, 16, 32],
            layer_num_features: vec![3, 7, 5],
            top_k_count: 1,
            tokenizer: std::sync::Arc::new(empty_tokenizer()),
        };
        assert_eq!(dmm.total_features(), 15);
    }

    #[test]
    fn feature_meta_returns_none_for_out_of_range_layer() {
        let (dmm, _guard) = make_mmap_fixture(&[&[(7, 0.5, &[])]], 1);
        assert!(dmm.feature_meta(99, 0).is_none());
    }

    #[test]
    fn feature_meta_returns_none_for_out_of_range_feature() {
        let (dmm, _guard) = make_mmap_fixture(&[&[(7, 0.5, &[])]], 1);
        assert!(dmm.feature_meta(0, 99).is_none());
    }

    #[test]
    fn feature_meta_returns_none_for_zero_record() {
        // (0, 0.0) is the sentinel for "no meta recorded for this slot".
        let (dmm, _guard) = make_mmap_fixture(&[&[(0, 0.0, &[])]], 1);
        assert!(dmm.feature_meta(0, 0).is_none());
    }

    #[test]
    fn feature_meta_decodes_token_id_and_score() {
        let (dmm, _guard) = make_mmap_fixture(&[&[(123, 1.5, &[])]], 1);
        let meta = dmm.feature_meta(0, 0).expect("meta present");
        assert_eq!(meta.top_token_id, 123);
        assert!((meta.c_score - 1.5).abs() < 1e-6);
    }

    #[test]
    fn feature_meta_skips_zero_top_k_slots() {
        // top_k_count=3 but only one real entry — the zero slots
        // must be filtered out of the returned `top_k`.
        let (dmm, _guard) = make_mmap_fixture(&[&[(7, 0.9, &[(42, 0.7)])]], 3);
        let meta = dmm.feature_meta(0, 0).unwrap();
        assert_eq!(meta.top_k.len(), 1, "zero-padded slots filtered out");
        assert_eq!(meta.top_k[0].token_id, 42);
        assert!((meta.top_k[0].logit - 0.7).abs() < 1e-6);
    }

    // ── helpers used by the unit tests above ───────────────────────

    fn memmap_empty() -> memmap2::Mmap {
        // Tempfile that lives long enough for the test scope —
        // returns an empty mmap suitable for record_size / num_features
        // tests where the bytes themselves are never read.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.bin");
        std::fs::write(&path, b"\0").unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
        // Leak the tempdir for the duration of the test — the mmap
        // outlives this scope but the OS reclaims pages on process
        // exit; a few bytes per test is fine.
        std::mem::forget(dir);
        mmap
    }

    fn empty_tokenizer() -> tokenizers::Tokenizer {
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::TokenizerBuilder;
        let model = WordLevel::builder()
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        TokenizerBuilder::<
            tokenizers::models::wordlevel::WordLevel,
            tokenizers::normalizers::NormalizerWrapper,
            tokenizers::pre_tokenizers::PreTokenizerWrapper,
            tokenizers::processors::PostProcessorWrapper,
            tokenizers::decoders::DecoderWrapper,
        >::default()
        .with_model(model)
        .build()
        .unwrap()
        .into()
    }
}
