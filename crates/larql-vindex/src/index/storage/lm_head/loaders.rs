//! LM-head loaders + the f16 → Q4_K synth path.
//!
//! Three on-disk paths (Q4_K, f32) plus one in-memory path
//! (synthesise from the f16 `embeddings.bin` for tied-embedding
//! models). All four populate `self.projections.lm_head_*` so the
//! KNN dispatch in `knn.rs` picks them up uniformly.

use std::sync::Arc;

use larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;

use crate::error::VindexError;
use crate::format::filenames::*;
use crate::index::core::VectorIndex;
use crate::mmap_util::mmap_optimized;

use super::{read_lm_head_manifest_kind, Q4_BYTES_PER_ELEM_DEN, Q4_BYTES_PER_ELEM_NUM};

impl VectorIndex {
    /// Load Q4 lm_head for GPU logits (replaces CPU f32 lm_head KNN).
    ///
    /// When `weight_manifest.json` is present and lists `lm_head.weight`, the
    /// entry's `kind` must be `kind::TENSOR_Q4K` — anything else is treated
    /// as a writer/reader contract violation and rejected, since the matvec
    /// kernel dispatched here (`q4k_matvec` via `lm_head_knn_backend`) is
    /// Q4_K-specific. This blocks the regression where a Q4_0 file shipped
    /// under the Q4_K filename produced silent garbage logits.
    ///
    /// Older vindexes without a manifest entry for lm_head still load (the
    /// extractor wrote the file directly), but no format check happens.
    pub fn load_lm_head_kquant(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = resolve_lm_head_kquant(dir).bin;
        if !path.exists() {
            return Err(VindexError::Parse(format!(
                "lm_head k-quant file not found (looked for {} and legacy {})",
                LM_HEAD_KQUANT_BIN, LEGACY_LM_HEAD_Q4_BIN
            )));
        }
        if let Some(manifest_kind) = read_lm_head_manifest_kind(dir) {
            if manifest_kind != crate::format::weights::write_f32::kind::TENSOR_Q4K {
                return Err(VindexError::Parse(format!(
                    "lm_head k-quant manifest mismatch: expected kind \"{}\", \
                     found \"{}\". This indicates the vindex was extracted with \
                     a writer that disagrees with the Q4_K matvec dispatch path \
                     — refusing to load to avoid silent garbage logits.",
                    crate::format::weights::write_f32::kind::TENSOR_Q4K,
                    manifest_kind
                )));
            }
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };
        // Derive `vocab_size` from the file size when it's still 0. Q4_K and
        // Q4_0 share the 9/16 byte-rate (`Q4_BYTES_PER_ELEM_*`), so the same
        // divisor handles both formats. Mirrors the pattern in `load_lm_head`
        // for f32 lm_head files.
        if self.vocab_size == 0 && self.hidden_size > 0 {
            let bytes = mmap.len();
            let denom = self.hidden_size * Q4_BYTES_PER_ELEM_NUM;
            // u64 multiply guards 32-bit hosts (e.g. armv7-android) where
            // `bytes * Q4_BYTES_PER_ELEM_DEN` can overflow `usize` before the
            // division for large lm_head mmaps.
            if let Some(vocab) = (bytes as u64)
                .checked_mul(Q4_BYTES_PER_ELEM_DEN as u64)
                .and_then(|v| v.checked_div(denom as u64))
                .and_then(|v| usize::try_from(v).ok())
            {
                if vocab > 0 {
                    self.vocab_size = vocab;
                }
            }
        }
        Arc::make_mut(&mut self.storage).set_lm_head_kquant_mmap(Arc::new(mmap));
        Ok(())
    }

    /// Whether Q4 lm_head is loaded (from file or synthesized from f16 embeddings).
    pub fn has_lm_head_kquant(&self) -> bool {
        self.storage.has_lm_head_kquant()
    }

    /// Synthesize Q4_0 lm_head in RAM from the f16 embeddings mmap.
    /// No-op if a Q4 source already exists or preconditions are not met.
    pub fn synthesize_lm_head_kquant(&mut self) {
        if self.storage.has_lm_head_kquant() {
            return;
        }
        let vocab = self.vocab_size;
        let hidden = self.hidden_size;
        // Q4_K quantises in `K_QUANT_BLOCK_ELEMS`-element super-blocks, so
        // `hidden` must be a multiple of that (matches the on-disk
        // `lm_head_q4.bin` writer in `format/weights/write_kquant/mod.rs`).
        // Earlier code used Q4_0 (32-element blocks) here but
        // `lm_head_knn_backend` dispatches `q4k_matvec` for both the mmap and
        // synth paths — keeping the synth bytes in Q4_K avoids the format-
        // collision bug that broke gemma3-4b-v2.vindex (writer Q4_K vs reader
        // Q4_0).
        if vocab == 0 || hidden == 0 || !hidden.is_multiple_of(K_QUANT_BLOCK_ELEMS) {
            return;
        }
        let f16_bytes: bytes::Bytes = match self.storage.lm_head_f16_view() {
            Some(b) => b.clone(),
            None => return,
        };
        let f16_buf: &[u8] = f16_bytes.as_ref();
        let expected = vocab * hidden * 2;
        if f16_buf.len() < expected {
            return;
        }
        // Decode the whole f16 mmap to f32 in one pass, then Q4_K-quantise
        // the flat `[vocab, hidden]` row-major data. Q4_K's 256-element
        // super-blocks fit cleanly into one row when `hidden` is a multiple
        // of 256, so a flat call gives the same row-by-row layout the
        // matvec kernel expects.
        let mut all_f32 = vec![0.0f32; vocab * hidden];
        for (i, slot) in all_f32.iter_mut().enumerate() {
            let off = i * 2;
            let bits = u16::from_le_bytes([f16_buf[off], f16_buf[off + 1]]);
            *slot = larql_models::quant::half::f16_to_f32(bits);
        }
        let q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(&all_f32);
        Arc::make_mut(&mut self.storage).set_lm_head_kquant_synth(Arc::new(q4k));
    }

    /// Adopt the vindex's f16 `embeddings.bin` mmap as an f16 view of the
    /// LM head. Safe only for tied-embedding models (Gemma 2/3/4, Llama
    /// when `tie_word_embeddings=true`) — the loader is responsible for
    /// gating. Caller must have already populated `vocab_size`.
    ///
    /// When set, `lm_head_knn_backend` prefers `ComputeBackend::f16_gemv`
    /// on the mmap'd bytes, avoiding the 5.6 GB f32 clone on Gemma 4 31B.
    pub fn set_lm_head_f16_mmap(&mut self, mmap: Arc<memmap2::Mmap>) {
        Arc::make_mut(&mut self.storage).set_lm_head_f16(mmap);
    }

    /// Whether an f16 mmap view of the LM head is available.
    pub fn has_lm_head_f16(&self) -> bool {
        self.storage.has_lm_head_f16() && self.vocab_size > 0
    }

    // ── LM head (output projection) for vindex logits ──

    /// Load lm_head from lm_head.bin for KNN logit lookup.
    pub fn load_lm_head(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join(LM_HEAD_BIN);
        if !path.exists() {
            return Err(VindexError::Parse("lm_head.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };
        // Detect vocab size from file size: vocab = file_bytes / (hidden_size * 4)
        let vocab = mmap.len() / (self.hidden_size * 4);
        self.vocab_size = vocab;
        Arc::make_mut(&mut self.storage).set_lm_head_f32(Arc::new(mmap));
        Ok(())
    }

    /// Whether lm_head is loaded for vindex logits.
    pub fn has_lm_head(&self) -> bool {
        self.storage.has_lm_head_f32() && self.vocab_size > 0
    }
}
