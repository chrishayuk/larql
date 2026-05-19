//! W2 feature-major down emit — transposes the down weights to
//! `[intermediate, hidden]` orientation and re-quantises at the same
//! precision the interleaved file uses, so per-feature decode at load
//! time can skip the `kquant_ffn_layer` cache and serve a single row.
//!
//! Lives only during the FFN write loop in
//! `super::write_model_weights_kquant_with_opts`. Each layer's down call
//! goes through `append_layer`; `finalize` flushes the bytes and emits
//! `down_features_q4k_manifest.json`. Both files are opt-in
//! (`KquantWriteOptions::feature_major_down`).
//!
//! See `ROADMAP.md` § W2 for the perf rationale (2440× at K=100,
//! 25× at full K on Gemma 4B Q4_K).
//!
//! Carved out of the monolithic `write_kquant.rs` in the 2026-04-25
//! modularity pass.

use std::io::{BufWriter, Write};
use std::path::Path;

use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};

use crate::error::VindexError;
use crate::format::weights::Q4kManifestEntry;

use super::{pad_rows_to_block, QuantBlockFormat};

/// In-flight state for the W2 feature-major down emission. Lives only
/// while the FFN write loop is running; collapsed into the manifest
/// JSON at end-of-loop. Each field has a name at the call sites
/// (replaces what used to be an anonymous 3-tuple inside the writer).
pub(crate) struct FeatureMajorDownState {
    file: BufWriter<std::fs::File>,
    next_offset: u64,
    manifest: Vec<Q4kManifestEntry>,
}

impl FeatureMajorDownState {
    pub(crate) fn new(path: &Path, capacity_layers: usize) -> Result<Self, VindexError> {
        Ok(Self {
            file: BufWriter::new(std::fs::File::create(path)?),
            next_offset: 0,
            manifest: Vec::with_capacity(capacity_layers),
        })
    }

    /// Transpose padded down (`[hidden, padded_intermediate]`) to
    /// feature-major (`[padded_intermediate, padded_hidden]`),
    /// re-pad rows to 256, and quantise at `format`. Mirrors the
    /// orientation used by `kquant_ffn_layer`'s in-memory transpose so
    /// the runtime decode path reads the same byte layout.
    pub(crate) fn append_layer(
        &mut self,
        key: String,
        padded_down: &[f32],
        rows_hidden: usize,
        cols_padded_intermediate: usize,
        format: QuantBlockFormat,
    ) -> Result<(), VindexError> {
        let n = rows_hidden * cols_padded_intermediate;
        debug_assert_eq!(padded_down.len(), n);
        let mut transposed = vec![0.0f32; n];
        for h in 0..rows_hidden {
            let src =
                &padded_down[h * cols_padded_intermediate..(h + 1) * cols_padded_intermediate];
            for (feat, &v) in src.iter().enumerate() {
                transposed[feat * rows_hidden + h] = v;
            }
        }
        let (fm_padded, fm_padded_cols) =
            pad_rows_to_block(&transposed, cols_padded_intermediate, rows_hidden);
        let bytes = match format {
            QuantBlockFormat::Q6K => quantize_q6_k(&fm_padded),
            QuantBlockFormat::Q4K => quantize_q4_k(&fm_padded),
            QuantBlockFormat::Other(ref tag) => {
                return Err(VindexError::Parse(format!(
                    "feature-major-down writer cannot emit format {tag:?}; \
                     add an encode function and a typed variant first"
                )));
            }
        };
        self.file.write_all(&bytes)?;
        let length = bytes.len() as u64;
        self.manifest.push(Q4kManifestEntry {
            key,
            shape: vec![cols_padded_intermediate, fm_padded_cols],
            format,
            offset: self.next_offset,
            length,
        });
        self.next_offset += length;
        Ok(())
    }

    /// Flush the bytes and write the manifest JSON sidecar.
    pub(crate) fn finalize(mut self, manifest_path: &Path) -> Result<(), VindexError> {
        self.file.flush()?;
        drop(self.file);
        let json = serde_json::to_string_pretty(&self.manifest)
            .map_err(|e| VindexError::Parse(e.to_string()))?;
        std::fs::write(manifest_path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn append_layer_rejects_unsupported_format() {
        let tmp = TempDir::new().unwrap();
        let mut state = FeatureMajorDownState::new(&tmp.path().join("down.bin"), 1).expect("new");
        // 1 hidden row × 256 padded intermediate cols — minimum that
        // satisfies the length debug-assert and pad_rows_to_block's
        // 256-multiple expectation.
        let padded = vec![0.0f32; 256];
        let err = state
            .append_layer(
                "blocks.0.down".to_string(),
                &padded,
                1,
                256,
                QuantBlockFormat::Other("Q5_K".to_string()),
            )
            .expect_err("Other format must be rejected by the writer");
        let msg = format!("{err}");
        assert!(
            msg.contains("feature-major-down writer cannot emit format"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn append_layer_q4k_roundtrip_then_finalize_writes_manifest() {
        let tmp = TempDir::new().unwrap();
        let bin_path = tmp.path().join("down.bin");
        let manifest_path = tmp.path().join("down_manifest.json");
        let mut state = FeatureMajorDownState::new(&bin_path, 2).expect("new");
        // 2 hidden rows × 256 padded intermediate cols. Non-zero values
        // so the quantiser produces real bytes (zero blocks compress to
        // a degenerate path).
        let padded: Vec<f32> = (0..2 * 256).map(|i| (i as f32) * 0.001).collect();
        state
            .append_layer(
                "blocks.0.down".into(),
                &padded,
                2,
                256,
                QuantBlockFormat::Q4K,
            )
            .expect("Q4_K append");
        state
            .append_layer(
                "blocks.1.down".into(),
                &padded,
                2,
                256,
                QuantBlockFormat::Q6K,
            )
            .expect("Q6_K append");
        state.finalize(&manifest_path).expect("finalize");
        let bin_size = std::fs::metadata(&bin_path).unwrap().len();
        assert!(bin_size > 0, "bin file must hold quantised bytes");
        let manifest_text = std::fs::read_to_string(&manifest_path).unwrap();
        assert!(manifest_text.contains("blocks.0.down"));
        assert!(manifest_text.contains("blocks.1.down"));
        assert!(manifest_text.contains("Q4_K"));
        assert!(manifest_text.contains("Q6_K"));
    }
}
