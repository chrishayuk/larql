//! Stage 6 — model weights (if extract level requires them).

use crate::config::types::QuantFormat;
use crate::error::VindexError;
use crate::extract::streaming::context::StreamingContext;

impl<'a> StreamingContext<'a> {
    /// Stage 6 — model weights (if extract level requires them).
    ///
    /// With quant=q4k we always materialise weights regardless of the
    /// declared level — the Q4_K writer emits all of attn, FFN, norms,
    /// lm_head in one pass and makes `--level browse --quant q4k`
    /// incoherent, so q4k implicitly promotes to "all".
    pub(in crate::extract::streaming) fn maybe_write_model_weights(
        &mut self,
    ) -> Result<(), VindexError> {
        let needs_weights = self.extract_level.writes_attn() || self.quant != QuantFormat::None;
        if !needs_weights {
            return Ok(());
        }
        // `StreamingWeights` is a safetensors-only writer subsystem
        // (Q4_K + f32 weight writers walk safetensors crate views
        // directly). GGUF input is supported at browse level (where
        // `needs_weights == false`) and below only; inference / Q4K
        // levels for GGUF need a separate writer pass that streams
        // per-tensor through `larql_models::quant::ggml::dequantize` —
        // tracked as a follow-on PR.
        let (shard_mmaps, tensor_index) = match (
            self.tensor_source.safetensors_mmap_refs(),
            self.tensor_source.safetensors_index(),
        ) {
            (Some(m), Some(i)) => (m, i),
            _ => {
                return Err(VindexError::Parse(
                    "GGUF input + extract-level requiring attention/FFN weights is not yet \
                     implemented (browse-level GGUF works; inference/Q4K GGUF requires \
                     per-tensor streaming through ggml::dequantize)"
                        .to_string(),
                ));
            }
        };
        let streaming_source = crate::format::weights::StreamingWeights {
            shard_mmaps: &shard_mmaps,
            tensor_index,
            arch: &*self.arch,
            num_layers: self.num_layers,
        };
        // Thread the extract level into the write options so the
        // writer can skip attn/FFN/lm_head sections per tier.
        let mut level_opts = self.weight_opts;
        level_opts.level = self.extract_level;
        match self.quant {
            QuantFormat::None => {
                crate::format::weights::write_model_weights_with_opts(
                    &streaming_source,
                    self.output_dir,
                    self.callbacks,
                    level_opts,
                )?;
            }
            QuantFormat::Q4K => {
                // Q4K doesn't write `up_weights.bin` / `down_weights.bin`
                // at all — the FFN weights live in `interleaved_kquant.bin`.
                // `ffn_compact` is a no-op here by construction. Level
                // gating for Q4K is a future refinement (today Q4K
                // always writes the full set).
                crate::format::weights::write_model_weights_kquant_with_opts(
                    &streaming_source,
                    self.output_dir,
                    self.callbacks,
                    self.q4k_opts,
                )?;
            }
        }
        Ok(())
    }
}
