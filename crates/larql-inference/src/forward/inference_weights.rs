//! Format-agnostic inference weight handle.
//!
//! `InferenceWeights` is the single loading point for any code that needs to
//! run `infer_patched` against a vindex. It detects the quantisation format
//! from `VindexConfig`, loads the right on-disk artefacts, and dispatches to
//! `infer_patched` or `infer_patched_q4k` without the caller branching on
//! `config.quant`.
//!
//! **Scope:** the INFER / INSERT KNN / EXPLAIN INFER pipeline. Specialised
//! callers (bench, generation, Metal) keep their own explicit paths.

use std::path::Path;

use tokenizers::Tokenizer;

use larql_vindex::{
    GateIndex, IndexLoadCallbacks, KnnStore, QuantFormat, VectorIndex, VindexConfig, VindexError,
};

use crate::model::ModelWeights;

use super::infer_patched::{infer_patched, infer_patched_q4k, InferPatchedResult};
use super::predict::predict;
use super::PredictResult;

/// An inference-ready weight handle that is agnostic to quantisation format.
///
/// Constructed via [`InferenceWeights::load`]. Callers use
/// [`InferenceWeights::infer_patched`] and [`InferenceWeights::as_weights`]
/// without branching on the underlying format.
#[allow(clippy::large_enum_variant)]
pub enum InferenceWeights {
    Dense(ModelWeights),
    Quantised {
        weights: ModelWeights,
        index: VectorIndex,
    },
}

impl InferenceWeights {
    /// Load weights for the vindex at `path`, choosing the right artefacts
    /// based on `config.quant`. Returns `VindexError` on any I/O or parse
    /// failure so callers can map it to their own error type.
    pub fn load(
        path: &Path,
        config: &VindexConfig,
        cb: &mut dyn IndexLoadCallbacks,
    ) -> Result<Self, VindexError> {
        if config.quant != QuantFormat::None {
            let mut idx = VectorIndex::load_vindex(path, cb)?;
            idx.load_attn_kquant(path)?;
            idx.load_interleaved_kquant(path)?;
            let weights = larql_vindex::load_model_weights_kquant(path, cb)?;
            Ok(Self::Quantised {
                weights,
                index: idx,
            })
        } else {
            let weights = larql_vindex::load_model_weights(path, cb)?;
            Ok(Self::Dense(weights))
        }
    }

    /// `true` if backed by a quantised (q4k or later) format.
    pub fn is_quantised(&self) -> bool {
        matches!(self, Self::Quantised { .. })
    }

    /// Borrow the underlying `ModelWeights` (arch + embeddings + norms).
    ///
    /// Always valid — both variants carry a `ModelWeights`. For the
    /// `Quantised` variant the attention/FFN tensor slots are empty; callers
    /// that need full attention tensors in memory must not use the dense path.
    pub fn as_weights(&self) -> &ModelWeights {
        match self {
            Self::Dense(w) => w,
            Self::Quantised { weights, .. } => weights,
        }
    }

    /// Mutably borrow the underlying `ModelWeights`.
    pub fn as_weights_mut(&mut self) -> &mut ModelWeights {
        match self {
            Self::Dense(w) => w,
            Self::Quantised { weights, .. } => weights,
        }
    }

    /// Run the shared INFER pipeline, dispatching to the correct forward path.
    ///
    /// Identical contract to [`infer_patched`] / [`infer_patched_q4k`]:
    /// unlimited walk FFN features, `KNN_COSINE_THRESHOLD = 0.75`, first
    /// stored layer wins. Callers do not branch on format.
    pub fn infer_patched(
        &mut self,
        tokenizer: &Tokenizer,
        gate_index: &dyn GateIndex,
        knn_store: Option<&KnnStore>,
        token_ids: &[u32],
        top_k: usize,
    ) -> InferPatchedResult {
        match self {
            Self::Dense(weights) => {
                infer_patched(weights, tokenizer, gate_index, knn_store, token_ids, top_k)
            }
            Self::Quantised { weights, index } => infer_patched_q4k(
                weights, tokenizer, gate_index, knn_store, token_ids, top_k, index,
            ),
        }
    }

    /// Dense forward pass (no walk FFN, no KNN). Used for the
    /// `INFER COMPARE` dense side-by-side column.
    pub fn predict_dense(
        &mut self,
        tokenizer: &Tokenizer,
        token_ids: &[u32],
        top_k: usize,
    ) -> PredictResult {
        match self {
            Self::Dense(weights) => predict(weights, tokenizer, token_ids, top_k),
            Self::Quantised { weights, index } => {
                crate::vindex::predict_kquant(weights, tokenizer, token_ids, top_k, index)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! `InferenceWeights` coverage.
    //!
    //! Dense-branch tests construct `InferenceWeights::Dense` directly
    //! from `test_utils::make_test_weights` and exercise every method.
    //! `load(Dense)` is covered separately via the on-disk
    //! `write_synthetic_model_dir` fixture.
    //!
    //! The `Quantised` branch needs a Q4K on-disk vindex that the
    //! synthetic fixtures don't yet write. It is constructed in memory
    //! here via `make_test_q4k_weights` + `make_test_q4k_vindex` so the
    //! match-arms still get hit, but `load(Quantised)` stays uncovered
    //! pending the Q4K disk fixture work.
    use super::*;
    use crate::test_utils::{
        make_test_q4k_vindex, make_test_q4k_weights, make_test_tokenizer, make_test_vindex,
        make_test_weights, write_synthetic_model_dir,
    };
    use larql_vindex::{load_vindex_config, SilentLoadCallbacks};

    fn dense_fixture() -> (InferenceWeights, tokenizers::Tokenizer) {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        (InferenceWeights::Dense(weights), tokenizer)
    }

    fn quantised_fixture() -> (InferenceWeights, tokenizers::Tokenizer) {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        (InferenceWeights::Quantised { weights, index }, tokenizer)
    }

    #[test]
    fn dense_is_quantised_reports_false() {
        let (iw, _) = dense_fixture();
        assert!(!iw.is_quantised());
    }

    #[test]
    fn quantised_is_quantised_reports_true() {
        let (iw, _) = quantised_fixture();
        assert!(iw.is_quantised());
    }

    #[test]
    fn dense_as_weights_returns_borrow() {
        let (iw, _) = dense_fixture();
        let w = iw.as_weights();
        assert_eq!(w.num_layers, 2);
        assert_eq!(w.hidden_size, 16);
    }

    #[test]
    fn quantised_as_weights_returns_borrow() {
        let (iw, _) = quantised_fixture();
        let w = iw.as_weights();
        // Q4K test fixture: 2 layers, hidden = Q4K_TEST_HIDDEN.
        assert_eq!(w.num_layers, 2);
        assert!(w.hidden_size > 0);
    }

    #[test]
    fn dense_as_weights_mut_returns_mutable_borrow() {
        let (mut iw, _) = dense_fixture();
        let w = iw.as_weights_mut();
        // Bump a field to prove we have &mut access — restore right after.
        let original = w.rope_base;
        w.rope_base = 12345.0;
        assert_eq!(w.rope_base, 12345.0);
        w.rope_base = original;
    }

    #[test]
    fn quantised_as_weights_mut_returns_mutable_borrow() {
        let (mut iw, _) = quantised_fixture();
        let w = iw.as_weights_mut();
        let original = w.rope_base;
        w.rope_base = 9999.0;
        assert_eq!(w.rope_base, 9999.0);
        w.rope_base = original;
    }

    #[test]
    fn dense_infer_patched_returns_predictions() {
        let (mut iw, tokenizer) = dense_fixture();
        let index = make_test_vindex(iw.as_weights());
        let result = iw.infer_patched(&tokenizer, &index, None, &[0u32, 1, 2], 5);
        assert!(!result.predictions.is_empty());
        // top_k clamped by vocab/available rows; just check we got a
        // shaped result.
        assert!(result.predictions.len() <= 5);
    }

    #[test]
    fn dense_predict_dense_returns_predictions() {
        let (mut iw, tokenizer) = dense_fixture();
        let result = iw.predict_dense(&tokenizer, &[0u32, 1, 2], 5);
        assert!(!result.predictions.is_empty());
        assert!(result.predictions.len() <= 5);
    }

    #[test]
    fn load_dense_round_trips_via_disk_fixture() {
        // Exercises the Dense branch of `InferenceWeights::load`: write
        // the synthetic model dir, load it back via `load`, observe
        // is_quantised=false and a usable ModelWeights.
        let dir = tempfile::tempdir().expect("tempdir");
        write_synthetic_model_dir(dir.path()).expect("write fixture");
        let config = load_vindex_config(dir.path()).expect("load_vindex_config");
        assert_eq!(config.quant, QuantFormat::None);

        let mut cb = SilentLoadCallbacks;
        let iw =
            InferenceWeights::load(dir.path(), &config, &mut cb).expect("load InferenceWeights");
        assert!(!iw.is_quantised());
        let w = iw.as_weights();
        assert_eq!(w.num_layers, 2);
        assert_eq!(w.hidden_size, 16);
    }
}
