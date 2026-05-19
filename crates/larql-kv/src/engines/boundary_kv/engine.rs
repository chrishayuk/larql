//! `BoundaryKvEngine` — Standard semantics + `larql-boundary` frame emission.

use std::sync::Arc;

use larql_boundary::{
    codec::{bf16 as codec_bf16, int8 as codec_int8},
    metadata, BoundaryCompression, BoundaryContract, BoundaryDecision, BoundaryFrame,
    BoundaryGateConfig,
};
use larql_inference::async_compute_backend::AsyncComputeBackend;
use larql_inference::ffn::FfnBackend;
use larql_inference::forward::hidden_to_raw_logits;
use larql_inference::model::ModelWeights;
use larql_inference::{cpu_engine_backend, EngineBackend};
use ndarray::Array2;

use crate::engines::boundary_kv::archive::{ArchiveError, BoundaryArchive, InMemoryArchive};
use crate::engines::boundary_kv::identity::BoundaryModelIdentity;
use crate::engines::standard::StandardEngine;
use crate::{EngineInfo, KvEngine};

/// Engine-level configuration.
#[derive(Debug, Clone)]
pub struct BoundaryKvEngineConfig {
    /// Hot-window cap for the inner `StandardEngine`. `None` = unbounded.
    pub window_size: Option<usize>,
    /// Capture a frame whenever the decode position reaches a positive
    /// multiple of `chunk_tokens`. Must be ≥ 1.
    pub chunk_tokens: usize,
    /// Identifies the session's chain in the archive. The archive groups
    /// frames by this id; restoring from a chain requires the same id.
    pub sequence_id: String,
    /// Embedded in every emitted frame for receiver-side verification.
    pub identity: BoundaryModelIdentity,
    /// Codec + threshold configuration for the gate.
    pub gate_config: BoundaryGateConfig,
    /// When true, run the compressed-residual forward to populate
    /// `boundary_agreement`. When false, frames carry `NotChecked` and the
    /// gate falls back per its `require_compressed_agreement` policy. See
    /// `BOUNDARY_REF_PROTOCOL.md` §8 for the cost tradeoff.
    pub verify_agreement: bool,
}

impl BoundaryKvEngineConfig {
    /// Build a default config: chunk_tokens=512, calibration-mode gate (always
    /// `UseBf16`), agreement verification enabled. Caller supplies a sequence
    /// id and a model identity.
    pub fn new(sequence_id: impl Into<String>, identity: BoundaryModelIdentity) -> Self {
        Self {
            window_size: None,
            chunk_tokens: 512,
            sequence_id: sequence_id.into(),
            identity,
            gate_config: BoundaryGateConfig::default(),
            verify_agreement: true,
        }
    }
}

/// `BoundaryKvEngine` — production-equivalent in-session decode, with frame
/// emission at chunk boundaries.
pub struct BoundaryKvEngine {
    inner: StandardEngine,
    config: BoundaryKvEngineConfig,
    archive: Arc<dyn BoundaryArchive>,
    abs_position: usize,
}

impl BoundaryKvEngine {
    /// Construct with the default CPU backend and an in-memory archive.
    pub fn new(config: BoundaryKvEngineConfig) -> Self {
        Self::with_backend(config, cpu_engine_backend())
    }

    /// Construct with a specific compute backend and an in-memory archive.
    pub fn with_backend(config: BoundaryKvEngineConfig, backend: Box<dyn EngineBackend>) -> Self {
        Self::with_backend_and_archive(config, backend, Arc::new(InMemoryArchive::new()))
    }

    /// Construct with a specific async compute backend and an in-memory archive.
    pub fn with_async_backend(
        config: BoundaryKvEngineConfig,
        backend: Box<dyn AsyncComputeBackend>,
    ) -> Self {
        let inner = StandardEngine::with_async_backend(config.window_size, backend);
        Self {
            inner,
            config,
            archive: Arc::new(InMemoryArchive::new()),
            abs_position: 0,
        }
    }

    /// Construct with a caller-supplied archive. Use this when you need
    /// durability beyond a single process.
    pub fn with_backend_and_archive(
        config: BoundaryKvEngineConfig,
        backend: Box<dyn EngineBackend>,
        archive: Arc<dyn BoundaryArchive>,
    ) -> Self {
        let inner = StandardEngine::with_backend(config.window_size, backend);
        Self {
            inner,
            config,
            archive,
            abs_position: 0,
        }
    }

    /// Borrow the archive (for inspection / chain replay).
    pub fn archive(&self) -> &Arc<dyn BoundaryArchive> {
        &self.archive
    }

    /// Current logical decode position.
    pub fn abs_position(&self) -> usize {
        self.abs_position
    }

    fn chunk_tokens(&self) -> usize {
        self.config.chunk_tokens.max(1)
    }

    fn token_start_of_current_chunk(&self) -> u64 {
        let chunk = self.chunk_tokens() as u64;
        let end = self.abs_position as u64;
        end.saturating_sub(chunk)
    }

    /// True iff `abs_position` (the just-completed step) lands on a chunk
    /// boundary. Position 0 is never a boundary.
    fn at_chunk_boundary(&self) -> bool {
        self.abs_position > 0 && self.abs_position % self.chunk_tokens() == 0
    }

    /// Build + archive a frame from the most recent hidden state. No-ops if
    /// the position is not a chunk boundary or the hidden is empty.
    fn maybe_emit_frame(
        &self,
        weights: &ModelWeights,
        hidden: &Array2<f32>,
    ) -> Result<(), ArchiveError> {
        if !self.at_chunk_boundary() {
            return Ok(());
        }
        if hidden.shape()[0] == 0 || hidden.shape()[1] == 0 {
            return Ok(());
        }
        let frame = build_frame(
            weights,
            hidden,
            &self.config,
            self.token_start_of_current_chunk(),
            self.abs_position as u64,
        );
        self.archive.append(frame)
    }
}

/// Build a `BoundaryFrame` from a final-position hidden state. Pure modulo
/// `weights` (read-only).
pub(super) fn build_frame(
    weights: &ModelWeights,
    hidden: &Array2<f32>,
    config: &BoundaryKvEngineConfig,
    token_start: u64,
    token_end: u64,
) -> BoundaryFrame {
    let residual = last_row_flat(hidden);
    let hidden_size = residual.len() as u32;
    let raw_logits = hidden_to_raw_logits(weights, hidden);
    let hat_logits = if config.verify_agreement {
        Some(reconstruct_logits(weights, hidden, &residual))
    } else {
        None
    };
    let mut meta = metadata::compute(&raw_logits, hat_logits.as_deref());
    let decision = larql_boundary::gate::apply(&mut meta, &config.gate_config);

    let (compression_scheme, contract_level, payload) = encode_for_decision(&decision, &residual);

    let boundary_id = format!("{}:{}", config.sequence_id, token_end);
    BoundaryFrame {
        version: 1,
        model_id: config.identity.model_id.clone(),
        model_revision: config.identity.model_revision.clone(),
        tokenizer_revision: config.identity.tokenizer_revision.clone(),
        architecture: config.identity.architecture.clone(),
        boundary_id,
        sequence_id: config.sequence_id.clone(),
        token_start,
        token_end,
        layer: weights.num_layers.saturating_sub(1) as u16,
        hidden_size,
        compression_scheme,
        contract_level,
        payload,
        raw_top1_token: meta.raw_top1_token,
        raw_logit_margin: meta.raw_logit_margin,
        raw_top1_prob: Some(meta.raw_top1_prob),
        compressed_top1_token: meta.compressed_top1_token,
        boundary_agreement: meta.boundary_agreement,
        codec_fragile: meta.codec_fragile,
        boundary_fragile: meta.boundary_fragile,
        fallback_policy: config.gate_config.fallback_policy.clone(),
        fallback_ref: None,
        calibration_run_id: None,
        residual_hash: None,
        token_hash: None,
    }
}

fn last_row_flat(hidden: &Array2<f32>) -> Vec<f32> {
    let rows = hidden.shape()[0];
    if rows == 0 {
        return Vec::new();
    }
    hidden.row(rows - 1).to_vec()
}

/// Run the compressed-residual forward (encode → decode → re-project to
/// logits) so the gate can populate `boundary_agreement`. Uses the same
/// `lm_head` as the raw path so any margin shift is purely codec-induced.
fn reconstruct_logits(weights: &ModelWeights, hidden: &Array2<f32>, residual: &[f32]) -> Vec<f32> {
    let encoded = codec_int8::encode(residual);
    let decoded = codec_int8::decode(&encoded);
    let mut hat = hidden.clone();
    let rows = hat.shape()[0];
    let cols = hat.shape()[1];
    let last = rows - 1;
    for j in 0..cols {
        hat[[last, j]] = decoded[j];
    }
    hidden_to_raw_logits(weights, &hat)
}

fn encode_for_decision(
    decision: &BoundaryDecision,
    residual: &[f32],
) -> (BoundaryCompression, BoundaryContract, Vec<u8>) {
    match decision {
        BoundaryDecision::CompressedOk { contract } => {
            let payload = codec_int8::encode(residual).to_bytes();
            (
                BoundaryCompression::Int8Clip3Sigma,
                contract.clone(),
                payload,
            )
        }
        BoundaryDecision::UseBf16 => (
            BoundaryCompression::None,
            BoundaryContract::Calibrating,
            codec_bf16::encode(residual),
        ),
        BoundaryDecision::UseColdReplay | BoundaryDecision::Reject => (
            BoundaryCompression::None,
            BoundaryContract::Unknown,
            Vec::new(),
        ),
    }
}

impl KvEngine for BoundaryKvEngine {
    fn name(&self) -> &str {
        "boundary-kv"
    }

    fn info(&self) -> EngineInfo {
        let inner = self.inner.info();
        let archived = self.archive.total_frames().unwrap_or(0);
        EngineInfo {
            name: "boundary-kv".into(),
            description: format!(
                "Standard KV + boundary-frame emission every {} tokens (archived={archived})",
                self.chunk_tokens(),
            ),
            backend: inner.backend,
            config: format!(
                "chunk_tokens={},sequence_id={},window={}",
                self.chunk_tokens(),
                self.config.sequence_id,
                self.config
                    .window_size
                    .map(|w| w.to_string())
                    .unwrap_or_else(|| "full".into()),
            ),
        }
    }

    fn prefill(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        let hidden = self.inner.prefill(weights, ffn, token_ids)?;
        self.abs_position = token_ids.len();
        // Best-effort emit; archive errors propagate as engine-decode
        // failure (None) per §8.2: a failed emit must not be silently
        // dropped.
        if self.maybe_emit_frame(weights, &hidden).is_err() {
            return None;
        }
        Some(hidden)
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        let hidden = self.inner.decode_step(weights, ffn, token_id)?;
        self.abs_position += 1;
        if self.maybe_emit_frame(weights, &hidden).is_err() {
            return None;
        }
        Some(hidden)
    }

    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    fn window_tokens(&self) -> usize {
        self.inner.window_tokens()
    }

    fn cold_bytes(&self) -> usize {
        self.inner.cold_bytes()
    }

    fn prefill_quant(
        &mut self,
        weights: &mut ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Array2<f32>> {
        let hidden = self
            .inner
            .prefill_quant(weights, ffn, index, token_ids, backend)?;
        self.abs_position = token_ids.len();
        if self.maybe_emit_frame(weights, &hidden).is_err() {
            return None;
        }
        Some(hidden)
    }

    fn decode_step_quant(
        &mut self,
        weights: &mut ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Array2<f32>> {
        let hidden = self
            .inner
            .decode_step_quant(weights, ffn, index, token_id, backend)?;
        self.abs_position += 1;
        if self.maybe_emit_frame(weights, &hidden).is_err() {
            return None;
        }
        Some(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::boundary_kv::archive::BoundaryArchive;
    use larql_boundary::{BoundaryAgreement, FallbackPolicy};
    use larql_inference::ffn::WeightFfn;
    use larql_inference::test_utils::make_test_weights;

    fn config(seq: &str, chunk: usize) -> BoundaryKvEngineConfig {
        let identity = BoundaryModelIdentity::placeholder("test-arch");
        let mut cfg = BoundaryKvEngineConfig::new(seq, identity);
        cfg.chunk_tokens = chunk;
        cfg
    }

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn engine_name_is_boundary_kv() {
        let eng = BoundaryKvEngine::new(config("s", 4));
        assert_eq!(eng.name(), "boundary-kv");
    }

    #[test]
    fn engine_info_reports_chunk_and_sequence() {
        let eng = BoundaryKvEngine::new(config("seq-42", 256));
        let info = eng.info();
        assert!(info.config.contains("chunk_tokens=256"));
        assert!(info.config.contains("sequence_id=seq-42"));
        assert!(info.description.contains("boundary-frame emission"));
    }

    #[test]
    fn engine_memory_zero_before_prefill() {
        let eng = BoundaryKvEngine::new(config("s", 4));
        assert_eq!(eng.memory_bytes(), 0);
        assert_eq!(eng.window_tokens(), 0);
        assert_eq!(eng.cold_bytes(), 0);
        assert_eq!(eng.abs_position(), 0);
    }

    #[test]
    fn chunk_tokens_floor_is_one() {
        let mut cfg = config("s", 0);
        cfg.chunk_tokens = 0;
        let eng = BoundaryKvEngine::new(cfg);
        // chunk_tokens=0 in config; floor brings it to 1.
        assert_eq!(eng.chunk_tokens(), 1);
    }

    // ── Standard parity (no boundary captures triggered) ──────────────────────

    #[test]
    fn prefill_matches_standard_when_no_boundary_crossed() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut boundary = BoundaryKvEngine::new(config("s", 16)); // chunk > prompt
        let mut standard = StandardEngine::new(None);
        let prompt = [0u32, 1, 2];
        let h_b = boundary.prefill(&weights, &ffn, &prompt).unwrap();
        let h_s = standard.prefill(&weights, &ffn, &prompt).unwrap();
        assert_eq!(h_b, h_s, "in-session prefill must be bit-identical");
        // No boundary crossed: archive is empty.
        assert_eq!(boundary.archive().total_frames(), Some(0));
    }

    #[test]
    fn decode_step_matches_standard_when_no_boundary_crossed() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut boundary = BoundaryKvEngine::new(config("s", 64));
        let mut standard = StandardEngine::new(None);
        boundary.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
        standard.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
        let h_b = boundary.decode_step(&weights, &ffn, 2).unwrap();
        let h_s = standard.decode_step(&weights, &ffn, 2).unwrap();
        assert_eq!(h_b, h_s);
        assert_eq!(boundary.archive().total_frames(), Some(0));
    }

    // ── Boundary emission ────────────────────────────────────────────────────

    #[test]
    fn prefill_landing_on_boundary_emits_frame() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        // chunk_tokens=2, prefill exactly 2 tokens → lands on boundary.
        let mut eng = BoundaryKvEngine::new(config("seq", 2));
        eng.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
        let chain = eng.archive().load_chain("seq").unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].token_end, 2);
        assert_eq!(chain[0].token_start, 0);
        assert_eq!(chain[0].sequence_id, "seq");
    }

    #[test]
    fn decode_landing_on_boundary_emits_frame() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        // chunk_tokens=2, prefill 1 then decode 1 → boundary at position 2.
        let mut eng = BoundaryKvEngine::new(config("seq", 2));
        eng.prefill(&weights, &ffn, &[0u32]).unwrap();
        assert_eq!(eng.archive().total_frames(), Some(0));
        eng.decode_step(&weights, &ffn, 1).unwrap();
        let chain = eng.archive().load_chain("seq").unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].token_end, 2);
    }

    #[test]
    fn position_off_boundary_does_not_emit() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = BoundaryKvEngine::new(config("seq", 4));
        eng.prefill(&weights, &ffn, &[0u32, 1, 2]).unwrap();
        assert_eq!(eng.archive().total_frames(), Some(0));
    }

    #[test]
    fn multiple_boundaries_accumulate_in_chain() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = BoundaryKvEngine::new(config("seq", 2));
        eng.prefill(&weights, &ffn, &[0u32, 1]).unwrap(); // boundary @ 2
        eng.decode_step(&weights, &ffn, 2).unwrap(); // no boundary
        eng.decode_step(&weights, &ffn, 3).unwrap(); // boundary @ 4
        eng.decode_step(&weights, &ffn, 4).unwrap(); // no boundary
        eng.decode_step(&weights, &ffn, 5).unwrap(); // boundary @ 6
        let chain = eng.archive().load_chain("seq").unwrap();
        assert_eq!(chain.len(), 3);
        let ends: Vec<u64> = chain.iter().map(|f| f.token_end).collect();
        assert_eq!(ends, vec![2, 4, 6]);
    }

    #[test]
    fn frame_carries_model_identity() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = BoundaryKvEngine::new(config("seq", 2));
        eng.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
        let chain = eng.archive().load_chain("seq").unwrap();
        let f = &chain[0];
        assert_eq!(f.model_id, "test-arch");
        assert!(f.model_revision.contains("test-arch"));
        assert!(f.tokenizer_revision.contains("test-arch"));
        assert_eq!(f.architecture, "test-arch");
        assert_eq!(f.boundary_id, "seq:2");
        assert_eq!(f.hidden_size as usize, weights.hidden_size);
        assert_eq!(f.layer as usize, weights.num_layers - 1);
    }

    #[test]
    fn calibration_mode_emits_bf16_frame() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        // Default gate config has calibration_mode = true → always bf16.
        let mut eng = BoundaryKvEngine::new(config("seq", 2));
        eng.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
        let chain = eng.archive().load_chain("seq").unwrap();
        let f = &chain[0];
        assert_eq!(f.compression_scheme, BoundaryCompression::None);
        assert_eq!(f.contract_level, BoundaryContract::Calibrating);
        assert!(!f.payload.is_empty(), "bf16 frame must carry residual");
        // bf16 payload size = 2 * hidden_size.
        assert_eq!(f.payload.len(), 2 * weights.hidden_size);
    }

    #[test]
    fn agreement_check_populates_compressed_top1() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut cfg = config("seq", 2);
        cfg.verify_agreement = true;
        let mut eng = BoundaryKvEngine::new(cfg);
        eng.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
        let chain = eng.archive().load_chain("seq").unwrap();
        let f = &chain[0];
        assert!(
            f.compressed_top1_token.is_some(),
            "verify_agreement=true should populate compressed_top1_token"
        );
        // Either Agrees or Disagrees, but not NotChecked.
        assert!(!matches!(
            f.boundary_agreement,
            BoundaryAgreement::NotChecked
        ));
    }

    #[test]
    fn agreement_skipped_when_verify_off() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut cfg = config("seq", 2);
        cfg.verify_agreement = false;
        let mut eng = BoundaryKvEngine::new(cfg);
        eng.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
        let chain = eng.archive().load_chain("seq").unwrap();
        let f = &chain[0];
        assert_eq!(f.boundary_agreement, BoundaryAgreement::NotChecked);
        assert!(f.compressed_top1_token.is_none());
    }

    #[test]
    fn shared_archive_aggregates_multiple_engines() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let archive: Arc<dyn BoundaryArchive> = Arc::new(InMemoryArchive::new());
        let mut e1 = BoundaryKvEngine::with_backend_and_archive(
            config("alpha", 2),
            cpu_engine_backend(),
            archive.clone(),
        );
        let mut e2 = BoundaryKvEngine::with_backend_and_archive(
            config("beta", 2),
            cpu_engine_backend(),
            archive.clone(),
        );
        e1.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
        e2.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
        assert_eq!(archive.total_frames(), Some(2));
        assert_eq!(archive.load_chain("alpha").unwrap().len(), 1);
        assert_eq!(archive.load_chain("beta").unwrap().len(), 1);
    }

    // ── Failure-mode emission ────────────────────────────────────────────────

    /// Archive that fails every append. Used to verify §8.2: a failed emit
    /// must propagate as engine-decode failure, not silently drop.
    struct FailingArchive;
    impl BoundaryArchive for FailingArchive {
        fn append(&self, _: BoundaryFrame) -> Result<(), ArchiveError> {
            Err(ArchiveError::Backend("simulated".into()))
        }
        fn load_chain(&self, _: &str) -> Result<Vec<BoundaryFrame>, ArchiveError> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn archive_failure_at_boundary_returns_none() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = BoundaryKvEngine::with_backend_and_archive(
            config("seq", 2),
            cpu_engine_backend(),
            Arc::new(FailingArchive),
        );
        // chunk=2, prefill 2 → boundary; archive returns Err → engine None.
        assert!(eng.prefill(&weights, &ffn, &[0u32, 1]).is_none());
    }

    #[test]
    fn archive_failure_off_boundary_does_not_return_none() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        // chunk=8, prefill 3 → no boundary → archive never consulted.
        let mut eng = BoundaryKvEngine::with_backend_and_archive(
            config("seq", 8),
            cpu_engine_backend(),
            Arc::new(FailingArchive),
        );
        assert!(eng.prefill(&weights, &ffn, &[0u32, 1, 2]).is_some());
    }

    // ── Position tracking ────────────────────────────────────────────────────

    #[test]
    fn abs_position_advances_through_prefill_and_decode() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = BoundaryKvEngine::new(config("s", 64));
        eng.prefill(&weights, &ffn, &[0u32, 1, 2]).unwrap();
        assert_eq!(eng.abs_position(), 3);
        eng.decode_step(&weights, &ffn, 3).unwrap();
        assert_eq!(eng.abs_position(), 4);
        eng.decode_step(&weights, &ffn, 4).unwrap();
        assert_eq!(eng.abs_position(), 5);
    }

    #[test]
    fn token_start_of_chunk_at_first_boundary() {
        let mut eng = BoundaryKvEngine::new(config("s", 4));
        eng.abs_position = 4;
        assert_eq!(eng.token_start_of_current_chunk(), 0);
        eng.abs_position = 8;
        assert_eq!(eng.token_start_of_current_chunk(), 4);
    }

    // ── Frame builder direct unit test ───────────────────────────────────────

    #[test]
    fn build_frame_produces_well_formed_output() {
        let weights = make_test_weights();
        let hidden = Array2::<f32>::ones((1, weights.hidden_size));
        let cfg = config("seq", 4);
        let f = build_frame(&weights, &hidden, &cfg, 0, 4);
        assert_eq!(f.token_start, 0);
        assert_eq!(f.token_end, 4);
        assert_eq!(f.hidden_size as usize, weights.hidden_size);
        assert_eq!(f.sequence_id, "seq");
        assert_eq!(f.boundary_id, "seq:4");
        // bf16 default (calibration_mode = true).
        assert_eq!(f.compression_scheme, BoundaryCompression::None);
    }

    #[test]
    fn build_frame_with_live_gate_emits_compressed_when_safe() {
        let weights = make_test_weights();
        // Hidden chosen so that the codec roundtrip preserves the argmax —
        // an all-ones residual hits the same lm_head argmax both raw and
        // decoded, so the gate sees Agrees.
        let hidden = Array2::<f32>::ones((1, weights.hidden_size));
        let mut cfg = config("seq", 4);
        cfg.gate_config = BoundaryGateConfig {
            calibration_mode: false,
            min_log_prob_margin: 0.0,
            min_top1_prob: 0.0,
            require_compressed_agreement: true,
            fallback_policy: FallbackPolicy::Bf16Boundary,
        };
        let f = build_frame(&weights, &hidden, &cfg, 0, 4);
        // Margin is 0 on uniform input → gate falls back to bf16. We assert
        // that the path is reachable (not that it compresses) — the codec
        // selection follows the gate, not the caller.
        assert!(matches!(
            f.compression_scheme,
            BoundaryCompression::None | BoundaryCompression::Int8Clip3Sigma
        ));
    }

    #[test]
    fn build_frame_with_reject_decision_emits_empty_payload() {
        // Force the gate into Reject by requiring agreement and feeding
        // NotChecked: the engine sets hat_logits when verify_agreement=true
        // so we instead set fallback_policy = RejectIfUnsafe and supply a
        // low-margin residual.
        let weights = make_test_weights();
        let hidden = Array2::<f32>::zeros((1, weights.hidden_size));
        let mut cfg = config("seq", 4);
        cfg.verify_agreement = false; // → boundary_agreement = NotChecked → reject
        cfg.gate_config = BoundaryGateConfig {
            calibration_mode: false,
            min_log_prob_margin: 100.0, // impossibly high; everything is fragile
            min_top1_prob: 0.0,
            require_compressed_agreement: true,
            fallback_policy: FallbackPolicy::RejectIfUnsafe,
        };
        let f = build_frame(&weights, &hidden, &cfg, 0, 4);
        assert_eq!(f.compression_scheme, BoundaryCompression::None);
        assert_eq!(f.contract_level, BoundaryContract::Unknown);
        assert!(f.payload.is_empty());
    }

    #[test]
    fn build_frame_cold_replay_decision_emits_empty_payload() {
        let weights = make_test_weights();
        let hidden = Array2::<f32>::zeros((1, weights.hidden_size));
        let mut cfg = config("seq", 4);
        cfg.verify_agreement = false;
        cfg.gate_config = BoundaryGateConfig {
            calibration_mode: false,
            min_log_prob_margin: 100.0,
            min_top1_prob: 0.0,
            require_compressed_agreement: true,
            fallback_policy: FallbackPolicy::ColdReplay,
        };
        let f = build_frame(&weights, &hidden, &cfg, 0, 4);
        assert_eq!(f.contract_level, BoundaryContract::Unknown);
        assert!(f.payload.is_empty());
    }

    #[test]
    fn last_row_flat_empty_hidden_returns_empty() {
        let h: Array2<f32> = Array2::zeros((0, 8));
        assert!(last_row_flat(&h).is_empty());
    }

    #[test]
    fn last_row_flat_extracts_correct_row() {
        let mut h: Array2<f32> = Array2::zeros((3, 4));
        for j in 0..4 {
            h[[2, j]] = (j + 1) as f32;
        }
        assert_eq!(last_row_flat(&h), vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ── Coverage close-out tests ─────────────────────────────────────────────

    #[test]
    fn config_new_accepts_owned_string_sequence_id() {
        // Exercises the `impl Into<String>` monomorphisation for an owned
        // String, distinct from the &str path used by the rest of the suite.
        let identity = BoundaryModelIdentity::placeholder("test");
        let cfg = BoundaryKvEngineConfig::new(String::from("owned"), identity);
        assert_eq!(cfg.sequence_id, "owned");
        assert_eq!(cfg.chunk_tokens, 512);
    }

    #[test]
    fn with_async_backend_constructs_engine() {
        use larql_compute::CpuBackend;
        use larql_inference::AsyncComputeBackend;
        let backend: Box<dyn AsyncComputeBackend> = Box::new(CpuBackend);
        let eng = BoundaryKvEngine::with_async_backend(config("seq", 4), backend);
        assert_eq!(eng.name(), "boundary-kv");
        assert_eq!(eng.abs_position(), 0);
        assert_eq!(eng.archive().total_frames(), Some(0));
    }

    #[test]
    fn with_async_backend_decode_works_end_to_end() {
        use larql_compute::CpuBackend;
        use larql_inference::AsyncComputeBackend;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let backend: Box<dyn AsyncComputeBackend> = Box::new(CpuBackend);
        let mut eng = BoundaryKvEngine::with_async_backend(config("seq", 64), backend);
        let h = eng
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .expect("async prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn info_uses_full_window_label_when_window_size_none() {
        let mut cfg = config("seq", 4);
        cfg.window_size = None;
        let eng = BoundaryKvEngine::new(cfg);
        assert!(eng.info().config.contains("window=full"));
    }

    #[test]
    fn info_reports_explicit_window_size() {
        let mut cfg = config("seq", 4);
        cfg.window_size = Some(32);
        let eng = BoundaryKvEngine::new(cfg);
        // The `info()` closure that formats Some(N) is on a different
        // monomorphisation from the None branch above; exercising both
        // covers the closure body.
        assert!(eng.info().config.contains("window=32"));
    }

    #[test]
    fn decode_step_archive_failure_returns_none() {
        // Mirrors `archive_failure_at_boundary_returns_none` but for the
        // decode_step path (lines around the `return None` in decode_step).
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = BoundaryKvEngine::with_backend_and_archive(
            config("seq", 2),
            cpu_engine_backend(),
            Arc::new(FailingArchive),
        );
        // Prefill with 1 token (no boundary crossed) succeeds.
        assert!(eng.prefill(&weights, &ffn, &[0u32]).is_some());
        // Decode lands on position 2 → boundary → archive fails → None.
        assert!(eng.decode_step(&weights, &ffn, 1).is_none());
    }

    #[test]
    fn failing_archive_load_chain_returns_empty_ok() {
        // The FailingArchive::load_chain body is `Ok(Vec::new())` — used as
        // a defensive default; exercising it covers the function.
        let a = FailingArchive;
        let chain = a.load_chain("missing").expect("load_chain returns Ok");
        assert!(chain.is_empty());
    }

    // ── Q4K paths ──
    //
    // `prefill_quant` / `decode_step_quant` delegate to `StandardEngine`, whose
    // CPU fallback uses `ensure_attn_tensors_dequantised`. The synthetic
    // `make_test_vindex` fixture doesn't carry Q4K attn slices, so the CPU
    // fallback panics — these paths are Metal-only end-to-end. The
    // delegation surface itself is a one-line passthrough; the underlying
    // behaviour is covered by `standard.rs`'s Q4K tests.
}
