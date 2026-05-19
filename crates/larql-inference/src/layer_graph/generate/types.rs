/// Sum of per-stage decode times across every successful step.
///
/// Dividing each field by `GenerateResult::decode_ms.len()` gives the
/// per-token average. Populated unconditionally — the six
/// `Instant::now()` calls per step are negligible next to the GPU
/// forward pass and the LM-head gemv.
#[derive(Debug, Default, Clone, Copy)]
pub struct StageTimings {
    pub embed_ms_total: f64,
    pub gpu_ms_total: f64,
    /// CPU fallback forward time when the backend lacks fused Q4 decode.
    pub cpu_fwd_ms_total: f64,
    /// Gate+up dispatch time within GPU fwd (populated when LARQL_PROFILE_SPLIT=1).
    pub gate_up_ms_total: f64,
    /// Activation+down+residual time within GPU fwd (populated when LARQL_PROFILE_SPLIT=1).
    pub down_ms_total: f64,
    pub norm_ms_total: f64,
    pub lm_head_ms_total: f64,
    pub detok_ms_total: f64,
    /// CPU-path-only: dequant time for Q4_K/Q6_K → f32 layer tensors.
    /// Lets the bench separate weight-unpack cost from gemm/attention.
    pub dequant_ms_total: f64,
}

/// Typed generation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerateError {
    UnsupportedBackend { reason: String },
    MissingWeights { reason: String },
    PromptTooLong { prompt_len: usize, max_len: usize },
    PrefillFailed { reason: String },
    EmptyOutput { reason: String },
    MaskRejectedAllCandidates,
    Other { reason: String },
}

impl GenerateError {
    pub fn unsupported_backend(reason: impl Into<String>) -> Self {
        Self::UnsupportedBackend {
            reason: reason.into(),
        }
    }

    pub fn missing_weights(reason: impl Into<String>) -> Self {
        Self::MissingWeights {
            reason: reason.into(),
        }
    }

    pub fn prompt_too_long(prompt_len: usize, max_len: usize) -> Self {
        Self::PromptTooLong {
            prompt_len,
            max_len,
        }
    }

    pub fn prefill_failed(reason: impl Into<String>) -> Self {
        Self::PrefillFailed {
            reason: reason.into(),
        }
    }

    pub fn empty_output(reason: impl Into<String>) -> Self {
        Self::EmptyOutput {
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for GenerateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenerateError::UnsupportedBackend { reason } => write!(f, "{reason}"),
            GenerateError::MissingWeights { reason } => write!(f, "{reason}"),
            GenerateError::PromptTooLong {
                prompt_len,
                max_len,
            } => write!(
                f,
                "prompt length {prompt_len} exceeds GPU KV cache capacity {max_len}"
            ),
            GenerateError::PrefillFailed { reason } => write!(f, "{reason}"),
            GenerateError::EmptyOutput { reason } => write!(f, "{reason}"),
            GenerateError::MaskRejectedAllCandidates => {
                write!(
                    f,
                    "constrained generation mask rejected every first-token candidate"
                )
            }
            GenerateError::Other { reason } => write!(f, "{reason}"),
        }
    }
}

impl std::error::Error for GenerateError {}

impl From<String> for GenerateError {
    fn from(reason: String) -> Self {
        Self::Other { reason }
    }
}

impl From<&str> for GenerateError {
    fn from(reason: &str) -> Self {
        Self::Other {
            reason: reason.to_string(),
        }
    }
}

/// Result of multi-token generation.
#[derive(Debug)]
pub struct GenerateResult {
    pub tokens: Vec<(String, f64)>,
    pub prefill_ms: f64,
    pub decode_ms: Vec<f64>,
    pub stage_timings: StageTimings,
    pub error: Option<GenerateError>,
}

impl StageTimings {
    /// Per-token average across `n` decode steps. Returns all-zero if
    /// `n == 0` (short-circuit no-decode paths safely).
    pub fn avg_per_step(&self, n: usize) -> StageTimings {
        if n == 0 {
            return Self::default();
        }
        let nf = n as f64;
        StageTimings {
            embed_ms_total: self.embed_ms_total / nf,
            gpu_ms_total: self.gpu_ms_total / nf,
            cpu_fwd_ms_total: self.cpu_fwd_ms_total / nf,
            gate_up_ms_total: self.gate_up_ms_total / nf,
            down_ms_total: self.down_ms_total / nf,
            norm_ms_total: self.norm_ms_total / nf,
            lm_head_ms_total: self.lm_head_ms_total / nf,
            detok_ms_total: self.detok_ms_total / nf,
            dequant_ms_total: self.dequant_ms_total / nf,
        }
    }
}

impl GenerateResult {
    pub fn empty_success() -> Self {
        Self {
            tokens: Vec::new(),
            prefill_ms: 0.0,
            decode_ms: Vec::new(),
            stage_timings: StageTimings::default(),
            error: None,
        }
    }

    pub fn empty_error(reason: impl Into<GenerateError>) -> Self {
        Self {
            tokens: Vec::new(),
            prefill_ms: 0.0,
            decode_ms: Vec::new(),
            stage_timings: StageTimings::default(),
            error: Some(reason.into()),
        }
    }

    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }

    pub fn into_result(mut self) -> Result<Self, GenerateError> {
        match self.error.take() {
            Some(err) => Err(err),
            None => Ok(self),
        }
    }

    pub fn error_message(&self) -> Option<String> {
        self.error.as_ref().map(ToString::to_string)
    }

    pub fn avg_decode_ms(&self) -> f64 {
        if self.decode_ms.is_empty() {
            0.0
        } else {
            self.decode_ms.iter().sum::<f64>() / self.decode_ms.len() as f64
        }
    }

    pub fn decode_tok_s(&self) -> f64 {
        let avg = self.avg_decode_ms();
        if avg > 0.0 {
            1000.0 / avg
        } else {
            0.0
        }
    }

    pub fn text(&self) -> String {
        self.tokens
            .iter()
            .map(|(t, _)| t.as_str())
            .collect::<Vec<_>>()
            .join("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── GenerateError constructors + Display ──────────────────────────────

    #[test]
    fn unsupported_backend_constructor_and_display() {
        let e = GenerateError::unsupported_backend("no metal");
        assert!(matches!(e, GenerateError::UnsupportedBackend { .. }));
        assert_eq!(format!("{e}"), "no metal");
    }

    #[test]
    fn missing_weights_constructor_and_display() {
        let e = GenerateError::missing_weights("lm_head absent");
        assert!(matches!(e, GenerateError::MissingWeights { .. }));
        assert_eq!(format!("{e}"), "lm_head absent");
    }

    #[test]
    fn prompt_too_long_constructor_and_display() {
        let e = GenerateError::prompt_too_long(8000, 4096);
        assert!(matches!(e, GenerateError::PromptTooLong { .. }));
        let s = format!("{e}");
        assert!(s.contains("8000"));
        assert!(s.contains("4096"));
    }

    #[test]
    fn prefill_failed_constructor_and_display() {
        let e = GenerateError::prefill_failed("OOM");
        assert!(matches!(e, GenerateError::PrefillFailed { .. }));
        assert_eq!(format!("{e}"), "OOM");
    }

    #[test]
    fn empty_output_constructor_and_display() {
        let e = GenerateError::empty_output("model halted");
        assert!(matches!(e, GenerateError::EmptyOutput { .. }));
        assert_eq!(format!("{e}"), "model halted");
    }

    #[test]
    fn mask_rejected_all_candidates_displays_stable_message() {
        let e = GenerateError::MaskRejectedAllCandidates;
        let s = format!("{e}");
        assert!(s.contains("mask rejected"));
    }

    #[test]
    fn other_variant_displays_reason() {
        let e = GenerateError::Other {
            reason: "unknown".into(),
        };
        assert_eq!(format!("{e}"), "unknown");
    }

    #[test]
    fn from_string_creates_other_variant() {
        let e: GenerateError = "boom".to_string().into();
        assert!(matches!(e, GenerateError::Other { .. }));
        assert_eq!(format!("{e}"), "boom");
    }

    #[test]
    fn from_str_creates_other_variant() {
        let e: GenerateError = "boom".into();
        assert!(matches!(e, GenerateError::Other { .. }));
        assert_eq!(format!("{e}"), "boom");
    }

    #[test]
    fn generate_error_is_std_error() {
        // Compile-time assertion: GenerateError implements std::error::Error.
        fn assert_error<E: std::error::Error>() {}
        assert_error::<GenerateError>();
    }

    // ── StageTimings ──────────────────────────────────────────────────────

    #[test]
    fn stage_timings_default_is_all_zero() {
        let t = StageTimings::default();
        assert_eq!(t.embed_ms_total, 0.0);
        assert_eq!(t.gpu_ms_total, 0.0);
        assert_eq!(t.lm_head_ms_total, 0.0);
    }

    #[test]
    fn stage_timings_avg_per_step_n_zero_returns_default() {
        let t = StageTimings {
            embed_ms_total: 10.0,
            gpu_ms_total: 20.0,
            ..Default::default()
        };
        let avg = t.avg_per_step(0);
        assert_eq!(avg.embed_ms_total, 0.0);
        assert_eq!(avg.gpu_ms_total, 0.0);
    }

    #[test]
    fn stage_timings_avg_per_step_divides_every_field() {
        let t = StageTimings {
            embed_ms_total: 10.0,
            gpu_ms_total: 20.0,
            cpu_fwd_ms_total: 30.0,
            gate_up_ms_total: 40.0,
            down_ms_total: 50.0,
            norm_ms_total: 60.0,
            lm_head_ms_total: 70.0,
            detok_ms_total: 80.0,
            dequant_ms_total: 90.0,
        };
        let avg = t.avg_per_step(10);
        assert_eq!(avg.embed_ms_total, 1.0);
        assert_eq!(avg.gpu_ms_total, 2.0);
        assert_eq!(avg.cpu_fwd_ms_total, 3.0);
        assert_eq!(avg.gate_up_ms_total, 4.0);
        assert_eq!(avg.down_ms_total, 5.0);
        assert_eq!(avg.norm_ms_total, 6.0);
        assert_eq!(avg.lm_head_ms_total, 7.0);
        assert_eq!(avg.detok_ms_total, 8.0);
        assert_eq!(avg.dequant_ms_total, 9.0);
    }

    // ── GenerateResult helpers ────────────────────────────────────────────

    #[test]
    fn empty_success_has_no_error_and_no_tokens() {
        let r = GenerateResult::empty_success();
        assert!(r.tokens.is_empty());
        assert!(r.error.is_none());
        assert!(!r.is_error());
        assert_eq!(r.error_message(), None);
    }

    #[test]
    fn empty_error_carries_reason() {
        let r = GenerateResult::empty_error("boom");
        assert!(r.is_error());
        assert_eq!(r.error_message().as_deref(), Some("boom"));
    }

    #[test]
    fn empty_error_accepts_typed_error() {
        let r = GenerateResult::empty_error(GenerateError::prompt_too_long(10, 5));
        assert!(matches!(r.error, Some(GenerateError::PromptTooLong { .. })));
    }

    #[test]
    fn into_result_returns_err_when_error_set() {
        let r = GenerateResult::empty_error("boom");
        let result = r.into_result();
        assert!(result.is_err());
        assert_eq!(format!("{}", result.unwrap_err()), "boom");
    }

    #[test]
    fn into_result_returns_ok_when_no_error() {
        let r = GenerateResult::empty_success();
        let result = r.into_result();
        assert!(result.is_ok());
    }

    #[test]
    fn avg_decode_ms_handles_empty_and_populated() {
        let r = GenerateResult {
            tokens: vec![],
            prefill_ms: 0.0,
            decode_ms: vec![],
            stage_timings: StageTimings::default(),
            error: None,
        };
        assert_eq!(r.avg_decode_ms(), 0.0);
        assert_eq!(r.decode_tok_s(), 0.0);

        let r = GenerateResult {
            tokens: vec![],
            prefill_ms: 0.0,
            decode_ms: vec![10.0, 20.0, 30.0],
            stage_timings: StageTimings::default(),
            error: None,
        };
        assert_eq!(r.avg_decode_ms(), 20.0);
        assert_eq!(r.decode_tok_s(), 50.0); // 1000/20
    }

    #[test]
    fn text_concatenates_token_strings() {
        let r = GenerateResult {
            tokens: vec![
                ("Hello".into(), 1.0),
                (", ".into(), 1.0),
                ("world".into(), 1.0),
                ("!".into(), 1.0),
            ],
            prefill_ms: 0.0,
            decode_ms: vec![],
            stage_timings: StageTimings::default(),
            error: None,
        };
        assert_eq!(r.text(), "Hello, world!");
    }

    #[test]
    fn text_empty_when_no_tokens() {
        let r = GenerateResult::empty_success();
        assert_eq!(r.text(), "");
    }
}
