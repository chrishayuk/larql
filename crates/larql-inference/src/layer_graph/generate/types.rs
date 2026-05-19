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
    /// Gate+up dispatch time within GPU fwd (populated when LARQL_PROFILE_SPLIT=1).
    pub gate_up_ms_total: f64,
    /// Activation+down+residual time within GPU fwd (populated when LARQL_PROFILE_SPLIT=1).
    pub down_ms_total: f64,
    pub norm_ms_total: f64,
    pub lm_head_ms_total: f64,
    pub detok_ms_total: f64,
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
            gate_up_ms_total: self.gate_up_ms_total / nf,
            down_ms_total: self.down_ms_total / nf,
            norm_ms_total: self.norm_ms_total / nf,
            lm_head_ms_total: self.lm_head_ms_total / nf,
            detok_ms_total: self.detok_ms_total / nf,
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn generate_result_into_result_returns_typed_error() {
        let result = GenerateResult::empty_error(GenerateError::unsupported_backend("no Q4"));
        let err = result.into_result().expect_err("expected typed error");
        assert!(matches!(err, GenerateError::UnsupportedBackend { .. }));
        assert_eq!(err.to_string(), "no Q4");
    }

    #[test]
    fn prompt_too_long_error_formats_capacity() {
        let err = GenerateError::prompt_too_long(17, 16);
        assert_eq!(
            err.to_string(),
            "prompt length 17 exceeds GPU KV cache capacity 16"
        );
    }
}
