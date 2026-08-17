//! Attention sinks: the judged semantics of a per-head learned logit that
//! competes in the softmax without a value slot (GPT-OSS).
//!
//! The reference behaviour, transcribed from the served path
//! (`larql-compute::attention::softmax::softmax_in_place`) — one scalar
//! per **query** head, appended to that head's score row before the
//! softmax and dropped from the output afterwards:
//!
//! ```text
//! m      = max(max_k score[k], sink)
//! den    = Σ_k exp(score[k] − m) + exp(sink − m)
//! p[k]   = exp(score[k] − m) / den            # Σ_k p[k] = 1 − p_sink
//! out    = Σ_k p[k] · v[k]                     # the sink has no v
//! ```
//!
//! Single-variant on purpose, like the attention-gate spec: the enum
//! grows only when a judged instance actually differs (a sink with a
//! value slot, a per-kv-head sink, …), and an unjudged difference must
//! fail closed rather than reuse the nearest variant. A stack shipping a
//! `self_attn.sinks` operand under no judgment fails operand closure.

use serde::{Deserialize, Serialize};

/// The fully judged semantics of a layer's attention sinks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionSinkSpec {
    /// One learned logit per query head, joining the softmax denominator
    /// (and its max) only — no value row, so the real keys' weights sum
    /// to `1 − p_sink`. Operand shape `[num_q_heads]`.
    SoftmaxDenominator,
}
