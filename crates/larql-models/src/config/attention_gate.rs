//! Attention output gating — a generic attention primitive, judged per
//! model from its reference implementation, never inferred from shapes.
//!
//! The spec describes *what generic attention operation the gate operand
//! implements* — source, projection target, nonlinearity, combination and
//! placement — in vocabulary that covers any gated-attention architecture
//! without naming one. The first judged instance (from the upstream
//! Transformers source):
//!
//! ```text
//! gate = sigmoid(gate_proj(attention_input))   # normalized layer input
//! attn_output = attn_output * gate             # after head aggregation
//! attn_output = o_proj(attn_output)            # before output projection
//! ```
//!
//! Every enum here is intentionally single-variant today: each variant
//! set grows only when a new judged instance actually differs, and an
//! unjudged difference must fail closed rather than reuse the nearest
//! variant.

use serde::{Deserialize, Serialize};

/// The fully judged semantics of one attention output gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionGateSpec {
    pub source: GateSource,
    pub activation: GateActivation,
    pub combine: GateCombine,
    pub placement: GatePlacement,
}

impl AttentionGateSpec {
    /// The second judged instance, Kimi-K3's MLA output gate
    /// (`mla_use_output_gate: true`, `modeling_kimi_linear.py`):
    ///
    /// ```text
    /// g = self.g_proj(hidden_states).sigmoid()   # the block's normalised input
    /// attn_output = attn_output * g              # after head aggregation
    /// attn_output = self.o_proj(attn_output)     # before the output projection
    /// ```
    ///
    /// Read from the reference line by line, never inferred from the
    /// `g_proj` operand's shape — which on K3 is byte-identical to the
    /// KDA layers' full-rank gate at `[12288, 7168]`. Stated once here so
    /// the resolved record and any executor that reads the spec cannot
    /// disagree on what the gate does.
    pub const fn from_attention_input_sigmoid_before_output_projection() -> Self {
        Self {
            source: GateSource::AttentionInput,
            activation: GateActivation::Sigmoid,
            combine: GateCombine::ElementwiseMultiply,
            placement: GatePlacement::AfterAggregationBeforeOutputProjection,
        }
    }
}

/// What the gate projection reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateSource {
    /// The attention block's input hidden state — already normalised,
    /// because the decoder layer norms before calling attention. The gate
    /// has its own projection operand.
    AttentionInput,
    /// The **second half of each head's query projection**. No separate
    /// gate operand exists: `q_proj` emits `2 · head_dim` rows per head
    /// and the op reads one matrix for both roles, the same way a K≡V
    /// layer reads one matrix for key and value.
    ///
    /// The layout is per-head interleaved, and that is the whole reason
    /// this is a variant rather than a flag:
    ///
    /// ```text
    /// [ q_h0 | gate_h0 | q_h1 | gate_h1 | … ]      <- Qwen3.8
    /// [ q_h0 | q_h1 | … | gate_h0 | gate_h1 | … ]  <- NOT this
    /// ```
    ///
    /// Both readings have identical dimensions, so nothing about the
    /// tensor shape distinguishes them; only the reference implementation
    /// does. Transcribed from HF `Qwen3_5Attention.forward`, which views
    /// the projection as `(…, heads, 2 · head_dim)` and chunks the LAST
    /// axis. The gate half bypasses `q_norm` and the rotary — both apply
    /// to the query half alone.
    FusedQueryProjection,
}

/// The gate nonlinearity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateActivation {
    Sigmoid,
}

/// How the gate combines with the attention output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateCombine {
    ElementwiseMultiply,
}

/// Where the gate applies in the attention pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GatePlacement {
    /// After head outputs are aggregated/flattened, before `o_proj`.
    AfterAggregationBeforeOutputProjection,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The second judged instance — Kimi-K3's MLA output gate — reads
    /// the block's normalised INPUT, unlike the first instance's fused
    /// query projection; the other three axes are the same generic
    /// operation. Pinned so the constructor cannot drift into the
    /// nearest existing variant.
    #[test]
    fn the_k3_mla_output_gate_is_the_attention_input_instance() {
        let spec = AttentionGateSpec::from_attention_input_sigmoid_before_output_projection();
        assert_eq!(spec.source, GateSource::AttentionInput);
        assert_eq!(spec.activation, GateActivation::Sigmoid);
        assert_eq!(spec.combine, GateCombine::ElementwiseMultiply);
        assert_eq!(
            spec.placement,
            GatePlacement::AfterAggregationBeforeOutputProjection
        );
        let fused = AttentionGateSpec {
            source: GateSource::FusedQueryProjection,
            ..spec
        };
        assert_ne!(
            spec, fused,
            "the two judged instances differ in source alone"
        );
        let json = serde_json::to_value(spec).unwrap();
        assert_eq!(json["source"], "attention_input");
    }

    #[test]
    fn spec_serialises_snake_case_and_round_trips() {
        let spec = AttentionGateSpec {
            source: GateSource::AttentionInput,
            activation: GateActivation::Sigmoid,
            combine: GateCombine::ElementwiseMultiply,
            placement: GatePlacement::AfterAggregationBeforeOutputProjection,
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains("\"sigmoid\""));
        assert!(json.contains("\"attention_input\""));
        let back: AttentionGateSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back, spec);
    }
}
