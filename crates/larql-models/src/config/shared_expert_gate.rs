//! The always-on shared expert's own gate.
//!
//! A mixture-of-experts layer may run a branch every token reads, beside
//! the router's top-k selection. Two lineages sum that branch in
//! differently, and the difference is a multiply this build must not
//! guess at:
//!
//! ```text
//! DeepSeek / Kimi:  out = routed(x) + shared(x)
//! Qwen2-MoE / 3.5:  out = routed(x) + sigmoid(shared_expert_gate(x)) * shared(x)
//! ```
//!
//! Transcribed from `Qwen2MoeSparseMoeBlock.forward` and
//! `Qwen3_5MoeSparseMoeBlock.forward`, which build
//! `shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)` and apply
//! `F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_output`.
//! Dropping it does not fail — it runs the shared branch at full weight
//! on every token, which is a plausible wrong answer, so the gate is a
//! declaration rather than something a reader infers from an operand it
//! happens to find.
//!
//! Deliberately NOT
//! [`AttentionGateSpec`](super::attention_gate::AttentionGateSpec): that
//! vocabulary is about where a gate sits in the attention pipeline
//! (`AttentionInput`, `AfterAggregationBeforeOutputProjection`) and none
//! of its placements describe an FFN branch. The two nonlinearity and
//! combination enums ARE shared, because those are the same facts.

use serde::{Deserialize, Serialize};

use super::attention_gate::{GateActivation, GateCombine};

/// The fully judged semantics of one shared-expert branch gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedExpertGateSpec {
    pub source: SharedExpertGateSource,
    pub activation: GateActivation,
    pub combine: GateCombine,
}

/// What the gate projection reads, and what it emits.
///
/// Single-variant today, and it grows only when a judged instance
/// actually differs — an unjudged difference must fail closed rather
/// than reuse the nearest variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SharedExpertGateSource {
    /// A dedicated `[1, hidden_size]` projection over the MoE block's
    /// input, emitting ONE logit per token. Its own operand — distinct
    /// from the shared expert's SwiGLU gate projection, which is
    /// `[shared_expert_intermediate_size, hidden_size]` and belongs to
    /// the branch's FFN rather than to this scalar.
    HiddenStateToScalar,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_serialises_snake_case_and_round_trips() {
        let spec = SharedExpertGateSpec {
            source: SharedExpertGateSource::HiddenStateToScalar,
            activation: GateActivation::Sigmoid,
            combine: GateCombine::ElementwiseMultiply,
        };
        let json = serde_json::to_string(&spec).expect("serialise");
        assert!(json.contains("hidden_state_to_scalar"), "{json}");
        assert_eq!(
            serde_json::from_str::<SharedExpertGateSpec>(&json).expect("round trip"),
            spec
        );
    }
}
