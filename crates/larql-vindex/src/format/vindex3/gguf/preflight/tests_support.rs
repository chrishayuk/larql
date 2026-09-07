//! A Qwen3.8-shaped execution surface, shared by the gguf gates.
//!
//! Deliberately complete: the first version omitted `attention` and
//! `ffn`, and the preflight refused it — correctly, since a qwen35
//! hybrid attends in every fourth layer. Keeping it whole means a test
//! that fails is telling you about the code, not about the fixture.

use crate::format::vindex3::graph::surface::ExecutionSurface;

pub fn qwen_shaped_surface() -> ExecutionSurface {
    use crate::format::vindex3::graph::surface::{
        AttentionSurface, FfnSurface, LinearAttentionSurface, NormSurface,
    };
    use larql_models::config::{
        Activation, FfnType, NormSpec, NormType, ParameterFreeQkNorm, QkNormScope,
    };
    ExecutionSurface {
        context_length: Some(262_144),
        // A qwen35 hybrid attends in every fourth layer, so a surface
        // with no attention or FFN is not one — the preflight said so
        // when this fixture first omitted them, which is the gate
        // working on its author.
        attention: Some(AttentionSurface {
            num_q_heads: 24,
            num_kv_heads: 4,
            head_dim: 256,
            query_scale: None,
            score_scale: 0.0625,
            logit_softcapping: None,
            qk_norm_scope: QkNormScope::PerHead,
            qk_norm_weight_offset: 1.0,
            parameter_free_qk_norm: ParameterFreeQkNorm::default(),
            output_gate: None,
            sinks: None,
            attention_bias: Some(false),
        }),
        ffn: Some(FfnSurface {
            intermediate_size: Some(17408),
            intermediate_size_by_layer: None,
            activation: Activation::Silu,
            ffn_type: FfnType::Gated,
            gate_policy: larql_models::ExpertGatePolicy::default(),
            moe: None,
        }),
        norm: NormSurface {
            pre: NormSpec {
                kind: NormType::RmsNorm,
                eps: 1e-6,
                weight_offset: 1.0,
            },
            post: None,
            final_norm: NormSpec {
                kind: NormType::RmsNorm,
                eps: 1e-6,
                weight_offset: 1.0,
            },
            placement: None,
        },
        head: None,
        residual_scale: None,
        residual_topology: larql_models::config::ResidualTopology::SingleStream,
        residual_in_fp32: None,
        linear_attention: Some(LinearAttentionSurface {
            key_heads: 16,
            key_head_dim: 128,
            value_heads: 48,
            value_head_dim: 128,
            conv_kernel: 4,
            state_dtype: Some(larql_models::inventory::report::RecurrentStateDtype::Float32),
        }),
        kda: None,
        kda_gate_lower_bound: None,
        kda_gate_form: None,
        kda_use_full_rank_gate: None,
        mla: None,
        conv_qkv: None,
        mamba2: None,
    }
}
