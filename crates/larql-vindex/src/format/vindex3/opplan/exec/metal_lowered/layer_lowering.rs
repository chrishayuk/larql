//! `LoweredSession::layer_lowering`: one layer's lowered op plan.

use crate::format::vindex3::graph::policy::AttentionSpan;
use crate::format::vindex3::opplan::LayerPlan;
use larql_compute_metal::lowering::attention::{
    AttnShape, AttnWeights, LoweredPosition, QkNormWeights,
};
use larql_compute_metal::lowering::ffn::{FfnActivation, FfnShape, FfnWeights};
use larql_compute_metal::lowering::stack::{
    HybridFfnLowering, LayerFfnLowering, LayerLowering, RoutedFfnLowering,
};
use larql_compute_metal::lowering::{DeviceBuffer, PostNorm};
use larql_models::config::PositionPolicy;
use routed::FfnResident;

#[allow(unused_imports)]
use super::*;

impl<'a> LoweredSession<'a> {
    pub(super) fn layer_lowering<'b>(
        &'b self,
        plan_layer: &'b LayerPlan,
        r: &'b LayerResident,
        t: usize,
    ) -> LayerLowering<'b> {
        let a = plan_layer.attention.softmax().unwrap_or_else(|| {
            panic!(
                "layer {} is not softmax; the lowering refused this plan in `new`",
                plan_layer.layer
            )
        });
        let post = |slot: &'b Option<(DeviceBuffer, f32, f32)>, scratch: &'b DeviceBuffer| {
            slot.as_ref().map(|(w, eps, off)| PostNorm {
                weight: w,
                eps: *eps,
                weight_offset: *off,
                scratch,
            })
        };
        LayerLowering {
            attn: AttnWeights {
                q: r.q.as_lowered(),
                k: r.k.as_lowered(),
                v: r.v.as_lowered(),
                o: r.o.as_lowered(),
                gate: r
                    .gate
                    .as_ref()
                    .filter(|_| !self.ablate.no_gate)
                    .map(DeviceMatrix::as_lowered),
                q_bias: r.q_bias.as_ref(),
                k_bias: r.k_bias.as_ref(),
                v_bias: r.v_bias.as_ref(),
                o_bias: r.o_bias.as_ref(),
                sinks: r.sinks.as_ref(),
                qk_norm: r.qk_norm.as_ref().filter(|_| !self.ablate.no_qk_norm).map(
                    |(q, k, offset)| QkNormWeights {
                        q,
                        k,
                        weight_offset: *offset,
                    },
                ),
                norm_weight: &r.pre_attn_norm,
                post_norm: post(&r.post_attn_norm, &self.scratch[7])
                    .filter(|_| !self.ablate.no_post_norms),
            },
            attn_shape: AttnShape {
                hidden: self.hidden,
                num_q_heads: a.num_q_heads,
                num_kv_heads: a.num_kv_heads,
                head_dim: a.head_dim,
                norm_eps: plan_layer.declared_norm_eps as f32,
                // The weight offset of the norm that conditions the
                // attention input. Under post-norm placement no norm does,
                // so the offset that would scale nothing is the identity.
                norm_weight_offset: plan_layer
                    .pre_attention_norm
                    .as_ref()
                    .map_or(0.0, |n| n.weight_offset),
                // The component's declared epsilon, which QK norm runs at.
                // Read from the layer's own field rather than off a norm
                // site that a post-norm stack does not carry.
                qk_norm_eps: plan_layer.declared_norm_eps as f32,
                parameter_free_q: a.parameter_free_qk_norm.q && !self.ablate.no_qk_norm,
                parameter_free_k: a.parameter_free_qk_norm.k && !self.ablate.no_qk_norm,
                parameter_free_v: a.parameter_free_qk_norm.v && !self.ablate.no_qk_norm,
                query_scale: a
                    .query_scale
                    .map(|s| s as f32)
                    .filter(|_| !self.ablate.no_query_scale),
                score_scale: a.score_scale as f32,
                position: match a.position {
                    _ if self.ablate.no_rope => LoweredPosition::None,
                    PositionPolicy::Rope { theta } => LoweredPosition::Rope { theta },
                    // Refused above, before the session exists.
                    PositionPolicy::MRope { .. } => {
                        unreachable!("M-RoPE is refused before the session is built")
                    }
                    // YaRN's ramped `inv_freq` rides the shared table
                    // (built for this layer's policy in `new`); the
                    // amplitude rides slot 6 of the rope kernel.
                    PositionPolicy::Yarn { theta, scaling } => {
                        let amplitude =
                            crate::format::vindex3::opplan::exec::kernels::yarn_frequencies(
                                &scaling, a.head_dim, theta,
                            )
                            .1;
                        LoweredPosition::Scaled { theta, amplitude }
                    }
                    // Llama-3 rides the same shared table as YaRN — the
                    // per-layer `inv_freq` built in `new` — but at unit
                    // amplitude: the family adjusts frequencies only.
                    // Written as an explicit 1.0 rather than reusing
                    // YaRN's arm, so an amplitude can never be inherited
                    // by a family that does not define one.
                    PositionPolicy::Llama3 { theta, .. } => LoweredPosition::Scaled {
                        theta,
                        amplitude: 1.0,
                    },
                    // Linear rides the shared table too — its `inv_freq`
                    // is the plain series divided by the factor, built in
                    // `new` — at unit amplitude, written explicitly for
                    // the reason Llama-3's is.
                    PositionPolicy::Linear { theta, .. } => LoweredPosition::Scaled {
                        theta,
                        amplitude: 1.0,
                    },
                    // No lowering exists for a relative scheme. It
                    // lowers to `None` — no rotation — and the executor
                    // refuses rather than running it unpositioned, so the
                    // absence is never mistaken for NoPE downstream.
                    PositionPolicy::Relative { .. } => LoweredPosition::None,
                    PositionPolicy::None => LoweredPosition::None,
                    // The proportional table (zeros above the fraction)
                    // rides this layer's own inv_freq at unit amplitude;
                    // the rotary-width basis was refused in `new`.
                    PositionPolicy::PartialRope { theta, .. } => LoweredPosition::Scaled {
                        theta,
                        amplitude: 1.0,
                    },
                },
                // A window applies only to a sliding span; a full layer
                // attends the whole prefix whatever the plan records.
                window: match a.span {
                    AttentionSpan::Sliding => a.window,
                    _ => None,
                },
                softcap: a.logit_softcapping,
                residual_scale: plan_layer.residual_scale,
                position_index: t,
                kv_len: t + 1,
            },
            ffn: match &r.ffn {
                FfnResident::Dense { gate, up, down } => LayerFfnLowering::Dense {
                    weights: FfnWeights {
                        gate: gate.as_lowered(),
                        up: up.as_lowered(),
                        down: down.as_lowered(),
                        norm_weight: &r.pre_ffn_norm,
                        post_norm: post(&r.post_ffn_norm, &self.scratch[14])
                            .filter(|_| !self.ablate.no_post_norms),
                    },
                    shape: FfnShape {
                        hidden: self.hidden,
                        intermediate: plan_layer
                            .ffn
                            .as_ref()
                            .and_then(|f| f.dense())
                            .map_or(self.hidden, |f| f.intermediate_size),
                        norm_eps: plan_layer
                            .pre_ffn_norm
                            .as_ref()
                            .expect("dense resident implies a pre-FFN norm")
                            .eps as f32,
                        norm_weight_offset: plan_layer
                            .pre_ffn_norm
                            .as_ref()
                            .expect("dense resident implies a pre-FFN norm")
                            .weight_offset,
                        activation: plan_layer.ffn.as_ref().and_then(|f| f.dense()).map_or(
                            FfnActivation::Silu,
                            |f| {
                                ffn_activation(f.activation, f.gate_policy)
                                    .expect("checked in `new`")
                            },
                        ),
                        residual_scale: plan_layer.residual_scale,
                    },
                    layer_scale: r.layer_scale,
                },
                FfnResident::Routed(routed) => {
                    LayerFfnLowering::Routed(Box::new(RoutedFfnLowering {
                        moe: routed.moe(),
                        scratch: &routed.scratch,
                        table: &routed.table,
                        eps: routed.eps,
                    }))
                }
                FfnResident::Hybrid(h) => {
                    let op = plan_layer
                        .ffn
                        .as_ref()
                        .and_then(|f| f.hybrid())
                        .expect("resident matches the plan");
                    LayerFfnLowering::Hybrid(Box::new(HybridFfnLowering {
                        dense: FfnWeights {
                            gate: h.gate.as_lowered(),
                            up: h.up.as_lowered(),
                            down: h.down.as_lowered(),
                            norm_weight: &r.pre_ffn_norm,
                            // The hybrid applies the layer's post-FFN norm
                            // itself, after summing the branches.
                            post_norm: None,
                        },
                        dense_shape: FfnShape {
                            hidden: self.hidden,
                            intermediate: op.dense.intermediate_size,
                            norm_eps: plan_layer
                                .pre_ffn_norm
                                .as_ref()
                                .expect("hybrid resident implies a pre-FFN norm")
                                .eps as f32,
                            norm_weight_offset: plan_layer
                                .pre_ffn_norm
                                .as_ref()
                                .expect("hybrid resident implies a pre-FFN norm")
                                .weight_offset,
                            activation: ffn_activation(op.dense.activation, op.dense.gate_policy)
                                .expect("checked in `new`"),
                            residual_scale: plan_layer.residual_scale,
                        },
                        routed: RoutedFfnLowering {
                            moe: h.routed.moe(),
                            scratch: &h.routed.scratch,
                            table: &h.routed.table,
                            eps: h.routed.eps,
                        },
                        router_conditioning: &h.router_conditioning,
                        per_expert_scale: &h.per_expert_scale,
                        pre_experts_norm: &h.pre_experts_norm,
                        post_dense_norm: &h.post_dense_norm,
                        post_experts_norm: &h.post_experts_norm,
                        branch_norm_eps: h.branch_norm_eps,
                        branch_norm_weight_offset: h.branch_norm_weight_offset,
                        post_ffn_norm: post(&r.post_ffn_norm, &self.scratch[14])
                            .filter(|_| !self.ablate.no_post_norms),
                        layer_scale: r.layer_scale,
                    }))
                }
            },
            k_cache: &r.k_cache,
            v_cache: &r.v_cache,
            inv_freq: r
                .rope_key
                .and_then(|k| self.inv_freq.get(&k))
                .unwrap_or(&self.scratch[0]),
        }
    }
}
