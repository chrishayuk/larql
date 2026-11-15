//! The model the graph declares, checked against the model the runtime
//! would build.
//!
//! llama.cpp reads geometry from keys but builds a good deal of behaviour
//! into its graph code: the FFN activation, the attention scale, the
//! embedding scale, which layers slide. [`read_programme`] reduces the
//! operation plan to the facts the metadata states, and refuses wherever
//! the plan declares something the target's built-in behaviour would not
//! reproduce. A refusal names both sides.

use larql_models::config::{is_full_at_period, PositionPolicy};

use super::target::DenseTarget;
use crate::format::vindex3::graph::policy::AttentionSpan;
use crate::format::vindex3::opplan::{ComponentOpPlan, LayerAttention, LayerFfn};

/// Tolerance for comparing a declared scale with the runtime's rule. The
/// graph stores scales as decimal floats (`embed_scale: 50.596443`), so
/// exact equality would refuse the real container over its sixth decimal.
pub const SCALE_TOLERANCE: f64 = 1e-4;

/// Rotary position as the target states it: a base and an optional
/// linear scaling factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeFacts {
    pub theta: f64,
    pub linear_factor: Option<f64>,
}

/// The sliding-window layers' shared facts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlidingFacts {
    pub window: usize,
    pub theta: f64,
    /// Every `period`-th layer attends fully.
    pub period: usize,
}

/// Everything the metadata states about the attention stack.
#[derive(Debug, Clone, PartialEq)]
pub struct Programme {
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub full_rope: RopeFacts,
    pub sliding: Option<SlidingFacts>,
}

fn rope_facts(layer: usize, p: &PositionPolicy) -> Result<RopeFacts, String> {
    match *p {
        PositionPolicy::Rope { theta } => Ok(RopeFacts {
            theta,
            linear_factor: None,
        }),
        PositionPolicy::Linear { theta, factor } => Ok(RopeFacts {
            theta,
            linear_factor: Some(factor),
        }),
        ref other => Err(format!(
            "layer {layer} declares position {other:?} — the dense GGUF lowering states plain \
             or linearly-scaled RoPE only"
        )),
    }
}

fn agree<T: PartialEq + std::fmt::Debug>(
    what: &str,
    seen: &mut Option<T>,
    value: T,
) -> Result<(), String> {
    match seen {
        None => {
            *seen = Some(value);
            Ok(())
        }
        Some(prev) if *prev == value => Ok(()),
        Some(prev) => Err(format!(
            "{what} differs across layers ({prev:?} vs {value:?}) — one GGUF key cannot state both"
        )),
    }
}

fn close(declared: f64, rule: f64) -> bool {
    (declared - rule).abs() <= SCALE_TOLERANCE * rule.abs().max(1.0)
}

/// The sliding period the layer table follows, or a refusal.
fn sliding_period(full_layers: &[usize], num_layers: usize) -> Result<usize, String> {
    let period = full_layers.first().map(|f| f + 1).ok_or_else(|| {
        "every layer slides — the target needs at least one full-attention layer to derive \
         its period"
            .to_string()
    })?;
    let regular = (0..num_layers).all(|l| is_full_at_period(l, period) == full_layers.contains(&l));
    if !regular {
        return Err(format!(
            "full-attention layers {full_layers:?} follow no single period — the target \
             describes interleaving by period only"
        ));
    }
    Ok(period)
}

/// Read the plan into a [`Programme`], refusing whatever the target's
/// built-in behaviour would not reproduce. Every refusal is collected.
pub fn read_programme(
    plan: &ComponentOpPlan,
    target: &DenseTarget,
    hidden_size: usize,
) -> Result<Programme, Vec<String>> {
    let mut refusals = Vec::new();
    let mut geometry = None;
    let mut inter = None;
    let mut full_rope = None;
    let mut sliding = None;
    let mut full_layers = Vec::new();

    if let Some(e) = &plan.embedding {
        let rule = target.embedding_scale.value(hidden_size);
        match e.scale {
            Some(s) if close(s as f64, rule) => {}
            other => refusals.push(format!(
                "the embedding scale is {other:?}; {} multiplies by {rule} unconditionally",
                target.architecture
            )),
        }
        if e.norm.is_some() {
            refusals.push("the embedding carries a norm the target has no op for".into());
        }
    }
    if let Some(o) = &plan.output {
        if o.multiplier.is_some() {
            refusals.push("the output head carries a multiplier the target has no key for".into());
        }
    }

    for layer in &plan.layers {
        let l = layer.layer;
        if layer.residual_scale.is_some() || layer.layer_scale.is_some() {
            refusals.push(format!(
                "layer {l} scales its residual; the target does not"
            ));
        }
        if let LayerAttention::Softmax(a) = &layer.attention {
            if let Err(e) = agree(
                "attention geometry",
                &mut geometry,
                (a.num_q_heads, a.num_kv_heads, a.head_dim),
            ) {
                refusals.push(e);
            }
            let rule = target.attention_scale.value(a.head_dim);
            if !close(a.score_scale, rule) {
                refusals.push(format!(
                    "layer {l} scales scores by {}; {} uses {rule}",
                    a.score_scale, target.architecture
                ));
            }
            if a.logit_softcapping.is_some() {
                refusals.push(format!(
                    "layer {l} soft-caps attention logits; the target does not"
                ));
            }
            if a.output_gate.is_some() || a.sinks.is_some() || a.v_from_k {
                refusals.push(format!(
                    "layer {l} carries an attention gate, sinks or K-as-V the target has no op for"
                ));
            }
            if [&a.q_bias, &a.k_bias, &a.v_bias, &a.o_bias]
                .iter()
                .any(|b| b.is_some())
            {
                refusals.push(format!(
                    "layer {l} carries attention biases the target has no name for"
                ));
            }
            let rope = match rope_facts(l, &a.position) {
                Ok(r) => r,
                Err(e) => {
                    refusals.push(e);
                    continue;
                }
            };
            let result = match a.span {
                AttentionSpan::Full => {
                    full_layers.push(l);
                    agree("full-attention RoPE", &mut full_rope, rope)
                }
                AttentionSpan::Sliding => match (a.window, rope.linear_factor) {
                    (Some(w), None) => {
                        agree("sliding window and RoPE", &mut sliding, (w, rope.theta))
                    }
                    _ => Err(format!(
                        "layer {l} slides without a window, or scales its sliding RoPE — the \
                         target states neither"
                    )),
                },
                other => Err(format!(
                    "layer {l} declares span {other:?}, which the target cannot state"
                )),
            };
            if let Err(e) = result {
                refusals.push(e);
            }
        }
        if let Some(LayerFfn::Dense(f)) = &layer.ffn {
            if f.activation != target.activation || f.gate.is_none() {
                refusals.push(format!(
                    "layer {l} runs a {:?} FFN (gated: {}); {} runs a gated {:?} FFN",
                    f.activation,
                    f.gate.is_some(),
                    target.architecture,
                    target.activation
                ));
            }
            if let Err(e) = agree("FFN width", &mut inter, f.intermediate_size) {
                refusals.push(e);
            }
        }
    }

    let sliding = match sliding {
        None => None,
        Some((window, theta)) => match sliding_period(&full_layers, plan.layers.len()) {
            Ok(period) => Some(SlidingFacts {
                window,
                theta,
                period,
            }),
            Err(e) => {
                refusals.push(e);
                None
            }
        },
    };
    match (geometry, inter, full_rope) {
        (Some((q, kv, hd)), Some(inter), Some(full_rope)) if refusals.is_empty() => Ok(Programme {
            num_q_heads: q,
            num_kv_heads: kv,
            head_dim: hd,
            intermediate_size: inter,
            full_rope,
            sliding,
        }),
        (g, i, r) => {
            if g.is_none() || i.is_none() || r.is_none() {
                refusals.push(
                    "the plan declares no softmax attention, no dense FFN, or no full-attention \
                     layer — nothing for the target's geometry keys to state"
                        .into(),
                );
            }
            Err(refusals)
        }
    }
}
