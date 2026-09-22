//! Norm-aware local replay of a softmax attention head mixture.
//!
//! This is an evidence path, not a second attention implementation. Q/K/V,
//! softmax and weighted-V aggregation have already run in canonical execution;
//! the caller supplies those captured head values. Replay starts exactly where
//! a head intervention starts: concatenate one value per query head, execute
//! the prepared image's effective `W_O`, then the declared post-attention norm,
//! residual-delta scale and single-stream residual add.

use super::backend::{PlanBackend, ProjectCall};
use super::prepared::{PreparedAttention, PreparedOperands};
use super::{scale_residual_delta, ComponentOpPlan};
use crate::error::VindexError;
use larql_compute::attention::softmax::softmax_in_place_f32;

/// Exact production-order replay of one softmax head from conditioned Q/K/V.
#[derive(Debug, Clone)]
pub struct SourceHeadReplay {
    pub weights: Vec<f32>,
    pub values: Vec<f32>,
}

/// Rerun one head's actual score scaling, optional softcap, softmax and
/// weighted-V aggregation. Q is supplied separately so GW-KEY-1 can keep it
/// natural while intervening independently on source K and V vectors.
pub fn replay_softmax_source_head(
    query: &[f32],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    score_scale: f64,
    logit_softcapping: Option<f32>,
) -> Result<SourceHeadReplay, VindexError> {
    let width = query.len();
    if width == 0
        || keys.is_empty()
        || keys.len() != values.len()
        || keys
            .iter()
            .chain(values)
            .any(|row| row.len() != width || row.iter().any(|value| !value.is_finite()))
        || query.iter().any(|value| !value.is_finite())
        || !score_scale.is_finite()
    {
        return Err(VindexError::Parse(
            "source-head replay requires aligned finite nonempty Q/K/V".into(),
        ));
    }
    let mut weights: Vec<f32> = keys
        .iter()
        .map(|key| {
            let dot: f32 = query.iter().zip(key).map(|(a, b)| a * b).sum();
            let scaled = dot * score_scale as f32;
            match logit_softcapping {
                Some(cap) => cap * (scaled / cap).tanh(),
                None => scaled,
            }
        })
        .collect();
    softmax_in_place_f32(&mut weights);
    let mut output = vec![0.0f32; width];
    for (weight, value) in weights.iter().zip(values) {
        for (acc, source) in output.iter_mut().zip(value) {
            *acc += *weight * source;
        }
    }
    Ok(SourceHeadReplay {
        weights,
        values: output,
    })
}

/// The exact local path's outputs for one head mixture.
#[derive(Debug, Clone)]
pub struct AttentionHeadReplay {
    /// One effective-`W_O` projection per query head, before the post norm.
    pub contributions: Vec<Vec<f32>>,
    /// L2 norm of each entry in [`Self::contributions`].
    pub contribution_norms: Vec<f64>,
    /// `W_O × concat(head_values)`, before post-attention norm.
    pub raw_attention_output: Vec<f32>,
    /// The branch delta after post-attention norm and residual scaling.
    pub applied_delta: Vec<f32>,
    /// The single residual carrier after the delta is added.
    pub carrier_after: Vec<f32>,
    /// Sum-of-head-contributions versus the one-shot `W_O` projection.
    pub raw_reconstruction_relative_l2: f64,
}

/// The exact local path without the eight additional decomposition
/// projections. GW-HEAD-1 uses this for exhaustive subset execution after the
/// structural decomposition gate has already passed.
#[derive(Debug, Clone)]
pub struct AttentionMixtureReplay {
    pub raw_attention_output: Vec<f32>,
    pub applied_delta: Vec<f32>,
    pub carrier_after: Vec<f32>,
}

fn relative_l2(actual: &[f32], reconstructed: &[f64]) -> f64 {
    let error = actual
        .iter()
        .zip(reconstructed)
        .map(|(&a, &b)| (f64::from(a) - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm = actual
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        error
    } else {
        error / norm
    }
}

/// One replay site, resolved and admitted once: the plan layer, its softmax
/// op, the prepared attention operands, and the widths both replays use.
struct ReplaySite<'p> {
    layer_plan: &'p super::super::LayerPlan,
    prepared_layer: &'p super::prepared::PreparedLayer,
    operands: &'p super::AttentionOperands,
    num_q_heads: usize,
    head_dim: usize,
    hidden: usize,
    width: usize,
}

/// Admit a replay: a single-stream image, a finite hidden-width carrier, a
/// softmax layer without output gating or bias, one finite value per query
/// head, a prepared layer inside the executed range, and a `W_O` of the
/// declared shape. Every refusal happens here, before any projection.
fn resolve_site<'p>(
    plan: &'p ComponentOpPlan,
    prepared: &'p PreparedOperands,
    layer: usize,
    entering_carrier: &[f32],
    head_values: &[Vec<f32>],
) -> Result<ReplaySite<'p>, VindexError> {
    if prepared.carries_hyper_connection() || prepared.carries_attention_residual() {
        return Err(VindexError::Parse(
            "attention-head replay v1 requires a single-stream carrier".into(),
        ));
    }
    let hidden = prepared.hidden();
    if entering_carrier.len() != hidden || entering_carrier.iter().any(|v| !v.is_finite()) {
        return Err(VindexError::Parse(
            "attention-head replay requires one finite hidden-width entering carrier".into(),
        ));
    }
    let layer_plan = plan.layers.get(layer).ok_or_else(|| {
        VindexError::Parse(format!(
            "attention-head replay layer {layer} is outside the plan"
        ))
    })?;
    let op = layer_plan.attention.softmax().ok_or_else(|| {
        VindexError::Parse(format!(
            "attention-head replay layer {layer} is not softmax"
        ))
    })?;
    if op.output_gate.is_some() || op.o_bias.is_some() {
        return Err(VindexError::Parse(
            "attention-head replay v1 does not admit output gating or output bias".into(),
        ));
    }
    if head_values.len() != op.num_q_heads
        || head_values
            .iter()
            .any(|head| head.len() != op.head_dim || head.iter().any(|v| !v.is_finite()))
    {
        return Err(VindexError::Parse(format!(
            "attention-head replay layer {layer} requires {} finite heads of width {}",
            op.num_q_heads, op.head_dim
        )));
    }
    let range = prepared.executed_layers();
    let local = layer.checked_sub(range.start).ok_or_else(|| {
        VindexError::Parse(format!("layer {layer} precedes prepared range {range:?}"))
    })?;
    let prepared_layer = prepared.layers().get(local).ok_or_else(|| {
        VindexError::Parse(format!("layer {layer} is outside prepared range {range:?}"))
    })?;
    let PreparedAttention::Softmax(operands) = &prepared_layer.attention else {
        return Err(VindexError::Parse(format!(
            "prepared layer {layer} is not softmax attention"
        )));
    };
    let width = op.num_q_heads * op.head_dim;
    if op.o.shape != [hidden, width] {
        return Err(VindexError::Parse(format!(
            "layer {layer} W_O shape {:?} is not [{hidden}, {width}]",
            op.o.shape
        )));
    }
    Ok(ReplaySite {
        layer_plan,
        prepared_layer,
        operands,
        num_q_heads: op.num_q_heads,
        head_dim: op.head_dim,
        hidden,
        width,
    })
}

/// `W_O` over one concatenated head vector, then the declared post norm,
/// residual scale and single-stream residual add.
fn mix<B: PlanBackend + ?Sized>(
    site: &ReplaySite<'_>,
    backend: &B,
    entering_carrier: &[f32],
    head_values: &[Vec<f32>],
) -> Result<AttentionMixtureReplay, VindexError> {
    let concat: Vec<f32> = head_values.iter().flatten().copied().collect();
    let raw_attention_output = backend.project(ProjectCall {
        weight: site.operands.w_o.slice(),
        out_dim: site.hidden,
        in_dim: site.width,
        x: &concat,
    })?;
    if raw_attention_output.len() != site.hidden
        || raw_attention_output.iter().any(|value| !value.is_finite())
    {
        return Err(VindexError::Parse(
            "attention-head replay produced an invalid raw output".into(),
        ));
    }
    let mut applied_delta = match &site.prepared_layer.post_attention {
        Some(norm) => norm.apply(backend, &raw_attention_output),
        None => raw_attention_output.clone(),
    };
    scale_residual_delta(site.layer_plan.residual_scale, &mut applied_delta);
    let mut carrier_after = entering_carrier.to_vec();
    backend.residual_add(&mut carrier_after, &applied_delta);
    if applied_delta
        .iter()
        .chain(&carrier_after)
        .any(|v| !v.is_finite())
    {
        return Err(VindexError::Parse(
            "attention-head replay produced a non-finite applied path".into(),
        ));
    }
    Ok(AttentionMixtureReplay {
        raw_attention_output,
        applied_delta,
        carrier_after,
    })
}

/// Execute one captured/replaced head mixture through the effective `W_O`,
/// declared post norm, residual scale and single-stream residual add.
pub fn replay_attention_mixture<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    prepared: &PreparedOperands,
    backend: &B,
    layer: usize,
    entering_carrier: &[f32],
    head_values: &[Vec<f32>],
) -> Result<AttentionMixtureReplay, VindexError> {
    let site = resolve_site(plan, prepared, layer, entering_carrier, head_values)?;
    mix(&site, backend, entering_carrier, head_values)
}

/// Replay one layer's captured query-head values through the image that
/// canonical execution prepared.
///
/// V1 intentionally refuses output gating and output bias: both are absent on
/// the frozen Gemma 3 surface, and admitting either without decomposing its
/// exact placement would make “per-head contribution” mean something else.
pub fn replay_attention_heads<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    prepared: &PreparedOperands,
    backend: &B,
    layer: usize,
    entering_carrier: &[f32],
    head_values: &[Vec<f32>],
) -> Result<AttentionHeadReplay, VindexError> {
    let site = resolve_site(plan, prepared, layer, entering_carrier, head_values)?;
    let mixture = mix(&site, backend, entering_carrier, head_values)?;
    let (hidden, width, head_dim) = (site.hidden, site.width, site.head_dim);
    let project = |x: &[f32]| {
        backend.project(ProjectCall {
            weight: site.operands.w_o.slice(),
            out_dim: hidden,
            in_dim: width,
            x,
        })
    };
    let mut contributions = Vec::with_capacity(site.num_q_heads);
    let mut contribution_norms = Vec::with_capacity(site.num_q_heads);
    let mut reconstructed = vec![0.0f64; hidden];
    for (head, values) in head_values.iter().enumerate() {
        let mut isolated = vec![0.0f32; width];
        isolated[head * head_dim..(head + 1) * head_dim].copy_from_slice(values);
        let contribution = project(&isolated)?;
        let norm = contribution
            .iter()
            .map(|&value| f64::from(value).powi(2))
            .sum::<f64>()
            .sqrt();
        if contribution.len() != hidden || !norm.is_finite() {
            return Err(VindexError::Parse(format!(
                "attention-head replay produced an invalid contribution for head {head}"
            )));
        }
        for (sum, &value) in reconstructed.iter_mut().zip(&contribution) {
            *sum += f64::from(value);
        }
        contributions.push(contribution);
        contribution_norms.push(norm);
    }
    let raw_reconstruction_relative_l2 = relative_l2(&mixture.raw_attention_output, &reconstructed);
    if !raw_reconstruction_relative_l2.is_finite() {
        return Err(VindexError::Parse(
            "attention-head replay reconstruction is non-finite".into(),
        ));
    }

    Ok(AttentionHeadReplay {
        contributions,
        contribution_norms,
        raw_attention_output: mixture.raw_attention_output,
        applied_delta: mixture.applied_delta,
        carrier_after: mixture.carrier_after,
        raw_reconstruction_relative_l2,
    })
}
