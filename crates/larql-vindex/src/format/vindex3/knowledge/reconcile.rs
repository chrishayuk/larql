//! GW-0B's physical join: reconstruct a dense FFN write as contributions
//! addressed in the same `(layer, feature)` vocabulary as exact WALK.
//!
//! This does not claim that a feature is a semantic edge. It supplies only a
//! checked physical join key. Attribution is refused unless the complete sum
//! reproduces the already-recorded carrier delta within caller-declared
//! tolerances.

use larql_models::config::{Activation, NormType};
use ndarray::{Array1, Array2};

use crate::error::VindexError;

use super::super::opplan::exec::kernels::activate;
use super::super::opplan::exec::operands::OperandStore;
use super::super::opplan::exec::prepared::PreparedOperands;
use super::super::opplan::{ComponentOpPlan, LayerFfn, NormOp};

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureContribution {
    pub layer: usize,
    pub feature: usize,
    /// `activation(gate) * up`, i.e. the coefficient of this down column.
    pub activation: f32,
    /// L2 norm of this feature's contribution after any post-FFN norm and
    /// residual-delta scale carried by the plan.
    pub contribution_l2: f64,
    /// Signed projection onto the observed write, divided by its squared
    /// norm. Fractions can be negative or exceed one because features cancel.
    pub observed_projection_fraction: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReconstructionProof {
    pub relative_l2_error: f64,
    pub relative_linf_error: f64,
    pub max_abs_error: f64,
    pub relative_l2_tolerance: f64,
    pub relative_linf_tolerance: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseFfnAttribution {
    pub layer: usize,
    pub features: usize,
    pub proof: ReconstructionProof,
    pub contributions: Vec<FeatureContribution>,
}

struct LoadedNorm {
    kind: NormType,
    eps: f64,
    weight_offset: f32,
    weight: Vec<f32>,
}

impl LoadedNorm {
    fn load(op: &NormOp, store: &OperandStore) -> Result<Self, VindexError> {
        Ok(Self {
            kind: op.kind,
            eps: op.eps,
            weight_offset: op.weight_offset,
            weight: store.load(&op.weight)?,
        })
    }

    fn apply(&self, x: &[f32]) -> Vec<f32> {
        match self.kind {
            NormType::RmsNorm => {
                let mean_square = x
                    .iter()
                    .map(|&value| f64::from(value) * f64::from(value))
                    .sum::<f64>()
                    / x.len() as f64;
                let inverse = (mean_square + self.eps).sqrt().recip() as f32;
                x.iter()
                    .zip(&self.weight)
                    .map(|(&value, &weight)| value * inverse * (self.weight_offset + weight))
                    .collect()
            }
            NormType::LayerNorm => {
                let mean = x.iter().map(|&value| f64::from(value)).sum::<f64>() / x.len() as f64;
                let variance = x
                    .iter()
                    .map(|&value| (f64::from(value) - mean).powi(2))
                    .sum::<f64>()
                    / x.len() as f64;
                let inverse = (variance + self.eps).sqrt().recip();
                x.iter()
                    .zip(&self.weight)
                    .map(|(&value, &weight)| {
                        ((f64::from(value) - mean) * inverse) as f32 * (self.weight_offset + weight)
                    })
                    .collect()
            }
        }
    }
}

/// One dense layer's effective execution operands and normalisation semantics.
/// Construct one layer at a time to keep the offline reconciliation's resident
/// set bounded.
pub struct DenseFfnLayerView {
    layer: usize,
    gate: Option<Array2<f32>>,
    up: Array2<f32>,
    down: Array2<f32>,
    activation: Activation,
    pre_norm: Option<LoadedNorm>,
    post_norm: Option<LoadedNorm>,
    residual_scale: f32,
}

impl DenseFfnLayerView {
    /// Bind the matrices from the prepared execution image, preserving a
    /// lossy residency choice such as Q8 rather than silently returning to
    /// the container's source BF16 values.
    pub fn from_prepared(
        plan: &ComponentOpPlan,
        store: &OperandStore,
        prepared: &PreparedOperands,
        layer: usize,
    ) -> Result<Self, VindexError> {
        if !plan.residual_topology.is_single_stream() {
            return Err(VindexError::Parse(
                "dense FFN reconciliation currently requires Single residual topology".to_string(),
            ));
        }
        let layer_plan = plan.layers.get(layer).ok_or_else(|| {
            VindexError::Parse(format!("layer {layer} is outside the component plan"))
        })?;
        let Some(LayerFfn::Dense(op)) = &layer_plan.ffn else {
            return Err(VindexError::Parse(format!(
                "layer {layer} is not an eligible dense FFN"
            )));
        };
        if op.gate_policy != larql_models::ExpertGatePolicy::Gated {
            return Err(VindexError::Parse(format!(
                "layer {layer} gate policy {:?} has no frozen GW-0B decomposition",
                op.gate_policy
            )));
        }
        let image = prepared.dense_ffn_image(plan, layer)?;
        let matrix = |values, rows, columns| {
            Array2::from_shape_vec((rows, columns), values).map_err(|error| {
                VindexError::Parse(format!("layer {layer} prepared FFN geometry: {error}"))
            })
        };
        Ok(Self {
            layer,
            gate: image
                .gate
                .map(|values| matrix(values, op.intermediate_size, plan_hidden(plan)?))
                .transpose()?,
            up: matrix(image.up, op.intermediate_size, plan_hidden(plan)?)?,
            down: matrix(image.down, plan_hidden(plan)?, op.intermediate_size)?,
            activation: op.activation,
            pre_norm: layer_plan
                .pre_ffn_norm
                .as_ref()
                .map(|norm| LoadedNorm::load(norm, store))
                .transpose()?,
            post_norm: layer_plan
                .post_ffn_norm
                .as_ref()
                .map(|norm| LoadedNorm::load(norm, store))
                .transpose()?,
            residual_scale: layer_plan.residual_scale.unwrap_or(1.0),
        })
    }

    pub fn features(&self) -> usize {
        self.down.ncols()
    }

    /// Reconstruct and rank the feature contributions to one recorded FFN
    /// write. Only the top `top_k` contributions are retained, but the proof is
    /// computed from the complete activation and complete down projection.
    pub fn attribute(
        &self,
        carrier_before: &[f32],
        observed_delta: &[f32],
        top_k: usize,
        relative_l2_tolerance: f64,
        relative_linf_tolerance: f64,
    ) -> Result<DenseFfnAttribution, VindexError> {
        if carrier_before.len() != self.up.ncols() || observed_delta.len() != self.down.nrows() {
            return Err(VindexError::Parse(format!(
                "layer {} attribution geometry mismatch: before {}, delta {}, expected hidden {}",
                self.layer,
                carrier_before.len(),
                observed_delta.len(),
                self.up.ncols()
            )));
        }
        let input = self.pre_norm.as_ref().map_or_else(
            || carrier_before.to_vec(),
            |norm| norm.apply(carrier_before),
        );
        let up = self.up.dot(&Array1::from_vec(input.clone()));
        let inner = match &self.gate {
            Some(gate) => {
                let gate = gate.dot(&Array1::from_vec(input));
                gate.iter()
                    .zip(&up)
                    .map(|(&gate, &up)| activate(self.activation, gate) * up)
                    .collect::<Vec<_>>()
            }
            None => up
                .iter()
                .map(|&up| activate(self.activation, up))
                .collect::<Vec<_>>(),
        };
        let raw = self.down.dot(&Array1::from_vec(inner.clone())).to_vec();
        let mut reconstructed = self
            .post_norm
            .as_ref()
            .map_or_else(|| raw.clone(), |norm| norm.apply(&raw));
        reconstructed
            .iter_mut()
            .for_each(|value| *value *= self.residual_scale);

        let errors = reconstructed
            .iter()
            .zip(observed_delta)
            .map(|(&actual, &observed)| f64::from(actual) - f64::from(observed))
            .collect::<Vec<_>>();
        let error_l2 = errors.iter().map(|error| error * error).sum::<f64>().sqrt();
        let observed_l2 = observed_delta
            .iter()
            .map(|&value| f64::from(value).powi(2))
            .sum::<f64>()
            .sqrt();
        let relative_l2_error = error_l2 / observed_l2.max(f64::MIN_POSITIVE);
        let max_abs_error = errors.iter().copied().map(f64::abs).fold(0.0, f64::max);
        let observed_max_abs = observed_delta
            .iter()
            .copied()
            .map(f64::from)
            .map(f64::abs)
            .fold(0.0, f64::max);
        let relative_linf_error = max_abs_error / observed_max_abs.max(f64::MIN_POSITIVE);
        let reconstructed_l2 = reconstructed
            .iter()
            .map(|&value| f64::from(value).powi(2))
            .sum::<f64>()
            .sqrt();
        let cosine = reconstructed
            .iter()
            .zip(observed_delta)
            .map(|(&left, &right)| f64::from(left) * f64::from(right))
            .sum::<f64>()
            / (reconstructed_l2 * observed_l2).max(f64::MIN_POSITIVE);
        let norm_ratio = reconstructed_l2 / observed_l2.max(f64::MIN_POSITIVE);
        let proof = ReconstructionProof {
            relative_l2_error,
            relative_linf_error,
            max_abs_error,
            relative_l2_tolerance,
            relative_linf_tolerance,
        };
        if !relative_l2_error.is_finite()
            || !max_abs_error.is_finite()
            || relative_l2_error > relative_l2_tolerance
            || relative_linf_error > relative_linf_tolerance
        {
            return Err(VindexError::Parse(format!(
                "layer {} FFN attribution refused: reconstructed write differs from observation \
                 (relative L2 {relative_l2_error:.3e} > {relative_l2_tolerance:.3e} or relative \
                 Linf {relative_linf_error:.3e} > {relative_linf_tolerance:.3e}; raw max abs \
                 {max_abs_error:.3e}, cosine {cosine:.9}, norm ratio {norm_ratio:.9})",
                self.layer
            )));
        }

        let observed_norm_squared = observed_delta
            .iter()
            .map(|&value| f64::from(value).powi(2))
            .sum::<f64>()
            .max(f64::MIN_POSITIVE);
        let post_stat = post_statistic(self.post_norm.as_ref(), &raw);
        let mut contributions = (0..self.features())
            .map(|feature| {
                let activation = inner[feature];
                let column = self.down.column(feature);
                let column_mean = if matches!(
                    self.post_norm.as_ref().map(|norm| norm.kind),
                    Some(NormType::LayerNorm)
                ) {
                    column.iter().map(|&value| f64::from(value)).sum::<f64>() / column.len() as f64
                } else {
                    0.0
                };
                let mut norm_squared = 0.0;
                let mut dot_observed = 0.0;
                for (index, &weight) in column.iter().enumerate() {
                    let transformed = transformed_weight(
                        weight,
                        column_mean,
                        index,
                        self.post_norm.as_ref(),
                        post_stat,
                    ) * self.residual_scale;
                    let value = f64::from(activation * transformed);
                    norm_squared += value * value;
                    dot_observed += value * f64::from(observed_delta[index]);
                }
                FeatureContribution {
                    layer: self.layer,
                    feature,
                    activation,
                    contribution_l2: norm_squared.sqrt(),
                    observed_projection_fraction: dot_observed / observed_norm_squared,
                }
            })
            .collect::<Vec<_>>();
        contributions.sort_by(|left, right| {
            right
                .contribution_l2
                .total_cmp(&left.contribution_l2)
                .then_with(|| left.feature.cmp(&right.feature))
        });
        contributions.truncate(top_k.min(contributions.len()));
        Ok(DenseFfnAttribution {
            layer: self.layer,
            features: self.features(),
            proof,
            contributions,
        })
    }
}

fn plan_hidden(plan: &ComponentOpPlan) -> Result<usize, VindexError> {
    plan.embedding
        .as_ref()
        .and_then(|embedding| embedding.table.shape.get(1))
        .copied()
        .ok_or_else(|| {
            VindexError::Parse("plan embedding does not declare hidden size".to_string())
        })
}

fn post_statistic(norm: Option<&LoadedNorm>, raw: &[f32]) -> f64 {
    match norm.map(|norm| norm.kind) {
        None => 1.0,
        Some(NormType::RmsNorm) => {
            let mean_square = raw
                .iter()
                .map(|&value| f64::from(value).powi(2))
                .sum::<f64>()
                / raw.len() as f64;
            (mean_square + norm.unwrap().eps).sqrt()
        }
        Some(NormType::LayerNorm) => {
            let mean = raw.iter().map(|&value| f64::from(value)).sum::<f64>() / raw.len() as f64;
            let variance = raw
                .iter()
                .map(|&value| (f64::from(value) - mean).powi(2))
                .sum::<f64>()
                / raw.len() as f64;
            (variance + norm.unwrap().eps).sqrt()
        }
    }
}

fn transformed_weight(
    weight: f32,
    column_mean: f64,
    index: usize,
    norm: Option<&LoadedNorm>,
    statistic: f64,
) -> f32 {
    let Some(norm) = norm else {
        return weight;
    };
    let centred = match norm.kind {
        NormType::RmsNorm => f64::from(weight),
        NormType::LayerNorm => f64::from(weight) - column_mean,
    };
    (centred / statistic) as f32 * (norm.weight_offset + norm.weight[index])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::vindex3::opplan::exec::backend::WeightSlice;
    use ndarray::array;

    fn view() -> DenseFfnLayerView {
        DenseFfnLayerView {
            layer: 3,
            gate: Some(array![[1.0, 0.0], [0.0, 1.0]]),
            up: array![[2.0, 0.0], [0.0, 3.0]],
            down: array![[1.0, 2.0], [3.0, 4.0]],
            activation: Activation::Silu,
            pre_norm: None,
            post_norm: None,
            residual_scale: 1.0,
        }
    }

    #[test]
    fn complete_feature_sum_is_a_checked_physical_join() {
        let view = view();
        let before = [0.5, -0.25];
        let a0 = activate(Activation::Silu, before[0]) * (2.0 * before[0]);
        let a1 = activate(Activation::Silu, before[1]) * (3.0 * before[1]);
        let observed = [a0 + 2.0 * a1, 3.0 * a0 + 4.0 * a1];
        let result = view.attribute(&before, &observed, 2, 1e-6, 1e-6).unwrap();
        assert_eq!(result.layer, 3);
        assert_eq!(result.features, 2);
        assert_eq!(result.contributions.len(), 2);
        assert!(result.proof.relative_l2_error <= 1e-6);
    }

    #[test]
    fn a_write_that_is_not_the_feature_sum_is_refused() {
        let error = view()
            .attribute(&[0.5, -0.25], &[8.0, 9.0], 2, 1e-6, 1e-6)
            .unwrap_err();
        assert!(error.to_string().contains("attribution refused"));
    }

    #[test]
    fn effective_q8_values_are_the_values_the_physical_join_uses() {
        let codes = [1, -2, 3, -4];
        let scales = [0.5, 2.0];
        let values = WeightSlice::Q8 {
            codes: &codes,
            scales: &scales,
            sums: &[],
            block: 2,
        }
        .decode_f32(1, 4)
        .unwrap();
        assert_eq!(values, vec![0.5, -1.0, 6.0, -8.0]);
    }
}
