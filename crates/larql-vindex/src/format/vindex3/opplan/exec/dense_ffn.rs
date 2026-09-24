//! Placement at the dense FFN operation boundary. Norms and residual updates
//! belong to the local interpreter; a provider consumes a normalized row and
//! returns only gate/activation/up/down computation.
use super::{
    accounting::Observed,
    backend::{MatrixClass, PlanBackend, WeightFormat},
    experts::FfnOperands,
    operands::OperandSource,
    prepared::{pinned_format, ExecutionSlice},
    realization::RealizationRecord,
};
use crate::{
    error::VindexError,
    format::vindex3::opplan::{planned::Operation, ComponentOpPlan, LayerFfn},
};

/// A bound operation provider. Transport and artifact validation live above
/// the format executor; there is no dependency on the legacy FFN engine.
pub trait DenseFfnProvider: Send + Sync {
    fn apply(&self, layer: usize, normalized: &[f32]) -> Result<Vec<f32>, VindexError>;
}

pub(super) fn validate_plan(plan: &ComponentOpPlan) -> Result<(), VindexError> {
    if plan.residual_topology != larql_models::config::ResidualTopology::SingleStream
        || plan.layers.is_empty()
        || plan
            .layers
            .iter()
            .any(|l| l.attention.softmax().is_none() || !matches!(l.ffn, Some(LayerFfn::Dense(_))))
    {
        return Err(VindexError::Parse(
            "dense FFN placement requires single-stream softmax layers with dense FFNs".into(),
        ));
    }
    Ok(())
}

pub fn validate_row(row: &[f32], hidden: usize) -> Result<(), VindexError> {
    if row.len() != hidden || row.iter().any(|v| !v.is_finite()) {
        return Err(VindexError::Parse(format!(
            "dense FFN requires {hidden} finite values"
        )));
    }
    Ok(())
}

/// A worker image contains only dense FFN matrices, never attention or norms.
pub struct PreparedDenseFfns {
    first: usize,
    hidden: usize,
    lowering: super::lowering::LoweringIdentity,
    ffns: Vec<FfnOperands>,
}
impl PreparedDenseFfns {
    pub(super) fn load<B: PlanBackend + ?Sized>(
        plan: &ComponentOpPlan,
        store: OperandSource<'_>,
        backend: &B,
        slice: &ExecutionSlice,
        pins: &[RealizationRecord],
        hidden: usize,
    ) -> Result<Self, VindexError> {
        let range = slice.layers(plan);
        let format = |op: &_| {
            pinned_format(
                pins,
                store,
                op,
                Operation::Project(MatrixClass::FfnProjection),
            )
        };
        let ffns = range
            .clone()
            .map(|i| {
                FfnOperands::load(
                    plan.layers[i].ffn.as_ref().expect("validated dense"),
                    store,
                    &format,
                    WeightFormat::F32.into(),
                    &format,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let weights = ffns
            .iter()
            .flat_map(FfnOperands::weight_slices)
            .collect::<Vec<_>>();
        backend.prepare(&weights);
        Ok(Self {
            first: range.start,
            hidden,
            lowering: backend.identity(),
            ffns,
        })
    }
    pub fn apply<B: PlanBackend + ?Sized>(
        &self,
        plan: &ComponentOpPlan,
        backend: &B,
        layer: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, VindexError> {
        if backend.identity() != self.lowering {
            return Err(VindexError::Parse(
                "dense FFN worker numerical provider changed".into(),
            ));
        }
        validate_row(x, self.hidden)?;
        let ffn = layer
            .checked_sub(self.first)
            .and_then(|i| self.ffns.get(i))
            .ok_or_else(|| {
                VindexError::Parse(format!("layer {layer} is outside this dense FFN worker"))
            })?;
        let op = plan
            .layers
            .get(layer)
            .and_then(|l| l.ffn.as_ref())
            .ok_or_else(|| VindexError::Parse("worker layer missing from plan".into()))?;
        let out = ffn.apply(op, backend, x, self.hidden)?;
        validate_row(&out, self.hidden)?;
        Ok(out)
    }
    pub(super) fn bound(&self, plan: &ComponentOpPlan) -> Result<Vec<Observed>, VindexError> {
        let mut out = Vec::new();
        for (i, ffn) in self.ffns.iter().enumerate() {
            let layer = self.first + i;
            for (operation, bound) in
                ffn.bound(plan.layers[layer].ffn.as_ref().expect("validated dense"))?
            {
                out.push(bound.observed(operation, Some(layer))?);
            }
        }
        Ok(out)
    }
    pub(super) fn matrices(&self) -> Vec<&super::weights::LoadedWeight> {
        self.ffns
            .iter()
            .flat_map(FfnOperands::loaded_matrices)
            .collect()
    }
}
