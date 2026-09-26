//! Reproduction provenance, bound to completed payloads by candidate authority.
//! Execution never loads calibration or interprets these recipe parameters.
use super::{
    calibration::CalibrationManifest, compiler::CandidateIndex, nvfp4_pack::EncoderRecipe,
};
use crate::error::VindexError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GptqParameters {
    pub hessian: String,
    pub damping: String,
    pub column_order: String,
    pub scales: String,
    pub factorization: String,
    pub schedule: String,
}
impl GptqParameters {
    pub fn frozen_v1() -> Self {
        Self {
            hessian: "raw-f64-XtX".into(),
            damping: "0.01*mean(diag(raw-H));dead-excluded".into(),
            column_order: "original-K;act-order=false".into(),
            scales: "nearest-v1-frozen".into(),
            factorization: "scalar-f64-cholesky-inverse-cholesky/v1".into(),
            schedule: "q,k,v,o,gate,up,down;ascending-layer/v1".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GptqDiagnostics {
    pub dead_columns: usize,
    pub alive_columns: usize,
    pub saturated_elements: usize,
    pub total_elements: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorDerivation {
    pub object: String,
    pub tensor: String,
    pub recipe: EncoderRecipe,
    pub source_values_sha256: String,
    pub payload_sha256: String,
    pub payload_bytes: u64,
    pub calibration: Option<CalibrationManifest>,
    pub parameters: Option<GptqParameters>,
    pub diagnostics: Option<GptqDiagnostics>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DerivationRecord {
    pub schema: String,
    pub source_semantic_sha256: String,
    pub allocation_sha256: String,
    /// Execution order, not tensor-name order. Only completed encodings enter.
    pub tensors: Vec<TensorDerivation>,
}

impl DerivationRecord {
    pub(crate) fn validate(&self, index: &CandidateIndex) -> Result<(), VindexError> {
        let bad = || VindexError::Parse("invalid completed calibration derivation".into());
        if self.schema != "represent-derivation/v1"
            || self.source_semantic_sha256 != index.source.identity.semantic_digest()
            || self.allocation_sha256
                != super::compile::hash_bytes(&serde_json::to_vec(&index.map).map_err(|_| bad())?)
            || self.tensors.len() != index.ledger.sealed.len()
        {
            return Err(bad());
        }
        let mut seen = std::collections::BTreeSet::new();
        for t in &self.tensors {
            let seal = index.ledger.get(&t.object, &t.tensor).ok_or_else(bad)?;
            if !seen.insert((&t.object, &t.tensor))
                || seal.encoding != "NVFP4"
                || seal.target_hash != t.payload_sha256
                || seal.target_len != t.payload_bytes
                || t.source_values_sha256.len() != 64
                || !t
                    .source_values_sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit())
            {
                return Err(bad());
            }
            if t.recipe == EncoderRecipe::gptq_v1() {
                let c = t.calibration.as_ref().ok_or_else(bad)?;
                let d = t.diagnostics.as_ref().ok_or_else(bad)?;
                c.key.validate()?;
                if t.parameters.as_ref() != Some(&GptqParameters::frozen_v1())
                    || c.schema != super::calibration::SCHEMA
                    || c.numeric != "f64-le-raw-sum"
                    || c.binding_sha256 != c.binding()?
                    || c.key.statistic != super::calibration::StatisticKind::DenseGram
                    || c.key.population != super::calibration::Population::Calibration
                    || c.key.site.object != t.object
                    || c.key.site.tensor != t.tensor
                    || d.alive_columns.checked_add(d.dead_columns) != Some(c.key.site.width)
                    || Some(d.total_elements) != c.key.site.rows.checked_mul(c.key.site.width)
                    || d.saturated_elements > d.total_elements
                {
                    return Err(bad());
                }
            } else if t.recipe != EncoderRecipe::nearest_v1()
                || t.calibration.is_some()
                || t.parameters.is_some()
                || t.diagnostics.is_some()
            {
                return Err(bad());
            }
        }
        Ok(())
    }
}
