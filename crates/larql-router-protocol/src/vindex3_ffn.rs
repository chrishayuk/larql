//! Stateless dense FFN RPC: one normalized row in, one contribution out.
use crate::vindex3::Binding as LayerBinding;
use serde::{Deserialize, Serialize};
pub const PATH: &str = "/v1/vindex3/ffn";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperandIdentity {
    pub layer: usize,
    pub operand: String,
    pub representation: String,
    pub codec: String,
    pub realization: String,
    pub extent: String,
    pub dependencies: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub program: LayerBinding,
    /// Effective preparation decisions, including provider configuration effects.
    pub operands: Vec<OperandIdentity>,
}
impl Binding {
    pub fn validate(&self) -> Result<(), String> {
        self.program.validate()?;
        if self.operands.is_empty()
            || self
                .operands
                .iter()
                .any(|o| !(self.program.start..self.program.end).contains(&o.layer))
        {
            return Err("invalid dense FFN operand binding".into());
        }
        Ok(())
    }
    pub fn validate_row(&self, layer: usize, row: &[f32]) -> Result<(), String> {
        self.validate()?;
        if !(self.program.start..self.program.end).contains(&layer) {
            return Err(format!("layer {layer} is outside this dense FFN worker"));
        }
        if row.len() != self.program.hidden || row.iter().any(|x| !x.is_finite()) {
            return Err(format!(
                "dense FFN requires {} finite values",
                self.program.hidden
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub binding: Binding,
    pub layer: usize,
    pub row: Vec<f32>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Response {
    pub binding: Binding,
    pub layer: usize,
    pub row: Vec<f32>,
}

/// Opt-in diagnostics header; excluded from the numerical binding and JSON body.
pub const PROFILE_HEADER: &str = "x-larql-ffn-profile";
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WorkerTiming {
    pub decode_ns: u64,
    pub queue_ns: u64,
    pub execute_ns: u64,
    pub ffn_ns: u64,
    pub encode_ns: u64,
    pub handler_ns: u64,
}
