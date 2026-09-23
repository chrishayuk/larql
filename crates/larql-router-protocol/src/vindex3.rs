//! V3 stateless layer-prefix RPC. Positions always start at zero; no remote
//! continuation cache, retries or hidden server affinity are part of this ABI.
use serde::{Deserialize, Serialize};
pub const SCHEMA: u32 = 1;
pub const PATH: &str = "/v1/vindex3/layers";
pub const MAX_POSITIONS: usize = 4096;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub schema: u32,
    pub artifact: String,
    pub backend: String,
    /// Numerical provider family and semantic revision.
    pub lowering: String,
    /// Half-open range of plan layer indices.
    pub start: usize,
    pub end: usize,
    pub layers: usize,
    pub hidden: usize,
}
impl Binding {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != SCHEMA
            || self.backend != "cpu"
            || self.lowering.is_empty()
            || self.hidden == 0
            || self.start >= self.end
            || self.end > self.layers
            || self.artifact.len() != 64
            || !self.artifact.bytes().all(|c| c.is_ascii_hexdigit())
        {
            return Err("invalid or unsupported V3 shard binding".into());
        }
        Ok(())
    }
    pub fn validate_rows(&self, rows: &[Vec<f32>]) -> Result<(), String> {
        self.validate()?;
        if rows.is_empty()
            || rows.len() > MAX_POSITIONS
            || rows
                .iter()
                .any(|r| r.len() != self.hidden || r.iter().any(|x| !x.is_finite()))
        {
            return Err(format!(
                "V3 shard input requires 1..={MAX_POSITIONS} rows of {} finite values",
                self.hidden
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub binding: Binding,
    pub rows: Vec<Vec<f32>>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Response {
    pub binding: Binding,
    pub rows: Vec<Vec<f32>>,
}
