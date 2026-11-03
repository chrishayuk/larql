//! A lowering that changes the model's output (MEM-002 I2b).
//!
//! The reference backend with its logits reversed: the same weights read from
//! the same files, a different result. It exists to show evidence of file
//! identity can't by itself say what ran.

use larql_vindex::error::VindexError;
use larql_vindex::format::vindex3::opplan::exec::backend::*;
use larql_vindex::format::vindex3::opplan::exec::lowering::LoweringIdentity;
use larql_vindex::format::vindex3::opplan::exec::realization::{
    RepresentationFacts, Selection, SelectionRefusal,
};
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend as Reference;
use larql_vindex::format::vindex3::opplan::planned::PlannedOperand;

const FAMILY: &str = "hostile-lowering";
const REVISION: u32 = 1;

pub struct Tamper;

impl PlanBackend for Tamper {
    fn name(&self) -> &str {
        "tamper"
    }
    fn identity(&self) -> LoweringIdentity {
        LoweringIdentity::new(FAMILY, REVISION)
    }
    fn select(
        &self,
        operand: &PlannedOperand,
        facts: &RepresentationFacts,
    ) -> Result<Selection, Box<SelectionRefusal>> {
        Reference.select(operand, facts)
    }
    fn embed(&self, table: &[f32], hidden: usize, token: u32, scale: Option<f32>) -> Vec<f32> {
        Reference.embed(table, hidden, token, scale)
    }
    fn norm(&self, call: NormCall<'_>) -> Vec<f32> {
        Reference.norm(call)
    }
    fn project(&self, call: ProjectCall<'_>) -> Result<Vec<f32>, VindexError> {
        Reference.project(call)
    }
    fn attention(&self, call: AttentionCall<'_>) -> Result<AttentionOut, VindexError> {
        Reference.attention(call)
    }
    fn attention_step(&self, call: AttentionStepCall<'_>) -> Result<AttentionStepOut, VindexError> {
        Reference.attention_step(call)
    }
    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        Reference.ffn(call)
    }
    fn routed_ffn(&self, call: RoutedFfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        Reference.routed_ffn(call)
    }
    fn output_head(
        &self,
        weights: WeightSlice<'_>,
        vocab: usize,
        hidden: usize,
        x: &[f32],
        multiplier: Option<f64>,
        scale: Option<f32>,
    ) -> Result<Vec<f32>, VindexError> {
        let mut logits = Reference.output_head(weights, vocab, hidden, x, multiplier, scale)?;
        logits.reverse();
        Ok(logits)
    }
    fn residual_add(&self, accumulator: &mut [f32], delta: &[f32]) {
        Reference.residual_add(accumulator, delta)
    }
}
