//! Coordinator-side transport seams for distributed VINDEX3 execution.
//!
//! These are signatures over this crate's wire types: a coordinator
//! (`larql-inference`) drives them, a transport (`larql-router`'s HTTP
//! clients, or an in-process test double) implements them. Living here
//! means a transport can be written against the protocol alone, without
//! depending on the inference engine that consumes it.
//!
//! `larql_inference::vindex3::{distributed, dense_ffn, routed_experts}`
//! re-export each trait at its original path.

use crate::{vindex3, vindex3_expert_rows, vindex3_experts, vindex3_ffn};

/// Layer-prefix shards. A transport must return the server's binding, not
/// manufacture one locally.
pub trait ShardTransport {
    fn bindings(&self) -> Vec<vindex3::Binding>;
    fn forward(&self, shard: usize, rows: Vec<Vec<f32>>) -> Result<vindex3::Response, String>;
}

/// Dense FFN placement: one normalised row in, one FFN output row back.
pub trait FfnTransport: Send + Sync {
    fn bindings(&self) -> Vec<vindex3_ffn::Binding>;
    fn forward(
        &self,
        shard: usize,
        layer: usize,
        normalized: &[f32],
    ) -> Result<vindex3_ffn::Response, String>;
    /// Return a row authenticated against the binding admitted at preparation.
    /// Compact transports verify their immutable handle and response correlation.
    fn forward_bound(
        &self,
        shard: usize,
        layer: usize,
        normalized: &[f32],
        expected: &vindex3_ffn::Binding,
    ) -> Result<Vec<f32>, String> {
        let response = self.forward(shard, layer, normalized)?;
        if response.binding != *expected || response.layer != layer {
            return Err("dense FFN response changed binding or layer".into());
        }
        Ok(response.row)
    }
}

/// One selected expert's transform output, as a transport delivers it.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpertOutput {
    pub expert: usize,
    pub row: Vec<f32>,
}

/// Selected-expert placement. Implementations receive expert IDs, never
/// routing weights, and must answer every requested expert exactly once.
pub trait ExpertTransport: Send + Sync {
    fn bindings(&self) -> Vec<vindex3_experts::Binding>;
    fn forward(
        &self,
        shard: usize,
        layer: usize,
        experts: &[usize],
        row: &[f32],
    ) -> Result<Vec<ExpertOutput>, String>;

    /// Several rows for one shard in ONE exchange, answered per
    /// `(row, expert)`. Deliberately not defaulted to repeated
    /// [`Self::forward`] calls: a transport either carries the multi-row
    /// frame or says it cannot, so a caller is never told a batch crossed
    /// once when it crossed once per row. An error is the whole answer;
    /// there is no partial result.
    fn forward_rows(
        &self,
        shard: usize,
        layer: usize,
        rows: &[vindex3_expert_rows::Row],
    ) -> Result<Vec<vindex3_expert_rows::Output>, String>;
}
