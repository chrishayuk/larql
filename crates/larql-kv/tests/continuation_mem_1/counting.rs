//! CONTINUATION-MEM-1 I5: a backend that IS its inner backend except that
//! its dense projector counts every projection by (in_dim, out_dim).
//!
//! Every method is spelled out rather than left to the trait's defaults,
//! for the reason the Arc delegation in `exec/backend.rs` gives: a default
//! `select` or `dense_projector` would quietly replace the inner backend's.

use std::collections::BTreeMap;
use std::sync::Mutex;

use larql_vindex::format::vindex3::opplan::exec::backend::{
    AttentionCall, AttentionOut, AttentionStepCall, AttentionStepOut, DispatchStats, FfnCall,
    FfnManyCall, HeadIntervene, NormCall, PlanBackend, ProjectCall, RoutedFfnCall, WeightSlice,
};
use larql_vindex::format::vindex3::opplan::exec::cpu::WeightRows;
use larql_vindex::format::vindex3::opplan::exec::gated_delta::DenseProjections;
use larql_vindex::format::vindex3::opplan::exec::lowering::LoweringIdentity;
use larql_vindex::format::vindex3::opplan::exec::observe::AttentionHeadRecord;
use larql_vindex::format::vindex3::opplan::exec::realization::{
    RepresentationFacts, Selection, SelectionRefusal,
};
use larql_vindex::format::vindex3::opplan::planned::PlannedOperand;
use larql_vindex::VindexError;

pub struct CountingProjections {
    inner: &'static dyn DenseProjections,
    shapes: Mutex<BTreeMap<(usize, usize), usize>>,
}

impl CountingProjections {
    pub fn count(&self, in_dim: usize, out_dim: usize) -> usize {
        self.shapes
            .lock()
            .unwrap()
            .get(&(in_dim, out_dim))
            .copied()
            .unwrap_or(0)
    }

    pub fn shapes(&self) -> BTreeMap<(usize, usize), usize> {
        self.shapes.lock().unwrap().clone()
    }

    fn bump(&self, in_dim: usize, out_dim: usize, n: usize) {
        *self
            .shapes
            .lock()
            .unwrap()
            .entry((in_dim, out_dim))
            .or_default() += n;
    }
}

impl DenseProjections for CountingProjections {
    fn project(&self, weight: WeightRows<'_>, x: &[f32], out_dim: usize) -> Vec<f32> {
        self.bump(x.len(), out_dim, 1);
        self.inner.project(weight, x, out_dim)
    }

    fn project_many(&self, weight: WeightRows<'_>, xs: &[&[f32]], out_dim: usize) -> Vec<Vec<f32>> {
        if let Some(first) = xs.first() {
            self.bump(first.len(), out_dim, xs.len());
        }
        self.inner.project_many(weight, xs, out_dim)
    }

    fn is_weight_stationary(&self, weight: WeightRows<'_>, in_dim: usize, n: usize) -> bool {
        self.inner.is_weight_stationary(weight, in_dim, n)
    }
}

pub struct Counting<B: 'static> {
    inner: &'static B,
    pub projections: CountingProjections,
}

impl<B: PlanBackend + 'static> Counting<B> {
    /// The inner backend is leaked so the counting projector can borrow
    /// its projector for `'static` — a test-lifetime cost, once.
    pub fn new(inner: B) -> Self {
        let inner: &'static B = Box::leak(Box::new(inner));
        Self {
            inner,
            projections: CountingProjections {
                inner: inner.dense_projector(),
                shapes: Mutex::new(BTreeMap::new()),
            },
        }
    }
}

impl<B: PlanBackend + 'static> PlanBackend for Counting<B> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn identity(&self) -> LoweringIdentity {
        self.inner.identity()
    }

    fn dispatch_stats(&self) -> Option<DispatchStats> {
        self.inner.dispatch_stats()
    }

    fn select(
        &self,
        operand: &PlannedOperand,
        facts: &RepresentationFacts,
    ) -> Result<Selection, Box<SelectionRefusal>> {
        self.inner.select(operand, facts)
    }

    fn dense_projector(&self) -> &dyn DenseProjections {
        &self.projections
    }

    fn prepare(&self, weights: &[WeightSlice<'_>]) {
        self.inner.prepare(weights)
    }

    fn embed(&self, table: &[f32], hidden: usize, token: u32, scale: Option<f32>) -> Vec<f32> {
        self.inner.embed(table, hidden, token, scale)
    }

    fn norm(&self, call: NormCall<'_>) -> Vec<f32> {
        self.inner.norm(call)
    }

    fn project(&self, call: ProjectCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.inner.project(call)
    }

    fn attention(&self, call: AttentionCall<'_>) -> Result<AttentionOut, VindexError> {
        self.inner.attention(call)
    }

    fn attention_step(&self, call: AttentionStepCall<'_>) -> Result<AttentionStepOut, VindexError> {
        self.inner.attention_step(call)
    }

    fn serves_attention_heads(&self) -> bool {
        self.inner.serves_attention_heads()
    }

    fn attention_step_observed(
        &self,
        call: AttentionStepCall<'_>,
        tap: &mut dyn FnMut(AttentionHeadRecord<'_>),
    ) -> Result<AttentionStepOut, VindexError> {
        self.inner.attention_step_observed(call, tap)
    }

    fn serves_head_intervention(&self) -> bool {
        self.inner.serves_head_intervention()
    }

    fn attention_step_intervened(
        &self,
        call: AttentionStepCall<'_>,
        tap: Option<&mut dyn FnMut(AttentionHeadRecord<'_>)>,
        head_intervene: &mut HeadIntervene<'_>,
    ) -> Result<AttentionStepOut, VindexError> {
        self.inner
            .attention_step_intervened(call, tap, head_intervene)
    }

    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.inner.ffn(call)
    }

    fn ffn_many(&self, call: FfnManyCall<'_>) -> Result<Vec<Vec<f32>>, VindexError> {
        self.inner.ffn_many(call)
    }

    fn scale_row(&self, row: &mut [f32], scale: f32) {
        self.inner.scale_row(row, scale)
    }

    fn routed_ffn(&self, call: RoutedFfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.inner.routed_ffn(call)
    }

    fn output_head(
        &self,
        projection: WeightSlice<'_>,
        vocab: usize,
        hidden: usize,
        x: &[f32],
        multiplier: Option<f64>,
        softcapping: Option<f32>,
    ) -> Result<Vec<f32>, VindexError> {
        self.inner
            .output_head(projection, vocab, hidden, x, multiplier, softcapping)
    }

    fn residual_add(&self, acc: &mut [f32], delta: &[f32]) {
        self.inner.residual_add(acc, delta)
    }
}
