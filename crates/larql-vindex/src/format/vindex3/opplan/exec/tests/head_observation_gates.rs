//! HEAD-OBS-1's two trait seams, exercised where the executor tests do not
//! reach them: the per-head observation methods forwarded through a shared
//! (`Arc`) backend, and the trait's own default for a backend that does not
//! serve heads. Tests only; nothing here changes execution.

use std::sync::Arc;

use super::decode::fixture;
use super::golden::{G_LAYERS, G_Q_HEADS, G_TOKENS};
use crate::error::VindexError;
use crate::format::vindex3::opplan::exec::backend::{
    AttentionCall, AttentionOut, AttentionStepCall, AttentionStepOut, FfnCall, NormCall,
    PlanBackend, ProjectCall, RoutedFfnCall, WeightSlice,
};
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::intervene::InterventionPlan;
use crate::format::vindex3::opplan::exec::intervene_heads::{
    HeadAddress, HeadIntervention, HeadInterventionPlan,
};
use crate::format::vindex3::opplan::exec::kv::RowKvState;
use crate::format::vindex3::opplan::exec::lowering::LoweringIdentity;
use crate::format::vindex3::opplan::exec::observe::NoopObserver;
use crate::format::vindex3::opplan::exec::observe_heads::HeadStats;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;

/// The head-sum residual the golden run is held to; the engineering freeze
/// bounds it at 1e-4 and measures ~1e-6.
const RESIDUAL_BOUND: f64 = 1e-4;

#[test]
fn head_observation_forwards_through_a_shared_backend() {
    let (_c, plan, store) = fixture();
    let backend = Arc::new(ReferenceBackend::new());
    assert!(
        backend.serves_attention_heads(),
        "the shared wrapper reports what the reference backend serves"
    );
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv).unwrap();
    let mut stats = HeadStats::new(&ops, &plan, &backend, None, 2);
    for &token in G_TOKENS.iter() {
        session.step_observed(token, &mut stats).unwrap();
    }
    assert!(stats.failure.is_none(), "{:?}", stats.failure);
    assert_eq!(
        stats.records,
        G_TOKENS.len() * G_LAYERS * G_Q_HEADS,
        "one record per head per attention write, through the wrapper"
    );
    assert_eq!(stats.writes.len(), G_TOKENS.len() * G_LAYERS);
    for write in &stats.writes {
        assert_eq!(write.rows.len(), G_Q_HEADS);
        assert!(
            write.residual <= RESIDUAL_BOUND,
            "layer {} position {}: head-sum residual {}",
            write.layer,
            write.position,
            write.residual
        );
    }
}

/// V3-INTERVENE-2's sibling seam, forwarded through the same shared
/// wrapper: a head intervention declared through `Arc<ReferenceBackend>`
/// reaches the kernel and fires exactly as it does unwrapped.
#[test]
fn head_intervention_forwards_through_a_shared_backend() {
    let (_c, plan, store) = fixture();
    let backend = Arc::new(ReferenceBackend::new());
    assert!(
        backend.serves_head_intervention(),
        "the shared wrapper reports what the reference backend serves"
    );
    let heads = HeadInterventionPlan::none()
        .with(HeadIntervention::zero(HeadAddress::new(0, 0, [3]).unwrap()))
        .unwrap();
    let mut session = DecodeSession::new(&plan, &store, &backend).unwrap();
    let mut fired = 0usize;
    for &token in G_TOKENS.iter() {
        let out = session
            .step_intervened(token, &mut NoopObserver, &InterventionPlan::none(), &heads)
            .unwrap();
        fired += out.head_firings.len();
    }
    assert_eq!(
        fired, 1,
        "the declared head firing reaches the kernel through the wrapper"
    );
}

/// Reference arithmetic behind a backend that declares nothing about heads:
/// every required method delegates, the two head methods are the trait's
/// defaults. Its `attention_step` invokes the default per-head path so the
/// default's refusal is observed, by name, on the executor's own call.
struct NoHeadsBackend(ReferenceBackend);

impl PlanBackend for NoHeadsBackend {
    fn name(&self) -> &str {
        "no-heads"
    }

    fn identity(&self) -> LoweringIdentity {
        LoweringIdentity::new("test-no-heads", 1)
    }

    fn embed(&self, table: &[f32], hidden: usize, token: u32, scale: Option<f32>) -> Vec<f32> {
        self.0.embed(table, hidden, token, scale)
    }

    fn norm(&self, call: NormCall<'_>) -> Vec<f32> {
        self.0.norm(call)
    }

    fn project(&self, call: ProjectCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.0.project(call)
    }

    fn attention(&self, call: AttentionCall<'_>) -> Result<AttentionOut, VindexError> {
        self.0.attention(call)
    }

    fn attention_step(&self, call: AttentionStepCall<'_>) -> Result<AttentionStepOut, VindexError> {
        // The default per-head path must refuse rather than compute; hand
        // its refusal back so the step carries the default's own words.
        match PlanBackend::attention_step_observed(self, call, &mut |_| {}) {
            Ok(_) => Err(VindexError::Parse(
                "the trait default served heads it does not have".to_string(),
            )),
            Err(e) => Err(e),
        }
    }

    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.0.ffn(call)
    }

    fn routed_ffn(&self, call: RoutedFfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.0.routed_ffn(call)
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
        self.0
            .output_head(projection, vocab, hidden, x, multiplier, softcapping)
    }

    fn residual_add(&self, acc: &mut [f32], delta: &[f32]) {
        self.0.residual_add(acc, delta)
    }
}

#[test]
fn a_backend_without_heads_takes_the_trait_defaults() {
    let (_c, plan, store) = fixture();
    let backend = NoHeadsBackend(ReferenceBackend::new());
    assert!(
        !backend.serves_attention_heads(),
        "the trait default declares no per-head observation"
    );
    let mut session = DecodeSession::new(&plan, &store, &backend).unwrap();
    let err = match session.step(G_TOKENS[0]) {
        Ok(_) => panic!("the default per-head path must refuse on the executor's call"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("per-head attention observation is not served by the no-heads backend"),
        "{err}"
    );
}

/// Reference arithmetic behind a backend that declares nothing about head
/// intervention either: `attention_step` hands the real call the executor
/// built straight to the trait's default `attention_step_intervened`, so
/// the default's own refusal is observed on a real `AttentionStepCall`
/// rather than a hand-built one.
struct NoHeadInterventionBackend(ReferenceBackend);

impl PlanBackend for NoHeadInterventionBackend {
    fn name(&self) -> &str {
        "no-head-intervention"
    }

    fn identity(&self) -> LoweringIdentity {
        LoweringIdentity::new("test-no-head-intervention", 1)
    }

    fn embed(&self, table: &[f32], hidden: usize, token: u32, scale: Option<f32>) -> Vec<f32> {
        self.0.embed(table, hidden, token, scale)
    }

    fn norm(&self, call: NormCall<'_>) -> Vec<f32> {
        self.0.norm(call)
    }

    fn project(&self, call: ProjectCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.0.project(call)
    }

    fn attention(&self, call: AttentionCall<'_>) -> Result<AttentionOut, VindexError> {
        self.0.attention(call)
    }

    fn attention_step(&self, call: AttentionStepCall<'_>) -> Result<AttentionStepOut, VindexError> {
        match PlanBackend::attention_step_intervened(self, call, None, &mut |_, _| {}) {
            Ok(_) => Err(VindexError::Parse(
                "the trait default served a head intervention it does not have".to_string(),
            )),
            Err(e) => Err(e),
        }
    }

    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.0.ffn(call)
    }

    fn routed_ffn(&self, call: RoutedFfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.0.routed_ffn(call)
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
        self.0
            .output_head(projection, vocab, hidden, x, multiplier, softcapping)
    }

    fn residual_add(&self, acc: &mut [f32], delta: &[f32]) {
        self.0.residual_add(acc, delta)
    }
}

#[test]
fn a_backend_without_head_intervention_takes_the_trait_default() {
    let (_c, plan, store) = fixture();
    let backend = NoHeadInterventionBackend(ReferenceBackend::new());
    assert!(
        !backend.serves_head_intervention(),
        "the trait default declares no head intervention"
    );
    let mut session = DecodeSession::new(&plan, &store, &backend).unwrap();
    let err = match session.step(G_TOKENS[0]) {
        Ok(_) => panic!("the default head-intervention path must refuse on the executor's call"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains(
            "per-head attention intervention is not served by the no-head-intervention backend"
        ),
        "{err}"
    );
}
