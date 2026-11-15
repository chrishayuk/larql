//! **Every position's logits from one block traversal.**
//!
//! `step_many_logits` must return, for each position of a block, exactly
//! the logits `step` returns at that position, and must leave exactly the
//! continuation state sequential decode leaves. `step_many` keeps its own
//! contract: last position only, unchanged.

use super::golden::{G_TOKENS, G_VOCAB, G_WINDOW};
use crate::error::VindexError;
use crate::format::vindex3::opplan::exec::backend::{
    AttentionCall, AttentionOut, AttentionStepCall, AttentionStepOut, FfnCall, NormCall,
    PlanBackend, ProjectCall, RoutedFfnCall, WeightSlice,
};
use crate::format::vindex3::opplan::exec::continuation::{LatentKvRows, RecurrentState};
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::kv::{
    ContinuationError, KvState, LayerKvGeometry, RowKvState,
};
use crate::format::vindex3::opplan::exec::lowering::LoweringIdentity;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use std::sync::atomic::{AtomicBool, Ordering};

/// Block widths under test.
const WIDTHS: [usize; 4] = [1, 2, 4, 8];
/// Longest prefix decoded before the block: blocks therefore start before
/// the sliding window fills, straddle its boundary and start beyond it.
const PREFIX_MAX: usize = 3 * G_WINDOW;
/// Positions decoded one at a time after the block, to prove the state the
/// block left behind and not only the logits it returned.
const FOLLOW_ON: usize = G_WINDOW + 2;

/// A deterministic token stream longer than the golden one, inside the vocabulary.
fn token(index: usize) -> u32 {
    (G_TOKENS[index % G_TOKENS.len()] as usize * (index / G_TOKENS.len() + 1) % G_VOCAB) as u32
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn gate<B: PlanBackend>(backend: &B) {
    let (_container, plan, store) = super::decode::fixture();
    for prefix in 0..=PREFIX_MAX {
        for width in WIDTHS {
            let before: Vec<u32> = (0..prefix).map(token).collect();
            let block: Vec<u32> = (prefix..prefix + width).map(token).collect();
            let after: Vec<u32> = (prefix + width..prefix + width + FOLLOW_ON)
                .map(token)
                .collect();

            // Sequential reference: one `step` per position throughout.
            let mut sequential_kv = RowKvState::default();
            let mut sequential =
                DecodeSession::with_kv_state(&plan, &store, backend, &mut sequential_kv).unwrap();
            for t in &before {
                sequential.step(*t).unwrap();
            }
            let expected: Vec<Vec<u32>> = block
                .iter()
                .map(|t| bits(&sequential.step(*t).unwrap().logits.unwrap()))
                .collect();
            let expected_after: Vec<Vec<u32>> = after
                .iter()
                .map(|t| bits(&sequential.step(*t).unwrap().logits.unwrap()))
                .collect();

            // The block, in one traversal, every position's logits.
            let mut kv = RowKvState::default();
            let mut session =
                DecodeSession::with_kv_state(&plan, &store, backend, &mut kv).unwrap();
            for t in &before {
                session.step(*t).unwrap();
            }
            let rows = session.step_many_logits(&block).unwrap();
            assert_eq!(rows.len(), width);
            let got: Vec<Vec<u32>> = rows.iter().map(|row| bits(row)).collect();
            assert_eq!(
                got, expected,
                "prefix {prefix}, width {width}: block logits"
            );
            assert_eq!(session.position(), prefix + width);
            // The state it left: every later position still agrees.
            let got_after: Vec<Vec<u32>> = after
                .iter()
                .map(|t| bits(&session.step(*t).unwrap().logits.unwrap()))
                .collect();
            assert_eq!(
                got_after, expected_after,
                "prefix {prefix}, width {width}: follow-on"
            );

            // `step_many` keeps its contract: the last position's logits.
            let mut last_kv = RowKvState::default();
            let mut last =
                DecodeSession::with_kv_state(&plan, &store, backend, &mut last_kv).unwrap();
            for t in &before {
                last.step(*t).unwrap();
            }
            let only_last = last.step_many(&block).unwrap().logits.unwrap();
            assert_eq!(bits(&only_last), expected[width - 1], "step_many changed");
        }
    }
}

#[test]
fn every_position_matches_sequential_decode_production() {
    gate(&ProductionBackend::new());
}

#[test]
fn every_position_matches_sequential_decode_reference() {
    gate(&ReferenceBackend::new());
}

#[test]
fn a_refused_block_moves_no_state() {
    let (_container, plan, store) = super::decode::fixture();
    let backend = ProductionBackend::new();
    let mut kv = RowKvState::default();
    {
        let mut session = DecodeSession::with_kv_state(&plan, &store, &backend, &mut kv).unwrap();
        session.step(token(0)).unwrap();
        let error = session.step_many_logits(&[]).unwrap_err().to_string();
        assert!(error.contains("given none"), "{error}");
        // A bad id in the LAST place: the earlier tokens must not have run.
        let error = session
            .step_many_logits(&[token(1), token(2), G_VOCAB as u32])
            .unwrap_err()
            .to_string();
        assert!(error.contains("outside the embedding table"), "{error}");
        assert_eq!(session.position(), 1);
    }
    assert_eq!(kv.position(), 1);
    assert_eq!(kv.rows(0).end(), 1, "a refused block appended rows");
}

/// A layer-range image has no output head, so no position has logits to
/// return: refused before any state moves.
#[test]
fn a_headless_image_is_refused_before_any_state_moves() {
    let (_container, plan, store) = super::decode::fixture();
    let backend = ProductionBackend::new();
    let slice = ExecutionSlice::LayerRange { start: 0, end: 1 };
    let sliced = PreparedOperands::load(&plan, &store, &backend, slice).unwrap();
    let mut kv = RowKvState::default();
    {
        let mut session = DecodeSession::over_prepared(&plan, &sliced, &backend, &mut kv).unwrap();
        let error = session
            .step_many_logits(&[token(0), token(1)])
            .unwrap_err()
            .to_string();
        assert!(error.contains("no output head"), "{error}");
    }
    assert_eq!(kv.position(), 0);
}

/// Tokens of a block that fails: not the tokens the continuation then
/// uses, so a row left behind would change the result.
fn failed_token(index: usize) -> u32 {
    ((token(index) as usize + 7) % G_VOCAB) as u32
}

const HEAD_FAILURE: &str = "the output head failed";

/// Production arithmetic, with an output head that fails while armed. The
/// head runs at the traversal's exit, after every layer of every position
/// has changed the continuation: the latest a block step can fail.
struct FailingHead {
    inner: ProductionBackend,
    armed: AtomicBool,
}

impl FailingHead {
    fn new() -> Self {
        Self {
            inner: ProductionBackend::new(),
            armed: AtomicBool::new(false),
        }
    }

    fn arm(&self, armed: bool) {
        self.armed.store(armed, Ordering::SeqCst);
    }
}

impl PlanBackend for FailingHead {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn identity(&self) -> LoweringIdentity {
        self.inner.identity()
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
    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.inner.ffn(call)
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
        if self.armed.load(Ordering::SeqCst) {
            return Err(VindexError::Parse(HEAD_FAILURE.to_string()));
        }
        self.inner
            .output_head(projection, vocab, hidden, x, multiplier, softcapping)
    }
    fn residual_add(&self, acc: &mut [f32], delta: &[f32]) {
        self.inner.residual_add(acc, delta)
    }
}

/// The two block steps, as one failing call each.
type BlockStep = fn(&mut DecodeSession<'_, FailingHead>, &[u32]) -> Result<(), VindexError>;

const BLOCK_STEPS: [(&str, BlockStep); 2] = [
    ("step_many", |session, tokens| {
        session.step_many(tokens).map(|_| ())
    }),
    ("step_many_logits", |session, tokens| {
        session.step_many_logits(tokens).map(|_| ())
    }),
];

/// A failed block leaves the continuation unchanged: the same position,
/// no row of the block, and every later step agreeing with a state that
/// never saw it. Through both block steps, with the failure at the exit,
/// after the traversal has changed every layer.
#[test]
fn a_block_that_fails_after_its_traversal_leaves_the_state_unchanged() {
    let (_container, plan, store) = super::decode::fixture();
    let backend = FailingHead::new();
    for (name, block_step) in BLOCK_STEPS {
        for prefix in [0, 1, G_WINDOW + 1] {
            let before: Vec<u32> = (0..prefix).map(token).collect();
            let failed: Vec<u32> = (prefix..prefix + G_WINDOW + 1).map(failed_token).collect();
            let after: Vec<u32> = (prefix..prefix + FOLLOW_ON).map(token).collect();

            let mut clean_kv = RowKvState::default();
            let mut clean =
                DecodeSession::with_kv_state(&plan, &store, &backend, &mut clean_kv).unwrap();
            for t in &before {
                clean.step(*t).unwrap();
            }
            let expected: Vec<Vec<u32>> = after
                .iter()
                .map(|t| bits(&clean.step(*t).unwrap().logits.unwrap()))
                .collect();

            let mut kv = RowKvState::default();
            let mut session =
                DecodeSession::with_kv_state(&plan, &store, &backend, &mut kv).unwrap();
            for t in &before {
                session.step(*t).unwrap();
            }
            backend.arm(true);
            let error = block_step(&mut session, &failed).unwrap_err().to_string();
            backend.arm(false);
            assert!(error.contains(HEAD_FAILURE), "{name}: {error}");
            assert!(!error.contains("unusable"), "{name}: {error}");
            assert_eq!(session.position(), prefix, "{name}: position moved");
            // The same session goes on, exactly as if the block had never run.
            let got: Vec<Vec<u32>> = after
                .iter()
                .map(|t| bits(&session.step(*t).unwrap().logits.unwrap()))
                .collect();
            assert_eq!(
                got, expected,
                "{name}, prefix {prefix}: steps after a failed block"
            );
            drop(session);
            assert_eq!(
                kv.rows(0).end(),
                prefix + after.len(),
                "{name}: rows remain"
            );
            assert!(kv.invalidated().is_none());
        }
    }
}

/// A provider that states no rollback: it forwards everything the trait
/// requires to real rows, and inherits the default refusal to rewind.
#[derive(Default)]
struct NoRollback(RowKvState);

impl KvState for NoRollback {
    fn prepare(&mut self, layers: &[LayerKvGeometry]) {
        self.0.prepare(layers)
    }
    fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) {
        self.0.append(layer, key, value)
    }
    fn rows(&self, layer: usize) -> super::super::kv_view::KvView<'_> {
        self.0.rows(layer)
    }
    fn prepare_layer(&mut self, layer: usize) {
        self.0.prepare_layer(layer)
    }
    fn position(&self) -> usize {
        self.0.position()
    }
    fn set_position(&mut self, position: usize) {
        self.0.set_position(position)
    }
    fn invalidate(&mut self, reason: &str) {
        self.0.invalidate(reason)
    }
    fn invalidated(&self) -> Option<&str> {
        self.0.invalidated()
    }
    fn recurrent_state(&mut self, layer: usize) -> Result<&mut RecurrentState, ContinuationError> {
        self.0.recurrent_state(layer)
    }
    fn latent_state(&mut self, layer: usize) -> Result<&mut LatentKvRows, ContinuationError> {
        self.0.latent_state(layer)
    }
}

/// A failed block over a provider that cannot put the state back leaves
/// the STATE unusable, not only the session that saw the failure: that
/// session refuses every further step by every entry point, and so does a
/// new session opened over the same state.
#[test]
fn a_failed_block_that_cannot_be_put_back_makes_the_state_unusable() {
    let (_container, plan, store) = super::decode::fixture();
    let backend = FailingHead::new();
    for (name, block_step) in BLOCK_STEPS {
        let mut kv = NoRollback::default();
        let mut session = DecodeSession::with_kv_state(&plan, &store, &backend, &mut kv).unwrap();
        session.step(token(0)).unwrap();
        backend.arm(true);
        let error = block_step(&mut session, &[token(1), token(2)])
            .unwrap_err()
            .to_string();
        backend.arm(false);
        assert!(error.contains(HEAD_FAILURE), "{name}: {error}");
        assert!(error.contains(UNUSABLE), "{name}: {error}");
        assert!(
            error.contains("does not support rollback"),
            "{name}: {error}"
        );
        assert_every_step_is_refused(&mut session, name);
        drop(session);

        let recorded = kv
            .invalidated()
            .expect("the provider records it")
            .to_string();
        assert!(recorded.contains(HEAD_FAILURE), "{name}: {recorded}");
        let mut later = DecodeSession::with_kv_state(&plan, &store, &backend, &mut kv).unwrap();
        assert_every_step_is_refused(&mut later, name);
    }
}

/// What every refusal of an unusable continuation says.
const UNUSABLE: &str = "unusable and must be discarded";

fn assert_every_step_is_refused<B: PlanBackend>(session: &mut DecodeSession<'_, B>, name: &str) {
    let refusals = [
        session.step(token(1)).map(|_| ()),
        session.step_many(&[token(1)]).map(|_| ()),
        session.step_many_logits(&[token(1)]).map(|_| ()),
    ];
    for refusal in refusals {
        let refusal = refusal.unwrap_err().to_string();
        assert!(refusal.contains(UNUSABLE), "{name}: {refusal}");
    }
}

/// An invalidated row state stays invalidated: a later session refuses to
/// step it, a batch traversal refuses to run over it, and it cannot be
/// rewound back into use.
#[test]
fn an_invalidated_row_state_cannot_be_reused_by_any_route() {
    const REASON: &str = "a failed advance could not be put back";
    let (_container, plan, store) = super::decode::fixture();
    let backend = ProductionBackend::new();
    let mut kv = RowKvState::default();
    {
        let mut session = DecodeSession::with_kv_state(&plan, &store, &backend, &mut kv).unwrap();
        session.step(token(0)).unwrap();
    }
    assert!(kv.invalidated().is_none());
    kv.invalidate(REASON);
    assert_eq!(kv.invalidated(), Some(REASON));

    {
        let mut session = DecodeSession::with_kv_state(&plan, &store, &backend, &mut kv).unwrap();
        assert_every_step_is_refused(&mut session, "row state");
        assert_eq!(session.position(), 1);
    }
    // The batch traversal, with no session in between.
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let batch = crate::format::vindex3::opplan::exec::traverse(
        &plan,
        &ops,
        &[token(1)],
        &backend,
        None,
        &mut |_| Ok(()),
        Some(&mut kv),
        crate::format::vindex3::opplan::exec::hyper_connection::Mutation::None,
    );
    let refusal = batch.map(|_| ()).unwrap_err().to_string();
    assert!(
        refusal.contains(UNUSABLE) && refusal.contains(REASON),
        "{refusal}"
    );
    assert_eq!(kv.rows(0).end(), 1, "a refused traversal appended rows");

    // A rewind restores a position. It does not restore trust.
    for target in [0, 1] {
        let refusal = kv.truncate(target).unwrap_err().to_string();
        assert!(refusal.contains(REASON), "{refusal}");
    }
    assert_eq!((kv.position(), kv.rows(0).end()), (1, 1));
    assert_eq!(
        kv.clone().invalidated(),
        Some(REASON),
        "a copy is no more usable"
    );
}
