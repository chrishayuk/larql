//! `DecodeSession::step_embedding`: an external row enters the stack
//! exactly as the token lookup's own row would.
//!
//! The oracle is the session itself. A token run records the carrier
//! entering layer 0 at every position; a fresh session fed those rows
//! through `step_embedding` must reproduce the token run's logits bit
//! for bit — on the single-stream, hyper-connected and
//! attention-residual topologies, whose entry each wraps the row
//! differently.
use super::decode::fixture;
use super::{attn_res_substrate, wave19_hc_decode, wave19_hc_substrate};
use crate::format::vindex3::fixtures::G_TOKENS;
use crate::format::vindex3::opplan::exec::{
    decode::DecodeSession,
    kv::RowKvState,
    observe::{StepEvent, StepObserver},
    prepared::{ExecutionSlice, PreparedOperands},
    reference::ReferenceBackend,
};
use crate::format::vindex3::opplan::ComponentOpPlan;

/// Tokens for the two topology substrates, inside their vocabulary.
const SUBSTRATE_TOKENS: [u32; 3] = [1, 4, 2];

/// Every row that entered layer 0, in position order.
#[derive(Default)]
struct EnteringRows(Vec<Vec<f32>>);

impl StepObserver for EnteringRows {
    fn event(&mut self, _: StepEvent) {}
    fn entering_carrier(&mut self, _position: usize, values: &[f32]) {
        self.0.push(values.to_vec());
    }
}

/// Run `tokens` through a whole-stack image, then run the rows they
/// entered as through `step_embedding` on a fresh session; the logits
/// must agree at every position.
fn assert_rows_reproduce_tokens(plan: &ComponentOpPlan, ops: &PreparedOperands, tokens: &[u32]) {
    let backend = ReferenceBackend::new();
    let mut rows = EnteringRows::default();
    let mut token_kv = RowKvState::default();
    let mut by_token = DecodeSession::over_prepared(plan, ops, &backend, &mut token_kv).unwrap();
    let expected: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&t| {
            by_token
                .step_observed(t, &mut rows)
                .unwrap()
                .logits
                .expect("a whole-stack image carries a head")
        })
        .collect();
    assert_eq!(rows.0.len(), tokens.len(), "one entering row per position");

    let mut row_kv = RowKvState::default();
    let mut by_row = DecodeSession::over_prepared(plan, ops, &backend, &mut row_kv).unwrap();
    for (position, (row, want)) in rows.0.iter().zip(&expected).enumerate() {
        let got = by_row.step_embedding(row).unwrap().logits;
        assert_eq!(got.as_ref(), Some(want), "position {position}");
    }
    assert_eq!(by_row.position(), tokens.len());
}

#[test]
fn a_single_stream_row_reproduces_the_token_lookup() {
    let (_dir, plan, store) = fixture();
    let ops =
        PreparedOperands::load(&plan, &store, &ReferenceBackend, ExecutionSlice::Full).unwrap();
    assert_rows_reproduce_tokens(&plan, &ops, &G_TOKENS);
}

#[test]
fn a_hyper_connected_row_replicates_like_the_token_lookup() {
    let sub = wave19_hc_substrate::build(wave19_hc_substrate::Variant::HeadBearing);
    let ops = wave19_hc_decode::prepare(&sub, ExecutionSlice::Full).unwrap();
    assert_rows_reproduce_tokens(&sub.plan, &ops, &SUBSTRATE_TOKENS);
}

#[test]
fn an_attention_residual_row_enters_as_the_first_prefix() {
    let sub = attn_res_substrate::substrate();
    let (_store, ops) = attn_res_substrate::prepare(&sub);
    assert_rows_reproduce_tokens(&sub.plan, &ops, &SUBSTRATE_TOKENS);
}

#[test]
fn a_malformed_row_refuses_before_the_position_advances() {
    let (_dir, plan, store) = fixture();
    let backend = ReferenceBackend;
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let hidden = ops.hidden();
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv).unwrap();

    let short = vec![0.0; hidden - 1];
    let mut non_finite = vec![0.0; hidden];
    non_finite[hidden / 2] = f32::NAN;
    for (what, row) in [("short", short), ("non-finite", non_finite)] {
        let err = session.step_embedding(&row).err().unwrap().to_string();
        assert!(err.contains("finite values"), "{what}: {err}");
        assert_eq!(session.position(), 0, "{what} advanced the position");
    }
}

#[test]
fn a_layer_range_image_refuses_an_external_row() {
    let (_dir, plan, store) = fixture();
    let backend = ReferenceBackend;
    let tail = PreparedOperands::load(
        &plan,
        &store,
        &backend,
        ExecutionSlice::LayerRange {
            start: 1,
            end: plan.layers.len(),
        },
    )
    .unwrap();
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&plan, &tail, &backend, &mut kv).unwrap();
    let err = session
        .step_embedding(&vec![0.0; tail.hidden()])
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("whole-stack image"), "{err}");
    assert_eq!(session.position(), 0);
}

#[test]
fn an_endpoints_image_refuses_a_session() {
    let (_dir, plan, store) = fixture();
    let backend = ReferenceBackend;
    let endpoints =
        PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Endpoints).unwrap();
    let mut kv = RowKvState::default();
    let err = DecodeSession::over_prepared(&plan, &endpoints, &backend, &mut kv)
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("distributed coordinator"), "{err}");
}
