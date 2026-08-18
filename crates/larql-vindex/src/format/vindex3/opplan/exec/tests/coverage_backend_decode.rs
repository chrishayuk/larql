//! Refusal and fallback arms of the backend seam and the decode loop.
//!
//! The seam's `as_f32` is fail-closed evidence of a wrong-format load;
//! `dispatch_stats` defaults to "no device to account for". The decode
//! session refuses a plan it cannot start (no embedding, an operand the
//! store cannot resolve), refuses a token outside the table, propagates
//! a backend refusal out of the step, and realises the *absent* arms of
//! the program — two-norm placement, no final norm, no output head —
//! rather than inventing an identity for them.

use super::{dense_f32_model, VOCAB};
use crate::error::VindexError;
use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::{
    AttentionCall, AttentionStepCall, AttentionStepOut, FfnCall, NormCall, PlanBackend,
    ProjectCall, RoutedFfnCall, WeightSlice,
};
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::execute_plan;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan, LayerFfn};

/// A short prompt over the dense fixture's vocabulary.
const TOKENS: [u32; 4] = [3, 17, 60, 0];
/// The first id past the table — one row too far, not wildly out.
const OUT_OF_TABLE_TOKEN: u32 = VOCAB as u32;
/// A tensor name no segment carries.
const UNRESOLVABLE_TENSOR: &str = "no.such.tensor";
/// Bytes standing in for a foreign-format slice; their content is never
/// read because the refusal happens before any conversion.
const FOREIGN_BYTES: [u8; 2] = [0x00, 0x3c];
const NVFP4_TENSOR_SCALE: f32 = 1.0;
const F32_VALUES: [f32; 2] = [0.5, -0.25];
const FFN_REFUSAL: &str = "this backend carries no FFN kernel";
const HEAD_REFUSAL: &str = "this backend carries no output-head kernel";

/// The dense two-norm fixture (Llama-shaped): embedding, final norm and
/// head present, `post_attention_norm` / `post_ffn_norm` absent.
fn dense_fixture() -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let dir = tempfile::tempdir().unwrap();
    dense_f32_model(dir.path());
    let inventory = larql_models::inventory::build_inventory(dir.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(
        &[("dense-artifact".to_string(), inventory)],
        container.path(),
    )
    .unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, plan, store)
}

/// Step every token through a fresh session; return the last logits.
fn decode_last_logits<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
) -> Option<Vec<f32>> {
    let mut session = DecodeSession::new(plan, store, backend).unwrap();
    let mut last = None;
    for &token in TOKENS.iter() {
        last = session.step(token).unwrap().logits;
    }
    assert_eq!(session.position(), TOKENS.len());
    last
}

fn parse_message(err: VindexError) -> String {
    match err {
        VindexError::Parse(message) => message,
        other => panic!("expected a Parse refusal, got {other:?}"),
    }
}

// ── backend seam ──

/// `as_f32` hands back exactly the f32 slice it was given and refuses
/// every other representation: a backend that declared f32 receiving
/// f16/MXFP4/NVFP4 is an interpreter bug, surfaced as an error rather
/// than converted.
#[test]
fn as_f32_returns_f32_and_refuses_every_other_representation() {
    let f32_slice = WeightSlice::F32(&F32_VALUES);
    assert_eq!(f32_slice.as_f32().unwrap(), &F32_VALUES);

    let foreign = [
        ("f16", WeightSlice::F16(&FOREIGN_BYTES)),
        (
            "mxfp4",
            WeightSlice::Mxfp4 {
                packed: &FOREIGN_BYTES,
                scales: &FOREIGN_BYTES,
            },
        ),
        (
            "nvfp4",
            WeightSlice::Nvfp4 {
                packed: &FOREIGN_BYTES,
                scales: &FOREIGN_BYTES,
                tensor_scale: NVFP4_TENSOR_SCALE,
            },
        ),
    ];
    for (label, slice) in foreign {
        let message = parse_message(slice.as_f32().expect_err(label));
        assert!(
            message.contains("declared f32 weights"),
            "{label}: {message}"
        );
    }
}

/// Backends with no device keep no dispatch accounting: the trait
/// default answers `None`, and both CPU backends leave it in place.
#[test]
fn cpu_backends_report_no_dispatch_stats() {
    assert_eq!(ReferenceBackend::new().dispatch_stats(), None);
    assert_eq!(ProductionBackend::new().dispatch_stats(), None);
    assert_eq!(RefusingBackend::new(RefusedOp::Ffn).dispatch_stats(), None);
}

// ── decode session ──

/// Two-norm placement decodes: attention and FFN outputs join the
/// residual un-normalised, and the stepped program still equals the
/// batch traversal bit-for-bit — the absent post-norms are absent on
/// both paths, not identities on one.
#[test]
fn a_two_norm_plan_decodes_and_matches_the_batch_traversal() {
    let (_container, plan, store) = dense_fixture();
    assert!(plan.layers[0].post_attention_norm.is_none());
    assert!(plan.layers[0].post_ffn_norm.is_none());
    let backend = ReferenceBackend::new();
    let batch = execute_plan(&plan, &store, &TOKENS, &backend).unwrap();
    let stepped = decode_last_logits(&plan, &store, &backend).expect("plan carries a head");
    assert_eq!(batch.logits, Some(stepped));
}

/// A plan with no embedding op cannot start a session: external
/// hidden-state input is a later rung, and the refusal names the
/// component.
#[test]
fn a_session_refuses_a_plan_without_an_embedding_op() {
    let (_container, mut plan, store) = dense_fixture();
    plan.embedding = None;
    let err = DecodeSession::new(&plan, &store, &ReferenceBackend::new())
        .err()
        .expect("no embedding op must refuse");
    let message = parse_message(err);
    assert!(message.contains("has no embedding op"), "{message}");
    assert!(message.contains(&plan.component), "{message}");
}

/// A token past the embedding table is refused before any arithmetic,
/// and the refused step consumes no position.
#[test]
fn a_token_outside_the_embedding_table_is_refused_without_advancing() {
    let (_container, plan, store) = dense_fixture();
    let backend = ReferenceBackend::new();
    let mut session = DecodeSession::new(&plan, &store, &backend).unwrap();
    session.step(TOKENS[0]).unwrap();
    let refused = session
        .step(OUT_OF_TABLE_TOKEN)
        .err()
        .expect("out-of-table token must refuse");
    let message = parse_message(refused);
    assert!(
        message.contains(&format!("token id {OUT_OF_TABLE_TOKEN}")),
        "{message}"
    );
    assert_eq!(session.position(), 1, "a refused step must not advance");
}

/// Without an output head a step yields no logits (never an empty or
/// invented vector) but still advances the position.
#[test]
fn a_plan_without_an_output_head_yields_no_logits() {
    let (_container, mut plan, store) = dense_fixture();
    plan.output = None;
    let logits = decode_last_logits(&plan, &store, &ReferenceBackend::new());
    assert!(logits.is_none(), "headless plan produced logits");
}

/// Without a final norm the head reads the raw residual: the logits
/// differ from the normed program's, so the absent norm was skipped and
/// not quietly applied.
#[test]
fn a_plan_without_a_final_norm_feeds_the_raw_residual_to_the_head() {
    let (_container, plan, store) = dense_fixture();
    let backend = ReferenceBackend::new();
    let normed = decode_last_logits(&plan, &store, &backend).unwrap();
    let mut unnormed_plan = plan.clone();
    unnormed_plan.final_norm = None;
    let unnormed = decode_last_logits(&unnormed_plan, &store, &backend).unwrap();
    assert_eq!(normed.len(), unnormed.len());
    assert_ne!(
        normed, unnormed,
        "dropping the final norm must change the logits"
    );
}

/// An operand the store cannot resolve fails the session at construction,
/// at whichever site names it — attention, FFN, or the head — with the
/// tensor in the message.
#[test]
fn a_session_fails_closed_on_an_unresolvable_operand_at_every_site() {
    let (_container, plan, store) = dense_fixture();
    let backend = ReferenceBackend::new();

    let mut attention_broken = plan.clone();
    attention_broken.layers[0].attention.q.tensor = UNRESOLVABLE_TENSOR.to_string();
    let mut ffn_broken = plan.clone();
    let LayerFfn::Dense(op) = &mut ffn_broken.layers[0].ffn else {
        panic!("dense fixture");
    };
    op.up.tensor = UNRESOLVABLE_TENSOR.to_string();
    let mut head_broken = plan;
    head_broken
        .output
        .as_mut()
        .expect("fixture carries a head")
        .projection
        .tensor = UNRESOLVABLE_TENSOR.to_string();

    for (site, broken) in [
        ("attention", attention_broken),
        ("ffn", ffn_broken),
        ("head", head_broken),
    ] {
        let err = DecodeSession::new(&broken, &store, &backend)
            .err()
            .unwrap_or_else(|| panic!("{site}: an unresolvable operand must refuse"));
        let message = parse_message(err);
        assert!(message.contains(UNRESOLVABLE_TENSOR), "{site}: {message}");
    }
}

/// A backend that refuses the FFN refuses the step: the session
/// propagates the refusal instead of substituting another backend's
/// arithmetic.
#[test]
fn a_backend_ffn_refusal_propagates_out_of_the_step() {
    let (_container, plan, store) = dense_fixture();
    let backend = RefusingBackend::new(RefusedOp::Ffn);
    let mut session = DecodeSession::new(&plan, &store, &backend).unwrap();
    let refused = session
        .step(TOKENS[0])
        .err()
        .expect("the FFN refusal must reach the caller");
    let message = parse_message(refused);
    assert_eq!(message, FFN_REFUSAL);
}

/// The same for the output head: every layer ran, and the head's
/// refusal is what the step returns.
#[test]
fn a_backend_output_head_refusal_propagates_out_of_the_step() {
    let (_container, plan, store) = dense_fixture();
    let backend = RefusingBackend::new(RefusedOp::OutputHead);
    let mut session = DecodeSession::new(&plan, &store, &backend).unwrap();
    let refused = session
        .step(TOKENS[0])
        .err()
        .expect("the head refusal must reach the caller");
    let message = parse_message(refused);
    assert_eq!(message, HEAD_REFUSAL);
}

/// Which operation a [`RefusingBackend`] has no kernel for.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RefusedOp {
    Ffn,
    OutputHead,
}

/// Reference arithmetic everywhere except one operation, which it
/// refuses — the shape of a device backend missing one kernel.
struct RefusingBackend {
    inner: ReferenceBackend,
    refuse: RefusedOp,
}

impl RefusingBackend {
    fn new(refuse: RefusedOp) -> Self {
        Self {
            inner: ReferenceBackend::new(),
            refuse,
        }
    }
}

impl PlanBackend for RefusingBackend {
    fn name(&self) -> &str {
        "refusing"
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

    fn attention(&self, call: AttentionCall<'_>) -> Result<Vec<Vec<f32>>, VindexError> {
        self.inner.attention(call)
    }

    fn attention_step(&self, call: AttentionStepCall<'_>) -> Result<AttentionStepOut, VindexError> {
        self.inner.attention_step(call)
    }

    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        if self.refuse == RefusedOp::Ffn {
            return Err(VindexError::Parse(FFN_REFUSAL.to_string()));
        }
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
        if self.refuse == RefusedOp::OutputHead {
            return Err(VindexError::Parse(HEAD_REFUSAL.to_string()));
        }
        self.inner
            .output_head(projection, vocab, hidden, x, multiplier, softcapping)
    }

    fn residual_add(&self, acc: &mut [f32], delta: &[f32]) {
        self.inner.residual_add(acc, delta)
    }
}
