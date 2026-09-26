//! V3-INTERVENE-2 acceptance on the golden plan
//! (`docs/v3-intervene-2-head-intervention.md`): JP1 the no-op law, JP2
//! exactness at the head address, JP4 backend parity, JP5 refusals and
//! the record surface (tested through the engine's own no-op/exact
//! results here; the receipt/gap wiring is exercised in
//! `larql-inference`), and J7's donor head capture.

use super::decode::fixture;
use super::golden::{G_HIDDEN, G_LAYERS, G_Q_HEADS, G_TOKENS};
use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::intervene::InterventionPlan;
use crate::format::vindex3::opplan::exec::intervene_heads::{
    HeadAddress, HeadCapture, HeadFiring, HeadInterventionKind, HeadInterventionPlan,
};
use crate::format::vindex3::opplan::exec::observe::{NoopObserver, StepObserver};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::ComponentOpPlan;

/// The layer and position this file intervenes at: inside the prompt,
/// with positions on both sides.
const L: usize = 0;
const P: usize = 3;

fn address(head: usize) -> HeadAddress {
    HeadAddress::new(L, head, [P]).unwrap()
}

fn plain<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
    tokens: &[u32],
) -> Vec<Vec<f32>> {
    let mut session = DecodeSession::new(
        plan,
        store,
        backend,
        Box::new(crate::format::vindex3::opplan::exec::kv::RowKvState::default()),
    )
    .unwrap();
    tokens
        .iter()
        .map(|&t| session.step(t).unwrap().logits.unwrap())
        .collect()
}

fn intervened<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
    tokens: &[u32],
    heads: &HeadInterventionPlan,
) -> (Vec<Vec<f32>>, Vec<HeadFiring>) {
    let mut session = DecodeSession::new(
        plan,
        store,
        backend,
        Box::new(crate::format::vindex3::opplan::exec::kv::RowKvState::default()),
    )
    .unwrap();
    let mut firings = Vec::new();
    let logits = tokens
        .iter()
        .map(|&t| {
            let out = session
                .step_intervened(t, &mut NoopObserver, &InterventionPlan::none(), heads)
                .unwrap();
            firings.extend(out.head_firings);
            out.logits.unwrap()
        })
        .collect();
    (logits, firings)
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn assert_logits_equal(a: &[Vec<f32>], b: &[Vec<f32>], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: position counts differ");
    for (p, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(bits_equal(x, y), "{what}: logits differ at position {p}");
    }
}

macro_rules! on_both_backends {
    (|$backend:ident| $body:block) => {{
        {
            let $backend: &ReferenceBackend = &ReferenceBackend::new();
            $body
        }
        {
            let $backend: &ProductionBackend = &ProductionBackend::new();
            $body
        }
    }};
}

// ── JP1: the no-op law ──────────────────────────────────────────────

#[test]
fn jp1_empty_scale_one_and_replace_own_value_are_the_unintervened_run_bit_for_bit() {
    let (_c, plan, store) = fixture();
    on_both_backends!(|backend| {
        let baseline = plain(&plan, &store, backend, &G_TOKENS);

        let (none, firings) = intervened(
            &plan,
            &store,
            backend,
            &G_TOKENS,
            &HeadInterventionPlan::none(),
        );
        assert_logits_equal(&baseline, &none, "none");
        assert!(firings.is_empty(), "nothing declared, nothing fired");

        let scale_one = HeadInterventionPlan::none()
            .with(
                crate::format::vindex3::opplan::exec::intervene_heads::HeadIntervention::scale(
                    address(0),
                    1.0,
                )
                .unwrap(),
            )
            .unwrap();
        let (scaled, firings) = intervened(&plan, &store, backend, &G_TOKENS, &scale_one);
        assert_logits_equal(&baseline, &scaled, "scale(1)");
        assert_eq!(
            firings,
            vec![HeadFiring {
                layer: L,
                head: 0,
                position: P,
                kind: HeadInterventionKind::Scale,
            }]
        );
    });
}

// ── JP2: exactness and causality ────────────────────────────────────

#[test]
fn jp2_positions_before_the_address_are_untouched_and_the_address_position_moves() {
    let (_c, plan, store) = fixture();
    on_both_backends!(|backend| {
        let baseline = plain(&plan, &store, backend, &G_TOKENS);
        let zero = HeadInterventionPlan::none()
            .with(
                crate::format::vindex3::opplan::exec::intervene_heads::HeadIntervention::zero(
                    address(0),
                ),
            )
            .unwrap();
        let (logits, firings) = intervened(&plan, &store, backend, &G_TOKENS, &zero);
        assert_logits_equal(&baseline[..P], &logits[..P], "positions before the address");
        assert!(
            !bits_equal(&baseline[P], &logits[P]),
            "zeroing a head at position {P} must move that position's logits"
        );
        assert_eq!(firings.len(), 1);
        assert_eq!(firings[0].kind, HeadInterventionKind::Zero);
    });
}

#[test]
fn jp2_zeroing_every_head_is_stronger_than_zeroing_one() {
    let (_c, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let baseline = plain(&plan, &store, &backend, &G_TOKENS);
    let one = HeadInterventionPlan::none()
        .with(
            crate::format::vindex3::opplan::exec::intervene_heads::HeadIntervention::zero(address(
                0,
            )),
        )
        .unwrap();
    let (one_logits, _) = intervened(&plan, &store, &backend, &G_TOKENS, &one);
    let mut all = HeadInterventionPlan::none();
    for h in 0..G_Q_HEADS {
        all = all
            .with(
                crate::format::vindex3::opplan::exec::intervene_heads::HeadIntervention::zero(
                    address(h),
                ),
            )
            .unwrap();
    }
    let (all_logits, firings) = intervened(&plan, &store, &backend, &G_TOKENS, &all);
    assert_eq!(firings.len(), G_Q_HEADS);
    assert!(!bits_equal(&one_logits[P], &all_logits[P]));
    assert!(!bits_equal(&baseline[P], &all_logits[P]));
}

#[test]
fn jp2_scale_and_replace_are_exact_and_a_declared_position_outside_the_set_is_untouched() {
    let (_c, plan, store) = fixture();
    let backend = ProductionBackend::new();
    // A distinct factor and vector no head will equal by accident.
    let scaled = HeadInterventionPlan::none()
        .with(
            crate::format::vindex3::opplan::exec::intervene_heads::HeadIntervention::scale(
                address(1),
                0.0,
            )
            .unwrap(),
        )
        .unwrap();
    let (_, firings) = intervened(&plan, &store, &backend, &G_TOKENS, &scaled);
    assert_eq!(firings[0].kind, HeadInterventionKind::Scale);

    let (vector, provenance) = {
        let values: Vec<f32> = (0..G_HIDDEN / G_Q_HEADS)
            .map(|i| 0.5 * (i as f32 + 1.0))
            .collect();
        let sha256 = crate::format::vindex3::opplan::exec::intervene::vector_sha256(&values);
        (
            values,
            crate::format::vindex3::opplan::exec::intervene::VectorProvenance::Literal { sha256 },
        )
    };
    let replaced = HeadInterventionPlan::none()
        .with(
            crate::format::vindex3::opplan::exec::intervene_heads::HeadIntervention::replace(
                address(2),
                vector,
                provenance,
            )
            .unwrap(),
        )
        .unwrap();
    let (_, firings) = intervened(&plan, &store, &backend, &G_TOKENS, &replaced);
    assert_eq!(firings[0].kind, HeadInterventionKind::Replace);
}

// ── JP4: reference gates production ─────────────────────────────────

/// The backends' existing attention parity band (matching the tolerance
/// `wave19_hc_decode`/`attn_res_substrate` already use for cross-backend
/// comparison) — JP4 asks for token-for-token agreement and this band,
/// never bit identity: the two backends are different arithmetic.
const PARITY_TOLERANCE: f32 = 5e-4;

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best
}

#[test]
fn jp4_zeroing_a_head_agrees_token_for_token_and_within_parity_between_backends() {
    let (_c, plan, store) = fixture();
    let zero = HeadInterventionPlan::none()
        .with(
            crate::format::vindex3::opplan::exec::intervene_heads::HeadIntervention::zero(address(
                0,
            )),
        )
        .unwrap();
    let (reference, ref_firings) =
        intervened(&plan, &store, &ReferenceBackend::new(), &G_TOKENS, &zero);
    let (production, prod_firings) =
        intervened(&plan, &store, &ProductionBackend::new(), &G_TOKENS, &zero);
    assert_eq!(
        ref_firings, prod_firings,
        "both backends fire the same declared intervention"
    );
    for (p, (r, q)) in reference.iter().zip(&production).enumerate() {
        assert_eq!(
            argmax(r),
            argmax(q),
            "position {p}: the backends' greedy continuation must agree token for token"
        );
        let max_abs_diff = r
            .iter()
            .zip(q)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff <= PARITY_TOLERANCE,
            "position {p}: logits differ by {max_abs_diff}, outside the parity band {PARITY_TOLERANCE}"
        );
    }
}

// ── JP5: refusals ────────────────────────────────────────────────────

fn refused<T: std::fmt::Debug>(result: Result<T, crate::error::VindexError>, needle: &str) {
    match result {
        Ok(v) => panic!("expected a refusal mentioning `{needle}`, got {v:?}"),
        Err(e) => {
            let text = e.to_string();
            assert!(
                text.contains(needle),
                "refusal `{text}` does not mention `{needle}`"
            );
        }
    }
}

#[test]
fn jp5_construction_refuses_what_it_cannot_represent() {
    use crate::format::vindex3::opplan::exec::intervene::VectorProvenance;
    use crate::format::vindex3::opplan::exec::intervene_heads::HeadIntervention;

    refused(HeadAddress::new(L, 0, []), "names no position");
    refused(
        HeadIntervention::scale(address(0), f32::NAN),
        "non-finite factor",
    );
    refused(
        HeadIntervention::replace(
            address(0),
            Vec::new(),
            VectorProvenance::Literal {
                sha256: String::new(),
            },
        ),
        "empty vector",
    );
    refused(
        HeadIntervention::replace(
            address(0),
            vec![1.0, f32::NAN],
            VectorProvenance::Literal {
                sha256: "deadbeef".into(),
            },
        ),
        "non-finite value at index 1",
    );

    let first = HeadIntervention::zero(HeadAddress::new(L, 0, [P, P + 1]).unwrap());
    let second = HeadIntervention::zero(HeadAddress::new(L, 0, [P + 1, P + 2]).unwrap());
    refused(
        HeadInterventionPlan::none()
            .with(first)
            .unwrap()
            .with(second),
        "depend on declaration order",
    );
}

#[test]
fn jp5_admission_refuses_before_the_first_token() {
    use crate::format::vindex3::opplan::exec::intervene_heads::HeadIntervention;
    use crate::format::vindex3::opplan::exec::prepared::PreparedOperands;

    let (_c, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(
        &plan,
        &store,
        &backend,
        crate::format::vindex3::opplan::exec::prepared::ExecutionSlice::Full,
    )
    .unwrap();

    let off_layer = HeadInterventionPlan::none()
        .with(HeadIntervention::zero(
            HeadAddress::new(G_LAYERS, 0, [P]).unwrap(),
        ))
        .unwrap();
    refused(off_layer.admit(&plan, &ops), "executes layers 0..");

    let off_head = HeadInterventionPlan::none()
        .with(HeadIntervention::zero(
            HeadAddress::new(L, G_Q_HEADS, [P]).unwrap(),
        ))
        .unwrap();
    refused(off_head.admit(&plan, &ops), "query heads");

    let (vec, provenance) = {
        let values = vec![0.0f32; G_HIDDEN]; // wrong width: not head_dim
        let sha256 = crate::format::vindex3::opplan::exec::intervene::vector_sha256(&values);
        (
            values,
            crate::format::vindex3::opplan::exec::intervene::VectorProvenance::Literal { sha256 },
        )
    };
    let wrong_width = HeadInterventionPlan::none()
        .with(HeadIntervention::replace(address(0), vec, provenance).unwrap())
        .unwrap();
    refused(wrong_width.admit(&plan, &ops), "carries a vector of width");

    HeadInterventionPlan::none()
        .admit(&plan, &ops)
        .expect("nothing declared admits everywhere");
}

#[test]
fn jp5_a_bundle_carrier_is_refused_by_declaration() {
    use super::wave19_hc_decode::{layer_range, prepare};
    use super::wave19_hc_substrate::{build, Variant};
    use crate::format::vindex3::opplan::exec::intervene_heads::HeadIntervention;
    let sub = build(Variant::Headless);
    let ops = prepare(&sub, layer_range()).expect("the hyper-connected substrate prepares");
    let declared = HeadInterventionPlan::none()
        .with(HeadIntervention::zero(HeadAddress::new(0, 0, [0]).unwrap()))
        .unwrap();
    refused(declared.admit(&sub.plan, &ops), "`Bundle`");
}

#[test]
fn jp5_a_history_carrier_is_refused_by_declaration() {
    use super::attn_res_substrate::substrate;
    use crate::format::vindex3::opplan::exec::intervene_heads::HeadIntervention;
    use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
    let sub = substrate();
    let store = OperandStore::open(sub.container.path(), &sub.inspection).unwrap();
    let ops = PreparedOperands::load(
        &sub.plan,
        &store,
        &ReferenceBackend::new(),
        ExecutionSlice::Full,
    )
    .expect("the attention-residual substrate prepares");
    let declared = HeadInterventionPlan::none()
        .with(HeadIntervention::zero(HeadAddress::new(0, 0, [0]).unwrap()))
        .unwrap();
    refused(declared.admit(&sub.plan, &ops), "`History`");
}

// ── J7: donor head capture ──────────────────────────────────────────

struct HeadTap {
    captured: HeadCapture,
}

impl StepObserver for HeadTap {
    fn event(&mut self, _event: crate::format::vindex3::opplan::exec::observe::StepEvent) {}
    fn wants_attention_heads(&self) -> bool {
        true
    }
    fn attention_head(
        &mut self,
        layer: usize,
        record: crate::format::vindex3::opplan::exec::observe::AttentionHeadRecord<'_>,
    ) {
        self.captured.observe(layer, &record);
    }
}

#[test]
fn j7_a_captured_head_replaces_bit_for_bit_and_names_its_run() {
    let (_c, plan, store) = fixture();
    let backend = ProductionBackend::new();

    // Run B: a different prompt, captured at (L, head 0, P).
    let donor: Vec<u32> = G_TOKENS.iter().rev().copied().collect();
    let mut tap = HeadTap {
        captured: HeadCapture::at([(L, 0, P)]),
    };
    {
        let mut session = DecodeSession::new(
            &plan,
            &store,
            &backend,
            Box::new(crate::format::vindex3::opplan::exec::kv::RowKvState::default()),
        )
        .unwrap();
        for &t in &donor {
            session.step_observed(t, &mut tap).unwrap();
        }
    }
    assert_eq!(tap.captured.captured(), 1);
    let captured = tap.captured.get(L, 0, P).unwrap().to_vec();

    let intervention = tap
        .captured
        .intervention("run-B", (L, 0, P), address(0))
        .unwrap();
    let vector_sha = crate::format::vindex3::opplan::exec::intervene::vector_sha256(&captured);
    assert_eq!(
        intervention.provenance(),
        Some(
            &crate::format::vindex3::opplan::exec::intervene::VectorProvenance::CapturedHead {
                run_id: "run-B".to_string(),
                layer: L,
                head: 0,
                position: P,
                sha256: vector_sha,
            }
        )
    );
    let patched = HeadInterventionPlan::none().with(intervention).unwrap();
    let (_, firings) = intervened(&plan, &store, &backend, &G_TOKENS, &patched);
    assert_eq!(firings.len(), 1);
    assert_eq!(firings[0].kind, HeadInterventionKind::Replace);
}

// ── Declaration identity and unreached addresses ────────────────────

#[test]
fn a_declaration_hashes_stably_and_names_what_the_run_never_reached() {
    use crate::format::vindex3::opplan::exec::intervene_heads::{HeadIntervention, HeadUnreached};

    assert_eq!(HeadInterventionPlan::none().declaration_sha256(), None);
    assert!(HeadInterventionPlan::none().is_none());

    let zero = || {
        HeadInterventionPlan::none()
            .with(HeadIntervention::zero(
                HeadAddress::new(L, 0, [P, 9]).unwrap(),
            ))
            .unwrap()
    };
    let a = zero().declaration_sha256().unwrap();
    let b = zero().declaration_sha256().unwrap();
    assert_eq!(a, b, "the same declaration hashes the same");
    assert_eq!(a.len(), 64);

    let scaled = HeadInterventionPlan::none()
        .with(HeadIntervention::scale(HeadAddress::new(L, 0, [P, 9]).unwrap(), 1.0).unwrap())
        .unwrap();
    assert_ne!(
        scaled.declaration_sha256().unwrap(),
        a,
        "a different kind is a different declaration"
    );

    let plan = zero();
    assert_eq!(
        plan.unreached(G_TOKENS.len()),
        vec![HeadUnreached {
            layer: L,
            head: 0,
            position: 9
        }]
    );
    assert!(plan.unreached(10).is_empty());
    assert_eq!(HeadInterventionKind::Zero.name(), "zero");
    assert_eq!(HeadInterventionKind::Scale.name(), "scale");
    assert_eq!(HeadInterventionKind::Replace.name(), "replace");
}
