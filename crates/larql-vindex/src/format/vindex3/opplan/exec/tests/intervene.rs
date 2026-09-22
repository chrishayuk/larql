//! V3-INTERVENE-1 acceptance on the golden plan (`docs/v3-intervene-1-carrier-intervention.md`):
//! IP1 the no-op law, IP2 exactness at the address and nowhere before it,
//! IP3 causality, IP4 the provenance round trip, IP5 every declared
//! refusal, and the executor's half of IP6 — the event precedes the write.

use std::collections::BTreeMap;

use super::decode::fixture;
use super::golden::{G_HIDDEN, G_LAYERS, G_TOKENS};
use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::intervene::{
    vector_sha256, Address, CarrierCapture, Firing, Intervention, InterventionKind,
    InterventionPlan, Unreached, VectorProvenance,
};
use crate::format::vindex3::opplan::exec::observe::{
    CarrierForm, CarrierWriteRecord, StepEvent, StepObserver, SublayerSite,
};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::ComponentOpPlan;

type Key = (usize, SublayerSite, usize);

/// The position this file intervenes at: inside the prompt, after the
/// golden window has truncated, with positions on both sides.
const P: usize = 3;

/// Every write's `before`, `delta` and `after`, keyed by address, plus the
/// event stream with the position each event fired at.
#[derive(Default)]
struct Writes {
    position: usize,
    chain: Option<Vec<f32>>,
    before: BTreeMap<Key, Vec<f32>>,
    delta: BTreeMap<Key, Vec<f32>>,
    after: BTreeMap<Key, Vec<f32>>,
    events: Vec<(usize, StepEvent)>,
}

impl StepObserver for Writes {
    fn event(&mut self, event: StepEvent) {
        if let StepEvent::Embedded { position } = event {
            self.position = position;
        }
        self.events.push((self.position, event));
    }

    fn entering_carrier(&mut self, _position: usize, values: &[f32]) {
        self.chain = Some(values.to_vec());
    }

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        let key = (record.layer, record.site, record.position);
        let before = self.chain.take().expect("a write follows a carrier");
        self.before.insert(key, before);
        self.delta.insert(key, record.delta.to_vec());
        self.after.insert(key, record.after.to_vec());
        let mut next = record.after.to_vec();
        if let Some(scale) = record.layer_scale {
            for v in &mut next {
                *v *= scale;
            }
        }
        self.chain = Some(next);
    }
}

fn last_layer() -> usize {
    G_LAYERS - 1
}

fn address(layer: usize, site: SublayerSite, positions: &[usize]) -> Address {
    Address::new(layer, site, positions.iter().copied()).unwrap()
}

fn literal(values: Vec<f32>) -> (Vec<f32>, VectorProvenance) {
    let sha256 = vector_sha256(&values);
    (values, VectorProvenance::Literal { sha256 })
}

/// A vector no carrier will equal by accident.
fn distinct() -> Vec<f32> {
    (0..G_HIDDEN)
        .map(|i| 0.25 * (i as f32 + 1.0) * if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect()
}

fn plain<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
    tokens: &[u32],
) -> (Vec<Vec<f32>>, Writes) {
    let mut session = DecodeSession::new(plan, store, backend).unwrap();
    let mut writes = Writes::default();
    let logits = tokens
        .iter()
        .map(|&t| {
            session
                .step_observed(t, &mut writes)
                .unwrap()
                .logits
                .unwrap()
        })
        .collect();
    (logits, writes)
}

fn intervened<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
    tokens: &[u32],
    interventions: &InterventionPlan,
) -> (Vec<Vec<f32>>, Writes, Vec<Firing>) {
    let mut session = DecodeSession::new(plan, store, backend).unwrap();
    let mut writes = Writes::default();
    let mut firings = Vec::new();
    let logits = tokens
        .iter()
        .map(|&t| {
            let out = session
                .step_intervened(t, &mut writes, interventions)
                .unwrap();
            firings.extend(out.firings);
            out.logits.unwrap()
        })
        .collect();
    (logits, writes, firings)
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

/// Every write strictly before the address in execution order — earlier
/// positions, or the same position at an earlier (layer, site) — is
/// bit-identical between the two runs.
fn assert_untouched_before(plain: &Writes, run: &Writes, at: Key) {
    let (layer, site, position) = at;
    for (key, after) in &plain.after {
        let (l, s, p) = *key;
        let earlier = p < position || (p == position && (l, s) < (layer, site));
        if !earlier {
            continue;
        }
        let other = run
            .after
            .get(key)
            .expect("the intervened run wrote every site");
        assert!(
            bits_equal(after, other),
            "write {key:?} moved before the address"
        );
    }
}

/// Run one closure body against both CPU backends. A macro rather than
/// a function because `DecodeSession` is generic over a sized backend,
/// so each expansion infers its own concrete type.
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

// ── IP1 ─────────────────────────────────────────────────────────────

#[test]
fn ip1_none_and_add_zero_are_the_unobserved_run_bit_for_bit() {
    let (_c, plan, store) = fixture();
    on_both_backends!(|backend| {
        let (baseline, _) = plain(&plan, &store, backend, &G_TOKENS);
        let (none, _, firings) =
            intervened(&plan, &store, backend, &G_TOKENS, &InterventionPlan::none());
        assert_logits_equal(&baseline, &none, "none");
        assert!(firings.is_empty(), "nothing declared, nothing fired");

        let (zeros, provenance) = literal(vec![0.0; G_HIDDEN]);
        let add_zero = InterventionPlan::none()
            .with(
                Intervention::add(
                    address(last_layer(), SublayerSite::Ffn, &[P]),
                    zeros,
                    provenance,
                )
                .unwrap(),
            )
            .unwrap();
        let (added, _, firings) = intervened(&plan, &store, backend, &G_TOKENS, &add_zero);
        assert_logits_equal(&baseline, &added, "add(0)");
        assert_eq!(
            firings,
            vec![Firing {
                layer: last_layer(),
                site: SublayerSite::Ffn,
                position: P,
                kind: InterventionKind::Add,
            }],
            "add(0) fired exactly once, at its address"
        );
    });
}

// ── IP2 ─────────────────────────────────────────────────────────────

#[test]
fn ip2_zero_add_and_replace_are_exact_at_the_address_and_untouched_before_it() {
    let (_c, plan, store) = fixture();
    let at: Key = (last_layer(), SublayerSite::Ffn, P);
    on_both_backends!(|backend| {
        let (_, base) = plain(&plan, &store, backend, &G_TOKENS);
        let base_after = &base.after[&at];
        let v = distinct();

        // Zero: after is zero, delta is exactly −before.
        let zero = InterventionPlan::none()
            .with(Intervention::zero(address(at.0, at.1, &[P])))
            .unwrap();
        let (_, run, _) = intervened(&plan, &store, backend, &G_TOKENS, &zero);
        assert!(run.after[&at]
            .iter()
            .all(|x| x.to_bits() == 0.0f32.to_bits()));
        let before = &run.before[&at];
        let neg: Vec<f32> = before.iter().map(|b| -b).collect();
        assert!(
            bits_equal(&run.delta[&at], &neg),
            "zero's delta is −before, exactly"
        );
        assert!(
            bits_equal(before, &base.before[&at]),
            "the carrier entering the site is the same"
        );
        assert_untouched_before(&base, &run, at);

        // Add: after is the unintervened after plus v, through the backend.
        let (vec, provenance) = literal(v.clone());
        let add = InterventionPlan::none()
            .with(Intervention::add(address(at.0, at.1, &[P]), vec, provenance).unwrap())
            .unwrap();
        let (_, run, _) = intervened(&plan, &store, backend, &G_TOKENS, &add);
        let mut expected = base_after.clone();
        backend.residual_add(&mut expected, &v);
        assert!(
            bits_equal(&run.after[&at], &expected),
            "add's after is after + v"
        );
        assert_untouched_before(&base, &run, at);

        // Replace: after is v.
        let (vec, provenance) = literal(v.clone());
        let replace = InterventionPlan::none()
            .with(Intervention::replace(address(at.0, at.1, &[P]), vec, provenance).unwrap())
            .unwrap();
        let (_, run, _) = intervened(&plan, &store, backend, &G_TOKENS, &replace);
        assert!(bits_equal(&run.after[&at], &v), "replace's after is v");
        assert_untouched_before(&base, &run, at);
        // The reported delta still includes what landed: before + delta
        // reaches v within one rounding of each element.
        let reconstructed: Vec<f32> = run.before[&at]
            .iter()
            .zip(&run.delta[&at])
            .map(|(b, d)| b + d)
            .collect();
        for (r, x) in reconstructed.iter().zip(&v) {
            assert!(
                (r - x).abs() <= f32::EPSILON * x.abs().max(1.0),
                "delta includes the patch"
            );
        }
    });
}

#[test]
fn ip2_the_attention_site_is_addressable_too() {
    let (_c, plan, store) = fixture();
    let at: Key = (0, SublayerSite::Attention, P);
    on_both_backends!(|backend| {
        let (_, base) = plain(&plan, &store, backend, &G_TOKENS);
        let zero = InterventionPlan::none()
            .with(Intervention::zero(address(at.0, at.1, &[P])))
            .unwrap();
        let (_, run, firings) = intervened(&plan, &store, backend, &G_TOKENS, &zero);
        assert!(run.after[&at].iter().all(|x| *x == 0.0));
        assert_untouched_before(&base, &run, at);
        assert_eq!(firings.len(), 1);
        assert_eq!(firings[0].site, SublayerSite::Attention);
    });
}

#[test]
fn a_plan_with_several_positions_fires_at_each() {
    let (_c, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let plan_two = InterventionPlan::none()
        .with(Intervention::zero(address(
            last_layer(),
            SublayerSite::Ffn,
            &[P, P + 1],
        )))
        .unwrap();
    let (_, run, firings) = intervened(&plan, &store, &backend, &G_TOKENS, &plan_two);
    assert_eq!(firings.len(), 2);
    assert_eq!(
        firings.iter().map(|f| f.position).collect::<Vec<_>>(),
        vec![P, P + 1]
    );
    for p in [P, P + 1] {
        assert!(run.after[&(last_layer(), SublayerSite::Ffn, p)]
            .iter()
            .all(|x| *x == 0.0));
    }
    assert!(
        !run.after[&(last_layer(), SublayerSite::Ffn, P - 1)]
            .iter()
            .all(|x| *x == 0.0),
        "a position outside the declared set is untouched"
    );
}

// ── IP3 ─────────────────────────────────────────────────────────────

#[test]
fn ip3_positions_before_the_address_are_untouched_and_the_address_position_moves() {
    let (_c, plan, store) = fixture();
    on_both_backends!(|backend| {
        let (baseline, _) = plain(&plan, &store, backend, &G_TOKENS);
        let (vec, provenance) = literal(distinct());
        let replace = InterventionPlan::none()
            .with(
                Intervention::replace(address(0, SublayerSite::Attention, &[P]), vec, provenance)
                    .unwrap(),
            )
            .unwrap();
        let (logits, _, _) = intervened(&plan, &store, backend, &G_TOKENS, &replace);
        assert_logits_equal(&baseline[..P], &logits[..P], "positions before the address");
        assert!(
            !bits_equal(&baseline[P], &logits[P]),
            "a replace at position {P} must move that position's logits"
        );
    });
}

// ── IP4 ─────────────────────────────────────────────────────────────

#[test]
fn ip4_a_captured_carrier_replaces_bit_for_bit_and_names_its_run() {
    let (_c, plan, store) = fixture();
    let at: Key = (last_layer(), SublayerSite::Ffn, P);
    on_both_backends!(|backend| {
        // Run B: a different prompt, captured at the address.
        let donor: Vec<u32> = G_TOKENS.iter().rev().copied().collect();
        let mut capture = CarrierCapture::at([at]);
        {
            let mut session = DecodeSession::new(&plan, &store, backend).unwrap();
            for &t in &donor {
                session.step_observed(t, &mut capture).unwrap();
            }
        }
        assert_eq!(capture.captured(), 1);
        let captured = capture.get(at.0, at.1, at.2).unwrap().to_vec();

        // Run A: the golden prompt, with B's carrier patched in.
        let intervention = capture
            .intervention(
                "run-B",
                at,
                InterventionKind::Replace,
                address(at.0, at.1, &[P]),
            )
            .unwrap();
        assert_eq!(
            intervention.provenance(),
            Some(&VectorProvenance::Captured {
                run_id: "run-B".to_string(),
                layer: at.0,
                site: at.1,
                position: at.2,
                sha256: vector_sha256(&captured),
            })
        );
        assert_eq!(
            intervention.vector_sha256(),
            Some(vector_sha256(&captured).as_str())
        );
        let patched = InterventionPlan::none().with(intervention).unwrap();
        let (_, run, firings) = intervened(&plan, &store, backend, &G_TOKENS, &patched);
        assert!(
            bits_equal(&run.after[&at], &captured),
            "run A carries run B's carrier bit for bit"
        );
        assert_eq!(firings.len(), 1);
        assert_eq!(firings[0].kind, InterventionKind::Replace);

        // The same carrier as an Add composes through the residual path.
        let added = capture
            .intervention(
                "run-B",
                at,
                InterventionKind::Add,
                address(at.0, at.1, &[P]),
            )
            .unwrap();
        assert_eq!(added.kind(), InterventionKind::Add);
    });
}

// ── IP5 ─────────────────────────────────────────────────────────────

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
fn ip5_construction_refuses_what_it_cannot_represent() {
    let at = || address(0, SublayerSite::Ffn, &[P]);
    refused(Address::new(0, SublayerSite::Ffn, []), "names no position");

    let (vec, provenance) = literal(vec![1.0, f32::NAN, 0.5]);
    refused(
        Intervention::add(at(), vec, provenance),
        "non-finite value at index 1",
    );

    refused(
        Intervention::replace(
            at(),
            Vec::new(),
            VectorProvenance::Literal {
                sha256: String::new(),
            },
        ),
        "empty vector",
    );

    let (vec, _) = literal(distinct());
    refused(
        Intervention::replace(
            at(),
            vec,
            VectorProvenance::Literal {
                sha256: "deadbeef".into(),
            },
        ),
        "not the bytes declared",
    );

    let first = Intervention::zero(address(0, SublayerSite::Ffn, &[P, P + 1]));
    let second = Intervention::zero(address(0, SublayerSite::Ffn, &[P + 1, P + 2]));
    refused(
        InterventionPlan::none().with(first).unwrap().with(second),
        "depend on declaration order",
    );

    let capture = CarrierCapture::at([(0, SublayerSite::Ffn, P)]);
    refused(
        capture.intervention(
            "none",
            (0, SublayerSite::Ffn, P),
            InterventionKind::Replace,
            at(),
        ),
        "no carrier was captured",
    );
    refused(
        capture.intervention(
            "none",
            (0, SublayerSite::Ffn, P),
            InterventionKind::Zero,
            at(),
        ),
        "no carrier was captured",
    );
}

#[test]
fn ip5_admission_refuses_before_the_first_token() {
    let (_c, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let cases: Vec<(InterventionPlan, &str)> = vec![
        (
            InterventionPlan::none()
                .with(Intervention::zero(address(
                    G_LAYERS,
                    SublayerSite::Ffn,
                    &[P],
                )))
                .unwrap(),
            "executes layers 0..",
        ),
        {
            let (vec, provenance) = literal(vec![0.0; G_HIDDEN + 1]);
            (
                InterventionPlan::none()
                    .with(
                        Intervention::add(address(0, SublayerSite::Ffn, &[P]), vec, provenance)
                            .unwrap(),
                    )
                    .unwrap(),
                "carries a vector of width",
            )
        },
    ];
    for (declared, needle) in cases {
        let mut session = DecodeSession::new(&plan, &store, &backend).unwrap();
        let mut sink = Writes::default();
        refused(
            session.step_intervened(G_TOKENS[0], &mut sink, &declared),
            needle,
        );
        assert_eq!(
            session.position(),
            0,
            "the refusal fired before any token executed"
        );
        assert!(sink.events.is_empty());
    }
}

#[test]
fn ip5_an_ffn_site_on_a_layer_without_an_ffn_program_is_refused() {
    let (_c, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let session = DecodeSession::new(&plan, &store, &backend).unwrap();
    drop(session);
    let mut mixer_only = plan.clone();
    mixer_only.layers[0].ffn = None;
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let declared = InterventionPlan::none()
        .with(Intervention::zero(address(0, SublayerSite::Ffn, &[P])))
        .unwrap();
    refused(declared.admit(&mixer_only, &ops), "declares no FFN program");
    declared
        .admit(&plan, &ops)
        .expect("the same address admits on the plan that writes it");
    InterventionPlan::none()
        .admit(&mixer_only, &ops)
        .expect("nothing declared admits everywhere");
}

#[test]
fn ip5_a_bundle_carrier_is_refused_by_declaration() {
    use super::wave19_hc_decode::{layer_range, prepare};
    use super::wave19_hc_substrate::{build, Variant};
    let sub = build(Variant::Headless);
    let ops = prepare(&sub, layer_range()).expect("the hyper-connected substrate prepares");
    let first = ops_first_layer(&ops, &sub.plan);
    let declared = InterventionPlan::none()
        .with(Intervention::zero(address(
            first,
            SublayerSite::Attention,
            &[0],
        )))
        .unwrap();
    refused(declared.admit(&sub.plan, &ops), "`Bundle`");
}

#[test]
fn ip5_a_history_carrier_is_refused_by_declaration() {
    use super::attn_res_substrate::substrate;
    let sub = substrate();
    let store = OperandStore::open(sub.container.path(), &sub.inspection).unwrap();
    let ops = PreparedOperands::load(
        &sub.plan,
        &store,
        &ReferenceBackend::new(),
        ExecutionSlice::Full,
    )
    .expect("the attention-residual substrate prepares");
    let declared = InterventionPlan::none()
        .with(Intervention::zero(address(
            0,
            SublayerSite::Attention,
            &[0],
        )))
        .unwrap();
    refused(declared.admit(&sub.plan, &ops), "`History`");
}

/// The first executed layer of a prepared image, read through the slice
/// the image was prepared with — the topology refusals fire before any
/// layer-range check, so the address only needs to be a real layer.
fn ops_first_layer(_ops: &PreparedOperands, _plan: &ComponentOpPlan) -> usize {
    0
}

// ── IP6, the executor's half ───────────────────────────────────────

#[test]
fn the_intervened_event_precedes_the_write_at_the_address_exactly_once() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let at: Key = (last_layer(), SublayerSite::Ffn, P);
    let (vec, provenance) = literal(distinct());
    let declared = InterventionPlan::none()
        .with(Intervention::add(address(at.0, at.1, &[P]), vec, provenance).unwrap())
        .unwrap();
    let (_, run, _) = intervened(&plan, &store, &backend, &G_TOKENS, &declared);
    let intervened: Vec<usize> = run
        .events
        .iter()
        .enumerate()
        .filter(|(_, (_, e))| matches!(e, StepEvent::Intervened { .. }))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(intervened.len(), 1, "one declared firing, one event");
    let i = intervened[0];
    assert_eq!(
        run.events[i],
        (
            P,
            StepEvent::Intervened {
                layer: at.0,
                site: at.1,
                kind: InterventionKind::Add
            }
        )
    );
    let next_write = run.events[i + 1..]
        .iter()
        .find(|(_, e)| matches!(e, StepEvent::CarrierWrite { .. }))
        .expect("a write closes the site");
    assert_eq!(
        next_write,
        &(
            P,
            StepEvent::CarrierWrite {
                layer: at.0,
                site: at.1,
                carrier: CarrierForm::Single
            }
        ),
        "the event precedes the write it describes, with nothing between"
    );
}

// ── Declaration identity and unreached addresses ───────────────────

#[test]
fn a_declaration_hashes_stably_and_names_what_the_run_never_reached() {
    assert_eq!(InterventionPlan::none().declaration_sha256(), None);
    assert!(InterventionPlan::none().is_none());
    assert_eq!(InterventionPlan::none().declared(), 0);

    let zero = || {
        InterventionPlan::none()
            .with(Intervention::zero(address(1, SublayerSite::Ffn, &[P, 9])))
            .unwrap()
    };
    let a = zero().declaration_sha256().unwrap();
    let b = zero().declaration_sha256().unwrap();
    assert_eq!(a, b, "the same declaration hashes the same");
    assert_eq!(a.len(), 64);

    let (vec, provenance) = literal(distinct());
    let add = InterventionPlan::none()
        .with(Intervention::add(address(1, SublayerSite::Ffn, &[P, 9]), vec, provenance).unwrap())
        .unwrap();
    assert_ne!(
        add.declaration_sha256().unwrap(),
        a,
        "a different kind is a different declaration"
    );
    assert_eq!(add.declared(), 1);
    assert_eq!(add.interventions()[0].kind().name(), "add");
    assert_eq!(InterventionKind::Zero.name(), "zero");
    assert_eq!(InterventionKind::Replace.name(), "replace");

    let plan = zero();
    assert_eq!(
        plan.unreached(G_TOKENS.len()),
        vec![Unreached {
            layer: 1,
            site: SublayerSite::Ffn,
            position: 9
        }]
    );
    assert!(plan.unreached(10).is_empty());
    assert_eq!(plan.unreached(0).len(), 2);
    assert!(plan.interventions()[0].address().covers(9));
    assert!(!plan.interventions()[0].address().covers(4));
    assert_eq!(
        plan.interventions()[0]
            .address()
            .positions()
            .collect::<Vec<_>>(),
        vec![P, 9]
    );
}
