//! **2: what may legitimately be tried, and why everything else may
//! not.**
//!
//! The property that matters is not "the expected candidate survived".
//! It is:
//!
//! > Every enumerated move either survives or carries exactly one
//! > sanctioned mechanical reason for not surviving.
//!
//! so that when an agent later asks *why aren't we exploring E24*, the
//! answer comes from a deterministic partition and not from a language
//! model reconstructing a rationale.

use std::collections::{BTreeMap, BTreeSet};

use super::super::compiler::SourceIdentity;
use super::super::map::{Exception, PrecisionMap};
use super::super::measurement::EvidenceScale;
use super::super::nvfp4_pack::{PackLayout, DTYPE_NVFP4};
use super::super::policy::Role;
use super::super::quality::{Distribution, LogitEvidence, QualityBank, RoutingEvidence};
use super::*;

// ---------------------------------------------------------------- fixtures

fn model() -> SourceIdentity {
    SourceIdentity {
        manifest_hash: "kimi-linear-48b".into(),
        graph_hash: "aligned-vindex3".into(),
        segments: BTreeMap::from([("target.decoder_stack".to_string(), "seg-dddd".to_string())]),
    }
}

/// Four projections at one depth. `no_proj` is 1-D, so no layout can
/// hold it — the structural case is a real property of the surface and
/// not a flag.
fn surface() -> TensorSurface {
    let mut entries: Vec<SurfaceTensor> = ["e_proj", "k_proj", "m_proj"]
        .into_iter()
        .map(|p| {
            SurfaceTensor::new(
                "target.decoder_stack",
                format!("0.self_attn.{p}.weight"),
                Role::DecoderLinear,
                vec![64, 64],
            )
        })
        .collect();
    entries.push(SurfaceTensor::new(
        "target.decoder_stack",
        "0.self_attn.no_proj.weight",
        Role::DecoderLinear,
        vec![64],
    ));
    TensorSurface::new(entries).expect("distinct tensors")
}

/// Compiles nothing by default; every edit is an explicit compile.
fn base_map() -> PrecisionMap {
    PrecisionMap {
        name: "base".into(),
        encoding: DTYPE_NVFP4.into(),
        roles: vec!["decoder-linear".into()],
        exceptions: vec![Exception {
            projection: None,
            layers: None,
            encoding: None,
        }],
    }
}

fn compile(projection: &str) -> Exception {
    Exception {
        projection: Some(projection.into()),
        layers: None,
        encoding: Some(DTYPE_NVFP4.into()),
    }
}

/// `E24`, `K25` and `M26` each compile one projection. `NO` compiles a
/// tensor the NVFP4 layout refuses, so it changes a decision and saves
/// nothing.
fn vocabulary() -> ActionVocabulary {
    ActionVocabulary::new([
        MapEdit::new("E24", compile("e_proj")),
        MapEdit::new("K25", compile("k_proj")),
        MapEdit::new("M26", compile("m_proj")),
        MapEdit::new("NO", compile("no_proj")),
    ])
    .expect("distinct names")
}

/// Prices a state from its resolved decisions and the surface's shapes:
/// a compiled matrix costs what the NVFP4 pack stores, and anything at
/// source precision costs BF16.
struct ShapeFootprint {
    surface: TensorSurface,
}

impl Footprint for ShapeFootprint {
    fn logical_bytes(&self, state: &RepresentationState) -> LogicalBytes {
        let total: u64 = state
            .decisions()
            .decisions()
            .iter()
            .map(|d| {
                let tensor = self
                    .surface
                    .get(&d.object, &d.tensor)
                    .expect("the state resolved against this surface");
                let elements: usize = tensor.shape.iter().product();
                match d.encoding.is_compiled() {
                    true => {
                        PackLayout::derive(&tensor.shape, &tensor.tensor)
                            .expect("a compiled decision means the layout admitted it")
                            .total_len as u64
                    }
                    false => elements as u64 * 2,
                }
            })
            .sum();
        LogicalBytes::new(total)
    }
}

fn footprint() -> ShapeFootprint {
    ShapeFootprint { surface: surface() }
}

fn bank() -> EvidenceBank {
    EvidenceBank::new("kimi-teacher-forced/v1", "17d59a6b", ["seq-000"], 32)
}

fn instrument() -> InstrumentSemantics {
    InstrumentSemantics::new("kl", "distribution", "teacher-forced", "q2a").truncated_to(2048)
}

fn intent(scale: EvidenceScale) -> MeasurementIntent {
    MeasurementIntent::new(bank().id(), scale, instrument().id())
}

fn observation() -> QualityBank {
    QualityBank {
        positions: 8192,
        logits: LogitEvidence {
            kl_p50: 0.0,
            kl_p95: 0.0,
            kl_p99: 1.0e-3,
            max_logit_delta: 0.0,
            top1_flips: 0,
            top10_changes: 0,
        },
        routing: RoutingEvidence {
            route_flips: 0,
            positions_with_route_change: 0,
            layers_with_route_change: 0,
            first_layer_with_route_change: None,
            route_margin: None,
            route_weight_mass_moved: Some(Distribution {
                count: 1,
                min: 0.0,
                p50: 0.0,
                p95: 0.0,
                p99: 0.0,
                max: 0.0,
            }),
        },
        min_covered_mass: Some(0.63),
        top10_margin: None,
        top10_candidate_margin: None,
        top10_mass_displaced: None,
        top10_rank_displacement: None,
        top1_margin: None,
        top1_candidate_margin: None,
        top1_mass_displaced: None,
    }
}

fn applied(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|n| (*n).to_string()).collect()
}

struct Rig {
    vocabulary: ActionVocabulary,
    base: PrecisionMap,
    surface: TensorSurface,
    footprint: ShapeFootprint,
    measurements: MeasurementRegistry,
    model: SourceIdentity,
}

impl Rig {
    fn new() -> Self {
        Self {
            vocabulary: vocabulary(),
            base: base_map(),
            surface: surface(),
            footprint: footprint(),
            measurements: MeasurementRegistry::new(),
            model: model(),
        }
    }

    fn generator(&self) -> Generator<'_> {
        Generator {
            model: &self.model,
            surface: &self.surface,
            base_map: &self.base,
            vocabulary: &self.vocabulary,
            layout: &PackLayoutAdmission,
            footprint: &self.footprint,
            policy: TransitionPolicy::StrictlyImprovingPhysical,
            measurements: &self.measurements,
        }
    }
}

// --------------------------------------------------- the conservation law

#[test]
fn every_enumerated_move_survives_or_carries_one_sanctioned_reason() {
    let rig = Rig::new();
    let g = rig.generator();
    let set = g
        .candidates(&applied(&["E24"]), &intent(EvidenceScale::Diagnostic))
        .expect("generate");

    // Universe from {E24} over a four-edit vocabulary: three additions,
    // plus 1x3 exchanges.
    let census = set.census();
    assert_eq!(census.enumerated, 6);
    assert_eq!(census.enumerated, set.len());
    assert!(census.conserves(), "{census}");
    assert_eq!(
        census.eligible
            + census.already_observed
            + census.physically_dominated
            + census.structurally_invalid,
        6
    );

    // Every disposition names its move, and no move is named twice.
    let labels: BTreeSet<&str> = set
        .dispositions()
        .iter()
        .map(|d| d.action().label.as_str())
        .collect();
    assert_eq!(labels.len(), 6, "each move appears once");
}

#[test]
fn the_census_reports_each_registered_category() {
    // From {E24}: `+K25` and `+M26` save bytes and are eligible; `+NO`
    // compiles a tensor the layout refuses, changing a decision while
    // saving nothing, so the transition policy dominates it; the
    // exchange `−E24 +NO` likewise. `−E24 +K25` and `−E24 +M26` swap one
    // compiled 2-D matrix for another and save nothing either.
    let rig = Rig::new();
    let set = rig
        .generator()
        .candidates(&applied(&["E24"]), &intent(EvidenceScale::Diagnostic))
        .expect("generate");
    let census = set.census();

    assert!(census.conserves(), "{census}");
    assert_eq!(census.eligible, 2, "+K25 and +M26");
    assert_eq!(census.physically_dominated, 4);
    assert_eq!(census.already_observed, 0);

    let eligible: BTreeSet<&str> = set.eligible().map(|c| c.action.label.as_str()).collect();
    assert_eq!(eligible, BTreeSet::from(["+K25", "+M26"]));

    for (action, reason) in set.pruned() {
        assert_eq!(
            reason.category(),
            "physically-dominated",
            "{}",
            action.label
        );
    }
}

#[test]
fn a_move_that_changes_no_decision_is_structurally_invalid() {
    // Re-applying an edit the map already holds resolves identically.
    // There is nothing for an instrument to observe that the parent has
    // not already shown.
    let rig = Rig::new();
    let g = rig.generator();
    let parent = applied(&["E24"]);

    // Build the degenerate move directly: the vocabulary's enumeration
    // never proposes `+E24` from `{E24}`, and the disposition must still
    // be right if something else does.
    let set = g
        .candidates(&parent, &intent(EvidenceScale::Diagnostic))
        .expect("generate");
    assert!(
        !set.dispositions()
            .iter()
            .any(|d| d.action().label == "+E24"),
        "enumeration does not re-propose an applied edit"
    );

    // A vocabulary holding two names for the same exception does produce
    // one: `+E24b` from `{E24}` changes no resolved decision.
    let vocabulary = ActionVocabulary::new([
        MapEdit::new("E24", compile("e_proj")),
        MapEdit::new("E24b", compile("e_proj")),
    ])
    .expect("distinct names");
    let rig2 = Rig {
        vocabulary,
        ..Rig::new()
    };
    let set = rig2
        .generator()
        .candidates(&applied(&["E24"]), &intent(EvidenceScale::Diagnostic))
        .expect("generate");
    // Both moves reachable from `{E24}` over that vocabulary resolve
    // identically to the parent: adding the alias, and swapping the
    // original for it.
    let census = set.census();
    assert!(census.conserves(), "{census}");
    assert_eq!(census.enumerated, 2);
    assert_eq!(census.structurally_invalid, 2);
    let pruned: Vec<&str> = set.pruned().map(|(a, _)| a.label.as_str()).collect();
    assert_eq!(pruned, vec!["+E24b", "−E24 +E24b"]);
    assert!(set
        .pruned()
        .all(|(_, r)| matches!(r, PreMeasurementPrune::StructurallyInvalid { .. })));
}

// ------------------------------------------------- dedup needs the context

#[test]
fn dedup_is_on_the_intended_experiment_and_not_on_the_state() {
    // The distinction 1c proved load-bearing: a diagnostic reading of a
    // state does not make an authority reading of it a repeat.
    let mut rig = Rig::new();
    let child = rig
        .generator()
        .realize(&applied(&["E24", "K25"]))
        .expect("realize");
    rig.measurements
        .record(
            intent(EvidenceScale::Diagnostic).key_for(child.physical_id()),
            observation(),
        )
        .expect("record");

    let diagnostic = rig
        .generator()
        .candidates(&applied(&["E24"]), &intent(EvidenceScale::Diagnostic))
        .expect("generate");
    assert_eq!(diagnostic.census().already_observed, 1);
    let (action, reason) = diagnostic
        .pruned()
        .find(|(_, r)| matches!(r, PreMeasurementPrune::AlreadyObserved { .. }))
        .expect("the diagnostic is a repeat");
    assert_eq!(action.label, "+K25");
    assert!(matches!(
        reason,
        PreMeasurementPrune::AlreadyObserved { .. }
    ));

    // The same move at authority scale is a different experiment, and it
    // arrives eligible — carrying the reading already held, so the
    // escalation is visible rather than silent.
    let authority = rig
        .generator()
        .candidates(&applied(&["E24"]), &intent(EvidenceScale::Authority))
        .expect("generate");
    assert_eq!(authority.census().already_observed, 0);
    let escalation = authority
        .eligible()
        .find(|c| c.action.label == "+K25")
        .expect("eligible at authority");
    assert_eq!(escalation.prior_observations.len(), 1);
    assert_eq!(
        escalation.prior_observations[0].scale(),
        EvidenceScale::Diagnostic
    );
    assert_eq!(escalation.intended_key.scale(), EvidenceScale::Authority);
}

#[test]
fn an_authority_refused_action_is_still_offered_from_another_state() {
    // **Ruling 1, the line that must not move.** `NO` — think E24 — is
    // refused from one state. That refusal attaches to the MEASURED MAP
    // and is not upward-closed under action-set inclusion, so the same
    // action from a different parent is a different, unmeasured state
    // and must still be enumerated. "More low precision" is not a
    // behavioural partial order (R4-F7, R4-F2, R5-F4, R5-F9, R5-F7).
    let mut rig = Rig::new();
    let refused = rig
        .generator()
        .realize(&applied(&["E24", "M26"]))
        .expect("realize");
    rig.measurements
        .record(
            intent(EvidenceScale::Authority).key_for(refused.physical_id()),
            observation(),
        )
        .expect("record");

    // From {K25}, the move `+M26` reaches a DIFFERENT state, and nothing
    // about the earlier refusal may remove it.
    let set = rig
        .generator()
        .candidates(&applied(&["K25"]), &intent(EvidenceScale::Authority))
        .expect("generate");
    assert!(
        set.eligible().any(|c| c.action.label == "+M26"),
        "a refusal elsewhere does not prune this move"
    );
    assert!(set.census().conserves());
}

// ------------------------------------------------------- physics is derived

#[test]
fn an_action_never_asserts_its_own_saving() {
    // R5-F5: a footprint column read as a saving overstated an expert
    // revert 3.39x. No `MapEdit` carries bytes; the delta is computed
    // from two footprints the generator resolved.
    let rig = Rig::new();
    let g = rig.generator();
    let parent = g.realize(&applied(&["E24"])).expect("realize");
    let set = g
        .candidates(&applied(&["E24"]), &intent(EvidenceScale::Diagnostic))
        .expect("generate");

    for candidate in set.eligible() {
        let expected = candidate
            .child
            .logical_bytes()
            .delta_from(parent.logical_bytes());
        assert_eq!(candidate.physical_delta, expected);
        assert!(candidate.physical_delta < 0, "eligible means it saves");
        assert_eq!(candidate.parent_state, *parent.physical_id());
        assert_eq!(candidate.parent_realization, *parent.realization_id());
    }

    // A 64x64 BF16 matrix is 8192 B. NVFP4 stores four 16-element
    // groups per row: 64 x 4 x 8 code bytes, plus 64 x 4 E4M3 group
    // scales, plus one f32 tensor scale — 2308 B. So one compile saves
    // 5884 B, and the generator says so without being told.
    let k25 = set
        .eligible()
        .find(|c| c.action.label == "+K25")
        .expect("+K25");
    assert_eq!(k25.physical_delta, 2308 - 8192);
}

#[test]
fn dominance_is_the_transition_policy_and_not_a_second_opinion() {
    // A generator that offered candidates the graph would refuse, or
    // pruned ones it would have taken, would be two answers to one
    // question.
    let rig = Rig { ..Rig::new() };
    let mut g = rig.generator();
    g.policy = TransitionPolicy::Unconstrained;
    let set = g
        .candidates(&applied(&["E24"]), &intent(EvidenceScale::Diagnostic))
        .expect("generate");

    let census = set.census();
    assert!(census.conserves(), "{census}");
    assert_eq!(
        census.physically_dominated, 0,
        "an unconstrained policy dominates nothing"
    );
    assert_eq!(census.eligible, 6);
    assert!(set.eligible().any(|c| c.physical_delta == 0));
}

// ------------------------------------------------- the vocabulary is input

#[test]
fn enumeration_covers_the_vocabulary_not_the_last_rounds_leftovers() {
    // R5-F6: neighbourhood 1 drew its in-moves from the candidates left
    // unpromoted at iteration 4 and never listed E20/E22/E23/E24/E25.
    // Two moves worth ~430 MB each were invisible. Enumeration is over
    // the declared vocabulary, always.
    let rig = Rig::new();
    let set = rig
        .generator()
        .candidates(&applied(&["E24"]), &intent(EvidenceScale::Diagnostic))
        .expect("generate");

    let proposed: BTreeSet<String> = set
        .dispositions()
        .iter()
        .flat_map(|d| d.action().added.iter().cloned())
        .collect();
    let expected: BTreeSet<String> = rig
        .vocabulary
        .names()
        .filter(|n| *n != "E24")
        .map(String::from)
        .collect();
    assert_eq!(proposed, expected, "every unapplied edit is an in-move");
}

#[test]
fn a_vocabulary_refuses_a_repeated_name_and_a_state_refuses_an_unknown_move() {
    let err = ActionVocabulary::new([
        MapEdit::new("E24", compile("e_proj")),
        MapEdit::new("E24", compile("k_proj")),
    ])
    .expect_err("one name, two edits");
    assert!(format!("{err}").contains("declared twice"), "{err}");

    let rig = Rig::new();
    let err = rig
        .generator()
        .realize(&applied(&["E99"]))
        .expect_err("not in the vocabulary");
    assert!(format!("{err}").contains("not in this vocabulary"), "{err}");
}

#[test]
fn an_applied_set_has_exactly_one_map_however_it_was_reached() {
    // The vocabulary's declaration order supplies the map order, so two
    // rounds that reach the same set reach the same bytes — which is
    // what makes the identity contract meaningful over search states.
    let rig = Rig::new();
    let g = rig.generator();
    let one = g.realize(&applied(&["E24", "M26"])).expect("realize");
    let other = g.realize(&applied(&["M26", "E24"])).expect("realize");
    assert_eq!(one.physical_id(), other.physical_id());
    assert_eq!(one.realization_id(), other.realization_id());
    assert_eq!(one.logical_bytes(), other.logical_bytes());

    let empty = ActionVocabulary::default();
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    assert!(!empty.contains("E24"));
    assert_eq!(rig.vocabulary.edits().len(), 4);
}

#[test]
fn a_candidate_set_reports_what_it_holds() {
    let rig = Rig::new();
    let set = rig
        .generator()
        .candidates(&applied(&["E24"]), &intent(EvidenceScale::Diagnostic))
        .expect("generate");
    assert!(!set.is_empty());
    assert_eq!(set.len(), set.dispositions().len());
    assert_eq!(set.eligible().count() + set.pruned().count(), set.len());

    let census = set.census();
    let rendered = format!("{census}");
    assert!(rendered.contains("generated"), "{rendered}");
    assert!(rendered.contains("eligible"), "{rendered}");
    assert!(Census::default().conserves(), "an empty round conserves");

    // An eligible disposition names the same move as its candidate.
    let first = set
        .dispositions()
        .iter()
        .find(|d| matches!(d, CandidateDisposition::Eligible(_)))
        .expect("one eligible");
    assert!(set
        .eligible()
        .any(|c| c.action.label == first.action().label));
    assert!(set.eligible().all(|c| c.applied.len() == 2));
}
