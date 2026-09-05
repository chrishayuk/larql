//! Rung 3b: the prepared plan resolves every planned operand's
//! representation, derives the candidates from what the registry declares,
//! pins one realization with its reason, and refuses — before any byte is
//! read — when nothing can serve. The old boolean ladder is kept here as
//! an oracle so the selector's answers are pinned to the pre-rung-3
//! behaviour, class by class, size by size, arm by arm.

use super::super::cpu::physical::{KQuantExecution, PhysicalProjectionPlan};
use super::super::operands::OperandStore;
use super::super::prepared::{ExecutionSlice, PreparedOperands};
use super::super::production::{select_cpu, ProductionBackend};
use super::super::realization::{
    RealizationForm, RealizationId, RepresentationFacts, SelectionReason,
};
use super::super::reference::ReferenceBackend;
use super::bf16_zlib_execution::{transcode, Transcode};
use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::{MatrixClass, WeightFormat};
use crate::format::vindex3::opplan::planned::{Operation, PlannedOperand};
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan, OperandRef};
use crate::format::vindex3::represent::codec::{RepresentationExtent, ResidencyProfile};

/// Below and above the size policy's cache threshold.
const SMALL: usize = 64 * 64;
const LARGE: usize = 17408 * 5120;
const PROJECTION_SUFFIX: &str = "_proj.weight";

struct Prepared {
    _src: tempfile::TempDir,
    _container: tempfile::TempDir,
    plan: ComponentOpPlan,
    store: OperandStore,
}

fn dense(into: Option<Transcode>) -> Prepared {
    let src = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(dense_f32_model, src.path(), container.path(), "dense");
    if let Some(into) = into {
        let transcoded = transcode(
            container.path(),
            |name, shape| shape.len() == 2 && name.ends_with(PROJECTION_SUFFIX),
            into,
        );
        assert!(!transcoded.is_empty());
    }
    let inspection = inspect_container(container.path(), false).unwrap();
    let plan = plan_component_ops(&inspection, container.path(), "target")
        .unwrap()
        .plan
        .unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    Prepared {
        _src: src,
        _container: container,
        plan,
        store,
    }
}

fn is_projection(operation: Operation) -> bool {
    matches!(operation, Operation::Project(_) | Operation::OutputHead)
}

// ── The selector answers what the boolean ladder answered ────────────

/// The pre-rung-3 policy, copied verbatim as a MIGRATION oracle: three
/// stored-dtype booleans and a size policy. The selector must land on the
/// same resident representation for every class, size, label and arm —
/// today. This ladder is transitional, not the new semantic authority:
/// once 3c's independent accounting and 3d's provider inventories exist,
/// new behaviour is allowed to diverge from it deliberately, and this
/// test is then narrowed or retired rather than preserving old mistakes.
fn old_ladder(
    class: MatrixClass,
    elements: usize,
    stored_bf16: bool,
    stored_nvfp4: bool,
    stored_kquant: bool,
    kquant: KQuantExecution,
) -> WeightFormat {
    match class {
        MatrixClass::RoutedExpertBank => WeightFormat::F32,
        _ if stored_nvfp4 => WeightFormat::Nvfp4,
        _ if stored_kquant && kquant == KQuantExecution::Direct => WeightFormat::KQuant,
        MatrixClass::AttentionProjection | MatrixClass::FfnProjection | MatrixClass::OutputHead => {
            PhysicalProjectionPlan::choose_for(Some(class), elements, stored_bf16).format()
        }
    }
}

fn synthetic(operation: Operation, elements: usize) -> PlannedOperand {
    PlannedOperand {
        operand: OperandRef {
            object: "target.decoder_stack".into(),
            tensor: "0.w".into(),
            dtype: String::new(),
            shape: vec![elements, 1],
        },
        operation,
        access: operation.access(),
        extent: RepresentationExtent::TERMINAL,
        layer: Some(0),
        logical_elements: elements,
    }
}

#[test]
fn the_selector_lands_where_the_boolean_ladder_landed_for_every_label_class_size_and_arm() {
    let labels = [
        "BF16",
        "F16",
        "F32",
        "Q4_K",
        "Q6_K",
        "Q8_0",
        "NVFP4",
        "MXFP4",
        "BF16_ZLIB",
    ];
    let mut compared = 0;
    for label in labels {
        let facts = RepresentationFacts::resolve(label);
        assert!(facts.registered.is_some(), "{label} is registered");
        let stored_bf16 = label == "BF16";
        let stored_nvfp4 = label == "NVFP4";
        let stored_kquant = matches!(label, "Q4_K" | "Q6_K" | "Q8_0");
        for (operation, class) in [
            (
                Operation::Project(MatrixClass::AttentionProjection),
                MatrixClass::AttentionProjection,
            ),
            (
                Operation::Project(MatrixClass::FfnProjection),
                MatrixClass::FfnProjection,
            ),
            (Operation::OutputHead, MatrixClass::OutputHead),
        ] {
            for elements in [SMALL, LARGE] {
                for arm in [KQuantExecution::Direct, KQuantExecution::Widen] {
                    let selected = select_cpu(&synthetic(operation, elements), &facts, arm)
                        .unwrap_or_else(|e| panic!("{label} {operation:?} {elements}: {e}"));
                    let expected = old_ladder(
                        class,
                        elements,
                        stored_bf16,
                        stored_nvfp4,
                        stored_kquant,
                        arm,
                    );
                    assert_eq!(
                        selected.realization.format(),
                        expected,
                        "{label} {operation:?} {elements} {arm:?}: {:?}",
                        selected.realization
                    );
                    // The pin is one of the candidates, always.
                    assert!(selected.candidates.contains(&selected.realization));
                    compared += 1;
                }
            }
        }
    }
    assert_eq!(compared, labels.len() * 3 * 2 * 2);
}

/// Direct and decode are DISTINCT realizations of the same operand, told
/// apart by the record and not by the bytes alone.
#[test]
fn direct_and_decode_are_distinct_realizations_of_one_stored_kquant() {
    let facts = RepresentationFacts::resolve("Q6_K");
    let operand = synthetic(Operation::Project(MatrixClass::FfnProjection), LARGE);
    let direct = select_cpu(&operand, &facts, KQuantExecution::Direct).unwrap();
    let decode = select_cpu(&operand, &facts, KQuantExecution::Widen).unwrap();
    assert_eq!(
        direct.realization.form,
        RealizationForm::Direct(PhysicalProjectionPlan::FusedKQuant)
    );
    assert_eq!(
        decode.realization.form,
        RealizationForm::Decode(PhysicalProjectionPlan::BlasF32)
    );
    assert_ne!(
        direct.residency, decode.residency,
        "different declared costs"
    );
    assert_eq!(decode.residency, ResidencyProfile::DECODED_F32);
    assert_eq!(
        direct.candidates, decode.candidates,
        "one candidate set, two orderings"
    );
}

// ── Records on real containers ────────────────────────────────────────

#[test]
fn a_prepared_plan_pins_one_realization_per_planned_operand_with_its_reason() {
    let fixture = dense(None);
    let prepared = PreparedOperands::load(
        &fixture.plan,
        &fixture.store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    let records = prepared.realizations();
    assert_eq!(records.len(), fixture.plan.planned_operands().len());
    for record in records {
        assert_eq!(record.representation, "F32");
        assert!(record
            .selection
            .candidates
            .contains(&record.selection.realization));
        match record.planned.operation {
            Operation::Embed => assert_eq!(
                record.selection.realization.form,
                RealizationForm::DecodedGather
            ),
            op if is_projection(op) => {
                // An f32 source's stored bytes ARE the f32 image: the
                // codec declares BLAS over them as a direct realization.
                assert_eq!(
                    record.selection.realization,
                    RealizationId::cpu(RealizationForm::Direct(PhysicalProjectionPlan::BlasF32))
                );
                assert_eq!(record.selection.reason, SelectionReason::DirectDeclared);
            }
            other => panic!("the dense plan has no {other:?}"),
        }
    }
    prepared.verify_pins().unwrap();
}

#[test]
fn the_reference_backend_pins_the_scalar_oracle_for_every_projection() {
    let fixture = dense(None);
    let prepared = PreparedOperands::load(
        &fixture.plan,
        &fixture.store,
        &ReferenceBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    for record in prepared.realizations() {
        if is_projection(record.planned.operation) {
            assert_eq!(
                record.selection.realization,
                RealizationId::cpu(RealizationForm::Decode(PhysicalProjectionPlan::ScalarF32))
            );
            assert_eq!(record.selection.reason, SelectionReason::ReferenceOracle);
            assert_eq!(
                record.selection.candidates.len(),
                1,
                "the oracle considers nothing else"
            );
        }
    }
}

#[test]
fn the_entropy_coded_container_pins_decode_and_says_why() {
    let fixture = dense(Some(Transcode::Bf16Zlib));
    let prepared = PreparedOperands::load(
        &fixture.plan,
        &fixture.store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    let mut seen = 0;
    for record in prepared.realizations() {
        if record.representation != "BF16_ZLIB" {
            continue;
        }
        assert!(is_projection(record.planned.operation));
        assert_eq!(
            record.selection.realization,
            RealizationId::cpu(RealizationForm::Decode(PhysicalProjectionPlan::BlasF32))
        );
        assert_eq!(
            record.selection.reason,
            SelectionReason::NoDirectRealization
        );
        assert_eq!(
            record.selection.candidates,
            vec![RealizationId::cpu(RealizationForm::Decode(
                PhysicalProjectionPlan::BlasF32
            ))],
            "no direct realization is declared, so decode is the only candidate"
        );
        assert_eq!(record.selection.residency, ResidencyProfile::DECODED_F32);
        seen += 1;
    }
    assert!(seen > 0);
}

// ── Refusals, before any byte ─────────────────────────────────────────

#[test]
fn an_unregistered_representation_is_refused_at_preparation_before_any_byte_is_read() {
    let fixture = dense(Some(Transcode::Unregistered));
    let before = fixture.store.load_count();
    let err = PreparedOperands::load(
        &fixture.plan,
        &fixture.store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .err()
    .map(|e| e.to_string())
    .expect("nothing is registered for the label");
    assert!(err.contains("unregistered representation"), "{err}");
    assert!(err.contains("BF16_ZLIB_UNREGISTERED"), "{err}");
    // Every refused operand is named, not just the first.
    assert!(
        err.matches("unregistered representation").count() > 1,
        "{err}"
    );
    assert_eq!(
        fixture.store.load_count(),
        before,
        "refused before any byte was read"
    );
}

// ── The pin is checked ────────────────────────────────────────────────

#[test]
fn a_tampered_pin_is_refused_by_the_prepared_plan() {
    let fixture = dense(None);
    let mut prepared = PreparedOperands::load(
        &fixture.plan,
        &fixture.store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    prepared.verify_pins().unwrap();
    let index = prepared
        .realizations()
        .iter()
        .position(|r| r.planned.operation == Operation::Project(MatrixClass::AttentionProjection))
        .unwrap();
    prepared.realizations_mut()[index].selection.realization =
        RealizationId::cpu(RealizationForm::Direct(PhysicalProjectionPlan::FusedBf16));
    let err = prepared.verify_pins().unwrap_err().to_string();
    assert!(err.contains("not the pinned realizations"), "{err}");
    assert!(err.contains("Bf16") && err.contains("F32"), "{err}");
}

/// The plan's own view and the loader agree on every fixture this suite
/// prepares; a loader asking for an operand the view does not list is a
/// refusal, not a default.
#[test]
fn every_loaded_matrix_has_a_record_and_no_record_is_left_unloaded() {
    let fixture = dense(None);
    let prepared = PreparedOperands::load(
        &fixture.plan,
        &fixture.store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    let projections = prepared
        .realizations()
        .iter()
        .filter(|r| is_projection(r.planned.operation))
        .count();
    let per_layer = 7;
    assert_eq!(projections, per_layer * fixture.plan.layers.len() + 1);
}

// ── The vocabulary itself ─────────────────────────────────────────────

use super::super::quantise::{Q4_BLOCK, Q8_BLOCK};
use super::super::realization::{
    class_of, common_selection, cpu_projection_candidates, resident_profile, RealizationBackend,
    RefusalKind, SelectionRefusal, SelectionRefusals,
};
use crate::format::vindex3::represent::codec::codecs::float::BF16;
use crate::format::vindex3::represent::codec::{CodecRegistry, RequiredAccess, ResidencyClass};

fn every_form() -> Vec<RealizationForm> {
    vec![
        RealizationForm::Direct(PhysicalProjectionPlan::FusedBf16),
        RealizationForm::Decode(PhysicalProjectionPlan::BlasF32),
        RealizationForm::Requantise(PhysicalProjectionPlan::FusedQ8),
        RealizationForm::SliceStored {
            convert: WeightFormat::F16,
        },
        RealizationForm::DecodedGather,
        RealizationForm::DeviceResident(WeightFormat::Nvfp4),
    ]
}

#[test]
fn every_form_names_its_backend_its_form_and_its_resident_representation() {
    let expected_format = [
        WeightFormat::Bf16,
        WeightFormat::F32,
        WeightFormat::Q8,
        WeightFormat::F16,
        WeightFormat::F32,
        WeightFormat::Nvfp4,
    ];
    let expected_plan = [
        Some(PhysicalProjectionPlan::FusedBf16),
        Some(PhysicalProjectionPlan::BlasF32),
        Some(PhysicalProjectionPlan::FusedQ8),
        None,
        None,
        None,
    ];
    for ((form, format), plan) in every_form()
        .into_iter()
        .zip(expected_format)
        .zip(expected_plan)
    {
        let cpu = RealizationId::cpu(form);
        assert_eq!(cpu.format(), format, "{form:?}");
        assert_eq!(cpu.cpu_plan(), plan, "{form:?}");
        assert!(cpu.name().starts_with("cpu:"), "{}", cpu.name());
        let device = RealizationId {
            backend: RealizationBackend::Device,
            form,
        };
        assert!(device.name().starts_with("device:"), "{}", device.name());
        assert_ne!(cpu, device);
    }
    let names: std::collections::BTreeSet<String> = every_form()
        .into_iter()
        .map(|f| RealizationId::cpu(f).name())
        .collect();
    assert_eq!(
        names.len(),
        every_form().len(),
        "every form renders distinctly"
    );
}

#[test]
fn resident_profiles_are_priced_from_the_executor_s_own_forms() {
    let f32_width = std::mem::size_of::<f32>() as f64;
    let cases = [
        (
            WeightFormat::F32,
            ResidencyClass::TransientDecoded,
            f32_width,
        ),
        (WeightFormat::Bf16, ResidencyClass::Rebound, 2.0),
        (WeightFormat::F16, ResidencyClass::TransientRequantised, 2.0),
        (
            WeightFormat::Q8,
            ResidencyClass::TransientRequantised,
            1.0 + f32_width / Q8_BLOCK as f64,
        ),
        (
            WeightFormat::Q4,
            ResidencyClass::TransientRequantised,
            0.5 + f32_width / Q4_BLOCK as f64,
        ),
        (WeightFormat::Nvfp4, ResidencyClass::Rebound, 4.5 / 8.0),
        (WeightFormat::Mxfp4, ResidencyClass::Stored, 4.25 / 8.0),
        (WeightFormat::KQuant, ResidencyClass::Stored, 8.5 / 8.0),
    ];
    for (format, class, bytes) in cases {
        let profile = resident_profile(format);
        assert_eq!(profile.class, class, "{format:?}");
        assert!(
            (profile.bytes_per_weight - bytes).abs() < 1e-12,
            "{format:?}: {profile:?}"
        );
    }
}

#[test]
fn reasons_and_refusal_kinds_carry_distinct_names() {
    let reasons = [
        SelectionReason::DirectDeclared,
        SelectionReason::NoDirectRealization,
        SelectionReason::ArmPrefersDecode,
        SelectionReason::SizePolicy,
        SelectionReason::BankSlicedAtLoad,
        SelectionReason::DeviceClassTable,
        SelectionReason::EmbeddingGather,
        SelectionReason::ReferenceOracle,
        SelectionReason::OverlaidEdit,
    ];
    let names: std::collections::BTreeSet<&str> = reasons.iter().map(|r| r.name()).collect();
    assert_eq!(names.len(), reasons.len());
    let kinds = [
        RefusalKind::UnregisteredRepresentation,
        RefusalKind::AccessRefused,
        RefusalKind::MissingRealization,
    ];
    let names: std::collections::BTreeSet<&str> = kinds.iter().map(|k| k.name()).collect();
    assert_eq!(names.len(), kinds.len());
}

#[test]
fn a_refusal_renders_the_operand_the_kind_and_every_candidate_it_considered() {
    let operand = synthetic(Operation::ExpertBankSlice, SMALL);
    let slice = RealizationId::cpu(RealizationForm::SliceStored {
        convert: WeightFormat::F32,
    });
    let decode = RealizationId::cpu(RealizationForm::Decode(PhysicalProjectionPlan::BlasF32));
    let refusal = SelectionRefusal {
        operand: operand.operand.clone(),
        operation: operand.operation,
        representation: "BF16_ZLIB".into(),
        requested: RequiredAccess::RowRandom,
        kind: RefusalKind::AccessRefused,
        considered: vec![
            (slice, "provides sequential access".into()),
            (decode, "not offered for a bank".into()),
        ],
    };
    let text = refusal.to_string();
    for expected in [
        "`0.w`",
        "expert-bank-slice",
        "row-random",
        "BF16_ZLIB",
        "access refused",
        &slice.name(),
        "provides sequential access",
        &decode.name(),
        "not offered for a bank",
    ] {
        assert!(text.contains(expected), "{expected} missing from: {text}");
    }
    let bare = SelectionRefusal {
        considered: vec![],
        kind: RefusalKind::MissingRealization,
        ..refusal.clone()
    };
    assert!(bare.to_string().contains("no realization to consider"));
    let all = SelectionRefusals(vec![refusal, bare]).to_string();
    assert!(
        all.starts_with("2 planned operand(s) have no admissible realization"),
        "{all}"
    );
    assert_eq!(
        all.matches("\n  ").count(),
        2,
        "one line per refusal: {all}"
    );
}

#[test]
fn facts_resolve_through_a_registry_and_an_overlay_empties_the_direct_candidates() {
    let scratch = CodecRegistry::new().register(Box::new(BF16)).unwrap();
    let bf16 = RepresentationFacts::resolve_in(&scratch, "BF16");
    assert_eq!(bf16.label, "BF16");
    assert!(bf16.registered.is_some());
    assert_eq!(
        bf16.direct_cpu_plans(),
        vec![
            PhysicalProjectionPlan::FusedBf16,
            PhysicalProjectionPlan::Bf16xQ8
        ]
    );
    assert!(bf16.provides(RequiredAccess::ElementRandom));
    assert_eq!(
        bf16.direct_residency(PhysicalProjectionPlan::FusedBf16),
        Some(ResidencyProfile::stored(16.0))
    );
    assert_eq!(
        bf16.direct_residency(PhysicalProjectionPlan::FusedNvfp4),
        None
    );
    bf16.admit_row_slicing().unwrap();
    // An overlay edit: still registered, still row-addressable in storage,
    // but nothing direct can honour an f32-space edit.
    let edited = bf16.clone().overlaid();
    assert!(edited.overlaid);
    assert!(edited.direct_cpu_plans().is_empty());
    assert!(edited.provides(RequiredAccess::RowRandom));
    // A label the scratch registry does not carry: no decode, no
    // capabilities, and a bank dialect the loader judges itself.
    let alien = RepresentationFacts::resolve_in(&scratch, "NVFP4");
    assert!(alien.registered.is_none());
    assert!(!alien.provides(RequiredAccess::Sequential));
    assert!(alien.direct_cpu_plans().is_empty());
    assert_eq!(
        alien.direct_residency(PhysicalProjectionPlan::FusedNvfp4),
        None
    );
    alien.admit_row_slicing().unwrap();
    // Candidates follow the facts: a bf16 source offers its kernels, the
    // decode, and the executor's re-quantised forms; an unregistered
    // label offers nothing.
    let requantise = [PhysicalProjectionPlan::FusedQ8];
    let candidates = cpu_projection_candidates(&bf16, PhysicalProjectionPlan::BlasF32, &requantise);
    assert_eq!(candidates.len(), 2 + 1 + 1);
    assert!(
        cpu_projection_candidates(&alien, PhysicalProjectionPlan::BlasF32, &requantise).is_empty()
    );
    let f16 = RepresentationFacts::resolve("F16");
    assert_eq!(
        cpu_projection_candidates(&f16, PhysicalProjectionPlan::BlasF32, &requantise),
        vec![RealizationId::cpu(RealizationForm::Decode(
            PhysicalProjectionPlan::BlasF32
        ))]
    );
}

#[test]
fn the_common_selections_cover_the_table_the_bank_and_the_shared_expert() {
    let registered = RepresentationFacts::resolve("BF16");
    let unregistered = RepresentationFacts::resolve("U8");
    let embed = synthetic(Operation::Embed, SMALL);
    let gathered = common_selection(&embed, &registered, WeightFormat::F32)
        .unwrap()
        .unwrap();
    assert_eq!(gathered.realization.form, RealizationForm::DecodedGather);
    assert_eq!(gathered.reason, SelectionReason::EmbeddingGather);
    let refused = common_selection(&embed, &unregistered, WeightFormat::F32)
        .unwrap()
        .unwrap_err();
    assert_eq!(refused.kind, RefusalKind::UnregisteredRepresentation);

    let bank = synthetic(Operation::ExpertBankSlice, SMALL);
    let sliced = common_selection(&bank, &registered, WeightFormat::F16)
        .unwrap()
        .unwrap();
    assert_eq!(
        sliced.realization.form,
        RealizationForm::SliceStored {
            convert: WeightFormat::F16
        }
    );
    assert_eq!(sliced.residency, resident_profile(WeightFormat::F16));
    let sequential = RepresentationFacts::resolve("BF16_ZLIB");
    let refused = common_selection(&bank, &sequential, WeightFormat::F32)
        .unwrap()
        .unwrap_err();
    assert_eq!(refused.kind, RefusalKind::AccessRefused);
    assert_eq!(refused.considered.len(), 1);

    let shared = synthetic(Operation::SharedExpertProject, SMALL);
    let refused = common_selection(&shared, &registered, WeightFormat::F32)
        .unwrap()
        .unwrap_err();
    assert_eq!(refused.kind, RefusalKind::MissingRealization);
    assert!(refused.considered.is_empty());

    for projection in [
        Operation::Project(MatrixClass::AttentionProjection),
        Operation::OutputHead,
    ] {
        assert!(common_selection(
            &synthetic(projection, SMALL),
            &registered,
            WeightFormat::F32
        )
        .is_none());
    }
    assert_eq!(
        class_of(Operation::OutputHead),
        Some(MatrixClass::OutputHead)
    );
    assert_eq!(class_of(Operation::Embed), None);
}

#[test]
fn the_cpu_selector_refuses_an_unregistered_projection_and_a_shared_expert() {
    let unregistered = RepresentationFacts::resolve("U8");
    let refused = select_cpu(
        &synthetic(Operation::Project(MatrixClass::FfnProjection), SMALL),
        &unregistered,
        KQuantExecution::Direct,
    )
    .unwrap_err();
    assert_eq!(refused.kind, RefusalKind::UnregisteredRepresentation);
    let refused = select_cpu(
        &synthetic(Operation::SharedExpertProject, SMALL),
        &RepresentationFacts::resolve("BF16"),
        KQuantExecution::Direct,
    )
    .unwrap_err();
    assert_eq!(refused.kind, RefusalKind::MissingRealization);
    // An overlay edit on a bf16 operand decodes, and the reason says so.
    let edited = RepresentationFacts::resolve("BF16").overlaid();
    let selected = select_cpu(
        &synthetic(Operation::Project(MatrixClass::FfnProjection), LARGE),
        &edited,
        KQuantExecution::Direct,
    )
    .unwrap();
    assert_eq!(
        selected.realization.form,
        RealizationForm::Decode(PhysicalProjectionPlan::BlasF32)
    );
    assert_eq!(selected.reason, SelectionReason::OverlaidEdit);
}

#[test]
fn the_reference_backend_refuses_an_unregistered_representation_before_any_byte() {
    let fixture = dense(Some(Transcode::Unregistered));
    let before = fixture.store.load_count();
    let err = PreparedOperands::load(
        &fixture.plan,
        &fixture.store,
        &ReferenceBackend::new(),
        ExecutionSlice::Full,
    )
    .err()
    .map(|e| e.to_string())
    .expect("the oracle cannot decode what nothing decodes");
    assert!(err.contains("unregistered representation"), "{err}");
    assert_eq!(fixture.store.load_count(), before);
}
