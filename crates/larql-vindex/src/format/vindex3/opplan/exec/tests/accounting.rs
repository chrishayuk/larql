//! Rung 3c: declarations bound AGAINST observations. The expectation side
//! is priced from the pin, the codec's declared residency, the executor's
//! block geometry and the container's recorded lengths; the observation
//! side is read off the objects the loader bound. Neither reads the other.

use super::super::accounting::{
    execution_touch, expectations, ledger_correspondence, reconcile, render_selection_summary,
    resident_profile_with, stored_footprint, BlockGeometry, Bound, Observed,
};
use super::super::backend::{MatrixClass, WeightFormat};
use super::super::cpu::ledger::{ledger, thread_projection_calls};
use super::super::cpu::physical::PhysicalProjectionPlan;
use super::super::operands::OperandStore;
use super::super::prepared::{ExecutionSlice, PreparedOperands};
use super::super::production::ProductionBackend;
use super::super::realization::{
    RealizationForm, RealizationId, RealizationRecord, Selection, SelectionReason,
};
use super::super::weights::{load_weight, DEVICE_PAGE_ALIGN};
use super::super::{execute_prepared_streaming, PlaneEvent};
use super::bf16_zlib_execution::{transcode, Transcode};
use crate::format::vindex3::fixtures::{
    dense_f32_model, dense_f32_model_with, encode_fixture_container, HeadStorage,
};
use crate::format::vindex3::inspect::{inspect_container, SystemInspection};
use crate::format::vindex3::opplan::planned::{Operation, PlannedOperand};
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan, OperandRef};
use crate::format::vindex3::represent::codec::codecs::{float, kquant, mxfp4, nvfp4};
use crate::format::vindex3::represent::codec::{
    CodecRegistry, RepresentationExtent, ResidencyProfile,
};

const F32_WIDTH: u64 = std::mem::size_of::<f32>() as u64;
const BF16_WIDTH: u64 = std::mem::size_of::<u16>() as u64;
const PROJECTION_SUFFIX: &str = "_proj.weight";
const TOKENS: [u32; 3] = [3, 17, 28];

struct Fixture {
    _src: tempfile::TempDir,
    container: tempfile::TempDir,
    inspection: SystemInspection,
    plan: ComponentOpPlan,
    store: OperandStore,
}

impl Fixture {
    /// The same container, decoding through `registry`.
    fn store_through(&self, registry: &'static CodecRegistry) -> OperandStore {
        OperandStore::open(self.container.path(), &self.inspection)
            .unwrap()
            .with_registry(registry)
    }
}

fn fixture(write: fn(&std::path::Path), into: Option<Transcode>) -> Fixture {
    let src = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(write, src.path(), container.path(), "dense");
    if let Some(into) = into {
        let done = transcode(
            container.path(),
            |name, shape| shape.len() == 2 && name.ends_with(PROJECTION_SUFFIX),
            into,
        );
        assert!(!done.is_empty());
    }
    let inspection = inspect_container(container.path(), false).unwrap();
    let plan = plan_component_ops(&inspection, container.path(), "target")
        .unwrap()
        .plan
        .unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    Fixture {
        _src: src,
        container,
        inspection,
        plan,
        store,
    }
}

fn tied_head() -> Fixture {
    fixture(|d| dense_f32_model_with(d, HeadStorage::Tied), None)
}

fn prepared(f: &Fixture) -> PreparedOperands {
    PreparedOperands::load(
        &f.plan,
        &f.store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap()
}

/// A record for one operand under one realization, as the selector would
/// have pinned it, with the residency a codec would have declared.
fn record(
    operand: &OperandRef,
    realization: RealizationId,
    residency: ResidencyProfile,
) -> RealizationRecord {
    let operation = Operation::Project(MatrixClass::FfnProjection);
    RealizationRecord {
        planned: PlannedOperand {
            operand: operand.clone(),
            operation,
            access: operation.access(),
            extent: RepresentationExtent::TERMINAL,
            layer: Some(0),
            declared_representation: None,
            logical_elements: operand.shape.iter().product(),
        },
        representation: operand.dtype.clone(),
        provider: None,
        selection: Selection {
            realization,
            residency,
            reason: SelectionReason::SizePolicy,
            candidates: vec![realization],
        },
    }
}

/// The first FFN projection of the dense plan.
fn an_ffn_projection(plan: &ComponentOpPlan) -> OperandRef {
    plan.planned_operands()
        .into_iter()
        .find(|p| p.operation == Operation::Project(MatrixClass::FfnProjection))
        .map(|p| p.operand)
        .unwrap()
}

// ── Every resident form the CPU loader produces, against its declaration ──

/// For each resident form: load the object, price the pin, reconcile.
/// The pricing that comes from the executor's block geometry is then
/// re-priced under a MUTATED geometry, and the reconciliation must break —
/// otherwise the comparison would be reading the declaration twice.
#[test]
fn every_resident_form_reconciles_with_its_declaration_and_a_mutated_geometry_breaks_it() {
    let f = fixture(dense_f32_model, None);
    let op = an_ffn_projection(&f.plan);
    let stored = |o: &OperandRef| f.store.stored_len(o);
    let executor = BlockGeometry::executor();
    let mutated = BlockGeometry {
        q8_block: executor.q8_block / 2,
        q4_block: executor.q4_block / 2,
        q8_indexed: !executor.q8_indexed,
    };
    let cases: [(WeightFormat, RealizationForm, bool); 6] = [
        (
            WeightFormat::F32,
            RealizationForm::Decode(PhysicalProjectionPlan::BlasF32),
            false,
        ),
        (
            WeightFormat::Q8,
            RealizationForm::Requantise(PhysicalProjectionPlan::FusedQ8),
            true,
        ),
        (
            WeightFormat::Q4,
            RealizationForm::Requantise(PhysicalProjectionPlan::FusedQ4),
            true,
        ),
        (
            WeightFormat::F16,
            RealizationForm::DeviceResident(WeightFormat::F16),
            false,
        ),
        (
            WeightFormat::Nvfp4,
            RealizationForm::DeviceResident(WeightFormat::Nvfp4),
            false,
        ),
        (
            WeightFormat::Mxfp4,
            RealizationForm::DeviceResident(WeightFormat::Mxfp4),
            false,
        ),
    ];
    for (format, form, geometry_priced) in cases {
        let loaded = load_weight((&f.store).into(), &op, format)
            .unwrap_or_else(|e| panic!("{format:?}: {e}"));
        let rec = record(&op, RealizationId::cpu(form), ResidencyProfile::DECODED_F32);
        let observed = vec![Bound::one(&op, &loaded)
            .observed(rec.planned.operation, rec.planned.layer)
            .unwrap()];
        let expected = expectations(std::slice::from_ref(&rec), stored, executor);
        let ok = reconcile(&expected, &observed).unwrap_or_else(|e| panic!("{format:?}: {e}"));
        assert_eq!(ok.matched, 1);
        assert!(
            ok.padding < loaded.padded_allocations() as u64 * DEVICE_PAGE_ALIGN as u64 + 1,
            "{format:?}: padding {} over {} padded allocation(s)",
            ok.padding,
            loaded.padded_allocations()
        );
        if loaded.padded_allocations() == 0 {
            assert_eq!(ok.padding, 0, "{format:?}: an exact form has no padding");
        }
        let under_mutation = expectations(std::slice::from_ref(&rec), stored, mutated);
        let broken = reconcile(&under_mutation, &observed).is_err();
        assert_eq!(
            broken, geometry_priced,
            "{format:?}: a mutated block geometry must break exactly the forms it prices"
        );
    }
}

#[test]
fn the_q8_pricing_follows_the_executor_s_index_flag_and_block() {
    let plain = BlockGeometry {
        q8_block: 64,
        q4_block: 64,
        q8_indexed: false,
    };
    let indexed = BlockGeometry {
        q8_indexed: true,
        ..plain
    };
    let p = resident_profile_with(WeightFormat::Q8, plain).bytes_per_weight;
    let i = resident_profile_with(WeightFormat::Q8, indexed).bytes_per_weight;
    assert!((p - (1.0 + 4.0 / 64.0)).abs() < 1e-12);
    assert!(
        (i - (1.0 + 6.0 / 64.0)).abs() < 1e-12,
        "one i16 sum per block"
    );
    let q4 = resident_profile_with(WeightFormat::Q4, plain).bytes_per_weight;
    assert!((q4 - (0.5 + 4.0 / 64.0)).abs() < 1e-12);
}

// ── A prepared plan reconciles, and the census is a third reading ─────

#[test]
fn a_prepared_dense_plan_reconciles_every_pin_and_the_census_is_the_same_bytes() {
    let f = fixture(dense_f32_model, None);
    let ops = prepared(&f);
    let done = ops.reconcile(&f.plan, (&f.store).into()).unwrap();
    assert_eq!(done.matched, ops.realizations().len());
    assert_eq!(done.padding, 0, "f32 images are exact");
    // The census walks the loaded objects by site; the observation walks
    // them by operand. Two readings of one resident image must agree.
    let observed = ops.bound(&f.plan).unwrap();
    let total: u64 = observed.iter().map(|o| o.resident_bytes).sum();
    let census = ops.residency_census();
    assert_eq!(total, (census.total() - census.glue.total()) as u64);
    assert_eq!(done.observed_resident, total);
}

#[test]
fn the_entropy_coded_container_s_stored_bytes_are_the_recorded_length_and_its_staging_is_the_image()
{
    let f = fixture(dense_f32_model, Some(Transcode::Bf16Zlib));
    let ops = prepared(&f);
    let expected = ops.expectations((&f.store).into(), BlockGeometry::executor());
    let mut seen = 0;
    for e in expected
        .iter()
        .filter(|e| f.store.stored_dtype(&e.operand) == Some("BF16_ZLIB"))
    {
        let recorded = f.store.stored_len(&e.operand).unwrap();
        assert_eq!(e.stored_bytes, recorded, "{}", e.operand.tensor);
        assert_ne!(
            e.stored_bytes,
            e.logical_elements as u64 * BF16_WIDTH,
            "instance, not shape"
        );
        assert_eq!(
            e.staging,
            e.logical_elements as u64 * F32_WIDTH,
            "decoded through f32"
        );
        assert_eq!(e.declared_resident, e.logical_elements as u64 * F32_WIDTH);
        assert_eq!(e.working_set(), e.staging + e.declared_resident);
        seen += 1;
    }
    assert!(seen > 0);
    ops.reconcile(&f.plan, (&f.store).into()).unwrap();
    // Footprint and touch: every operand is read once per operation, and
    // no operand is shared, so the two agree on this plan.
    assert_eq!(
        stored_footprint(&expected).bytes,
        execution_touch(&expected)
    );
}

#[test]
fn a_tied_head_counts_once_in_the_stored_footprint_and_once_per_operation_in_the_touch() {
    let f = tied_head();
    let ops = prepared(&f);
    let expected = ops.expectations((&f.store).into(), BlockGeometry::executor());
    let table = f.plan.embedding.as_ref().unwrap().table.clone();
    let table_bytes = f.store.stored_len(&table).unwrap();
    let footprint = stored_footprint(&expected);
    let touch = execution_touch(&expected);
    let operations = expected.len();
    assert_eq!(
        footprint.operands,
        operations - 1,
        "the table is one stored operand under two operations"
    );
    assert_eq!(
        touch,
        footprint.bytes + table_bytes,
        "the table is read once per operation"
    );
    ops.reconcile(&f.plan, (&f.store).into()).unwrap();
}

// ── Providers: gone or changed invalidates the image ─────────────────

fn rung_one_registry(with_f32: bool) -> CodecRegistry {
    let mut r = CodecRegistry::new()
        .register(Box::new(float::BF16))
        .and_then(|r| r.register(Box::new(float::F16)))
        .unwrap();
    if with_f32 {
        r = r.register(Box::new(float::F32)).unwrap();
    }
    r.register(Box::new(kquant::Q4_K))
        .and_then(|r| r.register(Box::new(kquant::Q6_K)))
        .and_then(|r| r.register(Box::new(kquant::Q8_0)))
        .and_then(|r| r.register(Box::new(nvfp4::NVFP4)))
        .and_then(|r| r.register(Box::new(mxfp4::MXFP4)))
        .unwrap()
}

#[test]
fn a_provider_that_disappears_invalidates_the_preparation_rather_than_falling_back() {
    let f = fixture(dense_f32_model, None);
    // The store carries the registry it decodes through, so selection,
    // provider identity and decode all read one registry.
    let with: &'static CodecRegistry = Box::leak(Box::new(rung_one_registry(true)));
    let ops = PreparedOperands::load(
        &f.plan,
        &f.store_through(with),
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    ops.ensure_providers_in(with).unwrap();
    ops.ensure_providers_in(CodecRegistry::builtin())
        .expect("the built-in registry offers the same F32 identity");
    let without: &'static CodecRegistry = Box::leak(Box::new(rung_one_registry(false)));
    let err = ops.ensure_providers_in(without).unwrap_err().to_string();
    assert!(
        err.contains("`F32`") && err.contains("no registered codec"),
        "{err}"
    );
    assert!(err.contains("re-prepare"), "{err}");
    // And preparing through a store whose registry lacks F32 refuses
    // outright.
    let Err(err) = PreparedOperands::load(
        &f.plan,
        &f.store_through(without),
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    ) else {
        panic!("F32 is not registered there");
    };
    let err = err.to_string();
    assert!(err.contains("unregistered representation"), "{err}");
}

// ── The four-way correspondence ───────────────────────────────────────

/// Planned operands, pins, bound objects and the ledger's account of what
/// ran describe one execution — EXACTLY, in positions. The projection
/// ledger is process-global, so the exact reading is only available in a
/// process that executes nothing else: this test re-launches the test
/// binary on itself, marked, and the marked run makes the assertions.
#[test]
fn planned_operands_pins_bound_objects_and_the_ledger_correspond() {
    const WITNESS: &str = "LARQL_LEDGER_WITNESS";
    const NAME: &str = "format::vindex3::opplan::exec::tests::accounting::planned_operands_pins_bound_objects_and_the_ledger_correspond";
    if std::env::var_os(WITNESS).is_some() {
        return exact_ledger_correspondence();
    }
    let exe = std::env::current_exe().expect("the test binary knows its own path");
    let output = std::process::Command::new(exe)
        .args(["--exact", NAME, "--test-threads=1", "--nocapture"])
        .env(WITNESS, "1")
        .output()
        .expect("the test binary re-launches");
    assert!(
        output.status.success(),
        "the exact correspondence did not hold in an isolated process:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn exact_ledger_correspondence() {
    let f = fixture(dense_f32_model, None);
    let ops = prepared(&f);
    let planned = f.plan.planned_operands();
    let pins = ops.realizations();
    let bound = ops.bound(&f.plan).unwrap();
    // One pin per planned operand, one bound object per pin.
    assert_eq!(pins.len(), planned.len());
    assert_eq!(bound.len(), pins.len());
    let projections = pins
        .iter()
        .filter(|r| r.selection.realization.cpu_plan().is_some())
        .count();

    ledger().reset();
    let backend = ProductionBackend::new();
    let mut sink = |_: PlaneEvent| Ok(());
    execute_prepared_streaming(&f.plan, &ops, &TOKENS, &backend, None, &mut sink).unwrap();
    let account = ledger_correspondence(pins, ledger()).unwrap();
    assert_eq!(
        account.len(),
        1,
        "one CPU plan on an f32 fixture: {account:?}"
    );
    let (plan, pinned, tally) = account[0];
    assert_eq!(plan, PhysicalProjectionPlan::BlasF32);
    assert_eq!(pinned, projections);
    eprintln!(
        "BlasF32 over {projections} pinned projections: calls {} slabs {} positions {} (this thread: {})",
        tally.calls,
        tally.slabs,
        tally.positions,
        thread_projection_calls()
    );
    // Positions are the unit that corresponds: every layer projection
    // processed every token, and the head processed the last one. Calls
    // are batched differently per site and threaded, so they are not.
    let in_layers = pins
        .iter()
        .filter(|r| r.planned.layer.is_some() && r.selection.realization.cpu_plan().is_some())
        .count() as u64;
    let heads = projections as u64 - in_layers;
    assert_eq!(
        tally.positions,
        in_layers * TOKENS.len() as u64 + heads,
        "{tally:?}"
    );
}

// ── Presentation over the records ────────────────────────────────────

#[test]
fn the_summary_renders_the_structured_records_and_adds_nothing() {
    let f = fixture(dense_f32_model, Some(Transcode::Bf16Zlib));
    let ops = prepared(&f);
    let text = render_selection_summary(ops.realizations());
    assert!(text.starts_with("realizations:\n"), "{text}");
    for record in ops.realizations() {
        assert!(text.contains(&record.representation), "{text}");
        assert!(
            text.contains(&record.selection.realization.name()),
            "{text}"
        );
        assert!(text.contains(record.selection.reason.name()), "{text}");
    }
    let lines = text.lines().count() - 1;
    let distinct: std::collections::BTreeSet<_> = ops
        .realizations()
        .iter()
        .map(|r| {
            (
                r.representation.clone(),
                r.selection.realization.name(),
                r.selection.reason.name(),
            )
        })
        .collect();
    assert_eq!(
        lines,
        distinct.len(),
        "one line per (representation, realization, reason)"
    );
}

/// The observation side refuses a pairing that contradicts itself,
/// and the reconciliation refuses a stray object and a missing one.
#[test]
fn reconciliation_refuses_strays_omissions_and_disagreeing_forms() {
    let f = fixture(dense_f32_model, None);
    let op = an_ffn_projection(&f.plan);
    let f32 = load_weight((&f.store).into(), &op, WeightFormat::F32).unwrap();
    // A different resident form for the same operand: the executor's own
    // re-quantisation, which any f32 source can take.
    let q8 = load_weight((&f.store).into(), &op, WeightFormat::Q8).unwrap();
    let rec = record(
        &op,
        RealizationId::cpu(RealizationForm::Decode(PhysicalProjectionPlan::BlasF32)),
        ResidencyProfile::DECODED_F32,
    );
    let expected = expectations(
        std::slice::from_ref(&rec),
        |o| f.store.stored_len(o),
        BlockGeometry::executor(),
    );
    let observed = |w| {
        vec![Bound::one(&op, w)
            .observed(rec.planned.operation, Some(0))
            .unwrap()]
    };
    reconcile(&expected, &observed(&f32)).unwrap();
    let err = reconcile(&expected, &observed(&q8))
        .unwrap_err()
        .to_string();
    assert!(err.contains("pinned F32 but Q8 is resident"), "{err}");
    let err = reconcile(&expected, &[]).unwrap_err().to_string();
    assert!(err.contains("nothing is resident for it"), "{err}");
    let err = reconcile(&[], &observed(&f32)).unwrap_err().to_string();
    assert!(err.contains("nothing was pinned for it"), "{err}");
    let mixed = Bound {
        operand: &op,
        weights: vec![&f32, &q8],
    }
    .observed(rec.planned.operation, Some(0))
    .unwrap_err()
    .to_string();
    assert!(
        mixed.contains("disagree on their representation"),
        "{mixed}"
    );
    let _: Observed = observed(&f32).remove(0);
}

// ── The prepared plan's pairing and provider arms ────────────────────

use super::gemma4::{closure as gemma4_closure, encoded as gemma4_encoded, miniature_gemma4};
use crate::format::vindex3::represent::codec::{
    AccessGranularity, CodecCapabilities, CodecError, CodecOperands, ExtentCertificate,
    RepresentationCodec, StreamSpec,
};
use crate::format::vindex3::represent::nvfp4_pack::CodecIdentity;

/// A codec that claims someone else's label under its own identity —
/// what a provider that CHANGED looks like to a prepared image.
pub(super) struct ProviderStub {
    pub label: &'static str,
}

impl RepresentationCodec for ProviderStub {
    fn encoding_label(&self) -> &'static str {
        self.label
    }
    fn identity(&self) -> CodecIdentity {
        CodecIdentity {
            family: format!("stub-{}", self.label),
            revision: 7,
            group_elems: 1,
            element: "stub".into(),
            group_scale: "none".into(),
            tensor_scale: "none".into(),
            layout: "stub".into(),
        }
    }
    fn streams(&self) -> &'static [StreamSpec] {
        &[crate::format::vindex3::represent::codec::streams::VALUES]
    }
    fn capabilities(&self) -> CodecCapabilities {
        CodecCapabilities {
            access: AccessGranularity::ElementRandom,
            group_elems: 1,
            row_align_elems: 1,
            physical_align_bytes: 1,
        }
    }
    fn extents(&self) -> Vec<ExtentCertificate> {
        vec![ExtentCertificate::terminal(32.0)]
    }
    fn stored_bytes(
        &self,
        _: &[usize],
        _: RepresentationExtent,
        _: &str,
    ) -> Result<u64, CodecError> {
        Ok(0)
    }
    fn validate(
        &self,
        _: &CodecOperands<'_>,
        _: &[usize],
        _: RepresentationExtent,
        _: &str,
    ) -> Result<(), CodecError> {
        Ok(())
    }
    fn decode_rows(
        &self,
        _: &CodecOperands<'_>,
        _: &[usize],
        _: std::ops::Range<usize>,
        _: RepresentationExtent,
        dst: &mut [f32],
        _: &str,
    ) -> Result<(), CodecError> {
        dst.fill(0.0);
        Ok(())
    }
    fn decode_residency(&self) -> ResidencyProfile {
        ResidencyProfile::DECODED_F32
    }
}

#[test]
fn a_hybrid_plan_pairs_dense_projections_one_each_and_banks_over_their_experts() {
    let dir = tempfile::tempdir().unwrap();
    miniature_gemma4(dir.path(), None);
    let container = gemma4_encoded(dir.path());
    let outcome = gemma4_closure(container.path());
    let plan = outcome.plan.expect("the Gemma-4 miniature plans");
    // Layer 3 binds one tensor as both its key and value projection: one
    // stored operand, two operation instances in one layer. The view lists
    // both, the loader binds both, the pairing matches both, and the
    // stored footprint counts the object once.
    let mut counts = std::collections::BTreeMap::new();
    for p in plan.planned_operands() {
        *counts
            .entry((p.operand.tensor.clone(), p.layer))
            .or_insert(0usize) += 1;
    }
    // Two shared objects: the aliased projection within layer 3, and the
    // head tied to the embedding table (two operations, no layer).
    let repeated: Vec<_> = counts.iter().filter(|(_, n)| **n > 1).collect();
    assert_eq!(repeated.len(), 2, "{repeated:?}");
    assert!(repeated.iter().all(|(_, n)| **n == 2), "{repeated:?}");
    assert!(repeated
        .iter()
        .any(|((t, l), _)| t.ends_with("k_proj.weight") && l.is_some()));
    assert!(
        repeated.iter().any(|((_, l), _)| l.is_none()),
        "the tied head"
    );
    let inspection = inspect_container(container.path(), false).unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let ops = PreparedOperands::load(
        &plan,
        &store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    let observed = ops.bound(&plan).unwrap();
    let hybrid_layers = plan
        .layers
        .iter()
        .filter(|l| {
            matches!(
                l.ffn,
                Some(crate::format::vindex3::opplan::LayerFfn::Hybrid(_))
            )
        })
        .count();
    assert!(hybrid_layers > 0, "the miniature has hybrid layers");
    let dense: Vec<_> = observed
        .iter()
        .filter(|o| o.operation == Operation::Project(MatrixClass::FfnProjection))
        .collect();
    let banks: Vec<_> = observed
        .iter()
        .filter(|o| o.operation == Operation::ExpertBankSlice)
        .collect();
    assert!(
        dense.len() >= 2 * hybrid_layers,
        "gate/up and down per hybrid dense branch"
    );
    assert_eq!(banks.len(), 2 * hybrid_layers, "two banks per hybrid layer");
    assert!(banks
        .iter()
        .all(|b| b.allocations == 0 && b.format == WeightFormat::F32));
    let done = ops.reconcile(&plan, (&store).into()).unwrap();
    assert_eq!(done.matched, ops.realizations().len());
    let expected = ops.expectations((&store).into(), BlockGeometry::executor());
    assert_eq!(
        stored_footprint(&expected).operands,
        expected.len() - 2,
        "the aliased projection and the tied table are one stored operand each under two instances"
    );
    assert!(execution_touch(&expected) > stored_footprint(&expected).bytes);
}

#[test]
fn verify_pins_refuses_a_tampered_ffn_pin_and_a_tampered_head_pin() {
    let f = fixture(dense_f32_model, None);
    for (operation, site) in [
        (Operation::Project(MatrixClass::FfnProjection), "ffn"),
        (Operation::OutputHead, "output head"),
    ] {
        let mut ops = prepared(&f);
        let index = ops
            .realizations()
            .iter()
            .position(|r| r.planned.operation == operation)
            .unwrap();
        ops.realizations_mut()[index].selection.realization =
            RealizationId::cpu(RealizationForm::Requantise(PhysicalProjectionPlan::FusedQ8));
        let err = ops.verify_pins().unwrap_err().to_string();
        assert!(
            err.contains(site) && err.contains("not the pinned realizations"),
            "{err}"
        );
    }
}

#[test]
fn a_provider_whose_identity_changed_invalidates_the_preparation() {
    let f = fixture(dense_f32_model, None);
    let ops = prepared(&f);
    let changed = CodecRegistry::new()
        .register(Box::new(ProviderStub { label: "F32" }))
        .unwrap();
    let err = ops.ensure_providers_in(&changed).unwrap_err().to_string();
    assert!(
        err.contains("`F32`") && err.contains("F32 r1") && err.contains("stub-F32 r7"),
        "{err}"
    );
    let providers = ops.providers();
    assert_eq!(
        providers.len(),
        1,
        "one stored label on the dense fixture: {providers:?}"
    );
    assert_eq!(providers[0].0, "F32");
    assert!(providers[0].1.is_some());
}

#[test]
fn bound_refuses_a_plan_that_is_a_different_program_from_the_prepared_one() {
    let dense = fixture(dense_f32_model, None);
    let ops = prepared(&dense);
    let mut headless = dense.plan.clone();
    headless.layers.clear();
    let err = ops.bound(&headless).unwrap_err().to_string();
    assert!(err.contains("prepared but not in the plan"), "{err}");
    let lllf = fixture(
        crate::format::vindex3::fixtures::hybrid_lllf_f32_model,
        None,
    );
    let err = ops.bound(&lllf.plan).unwrap_err().to_string();
    assert!(err.contains("different programs"), "{err}");
}
