//! K3-RESIDENCY-VERTICAL-1, V2: the resources a plan demands are seven
//! numbers aggregated by seven rules, a residency budget constrains the
//! PHYSICAL working set and the per-token touch — never a mapping's
//! address space — and selection under it CHOOSES among each operand's
//! own candidates, or refuses before any payload byte with the
//! irreducible deficit and the alternatives it tried.
//!
//! Three arms, on the bytes-backed per-expert miniature: unbounded (the
//! backend's own preference, the mapped bank selected), constrained (at
//! least one realization changes and the plan then fits), impossible
//! (refused by name with the deficit).

use std::path::Path;

use super::super::accounting::{
    expectations, BlockGeometry, ResidencyBudget, ResourceLedger, ThroughputBudget,
};
use super::super::operands::OperandStore;
use super::super::prepared::{select_realizations_within, ExecutionSlice, PreparedOperands};
use super::super::production::ProductionBackend;
use super::super::realization::{RealizationForm, SelectionReason};
use crate::format::vindex3::fixtures::encode_fixture_container;
use crate::format::vindex3::fixtures_kimi::{
    kimi_per_expert_moe_f32_model, MOE_EXPERTS, MOE_TOP_K,
};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::planned::Operation;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

struct Subject {
    _src: tempfile::TempDir,
    container: tempfile::TempDir,
}

impl Subject {
    /// The miniature with every two-dimensional weight stored as bf16 —
    /// the form the real container stores, and the one under which a
    /// projection has cheaper-resident candidates (the fused bf16
    /// kernel, the executor's Q8 and Q4 re-quantisations) for a budget to
    /// choose among. An f32 source offers only BLAS and the scalar
    /// transcription, which cost the same, so nothing could be chosen.
    fn build(write: fn(&Path)) -> Self {
        use super::bf16_zlib_execution::{transcode, Transcode};
        let src = tempfile::tempdir().unwrap();
        let container = tempfile::tempdir().unwrap();
        encode_fixture_container(write, src.path(), container.path(), "kimi-moe");
        let done = transcode(
            container.path(),
            |name, shape| shape.len() == 2 && name.ends_with(".weight"),
            Transcode::Bf16,
        );
        assert!(!done.is_empty(), "the miniature has matrices to transcode");
        Self {
            _src: src,
            container,
        }
    }

    fn open(&self) -> (ComponentOpPlan, OperandStore) {
        let inspection = inspect_container(self.container.path(), false).unwrap();
        let plan = plan_component_ops(&inspection, self.container.path(), "target")
            .unwrap()
            .plan
            .expect("the per-expert miniature plans");
        let store = OperandStore::open(self.container.path(), &inspection).unwrap();
        (plan, store)
    }
}

fn ledger_of(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    budget: &ResidencyBudget,
) -> ResourceLedger {
    let records = select_realizations_within(
        plan,
        store.into(),
        &ProductionBackend::new(),
        &ExecutionSlice::Full,
        budget,
    )
    .unwrap();
    let priced = expectations(
        &records,
        |op| store.stored_len(op),
        BlockGeometry::executor(),
    );
    ResourceLedger::aggregate(&priced)
}

/// Seven rules: stored once per object, mapped once per mapping,
/// resident summed over committed allocations, transient as the PEAK
/// not the total, touch and page-in per token, device apart. On the
/// miniature the bank is mapped and not resident, and a token touches
/// `top_k / experts` of it.
#[test]
fn the_ledger_aggregates_each_resource_by_its_own_rule() {
    let subject = Subject::build(kimi_per_expert_moe_f32_model);
    let (plan, store) = subject.open();
    let records = select_realizations_within(
        &plan,
        (&store).into(),
        &ProductionBackend::new(),
        &ExecutionSlice::Full,
        &ResidencyBudget::UNBOUNDED,
    )
    .unwrap();
    let priced = expectations(
        &records,
        |op| store.stored_len(op),
        BlockGeometry::executor(),
    );
    let ledger = ResourceLedger::aggregate(&priced);
    let is_bank = |op: &Operation| matches!(op, Operation::ExpertProject { .. });
    let bank_stored: u64 = priced
        .iter()
        .filter(|e| is_bank(&e.operation))
        .map(|e| e.stored_bytes)
        .sum();
    let owned_resident: u64 = priced
        .iter()
        .filter(|e| !is_bank(&e.operation))
        .map(|e| e.declared_resident)
        .sum();
    assert_eq!(ledger.mapped, bank_stored, "the bank is address space");
    assert_eq!(
        ledger.resident, owned_resident,
        "everything else is committed"
    );
    assert_eq!(
        ledger.transient_peak,
        priced.iter().map(|e| e.staging).max().unwrap(),
        "transient is the peak, not the sum"
    );
    assert!(
        ledger.transient_peak < priced.iter().map(|e| e.staging).sum::<u64>(),
        "the sum would overstate the peak"
    );
    let expected_page_in =
        (bank_stored as f64 * MOE_TOP_K as f64 / MOE_EXPERTS as f64).round() as u64;
    // Rounding is per operand, so the sum may differ by at most one byte
    // per bank matrix.
    let bank_matrices = priced.iter().filter(|e| is_bank(&e.operation)).count() as u64;
    assert!(ledger.page_in_per_token.abs_diff(expected_page_in) <= bank_matrices);
    assert_eq!(
        ledger.touch_per_token,
        owned_resident + ledger.page_in_per_token,
        "a token streams every committed image and its share of the mapping"
    );
    assert_eq!(ledger.device, 0);
    assert_eq!(
        ledger.physical_working_set(),
        ledger.resident + ledger.transient_peak + ledger.page_in_per_token
    );
    // Stored footprint counts each object once — the same tensors the
    // stored footprint instrument counts.
    let stored: u64 = priced.iter().map(|e| e.stored_bytes).sum();
    assert_eq!(
        ledger.stored, stored,
        "no operand is bound twice on this plan"
    );
}

/// Unbounded: the backend's own preference, the mapped bank selected and
/// no record re-selected. Constrained just below that working set: at
/// least one realization CHANGES — to a cheaper-resident candidate the
/// backend had considered — and the plan then fits, prepares, and
/// reconciles in the re-selected form.
#[test]
fn a_constrained_budget_changes_a_realization_and_the_plan_still_prepares() {
    let subject = Subject::build(kimi_per_expert_moe_f32_model);
    let (plan, store) = subject.open();
    let unbounded = ledger_of(&plan, &store, &ResidencyBudget::UNBOUNDED);
    let records = select_realizations_within(
        &plan,
        (&store).into(),
        &ProductionBackend::new(),
        &ExecutionSlice::Full,
        &ResidencyBudget::UNBOUNDED,
    )
    .unwrap();
    assert!(records
        .iter()
        .all(|r| r.selection.reason != SelectionReason::BudgetPolicy));
    assert!(records.iter().any(|r| matches!(
        r.selection.realization.form,
        RealizationForm::MappedStored { .. }
    )));

    let budget = ResidencyBudget::physical(unbounded.physical_working_set() - 1);
    let constrained = select_realizations_within(
        &plan,
        (&store).into(),
        &ProductionBackend::new(),
        &ExecutionSlice::Full,
        &budget,
    )
    .unwrap();
    let changed: Vec<_> = constrained
        .iter()
        .zip(&records)
        .filter(|(c, u)| c.selection.realization != u.selection.realization)
        .collect();
    assert!(!changed.is_empty(), "selection, not validation");
    for (c, u) in &changed {
        assert_eq!(c.selection.reason, SelectionReason::BudgetPolicy);
        assert!(
            u.selection.candidates.contains(&c.selection.realization),
            "re-selected among the backend's own candidates"
        );
        assert!(
            c.selection.residency.bytes_per_weight < u.selection.residency.bytes_per_weight,
            "cheaper resident"
        );
    }
    let fitted = ledger_of(&plan, &store, &budget);
    assert!(fitted.physical_working_set() <= budget.physical_bytes.unwrap());
    assert!(fitted.physical_working_set() < unbounded.physical_working_set());
    // The mapped bank is untouched by the budget: it was never resident.
    assert_eq!(fitted.mapped, unbounded.mapped);

    let loads_before = store.load_count();
    let ops = PreparedOperands::load_within(
        &plan,
        &store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
        &budget,
    )
    .unwrap();
    assert!(
        store.load_count() > loads_before,
        "binding reads the owned operands"
    );
    let reconciled = ops.reconcile(&plan, (&store).into()).unwrap();
    assert_eq!(reconciled.matched, constrained.len());
    assert!(ops
        .realizations()
        .iter()
        .any(|r| r.selection.reason == SelectionReason::BudgetPolicy));
}

/// Impossible: no candidate brings the plan inside, so the preparation
/// is refused BEFORE any payload byte, naming the deficit, the largest
/// committed operands with nowhere cheaper to go, and the alternatives
/// already taken. A throughput budget no plan can meet refuses the same
/// way, naming the per-token deficit.
#[test]
fn an_impossible_budget_refuses_before_io_with_the_deficit_and_the_alternatives() {
    let subject = Subject::build(kimi_per_expert_moe_f32_model);
    let (plan, store) = subject.open();
    let loads_before = store.load_count();
    let err = select_realizations_within(
        &plan,
        (&store).into(),
        &ProductionBackend::new(),
        &ExecutionSlice::Full,
        &ResidencyBudget::physical(1),
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("within the residency budget before any payload byte"),
        "{err}"
    );
    assert!(err.contains("irreducible deficit"), "{err}");
    assert!(
        err.contains("largest committed operands with no cheaper candidate"),
        "{err}"
    );
    assert!(err.contains("alternatives already taken"), "{err}");
    assert_eq!(store.load_count(), loads_before, "refused before any byte");

    let starved = ResidencyBudget::UNBOUNDED.with_throughput(ThroughputBudget {
        bytes_per_second: 1,
        target_tokens_per_second: 1.0,
    });
    let err = select_realizations_within(
        &plan,
        (&store).into(),
        &ProductionBackend::new(),
        &ExecutionSlice::Full,
        &starved,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("GB per token"), "{err}");
    assert_eq!(store.load_count(), loads_before);
    // And the machine's own budget is a real number here.
    assert!(ResidencyBudget::machine()
        .physical_bytes
        .is_some_and(|b| b > 0));
    assert_eq!(
        ThroughputBudget {
            bytes_per_second: 1_000,
            target_tokens_per_second: 4.0
        }
        .bytes_per_token(),
        250
    );
}
