//! `larql vindex3 ops --realizations` — the prepared plan's admission,
//! priced in seven resources, held against a residency budget, with no
//! payload byte read.
//!
//! The rung-3 machinery decides everything about an execution before it
//! reads the weights: which realization each planned operand is pinned
//! to, or why none is admissible, and what the pins demand of each
//! resource. All of that is a function of the plan, the backend and the
//! container's tensor TABLES, so this verb reports it from those alone
//! — and under a budget it reports the re-selections the budget forced,
//! or the refusal with its deficit. It is the front door of the K3
//! vertical's acceptance statement: the complete graph selects a
//! physically executable set of realizations within the machine budget,
//! or refuses before reading payload bytes with the exact deficit and
//! alternatives.

use std::path::Path;

use larql_vindex::format::vindex3::inspect::SystemInspection;
use larql_vindex::format::vindex3::opplan::exec::accounting::{
    execution_touch, expectations, render_selection_summary, stored_footprint, BlockGeometry,
    ResidencyBudget, ResourceLedger, ThroughputBudget,
};
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prepared::{
    select_realizations_within, ExecutionSlice, PreparedOperands,
};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::exec::realization::SelectionReason;
use larql_vindex::format::vindex3::opplan::ComponentOpPlan;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const GB: f64 = 1e9;

/// What the caller asked the report to hold the plan against.
#[derive(Debug, Clone, Copy)]
pub(super) struct Ask {
    /// Physical budget in GiB; `None` = this machine's memory.
    pub budget_gib: Option<f64>,
    /// Host bandwidth in GB/s; `None` = no throughput constraint.
    pub bandwidth_gbs: Option<f64>,
    pub target_tok_s: f64,
    pub bind: bool,
}

impl Ask {
    pub(super) fn budget(&self) -> ResidencyBudget {
        let mut budget = match self.budget_gib {
            Some(gib) => ResidencyBudget::physical((gib * GIB) as u64),
            None => ResidencyBudget::machine(),
        };
        if let Some(gbs) = self.bandwidth_gbs {
            budget = budget.with_throughput(ThroughputBudget {
                bytes_per_second: (gbs * GB) as u64,
                target_tokens_per_second: self.target_tok_s,
            });
        }
        budget
    }
}

pub(super) fn report(
    plan: &ComponentOpPlan,
    container: &Path,
    inspection: &SystemInspection,
    ask: Ask,
) -> Result<(), Box<dyn std::error::Error>> {
    // Tensor tables only: `open` reads segment headers, never a payload.
    let store = OperandStore::open(container, inspection)?;
    let budget = ask.budget();
    report_through(plan, &store, &budget)?;
    if ask.bind {
        bind_and_reconcile(plan, &store, &budget)?;
    }
    Ok(())
}

/// Prepare the plan for real under the budget — every pin bound to its
/// object — and hold what the loader bound against what the pins
/// declared: committed bytes reconciled exactly, mappings held to their
/// address space, and the pages of those mappings physically resident
/// at this moment reported beside them.
fn bind_and_reconcile(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    budget: &ResidencyBudget,
) -> Result<(), Box<dyn std::error::Error>> {
    let started = std::time::Instant::now();
    let ops = PreparedOperands::load_within(
        plan,
        store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
        budget,
    )?;
    let bound_in = started.elapsed().as_secs_f64();
    let reconciled = ops.reconcile(plan, store.into())?;
    println!("bound, from the container:");
    println!("  prepared in           {bound_in:>10.1} s");
    println!(
        "  objects mapped        {:>10}  ({} regions bound, none read)",
        store.mapped_objects(),
        store.mapped_regions()
    );
    println!("  payload reads         {:>10}", store.load_count());
    println!("  pins reconciled       {:>10}", reconciled.matched);
    println!(
        "  committed             {:>10.2} GB declared, {:.2} GB observed, {:.3} GB padding",
        reconciled.declared_resident as f64 / GB,
        reconciled.observed_resident as f64 / GB,
        reconciled.padding as f64 / GB
    );
    println!(
        "  mapped                {:>10.2} GB address space, {:.3} GB physically resident now",
        reconciled.mapped as f64 / GB,
        reconciled.mapped_resident as f64 / GB
    );
    println!("  verdict               RECONCILED");
    Ok(())
}

/// The report over a store the caller opened — the registry it decodes
/// through is the one selection consults, so a store bound to a registry
/// that lacks a representation refuses here, before I/O, by name.
pub(super) fn report_through(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    budget: &ResidencyBudget,
) -> Result<(), Box<dyn std::error::Error>> {
    let backend = ProductionBackend::new();
    let planned = plan.planned_operands().len();
    println!("planned operands: {planned}");
    println!("budget: {}", describe(budget));
    let records = match select_realizations_within(
        plan,
        store.into(),
        &backend,
        &ExecutionSlice::Full,
        budget,
    ) {
        Ok(records) => records,
        Err(refused) => {
            println!("REFUSED before I/O:");
            println!("{refused}");
            return Err(
                "the plan has no admissible preparation within the budget; nothing was read".into(),
            );
        }
    };
    let re_selected = records
        .iter()
        .filter(|r| r.selection.reason == SelectionReason::BudgetPolicy)
        .count();
    println!(
        "pinned: {} of {planned} planned operands ({re_selected} re-selected for the budget)",
        records.len()
    );
    print!("{}", render_selection_summary(&records));

    let priced = expectations(
        &records,
        |op| store.stored_len(op),
        BlockGeometry::executor(),
    );
    let ledger = ResourceLedger::aggregate(&priced);
    let footprint = stored_footprint(&priced);
    let touch_once = execution_touch(&priced);
    println!("resources, from tensor tables (no payload read):");
    println!(
        "  stored footprint      {:>10.2} GB  once per object ({} distinct operands)",
        ledger.stored as f64 / GB,
        footprint.operands
    );
    println!(
        "  mapped address space  {:>10.2} GB  once per mapping; resident only as touched",
        ledger.mapped as f64 / GB
    );
    println!(
        "  persistent resident   {:>10.2} GB  committed allocations, summed",
        ledger.resident as f64 / GB
    );
    println!(
        "  transient peak        {:>10.2} GB  largest overlapping staging, not the total",
        ledger.transient_peak as f64 / GB
    );
    println!(
        "  execution touch       {:>10.2} GB per token  ({:.2} GB stored bytes read once at load)",
        ledger.touch_per_token as f64 / GB,
        touch_once as f64 / GB
    );
    println!(
        "  expected page-in      {:>10.3} GB per token  cold, from the mappings",
        ledger.page_in_per_token as f64 / GB
    );
    println!(
        "  device memory         {:>10.2} GB",
        ledger.device as f64 / GB
    );
    println!(
        "  physical working set  {:>10.2} GiB  (resident + transient peak + page-in per token)",
        ledger.physical_working_set() as f64 / GIB
    );
    let deficit = budget.deficit(&ledger);
    if deficit.is_zero() {
        println!("  verdict               WITHIN BUDGET");
        Ok(())
    } else {
        println!(
            "  verdict               OVER BUDGET by {:.2} GiB physical, {:.2} GB per token",
            deficit.physical as f64 / GIB,
            deficit.touch_per_token as f64 / GB
        );
        Err("the plan exceeds the budget; nothing was read".into())
    }
}

fn describe(budget: &ResidencyBudget) -> String {
    let physical = budget
        .physical_bytes
        .map(|b| format!("{:.2} GiB physical", b as f64 / GIB))
        .unwrap_or_else(|| "no physical limit".to_string());
    let throughput = budget
        .throughput
        .map(|t| {
            format!(
                "{:.2} GB per token ({:.1} GB/s at {:.1} tok/s)",
                t.bytes_per_token() as f64 / GB,
                t.bytes_per_second as f64 / GB,
                t.target_tokens_per_second
            )
        })
        .unwrap_or_else(|| "no throughput limit".to_string());
    format!("{physical}; {throughput}")
}
