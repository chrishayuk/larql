//! `larql vindex3 ops --realizations` — the prepared plan's admission,
//! priced, with no payload byte read.
//!
//! The rung-3 machinery decides everything about an execution before it
//! reads the weights: which realization each planned operand is pinned
//! to, or why none is admissible, and what the pins declare they will
//! hold. All of that is a function of the plan, the backend and the
//! container's tensor TABLES, so this verb reports it from those alone.
//! It is the front door of the K3 vertical's exit criterion: a plan
//! either declares a working set the budget holds, or refuses before I/O
//! with every missing realization named.

use std::path::Path;

use larql_vindex::format::vindex3::inspect::SystemInspection;
use larql_vindex::format::vindex3::opplan::exec::accounting::{
    execution_touch, expectations, render_selection_summary, stored_footprint, BlockGeometry,
};
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prepared::{select_realizations, ExecutionSlice};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::ComponentOpPlan;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const GB: f64 = 1e9;

pub(super) fn report(
    plan: &ComponentOpPlan,
    container: &Path,
    inspection: &SystemInspection,
    budget_gib: Option<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Tensor tables only: `open` reads segment headers, never a payload.
    let store = OperandStore::open(container, inspection)?;
    report_through(plan, &store, budget_gib)
}

/// The report over a store the caller opened — the registry it decodes
/// through is the one selection consults, so a store bound to a registry
/// that lacks a representation refuses here, before I/O, by name.
pub(super) fn report_through(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    budget_gib: Option<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let backend = ProductionBackend::new();
    let planned = plan.planned_operands().len();
    println!("planned operands: {planned}");
    let records = match select_realizations(plan, store.into(), &backend, &ExecutionSlice::Full) {
        Ok(records) => records,
        Err(refused) => {
            println!("REFUSED before I/O:");
            println!("{refused}");
            return Err("the plan has no admissible preparation; nothing was read".into());
        }
    };
    println!("pinned: {} of {planned} planned operands", records.len());
    print!("{}", render_selection_summary(&records));

    let priced = expectations(
        &records,
        |op| store.stored_len(op),
        BlockGeometry::executor(),
    );
    let resident: u64 = priced.iter().map(|e| e.declared_resident).sum();
    let staging_peak: u64 = priced.iter().map(|e| e.staging).max().unwrap_or(0);
    let staging_total: u64 = priced.iter().map(|e| e.staging).sum();
    let footprint = stored_footprint(&priced);
    let touch = execution_touch(&priced);
    let budget = budget_gib.unwrap_or_else(physical_memory_gib);
    println!("declared, from tensor tables (no payload read):");
    println!(
        "  stored footprint     {:>10.2} GB  ({} distinct operands)",
        footprint.bytes as f64 / GB,
        footprint.operands
    );
    println!(
        "  execution touch      {:>10.2} GB  (stored bytes read, once per operation)",
        touch as f64 / GB
    );
    println!(
        "  declared resident    {:>10.2} GB  ({:.2} GiB)",
        resident as f64 / GB,
        resident as f64 / GIB
    );
    println!(
        "  staging              {:>10.2} GB peak single operand, {:.2} GB if all at once",
        staging_peak as f64 / GB,
        staging_total as f64 / GB
    );
    let working_set = resident + staging_peak;
    println!(
        "  working set          {:>10.2} GiB  (resident + one operand staging)",
        working_set as f64 / GIB
    );
    println!("  budget               {budget:>10.2} GiB");
    if working_set as f64 <= budget * GIB {
        println!("  verdict              WITHIN BUDGET");
        Ok(())
    } else {
        println!(
            "  verdict              OVER BUDGET by {:.2} GiB",
            working_set as f64 / GIB - budget
        );
        Err("the declared working set exceeds the budget; nothing was read".into())
    }
}

/// The machine's physical memory, in GiB — the default budget.
pub(super) fn physical_memory_gib() -> f64 {
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
        {
            if let Ok(bytes) = String::from_utf8_lossy(&out.stdout).trim().parse::<u64>() {
                return bytes as f64 / GIB;
            }
        }
    }
    0.0
}
