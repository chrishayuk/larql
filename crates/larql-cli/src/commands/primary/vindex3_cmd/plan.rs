//! `vindex3 plan`: the G1 semantic representability gate.

use larql_vindex::format::vindex3::plan::admission::admit;
use larql_vindex::format::vindex3::plan::capability::{Capability, CapabilityStatus};
use larql_vindex::format::vindex3::plan::{plan_resolved, SystemPlan};

use crate::commands::primary::hf_auto::refusal;

#[allow(unused_imports)]
use super::*;

pub(super) fn run_plan(args: PlanArgs) -> Result<(), Box<dyn std::error::Error>> {
    let resolved = artifact::resolve_all(&args.artifacts)?;
    for entry in &resolved {
        report_staging(entry);
    }
    let plan = plan_resolved(&args.artifacts, resolved)?;
    let json = serde_json::to_string_pretty(&plan)?;
    match &args.output {
        Some(path) => {
            std::fs::write(path, &json)?;
            eprintln!("plan written to {}", path.display());
        }
        None => println!("{json}"),
    }
    let summary = &plan.summary;
    eprintln!(
        "plan: {} representable, {} mismatched, {} unrepresented, {} interfaces — {} blocking",
        summary.representable,
        summary.mismatched,
        summary.unrepresented,
        summary.interfaces,
        summary.blocking,
    );
    match args.capability.map(Capability::from) {
        None if plan.admissible => Ok(()),
        None => Err(format!(
            "plan not admissible: {} blocking finding(s); \
             every one is a schema or resolution gap to close before conversion",
            summary.blocking
        )
        .into()),
        Some(capability) => judge_capability(&plan, capability),
    }
}

/// The scoped verdict, from the same gate `encode --capability` applies.
/// A refusal lists the blockers inside the closure, one per line.
fn judge_capability(
    plan: &SystemPlan,
    capability: Capability,
) -> Result<(), Box<dyn std::error::Error>> {
    match admit(plan, Some(capability)) {
        Ok(status) => {
            eprintln!("{}", admitted_line(plan, capability, status));
            Ok(())
        }
        Err(refused) => {
            let (_, blockers) = refusal::judge(plan, Some(capability));
            Err(format!("{refused}{}", refusal::blocker_lines(&blockers)).into())
        }
    }
}

/// The closing line of an admissible scoped plan: the capability, its
/// closure, and what the whole-model verdict still says.
pub(crate) fn admitted_line(
    plan: &SystemPlan,
    capability: Capability,
    status: Option<&CapabilityStatus>,
) -> String {
    let required = status.map_or(0, |s| s.required);
    let whole_model = if plan.admissible {
        "whole model admissible too".to_string()
    } else {
        format!(
            "whole-model completeness NOT asserted ({} blocking finding(s) outside this closure)",
            plan.summary.blocking
        )
    };
    format!(
        "admissible for {capability:?}: 0 of {required} finding(s) in its closure block; \
         {whole_model}"
    )
}
