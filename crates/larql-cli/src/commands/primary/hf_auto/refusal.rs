//! The refusal a first-run encode owes when the plan is not admissible:
//! the blocking findings, one line each, and where to read the rest.
//!
//! `larql vindex3 plan` prints the whole plan as JSON — the right output
//! for a verb whose job is the plan, and the wrong one for `larql run`,
//! whose user asked for text. This renders the same findings short.

use larql_vindex::format::vindex3::plan::capability::{Capability, CapabilityStatus};
use larql_vindex::format::vindex3::plan::{FindingCategory, PlannedFinding, SystemPlan};

/// Findings listed before the rest are summarised as a count.
pub const MAX_LISTED_BLOCKERS: usize = 12;

/// A finding's detail is cut to this many characters in the list.
pub const MAX_DETAIL_CHARS: usize = 110;

/// One blocking finding, as the refusal shows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Blocker {
    pub component: String,
    pub subject: String,
    pub category: &'static str,
    pub detail: String,
}

/// Why the gate closed, in the words the refusal leads with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The gate passes.
    Admissible,
    /// Findings inside the gated scope block.
    Blocked,
    /// The scope's semantics are understood but this build cannot run it.
    Unsupported,
    /// The checkpoint ships nothing the scope needs.
    Unavailable,
}

/// Judge `plan` for `scope` (`None` = whole-model completeness, the
/// `larql vindex3 encode` default) and collect the findings that block.
pub fn judge(plan: &SystemPlan, scope: Option<Capability>) -> (Verdict, Vec<Blocker>) {
    let findings = plan.artifacts.iter().flat_map(|a| a.findings.iter());
    let Some(capability) = scope else {
        let blockers: Vec<Blocker> = findings.filter(|f| f.blocks()).map(blocker).collect();
        let verdict = if plan.admissible {
            Verdict::Admissible
        } else {
            Verdict::Blocked
        };
        return (verdict, blockers);
    };
    let Some(status) = plan
        .capabilities
        .iter()
        .find(|c| c.capability == capability)
    else {
        return (Verdict::Unsupported, Vec::new());
    };
    let blockers = findings
        .filter(|f| status.blocker_ids.contains(&f.id))
        .map(blocker)
        .collect();
    (capability_verdict(status), blockers)
}

fn capability_verdict(status: &CapabilityStatus) -> Verdict {
    if !status.admissible {
        Verdict::Blocked
    } else if !status.available {
        Verdict::Unavailable
    } else if !status.supported {
        Verdict::Unsupported
    } else {
        Verdict::Admissible
    }
}

fn blocker(finding: &PlannedFinding) -> Blocker {
    Blocker {
        component: finding.component.clone(),
        subject: finding.subject.clone(),
        category: category_label(finding.category),
        detail: finding.detail.clone(),
    }
}

/// The plan JSON's own spelling of a category.
pub fn category_label(category: FindingCategory) -> &'static str {
    match category {
        FindingCategory::Representable => "representable",
        FindingCategory::Mismatched => "mismatched",
        FindingCategory::Unrepresented => "unrepresented",
        FindingCategory::Interface => "interface",
    }
}

/// The refusal text for `spec`, gated on `scope_label`.
pub fn render(spec: &str, scope_label: &str, verdict: Verdict, blockers: &[Blocker]) -> String {
    let headline = match verdict {
        Verdict::Admissible => format!("{spec} is admissible for {scope_label}"),
        Verdict::Blocked => format!(
            "cannot run {spec}: LARQL does not yet represent everything {scope_label} needs \
             ({} blocking finding(s)); only headers were read",
            blockers.len()
        ),
        Verdict::Unsupported => {
            format!("cannot run {spec}: this build has no executor for {scope_label}")
        }
        Verdict::Unavailable => {
            format!("cannot run {spec}: the checkpoint ships nothing {scope_label} needs")
        }
    };
    let mut out = headline;
    for b in blockers.iter().take(MAX_LISTED_BLOCKERS) {
        out.push_str(&format!(
            "\n  - [{}] {}: {} — {}",
            b.category,
            b.component,
            b.subject,
            truncate(&b.detail, MAX_DETAIL_CHARS)
        ));
    }
    if let Some(more) = blockers.len().checked_sub(MAX_LISTED_BLOCKERS) {
        if more > 0 {
            out.push_str(&format!("\n  … and {more} more"));
        }
    }
    out.push_str(&format!("\n{}", plan_hint(spec, scope_label)));
    out
}

/// Where to read the whole plan. `larql vindex3 plan` judges whole-model
/// completeness and has no capability flag, so it can report blockers
/// outside the gated scope; the hint says which findings gated this run.
pub fn plan_hint(spec: &str, scope_label: &str) -> String {
    format!(
        "full plan: larql vindex3 plan {spec} --output plan.json \
         (whole-model; this run was gated on {scope_label} only, the findings above)"
    )
}

/// `s` cut to at most `max` characters, on a character boundary, with an
/// ellipsis when anything was cut. Only the first line is kept, with
/// runs of whitespace (a wrapped source string) collapsed to one space.
pub fn truncate(s: &str, max: usize) -> String {
    let line = s
        .lines()
        .next()
        .unwrap_or_default()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if line.chars().count() <= max {
        return line;
    }
    let cut: String = line.chars().take(max.saturating_sub(1)).collect();
    format!("{}…", cut.trim_end())
}
