//! The refusal a first-run encode owes when the plan is not admissible:
//! the blocking findings, one line each, and where to read the rest.
//!
//! `larql vindex3 plan` prints the whole plan as JSON — the right output
//! for a verb whose job is the plan, and the wrong one for `larql run`,
//! whose user asked for text. This renders the same findings short.

use larql_vindex::format::vindex3::plan::capability::{Capability, CapabilityStatus};
use larql_vindex::format::vindex3::plan::{FindingCategory, PlannedFinding, SystemPlan};

use super::super::vindex3_cmd::EncodeCapability;

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

/// What a `None` scope is called.
pub const WHOLE_MODEL_LABEL: &str = "the whole model";

/// The gate's name in prose: the capability's own snake_case spelling
/// with spaces, or the whole model.
pub fn scope_label(scope: Option<Capability>) -> String {
    match scope {
        Some(capability) => format!("{capability:?}")
            .chars()
            .enumerate()
            .flat_map(|(i, c)| {
                let space = (i > 0 && c.is_ascii_uppercase()).then_some(' ');
                space
                    .into_iter()
                    .chain(std::iter::once(c.to_ascii_lowercase()))
            })
            .collect(),
        None => WHOLE_MODEL_LABEL.to_string(),
    }
}

/// The refusal text for `spec`, gated on `scope`.
pub fn render(
    spec: &str,
    scope: Option<Capability>,
    verdict: Verdict,
    blockers: &[Blocker],
) -> String {
    let scope_label = scope_label(scope);
    let mut out = match verdict {
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
    out.push_str(&blocker_lines(blockers));
    out.push_str(&format!("\n{}", plan_hint(spec, scope)));
    out
}

/// The blockers, one indented line each after a leading newline, capped
/// at [`MAX_LISTED_BLOCKERS`] with the rest counted. Empty for none.
pub fn blocker_lines(blockers: &[Blocker]) -> String {
    let mut out = String::new();
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
    out
}

/// Where to read the whole plan, judged by the same gate this run used:
/// `larql vindex3 plan --capability` applies `encode`'s scoped gate, so it
/// reports exactly the blockers above.
pub fn plan_hint(spec: &str, scope: Option<Capability>) -> String {
    let flag = scope
        .and_then(EncodeCapability::flag_value)
        .map(|value| format!(" {CAPABILITY_FLAG} {value}"))
        .unwrap_or_default();
    format!("full plan: larql vindex3 plan {spec}{flag}")
}

/// The flag `larql vindex3 plan` and `encode` scope their gate with.
pub const CAPABILITY_FLAG: &str = "--capability";

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
