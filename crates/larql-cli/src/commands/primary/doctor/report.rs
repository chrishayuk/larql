//! The doctor's checklist: one [`Check`] per line, rendered as text or
//! JSON, and the exit decision.

use serde::Serialize;

use super::requirements::GETTING_STARTED_DOC;

/// How one check came out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Status {
    /// Fine.
    Ok,
    /// Worth fixing, but the quick start still works.
    Warn,
    /// Stops the quick start working; the only status that fails the run.
    Fail,
    /// Not checked (`--offline`, or the answer was unavailable).
    Skip,
}

impl Status {
    pub(crate) fn symbol(self) -> &'static str {
        match self {
            Status::Ok => "✓",
            Status::Warn => "!",
            Status::Fail => "✗",
            Status::Skip => "-",
        }
    }
}

/// The heading a check is listed under, in display order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Group {
    Larql,
    Platform,
    Server,
    Caches,
    HuggingFace,
    Updates,
}

impl Group {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Group::Larql => "LARQL",
            Group::Platform => "Platform",
            Group::Server => "Server",
            Group::Caches => "Caches",
            Group::HuggingFace => "Hugging Face",
            Group::Updates => "Updates",
        }
    }
}

/// One line of the checklist, with a fix hint when it is not fine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct Check {
    pub group: Group,
    pub name: String,
    pub status: Status,
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

impl Check {
    fn new(group: Group, name: &str, status: Status, summary: String) -> Self {
        Check {
            group,
            name: name.to_string(),
            status,
            summary,
            hint: None,
        }
    }

    pub(crate) fn ok(group: Group, name: &str, summary: impl Into<String>) -> Self {
        Self::new(group, name, Status::Ok, summary.into())
    }

    pub(crate) fn skip(group: Group, name: &str, summary: impl Into<String>) -> Self {
        Self::new(group, name, Status::Skip, summary.into())
    }

    pub(crate) fn warn(
        group: Group,
        name: &str,
        summary: impl Into<String>,
        hint: impl Into<String>,
    ) -> Self {
        Self::new(group, name, Status::Warn, summary.into()).with_hint(hint)
    }

    pub(crate) fn fail(
        group: Group,
        name: &str,
        summary: impl Into<String>,
        hint: impl Into<String>,
    ) -> Self {
        Self::new(group, name, Status::Fail, summary.into()).with_hint(hint)
    }

    fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }
}

/// Indent of a check line under its heading.
const CHECK_INDENT: &str = "  ";
/// Indent of a fix hint under its check.
const HINT_INDENT: &str = "      ";
/// Marks a fix hint.
const HINT_ARROW: &str = "→";
/// Gap between the name column and the summary.
const NAME_GAP: usize = 2;

/// The whole checklist, in the order the checks ran.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Report {
    pub checks: Vec<Check>,
}

impl Report {
    pub(crate) fn count(&self, status: Status) -> usize {
        self.checks.iter().filter(|c| c.status == status).count()
    }

    /// Checks that stop the quick start working.
    pub(crate) fn problems(&self) -> usize {
        self.count(Status::Fail)
    }

    /// The checklist as terminal text: a heading per group, a line per
    /// check, a hint under each one that is not fine, and a verdict.
    pub(crate) fn render_text(&self) -> String {
        let width = self.checks.iter().map(|c| c.name.chars().count()).max();
        let width = width.unwrap_or(0) + NAME_GAP;
        let mut out = String::new();
        let mut current = None;
        for check in &self.checks {
            if current != Some(check.group) {
                if current.is_some() {
                    out.push('\n');
                }
                out.push_str(check.group.label());
                out.push('\n');
                current = Some(check.group);
            }
            out.push_str(&format!(
                "{CHECK_INDENT}{} {:<width$}{}\n",
                check.status.symbol(),
                check.name,
                check.summary
            ));
            if let Some(hint) = &check.hint {
                out.push_str(&format!("{HINT_INDENT}{HINT_ARROW} {hint}\n"));
            }
        }
        out.push('\n');
        out.push_str(&self.verdict());
        out.push('\n');
        out
    }

    /// The closing line.
    pub(crate) fn verdict(&self) -> String {
        let problems = self.problems();
        let warnings = self.count(Status::Warn);
        if problems > 0 {
            format!(
                "{problems} problem(s) will stop the quick start ({GETTING_STARTED_DOC}); \
                 fix the {} lines first.",
                Status::Fail.symbol()
            )
        } else if warnings > 0 {
            format!("Ready for the quick start, with {warnings} warning(s) above.")
        } else {
            "Ready for the quick start. No problems found.".to_string()
        }
    }

    /// The checklist for machines.
    pub(crate) fn to_json(&self, larql_version: &str) -> serde_json::Value {
        serde_json::json!({
            "larql_version": larql_version,
            "ready": self.problems() == 0,
            "problems": self.problems(),
            "warnings": self.count(Status::Warn),
            "checks": self.checks,
        })
    }

    /// `Err` when any check failed, so the process exits non-zero.
    pub(crate) fn outcome(&self) -> Result<(), String> {
        match self.problems() {
            0 => Ok(()),
            n => Err(format!("larql doctor found {n} problem(s)")),
        }
    }
}
