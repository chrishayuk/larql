//! **Where a rendered field came from.**
//!
//! Stage 1d's anti-cheat asks a stored snapshot whether it carries a
//! CONCLUSION. This is its mirror, and the facade's whole theorem:
//!
//! > **Anything an agent can learn through this facade is already
//! > derivable from the deterministic optimiser substrate.**
//!
//! A view is not trusted to obey that by inspection. Every field it
//! renders declares the substrate call that produced it, and a test
//! walks the serialised form to check the two agree in BOTH
//! directions — an undeclared field fails, and so does a declaration
//! for a field that is no longer rendered. The registry is therefore
//! not documentation that can rot; it is the only way a field is
//! allowed to exist.
//!
//! # Why a whole subtree may be one origin
//!
//! Most of the substrate's types already derive `Serialize`, and where
//! one does the honest rendering is the type itself: a [`Margin`] in a
//! response is the optimiser's own `Margin`, not a reshaping of it that
//! could disagree with it. Such a field declares ONE origin and the
//! walk stops there. What the walk is hunting is the opposite case — a
//! scalar the view invented, sitting in the response with no call
//! behind it.
//!
//! [`Margin`]: super::super::constraint::Margin

use std::collections::BTreeSet;

use serde::Serialize;

/// Separates a field from its parent in a rendered path.
const PATH_SEP: char = '.';

/// Stands in for every index of an array, so a path is a shape and not
/// a position: `states[].adjudications[].binding`.
const INDEX: &str = "[]";

/// One rendered field, and the substrate call behind it.
///
/// `call` is written as it appears in the source — `Adjudication::
/// binding()`, `SearchSnapshot::frontier()` — because the point of
/// recording it is that a reader can go and check.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Origin {
    /// The field's path in the rendered form, arrays written `[]`.
    pub field: String,
    /// The substrate call that produced it.
    pub call: &'static str,
}

impl Origin {
    pub fn new(field: impl Into<String>, call: &'static str) -> Self {
        Self {
            field: field.into(),
            call,
        }
    }

    /// The same origin, one level down: `states[]` + `admitted` reads
    /// `states[].admitted`.
    ///
    /// A nested view declares its fields once and every container
    /// re-roots them, so a field cannot be declared in one place and
    /// forgotten in another.
    pub fn under(&self, prefix: &str) -> Self {
        Self {
            field: match prefix.is_empty() {
                true => self.field.clone(),
                false => format!("{prefix}{PATH_SEP}{}", self.field),
            },
            call: self.call,
        }
    }
}

/// A view whose every field names its origin.
pub trait Rendered: Serialize {
    /// Every field this view renders, and the substrate call behind it.
    ///
    /// Returned by value rather than as a `&'static [_]` because a
    /// container builds its registry by re-rooting the registries of
    /// the views it nests, and a nesting depth is not known at compile
    /// time. It is called once per check, never per response.
    fn origins() -> Vec<Origin>;
}

/// What a walk of a rendered value found.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Coverage {
    /// Paths that reached a declared origin.
    pub covered: BTreeSet<String>,
    /// Leaves with no declared origin — a field the view invented.
    pub undeclared: BTreeSet<String>,
}

impl Coverage {
    /// Origins declared but never reached: a stale registry entry, which
    /// is the same defect as an undeclared field seen from the other
    /// side.
    pub fn unreached<'a>(&self, origins: &'a [Origin]) -> Vec<&'a str> {
        origins
            .iter()
            .map(|o| o.field.as_str())
            .filter(|f| !self.covered.contains(*f))
            .collect()
    }
}

/// Walk a rendered value against a view's declared origins.
///
/// Descent stops at any path that has an origin, so an embedded
/// substrate type is covered whole. A leaf reached without one is
/// recorded as undeclared, which is what makes the check bite.
pub fn walk<T: Rendered>(view: &T) -> Result<Coverage, serde_json::Error> {
    let value = serde_json::to_value(view)?;
    let origins = T::origins();
    let declared: BTreeSet<&str> = origins.iter().map(|o| o.field.as_str()).collect();
    let mut coverage = Coverage::default();
    visit(&value, String::new(), &declared, &mut coverage);
    Ok(coverage)
}

fn visit(
    value: &serde_json::Value,
    path: String,
    declared: &BTreeSet<&str>,
    coverage: &mut Coverage,
) {
    if !path.is_empty() && declared.contains(path.as_str()) {
        coverage.covered.insert(path);
        return;
    }
    match value {
        serde_json::Value::Object(fields) => {
            for (name, child) in fields {
                visit(child, join(&path, name), declared, coverage);
            }
        }
        serde_json::Value::Array(items) => {
            let indexed = format!("{path}{INDEX}");
            // An empty array reaches no leaf, so no element can vouch
            // for its shape. The registry still can: a declaration for
            // anything under `path[]` says what would have been there.
            // Absent that, the array is a field nobody declared.
            if items.is_empty() {
                if !declared.iter().any(|f| f.starts_with(&indexed)) {
                    coverage.undeclared.insert(path);
                }
                return;
            }
            for item in items {
                visit(item, indexed.clone(), declared, coverage);
            }
        }
        // A field that is absent reaches no leaf either, and the same
        // rule applies: a declaration for anything beneath it says what
        // would have been there. `Some(x)` is covered by descending
        // into x, so this arm is only ever reached by a real absence.
        serde_json::Value::Null => {
            let beneath = format!("{path}{PATH_SEP}");
            if !declared.iter().any(|f| f.starts_with(&beneath)) {
                coverage.undeclared.insert(path);
            }
        }
        _ => {
            coverage.undeclared.insert(path);
        }
    }
}

fn join(parent: &str, name: &str) -> String {
    match parent.is_empty() {
        true => name.to_string(),
        false => format!("{parent}{PATH_SEP}{name}"),
    }
}
