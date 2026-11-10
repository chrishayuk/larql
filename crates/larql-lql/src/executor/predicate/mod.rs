//! `WHERE` predicates, validated before anything reads or writes.
//!
//! The parser accepts any `field op value` triple. What a statement can
//! actually *honour* is narrower, and the gap used to be closed by
//! ignoring whatever was not understood — so `DELETE FROM EDGES WHERE
//! confidence < 0.1` matched every feature, and `WHERE layer > 20` meant
//! `layer = 20`. On a destructive verb, a predicate that silently widens
//! is the worst available reading.
//!
//! [`EdgePredicate::parse`] is the one place a condition list becomes
//! filters. It either understands **every** condition or refuses the
//! whole clause; a mixed valid/invalid clause is refused as a unit, and
//! the refusal happens before the caller has touched any session state.
//!
//! SELECT, DELETE and UPDATE share this grammar, and the relation and
//! score matchers below are the only implementations of those
//! comparisons — one predicate means one thing on every verb:
//!
//! ```text
//! layer   = <non-negative integer>
//! feature = <non-negative integer>
//! entity  = <string>                      (alias: token, where listed)
//! relation {= | != | LIKE} <string>
//! score   {= | != | < | <= | > | >=} <number>   (alias: confidence)
//! ```
//!
//! At most one condition per field: `layer >= 20 AND layer <= 30` is a
//! range, which no executor evaluates, so it is refused rather than read
//! as its first half.

#[cfg(test)]
mod tests;

use crate::ast::{CompareOp, Condition, Value};
use crate::error::LqlError;

/// Tolerance for `score = x` / `score != x`. Scores are f32 computed
/// from weights; exact equality on a typed literal would never match.
pub(crate) const SCORE_EQ_TOLERANCE: f32 = 1e-3;

/// The `LIKE` wildcard.
const LIKE_WILDCARD: char = '%';

/// A field a `WHERE` clause can constrain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PredicateField {
    Layer,
    Feature,
    Entity,
    Relation,
    Score,
}

impl PredicateField {
    /// The spelling used in refusals.
    fn name(self) -> &'static str {
        match self {
            Self::Layer => "layer",
            Self::Feature => "feature",
            Self::Entity => "entity",
            Self::Relation => "relation",
            Self::Score => "score",
        }
    }

    /// What may follow the field, for refusals.
    fn usage(self) -> &'static str {
        match self {
            Self::Layer => "layer = <non-negative integer>",
            Self::Feature => "feature = <non-negative integer>",
            Self::Entity => "entity = <string>",
            Self::Relation => "relation {= | != | LIKE} <string>",
            Self::Score => "score {= | != | < | <= | > | >=} <number>",
        }
    }
}

/// Which fields a row source has. A field outside the scope is refused
/// like an unknown one: the source has nothing to compare it against.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PredicateScope {
    /// The row source, for refusals (`EDGES`, `FEATURES`, …).
    source: &'static str,
    /// Accepted spellings and the field each one names.
    fields: &'static [(&'static str, PredicateField)],
}

impl PredicateScope {
    /// `SELECT … FROM EDGES`, `DELETE FROM EDGES`, `UPDATE EDGES`.
    pub(crate) const EDGES: Self = Self {
        source: "EDGES",
        fields: &[
            ("layer", PredicateField::Layer),
            ("feature", PredicateField::Feature),
            ("entity", PredicateField::Entity),
            ("relation", PredicateField::Relation),
            ("score", PredicateField::Score),
            ("confidence", PredicateField::Score),
        ],
    };

    /// `SELECT … FROM FEATURES`.
    pub(crate) const FEATURES: Self = Self {
        source: "FEATURES",
        fields: &[
            ("layer", PredicateField::Layer),
            ("feature", PredicateField::Feature),
            ("entity", PredicateField::Entity),
            ("token", PredicateField::Entity),
        ],
    };

    /// `SHOW FEATURES <layer> WHERE …` — the layer is positional.
    pub(crate) const SHOW_FEATURES: Self = Self {
        source: "SHOW FEATURES",
        fields: &[
            ("token", PredicateField::Entity),
            ("entity", PredicateField::Entity),
            ("score", PredicateField::Score),
            ("confidence", PredicateField::Score),
        ],
    };

    /// `SELECT … FROM ENTITIES`.
    pub(crate) const ENTITIES: Self = Self {
        source: "ENTITIES",
        fields: &[
            ("layer", PredicateField::Layer),
            ("entity", PredicateField::Entity),
            ("token", PredicateField::Entity),
        ],
    };

    fn resolve(&self, spelling: &str) -> Option<PredicateField> {
        self.fields
            .iter()
            .find(|(name, _)| *name == spelling)
            .map(|(_, field)| *field)
    }

    fn spellings(&self) -> String {
        self.fields
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// How a `relation` condition compares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RelationOp {
    Eq,
    Neq,
    Like,
}

/// A validated `relation` condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RelationPredicate<'a> {
    pub op: RelationOp,
    pub wanted: &'a str,
}

impl RelationPredicate<'_> {
    /// Does a feature's relation label satisfy the condition? An
    /// unlabelled feature (`""`) satisfies none of them — including
    /// `!=`, which selects features labelled *something else*, not
    /// features with no label.
    pub(crate) fn matches(&self, label: &str) -> bool {
        if label.is_empty() {
            return false;
        }
        let label = label.to_lowercase();
        let wanted = self.wanted.to_lowercase();
        match self.op {
            RelationOp::Eq => overlaps(&label, &wanted),
            RelationOp::Neq => !overlaps(&label, &wanted),
            RelationOp::Like => like(&label, &wanted),
        }
    }
}

/// Relation equality is substring overlap in either direction, so
/// `capital` addresses a feature labelled `capital-of` and the reverse.
fn overlaps(label: &str, wanted: &str) -> bool {
    label.contains(wanted) || wanted.contains(label)
}

/// SQL-style `LIKE` with a leading and/or trailing `%`.
fn like(label: &str, pattern: &str) -> bool {
    let needle = pattern.trim_matches(LIKE_WILDCARD);
    if needle.is_empty() {
        return true;
    }
    match (
        pattern.starts_with(LIKE_WILDCARD),
        pattern.ends_with(LIKE_WILDCARD),
    ) {
        (true, true) => label.contains(needle),
        (true, false) => label.ends_with(needle),
        (false, true) => label.starts_with(needle),
        (false, false) => label == needle,
    }
}

/// How a `score` condition compares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScoreOp {
    Eq,
    Neq,
    Gt,
    Lt,
    Gte,
    Lte,
}

/// A validated `score` / `confidence` condition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ScorePredicate {
    pub op: ScoreOp,
    pub threshold: f32,
}

impl ScorePredicate {
    pub(crate) fn matches(&self, score: f32) -> bool {
        let t = self.threshold;
        match self.op {
            ScoreOp::Gt => score > t,
            ScoreOp::Lt => score < t,
            ScoreOp::Gte => score >= t,
            ScoreOp::Lte => score <= t,
            ScoreOp::Eq => (score - t).abs() < SCORE_EQ_TOLERANCE,
            ScoreOp::Neq => (score - t).abs() >= SCORE_EQ_TOLERANCE,
        }
    }
}

/// A `WHERE` clause every condition of which was understood.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct EdgePredicate<'a> {
    pub layer: Option<usize>,
    pub feature: Option<usize>,
    pub entity: Option<&'a str>,
    pub relation: Option<RelationPredicate<'a>>,
    pub score: Option<ScorePredicate>,
}

impl<'a> EdgePredicate<'a> {
    /// Validate `conditions` against `scope`. Every condition must name
    /// a field the scope has, at most once, with an operator and a value
    /// type that field supports — otherwise the whole clause is refused.
    /// Reads nothing but its arguments.
    pub(crate) fn parse(
        conditions: &'a [Condition],
        scope: PredicateScope,
    ) -> Result<Self, LqlError> {
        let mut predicate = Self::default();
        let mut seen: Vec<PredicateField> = Vec::new();

        for condition in conditions {
            let field = scope.resolve(&condition.field).ok_or_else(|| {
                refusal(format!(
                    "`{}` is not a field of {}; a WHERE clause here may use: {}",
                    condition.field,
                    scope.source,
                    scope.spellings()
                ))
            })?;
            if seen.contains(&field) {
                return Err(refusal(format!(
                    "`{}` is constrained more than once; one condition per field is \
                     supported (ranges are not evaluated)",
                    field.name()
                )));
            }
            seen.push(field);

            match field {
                PredicateField::Layer => {
                    predicate.layer = Some(index_value(field, condition)?);
                }
                PredicateField::Feature => {
                    predicate.feature = Some(index_value(field, condition)?);
                }
                PredicateField::Entity => {
                    require_op(field, condition, &[CompareOp::Eq])?;
                    predicate.entity = Some(string_value(field, condition)?);
                }
                PredicateField::Relation => {
                    let op = match condition.op {
                        CompareOp::Eq => RelationOp::Eq,
                        CompareOp::Neq => RelationOp::Neq,
                        CompareOp::Like => RelationOp::Like,
                        _ => return Err(unsupported(field, condition)),
                    };
                    predicate.relation = Some(RelationPredicate {
                        op,
                        wanted: string_value(field, condition)?,
                    });
                }
                PredicateField::Score => {
                    let op = match condition.op {
                        CompareOp::Eq => ScoreOp::Eq,
                        CompareOp::Neq => ScoreOp::Neq,
                        CompareOp::Gt => ScoreOp::Gt,
                        CompareOp::Lt => ScoreOp::Lt,
                        CompareOp::Gte => ScoreOp::Gte,
                        CompareOp::Lte => ScoreOp::Lte,
                        _ => return Err(unsupported(field, condition)),
                    };
                    let threshold = match condition.value {
                        Value::Number(n) => n as f32,
                        Value::Integer(n) => n as f32,
                        _ => return Err(unsupported(field, condition)),
                    };
                    predicate.score = Some(ScorePredicate { op, threshold });
                }
            }
        }
        Ok(predicate)
    }
}

fn refusal(reason: String) -> LqlError {
    LqlError::Execution(format!("unsupported WHERE clause: {reason}"))
}

fn unsupported(field: PredicateField, condition: &Condition) -> LqlError {
    refusal(format!(
        "`{} {} {}` is not evaluated; supported: {}",
        condition.field,
        op_symbol(&condition.op),
        value_kind(&condition.value),
        field.usage()
    ))
}

fn require_op(
    field: PredicateField,
    condition: &Condition,
    allowed: &[CompareOp],
) -> Result<(), LqlError> {
    let ok = allowed
        .iter()
        .any(|op| std::mem::discriminant(op) == std::mem::discriminant(&condition.op));
    if ok {
        Ok(())
    } else {
        Err(unsupported(field, condition))
    }
}

/// A string operand. The empty string is refused: every matcher here is
/// a substring test, and everything contains `""` — it would select all
/// rows while reading like a filter.
fn string_value(field: PredicateField, condition: &Condition) -> Result<&str, LqlError> {
    match &condition.value {
        Value::String(s) if s.is_empty() => Err(refusal(format!(
            "`{}` is compared with an empty string, which every value contains; \
             supported: {}",
            condition.field,
            field.usage()
        ))),
        Value::String(s) => Ok(s.as_str()),
        _ => Err(unsupported(field, condition)),
    }
}

/// A layer or feature index: `=` and a non-negative integer, nothing
/// else. A negative index addresses no slot, so it is refused rather
/// than turned into a sentinel.
fn index_value(field: PredicateField, condition: &Condition) -> Result<usize, LqlError> {
    require_op(field, condition, &[CompareOp::Eq])?;
    match condition.value {
        Value::Integer(n) => usize::try_from(n).map_err(|_| unsupported(field, condition)),
        _ => Err(unsupported(field, condition)),
    }
}

fn op_symbol(op: &CompareOp) -> &'static str {
    match op {
        CompareOp::Eq => "=",
        CompareOp::Neq => "!=",
        CompareOp::Gt => ">",
        CompareOp::Lt => "<",
        CompareOp::Gte => ">=",
        CompareOp::Lte => "<=",
        CompareOp::Like => "LIKE",
        CompareOp::In => "IN",
    }
}

fn value_kind(value: &Value) -> String {
    match value {
        Value::String(_) => "<string>".into(),
        Value::Number(n) => format!("{n}"),
        Value::Integer(n) => format!("{n}"),
        Value::List(_) => "(<list>)".into(),
    }
}
