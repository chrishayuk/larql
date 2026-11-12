//! The SET side of `UPDATE EDGES`, validated whole before anything runs.
//!
//! An assignment the statement cannot apply used to be dropped: an unknown
//! field, or a value of the wrong type, changed nothing and the statement
//! still answered `Updated N features`. Every assignment is now checked
//! first, and one that cannot be applied refuses the statement by name.
//!
//! `UPDATE EDGES` rewrites a feature's metadata record. A record has two
//! settable facts, each with two spellings:
//!
//! - `target` / `top_token`: the text of the record's token;
//! - `confidence` / `c_score`: its score.

use crate::ast::{Assignment, Value};
use crate::error::LqlError;

/// The spellings of the token-text field.
const TARGET_FIELDS: &[&str] = &["target", "top_token"];
/// The spellings of the score field.
const CONFIDENCE_FIELDS: &[&str] = &["confidence", "c_score"];

/// A validated SET clause: what the statement changes on each record.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct UpdateSet {
    /// The new token text, if the clause sets it.
    pub target: Option<String>,
    /// The new score, if the clause sets it.
    pub confidence: Option<f32>,
}

impl UpdateSet {
    /// Check every assignment and collect what the clause sets.
    pub(crate) fn parse(set: &[Assignment]) -> Result<Self, LqlError> {
        let mut parsed = Self::default();
        for assignment in set {
            let field = assignment.field.as_str();
            if TARGET_FIELDS.contains(&field) {
                if parsed.target.is_some() {
                    return Err(set_twice(field, TARGET_FIELDS));
                }
                parsed.target = Some(target_value(field, &assignment.value)?);
            } else if CONFIDENCE_FIELDS.contains(&field) {
                if parsed.confidence.is_some() {
                    return Err(set_twice(field, CONFIDENCE_FIELDS));
                }
                parsed.confidence = Some(confidence_value(field, &assignment.value)?);
            } else {
                return Err(refused(format!(
                    "`{field}` is not a field UPDATE EDGES can set (settable: {}, {})",
                    TARGET_FIELDS.join(" / "),
                    CONFIDENCE_FIELDS.join(" / ")
                )));
            }
        }
        if parsed == Self::default() {
            return Err(refused("the SET clause sets nothing".into()));
        }
        Ok(parsed)
    }

    /// Write the clause onto one record.
    pub(crate) fn apply(&self, meta: &mut larql_vindex::FeatureMeta) {
        if let Some(target) = &self.target {
            meta.top_token = target.clone();
        }
        if let Some(confidence) = self.confidence {
            meta.c_score = confidence;
        }
    }
}

fn refused(reason: String) -> LqlError {
    LqlError::Execution(format!("UPDATE refused: {reason}. Nothing was updated."))
}

fn set_twice(field: &str, spellings: &[&str]) -> LqlError {
    refused(format!(
        "`{field}` is set more than once ({} name the same field)",
        spellings.join(" and ")
    ))
}

fn target_value(field: &str, value: &Value) -> Result<String, LqlError> {
    match value {
        Value::String(text) if !text.is_empty() => Ok(text.clone()),
        Value::String(_) => Err(refused(format!(
            "`{field}` cannot be set to an empty string"
        ))),
        other => Err(refused(format!(
            "`{field}` takes a string, and got {}",
            describe(other)
        ))),
    }
}

fn confidence_value(field: &str, value: &Value) -> Result<f32, LqlError> {
    let number = match value {
        Value::Number(n) => *n as f32,
        Value::Integer(n) => *n as f32,
        other => {
            return Err(refused(format!(
                "`{field}` takes a number, and got {}",
                describe(other)
            )))
        }
    };
    // A literal too large for f32 becomes infinite in the cast.
    if !number.is_finite() {
        return Err(refused(format!("`{field}` must be a finite number")));
    }
    Ok(number)
}

fn describe(value: &Value) -> &'static str {
    match value {
        Value::String(_) => "a string",
        Value::Number(_) | Value::Integer(_) => "a number",
        Value::List(_) => "a list",
    }
}
