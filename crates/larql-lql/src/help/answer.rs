//! `HELP;` and `HELP <topic>;` as the executor runs them.

use crate::error::LqlError;

use super::{lookup, render};

/// The lines a `HELP` statement prints. It reads no session state and
/// writes nothing, so it answers the same with or without a bound backend.
/// A topic that names no page is an error listing the nearest ones: a typo
/// never silently returns the overview.
pub fn help_lines(topic: Option<&str>) -> Result<Vec<String>, LqlError> {
    let Some(topic) = topic else {
        return Ok(render::overview());
    };
    match lookup::resolve(topic) {
        Ok(page) => Ok(render::page(page)),
        Err(nearest) => Err(LqlError::UnknownHelpTopic {
            topic: lookup::normalise(topic),
            nearest: if nearest.is_empty() {
                String::new()
            } else {
                format!(" Nearest: {}.", nearest.join(", "))
            },
        }),
    }
}
