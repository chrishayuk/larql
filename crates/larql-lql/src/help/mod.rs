//! In-language help: what `HELP;` and `HELP <topic>;` print.
//!
//! The catalogue cannot fall behind the grammar:
//!
//! - [`crate::ast::Statement::help_topic`] is an exhaustive match, so a new
//!   statement does not compile until it names a topic;
//! - [`catalogue::statement`] is an exhaustive match, so a topic does not
//!   compile until it has a page;
//! - `tests/help/catalogue.rs` parses every example of every page with the
//!   real parser and checks it is the statement the page documents, and
//!   that every page is listed by `HELP;` and found by `HELP <topic>;`.
//!
//! The `USE` banner and the parser's "expected ..." messages are built from
//! the same entries ([`render::vindex3_supported`], [`grammar`]).

mod answer;
pub mod catalogue;
pub mod entry;
pub mod grammar;
pub mod lookup;
pub mod render;
mod statement_topic;
pub mod topic;

pub use answer::help_lines;
pub use entry::{ConceptHelp, GroupHelp, Need, StatementHelp};
pub use topic::{ConceptTopic, HelpGroup, HelpPage, StatementTopic};
