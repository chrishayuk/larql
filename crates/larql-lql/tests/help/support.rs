//! Shared helpers: run a statement, and iterate the catalogue.

use larql_lql::help::{catalogue, StatementHelp, StatementTopic};
use larql_lql::{parse, LqlError, Session, Statement};

/// Parse, panicking with the source on a parse error.
pub fn parsed(source: &str) -> Statement {
    parse(source).unwrap_or_else(|e| panic!("parse {source}: {e}"))
}

/// The message of the parse error `source` must produce.
pub fn parse_error(source: &str) -> String {
    match parse(source) {
        Ok(statement) => panic!("{source} parsed as {statement:?}"),
        Err(e) => e.to_string(),
    }
}

pub fn execute(session: &mut Session, source: &str) -> Result<Vec<String>, LqlError> {
    session.execute(&parsed(source))
}

/// The lines `source` prints in a fresh, unbound session.
pub fn help(source: &str) -> Vec<String> {
    execute(&mut Session::new(), source).unwrap_or_else(|e| panic!("{source}: {e}"))
}

/// Every statement page with its topic, in overview order.
pub fn statement_pages() -> impl Iterator<Item = (StatementTopic, &'static StatementHelp)> {
    StatementTopic::ALL
        .iter()
        .map(|topic| (*topic, catalogue::statement(*topic)))
}
