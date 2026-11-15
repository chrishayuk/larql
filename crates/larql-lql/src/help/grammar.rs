//! What the catalogue says about how statements begin. The parser's error
//! messages read these, so a message that lists the valid statements or the
//! valid words after `REMOVE` is derived from the same entries `HELP` prints
//! and cannot fall behind them.

use super::catalogue;
use super::topic::StatementTopic;

/// The word every statement kind starts with, sorted, each once. Taken from
/// the first example of each page: an example is a real statement, checked
/// against the parser by the catalogue's tests.
pub fn leading_keywords() -> Vec<&'static str> {
    let mut words: Vec<&'static str> = StatementTopic::ALL
        .iter()
        .filter_map(|topic| catalogue::statement(*topic).examples.first())
        .filter_map(|example| example.split_whitespace().next())
        .map(|word| word.trim_end_matches(';'))
        .collect();
    words.sort_unstable();
    words.dedup();
    words
}

/// What may follow a leading keyword that begins more than one word of a
/// statement's name: after `REMOVE`, `DIRECTION` and `PATCH`; after `ADD`,
/// `LOW RANK`. Empty for a keyword that is a whole name by itself.
pub fn continuations(leading: &str) -> Vec<&'static str> {
    StatementTopic::ALL
        .iter()
        .map(|topic| catalogue::statement(*topic).name)
        .filter_map(|name| name.strip_prefix(leading)?.strip_prefix(' '))
        .collect()
}
