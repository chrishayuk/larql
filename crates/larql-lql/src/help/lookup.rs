//! Find the page a `HELP <topic>;` asks for, or the nearest topics when
//! there is none.

use super::catalogue;
use super::topic::{ConceptTopic, HelpGroup, HelpPage, StatementTopic};

/// The most topics an unknown-topic error suggests by spelling.
const MAX_SUGGESTIONS: usize = 5;

/// A topic is suggested when it is within this many edits of the query...
const MIN_EDIT_BUDGET: usize = 2;
/// ...or within one edit per this many characters of the query, whichever
/// is larger.
const CHARS_PER_EDIT: usize = 3;

/// Every page, in overview order.
pub fn pages() -> impl Iterator<Item = HelpPage> {
    let statements = StatementTopic::ALL.iter().map(|t| HelpPage::Statement(*t));
    let concepts = ConceptTopic::ALL.iter().map(|t| HelpPage::Concept(*t));
    let groups = HelpGroup::ALL.iter().map(|g| HelpPage::Group(*g));
    statements.chain(concepts).chain(groups)
}

impl HelpPage {
    /// The name `HELP <name>;` is asked with.
    pub fn name(&self) -> &'static str {
        match self {
            HelpPage::Statement(t) => catalogue::statement(*t).name,
            HelpPage::Concept(t) => catalogue::concept(*t).name,
            HelpPage::Group(g) => catalogue::group(*g).name,
        }
    }

    /// The other words that find the page.
    pub fn aliases(&self) -> &'static [&'static str] {
        match self {
            HelpPage::Statement(t) => catalogue::statement(*t).aliases,
            HelpPage::Concept(t) => catalogue::concept(*t).aliases,
            HelpPage::Group(g) => catalogue::group(*g).aliases,
        }
    }

    /// The name and every alias.
    pub fn keys(&self) -> impl Iterator<Item = &'static str> {
        std::iter::once(self.name()).chain(self.aliases().iter().copied())
    }
}

/// A topic as it is compared: upper case, single spaces, no trailing `;`.
pub fn normalise(topic: &str) -> String {
    topic
        .trim()
        .trim_end_matches(';')
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_uppercase()
}

/// The page a topic names. A topic names a page by its name, by an alias,
/// or by leading words that only that page's names start with (`ADD` finds
/// `ADD LOW RANK`; `REMOVE` finds nothing, since two statements start so).
pub fn resolve(topic: &str) -> Result<HelpPage, Vec<&'static str>> {
    let query = normalise(topic);
    if let Some(page) = pages().find(|page| page.keys().any(|key| key == query)) {
        return Ok(page);
    }
    let mut starting = pages().filter(|page| page.keys().any(|key| starts_with_words(key, &query)));
    match (starting.next(), starting.next()) {
        (Some(only), None) => Ok(only),
        _ => Err(nearest(&query)),
    }
}

/// Whether `key` begins with all of `words` and then goes on.
fn starts_with_words(key: &str, words: &str) -> bool {
    key.strip_prefix(words)
        .is_some_and(|rest| rest.starts_with(' '))
}

/// The topics nearest to a query that names none: every page whose name
/// starts with the query's words, then the closest names by spelling.
pub fn nearest(query: &str) -> Vec<&'static str> {
    let query = normalise(query);
    let mut names: Vec<&'static str> = pages()
        .filter(|page| page.keys().any(|key| starts_with_words(key, &query)))
        .map(|page| page.name())
        .collect();
    let budget = MIN_EDIT_BUDGET.max(query.chars().count() / CHARS_PER_EDIT);
    let mut by_spelling: Vec<(usize, &'static str)> = pages()
        .filter_map(|page| {
            let distance = page.keys().map(|key| edit_distance(key, &query)).min()?;
            (distance <= budget).then_some((distance, page.name()))
        })
        .collect();
    by_spelling.sort_by_key(|(distance, _)| *distance);
    for (_, name) in by_spelling.into_iter().take(MAX_SUGGESTIONS) {
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

/// The word of `candidates` closest in spelling to `word`, when one is close
/// enough to be what was meant.
pub fn closest<'a>(word: &str, candidates: &[&'a str]) -> Option<&'a str> {
    let word = normalise(word);
    let budget = MIN_EDIT_BUDGET.max(word.chars().count() / CHARS_PER_EDIT);
    candidates
        .iter()
        .map(|candidate| (edit_distance(candidate, &word), *candidate))
        .filter(|(distance, _)| *distance <= budget)
        .min_by_key(|(distance, _)| *distance)
        .map(|(_, candidate)| candidate)
}

/// Levenshtein distance between two strings, by character.
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let substitution = diagonal + usize::from(ca != *cb);
            diagonal = row[j + 1];
            row[j + 1] = substitution.min(row[j] + 1).min(diagonal + 1);
        }
    }
    row[b.len()]
}
