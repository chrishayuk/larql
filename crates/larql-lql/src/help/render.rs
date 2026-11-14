//! Turn catalogue entries into the lines `HELP` prints.

use super::catalogue;
use super::entry::{ConceptHelp, Need, StatementHelp};
use super::topic::{ConceptTopic, HelpGroup, HelpPage, StatementTopic};

/// Indentation of a line under a heading.
const INDENT: &str = "  ";
/// The column the overview starts a process topic's summary at.
const CONCEPT_NAME_WIDTH: usize = 12;

/// How the overview says to ask for more.
const MORE: &[&str] = &[
    "More: HELP <statement>; prints one page: syntax, examples, what it requires, when its effect",
    "is visible, and what it refuses. Name the statement by its leading keywords, for example",
    "HELP REMOVE DIRECTION; or HELP COMPILE;. HELP <group>; prints one group, for example",
    "HELP WEIGHTS;. An unknown topic is an error that lists the nearest ones.",
];

/// The statements of a group, in overview order.
fn statements_in(group: HelpGroup) -> impl Iterator<Item = &'static StatementHelp> {
    StatementTopic::ALL
        .iter()
        .map(|topic| catalogue::statement(*topic))
        .filter(move |entry| entry.group == group)
}

/// The process topics of a group, in overview order.
fn concepts_in(group: HelpGroup) -> impl Iterator<Item = &'static ConceptHelp> {
    ConceptTopic::ALL
        .iter()
        .map(|topic| catalogue::concept(*topic))
        .filter(move |entry| entry.group == group)
}

/// How the overview lists one statement: its skeleton, and its name when
/// the skeleton does not begin with it.
fn overview_line(entry: &StatementHelp) -> String {
    if entry.skeleton().starts_with(entry.name) {
        format!("{INDENT}{}", entry.skeleton())
    } else {
        format!("{INDENT}{}   (HELP {};)", entry.skeleton(), entry.name)
    }
}

fn concept_line(entry: &ConceptHelp) -> String {
    format!(
        "{INDENT}{:<CONCEPT_NAME_WIDTH$} {}",
        entry.name, entry.summary
    )
}

/// The statements and process topics of one group, one line each.
fn group_members(group: HelpGroup) -> Vec<String> {
    statements_in(group)
        .map(overview_line)
        .chain(concepts_in(group).map(concept_line))
        .collect()
}

/// `HELP;`: every statement's syntax skeleton, by group, and how to ask for
/// more.
pub fn overview() -> Vec<String> {
    let mut out = vec![
        "LQL statements, grouped in the order an editing loop uses them.".to_string(),
        "Notation: UPPERCASE is a keyword (case-insensitive), \"<x>\" a quoted string, <n> a number,"
            .to_string(),
        "[x] optional, {a | b} a choice, ... a repeat. Every statement ends with \";\".".to_string(),
    ];
    for group in HelpGroup::ALL {
        let entry = catalogue::group(*group);
        out.push(String::new());
        out.push(format!("{} (HELP {};)", entry.title, entry.name));
        out.extend(group_members(*group));
    }
    out.push(String::new());
    out.extend(MORE.iter().map(|line| line.to_string()));
    out
}

/// Append a headed section, unless it is empty.
fn section<'a>(
    out: &mut Vec<String>,
    heading: &str,
    lines: impl IntoIterator<Item = &'a str>,
    bullet: &str,
) {
    let mut lines = lines.into_iter().peekable();
    if lines.peek().is_none() {
        return;
    }
    out.push(String::new());
    out.push(format!("{heading}:"));
    out.extend(lines.map(|line| {
        // A line that is already indented continues the bullet above it.
        if line.starts_with(' ') {
            format!("{INDENT}{line}")
        } else {
            format!("{INDENT}{bullet}{line}")
        }
    }));
}

fn see_also(out: &mut Vec<String>, related: &[&str]) {
    if !related.is_empty() {
        out.push(String::new());
        out.push(format!("See also: {}", related.join(", ")));
    }
}

fn statement_page(entry: &StatementHelp) -> Vec<String> {
    let group = catalogue::group(entry.group);
    let mut out = vec![format!("{}  ({})", entry.name, group.title)];
    section(&mut out, "Syntax", entry.syntax.iter().copied(), "");
    section(&mut out, "What it does", entry.does.iter().copied(), "");
    section(&mut out, "Examples", entry.examples.iter().copied(), "");
    section(
        &mut out,
        "Requires",
        entry.needs.iter().map(Need::text),
        "- ",
    );
    section(
        &mut out,
        "When it takes effect",
        entry.effect.iter().copied(),
        "",
    );
    section(&mut out, "Operand file", entry.operands.iter().copied(), "");
    section(
        &mut out,
        "Refusals and limits",
        entry.refusals.iter().copied(),
        "- ",
    );
    see_also(&mut out, entry.related);
    out
}

fn concept_page(entry: &ConceptHelp) -> Vec<String> {
    let group = catalogue::group(entry.group);
    let mut out = vec![format!("{}  ({})", entry.name, group.title), String::new()];
    out.extend(entry.body.iter().map(|line| line.to_string()));
    see_also(&mut out, entry.related);
    out
}

fn group_page(group: HelpGroup) -> Vec<String> {
    let entry = catalogue::group(group);
    let mut out = vec![entry.title.to_string(), String::new()];
    out.extend(entry.intro.iter().map(|line| line.to_string()));
    out.push(String::new());
    out.extend(group_members(group));
    out.push(String::new());
    out.push("HELP <statement>; prints the full page of any statement above.".to_string());
    out
}

/// The lines of one page.
pub fn page(page: HelpPage) -> Vec<String> {
    match page {
        HelpPage::Statement(topic) => statement_page(catalogue::statement(topic)),
        HelpPage::Concept(topic) => concept_page(catalogue::concept(topic)),
        HelpPage::Group(group) => group_page(group),
    }
}

/// The names of the statements for which `keep` holds, in overview order.
fn statement_names(keep: impl Fn(&StatementHelp) -> bool) -> String {
    StatementTopic::ALL
        .iter()
        .map(|topic| catalogue::statement(*topic))
        .filter(|entry| keep(entry))
        .map(|entry| entry.name)
        .collect::<Vec<_>>()
        .join(", ")
}

/// The statements a VINDEX3 binding serves: every statement whose page does
/// not say it needs a VINDEX2 vindex. This is what the `USE` banner, `STATS`
/// and the VINDEX3 capability refusal print, so they cannot name a statement
/// the catalogue does not document, or omit one it does.
pub fn vindex3_supported() -> String {
    statement_names(|entry| !entry.needs(Need::Vindex2))
}

/// The statements a VINDEX3 binding serves only with a tokenizer.
pub fn vindex3_needs_tokenizer() -> String {
    statement_names(|entry| entry.needs(Need::Tokenizer) && !entry.needs(Need::Vindex2))
}
