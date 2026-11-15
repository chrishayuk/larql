//! Drift protection: the catalogue against the parser and against itself.

use std::collections::BTreeSet;

use larql_lql::help::{catalogue, lookup, render, ConceptTopic, HelpGroup, HelpPage};
use larql_lql::{CapabilityProfile, Statement};
use larql_vindex::format::vindex3::weight_patch::{MAX_PATCH_ELEMENTS, MAX_STRUCTURED_RANK};

use crate::support::*;

#[test]
fn every_example_parses_as_the_statement_its_page_documents() {
    let mut wrong = Vec::new();
    for (topic, page) in statement_pages() {
        assert!(!page.examples.is_empty(), "{} has no example", page.name);
        for example in page.examples {
            match larql_lql::parse(example) {
                Ok(statement) if statement.help_topic() == topic => {}
                Ok(statement) => wrong.push(format!(
                    "{example}: parsed as {:?}, on the page of {topic:?}",
                    statement.help_topic()
                )),
                Err(e) => wrong.push(format!("{example}: {e}")),
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn a_page_is_named_by_its_statements_verb() {
    // `HELP <verb>;` must find the page of the statement a refusal names by
    // that verb. The pipe is the exception: its verb is not a word.
    for (_, page) in statement_pages() {
        let statement = parsed(page.examples[0]);
        if matches!(statement, Statement::Pipe { .. }) {
            continue;
        }
        assert_eq!(page.name, statement.verb(), "{}", page.examples[0]);
    }
}

#[test]
fn every_page_has_what_a_caller_needs() {
    for (_, page) in statement_pages() {
        let name = page.name;
        assert!(!page.syntax.is_empty(), "{name}: no syntax");
        assert!(!page.does.is_empty(), "{name}: no description");
        assert!(!page.needs.is_empty(), "{name}: no requirement stated");
        assert!(!page.effect.is_empty(), "{name}: no effect stated");
        assert!(!page.related.is_empty(), "{name}: no related page");
    }
    for topic in ConceptTopic::ALL {
        let page = catalogue::concept(*topic);
        assert!(
            !page.summary.is_empty() && !page.body.is_empty(),
            "{}",
            page.name
        );
    }
}

#[test]
fn every_page_is_listed_by_the_overview_and_found_by_its_name() {
    let overview = help("HELP;");
    let listed = |line: &str| overview.iter().any(|l| l.trim() == line);
    for (topic, page) in statement_pages() {
        let line = if page.skeleton().starts_with(page.name) {
            page.skeleton().to_string()
        } else {
            format!("{}   (HELP {};)", page.skeleton(), page.name)
        };
        assert!(listed(&line), "HELP; does not list {}", page.name);
        assert_eq!(
            lookup::resolve(page.name),
            Ok(HelpPage::Statement(topic)),
            "{}",
            page.name
        );
    }
    for topic in ConceptTopic::ALL {
        let page = catalogue::concept(*topic);
        assert!(
            overview.iter().any(|l| l.trim().starts_with(page.name)),
            "HELP; does not list {}",
            page.name
        );
        assert_eq!(lookup::resolve(page.name), Ok(HelpPage::Concept(*topic)));
    }
    for group in HelpGroup::ALL {
        let page = catalogue::group(*group);
        assert!(
            listed(&format!("{} (HELP {};)", page.title, page.name)),
            "HELP; has no heading for {}",
            page.name
        );
        assert_eq!(lookup::resolve(page.name), Ok(HelpPage::Group(*group)));
    }
}

#[test]
fn the_overview_lists_groups_in_the_order_of_an_editing_loop() {
    let overview = help("HELP;");
    let position = |group: HelpGroup| {
        let heading = format!("(HELP {};)", catalogue::group(group).name);
        overview
            .iter()
            .position(|l| l.ends_with(&heading))
            .unwrap_or_else(|| panic!("no heading {heading}"))
    };
    let positions: Vec<usize> = HelpGroup::ALL.iter().map(|g| position(*g)).collect();
    assert!(positions.windows(2).all(|w| w[0] < w[1]), "{positions:?}");
    assert_eq!(
        HelpGroup::ALL,
        [
            HelpGroup::Bind,
            HelpGroup::Inspect,
            HelpGroup::Run,
            HelpGroup::Weights,
            HelpGroup::Patches,
            HelpGroup::Compile,
            HelpGroup::Verify,
            HelpGroup::Abandon,
            HelpGroup::Knowledge,
            HelpGroup::Process,
        ]
    );
}

#[test]
fn every_page_renders_through_the_statement() {
    for page in lookup::pages() {
        let lines = help(&format!("HELP {};", page.name()));
        assert!(lines.len() > 1, "{}: {lines:?}", page.name());
        assert!(lines[0].contains(page.name()) || matches!(page, HelpPage::Group(_)));
        for alias in page.aliases() {
            assert_eq!(
                help(&format!("HELP \"{alias}\";")),
                lines,
                "alias {alias} of {}",
                page.name()
            );
        }
    }
}

#[test]
fn a_statement_page_has_every_section_it_has_content_for() {
    let page = help("HELP REMOVE DIRECTION;").join("\n");
    for heading in [
        "Syntax:",
        "What it does:",
        "Examples:",
        "Requires:",
        "When it takes effect:",
        "Operand file:",
        "Refusals and limits:",
        "See also:",
    ] {
        assert!(page.contains(heading), "no {heading} in:\n{page}");
    }
    // A page with no operand file has no such section.
    assert!(!help("HELP COMPILE;").join("\n").contains("Operand file:"));
}

#[test]
fn names_and_aliases_find_exactly_one_page() {
    let mut seen = BTreeSet::new();
    for page in lookup::pages() {
        for key in page.keys() {
            assert_eq!(key, lookup::normalise(key), "{key} is not in normal form");
            assert!(seen.insert(key), "{key} names two pages");
        }
    }
}

#[test]
fn every_related_page_exists() {
    let dangling: Vec<String> = statement_pages()
        .flat_map(|(_, page)| page.related.iter().map(move |r| (page.name, *r)))
        .chain(ConceptTopic::ALL.iter().flat_map(|topic| {
            let page = catalogue::concept(*topic);
            page.related.iter().map(move |r| (page.name, *r))
        }))
        .filter(|(_, related)| lookup::resolve(related).is_err())
        .map(|(name, related)| format!("{name} -> {related}"))
        .collect();
    assert!(dangling.is_empty(), "{dangling:#?}");
}

#[test]
fn every_help_topic_a_page_mentions_exists() {
    // "HELP BATCH;" inside a page is an instruction to the reader: it must work.
    let mut dangling = Vec::new();
    for page in lookup::pages() {
        for line in render::page(page).iter().chain(&help("HELP;")) {
            let mut rest = line.as_str();
            while let Some(at) = rest.find("HELP ") {
                let after = &rest[at + "HELP ".len()..];
                let Some(end) = after.find(';') else { break };
                let topic = &after[..end];
                let placeholder = topic.starts_with('<') || topic.starts_with('[');
                if !placeholder && lookup::resolve(topic).is_err() {
                    dangling.push(format!("{}: HELP {topic};", page.name()));
                }
                rest = &after[end..];
            }
        }
    }
    assert!(dangling.is_empty(), "{dangling:#?}");
}

#[test]
fn the_limits_a_page_quotes_are_the_engines() {
    let update = help("HELP UPDATE WEIGHTS;").join("\n");
    assert!(update.contains(&MAX_PATCH_ELEMENTS.to_string()), "{update}");
    let compile = help("HELP COMPILE;").join("\n");
    assert!(
        compile.contains(&MAX_PATCH_ELEMENTS.to_string()),
        "{compile}"
    );
    let low_rank = help("HELP ADD LOW RANK;").join("\n");
    assert!(
        low_rank.contains(&format!("at most {MAX_STRUCTURED_RANK}")),
        "{low_rank}"
    );
    let infer = help("HELP INFER;").join("\n");
    assert!(
        infer.contains(&format!(
            "GENERATE above {}",
            CapabilityProfile::PUBLIC_EXPLORER_MAX_GENERATE
        )),
        "{infer}"
    );
}

#[test]
fn the_weight_edit_pages_say_what_an_agent_must_know() {
    for topic in ["UPDATE WEIGHTS", "REMOVE DIRECTION", "ADD LOW RANK"] {
        let page = help(&format!("HELP {topic};")).join("\n");
        // Visibility: what executes a staged op and what does not is held to the engine in `live_effect.rs`.
        assert!(page.contains("Staging writes nothing to disk"), "{topic}");
        assert!(page.contains("COMPILE CURRENT INTO VINDEX"), "{topic}");
        assert!(page.contains("DISCARD WEIGHT EDITS"), "{topic}");
        assert!(page.contains("VINDEX3 container"), "{topic}");
    }
    let remove = help("HELP REMOVE DIRECTION;").join("\n");
    assert!(remove.contains(r#"{"bits": ["<hex>", ...], "origin": "<note>"}"#));
    assert!(remove.contains("16 lowercase hex digits"));
    let low_rank = help("HELP ADD LOW RANK;").join("\n");
    assert!(low_rank
        .contains(r#"{"rank": K, "a": ["<hex>", ...], "b": ["<hex>", ...], "origin": "<note>"}"#));
    let exit = help("HELP EXIT CODES;").join("\n");
    for code in [
        larql_lql::EXIT_OK,
        larql_lql::EXIT_ERROR,
        larql_lql::EXIT_CONTRACT_VIOLATED,
    ] {
        assert!(exit.contains(&format!("\n  {code}  ")), "no exit {code}");
    }
}

#[test]
fn the_batch_page_names_every_statement_that_writes() {
    // The write guard's list is `Statement::writes_artifact`; the page that
    // explains the guard must not fall behind it.
    let batch = help("HELP BATCH;").join("\n");
    for (_, page) in statement_pages() {
        let writes = page
            .examples
            .iter()
            .any(|example| parsed(example).writes_artifact());
        if writes {
            assert!(
                batch.contains(page.name),
                "HELP BATCH; does not name {}",
                page.name
            );
        }
    }
    // A pipe that contains a writer is one.
    assert!(parsed("STATS |> SAVE PATCH;").writes_artifact());
    assert!(batch.contains("a pipe containing one"), "{batch}");
}
