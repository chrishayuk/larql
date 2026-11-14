//! How a topic finds its page, and what an unknown topic gets.

use larql_lql::help::{lookup, HelpGroup, HelpPage, StatementTopic};
use larql_lql::{LqlError, Session};

use crate::support::*;

fn unknown(source: &str) -> String {
    match execute(&mut Session::new(), source) {
        Err(e @ LqlError::UnknownHelpTopic { .. }) => e.to_string(),
        other => panic!("{source}: expected an unknown-topic error, got {other:?}"),
    }
}

#[test]
fn a_topic_is_case_insensitive_and_may_be_quoted() {
    let page = help("HELP REMOVE DIRECTION;");
    for spelling in [
        "help remove direction;",
        "Help Remove Direction",
        "HELP \"remove direction\";",
        "HELP 'REMOVE   DIRECTION';",
    ] {
        assert_eq!(help(spelling), page, "{spelling}");
    }
}

#[test]
fn a_topic_made_of_keywords_and_words_is_read_whole() {
    // VERIFY and PATCH are reserved words, LEDGER and DIRECTION are not.
    for (source, topic) in [
        ("HELP VERIFY LEDGER;", StatementTopic::VerifyLedger),
        ("HELP VERIFY PATCH;", StatementTopic::VerifyPatch),
        (
            "HELP SHOW COMPACT STATUS;",
            StatementTopic::ShowCompactStatus,
        ),
        ("HELP COMPILE CURRENT INTO VINDEX;", StatementTopic::Compile),
        ("HELP UPDATE EDGES;", StatementTopic::Update),
        ("HELP PIPE;", StatementTopic::Pipe),
    ] {
        let larql_lql::Statement::Help { topic: Some(words) } = parsed(source) else {
            panic!("{source} is not HELP <topic>");
        };
        assert_eq!(
            lookup::resolve(&words),
            Ok(HelpPage::Statement(topic)),
            "{source}"
        );
    }
}

#[test]
fn a_group_and_a_statement_of_one_word_are_told_apart() {
    assert_eq!(
        lookup::resolve("verify"),
        Ok(HelpPage::Group(HelpGroup::Verify))
    );
    assert_eq!(
        lookup::resolve("weights"),
        Ok(HelpPage::Group(HelpGroup::Weights))
    );
    assert_eq!(
        lookup::resolve("compile"),
        Ok(HelpPage::Statement(StatementTopic::Compile))
    );
    assert_eq!(
        lookup::resolve("update"),
        Ok(HelpPage::Statement(StatementTopic::Update))
    );
}

#[test]
fn leading_words_only_one_page_starts_with_find_it() {
    assert_eq!(
        lookup::resolve("EXTRACT"),
        Ok(HelpPage::Statement(StatementTopic::Extract))
    );
    assert_eq!(
        lookup::resolve("SHOW COMPACT"),
        Ok(HelpPage::Statement(StatementTopic::ShowCompactStatus))
    );
    // Aliases of one page count once.
    assert_eq!(
        lookup::resolve("COMPILE INTO"),
        Ok(HelpPage::Statement(StatementTopic::Compile))
    );
}

#[test]
fn leading_words_several_pages_start_with_are_an_error_naming_them_all() {
    let message = unknown("HELP REMOVE;");
    assert_eq!(
        message,
        "no help topic \"REMOVE\". Nearest: REMOVE DIRECTION, REMOVE PATCH. Run HELP; for every topic."
    );
    let show = unknown("HELP SHOW;");
    for name in [
        "SHOW COMPONENTS",
        "SHOW REPRESENTATIONS",
        "SHOW PROVENANCE",
        "SHOW AUTHORITY",
        "SHOW LAYERS",
        "SHOW MODELS",
        "SHOW RELATIONS",
        "SHOW FEATURES",
        "SHOW ENTITIES",
        "SHOW PATCHES",
        "SHOW COMPACT STATUS",
    ] {
        assert!(show.contains(name), "{name} missing from: {show}");
    }
}

#[test]
fn a_mistyped_topic_is_an_error_naming_the_nearest_never_the_overview() {
    assert_eq!(
        unknown("HELP REMOVE DIRECTON;"),
        "no help topic \"REMOVE DIRECTON\". Nearest: REMOVE DIRECTION. Run HELP; for every topic."
    );
    assert!(unknown("HELP VERIFY LEDGERS;").contains("Nearest: VERIFY LEDGER."));
    assert!(unknown("HELP EXIT CODEZ;").contains("EXIT CODES"));
    // An alias close in spelling suggests the page's own name.
    assert!(unknown("HELP SESSIONZ;").contains("BATCH"));
}

#[test]
fn a_topic_near_nothing_says_so_plainly() {
    assert_eq!(
        unknown("HELP zzzzzzzz;"),
        "no help topic \"ZZZZZZZZ\". Run HELP; for every topic."
    );
}

#[test]
fn suggestions_are_few_and_closest_first() {
    let nearest = lookup::nearest("SHOW LAYER");
    assert_eq!(nearest.first(), Some(&"SHOW LAYERS"), "{nearest:?}");
    assert!(nearest.len() <= 5, "{nearest:?}");
    // The same page is never suggested twice.
    let mut distinct = nearest.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), nearest.len(), "{nearest:?}");
}

#[test]
fn closest_names_a_word_only_when_one_is_close() {
    let words = ["DIRECTION", "PATCH"];
    assert_eq!(lookup::closest("directon", &words), Some("DIRECTION"));
    assert_eq!(lookup::closest("PATCHE", &words), Some("PATCH"));
    assert_eq!(lookup::closest("LEDGER", &words), None);
}
