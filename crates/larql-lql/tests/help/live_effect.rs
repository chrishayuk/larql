//! What the help says about when a stored-weight op takes effect, held to what a session does.
//!
//! A staged op is executed by `INFER` and `TRACE` in the session that staged it. The pages once said the
//! opposite, because that was true when they were written. Three things keep them from saying it again:
//!
//! - the pages and the engine's replies name the executing statements with the same words
//!   ([`larql_lql::live_scope`]), so a page cannot name others;
//! - a session is run here, and what its replies and its `INFER` do is what the pages are required to say;
//! - no page may carry a sentence that says a staged op is not executed.

use larql_lql::help::{lookup, render};
use larql_lql::live_scope::{EXECUTED_BY, IN_EFFECT, NOTHING_IN_EFFECT, READS_STORED};
use larql_lql::Session;
use larql_vindex::format::vindex3::fixtures::{bf16_container_with_q8_pack, DENSE_VOCAB};

use crate::support::*;

/// The pages that stage a stored-weight op, withdraw one, or run the model.
const STAGING_PAGES: [&str; 3] = ["UPDATE WEIGHTS", "REMOVE DIRECTION", "ADD LOW RANK"];
/// Every page that must name the statements that execute a pending op.
const PAGES_NAMING_THE_EXECUTORS: [&str; 10] = [
    "UPDATE WEIGHTS",
    "REMOVE DIRECTION",
    "ADD LOW RANK",
    "WEIGHTS",
    "APPLY PATCH",
    "SHOW PATCHES",
    "DISCARD WEIGHT EDITS",
    "REMOVE PATCH",
    "ABANDON",
    "RUN",
];
/// What a page said when a staged op was not executed. None may come back.
const CLAIMS_OF_THE_OLD_BEHAVIOUR: [&str; 9] = [
    "NOT seen",
    "is NOT applied",
    "does not see",
    "do not see",
    "never see a staged",
    "still see the stored weights",
    "runs the unedited weights",
    "exists only in",
    "take effect only in a container",
];
/// A prompt of the fixture's vocabulary, and a delta large enough to move what INFER prints.
const PROMPT: &str = "[3] [17] [5] [42]";
const STAGE: &str = "UPDATE WEIGHTS SET DELTA = 64.0 WHERE TENSOR = \"1.mlp.down_proj.weight\" \
                     AND ROW = 5 AND COL IN (0, 1, 2, 3, 4, 5, 6, 7);";

/// A page as running text: its lines joined and its indentation dropped, so a sentence that wraps is found.
fn page(topic: &str) -> String {
    help(&format!("HELP {topic};"))
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// A session on the BF16 fixture with a tokenizer, so INFER runs.
fn inferable() -> (tempfile::TempDir, Session) {
    let dir = tempfile::tempdir().unwrap();
    let container = bf16_container_with_q8_pack(dir.path());
    let vocab: serde_json::Map<String, serde_json::Value> = (0..DENSE_VOCAB)
        .map(|i| (format!("[{i}]"), i.into()))
        .collect();
    let tokenizer = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[0]"}
    });
    std::fs::write(container.join("tokenizer.json"), tokenizer.to_string()).unwrap();
    let mut session = Session::new();
    let bind = format!(
        "USE \"{}\";",
        container.display().to_string().replace('\\', "\\\\")
    );
    execute(&mut session, &bind).unwrap();
    (dir, session)
}

/// The prediction lines of INFER, without its timing.
fn infer(session: &mut Session) -> Vec<String> {
    let mut lines = execute(session, &format!("INFER \"{PROMPT}\" TOP 20;")).unwrap();
    lines.pop();
    lines
}

/// The engine first: a staged op changes what INFER prints, the reply says so in the shared words, and a
/// discard restores the stored weights and says that. Then the pages, which must say what was just observed.
#[test]
fn the_pages_say_what_a_session_does_with_a_staged_op() {
    let (_dir, mut session) = inferable();
    let base = infer(&mut session);
    let staged = execute(&mut session, STAGE).unwrap().join("\n");
    assert_ne!(
        infer(&mut session),
        base,
        "a staged op is executed by INFER"
    );
    assert!(staged.contains(IN_EFFECT), "{staged}");
    assert!(staged.contains(READS_STORED), "{staged}");
    let discarded = execute(&mut session, "DISCARD WEIGHT EDITS;")
        .unwrap()
        .join("\n");
    assert!(discarded.contains(NOTHING_IN_EFFECT), "{discarded}");
    assert_eq!(infer(&mut session), base, "a discarded op is not");

    for topic in STAGING_PAGES {
        let page = page(topic);
        assert!(
            page.contains(&format!(
                "{EXECUTED_BY} in this session compute with the patched weights"
            )),
            "{topic} does not say {EXECUTED_BY} compute with a staged op:\n{page}"
        );
        assert!(
            page.contains(&format!(
                "{READS_STORED} ... CURRENT still read the stored weights"
            )),
            "{topic}:\n{page}"
        );
        assert!(
            page.contains(&format!(
                "{EXECUTED_BY} are refused, not run on the stored weights"
            )),
            "{topic}:\n{page}"
        );
    }
    let infer_page = page("INFER");
    assert!(
        infer_page.contains("stored-weight op") && infer_page.contains("is executed"),
        "{infer_page}"
    );
    let discard_page = page("DISCARD WEIGHT EDITS");
    assert!(
        discard_page.contains(&format!(
            "{EXECUTED_BY} stop computing with the discarded ops"
        )),
        "{discard_page}"
    );
}

/// The pages of INFER and TRACE say they need a tokenizer, and `banner.rs` holds that to a container without
/// one. It stays true with an op pending: the tokenizer is asked for before the op is, so the refusal a
/// reader meets is the one the page names.
#[test]
fn without_a_tokenizer_a_pending_op_does_not_change_what_infer_and_trace_refuse_for() {
    let dir = tempfile::tempdir().unwrap();
    let container = bf16_container_with_q8_pack(dir.path());
    let mut session = Session::new();
    let bind = format!(
        "USE \"{}\";",
        container.display().to_string().replace('\\', "\\\\")
    );
    execute(&mut session, &bind).unwrap();
    let staged = execute(&mut session, STAGE).unwrap().join("\n");
    assert!(staged.contains(IN_EFFECT), "{staged}");
    for statement in [
        format!("INFER \"{PROMPT}\" TOP 3;"),
        format!("TRACE \"{PROMPT}\";"),
    ] {
        let refusal = execute(&mut session, &statement).unwrap_err().to_string();
        assert!(refusal.contains("tokenizer"), "{statement}: {refusal}");
    }
}

#[test]
fn every_page_about_pending_ops_names_the_statements_that_execute_them() {
    for topic in PAGES_NAMING_THE_EXECUTORS {
        let page = page(topic);
        assert!(page.contains(EXECUTED_BY), "{topic}:\n{page}");
    }
    // TRACE and INFER are those statements: each page says what it does with a pending op.
    for topic in ["INFER", "TRACE"] {
        let page = page(topic);
        assert!(
            page.contains("pending") && page.contains("stored-weight op"),
            "{topic}:\n{page}"
        );
    }
}

#[test]
fn no_page_says_a_staged_op_is_not_executed() {
    let mut stale = Vec::new();
    let overview = help("HELP;");
    for page in lookup::pages() {
        let text = render::page(page).join(" ");
        for claim in CLAIMS_OF_THE_OLD_BEHAVIOUR {
            if text.contains(claim) {
                stale.push(format!("HELP {}; says \"{claim}\"", page.name()));
            }
        }
    }
    for claim in CLAIMS_OF_THE_OLD_BEHAVIOUR {
        if overview.join(" ").contains(claim) {
            stale.push(format!("HELP; says \"{claim}\""));
        }
    }
    assert!(stale.is_empty(), "{stale:#?}");
}
