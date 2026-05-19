//! LQL REPL — interactive shell with history, arrow keys, and line editing.

use crate::executor::Session;
use crate::parser;

use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

const BANNER: &str = r#"
   ╦   ╔═╗ ╦═╗ ╔═╗ ╦
   ║   ╠═╣ ╠╦╝ ║═╬╗║
   ╩═╝ ╩ ╩ ╩╚═ ╚═╝╚╩═╝
   Lazarus Query Language v0.1
"#;

const PROMPT: &str = "larql> ";
const CONTINUATION: &str = "   ... ";

/// History file location — stored in ~/.larql_history
fn history_path() -> Option<std::path::PathBuf> {
    dirs_or_home().map(|d| d.join(".larql_history"))
}

fn dirs_or_home() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(std::path::PathBuf::from)
}

/// Run the interactive REPL.
pub fn run_repl() {
    println!("{BANNER}");

    let mut session = Session::new();
    let mut rl = match DefaultEditor::new() {
        Ok(rl) => rl,
        Err(e) => {
            eprintln!("Failed to initialise line editor: {e}");
            eprintln!("Falling back to basic mode.");
            run_repl_basic();
            return;
        }
    };

    // Load history
    if let Some(ref path) = history_path() {
        let _ = rl.load_history(path);
    }

    let mut statement_buf = String::new();

    loop {
        let prompt = if statement_buf.is_empty() {
            PROMPT
        } else {
            CONTINUATION
        };

        match rl.readline(prompt) {
            Ok(line) => match classify_input(&line, &mut statement_buf) {
                LineAction::Exit => break,
                LineAction::Clear => {
                    print!("\x1B[2J\x1B[1;1H");
                    use std::io::Write;
                    std::io::stdout().flush().ok();
                    continue;
                }
                LineAction::Help => {
                    print_help();
                    continue;
                }
                LineAction::Skip | LineAction::NeedMore => continue,
                LineAction::Execute(input) => {
                    if input.is_empty() {
                        continue;
                    }
                    let _ = rl.add_history_entry(&input);
                    execute_and_print(&mut session, &input);
                }
            },
            Err(ReadlineError::Interrupted) => {
                // Ctrl-C: clear current buffer
                if !statement_buf.is_empty() {
                    statement_buf.clear();
                    println!("(cancelled)");
                }
            }
            Err(ReadlineError::Eof) => break, // Ctrl-D
            Err(e) => {
                eprintln!("Read error: {e}");
                break;
            }
        }
    }

    // Save history
    if let Some(ref path) = history_path() {
        let _ = rl.save_history(path);
    }

    println!("Goodbye.");
}

/// Basic fallback REPL without line editing (used if rustyline fails).
fn run_repl_basic() {
    use std::io::{self, BufRead, Write};

    let mut session = Session::new();
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut line_buf = String::new();
    let mut statement_buf = String::new();

    loop {
        if statement_buf.is_empty() {
            print!("{PROMPT}");
        } else {
            print!("{CONTINUATION}");
        }
        io::stdout().flush().ok();

        line_buf.clear();
        match reader.read_line(&mut line_buf) {
            Ok(0) => break,
            Err(_) => break,
            Ok(_) => {}
        }

        match classify_input(&line_buf, &mut statement_buf) {
            LineAction::Exit => break,
            LineAction::Clear => {
                print!("\x1B[2J\x1B[1;1H");
                use std::io::Write;
                std::io::stdout().flush().ok();
                continue;
            }
            LineAction::Help => {
                print_help();
                continue;
            }
            LineAction::Skip | LineAction::NeedMore => continue,
            LineAction::Execute(input) => {
                if input.is_empty() {
                    continue;
                }
                execute_and_print(&mut session, &input);
            }
        }
    }
    println!("Goodbye.");
}

/// Run a single LQL statement (non-interactive).
pub fn run_statement(input: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let stmt = parser::parse(input)?;
    let mut session = Session::new();
    Ok(session.execute(&stmt)?)
}

/// Run a batch of LQL statements from a file or string.
pub fn run_batch(input: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut session = Session::new();
    let mut all_output = Vec::new();

    // Split on semicolons, handling strings
    for stmt_text in split_statements(input) {
        // Strip leading comment lines so "-- comment\nSTATS;" isn't skipped
        let trimmed: String = stmt_text
            .lines()
            .filter(|l| !l.trim().starts_with("--"))
            .collect::<Vec<_>>()
            .join("\n");
        let trimmed = trimmed.trim();
        if trimmed.is_empty() {
            continue;
        }
        match parser::parse(trimmed) {
            Ok(stmt) => match session.execute(&stmt) {
                Ok(lines) => all_output.extend(lines),
                Err(e) => all_output.push(format!("Error: {e}")),
            },
            Err(e) => all_output.push(format!("Error: {e}")),
        }
    }

    Ok(all_output)
}

fn is_complete_statement(s: &str) -> bool {
    s.ends_with(';')
}

/// Outcome of feeding one raw line from the REPL editor into the input
/// classifier. The classifier owns the meta-command vocabulary and the
/// multi-line accumulation rule so both `run_repl` (rustyline-backed)
/// and `run_repl_basic` (stdin-fallback) share the policy.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum LineAction {
    /// `exit` / `quit` / `\q` at a fresh prompt — leave the loop.
    Exit,
    /// `clear` / `clear;` at a fresh prompt — clear screen, prompt again.
    Clear,
    /// `help` / `\h` / `\?` at a fresh prompt — print help, prompt again.
    Help,
    /// Empty line at a fresh prompt — re-prompt without touching the buffer.
    Skip,
    /// Line appended to the accumulation buffer but statement isn't
    /// complete yet — keep collecting lines.
    NeedMore,
    /// Buffer is complete; the inner string is the trimmed statement
    /// ready for `parser::parse`. The classifier has already cleared
    /// the accumulation buffer.
    Execute(String),
}

/// Feed one raw line from the editor into the REPL input state. Returns
/// the action the loop should take. Pure-ish: the only side effect is
/// mutating `buf` (the multi-line accumulation buffer the loop owns).
///
/// Behaviour:
/// - Meta commands (`exit` / `quit` / `\q` / `clear` / `help` / empty)
///   only fire when `buf` is empty — once a statement starts they're
///   treated as ordinary text so the user can include them in SQL.
/// - A statement is "complete" when it ends with `;` *or* when its
///   first word is one of the two semicolon-optional bare commands
///   (`STATS`, `SHOW MODELS`). Anything else returns `NeedMore`.
/// - On `Execute`, `buf` is drained — caller doesn't need to clear it.
pub(crate) fn classify_input(line: &str, buf: &mut String) -> LineAction {
    let trimmed = line.trim();
    if buf.is_empty() {
        match trimmed.to_lowercase().as_str() {
            "exit" | "quit" | "\\q" => return LineAction::Exit,
            "clear" | "clear;" => return LineAction::Clear,
            "help" | "\\h" | "\\?" => return LineAction::Help,
            "" => return LineAction::Skip,
            _ => {}
        }
    }

    if !buf.is_empty() {
        buf.push('\n');
    }
    buf.push_str(line);

    let trimmed_stmt = buf.trim();
    if !trimmed_stmt.ends_with(';')
        && !trimmed_stmt.to_uppercase().starts_with("STATS")
        && !trimmed_stmt.to_uppercase().starts_with("SHOW MODELS")
        && !is_complete_statement(trimmed_stmt)
    {
        return LineAction::NeedMore;
    }

    let stmt = std::mem::take(buf).trim().to_string();
    LineAction::Execute(stmt)
}

/// Parse `input`, execute against `session`, print results (or error)
/// to stdout/stderr. Shared by `run_repl` and `run_repl_basic` so the
/// error-path formatting is identical across the two front-ends.
fn execute_and_print(session: &mut Session, input: &str) {
    match parser::parse(input) {
        Ok(stmt) => match session.execute(&stmt) {
            Ok(lines) => {
                for line in &lines {
                    println!("{line}");
                }
            }
            Err(e) => {
                eprintln!("Error: {e}");
            }
        },
        Err(e) => {
            eprintln!("Error: {e}");
        }
    }
}

fn split_statements(input: &str) -> Vec<String> {
    let mut stmts = Vec::new();
    let mut current = String::new();
    let mut in_string = false;
    let mut string_char = '"';

    for ch in input.chars() {
        if in_string {
            current.push(ch);
            if ch == string_char {
                in_string = false;
            }
        } else if ch == '"' || ch == '\'' {
            in_string = true;
            string_char = ch;
            current.push(ch);
        } else if ch == ';' {
            current.push(ch);
            stmts.push(std::mem::take(&mut current));
        } else {
            current.push(ch);
        }
    }

    let trimmed = current.trim();
    if !trimmed.is_empty() {
        stmts.push(current);
    }

    stmts
}

fn print_help() {
    println!(
        r#"
LQL Commands:

  Lifecycle:
    EXTRACT MODEL <id> INTO <path>;     Decompile model → vindex
    COMPILE <vindex> INTO MODEL <path>;  Recompile vindex → weights
    USE <path>;                          Set active vindex
    USE MODEL <id>;                      Set active model (live weights)

  Query (pure vindex, no model needed):
    WALK <prompt> [TOP n];               Feature scan for a token
    SELECT ... FROM EDGES WHERE ...;     Query edges
    DESCRIBE <entity>;                   Knowledge about an entity
    EXPLAIN WALK <prompt>;               Feature trace (no attention)
    EXPLAIN INFER <prompt>;              Feature trace (with attention)

  Inference (requires model weights):
    INFER <prompt> [TOP n] [COMPARE];    Full prediction with attention

  Mutation:
    INSERT INTO EDGES (...) VALUES (...); Add edge
    DELETE FROM EDGES WHERE ...;          Remove edges
    UPDATE EDGES SET ... WHERE ...;       Modify edges

  Introspection:
    SHOW RELATIONS;                      List relation types
    SHOW LAYERS;                         Layer summary
    SHOW FEATURES <layer>;               Feature details
    SHOW MODELS;                         List vindexes
    STATS;                               Summary stats

  Meta:
    clear                                Clear the screen
    help, \h, \?                         Show this help
    exit, quit, \q                       Exit REPL
"#
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Statement splitting ──

    #[test]
    fn split_single_statement() {
        let stmts = split_statements("STATS;");
        assert_eq!(stmts.len(), 1);
        assert_eq!(stmts[0].trim(), "STATS;");
    }

    #[test]
    fn split_multiple_statements() {
        let stmts = split_statements("STATS; SHOW MODELS;");
        assert_eq!(stmts.len(), 2);
    }

    #[test]
    fn split_preserves_strings_with_semicolons() {
        let stmts = split_statements(r#"WALK "hello; world" TOP 5;"#);
        assert_eq!(stmts.len(), 1);
        assert!(stmts[0].contains("hello; world"));
    }

    #[test]
    fn split_multiline() {
        let stmts = split_statements("STATS;\nSHOW MODELS;\nSHOW LAYERS;");
        assert_eq!(stmts.len(), 3);
    }

    #[test]
    fn split_empty_input() {
        let stmts = split_statements("");
        assert!(stmts.is_empty());
    }

    #[test]
    fn split_trailing_text_without_semicolon() {
        let stmts = split_statements("STATS; SHOW MODELS");
        assert_eq!(stmts.len(), 2);
    }

    // ── Completeness check ──

    #[test]
    fn is_complete_with_semicolon() {
        assert!(is_complete_statement("STATS;"));
    }

    #[test]
    fn is_not_complete_without_semicolon() {
        assert!(!is_complete_statement("STATS"));
    }

    #[test]
    fn is_not_complete_multiline_partial() {
        assert!(!is_complete_statement("SELECT *\n  FROM EDGES"));
    }

    // ── Batch execution ──

    #[test]
    fn batch_show_models_runs() {
        let result = run_batch("SHOW MODELS;").unwrap();
        // Should return at least a header line
        assert!(!result.is_empty());
    }

    #[test]
    fn batch_errors_are_captured() {
        let result = run_batch("STATS;").unwrap();
        // STATS without USE should produce an error line
        assert!(result.iter().any(|l| l.contains("Error")));
    }

    #[test]
    fn batch_multiple_statements() {
        let result = run_batch("SHOW MODELS; SHOW MODELS;").unwrap();
        // Two SHOW MODELS should produce output from both
        assert!(result.len() >= 4); // at least 2 headers + 2 separators
    }

    #[test]
    fn batch_comments_skipped() {
        let result = run_batch("-- comment\nSHOW MODELS;").unwrap();
        assert!(!result.is_empty());
        // Comment shouldn't produce error
        assert!(!result.iter().any(|l| l.contains("comment")));
    }

    #[test]
    fn batch_parse_error_captured() {
        let result = run_batch("FOOBAR;").unwrap();
        assert!(result.iter().any(|l| l.contains("Error")));
    }

    // ── run_statement ──

    #[test]
    fn run_statement_show_models() {
        let result = run_statement("SHOW MODELS;");
        assert!(result.is_ok());
    }

    #[test]
    fn run_statement_parse_error() {
        let result = run_statement("NOT A VALID STATEMENT;");
        assert!(result.is_err());
    }

    // ── Comment / whitespace handling ──────────────────────────────

    #[test]
    fn batch_empty_string_returns_empty() {
        let result = run_batch("").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn batch_only_comments_returns_empty() {
        let result = run_batch("-- one\n-- two\n").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn batch_only_whitespace_returns_empty() {
        // Whitespace-only input has no statement text after splitting;
        // run_batch should silently produce no output.
        let result = run_batch("    \n  \t  ").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn batch_mixed_comment_and_statement() {
        let result = run_batch("-- header\nSHOW MODELS;\n-- trailer\n").unwrap();
        assert!(!result.is_empty());
        assert!(!result.iter().any(|l| l.contains("header")));
    }

    // ── String / quote handling in split_statements ──────────────────

    #[test]
    fn split_preserves_single_quoted_strings() {
        let stmts = split_statements(r#"WALK 'hello; world' TOP 5;"#);
        assert_eq!(stmts.len(), 1);
        assert!(stmts[0].contains("hello; world"));
    }

    #[test]
    fn split_handles_double_then_single_quotes() {
        let stmts = split_statements(r#"A "x;" B 'y;' C;"#);
        assert_eq!(stmts.len(), 1);
    }

    #[test]
    fn split_two_statements_each_with_quoted_semicolon() {
        let stmts = split_statements(r#"A "x;y"; B 'p;q';"#);
        assert_eq!(stmts.len(), 2);
    }

    // ── Help banner ──────────────────────────────────────────────

    #[test]
    fn print_help_does_not_panic() {
        // Trivially invokes print_help() to cover the println! and
        // confirm the BANNER text is well-formed.
        super::print_help();
    }

    #[test]
    fn banner_constants_have_expected_shape() {
        // Sanity: BANNER is rendered at REPL start-up; the PROMPT
        // ends with "> " so users can tell the REPL is taking input,
        // and CONTINUATION is at least as wide so multi-line entries
        // visually line up with the continuation prompt.
        assert!(super::BANNER.contains("Lazarus") || super::BANNER.contains("LARQL"));
        assert!(super::PROMPT.contains(">"));
        assert_eq!(super::PROMPT.len(), super::CONTINUATION.len());
    }

    // ── History-path helpers ───────────────────────────────────────

    #[test]
    fn history_path_returns_either_some_or_none() {
        // Resolution depends on the test environment ($HOME / dirs);
        // the contract is just that it doesn't panic in either case
        // and the path (if any) ends with the history-file name.
        if let Some(path) = super::history_path() {
            let s = path.to_string_lossy();
            assert!(s.ends_with(".larql_history") || s.contains("history"));
        }
    }

    #[test]
    fn dirs_or_home_returns_either_some_or_none() {
        let _ = super::dirs_or_home();
    }

    // ── classify_input: meta commands ──────────────────────────────────

    #[test]
    fn classify_exit_quit_q_at_fresh_prompt() {
        for word in ["exit", "quit", "\\q", "EXIT", "Quit", "  exit  "] {
            let mut buf = String::new();
            assert_eq!(
                classify_input(word, &mut buf),
                LineAction::Exit,
                "expected Exit for {word:?}",
            );
            assert!(buf.is_empty());
        }
    }

    #[test]
    fn classify_clear_at_fresh_prompt() {
        for word in ["clear", "clear;", "CLEAR", "  Clear  "] {
            let mut buf = String::new();
            assert_eq!(classify_input(word, &mut buf), LineAction::Clear);
            assert!(buf.is_empty());
        }
    }

    #[test]
    fn classify_help_at_fresh_prompt() {
        for word in ["help", "\\h", "\\?", "HELP"] {
            let mut buf = String::new();
            assert_eq!(classify_input(word, &mut buf), LineAction::Help);
            assert!(buf.is_empty());
        }
    }

    #[test]
    fn classify_empty_line_at_fresh_prompt_skips() {
        let mut buf = String::new();
        assert_eq!(classify_input("", &mut buf), LineAction::Skip);
        assert_eq!(classify_input("    ", &mut buf), LineAction::Skip);
        assert!(buf.is_empty());
    }

    #[test]
    fn classify_meta_commands_only_fire_at_fresh_prompt() {
        // Once a statement is in flight, `exit` / `help` / `clear` are
        // appended verbatim so the user can include them in literals.
        let mut buf = String::from("SELECT * FROM EDGES");
        let outcome = classify_input("exit", &mut buf);
        // The buf doesn't end with `;` and isn't a bare STATS/SHOW MODELS,
        // so it stays incomplete → NeedMore.
        assert_eq!(outcome, LineAction::NeedMore);
        assert!(buf.contains("exit"));
    }

    // ── classify_input: accumulation + completeness ────────────────────

    #[test]
    fn classify_single_complete_statement_executes() {
        let mut buf = String::new();
        let outcome = classify_input("SHOW MODELS;", &mut buf);
        match outcome {
            LineAction::Execute(s) => assert_eq!(s, "SHOW MODELS;"),
            other => panic!("expected Execute, got {other:?}"),
        }
        assert!(buf.is_empty(), "buffer should be drained on Execute");
    }

    #[test]
    fn classify_multi_line_accumulates_until_semicolon() {
        let mut buf = String::new();
        assert_eq!(classify_input("SELECT *", &mut buf), LineAction::NeedMore);
        assert_eq!(
            classify_input("  FROM EDGES", &mut buf),
            LineAction::NeedMore
        );
        let outcome = classify_input("  WHERE layer = 5;", &mut buf);
        match outcome {
            LineAction::Execute(s) => {
                assert!(s.contains("SELECT *"));
                assert!(s.contains("FROM EDGES"));
                assert!(s.contains("WHERE layer = 5;"));
            }
            other => panic!("expected Execute, got {other:?}"),
        }
        assert!(buf.is_empty());
    }

    #[test]
    fn classify_bare_stats_executes_without_semicolon() {
        let mut buf = String::new();
        match classify_input("STATS", &mut buf) {
            LineAction::Execute(s) => assert_eq!(s, "STATS"),
            other => panic!("expected Execute (bare STATS), got {other:?}"),
        }
    }

    #[test]
    fn classify_bare_show_models_executes_without_semicolon() {
        let mut buf = String::new();
        match classify_input("SHOW MODELS", &mut buf) {
            LineAction::Execute(s) => assert_eq!(s, "SHOW MODELS"),
            other => panic!("expected Execute (bare SHOW MODELS), got {other:?}"),
        }
    }

    #[test]
    fn classify_stats_keyword_case_insensitive() {
        let mut buf = String::new();
        assert!(matches!(
            classify_input("stats", &mut buf),
            LineAction::Execute(_)
        ));
        let mut buf2 = String::new();
        assert!(matches!(
            classify_input("StAtS", &mut buf2),
            LineAction::Execute(_)
        ));
    }

    #[test]
    fn classify_incomplete_statement_returns_need_more() {
        let mut buf = String::new();
        assert_eq!(classify_input("SELECT 1", &mut buf), LineAction::NeedMore);
        assert!(!buf.is_empty());
    }

    #[test]
    fn classify_drains_buffer_on_execute() {
        let mut buf = String::from("partial");
        // First call: append, statement still incomplete (no `;`).
        assert_eq!(
            classify_input(" continuation;", &mut buf),
            LineAction::Execute(String::from("partial\n continuation;")),
        );
        assert!(buf.is_empty(), "Execute must drain the buffer");
    }

    // ── execute_and_print smoke ────────────────────────────────────────

    #[test]
    fn execute_and_print_runs_show_models_without_panic() {
        let mut session = crate::executor::Session::new();
        super::execute_and_print(&mut session, "SHOW MODELS;");
    }

    #[test]
    fn execute_and_print_parse_error_does_not_panic() {
        let mut session = crate::executor::Session::new();
        super::execute_and_print(&mut session, "GARBAGE STATEMENT");
    }

    #[test]
    fn execute_and_print_execution_error_does_not_panic() {
        let mut session = crate::executor::Session::new();
        // STATS without USE → executor error path, not a parse error.
        super::execute_and_print(&mut session, "STATS;");
    }
}
