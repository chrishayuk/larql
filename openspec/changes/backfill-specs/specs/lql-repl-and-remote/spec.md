## ADDED Requirements

### Requirement: REPL statement splitting and completeness

`larql_lql::repl::split_statements` SHALL split an input string into
individual LQL statements terminated by `;` while preserving
semicolons embedded inside string literals, tolerating multi-line
input, and returning an empty vector for empty input. Trailing text
without a closing `;` MUST still surface as a final partial
statement so the REPL can show a clear error rather than swallow
the input. `is_complete_statement` SHALL return `true` only when
the input ends with a semicolon (after trimming) so the REPL can
buffer multi-line input until the user finishes the statement.

#### Scenario: Single statement, multi-statement, and multi-line splits
- **WHEN** `STATS;`, `STATS; SHOW MODELS;`, and `STATS;\nSHOW MODELS;\nSHOW LAYERS;` are split
- **THEN** the splitter MUST yield one, two, and three statements respectively
<!-- test: larql_lql::repl::tests::split_single_statement -->
<!-- test: larql_lql::repl::tests::split_multiple_statements -->
<!-- test: larql_lql::repl::tests::split_multiline -->

#### Scenario: Splitter preserves semicolons inside strings
- **WHEN** `WALK "hello; world" TOP 5;` is split
- **THEN** the result MUST be a single statement whose text contains the embedded `hello; world`
<!-- test: larql_lql::repl::tests::split_preserves_strings_with_semicolons -->

#### Scenario: Empty input and trailing-no-semicolon edge cases
- **WHEN** `""` and `STATS; SHOW MODELS` (no trailing semicolon) are split
- **THEN** the splitter MUST return an empty vector for empty input and two statements for the trailing-no-semicolon form
<!-- test: larql_lql::repl::tests::split_empty_input -->
<!-- test: larql_lql::repl::tests::split_trailing_text_without_semicolon -->

#### Scenario: Statement completeness gates REPL buffering
- **WHEN** `is_complete_statement` is called on `STATS;`, `STATS`, and a partial `SELECT *\n  FROM EDGES`
- **THEN** the result MUST be `true` only for the explicitly terminated input
<!-- test: larql_lql::repl::tests::is_complete_with_semicolon -->
<!-- test: larql_lql::repl::tests::is_not_complete_without_semicolon -->
<!-- test: larql_lql::repl::tests::is_not_complete_multiline_partial -->

### Requirement: Batch and run-statement execution surface

`larql_lql::repl::run_batch` SHALL execute every statement in a
multi-statement input string in order, capture per-statement
errors as `Error: …` lines (rather than aborting), skip `--`
comments, and emit at least a header per statement so the user can
correlate output with input. `larql_lql::repl::run_statement`
SHALL parse and execute exactly one statement, returning the
output `Vec<String>` on success or a boxed `std::error::Error` on
parse/execution failure.

#### Scenario: run_batch executes successful and erroring statements
- **WHEN** `run_batch("SHOW MODELS;")`, `run_batch("STATS;")`, `run_batch("SHOW MODELS; SHOW MODELS;")`, `run_batch("-- comment\nSHOW MODELS;")`, and `run_batch("FOOBAR;")` are called
- **THEN** the SHOW MODELS forms MUST emit non-empty output, STATS MUST emit an `Error` line because no backend is loaded, the comment MUST be skipped, and the parse error MUST be captured as an `Error` line rather than a panic
<!-- test: larql_lql::repl::tests::batch_show_models_runs -->
<!-- test: larql_lql::repl::tests::batch_errors_are_captured -->
<!-- test: larql_lql::repl::tests::batch_multiple_statements -->
<!-- test: larql_lql::repl::tests::batch_comments_skipped -->
<!-- test: larql_lql::repl::tests::batch_parse_error_captured -->

#### Scenario: run_statement returns Ok for valid input and Err for invalid input
- **WHEN** `run_statement("SHOW MODELS;")` and `run_statement("NOT A VALID STATEMENT;")` are called
- **THEN** the first MUST return `Ok(_)` and the second MUST return `Err(_)`
<!-- test: larql_lql::repl::tests::run_statement_show_models -->
<!-- test: larql_lql::repl::tests::run_statement_parse_error -->

#### Scenario: Interactive REPL initialises without crashing
- **WHEN** the basic REPL bootstrap path is exercised
- **THEN** the constructor MUST succeed (history file resolution and prompt setup MUST not panic)
<!-- test: larql_lql::repl::tests::batch_show_models_runs -->

### Requirement: USE REMOTE establishes a federated session

`Session::exec_use_remote` SHALL trim a trailing `/` from the URL,
build a 30-second timeout `reqwest::blocking::Client`, probe
`/v1/stats` to verify connectivity, parse `model`, `layers`, and
`features` from the JSON response, generate a unique session id
based on PID + millis-since-epoch, replace `self.backend` with
`Backend::Remote { url, client, local_patches: Vec::new(),
session_id }`, clear any active patch recording, and return a
"Connected: …" banner. Connection failures and non-2xx responses
MUST surface as `LqlError::Execution` with the offending URL or
status. After the call, `is_remote()` MUST return `true` and
remote dispatch SHALL be available for the documented verbs.

#### Scenario: USE REMOTE flips backend to Remote and remote dispatch is gated
- **WHEN** the executor dispatches a statement on a `Backend::Remote` session
- **THEN** `is_remote()` MUST return `true` and the executor MUST forward via `execute_remote`, supporting only DESCRIBE / WALK / INFER / EXPLAIN / SELECT / STATS / SHOW RELATIONS / INSERT / DELETE / UPDATE / APPLY PATCH / SHOW PATCHES / REMOVE PATCH / USE / Pipe — every other verb MUST return an `LqlError::Execution` whose message hints that TRACE requires a local vindex
<!-- test: larql_lql::executor::tests::no_backend_trace -->
<!-- test: larql_lql::parser::tests::parse_demo_script_act3 -->

#### Scenario: Remote INSERT drops ALPHA / MODE since the protocol lacks a schema
- **WHEN** `Statement::Insert { alpha, mode, .. }` is forwarded to a remote backend
- **THEN** the executor MUST omit `alpha` and `mode` from the HTTP request and forward only `entity`, `relation`, `target`, `layer`, and `confidence`; the local backend SHALL keep honouring `alpha` and `mode` directly via `exec_insert`
<!-- test: larql_lql::executor::tests::knn_store_insert_at_layer_hint -->

### Requirement: Statement-level error handling and protocol resilience

The remote executor SHALL parse errors from the server as
`LqlError::Execution(message)` so the REPL surfaces them
identically to local errors. Per-statement failures (whether
parse, local execution, or remote 5xx) MUST never abort the REPL
loop or `run_batch` and MUST always be captured as `Error:` lines
in the user's output stream.

#### Scenario: Parse errors propagate via run_statement and run_batch
- **WHEN** an invalid statement is fed to either entry point
- **THEN** `run_statement` MUST return `Err`, and `run_batch` MUST capture the error in its output without aborting subsequent statements
<!-- test: larql_lql::repl::tests::run_statement_parse_error -->
<!-- test: larql_lql::repl::tests::batch_parse_error_captured -->
<!-- test: larql_lql::repl::tests::batch_errors_are_captured -->

#### Scenario: REPL splitter and pipe operator co-operate
- **WHEN** the REPL receives a multi-statement input that includes a piped statement (`WALK "x" |> EXPLAIN WALK "x";`) followed by another statement
- **THEN** `split_statements` MUST treat the pipe as part of one statement and the parser MUST emit a `Statement::Pipe`
<!-- test: larql_lql::parser::tests::parse_pipe_walk_to_explain -->
<!-- test: larql_lql::repl::tests::split_multiple_statements -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_lql::repl::tests::**::* -->
