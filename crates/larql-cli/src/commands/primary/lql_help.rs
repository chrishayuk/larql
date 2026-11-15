//! What `larql lql --help` says below its options: the process model an
//! agent driving the command needs, and where the language documents
//! itself (`larql lql "HELP;"`).
//!
//! The statement syntax is not repeated here. It lives in the LQL help
//! catalogue (`larql_lql::help`), which is checked against the parser; this
//! text only says how the command runs statements and reports on them.

use larql_lql::{EXIT_CONTRACT_VIOLATED, EXIT_ERROR, EXIT_OK};

/// One line under `larql --help`.
pub const ABOUT: &str = "Run LQL statements in one process (larql lql \"HELP;\" lists them)";

/// The text after the options of `larql lql --help`.
pub fn after_help() -> String {
    format!(
        "\
Statements:
  Several statements separated by `;` run in order in ONE process and share ONE session:
  USE binds once, and staged edits and applied patches carry from one statement to the next.
  Pass them as one argument:  larql lql 'USE \"model.vindex3\"; STATS; SHOW LAYERS;'
  State persists across invocations only through files: a .vlp patch (BEGIN PATCH ... SAVE
  PATCH, then APPLY PATCH) or a compiled container (COMPILE CURRENT INTO VINDEX \"out\", then
  USE \"out\"). A new invocation starts unbound, with nothing staged.
  Output goes to standard output, and so does one `Error: ...` line per failed statement.

After a failure:
  A failed statement does not stop the batch: later statements still run. Once any statement
  has failed, every later statement that would write an artifact (COMPILE, SAVE PATCH,
  EXTRACT and the others `HELP BATCH;` lists) is refused, so a batch never writes a container
  or patch built from a session in which an earlier step was refused.

Exit status:
  {EXIT_OK}  every statement succeeded
  {EXIT_ERROR}  any statement failed: a parse error, a refusal, an execution error, a verification
     that could not be evaluated, a missing verifier; also a usage error in this command
  {EXIT_CONTRACT_VIOLATED}  a verification contract was VIOLATED and nothing else failed
  `larql repl` runs the same language interactively and does not report failure through its
  exit status: it exits 0.

Learn the language from the tool:
  larql lql \"HELP;\"                    every statement's syntax, in the order an edit uses them
  larql lql \"HELP REMOVE DIRECTION;\"   one statement: syntax, examples, requirements, refusals
  larql lql \"HELP WEIGHTS;\"            one group
  larql lql \"HELP BATCH;\"              this process model in full;  \"HELP EXIT CODES;\"  the status

A statement text that begins with `--` (an LQL comment) must follow a bare `--`:
  larql lql -- '-- note
  STATS;'"
    )
}
