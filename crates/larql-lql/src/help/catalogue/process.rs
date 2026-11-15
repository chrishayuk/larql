//! Process model: how statements are run, and what carries between them.

use crate::help::entry::{ConceptHelp, GroupHelp, Need, StatementHelp};
use crate::help::topic::HelpGroup;

pub(super) static GROUP: GroupHelp = GroupHelp {
    name: "PROCESS",
    aliases: &["PROCESS MODEL", "RUNNING"],
    title: "Process model",
    intro: &[
        "larql lql \"<statements>\" runs every statement in one process and one session, then exits.",
        "Nothing carries from one process to the next except files.",
    ],
};

pub(super) static HELP: StatementHelp = StatementHelp {
    name: "HELP",
    aliases: &[],
    group: HelpGroup::Process,
    syntax: &["HELP [<topic>];"],
    does: &[
        "HELP; prints every statement's syntax, grouped in the order an editing loop uses them.",
        "HELP <topic>; prints one page. A topic is a statement's leading words (HELP REMOVE",
        "DIRECTION;), a group (HELP WEIGHTS;) or a process topic (HELP BATCH;, HELP EXIT CODES;).",
        "Topics are case-insensitive and may be quoted.",
    ],
    examples: &[
        "HELP;",
        "HELP REMOVE DIRECTION;",
        "HELP WEIGHTS;",
        "HELP EXIT CODES;",
    ],
    needs: &[Need::Nothing],
    effect: &["Read-only. It writes nothing and is allowed under every capability profile."],
    operands: &[],
    refusals: &[
        "An unknown topic is an error that lists the nearest topics; it is not the overview.",
    ],
    related: &["BATCH", "EXIT CODES"],
};

pub(super) static PIPE: StatementHelp = StatementHelp {
    name: "PIPE",
    aliases: &["|>"],
    group: HelpGroup::Process,
    syntax: &["<statement> |> <statement>;"],
    does: &[
        "Runs the left statement, then the right, as one statement, and prints both outputs. It",
        "joins exactly two statements; no data flows from one to the other.",
    ],
    examples: &["STATS |> SHOW LAYERS;"],
    needs: &[Need::Other("Whatever its two statements need.")],
    effect: &["That of its two statements. If the left one fails, the right one does not run."],
    operands: &[],
    refusals: &[
        "A second |> in one statement: a parse error.",
        "A capability profile that refuses either side refuses the whole pipe before anything runs.",
    ],
    related: &["BATCH"],
};

pub(super) static BATCH: ConceptHelp = ConceptHelp {
    name: "BATCH",
    aliases: &["BATCHES", "SESSION", "SESSIONS", "SCRIPT"],
    group: HelpGroup::Process,
    summary: "several statements in one process share one session; only files persist",
    body: &[
        "One process, one session.",
        "  larql lql 'USE \"base\"; UPDATE WEIGHTS ...; COMPILE CURRENT INTO VINDEX \"out\";'",
        "  runs its statements in order in one session. They are split on \";\" outside quotes.",
        "  The binding made by USE, staged stored-weight ops, applied patches and an open patch",
        "  recording all carry from one statement to the next.",
        "",
        "Nothing carries between processes except files.",
        "  A second larql lql starts unbound with nothing staged. To continue work, write a file:",
        "    a patch       BEGIN PATCH \"e.vlp\"; ...; SAVE PATCH;   then   USE ...; APPLY PATCH \"e.vlp\";",
        "    a container   COMPILE CURRENT INTO VINDEX \"out\";       then   USE \"out\";",
        "",
        "A failed statement does not stop the batch.",
        "  Its failure is a line starting \"Error:\" on standard output, in place of its output. Later",
        "  statements still run, so one run reports every problem.",
        "",
        "After a failure, nothing is written.",
        "  Once any statement has failed (a parse error, a refusal, an execution error), every later",
        "  statement that would write an artifact is refused and counts as a failure: EXTRACT,",
        "  COMPILE, COMPACT INTO VINDEX, COMPACT MAJOR, SAVE PATCH, DIFF ... INTO PATCH, TRACE ...",
        "  SAVE, EXPORT WEIGHT, CAPTURE, and a pipe containing one. So a COMPILE never writes a",
        "  container built from a session in which an earlier edit was refused. A violated VERIFY",
        "  contract is not a failure and does not stop later writes.",
        "",
        "Comments and the command line.",
        "  A \"--\" comment runs to the end of its line. Keep quote characters out of comments: the",
        "  statement splitter reads one as the start of a string and joins the statements after it.",
        "  A statement text that begins with \"--\" must be passed after a bare \"--\" argument, or",
        "  the command line takes it for a flag.",
        "",
        "larql repl runs the same statements interactively, keeps running after a failure, does not",
        "refuse writes after one, and always exits 0.",
    ],
    related: &["EXIT CODES", "USE", "SAVE PATCH", "COMPILE", "HELP"],
};

pub(super) static EXIT_CODES: ConceptHelp = ConceptHelp {
    name: "EXIT CODES",
    aliases: &["EXIT CODE", "EXIT STATUS", "EXIT"],
    group: HelpGroup::Process,
    summary: "0 all succeeded, 1 any failure, 2 a contract was violated and nothing failed",
    body: &[
        "larql lql exits with:",
        "  0  Every statement succeeded, and every VERIFY contract asked for was satisfied.",
        "  1  At least one statement failed: a parse error, a refusal, an execution error, a VERIFY",
        "     whose contract could not be evaluated, a missing verifier, or a statement refused",
        "     because an earlier one failed. A usage error in the command itself is also 1.",
        "  2  Every statement succeeded, and at least one VERIFY contract was evaluated and violated.",
        "",
        "A failure outranks a violation: a batch with both exits 1.",
        "The status covers the whole batch. A later statement that succeeds does not hide an earlier",
        "failure, so a contract that could not be evaluated never reads as satisfied.",
        "Every diagnostic is on standard output, one \"Error: ...\" line per failed statement.",
        "",
        "larql repl does not report failure through its exit status: it exits 0.",
    ],
    related: &["BATCH", "VERIFIER", "VERIFY"],
};

pub(super) static VERIFIER: ConceptHelp = ConceptHelp {
    name: "VERIFIER",
    aliases: &["VERIFIERS", "LARQL_VERIFIER"],
    group: HelpGroup::Process,
    summary: "how a VERIFY statement finds the external program it runs",
    body: &[
        "A VERIFY statement runs an external program and reports its answer. LQL evaluates nothing",
        "itself. The first of these that applies names the program:",
        "  1. the embedding program's choice (Session::set_verifier);",
        "  2. the LARQL_VERIFIER environment variable;",
        "  3. larql-verifier on PATH, absolute entries only;",
        "  4. verifier/.venv/bin/larql-verifier in a directory above the running larql binary, which",
        "     is where a checkout's own verifier is installed.",
        "The current directory is never searched.",
        "",
        "If none applies, every VERIFY statement is refused (exit 1); none passes by default.",
        "Setting 1 or 2 to an empty value turns discovery off, and the statements are refused.",
        "",
        "The verifier is given the contract id and canonical absolute paths. It has one hour. LQL does",
        "not authenticate it: replacing the verifier can make a contract appear satisfied, so a",
        "result must not gate a release on this basis alone.",
    ],
    related: &["VERIFY", "EXIT CODES"],
};
