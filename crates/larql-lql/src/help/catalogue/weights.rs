//! Edit weights: stage ops on the stored weights of a VINDEX3 container.

use crate::help::entry::{GroupHelp, Need, StatementHelp};
use crate::help::topic::HelpGroup;
use crate::live_scope::{executed_by, reads_stored};

pub(super) static GROUP: GroupHelp = GroupHelp {
    name: "WEIGHTS",
    aliases: &["WEIGHT", "EDIT WEIGHTS", "WEIGHT EDITS", "STORED WEIGHTS"],
    title: "Edit stored weights (VINDEX3)",
    intro: &[
        "Each statement stages one op on one stored BF16 matrix of the bound container. Staging",
        "checks everything it can and writes nothing. The bound container is never modified.",
        concat!("A staged op takes effect at once in the session: ", executed_by!(), " compute with the"),
        "patched weights, held in memory, and the reply lists each patched tensor with its SHA-256.",
        "COMPILE CURRENT INTO VINDEX writes a container that stores exactly those bytes.",
        concat!("If the session cannot execute an op exactly, ", executed_by!(), " are refused while it is"),
        "pending: they never run the stored weights in its place. HELP UPDATE WEIGHTS; lists when.",
        "One compile takes several ops on several tensors of one canonical representation.",
        "A container compiled with stored-weight ops carries a ledger and cannot be edited again:",
        "stage every op on the unedited base and compile once.",
        "The loop: USE, stage, [BEGIN PATCH ... SAVE PATCH], COMPILE, VERIFY. HELP BATCH; for how",
        "state carries between statements and processes.",
    ],
};

/// How a tensor is named, for all three statements.
const TENSOR_NAME: Need = Need::Other(
    "The tensor's name as the container stores it, such as \"0.self_attn.q_proj.weight\", or <representation>::<tensor> where that bare name is ambiguous: SHOW TENSORS; lists every name.",
);

/// What every staged op has in common about visibility.
const STAGED_EFFECT: &[&str] = &[
    concat!("Staging writes nothing to disk. From the next statement, ", executed_by!(), " in this session"),
    "compute with the patched weights: the reply says so and lists each patched tensor with the",
    "SHA-256 of its bytes, which is the digest that tensor has in a container compiled from the op.",
    concat!(reads_stored!(), " ... CURRENT still read the stored weights."),
    concat!(executed_by!(), " are refused, not run on the stored weights, while an op is pending and:"),
    "  the session's representation source is not `stored`;",
    "  a knowledge edit (INSERT, UPDATE EDGES, DELETE) is pending too;",
    "  the tensor is one execution reads past the patched bytes (an expert bank, a shared expert, a",
    "  routed or hybrid layer, a mixer other than softmax attention) or does not project through;",
    "  or COMPILE would refuse the op before writing.",
    "To keep the edit, COMPILE CURRENT INTO VINDEX \"<out>\". To withdraw it, DISCARD WEIGHT EDITS.",
    "While a BEGIN PATCH recording is open the op is also recorded, so SAVE PATCH can write it.",
];

pub(super) static UPDATE_WEIGHTS: StatementHelp = StatementHelp {
    name: "UPDATE WEIGHTS",
    aliases: &["WEIGHT DELTA"],
    group: HelpGroup::Weights,
    syntax: &[
        "UPDATE WEIGHTS SET DELTA = <number> WHERE TENSOR = \"<tensor>\" AND ROW = <n> AND {COL = <n> | COL IN (<n>, ...)};",
    ],
    does: &[
        "Stages a delta on named elements of one stored BF16 matrix. Each element becomes the",
        "bfloat16 nearest to (stored value + delta), ties to even, rounded once. The delta applies",
        "to the value stored in the bound container, so staging never accumulates.",
    ],
    examples: &[
        "UPDATE WEIGHTS SET DELTA = 0.0625 WHERE TENSOR = \"0.mlp.down_proj.weight\" AND ROW = 5 AND COL IN (12, 13);",
        "UPDATE WEIGHTS SET DELTA = -0.125 WHERE TENSOR = \"0.mlp.up_proj.weight\" AND ROW = 5 AND COL = 14;",
    ],
    needs: &[Need::Vindex3, TENSOR_NAME],
    effect: STAGED_EFFECT,
    operands: &[],
    refusals: &[
        "The container already carries a weight-patch ledger (it is a compiled edit): refused at staging.",
        "The tensor is not held by a canonical representation, is not BF16, or is not a matrix.",
        "A row or column outside the tensor.",
        "An element already edited in this session or by an applied patch: edits to one element are not composed.",
        "The tensor is already edited by REMOVE DIRECTION or ADD LOW RANK: one tensor takes one form.",
        "A delta that is not finite, or a result that overflows bfloat16.",
        "More than 262144 listed elements in one compile.",
        "A delta of 0 is NOT refused: it stages, compiles and changes no bits.",
    ],
    related: &[
        "REMOVE DIRECTION",
        "ADD LOW RANK",
        "DISCARD WEIGHT EDITS",
        "BEGIN PATCH",
        "COMPILE",
        "WEIGHTS",
    ],
};

pub(super) static REMOVE_DIRECTION: StatementHelp = StatementHelp {
    name: "REMOVE DIRECTION",
    aliases: &[],
    group: HelpGroup::Weights,
    syntax: &[
        "REMOVE DIRECTION \"<operand.json>\" FROM \"<tensor>\" COLUMNS (<n>, ...) SCALE <number> NUMERICS \"<profile>\";",
    ],
    does: &[
        "Stages the removal of a direction u from the chosen columns S of one stored BF16 matrix:",
        "W[:, S] <- W[:, S] - scale * u * (u^T W[:, S]). The op is declared by its operand, columns,",
        "scale and numerics profile; COMPILE writes the exact BF16 results the profile defines.",
        "No f32 delta is computed anywhere.",
    ],
    examples: &[
        "REMOVE DIRECTION \"u.json\" FROM \"1.mlp.down_proj.weight\" COLUMNS (33, 34) SCALE 0.5 NUMERICS \"f64-seq-rne1/v1\";",
        "REMOVE DIRECTION \"u.json\" FROM \"1.self_attn.o_proj.weight\" COLUMNS (40) SCALE -1 NUMERICS \"f64-seq-rne2/v1\";",
    ],
    needs: &[
        Need::Vindex3,
        TENSOR_NAME,
        Need::Other("The operand file, read when the statement runs. A relative path is relative to the working directory."),
    ],
    effect: STAGED_EFFECT,
    operands: &[
        "A JSON object: {\"bits\": [\"<hex>\", ...], \"origin\": \"<note>\"}",
        "  bits    One string per row of the tensor: that component of u as the bit pattern of an",
        "          f64, exactly 16 lowercase hex digits. \"3ff0000000000000\" is 1.0. Never decimals.",
        "  origin  Optional. Where the direction came from. The op's digest covers it; nothing",
        "          computes with it.",
        "Any other field is refused.",
    ],
    refusals: &[
        "NUMERICS other than \"f64-seq-rne1/v1\" or \"f64-seq-rne2/v1\". \"f32-seq-rne1/v1\" is refused here.",
        "The operand file is missing, is not this JSON, or holds a string that is not 16 lowercase hex digits.",
        "COLUMNS that are empty, not strictly increasing, or outside the tensor.",
        "A column another structured op already writes: two structured ops do not share a column.",
        "The tensor is already edited by UPDATE WEIGHTS: one tensor takes one form.",
        "An op that changes no element, such as SCALE 0.",
        "SCALE as an integer beyond 2^53 (not exactly an f64): a parse error.",
        "On a container that already carries a ledger the op stages, and COMPILE then refuses.",
    ],
    related: &[
        "ADD LOW RANK",
        "UPDATE WEIGHTS",
        "DISCARD WEIGHT EDITS",
        "COMPILE",
        "VERIFY PATCH",
        "WEIGHTS",
    ],
};

pub(super) static ADD_LOW_RANK: StatementHelp = StatementHelp {
    name: "ADD LOW RANK",
    aliases: &["ADD", "LOW RANK"],
    group: HelpGroup::Weights,
    syntax: &[
        "ADD LOW RANK \"<operands.json>\" TO \"<tensor>\" COLUMNS (<n>, ...) NUMERICS \"<profile>\";",
    ],
    does: &[
        "Stages a low-rank update of the chosen columns J of one stored BF16 matrix:",
        "W[r, J[p]] <- W[r, J[p]] + sum over k of a[r, k] * b[k, p]. Rows are the output axis and",
        "COLUMNS index the input axis. COMPILE writes the exact BF16 results the profile defines.",
        "There is no scale: fold it into a or b.",
    ],
    examples: &[
        "ADD LOW RANK \"a_b.json\" TO \"1.mlp.gate_proj.weight\" COLUMNS (8, 9) NUMERICS \"f64-seq-rne1/v1\";",
        "ADD LOW RANK \"a_b_f32.json\" TO \"1.self_attn.o_proj.weight\" COLUMNS (0, 1, 2, 63) NUMERICS \"f32-seq-rne1/v1\";",
    ],
    needs: &[
        Need::Vindex3,
        TENSOR_NAME,
        Need::Other("The operand file, read when the statement runs. A relative path is relative to the working directory."),
    ],
    effect: STAGED_EFFECT,
    operands: &[
        "A JSON object: {\"rank\": K, \"a\": [\"<hex>\", ...], \"b\": [\"<hex>\", ...], \"origin\": \"<note>\"}",
        "  rank    K, at most 256.",
        "  a       rows x K values, row-major: a[r, k] is at index r * K + k.",
        "  b       K x (number of COLUMNS) values, row-major: b[k, p] is at index k * |COLUMNS| + p.",
        "          b is compact: it covers only the listed columns, in the order listed.",
        "  origin  Optional. The op's digest covers it; nothing computes with it.",
        "Each value is a bit pattern in the profile's working width, lowercase hex: 16 digits (f64)",
        "for \"f64-seq-rne1/v1\" and \"f64-seq-rne2/v1\", 8 digits (f32) for \"f32-seq-rne1/v1\".",
        "Any other field is refused.",
    ],
    refusals: &[
        "NUMERICS that is not one of the three profiles above.",
        "a or b of the wrong length for the tensor's rows, the rank and the number of columns.",
        "A value that is not hex of the profile's width.",
        "COLUMNS that are empty, not strictly increasing, or outside the tensor.",
        "A column another structured op already writes: two structured ops do not share a column.",
        "The tensor is already edited by UPDATE WEIGHTS: one tensor takes one form.",
        "An op whose results all round back to the stored bits.",
        "An op with a zero factor in every term is NOT refused: it is a declared sham and compiles to the base's bytes.",
        "On a container that already carries a ledger the op stages, and COMPILE then refuses.",
    ],
    related: &[
        "REMOVE DIRECTION",
        "UPDATE WEIGHTS",
        "DISCARD WEIGHT EDITS",
        "COMPILE",
        "VERIFY LEDGER",
        "WEIGHTS",
    ],
};
