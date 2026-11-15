//! Compile: write the session's edits into a new container.

use crate::help::entry::{GroupHelp, Need, StatementHelp};
use crate::help::topic::HelpGroup;
use crate::live_scope::executed_by;

pub(super) static GROUP: GroupHelp = GroupHelp {
    name: "COMPILING",
    aliases: &["WRITE", "OUTPUT", "CONTAINERS"],
    title: "Compile",
    intro: &[
        "COMPILE is the only statement that writes staged stored-weight ops into a container. Until",
        concat!("then ", executed_by!(), " in the session that staged them compute with them from memory, and"),
        "nothing else does. It writes a new container and never modifies the bound one.",
        "The output path must not exist yet: nothing is ever replaced.",
    ],
};

pub(super) static COMPILE: StatementHelp = StatementHelp {
    name: "COMPILE",
    aliases: &[
        "COMPILE CURRENT",
        "COMPILE INTO VINDEX",
        "COMPILE INTO MODEL",
        "COMPILE CURRENT INTO VINDEX",
        "COMPILE CURRENT INTO MODEL",
    ],
    group: HelpGroup::Compile,
    syntax: &[
        "COMPILE CURRENT INTO VINDEX \"<out>\" [ON CONFLICT {LAST_WINS | HIGHEST_CONFIDENCE | FAIL}];",
        "COMPILE \"<vindex>\" INTO VINDEX \"<out>\" [ON CONFLICT {LAST_WINS | HIGHEST_CONFIDENCE | FAIL}];",
        "COMPILE {CURRENT | \"<vindex>\"} INTO MODEL \"<out>\" [FORMAT {SAFETENSORS | GGUF}];",
    ],
    does: &[
        "Writes a new, self-contained vindex or container at <out> that holds the session's edits.",
        "With stored-weight ops pending on a VINDEX3 container it applies all of them as one patch:",
        "it rewrites the edited canonical (BF16) segment, hard-links every other segment, re-derives",
        "the derived (quantised) representations from the edited weights, and writes the ledger",
        "weight_patches.json. With knowledge edits it bakes those. With nothing pending it writes a",
        "clean copy with no ledger.",
    ],
    examples: &[
        "COMPILE CURRENT INTO VINDEX \"edited.vindex3\";",
        "COMPILE CURRENT INTO VINDEX \"merged.vindex\" ON CONFLICT FAIL;",
        "COMPILE CURRENT INTO MODEL \"edited-model\" FORMAT SAFETENSORS;",
    ],
    needs: &[
        Need::Backend,
        Need::Other("<out> must not exist. (With no stored-weight op pending, an empty directory is accepted too.)"),
    ],
    effect: &[
        "<out> is complete or absent: it is assembled beside the destination, inspected, and renamed",
        "into place. The session stays bound to the source and its ops stay staged, so INFER after",
        "COMPILE still computes with them in memory. USE \"<out>\" binds the container that stores them.",
        "The ledger's schema follows the ops compiled: weight-patches/v2 for UPDATE WEIGHTS only,",
        "weight-patches/v3 when a REMOVE DIRECTION is present and no ADD LOW RANK,",
        "weight-patches/v4 when any ADD LOW RANK is present.",
    ],
    operands: &[],
    refusals: &[
        "<out> exists. With stored-weight ops pending even an empty directory is refused (\"containers",
        "  are immutable\"); otherwise a directory that holds anything is. Remove it yourself or pick a new path.",
        "Stored-weight ops on a container that already carries a weight-patch ledger: a compiled",
        "  weight edit cannot be edited again. Stage every op on the unedited base and compile once.",
        "Stored-weight ops and knowledge edits (INSERT, UPDATE EDGES, DELETE) pending together.",
        "More than 262144 listed elements across the compile. Elements changed by UPDATE WEIGHTS and by",
        "  REMOVE DIRECTION count. A tensor any ADD LOW RANK edits is recorded by its ops, not its",
        "  elements, and is bounded by rank, operand size, tensor size and total work instead.",
        "The source is not a canonical container, or carries alternate variants.",
        "INTO MODEL on a VINDEX3 container.",
        "The path form with a VINDEX3 container: it answers \"No backend loaded\". USE it and compile CURRENT.",
        "ON CONFLICT with INTO MODEL: a parse error.",
        "In a batch, after any earlier statement failed (HELP BATCH;).",
    ],
    related: &[
        "USE",
        "SHOW PATCHES",
        "VERIFY LEDGER",
        "VERIFY PATCH",
        "COMPACT INTO VINDEX",
        "DIFF",
    ],
};

pub(super) static COMPACT_INTO: StatementHelp = StatementHelp {
    name: "COMPACT INTO VINDEX",
    aliases: &["COMPACT INTO"],
    group: HelpGroup::Compile,
    syntax: &["COMPACT INTO VINDEX \"<out>\";"],
    does: &[
        "Writes a copy of the bound VINDEX3 container with the same meaning and cleaner storage:",
        "referenced segments are carried byte-identically and unreferenced files are dropped.",
        "DIFF of the source against the output must report no differences.",
    ],
    examples: &["COMPACT INTO VINDEX \"compact.vindex3\";"],
    needs: &[
        Need::Vindex3,
        Need::Other("<out> absent, or an empty directory."),
    ],
    effect: &["Writes <out>. The session stays bound to the source."],
    operands: &[],
    refusals: &[
        "Any edit is pending (knowledge or stored-weight): COMPILE first, then compact the result.",
        "<out> exists and is not empty.",
        "In a batch, after any earlier statement failed (HELP BATCH;).",
    ],
    related: &["COMPILE", "DIFF"],
};
