//! Patches: record edits into a `.vlp` file and apply them in another session.

use crate::help::entry::{GroupHelp, Need, StatementHelp};
use crate::help::topic::HelpGroup;

pub(super) static GROUP: GroupHelp = GroupHelp {
    name: "PATCHES",
    aliases: &["PATCH", "VLP"],
    title: "Patches (.vlp files)",
    intro: &[
        "A patch is a JSON file of operations. It is the only way staged edits leave a process",
        "without being compiled: BEGIN PATCH and SAVE PATCH write one, APPLY PATCH stages its",
        "operations again in another session, REMOVE PATCH takes them back out.",
        "A stored-weight op in a patch is bound to the exact container it was written against (the",
        "SHA-256 of its index.json and of the edited representation's payload) and carries a digest",
        "of its own fields. It applies to that container and to no other.",
    ],
};

pub(super) static BEGIN_PATCH: StatementHelp = StatementHelp {
    name: "BEGIN PATCH",
    aliases: &["BEGIN"],
    group: HelpGroup::Patches,
    syntax: &["BEGIN PATCH \"<file.vlp>\";"],
    does: &[
        "Opens a recording under a file name. Every edit staged after it (knowledge edits and",
        "stored-weight ops) is also recorded, until SAVE PATCH writes the file.",
        "Knowledge edits made before it in the session are adopted into the recording.",
    ],
    examples: &["BEGIN PATCH \"edits.vlp\";"],
    needs: &[Need::Other(
        "Nothing to open the recording; the statements it records need a bound container.",
    )],
    effect: &["Session state only. Nothing is written until SAVE PATCH."],
    operands: &[],
    refusals: &[
        "A recording is already open: \"patch session already active\".",
        "An empty file name.",
    ],
    related: &[
        "SAVE PATCH",
        "APPLY PATCH",
        "DISCARD WEIGHT EDITS",
        "PATCHES",
    ],
};

pub(super) static SAVE_PATCH: StatementHelp = StatementHelp {
    name: "SAVE PATCH",
    aliases: &["SAVE"],
    group: HelpGroup::Patches,
    syntax: &["SAVE PATCH;"],
    does: &[
        "Writes the open recording to the file BEGIN PATCH named and closes the recording. A patch",
        "that carries stored-weight ops is written as version 2; otherwise version 1.",
    ],
    examples: &["SAVE PATCH;"],
    needs: &[Need::Other("An open recording: BEGIN PATCH \"<file.vlp>\" earlier in the same process.")],
    effect: &[
        "The file exists when the statement returns. The ops stay staged in the session too, so a",
        "COMPILE after SAVE PATCH still writes them.",
    ],
    operands: &[],
    refusals: &[
        "No recording is open: \"no active patch session\".",
        "The recording is anonymous (knowledge edits with no BEGIN PATCH): name it with BEGIN PATCH first.",
        "In a batch, after any earlier statement failed (HELP BATCH;).",
    ],
    related: &["BEGIN PATCH", "APPLY PATCH", "PATCHES"],
};

pub(super) static APPLY_PATCH: StatementHelp = StatementHelp {
    name: "APPLY PATCH",
    aliases: &["APPLY"],
    group: HelpGroup::Patches,
    syntax: &["APPLY PATCH \"<file.vlp>\";"],
    does: &[
        "Stages every operation of a patch file in the session, or none of them. Patches stack in",
        "the order applied.",
    ],
    examples: &["APPLY PATCH \"edits.vlp\";"],
    needs: &[Need::Backend],
    effect: &[
        "Knowledge operations take effect at once in the session. Stored-weight ops are staged like",
        "the statements that made them: they take effect only in a container written by COMPILE.",
    ],
    operands: &[],
    refusals: &[
        "The file does not exist: \"patch not found\".",
        "A stored-weight op written against another container. A container compiled from it is another.",
        "A stored-weight op whose digest does not match its contents (the file was altered).",
        "The patch applied twice, or two patches that edit one element or share a structured column.",
        "A patch whose base_model is not the bound model's name.",
        "A patch with stored-weight ops on a VINDEX2 vindex.",
    ],
    related: &["REMOVE PATCH", "SHOW PATCHES", "COMPILE", "PATCHES"],
};

pub(super) static SHOW_PATCHES: StatementHelp = StatementHelp {
    name: "SHOW PATCHES",
    aliases: &[],
    group: HelpGroup::Patches,
    syntax: &["SHOW PATCHES;"],
    does: &[
        "Lists the applied patches and, on a VINDEX3 container, the stored-weight ops pending",
        "COMPILE. Use it to see what a COMPILE would write.",
    ],
    examples: &["SHOW PATCHES;"],
    needs: &[Need::Backend],
    effect: &["Read-only."],
    operands: &[],
    refusals: &["Nothing is bound."],
    related: &["APPLY PATCH", "REMOVE PATCH", "DISCARD WEIGHT EDITS"],
};
