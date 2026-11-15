//! Abandon: withdraw staged edits without writing anything.

use crate::help::entry::{GroupHelp, Need, StatementHelp};
use crate::help::topic::HelpGroup;
use crate::live_scope::executed_by;

pub(super) static GROUP: GroupHelp = GroupHelp {
    name: "ABANDON",
    aliases: &["UNDO", "DISCARD", "WITHDRAW"],
    title: "Abandon staged edits",
    intro: &[
        "Staged edits live in the process. Three things remove them:",
        "  DISCARD WEIGHT EDITS   drops the stored-weight ops staged directly in the session.",
        "  REMOVE PATCH           drops an applied patch and the ops it carried.",
        "  USE                    binds afresh and drops everything staged.",
        concat!(
            "Each takes effect at once: ",
            executed_by!(),
            " compute with the stored-weight ops that are left,"
        ),
        "and with the stored weights when none is.",
        "Ending the process drops everything too: a new larql lql starts with nothing staged.",
        "A compiled container is a file and is not undone by any statement: delete the directory.",
    ],
};

pub(super) static DISCARD_WEIGHT_EDITS: StatementHelp = StatementHelp {
    name: "DISCARD WEIGHT EDITS",
    aliases: &["DISCARD WEIGHT"],
    group: HelpGroup::Abandon,
    syntax: &["DISCARD WEIGHT EDITS;"],
    does: &[
        "Drops every stored-weight op staged directly in the session (UPDATE WEIGHTS, REMOVE",
        "DIRECTION, ADD LOW RANK), and the copies an open BEGIN PATCH recording holds. It is all or",
        "nothing: there is no form that drops one op.",
    ],
    examples: &["DISCARD WEIGHT EDITS;"],
    needs: &[Need::Vindex3],
    effect: &[
        concat!("At once. ", executed_by!(), " stop computing with the discarded ops: the reply says what is"),
        "still executed, or that the stored weights are. The elements and columns the ops claimed are",
        "free again, so the same op can be staged afresh. Nothing is written. With nothing staged it",
        "says so and is not an error.",
        "Ops that came from an applied patch stay and still execute: the reply counts them, and",
        "REMOVE PATCH removes them.",
    ],
    operands: &[],
    refusals: &["Nothing is bound, or a VINDEX2 vindex is bound."],
    related: &["REMOVE PATCH", "SHOW PATCHES", "USE", "ABANDON"],
};

pub(super) static REMOVE_PATCH: StatementHelp = StatementHelp {
    name: "REMOVE PATCH",
    aliases: &[],
    group: HelpGroup::Abandon,
    syntax: &["REMOVE PATCH \"<file.vlp>\";"],
    does: &[
        "Removes an applied patch from the session's stack, with every operation it carried,",
        "stored-weight ops included. The patch file is not touched.",
    ],
    examples: &["REMOVE PATCH \"edits.vlp\";"],
    needs: &[Need::Backend],
    effect: &[
        concat!(
            "At once. ",
            executed_by!(),
            " stop computing with the patch's stored-weight ops. Stored-weight"
        ),
        "ops staged directly in the session are kept and still execute. Knowledge edits staged",
        "directly in the session (outside any patch) are reset.",
    ],
    operands: &[],
    refusals: &["The patch is not applied in this session: \"patch not found\"."],
    related: &[
        "APPLY PATCH",
        "SHOW PATCHES",
        "DISCARD WEIGHT EDITS",
        "ABANDON",
    ],
};
