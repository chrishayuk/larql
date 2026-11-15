//! Bind: point the session at a vindex or container.

use crate::help::entry::{GroupHelp, Need, StatementHelp};
use crate::help::topic::HelpGroup;

pub(super) static GROUP: GroupHelp = GroupHelp {
    name: "BIND",
    aliases: &["BINDING", "LIFECYCLE"],
    title: "Bind a model",
    intro: &[
        "Every statement that reads or edits a model acts on what the session is bound to.",
        "A VINDEX3 container is a directory holding index.json and segments/. A VINDEX2 vindex is",
        "the older directory format. USE decides which by the directory's own marker.",
    ],
};

pub(super) static USE: StatementHelp = StatementHelp {
    name: "USE",
    aliases: &["USE MODEL", "USE REMOTE"],
    group: HelpGroup::Bind,
    syntax: &[
        "USE \"<path>\";",
        "USE MODEL \"<id>\" [AUTO_EXTRACT];",
        "USE REMOTE \"<url>\";",
    ],
    does: &[
        "Binds the session to a vindex or container; later statements in the same process act on it.",
        "<path> is a VINDEX2 vindex directory, a VINDEX3 container directory, an hf://owner/repo",
        "published vindex (fetched into the local cache), or a model name the embedding program",
        "resolves (larql lql resolves names already on this machine and never downloads one).",
        "USE MODEL loads a checkpoint's live weights. USE REMOTE forwards statements to a larql server.",
    ],
    examples: &[
        "USE \"model.vindex3\";",
        "USE MODEL \"google/gemma-3-4b-it\";",
        "USE REMOTE \"http://localhost:8080\";",
    ],
    needs: &[Need::Nothing],
    effect: &[
        "At once, until the process ends or another USE runs. The reply names the generation, the",
        "statements the binding serves, and whether a tokenizer is present.",
        "A second USE replaces the binding and drops everything staged: stored-weight ops, applied",
        "patches and an open patch recording. Nothing warns about what was dropped.",
    ],
    operands: &[],
    refusals: &[
        "<path> is not a directory, a published vindex or a known model name.",
        "The PUBLIC_EXPLORER profile refuses USE: the embedding server binds, then tightens.",
    ],
    related: &["STATS", "SHOW MODELS", "EXTRACT", "BATCH"],
};

pub(super) static EXTRACT: StatementHelp = StatementHelp {
    name: "EXTRACT",
    aliases: &["EXTRACT MODEL"],
    group: HelpGroup::Bind,
    syntax: &[
        "EXTRACT MODEL \"<id>\" INTO \"<out>\" [COMPONENTS <c>, ...] [LAYERS <a>-<b>] [WITH {INFERENCE | ALL}] [FORMAT {VINDEX2 | VINDEX3}];",
    ],
    does: &[
        "Builds a new vindex at <out> from a model checkpoint (a Hugging Face id or a local",
        "directory), then binds the session to it. FORMAT picks the generation; without it the",
        "vindex crate's extraction-generation policy decides.",
        "A VINDEX3 extraction encodes the checkpoint's own bytes into a container.",
        "WITH shapes a VINDEX2 extraction only: none serves browsing, WITH INFERENCE adds what INFER",
        "needs, WITH ALL adds what COMPILE needs.",
        "COMPONENTS and LAYERS are parsed and ignored.",
    ],
    examples: &[
        "EXTRACT MODEL \"Qwen/Qwen3-0.6B\" INTO \"qwen3-0.6b.vindex3\";",
        "EXTRACT MODEL \"google/gemma-3-4b-it\" INTO \"gemma3-4b.vindex\" WITH ALL FORMAT VINDEX2;",
    ],
    needs: &[
        Need::Nothing,
        Need::Other("The model's checkpoint: a local directory, or a Hugging Face id."),
    ],
    effect: &[
        "Writes <out> and binds it. A failed extraction can leave a partial directory at <out>,",
        "which a re-run refuses as not empty.",
    ],
    operands: &[],
    refusals: &[
        "<out> exists and is not an empty directory. Nothing is ever replaced.",
        "A GGUF file as the source of a VINDEX3 extraction: use FORMAT VINDEX2.",
        "In a batch, after any earlier statement failed (HELP BATCH;).",
    ],
    related: &["USE", "COMPILE", "BATCH"],
};
