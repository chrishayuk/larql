//! Run: execute the bound model on a prompt.

use crate::help::entry::{GroupHelp, Need, StatementHelp};
use crate::help::topic::HelpGroup;

pub(super) static GROUP: GroupHelp = GroupHelp {
    name: "RUN",
    aliases: &["INFERENCE", "EXECUTE"],
    title: "Run the model",
    intro: &[
        "These run the bound model as it is stored, plus the session's knowledge edits (INSERT,",
        "DELETE, UPDATE EDGES). They never see a staged stored-weight op: to run edited weights,",
        "COMPILE them into a container and USE that container.",
    ],
};

pub(super) static INFER: StatementHelp = StatementHelp {
    name: "INFER",
    aliases: &[],
    group: HelpGroup::Run,
    syntax: &[
        "INFER \"<prompt>\" [TOP <n>] [GENERATE <n>] [COMPARE] [ROUTE VERIFY [FALLBACK] [TOPK <n>] [EXIT]];",
    ],
    does: &[
        "Runs a full forward pass and prints the top next-token predictions. GENERATE <n> continues",
        "greedily for <n> tokens (VINDEX3 containers only). ROUTE VERIFY selects the checked router",
        "for facts stored by INSERT ... MODE KNN.",
    ],
    examples: &[
        "INFER \"The capital of France is\" TOP 5;",
        "INFER \"The capital of France is\" GENERATE 16;",
    ],
    needs: &[
        Need::Backend,
        Need::Tokenizer,
        Need::Other("On a VINDEX2 vindex: model weights (EXTRACT ... WITH INFERENCE or WITH ALL)."),
    ],
    effect: &[
        "Read-only. It computes with the weights of the bound container as stored. A stored-weight op",
        "staged in this session (UPDATE WEIGHTS, REMOVE DIRECTION, ADD LOW RANK) is NOT applied.",
    ],
    operands: &[],
    refusals: &[
        "The container carries no tokenizer.json. A prompt of token ids is refused too.",
        "GENERATE on anything but a VINDEX3 container.",
        "GENERATE above 32 under the PUBLIC_EXPLORER profile.",
    ],
    related: &["EXPLAIN", "TRACE", "COMPILE", "USE"],
};

pub(super) static WALK: StatementHelp = StatementHelp {
    name: "WALK",
    aliases: &[],
    group: HelpGroup::Run,
    syntax: &[
        "WALK \"<prompt>\" [TOP <n>] [LAYERS {<a>-<b> | ALL}] [MODE {HYBRID | PURE | DENSE}] [COMPARE];",
    ],
    does: &[
        "Scans which gate features fire for the prompt's last token embedding, per layer. It is a",
        "browse of the feature space, not inference: no attention runs.",
    ],
    examples: &["WALK \"France\" TOP 10;", "WALK \"The capital of France is\" TOP 5 LAYERS 24-33;"],
    needs: &[Need::Backend, Need::Tokenizer],
    effect: &["Read-only."],
    operands: &[],
    refusals: &["The container carries no tokenizer.json."],
    related: &["DESCRIBE", "EXPLAIN", "INFER"],
};

pub(super) static EXPLAIN: StatementHelp = StatementHelp {
    name: "EXPLAIN",
    aliases: &["EXPLAIN WALK", "EXPLAIN INFER"],
    group: HelpGroup::Run,
    syntax: &[
        "EXPLAIN {INFER | WALK} \"<prompt>\" [LAYERS <a>-<b>] [VERBOSE] [TOP <n>] [RELATIONS ONLY] [WITH ATTENTION] [SYNTAX | KNOWLEDGE | OUTPUT | ALL LAYERS];",
    ],
    does: &[
        "EXPLAIN WALK prints the per-layer feature trace of a WALK. EXPLAIN INFER on a VINDEX3",
        "container prints the executable plan without running a token: each layer's ops and the",
        "tensors they read. The part of a tensor after \"::\" is the name the weight-edit statements take.",
    ],
    examples: &["EXPLAIN INFER \"x\";", "EXPLAIN WALK \"The capital of France is\" TOP 3;"],
    needs: &[
        Need::Backend,
        Need::Other("EXPLAIN WALK needs a tokenizer. EXPLAIN INFER on a VINDEX3 container needs none."),
    ],
    effect: &["Read-only."],
    operands: &[],
    refusals: &["EXPLAIN WALK when the container carries no tokenizer.json."],
    related: &["INFER", "WALK", "UPDATE WEIGHTS"],
};

pub(super) static TRACE: StatementHelp = StatementHelp {
    name: "TRACE",
    aliases: &[],
    group: HelpGroup::Run,
    syntax: &[
        "TRACE \"<prompt>\" [FOR \"<token>\"] [DECOMPOSE] [LAYERS <a>-<b>] [POSITIONS {LAST | ALL}] [SAVE \"<file>\"];",
    ],
    does: &[
        "Runs the model and reports the residual stream layer by layer. FOR tracks one answer token;",
        "DECOMPOSE splits attention from FFN; SAVE writes the trace to a file.",
    ],
    examples: &[
        "TRACE \"The capital of France is\";",
        "TRACE \"The capital of France is\" FOR \"Paris\";",
        "TRACE \"The capital of France is\" SAVE \"france.trace\";",
    ],
    needs: &[
        Need::Backend,
        Need::Tokenizer,
        Need::Other("On a VINDEX2 vindex: model weights (EXTRACT ... WITH ALL)."),
    ],
    effect: &["Read-only unless SAVE is given. Like INFER, it does not see staged stored-weight ops."],
    operands: &[],
    refusals: &[
        "The container carries no tokenizer.json.",
        "Any clause (FOR, DECOMPOSE, LAYERS, POSITIONS, SAVE) on a VINDEX3 container: only the plain form runs there.",
        "SAVE in a batch after any earlier statement failed (HELP BATCH;).",
    ],
    related: &["INFER", "EXPLAIN"],
};
