//! Run: execute the bound model on a prompt.

use crate::help::entry::{GroupHelp, Need, StatementHelp};
use crate::help::topic::HelpGroup;
use crate::live_scope::executed_by;

pub(super) static GROUP: GroupHelp = GroupHelp {
    name: "RUN",
    aliases: &["INFERENCE", "EXECUTE"],
    title: "Run the model",
    intro: &[
        concat!("These run the bound model with the session's edits. ", executed_by!(), " compute with pending"),
        "stored-weight ops (UPDATE WEIGHTS, REMOVE DIRECTION, ADD LOW RANK, or a patch applied with",
        "APPLY PATCH) as a container compiled from them would, or are refused with the reason: they",
        "never run the stored weights while such an op is pending. Knowledge edits (INSERT, DELETE,",
        "UPDATE EDGES) are seen too, but not together with stored-weight ops: that mix is refused.",
        "WALK reads the stored weights.",
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
        "Read-only. It computes with the bound container's weights and the session's edits. A pending",
        "stored-weight op (UPDATE WEIGHTS, REMOVE DIRECTION, ADD LOW RANK, or one a patch applied)",
        "is executed: the tensors it edits are computed with as a container compiled from the op",
        "stores them. If the session cannot do that exactly, INFER is refused with the reason and",
        "does not run the stored weights in their place (HELP UPDATE WEIGHTS; lists when).",
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
    effect: &[
        "Read-only unless SAVE is given.",
        concat!(executed_by!(), " run the same program: TRACE computes with pending stored-weight ops, or is"),
        "refused, exactly as INFER is.",
    ],
    operands: &[],
    refusals: &[
        "The container carries no tokenizer.json.",
        "Any clause (FOR, DECOMPOSE, LAYERS, POSITIONS, SAVE) on a VINDEX3 container: only the plain form runs there.",
        "SAVE in a batch after any earlier statement failed (HELP BATCH;).",
    ],
    related: &["INFER", "EXPLAIN"],
};

pub(super) static CAPTURE: StatementHelp = StatementHelp {
    name: "CAPTURE",
    aliases: &["ACTIVATIONS"],
    group: HelpGroup::Run,
    syntax: &[
        "CAPTURE <site> [AT LAYER <n> | AT LAYERS (<n>, ...)] FOR \"<prompt>\"",
        "        [CONTINUE \"<text>\" | CONTINUE TOKENS (<id>, ...) | GENERATE <n>]",
        "        [POSITIONS {PROMPT_END | REPLY | ALL | LAST | (<p>, ...)}] TO \"<file>.npy\";",
    ],
    does: &[
        "Runs the bound program over a sequence, as INFER runs it, and writes the values it computed",
        "at one site to a NumPy .npy file (float32, format 1.0) with a sidecar <file>.npy.json.",
        "Nothing is recomputed outside the executor: every value is one the run produced.",
        "Sites inside a layer, which need AT LAYER or AT LAYERS:",
        "  residual         the residual after the layer",
        "  post_attention   the residual after the layer's attention, before its FFN",
        "  ffn.input        the FFN's normed input (what its first projections multiply)",
        "  ffn.down_input   the input of the FFN's down projection",
        "  ffn.down_output  the down projection's output, before any post-FFN norm (dense FFN only)",
        "Sites after the last layer, which take no layer: final (the final-normed hidden state the",
        "output head reads) and logits (the whole vocabulary).",
        "The sequence is the prompt, encoded exactly as INFER encodes it (no chat template), then",
        "the continuation: CONTINUE \"<text>\" appends the text's tokens with no special tokens added,",
        "CONTINUE TOKENS appends the ids given, and GENERATE <n> appends the tokens INFER ... GENERATE",
        "<n> produces for the prompt. Positions count from 0 over the whole sequence. PROMPT_END is",
        "the prompt's last position, whose logits predict the first token after it; REPLY is every",
        "position from PROMPT_END to the second-to-last; LAST, the default, is the last position.",
        "The array is [positions, width], or [layers, positions, width] with AT LAYERS.",
        "At the last position, logits are the vector INFER ranks for the same tokens, bit for bit.",
        "residual, post_attention, ffn.input, final and logits come from INFER's own pass;",
        "ffn.down_input and ffn.down_output come from TRACE's decode steps, the only path that",
        "exposes the down projection, and the sidecar says whether those steps computed the same",
        "FFN input as INFER's pass.",
        "The sidecar records the statement, the container and the SHA-256 of its index.json, the",
        "model, the site, layers and positions (with the token at each and the token it predicts),",
        "every token id of the sequence and how many are the prompt's, \"executes\" (\"stored\" or",
        "\"patched\"), every patch in effect (its .vlp path and SHA-256 and each op's digest), the",
        "representation source and lowering, the dtype and shape, the SHA-256 of the .npy file and",
        "the LARQL version.",
    ],
    examples: &[
        "CAPTURE logits FOR \"The capital of France is\" TO \"logits.npy\";",
        "CAPTURE residual AT LAYER 1 FOR \"The capital of France is\" POSITIONS ALL TO \"residual.npy\";",
        "CAPTURE ffn.down_input AT LAYERS (0, 1) FOR \"The capital of France is\" CONTINUE \" Paris\" POSITIONS REPLY TO \"down_input.npy\";",
        "CAPTURE final FOR \"The capital of France is\" GENERATE 4 POSITIONS REPLY TO \"final.npy\";",
    ],
    needs: &[Need::Vindex3, Need::Tokenizer],
    effect: &[
        "Writes <file>.npy and <file>.npy.json and changes nothing in the session.",
        concat!(executed_by!(), " run the same program: CAPTURE computes with pending stored-weight ops,"),
        "or is refused, exactly as INFER is, and the sidecar names the ops it computed with.",
    ],
    operands: &[],
    refusals: &[
        "The session is bound to something other than a VINDEX3 container, or the container carries no tokenizer.json.",
        "An unknown site; AT LAYER missing on a site inside a layer, or given for final or logits; a layer the program does not have, or listed twice.",
        "A position outside the sequence or listed twice; POSITIONS REPLY with no continuation.",
        "A token id outside the embedding's vocabulary; a CONTINUE text that encodes to no tokens.",
        "ffn.down_output at a layer whose FFN is not dense; ffn.down_input where the backend does not serve it.",
        "Pending stored-weight ops the session cannot execute exactly (HELP UPDATE WEIGHTS; lists when).",
        "An output that does not end in .npy, or an output or sidecar that already exists: nothing is overwritten.",
        "After a failed statement in a batch (HELP BATCH;).",
    ],
    related: &["INFER", "TRACE", "EXPORT WEIGHT", "APPLY PATCH", "BATCH"],
};
