//! Knowledge edits: edges in the session's overlay, not stored weights.

use crate::help::entry::{GroupHelp, Need, StatementHelp};
use crate::help::topic::HelpGroup;

pub(super) static GROUP: GroupHelp = GroupHelp {
    name: "KNOWLEDGE",
    aliases: &["KNOWLEDGE EDITS", "EDGES", "MUTATION"],
    title: "Knowledge edits",
    intro: &[
        "These edit what the model retrieves, through an overlay on the bound vindex or container.",
        "They take effect at once: DESCRIBE and INFER in the same session see them. They need a",
        "tokenizer. The base files are never modified.",
        "Without BEGIN PATCH they start an anonymous recording, which BEGIN PATCH then adopts.",
        "COMPILE refuses knowledge edits and stored-weight ops in one container, and while both are",
        "pending INFER and TRACE are refused too (HELP WEIGHTS;).",
    ],
};

pub(super) static INSERT: StatementHelp = StatementHelp {
    name: "INSERT",
    aliases: &["INSERT INTO EDGES"],
    group: HelpGroup::Knowledge,
    syntax: &[
        "INSERT INTO EDGES (entity, relation, target) VALUES (\"<entity>\", \"<relation>\", \"<target>\") [AT LAYER <n>] [CONFIDENCE <f>] [ALPHA <f>] [MODE {KNN | COMPOSE}];",
    ],
    does: &[
        "Adds a fact. MODE KNN (the default) stores a retrieval key that overrides INFER's top",
        "prediction when it matches. MODE COMPOSE installs an FFN slot that takes part in the",
        "forward pass; ALPHA applies to COMPOSE only.",
    ],
    examples: &[
        "INSERT INTO EDGES (entity, relation, target) VALUES (\"Atlantis\", \"capital\", \"Poseidon\");",
        "INSERT INTO EDGES (entity, relation, target) VALUES (\"Atlantis\", \"capital\", \"Poseidon\") AT LAYER 26 ALPHA 0.1 MODE COMPOSE;",
    ],
    needs: &[Need::Backend, Need::Tokenizer],
    effect: &["At once, in this session. Lost at process end unless saved as a patch or compiled."],
    operands: &[],
    refusals: &[
        "The container carries no tokenizer.json.",
        "MODE COMPOSE while a stored-weight op is pending: the install is computed on the stored weights.",
        "On a remote backend ALPHA and MODE are not forwarded.",
    ],
    related: &["DESCRIBE", "INFER", "BEGIN PATCH", "REBALANCE", "COMPILE"],
};

pub(super) static DELETE: StatementHelp = StatementHelp {
    name: "DELETE",
    aliases: &["DELETE FROM EDGES"],
    group: HelpGroup::Knowledge,
    syntax: &["DELETE FROM EDGES WHERE <condition> [AND <condition>]...;"],
    does: &[
        "Removes the edges a WHERE clause selects. Run SELECT * FROM EDGES with the same clause",
        "first: it lists exactly the slots DELETE targets. layer and feature together name one slot.",
    ],
    examples: &["DELETE FROM EDGES WHERE layer = 26 AND feature = 8821;"],
    needs: &[Need::Backend, Need::Tokenizer],
    effect: &["At once, in this session. A statement that matches nothing changes nothing."],
    operands: &[],
    refusals: &[
        "entity together with relation: list with SELECT, then address each slot by layer and feature.",
        "A layer and feature pair that names no slot.",
        "A relation condition when no relation labels are loaded.",
        "Any condition outside the WHERE grammar (HELP SELECT;).",
    ],
    related: &["SELECT", "UPDATE", "BEGIN PATCH"],
};

pub(super) static UPDATE: StatementHelp = StatementHelp {
    name: "UPDATE",
    aliases: &["UPDATE EDGES"],
    group: HelpGroup::Knowledge,
    syntax: &["UPDATE EDGES SET <field> = <value> [, <field> = <value>]... WHERE <condition> [AND <condition>]...;"],
    does: &[
        "Changes the edges a WHERE clause selects. Settable fields are target (a non-empty string)",
        "and confidence (a finite number). This is not UPDATE WEIGHTS, which edits stored weights.",
    ],
    examples: &["UPDATE EDGES SET target = \"London\", confidence = 0.95 WHERE layer = 26 AND feature = 8821;"],
    needs: &[Need::Backend, Need::Tokenizer],
    effect: &["At once, in this session."],
    operands: &[],
    refusals: &[
        "An unknown field, a value of the wrong type, or a field set twice: nothing is updated.",
        "The same WHERE refusals as DELETE.",
    ],
    related: &["SELECT", "DELETE", "UPDATE WEIGHTS"],
};

pub(super) static MERGE: StatementHelp = StatementHelp {
    name: "MERGE",
    aliases: &[],
    group: HelpGroup::Knowledge,
    syntax: &[
        "MERGE \"<source>\" [INTO \"<target>\"] [ON CONFLICT {KEEP_SOURCE | KEEP_TARGET | HIGHEST_CONFIDENCE}];",
    ],
    does: &[
        "Merges another vindex's edges into the bound session's overlay. Each record written is",
        "recorded as an update, so SAVE PATCH includes it.",
    ],
    examples: &["MERGE \"medical-knowledge.vindex\" ON CONFLICT HIGHEST_CONFIDENCE;"],
    needs: &[Need::Backend],
    effect: &["At once, in this session. No file is written."],
    operands: &[],
    refusals: &[
        "The source does not exist, or was built from another model.",
        "INTO naming anything but the bound vindex.",
    ],
    related: &["APPLY PATCH", "DIFF"],
};

pub(super) static REBALANCE: StatementHelp = StatementHelp {
    name: "REBALANCE",
    aliases: &[],
    group: HelpGroup::Knowledge,
    syntax: &["REBALANCE [UNTIL CONVERGED] [MAX <n>] [FLOOR <p>] [CEILING <p>];"],
    does: &[
        "Adjusts every COMPOSE-mode INSERT of the session until each fact's target probability is",
        "between FLOOR and CEILING, for at most MAX rounds. KNN inserts need none; with no COMPOSE",
        "insert it says so and does nothing.",
    ],
    examples: &["REBALANCE;", "REBALANCE MAX 16 FLOOR 0.3 CEILING 0.9;"],
    needs: &[Need::Other(
        "COMPOSE-mode inserts in the session. With none, bound or not, it reports nothing to rebalance.",
    )],
    effect: &["At once, in this session."],
    operands: &[],
    refusals: &[],
    related: &["INSERT", "COMPILE"],
};

pub(super) static COMPACT_MINOR: StatementHelp = StatementHelp {
    name: "COMPACT MINOR",
    aliases: &[],
    group: HelpGroup::Knowledge,
    syntax: &["COMPACT MINOR;"],
    does: &[
        "Promotes the session's KNN entries to COMPOSE edges (VINDEX2 storage tiers L0 to L1).",
    ],
    examples: &["COMPACT MINOR;"],
    needs: &[Need::Vindex2],
    effect: &["At once, in this session."],
    operands: &[],
    refusals: &["A VINDEX3 container is bound: its compaction is COMPACT INTO VINDEX."],
    related: &[
        "COMPACT MAJOR",
        "SHOW COMPACT STATUS",
        "COMPACT INTO VINDEX",
    ],
};

pub(super) static COMPACT_MAJOR: StatementHelp = StatementHelp {
    name: "COMPACT MAJOR",
    aliases: &[],
    group: HelpGroup::Knowledge,
    syntax: &["COMPACT MAJOR [FULL] [WITH LAMBDA = <f>];"],
    does: &[
        "Promotes COMPOSE edges to the MEMIT store (VINDEX2 storage tiers L1 to L2) and persists that",
        "store into the bound vindex directory.",
    ],
    examples: &["COMPACT MAJOR;", "COMPACT MAJOR FULL WITH LAMBDA = 0.001;"],
    needs: &[Need::Vindex2],
    effect: &["Writes into the bound vindex directory: the one statement here that does."],
    operands: &[],
    refusals: &[
        "A VINDEX3 container is bound.",
        "In a batch, after any earlier statement failed (HELP BATCH;).",
    ],
    related: &["COMPACT MINOR", "SHOW COMPACT STATUS"],
};

pub(super) static SHOW_COMPACT_STATUS: StatementHelp = StatementHelp {
    name: "SHOW COMPACT STATUS",
    aliases: &[],
    group: HelpGroup::Knowledge,
    syntax: &["SHOW COMPACT STATUS;"],
    does: &["Prints how many entries each VINDEX2 storage tier (L0, L1, L2) holds."],
    examples: &["SHOW COMPACT STATUS;"],
    needs: &[Need::Vindex2],
    effect: &["Read-only."],
    operands: &[],
    refusals: &["A VINDEX3 container is bound."],
    related: &["COMPACT MINOR", "COMPACT MAJOR"],
};
