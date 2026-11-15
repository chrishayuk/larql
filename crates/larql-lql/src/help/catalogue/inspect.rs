//! Inspect: read what the bound vindex or container declares and holds.

use crate::help::entry::{GroupHelp, Need, StatementHelp};
use crate::help::topic::HelpGroup;

pub(super) static GROUP: GroupHelp = GroupHelp {
    name: "INSPECT",
    aliases: &["INTROSPECTION", "BROWSE"],
    title: "Inspect",
    intro: &[
        "Read-only statements. STATS, SHOW COMPONENTS / REPRESENTATIONS / PROVENANCE / AUTHORITY /",
        "LAYERS / MODELS and DIFF answer from what the container declares and need no tokenizer.",
        "SHOW RELATIONS / FEATURES / ENTITIES, DESCRIBE and SELECT decode token ids and need one.",
    ],
};

/// The WHERE grammar every statement over EDGES shares.
const WHERE_EDGES: &str =
    "WHERE takes layer = <n>, feature = <n>, entity = \"<s>\", relation {= | != | LIKE} \"<s>\" and score <op> <number>, joined by AND, one per field.";

pub(super) static STATS: StatementHelp = StatementHelp {
    name: "STATS",
    aliases: &[],
    group: HelpGroup::Inspect,
    syntax: &["STATS [\"<vindex>\"];"],
    does: &[
        "Prints the bound model's summary: generation, layers, attention geometry, tokenizer",
        "presence and the statements the binding serves. The optional path is parsed and ignored:",
        "STATS always describes the bound session.",
    ],
    examples: &["STATS;"],
    needs: &[Need::Backend],
    effect: &["Read-only."],
    operands: &[],
    refusals: &["Nothing is bound: \"No backend loaded\", even when a path is given."],
    related: &["USE", "SHOW LAYERS", "SHOW COMPONENTS"],
};

pub(super) static SHOW_COMPONENTS: StatementHelp = StatementHelp {
    name: "SHOW COMPONENTS",
    aliases: &[],
    group: HelpGroup::Inspect,
    syntax: &["SHOW COMPONENTS;"],
    does: &["Lists the components the container's system graph declares: role, layers, hidden size, attention."],
    examples: &["SHOW COMPONENTS;"],
    needs: &[Need::Vindex3],
    effect: &["Read-only."],
    operands: &[],
    refusals: &["The session is bound to something other than a VINDEX3 container."],
    related: &["SHOW REPRESENTATIONS", "SHOW PROVENANCE", "SHOW AUTHORITY"],
};

pub(super) static SHOW_REPRESENTATIONS: StatementHelp = StatementHelp {
    name: "SHOW REPRESENTATIONS",
    aliases: &[],
    group: HelpGroup::Inspect,
    syntax: &["SHOW REPRESENTATIONS [\"<object>\"];"],
    does: &[
        "Lists the representations physically present in the container: object, encoding (BF16,",
        "Q8_0, ...), tensor count and bytes. The optional string keeps entries whose object contains it.",
    ],
    examples: &["SHOW REPRESENTATIONS;", "SHOW REPRESENTATIONS \"decoder\";"],
    needs: &[Need::Vindex3],
    effect: &["Read-only."],
    operands: &[],
    refusals: &["The session is bound to something other than a VINDEX3 container."],
    related: &["SHOW PROVENANCE", "SHOW COMPONENTS"],
};

pub(super) static SHOW_PROVENANCE: StatementHelp = StatementHelp {
    name: "SHOW PROVENANCE",
    aliases: &[],
    group: HelpGroup::Inspect,
    syntax: &["SHOW PROVENANCE [\"<object>\"];"],
    does: &[
        "Prints, per representation, its segment file, payload and segment SHA-256, and what it was",
        "compiled from. Use it to see that a COMPILE changed the segments it should have.",
    ],
    examples: &["SHOW PROVENANCE;", "SHOW PROVENANCE \"decoder\";"],
    needs: &[Need::Vindex3],
    effect: &["Read-only."],
    operands: &[],
    refusals: &["The session is bound to something other than a VINDEX3 container."],
    related: &["SHOW REPRESENTATIONS", "DIFF", "COMPILE"],
};

pub(super) static SHOW_AUTHORITY: StatementHelp = StatementHelp {
    name: "SHOW AUTHORITY",
    aliases: &[],
    group: HelpGroup::Inspect,
    syntax: &["SHOW AUTHORITY;"],
    does: &["Prints the container's authority declaration (canonical or not) and the profiles it declares."],
    examples: &["SHOW AUTHORITY;"],
    needs: &[Need::Vindex3],
    effect: &["Read-only."],
    operands: &[],
    refusals: &["The session is bound to something other than a VINDEX3 container."],
    related: &["SHOW COMPONENTS", "STATS"],
};

pub(super) static SHOW_LAYERS: StatementHelp = StatementHelp {
    name: "SHOW LAYERS",
    aliases: &[],
    group: HelpGroup::Inspect,
    syntax: &["SHOW LAYERS [[RANGE] <a>-<b>];"],
    does: &["Prints one row per layer. On a VINDEX3 container: attention mode, window, heads and head size, from the plan."],
    examples: &["SHOW LAYERS;", "SHOW LAYERS RANGE 0-1;"],
    needs: &[Need::Backend],
    effect: &["Read-only."],
    operands: &[],
    refusals: &["Nothing is bound.", "A range whose start is above its end."],
    related: &["STATS", "EXPLAIN"],
};

pub(super) static SHOW_MODELS: StatementHelp = StatementHelp {
    name: "SHOW MODELS",
    aliases: &[],
    group: HelpGroup::Inspect,
    syntax: &["SHOW MODELS;"],
    does: &["Lists the vindexes and containers in the process's working directory: size, generation, layers."],
    examples: &["SHOW MODELS;"],
    needs: &[Need::Nothing],
    effect: &["Read-only."],
    operands: &[],
    refusals: &[],
    related: &["USE"],
};

pub(super) static SHOW_RELATIONS: StatementHelp = StatementHelp {
    name: "SHOW RELATIONS",
    aliases: &[],
    group: HelpGroup::Inspect,
    syntax: &["SHOW RELATIONS [AT LAYER <n>] [WITH EXAMPLES] [VERBOSE | BRIEF | RAW];"],
    does: &["Lists the relation types found among the bound model's features."],
    examples: &[
        "SHOW RELATIONS;",
        "SHOW RELATIONS AT LAYER 26 WITH EXAMPLES;",
    ],
    needs: &[Need::Backend, Need::Tokenizer],
    effect: &["Read-only."],
    operands: &[],
    refusals: &["The container carries no tokenizer.json."],
    related: &["DESCRIBE", "SELECT", "SHOW FEATURES"],
};

pub(super) static SHOW_FEATURES: StatementHelp = StatementHelp {
    name: "SHOW FEATURES",
    aliases: &[],
    group: HelpGroup::Inspect,
    syntax: &["SHOW FEATURES <layer> [WHERE <condition> [AND <condition>]...] [LIMIT <n>];"],
    does: &[
        "Lists the features of one layer with their top tokens.",
        "WHERE takes entity = \"<s>\" (alias token) and score <op> <number> (alias confidence).",
    ],
    examples: &[
        "SHOW FEATURES 26 LIMIT 20;",
        "SHOW FEATURES 26 WHERE token = \"Paris\";",
    ],
    needs: &[Need::Backend, Need::Tokenizer],
    effect: &["Read-only."],
    operands: &[],
    refusals: &[
        "The container carries no tokenizer.json.",
        "A WHERE condition outside the accepted ones; none is dropped silently.",
    ],
    related: &["SELECT", "SHOW ENTITIES"],
};

pub(super) static SHOW_ENTITIES: StatementHelp = StatementHelp {
    name: "SHOW ENTITIES",
    aliases: &[],
    group: HelpGroup::Inspect,
    syntax: &["SHOW ENTITIES [[AT LAYER] <n>] [LIMIT <n>];"],
    does: &["Lists the distinct entity-like tokens across the loaded layers, or at one layer."],
    examples: &["SHOW ENTITIES;", "SHOW ENTITIES AT LAYER 26 LIMIT 50;"],
    needs: &[Need::Backend, Need::Tokenizer],
    effect: &["Read-only."],
    operands: &[],
    refusals: &["The container carries no tokenizer.json."],
    related: &["DESCRIBE", "SHOW FEATURES"],
};

pub(super) static DESCRIBE: StatementHelp = StatementHelp {
    name: "DESCRIBE",
    aliases: &[],
    group: HelpGroup::Inspect,
    syntax: &[
        "DESCRIBE \"<entity>\" [AT LAYER <n>] [RELATIONS ONLY] [VERBOSE | BRIEF | RAW] [SYNTAX | KNOWLEDGE | OUTPUT | ALL LAYERS];",
    ],
    does: &[
        "Shows what the model's features associate with an entity, grouped by layer band. The default",
        "is the knowledge band, in BRIEF form. A knowledge edit (INSERT) made in the session is included.",
    ],
    examples: &["DESCRIBE \"France\";", "DESCRIBE \"France\" ALL LAYERS VERBOSE;"],
    needs: &[Need::Backend, Need::Tokenizer],
    effect: &["Read-only."],
    operands: &[],
    refusals: &["The container carries no tokenizer.json."],
    related: &["WALK", "SELECT", "INSERT"],
};

pub(super) static SELECT: StatementHelp = StatementHelp {
    name: "SELECT",
    aliases: &[],
    group: HelpGroup::Inspect,
    syntax: &[
        "SELECT {* | <field>, ...} FROM {EDGES | FEATURES | ENTITIES} [NEAREST TO \"<entity>\" AT LAYER <n>] [WHERE <condition> [AND <condition>]...] [ORDER BY <field> [ASC | DESC]] [LIMIT <n>];",
    ],
    does: &[
        "Lists edges, features or entities. Over EDGES it is the preview of a mutation: DELETE and",
        "UPDATE with the same WHERE clause target exactly the slots it lists.",
        WHERE_EDGES,
    ],
    examples: &[
        "SELECT * FROM EDGES WHERE entity = \"France\" LIMIT 10;",
        "SELECT * FROM FEATURES WHERE layer = 0 LIMIT 8;",
    ],
    needs: &[Need::Backend, Need::Tokenizer],
    effect: &["Read-only. The listing is paged (20 rows unless LIMIT says otherwise)."],
    operands: &[],
    refusals: &[
        "The container carries no tokenizer.json.",
        "A WHERE clause with any condition outside the grammar: the whole statement is refused.",
        "WHERE together with NEAREST TO.",
    ],
    related: &["DESCRIBE", "DELETE", "UPDATE"],
};

pub(super) static DIFF: StatementHelp = StatementHelp {
    name: "DIFF",
    aliases: &["DIFF INTO PATCH"],
    group: HelpGroup::Inspect,
    syntax: &[
        "DIFF {CURRENT | \"<a>\"} {CURRENT | \"<b>\"} [PHYSICAL] [LAYER <n>] [RELATION \"<r>\"] [LIMIT <n>] [INTO PATCH \"<file.vlp>\"];",
    ],
    does: &[
        "Compares two vindexes or containers. The default report is logical: knowledge edges and",
        "feature slots that differ. PHYSICAL (VINDEX3) lists the segments whose hash differs.",
        "INTO PATCH writes the difference as a .vlp patch (VINDEX2 sides only).",
    ],
    examples: &[
        "DIFF \"base.vindex3\" \"edited.vindex3\";",
        "DIFF \"base.vindex3\" \"edited.vindex3\" PHYSICAL;",
        "DIFF \"base.vindex\" CURRENT INTO PATCH \"changes.vlp\";",
    ],
    needs: &[Need::Other(
        "A bound vindex or container when either side is CURRENT.",
    )],
    effect: &[
        "Read-only without INTO PATCH. CURRENT is the bound container as stored: stored-weight ops",
        "that are staged and not compiled are not part of it, and the reply says how many are pending.",
        "Compile them, then DIFF the base against the compiled container.",
    ],
    operands: &[],
    refusals: &[
        "INTO PATCH on a VINDEX3 container.",
        "PHYSICAL with a VINDEX2 side.",
        "INTO PATCH in a batch after any earlier statement failed (HELP BATCH;).",
    ],
    related: &["COMPILE", "SHOW PROVENANCE", "APPLY PATCH"],
};
