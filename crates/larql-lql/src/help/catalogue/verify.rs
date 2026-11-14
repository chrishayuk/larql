//! Verify: ask an external verifier whether a container meets a contract.

use crate::help::entry::{GroupHelp, Need, StatementHelp};
use crate::help::topic::HelpGroup;

pub(super) static GROUP: GroupHelp = GroupHelp {
    name: "VERIFY",
    aliases: &["VERIFICATION", "CONTRACTS"],
    title: "Verify",
    intro: &[
        "LQL never evaluates a contract. Each VERIFY statement runs an external verifier program and",
        "reports its answer. With no verifier configured the statement is refused; it never passes by",
        "default (HELP VERIFIER;).",
        "They take paths, not the bound session, and need nothing bound. An answer is one of:",
        "  satisfied            the report is printed.",
        "  violated             the report is printed; the statement succeeds; larql lql exits 2",
        "                       if nothing else failed.",
        "  could_not_evaluate   an error: larql lql exits 1.",
        "Which statement fits a compiled weight edit depends on its ledger (HELP COMPILE;):",
        "  weight-patches/v4  any ADD LOW RANK                  VERIFY LEDGER",
        "  weight-patches/v3  REMOVE DIRECTION, no ADD LOW RANK  VERIFY PATCH with the saved .vlp,",
        "                                                       if it holds only REMOVE DIRECTION ops",
        "  weight-patches/v2  UPDATE WEIGHTS only               neither statement judges it",
        "A result is a numerical measurement by a program LQL does not authenticate. It is not a proof.",
    ],
};

/// What every VERIFY statement says about its outcome.
const VERIFY_EFFECT: &[&str] = &[
    "Read-only. \"violated\" is an answer, not a failure: the statement returns its report, later",
    "statements still run and still write, and larql lql exits 2 when nothing else failed.",
    "\"could_not_evaluate\", a verifier that fails, and an answer that is malformed or about another",
    "question are errors (exit 1).",
];

pub(super) static VERIFY_PATCH: StatementHelp = StatementHelp {
    name: "VERIFY PATCH",
    aliases: &[],
    group: HelpGroup::Verify,
    syntax: &[
        "VERIFY PATCH \"<patch.vlp>\" USING CONTRACT \"<id>\" FROM \"<base>\" AGAINST \"<container>\";",
    ],
    does: &[
        "Asks the verifier whether <container> holds what the patch declares of <base>. The verifier",
        "recomputes the expected weights from the patch's operands; it does not read the ledger.",
        "The repository's verifier knows one contract here, STRUCTURED_REMOVE_DIRECTION, for patches",
        "that hold only REMOVE DIRECTION ops.",
    ],
    examples: &[
        "VERIFY PATCH \"removal.vlp\" USING CONTRACT \"STRUCTURED_REMOVE_DIRECTION\" FROM \"base.vindex3\" AGAINST \"edited.vindex3\";",
    ],
    needs: &[
        Need::Verifier,
        Need::Nothing,
        Need::Other("The patch file (SAVE PATCH), the base container and the compiled container, as paths."),
    ],
    effect: VERIFY_EFFECT,
    operands: &[],
    refusals: &[
        "No verifier is configured.",
        "The patch is not a file, or the base or the container is not a directory.",
        "could_not_evaluate: a patch with any op other than REMOVE DIRECTION, a patch bound to another base, an unknown contract.",
        "It does not judge the derived (quantised) representations, the op digest or the operand's origin.",
    ],
    related: &["VERIFY LEDGER", "REMOVE DIRECTION", "SAVE PATCH", "VERIFIER", "EXIT CODES"],
};

pub(super) static VERIFY_LEDGER: StatementHelp = StatementHelp {
    name: "VERIFY LEDGER",
    aliases: &[],
    group: HelpGroup::Verify,
    syntax: &["VERIFY LEDGER \"<container>\" USING CONTRACT \"<id>\" FROM \"<base>\";"],
    does: &[
        "Asks the verifier whether <container> is what its own weight-patch ledger declares of",
        "<base>. The verifier recomputes every op the ledger records from the base's weights. The",
        "repository's verifier knows one contract here, WEIGHT_PATCHES_V4. It gives two verdicts,",
        "numerical satisfaction and ledger integrity; the statement is satisfied only when both are.",
    ],
    examples: &[
        "VERIFY LEDGER \"edited.vindex3\" USING CONTRACT \"WEIGHT_PATCHES_V4\" FROM \"base.vindex3\";",
    ],
    needs: &[
        Need::Verifier,
        Need::Nothing,
        Need::Other("The compiled container and the base it was compiled from, as paths."),
    ],
    effect: VERIFY_EFFECT,
    operands: &[],
    refusals: &[
        "No verifier is configured.",
        "The container or the base is not a directory.",
        "could_not_evaluate: a ledger that is not weight-patches/v4. A v2 or v3 ledger is not judged.",
        "No patch file is involved: LQL passes the verifier the contract, the base and the container only.",
        "It judges the canonical BF16 weights. The re-derived (quantised) representations are not judged.",
    ],
    related: &["VERIFY PATCH", "ADD LOW RANK", "COMPILE", "VERIFIER", "EXIT CODES"],
};

pub(super) static VERIFY_REPRESENTATION: StatementHelp = StatementHelp {
    name: "VERIFY REPRESENTATION",
    aliases: &[],
    group: HelpGroup::Verify,
    syntax: &[
        "VERIFY REPRESENTATION \"<container>\" USING CONTRACT \"<id>\" AGAINST \"<source>\";",
    ],
    does: &[
        "Asks the verifier to measure one container against a numerical contract, relative to the",
        "source checkpoint directory its weights came from. The repository's verifier knows Q8_0_T2.",
    ],
    examples: &[
        "VERIFY REPRESENTATION \"model-q8_0.vindex3\" USING CONTRACT \"Q8_0_T2\" AGAINST \"checkpoint\";",
    ],
    needs: &[
        Need::Verifier,
        Need::Nothing,
        Need::Other("The container and the source checkpoint, as directories."),
    ],
    effect: VERIFY_EFFECT,
    operands: &[],
    refusals: &[
        "No verifier is configured.",
        "The container or the source is not a directory.",
        "could_not_evaluate: an unknown contract, or a source the verifier does not pin.",
        "Measured against the unedited checkpoint, a container with a compiled weight edit reports the edited elements.",
    ],
    related: &["VERIFY OPERATOR", "VERIFIER", "EXIT CODES"],
};

pub(super) static VERIFY_OPERATOR: StatementHelp = StatementHelp {
    name: "VERIFY OPERATOR",
    aliases: &[],
    group: HelpGroup::Verify,
    syntax: &[
        "VERIFY OPERATOR \"<tensor>\" USING CONTRACT \"<id>\" IN WITNESS \"<witness.json>\" AGAINST \"<container>\";",
    ],
    does: &[
        "Asks the verifier to judge one recorded kernel call against a numerical contract. The",
        "witness is a file an earlier run wrote (larql vindex3 exec --gemv-witness). The statement",
        "never runs the engine. Contracts: Q8_0_GEMV_SEQ, Q8_0_GEMV_B1.",
    ],
    examples: &[
        "VERIFY OPERATOR \"0.self_attn.q_proj.weight\" USING CONTRACT \"Q8_0_GEMV_SEQ\" IN WITNESS \"witness.json\" AGAINST \"model-q8_0.vindex3\";",
    ],
    needs: &[
        Need::Verifier,
        Need::Nothing,
        Need::Other("The witness file and the container it was recorded on, as paths."),
    ],
    effect: VERIFY_EFFECT,
    operands: &[],
    refusals: &[
        "No verifier is configured.",
        "The witness is not a file, or the container is not a directory.",
        "could_not_evaluate: the verifier cannot tie the witness to the operator and the container's bytes.",
    ],
    related: &["VERIFY REPRESENTATION", "VERIFIER", "EXIT CODES"],
};
