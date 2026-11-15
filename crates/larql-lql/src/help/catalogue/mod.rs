//! The help catalogue: one static entry per topic, kept in the file of its
//! group. The three lookups below are exhaustive matches with no wildcard
//! arm, so a topic cannot be declared without an entry.

mod abandon;
mod bind;
mod compile;
mod inspect;
mod knowledge;
mod patches;
mod process;
mod run;
mod verify;
mod weights;

use super::entry::{ConceptHelp, GroupHelp, StatementHelp};
use super::topic::{ConceptTopic, HelpGroup, StatementTopic};

/// The page of one statement kind.
pub fn statement(topic: StatementTopic) -> &'static StatementHelp {
    match topic {
        StatementTopic::Use => &bind::USE,
        StatementTopic::Extract => &bind::EXTRACT,
        StatementTopic::Stats => &inspect::STATS,
        StatementTopic::ShowComponents => &inspect::SHOW_COMPONENTS,
        StatementTopic::ShowRepresentations => &inspect::SHOW_REPRESENTATIONS,
        StatementTopic::ShowTensors => &inspect::SHOW_TENSORS,
        StatementTopic::ExportWeight => &inspect::EXPORT_WEIGHT,
        StatementTopic::ShowProvenance => &inspect::SHOW_PROVENANCE,
        StatementTopic::ShowAuthority => &inspect::SHOW_AUTHORITY,
        StatementTopic::ShowLayers => &inspect::SHOW_LAYERS,
        StatementTopic::ShowModels => &inspect::SHOW_MODELS,
        StatementTopic::ShowRelations => &inspect::SHOW_RELATIONS,
        StatementTopic::ShowFeatures => &inspect::SHOW_FEATURES,
        StatementTopic::ShowEntities => &inspect::SHOW_ENTITIES,
        StatementTopic::Describe => &inspect::DESCRIBE,
        StatementTopic::Select => &inspect::SELECT,
        StatementTopic::Diff => &inspect::DIFF,
        StatementTopic::Infer => &run::INFER,
        StatementTopic::Walk => &run::WALK,
        StatementTopic::Explain => &run::EXPLAIN,
        StatementTopic::Trace => &run::TRACE,
        StatementTopic::Capture => &run::CAPTURE,
        StatementTopic::UpdateWeights => &weights::UPDATE_WEIGHTS,
        StatementTopic::RemoveDirection => &weights::REMOVE_DIRECTION,
        StatementTopic::AddLowRank => &weights::ADD_LOW_RANK,
        StatementTopic::BeginPatch => &patches::BEGIN_PATCH,
        StatementTopic::SavePatch => &patches::SAVE_PATCH,
        StatementTopic::ApplyPatch => &patches::APPLY_PATCH,
        StatementTopic::ShowPatches => &patches::SHOW_PATCHES,
        StatementTopic::Compile => &compile::COMPILE,
        StatementTopic::CompactInto => &compile::COMPACT_INTO,
        StatementTopic::VerifyPatch => &verify::VERIFY_PATCH,
        StatementTopic::VerifyLedger => &verify::VERIFY_LEDGER,
        StatementTopic::VerifyRepresentation => &verify::VERIFY_REPRESENTATION,
        StatementTopic::VerifyOperator => &verify::VERIFY_OPERATOR,
        StatementTopic::DiscardWeightEdits => &abandon::DISCARD_WEIGHT_EDITS,
        StatementTopic::RemovePatch => &abandon::REMOVE_PATCH,
        StatementTopic::Insert => &knowledge::INSERT,
        StatementTopic::Delete => &knowledge::DELETE,
        StatementTopic::Update => &knowledge::UPDATE,
        StatementTopic::Merge => &knowledge::MERGE,
        StatementTopic::Rebalance => &knowledge::REBALANCE,
        StatementTopic::CompactMinor => &knowledge::COMPACT_MINOR,
        StatementTopic::CompactMajor => &knowledge::COMPACT_MAJOR,
        StatementTopic::ShowCompactStatus => &knowledge::SHOW_COMPACT_STATUS,
        StatementTopic::Help => &process::HELP,
        StatementTopic::Pipe => &process::PIPE,
    }
}

/// The page of one process topic.
pub fn concept(topic: ConceptTopic) -> &'static ConceptHelp {
    match topic {
        ConceptTopic::Batch => &process::BATCH,
        ConceptTopic::ExitCodes => &process::EXIT_CODES,
        ConceptTopic::Verifier => &process::VERIFIER,
    }
}

/// The heading and introduction of one group.
pub fn group(group: HelpGroup) -> &'static GroupHelp {
    match group {
        HelpGroup::Bind => &bind::GROUP,
        HelpGroup::Inspect => &inspect::GROUP,
        HelpGroup::Run => &run::GROUP,
        HelpGroup::Weights => &weights::GROUP,
        HelpGroup::Patches => &patches::GROUP,
        HelpGroup::Compile => &compile::GROUP,
        HelpGroup::Verify => &verify::GROUP,
        HelpGroup::Abandon => &abandon::GROUP,
        HelpGroup::Knowledge => &knowledge::GROUP,
        HelpGroup::Process => &process::GROUP,
    }
}
