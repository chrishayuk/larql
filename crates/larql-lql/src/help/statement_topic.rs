//! Which help topic documents each statement.
//!
//! The match is exhaustive with no wildcard arm, so a new statement does not
//! compile until it names a topic, and [`super::catalogue::statement`] does
//! not compile until that topic has an entry. `tests/help/catalogue.rs`
//! parses every example of every entry and checks it lands on the topic
//! that holds it.

use crate::ast::Statement;

use super::topic::StatementTopic;

impl Statement {
    /// The help topic that documents this statement: what `HELP <verb>;`
    /// prints.
    pub fn help_topic(&self) -> StatementTopic {
        match self {
            Statement::Extract { .. } => StatementTopic::Extract,
            Statement::Compile { .. } => StatementTopic::Compile,
            Statement::Diff { .. } => StatementTopic::Diff,
            Statement::Use { .. } => StatementTopic::Use,
            Statement::Walk { .. } => StatementTopic::Walk,
            Statement::Infer { .. } => StatementTopic::Infer,
            Statement::Select { .. } => StatementTopic::Select,
            Statement::Describe { .. } => StatementTopic::Describe,
            Statement::Explain { .. } => StatementTopic::Explain,
            Statement::Insert { .. } => StatementTopic::Insert,
            Statement::Delete { .. } => StatementTopic::Delete,
            Statement::Update { .. } => StatementTopic::Update,
            Statement::UpdateWeights { .. } => StatementTopic::UpdateWeights,
            Statement::RemoveDirection { .. } => StatementTopic::RemoveDirection,
            Statement::AddLowRank { .. } => StatementTopic::AddLowRank,
            Statement::Merge { .. } => StatementTopic::Merge,
            Statement::Rebalance { .. } => StatementTopic::Rebalance,
            Statement::ShowRelations { .. } => StatementTopic::ShowRelations,
            Statement::ShowLayers { .. } => StatementTopic::ShowLayers,
            Statement::ShowFeatures { .. } => StatementTopic::ShowFeatures,
            Statement::ShowEntities { .. } => StatementTopic::ShowEntities,
            Statement::ShowModels => StatementTopic::ShowModels,
            Statement::ShowComponents => StatementTopic::ShowComponents,
            Statement::ShowRepresentations { .. } => StatementTopic::ShowRepresentations,
            Statement::ShowTensors { .. } => StatementTopic::ShowTensors,
            Statement::ExportWeight { .. } => StatementTopic::ExportWeight,
            Statement::ShowProvenance { .. } => StatementTopic::ShowProvenance,
            Statement::ShowAuthority => StatementTopic::ShowAuthority,
            Statement::VerifyRepresentation { .. } => StatementTopic::VerifyRepresentation,
            Statement::VerifyOperator { .. } => StatementTopic::VerifyOperator,
            Statement::VerifyPatch { .. } => StatementTopic::VerifyPatch,
            Statement::VerifyLedger { .. } => StatementTopic::VerifyLedger,
            Statement::DiscardWeightEdits => StatementTopic::DiscardWeightEdits,
            Statement::Stats { .. } => StatementTopic::Stats,
            Statement::ShowCompactStatus => StatementTopic::ShowCompactStatus,
            Statement::CompactInto { .. } => StatementTopic::CompactInto,
            Statement::CompactMinor => StatementTopic::CompactMinor,
            Statement::CompactMajor { .. } => StatementTopic::CompactMajor,
            Statement::BeginPatch { .. } => StatementTopic::BeginPatch,
            Statement::SavePatch => StatementTopic::SavePatch,
            Statement::ApplyPatch { .. } => StatementTopic::ApplyPatch,
            Statement::ShowPatches => StatementTopic::ShowPatches,
            Statement::RemovePatch { .. } => StatementTopic::RemovePatch,
            Statement::Trace { .. } => StatementTopic::Trace,
            Statement::Help { .. } => StatementTopic::Help,
            Statement::Pipe { .. } => StatementTopic::Pipe,
        }
    }
}
