//! Which statements write an artifact (a container, model, patch file,
//! saved trace, or state persisted into the bound vindex) to disk. Caches
//! filled as a side effect, such as `USE "hf://..."` downloading into the
//! model cache, are not artifacts.
//!
//! A batch uses this to refuse every write after a statement has failed
//! (see [`crate::repl::run_batch_outcome`]): a write that follows a failure
//! would be built from a session the batch's author did not intend, such as
//! a `COMPILE` after a refused `APPLY PATCH`, and its output would look like
//! success.
//!
//! The match is exhaustive with no wildcard arm, so a new statement cannot
//! be added without deciding whether it writes.

use crate::ast::Statement;

impl Statement {
    /// Whether executing this statement can write an artifact to disk.
    /// Statements that only change session state (staging an edit, applying
    /// a patch to the overlay, merging into it) do not: nothing they do
    /// leaves the process until a writing statement runs.
    pub fn writes_artifact(&self) -> bool {
        match self {
            Statement::Extract { .. }
            | Statement::Compile { .. }
            | Statement::CompactInto { .. }
            // Persists the MEMIT store into the bound vindex directory.
            | Statement::CompactMajor { .. }
            | Statement::SavePatch => true,
            Statement::Diff { into_patch, .. } => into_patch.is_some(),
            Statement::Trace { save, .. } => save.is_some(),
            Statement::Pipe { left, right } => left.writes_artifact() || right.writes_artifact(),
            Statement::Use { .. }
            | Statement::Walk { .. }
            | Statement::Infer { .. }
            | Statement::Select { .. }
            | Statement::Describe { .. }
            | Statement::Explain { .. }
            | Statement::Insert { .. }
            | Statement::Delete { .. }
            | Statement::Update { .. }
            | Statement::UpdateWeights { .. }
            | Statement::RemoveDirection { .. }
            | Statement::AddLowRank { .. }
            | Statement::Merge { .. }
            | Statement::Rebalance { .. }
            | Statement::ShowRelations { .. }
            | Statement::ShowLayers { .. }
            | Statement::ShowFeatures { .. }
            | Statement::ShowEntities { .. }
            | Statement::ShowModels
            | Statement::ShowComponents
            | Statement::ShowRepresentations { .. }
            | Statement::ShowProvenance { .. }
            | Statement::ShowAuthority
            | Statement::VerifyRepresentation { .. }
            | Statement::VerifyOperator { .. }
            | Statement::VerifyPatch { .. }
            | Statement::Stats { .. }
            | Statement::ShowCompactStatus
            | Statement::CompactMinor
            | Statement::BeginPatch { .. }
            | Statement::ApplyPatch { .. }
            | Statement::ShowPatches
            | Statement::RemovePatch { .. } => false,
        }
    }
}
