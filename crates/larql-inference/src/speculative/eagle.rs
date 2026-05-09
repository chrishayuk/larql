//! EAGLE-style draft head — phase 1 stub.
//!
//! Real implementation per design.md §2.1: a 1-layer transformer
//! block + LM head loaded from a separate checkpoint, sharing
//! tokenization and embedding dim with the target. This stub
//! returns an explicit "not yet implemented" error so phase 2/3
//! can wire the dispatch path without depending on the trained
//! checkpoint.

use super::{DraftToken, Drafter, TokenId};

#[derive(Clone, Debug)]
pub struct EagleDraftHead {
    /// Path the head was loaded from (or would be). Stored so we
    /// can surface a useful error before training is done.
    checkpoint_path: Option<std::path::PathBuf>,
}

impl EagleDraftHead {
    /// Construct without a real checkpoint. `propose` will return
    /// an empty draft, which the caller MUST interpret as "no
    /// speculation possible — fall back to non-speculative".
    pub fn unloaded() -> Self {
        Self {
            checkpoint_path: None,
        }
    }

    /// Phase 1: stub that records the path. Returns Ok so the
    /// downstream wiring code paths can compile and run; the
    /// actual checkpoint reader lands in phase 1 task 1.2.
    pub fn from_checkpoint(path: impl Into<std::path::PathBuf>) -> Result<Self, EagleError> {
        Ok(Self {
            checkpoint_path: Some(path.into()),
        })
    }

    pub fn checkpoint_path(&self) -> Option<&std::path::Path> {
        self.checkpoint_path.as_deref()
    }
}

impl Drafter for EagleDraftHead {
    fn propose(&mut self, _h_target: &[f32], _n: usize) -> Vec<DraftToken> {
        // Stub: no checkpoint reader yet, no proposals. Returning
        // empty causes the speculative dispatch (when it lands in
        // phase 3) to fall back to non-speculative for this step.
        // This keeps the entire path safe behind the feature flag.
        Vec::new()
    }

    fn reset(&mut self) {
        // No state yet.
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EagleError {
    #[error("EAGLE draft head: checkpoint reader not yet implemented (phase 1 task 1.2)")]
    NotImplemented,
    #[error("EAGLE draft head: I/O reading checkpoint: {0}")]
    Io(#[from] std::io::Error),
}

/// Phase 1 placeholder for `propose_one` so call sites in phase 2
/// can compile. Real impl in task 1.2.
pub fn propose_one_stub(_h_target: &[f32]) -> Option<DraftToken> {
    let _ = TokenId::default();
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unloaded_propose_returns_empty() {
        let mut head = EagleDraftHead::unloaded();
        let h = vec![0.0f32; 2560];
        let proposals = head.propose(&h, 4);
        assert!(proposals.is_empty());
    }

    #[test]
    fn from_checkpoint_records_path() {
        let head = EagleDraftHead::from_checkpoint("/tmp/draft.safetensors").unwrap();
        assert_eq!(
            head.checkpoint_path()
                .map(|p| p.to_string_lossy().into_owned()),
            Some("/tmp/draft.safetensors".to_string())
        );
    }

    #[test]
    fn reset_does_not_panic() {
        let mut head = EagleDraftHead::unloaded();
        head.reset();
    }
}
