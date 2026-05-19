//! Persistence layer for emitted boundary frames.
//!
//! The engine writes through a [`BoundaryArchive`] — a tiny trait whose only
//! contract is "append + look up by sequence id." Concrete archive
//! implementations (filesystem, gRPC, etc.) live outside `larql-kv`. v0.1
//! ships [`InMemoryArchive`] which keeps everything in process memory.

use larql_boundary::BoundaryFrame;
use std::collections::HashMap;
use std::sync::Mutex;

/// Errors a [`BoundaryArchive`] may surface.
#[derive(Debug, thiserror::Error)]
pub enum ArchiveError {
    /// The underlying store returned an error (filesystem I/O, network, etc.).
    #[error("archive backend error: {0}")]
    Backend(String),
    /// The store is in an inconsistent state (e.g. corrupted record).
    #[error("archive consistency error: {0}")]
    Consistency(String),
}

/// Persistence interface for [`BoundaryFrame`] chains.
///
/// Implementations must be `Send + Sync`: the engine is constructed with a
/// boxed archive and the engine itself is `Send`.
pub trait BoundaryArchive: Send + Sync {
    /// Durably append a frame for the frame's `sequence_id`. Implementations
    /// that buffer must ensure the frame survives a process crash before
    /// returning `Ok(())`, per `BOUNDARY_REF_PROTOCOL.md` §13 Option A.
    fn append(&self, frame: BoundaryFrame) -> Result<(), ArchiveError>;

    /// Return the chain for `sequence_id`, ordered by `token_end` ascending.
    /// Returns an empty vector when no frames exist for the sequence.
    fn load_chain(&self, sequence_id: &str) -> Result<Vec<BoundaryFrame>, ArchiveError>;

    /// Total number of frames archived across all sequences. Diagnostic
    /// counter — implementations that cannot count cheaply may return `None`.
    fn total_frames(&self) -> Option<usize> {
        None
    }
}

/// Default archive: keeps frames in process memory.
///
/// Suitable for tests, single-process sessions, and any use case where
/// durability across process restarts is provided externally (e.g. the
/// caller serialises the chain through its own persistence layer).
#[derive(Default)]
pub struct InMemoryArchive {
    inner: Mutex<HashMap<String, Vec<BoundaryFrame>>>,
}

impl InMemoryArchive {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of sequences with at least one archived frame. Useful for tests.
    pub fn sequence_count(&self) -> usize {
        self.inner.lock().map(|m| m.len()).unwrap_or(0)
    }
}

impl BoundaryArchive for InMemoryArchive {
    fn append(&self, frame: BoundaryFrame) -> Result<(), ArchiveError> {
        let mut map = self
            .inner
            .lock()
            .map_err(|e| ArchiveError::Backend(format!("mutex poisoned: {e}")))?;
        map.entry(frame.sequence_id.clone())
            .or_default()
            .push(frame);
        Ok(())
    }

    fn load_chain(&self, sequence_id: &str) -> Result<Vec<BoundaryFrame>, ArchiveError> {
        let map = self
            .inner
            .lock()
            .map_err(|e| ArchiveError::Backend(format!("mutex poisoned: {e}")))?;
        let mut chain = map.get(sequence_id).cloned().unwrap_or_default();
        chain.sort_by_key(|f| f.token_end);
        Ok(chain)
    }

    fn total_frames(&self) -> Option<usize> {
        let map = self.inner.lock().ok()?;
        Some(map.values().map(|v| v.len()).sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_boundary::{
        BoundaryAgreement, BoundaryCompression, BoundaryContract, FallbackPolicy,
    };

    fn frame(sequence_id: &str, token_end: u64) -> BoundaryFrame {
        BoundaryFrame {
            version: 1,
            model_id: "test".into(),
            model_revision: "rev".into(),
            tokenizer_revision: "tok".into(),
            architecture: "arch".into(),
            boundary_id: format!("{sequence_id}-{token_end}"),
            sequence_id: sequence_id.into(),
            token_start: token_end.saturating_sub(512),
            token_end,
            layer: 0,
            hidden_size: 0,
            compression_scheme: BoundaryCompression::None,
            contract_level: BoundaryContract::Exact,
            payload: vec![],
            raw_top1_token: 0,
            raw_logit_margin: 0.0,
            raw_top1_prob: None,
            compressed_top1_token: None,
            boundary_agreement: BoundaryAgreement::NotChecked,
            codec_fragile: false,
            boundary_fragile: false,
            fallback_policy: FallbackPolicy::None,
            fallback_ref: None,
            calibration_run_id: None,
            residual_hash: None,
            token_hash: None,
        }
    }

    #[test]
    fn empty_archive_loads_empty_chain() {
        let a = InMemoryArchive::new();
        let chain = a.load_chain("missing").unwrap();
        assert!(chain.is_empty());
        assert_eq!(a.sequence_count(), 0);
        assert_eq!(a.total_frames(), Some(0));
    }

    #[test]
    fn append_then_load_returns_frame() {
        let a = InMemoryArchive::new();
        a.append(frame("seq-a", 512)).unwrap();
        let chain = a.load_chain("seq-a").unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].token_end, 512);
    }

    #[test]
    fn load_chain_is_sorted_by_token_end() {
        let a = InMemoryArchive::new();
        a.append(frame("seq", 1024)).unwrap();
        a.append(frame("seq", 512)).unwrap();
        a.append(frame("seq", 1536)).unwrap();
        let chain = a.load_chain("seq").unwrap();
        let ends: Vec<u64> = chain.iter().map(|f| f.token_end).collect();
        assert_eq!(ends, vec![512, 1024, 1536]);
    }

    #[test]
    fn multiple_sequences_are_isolated() {
        let a = InMemoryArchive::new();
        a.append(frame("seq-a", 512)).unwrap();
        a.append(frame("seq-b", 512)).unwrap();
        a.append(frame("seq-a", 1024)).unwrap();
        assert_eq!(a.sequence_count(), 2);
        assert_eq!(a.load_chain("seq-a").unwrap().len(), 2);
        assert_eq!(a.load_chain("seq-b").unwrap().len(), 1);
        assert_eq!(a.total_frames(), Some(3));
    }

    #[test]
    fn archive_error_display_includes_message() {
        let e = ArchiveError::Backend("disk full".into());
        assert!(e.to_string().contains("disk full"));
        let e = ArchiveError::Consistency("corrupted".into());
        assert!(e.to_string().contains("corrupted"));
    }

    /// Custom archive that inherits the default `total_frames` impl
    /// (which returns `None`). Used to verify the trait-default branch.
    struct DefaultMethodArchive;
    impl BoundaryArchive for DefaultMethodArchive {
        fn append(&self, _: BoundaryFrame) -> Result<(), ArchiveError> {
            Ok(())
        }
        fn load_chain(&self, _: &str) -> Result<Vec<BoundaryFrame>, ArchiveError> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn default_method_archive_exercises_all_methods() {
        let a = DefaultMethodArchive;
        // total_frames hits the default trait impl body (not InMemoryArchive's
        // override).
        assert_eq!(a.total_frames(), None);
        // Exercise append + load_chain so the impl methods on the test-only
        // archive don't drag the file's function-coverage denominator.
        a.append(frame("seq", 1)).unwrap();
        assert!(a.load_chain("seq").unwrap().is_empty());
    }
}
