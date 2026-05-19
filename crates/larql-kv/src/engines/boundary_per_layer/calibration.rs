//! `BoundaryCalibrationStore` — records that bound the end-to-end KL of a
//! per-layer codec policy on a given model.
//!
//! Per the spec (§4.7), the store is populated by an offline sweep harness
//! that runs the policy against `MarkovResidualEngine` on a representative
//! corpus and measures KL. The engine refuses to construct without a
//! matching record. The harness itself lives outside this crate; v0.1 of
//! the calibration store offers only insert + lookup APIs and an in-memory
//! implementation.

use std::collections::HashMap;
use std::sync::Mutex;

/// Per-policy calibration record. The `kl_bound_nats` is the maximum
/// end-to-end KL divergence measured for this policy on the corpus
/// referenced by `corpus_id`.
#[derive(Debug, Clone)]
pub struct BoundaryCalibrationRecord {
    pub policy_fingerprint: String,
    pub corpus_id: String,
    pub kl_bound_nats: f32,
    pub samples: usize,
}

impl BoundaryCalibrationRecord {
    /// Convenience constructor for an "uncalibrated but trivially safe"
    /// record. v0.1 uses this for `bf16_uniform` policies which inherit
    /// `MarkovResidualCodecEngine`'s `Bf16` calibration without further work.
    pub fn bf16_uniform_default(policy_fingerprint: impl Into<String>) -> Self {
        Self {
            policy_fingerprint: policy_fingerprint.into(),
            corpus_id: "bf16-trivial".into(),
            kl_bound_nats: 0.01,
            samples: 0,
        }
    }
}

/// Errors a [`BoundaryCalibrationStore`] may surface.
#[derive(Debug, thiserror::Error)]
pub enum CalibrationError {
    #[error("backend error: {0}")]
    Backend(String),
    #[error("no calibration record for policy fingerprint {0}")]
    NoRecord(String),
    #[error(
        "calibration KL {measured:.3} nats exceeds caller budget {budget:.3} nats (policy {fingerprint})"
    )]
    BudgetExceeded {
        fingerprint: String,
        measured: f32,
        budget: f32,
    },
}

/// Lookup interface for per-policy calibration records.
pub trait BoundaryCalibrationStore: Send + Sync {
    /// Persist a calibration record. Implementations may dedupe by
    /// `policy_fingerprint`.
    fn put(&self, record: BoundaryCalibrationRecord) -> Result<(), CalibrationError>;

    /// Return the most recent record for `policy_fingerprint`, or
    /// `Err(NoRecord)` if none exists.
    fn get(&self, policy_fingerprint: &str) -> Result<BoundaryCalibrationRecord, CalibrationError>;
}

/// Default v0.1 implementation: keeps records in process memory.
#[derive(Default)]
pub struct InMemoryCalibrationStore {
    inner: Mutex<HashMap<String, BoundaryCalibrationRecord>>,
}

impl InMemoryCalibrationStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_count(&self) -> usize {
        self.inner.lock().map(|m| m.len()).unwrap_or(0)
    }
}

impl BoundaryCalibrationStore for InMemoryCalibrationStore {
    fn put(&self, record: BoundaryCalibrationRecord) -> Result<(), CalibrationError> {
        let mut map = self
            .inner
            .lock()
            .map_err(|e| CalibrationError::Backend(format!("mutex poisoned: {e}")))?;
        map.insert(record.policy_fingerprint.clone(), record);
        Ok(())
    }

    fn get(&self, policy_fingerprint: &str) -> Result<BoundaryCalibrationRecord, CalibrationError> {
        let map = self
            .inner
            .lock()
            .map_err(|e| CalibrationError::Backend(format!("mutex poisoned: {e}")))?;
        map.get(policy_fingerprint)
            .cloned()
            .ok_or_else(|| CalibrationError::NoRecord(policy_fingerprint.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_store_returns_no_record() {
        let s = InMemoryCalibrationStore::new();
        let err = s.get("missing").unwrap_err();
        assert!(matches!(err, CalibrationError::NoRecord(_)));
        assert_eq!(s.record_count(), 0);
    }

    #[test]
    fn put_then_get_returns_record() {
        let s = InMemoryCalibrationStore::new();
        let r = BoundaryCalibrationRecord {
            policy_fingerprint: "fp-1".into(),
            corpus_id: "corpus-A".into(),
            kl_bound_nats: 0.05,
            samples: 300,
        };
        s.put(r.clone()).unwrap();
        let got = s.get("fp-1").unwrap();
        assert_eq!(got.policy_fingerprint, "fp-1");
        assert_eq!(got.corpus_id, "corpus-A");
        assert!((got.kl_bound_nats - 0.05).abs() < 1e-6);
        assert_eq!(got.samples, 300);
    }

    #[test]
    fn put_overwrites_previous_record() {
        let s = InMemoryCalibrationStore::new();
        let mut r = BoundaryCalibrationRecord {
            policy_fingerprint: "fp".into(),
            corpus_id: "c".into(),
            kl_bound_nats: 0.1,
            samples: 100,
        };
        s.put(r.clone()).unwrap();
        r.kl_bound_nats = 0.05;
        r.samples = 500;
        s.put(r).unwrap();
        let got = s.get("fp").unwrap();
        assert_eq!(got.samples, 500);
    }

    #[test]
    fn record_count_reflects_inserts() {
        let s = InMemoryCalibrationStore::new();
        for i in 0..3 {
            s.put(BoundaryCalibrationRecord {
                policy_fingerprint: format!("fp-{i}"),
                corpus_id: "c".into(),
                kl_bound_nats: 0.0,
                samples: 0,
            })
            .unwrap();
        }
        assert_eq!(s.record_count(), 3);
    }

    #[test]
    fn bf16_uniform_default_has_small_kl() {
        let r = BoundaryCalibrationRecord::bf16_uniform_default("fp");
        assert!(r.kl_bound_nats < 0.1);
        assert_eq!(r.samples, 0);
        assert_eq!(r.corpus_id, "bf16-trivial");
    }

    #[test]
    fn calibration_error_display_includes_fields() {
        let e = CalibrationError::NoRecord("xyz".into());
        assert!(e.to_string().contains("xyz"));
        let e = CalibrationError::BudgetExceeded {
            fingerprint: "fp".into(),
            measured: 0.5,
            budget: 0.1,
        };
        let s = e.to_string();
        assert!(s.contains("0.500"));
        assert!(s.contains("0.100"));
        assert!(s.contains("fp"));
    }

    #[test]
    fn backend_error_display_round_trips_message() {
        let e = CalibrationError::Backend("disk full".into());
        assert!(e.to_string().contains("disk full"));
    }
}
