//! [`BuildRecord`] — the structured result `larql recipe build` prints
//! (matching `larql dec-bench`'s `--output-file` JSON pattern). MIRROR
//! and REGISTER aren't implemented in this driver (see `build/mod.rs`'s
//! doc comment) — this record is what an external wrapper would read
//! to do either, the same way `dec0-loopback.sh` wraps `dec-bench`'s
//! own JSON output today.

use serde::Serialize;

/// Which stage a build stopped at, if it didn't reach RELEASE.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// Scratch directory usability (§7 stage 1).
    Preflight,
    /// Pinned-revision download (§7 stage 2).
    Fetch,
    /// `larql extract` (§7 stage 3).
    Extract,
    /// `larql slice` per non-full output (§7 stage 4).
    Slice,
    /// Output size measurement (§7 stage 5 — manifest writing itself
    /// is EXTRACT/SLICE's job, not this driver's).
    Manifest,
    /// Checksum integrity (part of §7 stage 7 — reconstruction/logit
    /// match are not implemented, see `stages::verify`'s doc comment).
    Verify,
    /// `larql hf publish --private` per output (§7 stage 8).
    Publish,
    /// Visibility flip to public (part of §7 stage 10 — tagging and
    /// collection membership are not implemented, see
    /// `stages::release`'s doc comment).
    Release,
}

/// Terminal outcome of a build.
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum BuildStatus {
    /// Every stage completed.
    Passed,
    /// A stage failed; the record's `outputs` reflect whatever
    /// completed before the failure.
    Failed {
        /// Which stage failed.
        stage: Stage,
        /// The stage's own error message.
        message: String,
    },
}

/// One declared output's progress through the pipeline.
#[derive(Debug, Serialize)]
pub struct OutputRecord {
    /// The output's `preset` name.
    pub preset: String,
    /// Measured size, once MANIFEST has run for this output.
    pub size_bytes: Option<u64>,
    /// The Hub repo it published to, once PUBLISH has run.
    pub repo: Option<String>,
    /// Whether RELEASE flipped this repo to public.
    pub released: bool,
}

impl OutputRecord {
    pub(super) fn new(preset: &str) -> Self {
        Self {
            preset: preset.to_string(),
            size_bytes: None,
            repo: None,
            released: false,
        }
    }
}

/// The full result of one `larql recipe build` run.
#[derive(Debug, Serialize)]
pub struct BuildRecord {
    /// The recipe's `build_id` (see [`crate::build_id`]).
    pub build_id: String,
    /// The recipe's `metadata.name`.
    pub recipe_name: String,
    /// Per-output progress.
    pub outputs: Vec<OutputRecord>,
    /// Terminal status. Flattened so the JSON carries one `status` key
    /// at the top level (`"passed"` or `"failed"`), not a nested
    /// `status.status`.
    #[serde(flatten)]
    pub status: BuildStatus,
}

impl BuildRecord {
    /// True if every stage completed.
    pub fn passed(&self) -> bool {
        matches!(self.status, BuildStatus::Passed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passed_true_only_for_passed_status() {
        let record = BuildRecord {
            build_id: "abc".into(),
            recipe_name: "test".into(),
            outputs: vec![],
            status: BuildStatus::Passed,
        };
        assert!(record.passed());
    }

    #[test]
    fn passed_false_for_failed_status() {
        let record = BuildRecord {
            build_id: "abc".into(),
            recipe_name: "test".into(),
            outputs: vec![],
            status: BuildStatus::Failed {
                stage: Stage::Extract,
                message: "boom".into(),
            },
        };
        assert!(!record.passed());
    }

    #[test]
    fn output_record_starts_empty() {
        let record = OutputRecord::new("client");
        assert_eq!(record.preset, "client");
        assert!(record.size_bytes.is_none());
        assert!(record.repo.is_none());
        assert!(!record.released);
    }

    #[test]
    fn serialises_to_the_expected_shape() {
        let record = BuildRecord {
            build_id: "abc123".into(),
            recipe_name: "gemma-3-4b-it".into(),
            outputs: vec![OutputRecord::new("full")],
            status: BuildStatus::Failed {
                stage: Stage::Publish,
                message: "401".into(),
            },
        };
        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("\"build_id\":\"abc123\""));
        assert!(json.contains("\"status\":\"failed\""));
        assert!(json.contains("\"stage\":\"publish\""));
        // Flattened, not nested — `status` is a plain string field at
        // the top level, not `{"status": {"status": "failed", ...}}`.
        assert!(!json.contains("\"status\":{"));
    }
}
