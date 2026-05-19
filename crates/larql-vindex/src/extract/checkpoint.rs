//! Streaming-extract checkpoint — lets `build_vindex_streaming` skip
//! phases that already completed in a previous run.
//!
//! Today's contract is **phase-level**: each phase (`gate`,
//! `down_meta`, `weights`, `q4k_weights`) marks itself complete at
//! the end. On resume the extract loop checks the checkpoint and
//! short-circuits any phase already marked done.
//!
//! Layer-level resume (skip individual finished layers within a
//! still-incomplete phase) is a future enhancement — it requires
//! mid-phase file truncation to the last clean layer boundary plus a
//! per-layer manifest of byte offsets, which is more delicate than a
//! phase flag.
//!
//! # File
//! Stored at `<output_dir>/.extract_checkpoint.json`. Atomic write
//! via `<file>.tmp` rename. Removed by `Checkpoint::clear` once the
//! whole extract succeeds — its presence in the output dir means a
//! previous run was interrupted.

use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::VindexLayerInfo;
use crate::error::VindexError;

/// Checkpoint filename inside the output directory. Hidden so it
/// doesn't clutter `ls` and so HF / vindex-loader code doesn't try to
/// upload it.
pub const CHECKPOINT_FILE: &str = ".extract_checkpoint.json";

/// Set of phases the streaming extractor runs. Phase order matters
/// for resume — completing a later phase implies all earlier phases
/// completed in the run that produced the checkpoint.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractPhase {
    /// `gate_vectors.bin` write.
    Gate,
    /// `down_meta.bin` write.
    DownMeta,
    /// `attn_weights.bin` / `up_weights.bin` / `down_weights.bin` /
    /// `norms.bin` / `lm_head.bin` (f32 path).
    Weights,
    /// `attn_weights_kquant.bin` + `interleaved_kquant.bin` (k-quant path).
    Q4kWeights,
}

/// On-disk checkpoint format.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Format version — bump when the JSON shape changes
    /// incompatibly.
    pub version: u32,
    /// Source model directory captured at extract start. If the
    /// checkpoint's `model_dir` differs from the resume run's
    /// `model_dir`, the checkpoint is silently invalidated (callers
    /// are extracting from a different source).
    #[serde(default)]
    pub model_dir: String,
    /// Source model name (`config.json#model_name`).
    #[serde(default)]
    pub model_name: String,
    /// Total layer count of the model — sanity check.
    #[serde(default)]
    pub num_layers: usize,
    /// Phases marked complete by the previous run.
    #[serde(default)]
    pub completed: Vec<ExtractPhase>,
    /// ISO 8601 timestamp of the last update.
    #[serde(default)]
    pub last_update: String,
    /// Per-layer info captured during the gate phase. Persisted so a
    /// resume run can skip the gate loop and still produce the
    /// correct `index.json` `layers` array. Populated by
    /// `mark_gate_complete`; left `None` until the gate phase
    /// finishes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate_layer_infos: Option<Vec<VindexLayerInfo>>,
}

impl Checkpoint {
    /// Try to load a checkpoint from `<output_dir>/.extract_checkpoint.json`.
    /// Returns `Ok(None)` if no checkpoint is present (fresh run);
    /// `Ok(Some(...))` if one was found; `Err` only on actual parse
    /// failures (corrupted JSON in an existing file).
    pub fn load(output_dir: &Path) -> Result<Option<Self>, VindexError> {
        let path = checkpoint_path(output_dir);
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)?;
        let cp: Checkpoint = serde_json::from_str(&text)
            .map_err(|e| VindexError::Parse(format!("checkpoint at {}: {e}", path.display())))?;
        Ok(Some(cp))
    }

    /// Save atomically (`*.tmp` + rename).
    pub fn save(&self, output_dir: &Path) -> Result<(), VindexError> {
        let path = checkpoint_path(output_dir);
        let tmp_path = path.with_extension("json.tmp");
        let json =
            serde_json::to_string_pretty(self).map_err(|e| VindexError::Parse(e.to_string()))?;
        let mut f = std::fs::File::create(&tmp_path)?;
        f.write_all(json.as_bytes())?;
        f.sync_all()?;
        drop(f);
        std::fs::rename(&tmp_path, &path)?;
        Ok(())
    }

    /// Remove the checkpoint file. Call after the whole extract
    /// succeeds so the next run treats the output dir as a finished
    /// vindex, not a half-finished one.
    pub fn clear(output_dir: &Path) -> Result<(), VindexError> {
        let path = checkpoint_path(output_dir);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    /// Mark `phase` complete and persist.
    pub fn mark(&mut self, phase: ExtractPhase, output_dir: &Path) -> Result<(), VindexError> {
        if !self.completed.contains(&phase) {
            self.completed.push(phase);
        }
        self.last_update = current_iso8601();
        self.save(output_dir)
    }

    /// Mark the gate phase complete and persist the `layer_infos`
    /// vector. The skip-on-resume path uses the persisted infos to
    /// rebuild the final `index.json` without re-running the gate
    /// loop.
    pub fn mark_gate_complete(
        &mut self,
        layer_infos: Vec<VindexLayerInfo>,
        output_dir: &Path,
    ) -> Result<(), VindexError> {
        self.gate_layer_infos = Some(layer_infos);
        self.mark(ExtractPhase::Gate, output_dir)
    }

    /// Whether `phase` was completed in a prior run.
    pub fn is_complete(&self, phase: ExtractPhase) -> bool {
        self.completed.contains(&phase)
    }

    /// Construct a fresh checkpoint at the start of an extract run.
    pub fn fresh(model_dir: &Path, model_name: &str, num_layers: usize) -> Self {
        Self {
            version: 1,
            model_dir: model_dir.display().to_string(),
            model_name: model_name.to_string(),
            num_layers,
            completed: Vec::new(),
            last_update: current_iso8601(),
            gate_layer_infos: None,
        }
    }

    /// Decide whether a previously-loaded checkpoint is **valid for
    /// resume** in the current run. Validation rules:
    /// - same `model_dir` (re-extracting from a different source =
    ///   start fresh)
    /// - same `model_name`
    /// - same `num_layers`
    /// - version matches
    ///
    /// On mismatch, returns `false` — caller should delete the
    /// stale checkpoint and start a fresh run.
    pub fn is_compatible_with(
        &self,
        model_dir: &Path,
        model_name: &str,
        num_layers: usize,
    ) -> bool {
        self.version == 1
            && self.model_dir == model_dir.display().to_string()
            && self.model_name == model_name
            && self.num_layers == num_layers
    }
}

fn checkpoint_path(output_dir: &Path) -> PathBuf {
    output_dir.join(CHECKPOINT_FILE)
}

fn current_iso8601() -> String {
    // Bare-minimum ISO-8601 in UTC without pulling chrono in.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{}Z", iso8601_from_unix(now))
}

/// Convert a Unix timestamp to a calendar `YYYY-MM-DDTHH:MM:SS`
/// string. Fixed-offset only; no leap-second / TZ handling.
fn iso8601_from_unix(secs: u64) -> String {
    let days = secs / 86400;
    let secs_of_day = secs % 86400;
    let h = secs_of_day / 3600;
    let m = (secs_of_day % 3600) / 60;
    let s = secs_of_day % 60;
    let (y, mo, d) = days_to_ymd(days as i64);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}")
}

/// Civil-from-days (Howard Hinnant's algorithm), 1970-01-01 = 0.
fn days_to_ymd(z: i64) -> (i32, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i32 + era as i32 * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir(label: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "larql_checkpoint_{}_{}_{}",
            label,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn missing_checkpoint_loads_as_none() {
        let dir = tempdir("missing");
        assert!(Checkpoint::load(&dir).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn round_trip_preserves_completed_phases() {
        let dir = tempdir("round");
        let mut cp = Checkpoint::fresh(Path::new("/src"), "model-x", 34);
        cp.mark(ExtractPhase::Gate, &dir).unwrap();
        cp.mark(ExtractPhase::DownMeta, &dir).unwrap();

        let loaded = Checkpoint::load(&dir).unwrap().expect("present");
        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.model_name, "model-x");
        assert_eq!(loaded.num_layers, 34);
        assert!(loaded.is_complete(ExtractPhase::Gate));
        assert!(loaded.is_complete(ExtractPhase::DownMeta));
        assert!(!loaded.is_complete(ExtractPhase::Weights));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mark_is_idempotent() {
        let dir = tempdir("idem");
        let mut cp = Checkpoint::fresh(Path::new("/src"), "m", 1);
        cp.mark(ExtractPhase::Gate, &dir).unwrap();
        cp.mark(ExtractPhase::Gate, &dir).unwrap();
        assert_eq!(cp.completed.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn clear_removes_file() {
        let dir = tempdir("clear");
        let mut cp = Checkpoint::fresh(Path::new("/src"), "m", 1);
        cp.mark(ExtractPhase::Gate, &dir).unwrap();
        assert!(checkpoint_path(&dir).exists());
        Checkpoint::clear(&dir).unwrap();
        assert!(!checkpoint_path(&dir).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compatibility_rejects_different_model() {
        let dir = tempdir("compat");
        let cp = Checkpoint::fresh(Path::new("/src/a"), "model-a", 34);
        cp.save(&dir).unwrap();
        let loaded = Checkpoint::load(&dir).unwrap().unwrap();

        // Same model — compatible.
        assert!(loaded.is_compatible_with(Path::new("/src/a"), "model-a", 34));
        // Different source dir — invalidate.
        assert!(!loaded.is_compatible_with(Path::new("/src/b"), "model-a", 34));
        // Different model name — invalidate.
        assert!(!loaded.is_compatible_with(Path::new("/src/a"), "model-b", 34));
        // Different layer count — invalidate.
        assert!(!loaded.is_compatible_with(Path::new("/src/a"), "model-a", 35));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn iso8601_known_dates() {
        // Sanity-check our hand-rolled civil calendar against known
        // Unix timestamps. 2026-04-25T00:00:00Z = 1777680000.
        assert_eq!(iso8601_from_unix(0), "1970-01-01T00:00:00");
        assert_eq!(iso8601_from_unix(1_777_680_000), "2026-05-02T00:00:00");
    }
}
