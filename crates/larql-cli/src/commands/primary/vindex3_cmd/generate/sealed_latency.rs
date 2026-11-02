//! `--sealed-latency`: every latency a residency-curve run measures, in a
//! file of its own.
//!
//! Sealing is a matter of access, not of discipline. The residency stream
//! and the intervention records are what an evaluator reads; this file is
//! excluded from them until it is unsealed, and the run's console prints
//! no timing while it is active. It is created exclusively, refused
//! attempts' timings go to it like any other, and its SHA-256 is written
//! into the residency stream's `after_run` header so it can be shown
//! untouched when unsealed.
//!
//! **What the seal does not cover.** It seals per-step, per-stage and bind
//! values. The residency stream still carries clock readings at admission
//! and after the run, so a reader can compute the whole attempt's
//! admission-to-after-run duration, which on a one-pass cold arm is close
//! to the pass's latency. The evidence files' own timestamps give the same
//! figure. Keeping that aggregate out of a decision is a matter of the
//! protocol's access rules; nothing here enforces it.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use super::admission::create_evidence_file;
use super::residency_record::{point_clock, RunIdentity};

/// The sealed file's schema. A change to any field's meaning increments it.
pub(in super::super) const SEALED_SCHEMA: &str = "larql.sealed-latency/1";

/// An open sealed-latency file.
pub(in super::super) struct SealedLatency {
    path: PathBuf,
    out: BufWriter<File>,
    run_id: String,
}

/// A sealed file, closed, and the digest of what it holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super) struct SealedDigest {
    pub(in super::super) path: PathBuf,
    pub(in super::super) sha256: String,
}

impl SealedLatency {
    /// Create the file (refusing an existing one) and write its header.
    pub(in super::super) fn create(path: &Path, run: &RunIdentity) -> Result<Self, String> {
        let file = create_evidence_file(path, "--sealed-latency")?;
        let mut sealed = Self {
            path: path.to_path_buf(),
            out: BufWriter::new(file),
            run_id: run.run_id.clone(),
        };
        sealed
            .write(json!({
                "record": "header",
                "pid": run.pid,
                "container": {
                    "path": run.container_path,
                    "index_json_sha256": run.index_json_sha256,
                },
            }))
            .map_err(|e| format!("cannot write --sealed-latency {}: {e}", path.display()))?;
        Ok(sealed)
    }

    pub(in super::super) fn path(&self) -> &Path {
        &self.path
    }

    /// One record, stamped with the schema, the run and the system-wide
    /// clock.
    fn write(&mut self, mut record: Value) -> std::io::Result<()> {
        let fields = record.as_object_mut().expect("a JSON object");
        fields.insert("schema".into(), json!(SEALED_SCHEMA));
        fields.insert("run_id".into(), json!(self.run_id));
        fields.insert("clock".into(), point_clock());
        serde_json::to_writer(&mut self.out, &record)?;
        self.out.write_all(b"\n")?;
        // Every record reaches the file as it is written (never inside a
        // timed window), so a killed process keeps what it measured.
        self.out.flush()
    }

    /// The bind time: sealed like every other latency, and no input to
    /// admission.
    pub(in super::super) fn bind(&mut self, seconds: f64) -> std::io::Result<()> {
        self.write(json!({ "record": "bind", "seconds": seconds }))
    }

    /// One timed step of one pass, with its breakdowns.
    pub(in super::super) fn step(
        &mut self,
        pass: usize,
        label: &str,
        step: &str,
        elapsed_ms: f64,
        breakdown: Value,
    ) -> std::io::Result<()> {
        self.write(json!({
            "record": "step",
            "pass": pass,
            "label": label,
            "step": step,
            "status": "ok",
            "elapsed_ms": elapsed_ms,
            "breakdown": breakdown,
        }))
    }

    /// A step that returned an error: its time is latency too.
    pub(in super::super) fn failed_step(
        &mut self,
        pass: usize,
        label: &str,
        step: &str,
        elapsed_ms: f64,
    ) -> std::io::Result<()> {
        self.write(json!({
            "record": "step",
            "pass": pass,
            "label": label,
            "step": step,
            "status": "error",
            "elapsed_ms": elapsed_ms,
        }))
    }

    /// Flush, sync and close the file, then hash what reached the disk.
    pub(in super::super) fn finish(mut self) -> Result<SealedDigest, String> {
        let shown = self.path.display().to_string();
        let describe =
            |e: &dyn std::fmt::Display| format!("cannot seal --sealed-latency {shown}: {e}");
        self.out.flush().map_err(|e| describe(&e))?;
        self.out.get_ref().sync_data().map_err(|e| describe(&e))?;
        let sha256 =
            larql_vindex::format::checksums::sha256_file(&self.path).map_err(|e| describe(&e))?;
        Ok(SealedDigest {
            path: self.path.clone(),
            sha256,
        })
    }
}
