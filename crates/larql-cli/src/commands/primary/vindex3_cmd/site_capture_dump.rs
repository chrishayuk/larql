//! `vindex3 exec --logit-dump … --capture-sites DIR` (PREC-DIAG-1 E0a):
//! persist a [`SiteCapture`] beside the teacher-forced logit dump.
//!
//! **One authority for how a capture ended: `capture.result.json`.** It
//! is written the moment the NEW directory is created, with
//! `status: "in_progress"`, and replaced (write to a temporary file, then
//! rename) by exactly one terminal status:
//!
//! - `"complete"` — only after every plane file AND `manifest.json` are
//!   on disk; it carries the manifest's sha256. A reader accepts a
//!   capture iff the status is `complete` and that digest matches.
//! - `"resource_limit"` — over budget, raised before anything is
//!   allocated or written.
//! - `"fault"` — any other refusal or failure, including a step error or a
//!   failed write after the capture began.
//!
//! A process that dies mid-run leaves `"in_progress"`, which is never a
//! complete capture. Every record carries the run's identity: build
//! commit (when the binary declares one), binary and container digests,
//! the token digest, backend and representation settings, the `LARQL_*`
//! environment, and the CPU projection plans the run called.

use std::path::{Path, PathBuf};

use larql_vindex::format::vindex3::opplan::exec::site_capture::{
    CaptureFault, CaptureRequest, CapturedSites, SiteCapture,
};
use larql_vindex::format::vindex3::opplan::ComponentOpPlan;
use serde_json::json;
use sha2::{Digest, Sha256};

use super::args::{SiteCaptureArgs, DEFAULT_CAPTURE_BUDGET_BYTES};

pub(super) const MANIFEST: &str = "manifest.json";
pub(super) const RESULT: &str = "capture.result.json";
const SCHEMA: &str = "larql.site-capture/1";
const IN_PROGRESS: &str = "in_progress";
const COMPLETE: &str = "complete";

/// What produced a capture, bound before anything loads.
#[derive(Debug, Clone)]
pub(super) struct CaptureIdentity {
    pub(super) build_commit: Option<String>,
    pub(super) binary_sha256: String,
    pub(super) container: PathBuf,
    pub(super) container_index_sha256: String,
    pub(super) component: String,
    pub(super) engine: String,
    pub(super) backend: String,
    pub(super) representation_source: String,
    pub(super) tokens_sha256: String,
    /// PREC-DIAG-1 E0b `--rope-angles`, by name.
    pub(super) rope_angles: String,
    pub(super) draft_depth: Option<usize>,
    pub(super) environment: Vec<(String, String)>,
    pub(super) platform: String,
}

impl CaptureIdentity {
    fn json(&self) -> serde_json::Value {
        json!({
            "build_commit": self.build_commit,
            "binary_sha256": self.binary_sha256,
            "container": self.container.display().to_string(),
            "container_index_sha256": self.container_index_sha256,
            "component": self.component,
            "engine": self.engine,
            "backend": self.backend,
            "representation_source": self.representation_source,
            "tokens_sha256": self.tokens_sha256,
            "rope_angles": self.rope_angles,
            "draft_depth": self.draft_depth,
            "environment": self.environment.iter()
                .map(|(k, v)| json!({"name": k, "value": v})).collect::<Vec<_>>(),
            "platform": self.platform,
        })
    }
}

/// The digest of a token sequence as the capture records it: sha256 over
/// the ids as little-endian u32.
pub(super) fn tokens_sha256(tokens: &[u32]) -> String {
    let mut digest = Sha256::new();
    for t in tokens {
        digest.update(t.to_le_bytes());
    }
    hex(&digest.finalize())
}

/// Where a requested capture goes, created before the weights load.
pub(super) struct CaptureTarget {
    dir: PathBuf,
    positions: Vec<usize>,
    budget_bytes: u64,
    identity: CaptureIdentity,
}

/// Parse `--capture-positions`: comma-separated, strictly increasing,
/// each `< sequence_len`. Refused, never sorted or clamped.
pub(super) fn parse_positions(text: &str, sequence_len: usize) -> Result<Vec<usize>, String> {
    let positions = text
        .split(',')
        .map(|p| {
            p.trim()
                .parse::<usize>()
                .map_err(|e| format!("--capture-positions {p:?}: {e}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if positions.is_empty() {
        return Err("--capture-positions names no position".to_string());
    }
    if !positions.windows(2).all(|w| w[0] < w[1]) {
        return Err("--capture-positions must be strictly increasing".to_string());
    }
    if let Some(&last) = positions.last() {
        if last >= sequence_len {
            return Err(format!(
                "--capture-positions {last} is outside a sequence of {sequence_len}"
            ));
        }
    }
    Ok(positions)
}

impl CaptureTarget {
    /// `None` when no capture was asked for. Creates the directory, which
    /// must not exist.
    pub(super) fn open(
        args: &SiteCaptureArgs,
        sequence_len: usize,
        identity: impl FnOnce() -> Result<CaptureIdentity, Box<dyn std::error::Error>>,
    ) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        let Some(dir) = &args.capture_sites else {
            return Ok(None);
        };
        let text = args
            .capture_positions
            .as_deref()
            .ok_or("--capture-sites requires --capture-positions")?;
        let positions = parse_positions(text, sequence_len)?;
        let identity = identity()?;
        std::fs::create_dir(dir).map_err(|e| {
            format!(
                "--capture-sites {}: {e} (the directory must be new)",
                dir.display()
            )
        })?;
        let target = Self {
            dir: dir.clone(),
            positions,
            budget_bytes: args
                .capture_budget_bytes
                .unwrap_or(DEFAULT_CAPTURE_BUDGET_BYTES),
            identity,
        };
        target.set_status(IN_PROGRESS, json!(null), "the capture has not finished")?;
        Ok(Some(target))
    }

    /// Replace `capture.result.json` (temporary file, then rename).
    fn set_status(
        &self,
        status: &str,
        detail: serde_json::Value,
        message: &str,
    ) -> std::io::Result<()> {
        let record = json!({
            "schema": SCHEMA,
            "status": status,
            "detail": detail,
            "message": message,
            "positions": self.positions,
            "budget_bytes": self.budget_bytes,
            "identity": self.identity.json(),
        });
        write_json_atomic(&self.dir.join(RESULT), &record)
    }

    /// The status `capture.result.json` holds now.
    pub(super) fn status(&self) -> Option<String> {
        let bytes = std::fs::read(self.dir.join(RESULT)).ok()?;
        let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
        value["status"].as_str().map(str::to_string)
    }

    /// After the run ended in `error`: if no terminal status was recorded,
    /// record `fault` so the directory cannot read as in progress or
    /// complete. Never overwrites a terminal status.
    pub(super) fn record_run_failure(&self, error: &dyn std::error::Error) {
        if self.status().as_deref() == Some(IN_PROGRESS) {
            let _ = self.set_status("fault", json!(null), &format!("the run failed: {error}"));
        }
    }

    /// Build the observer over the prepared image, or record why not.
    pub(super) fn observer(
        &self,
        plan: &ComponentOpPlan,
        layers: std::ops::Range<usize>,
        hidden: usize,
        sequence_len: usize,
    ) -> Result<SiteCapture, Box<dyn std::error::Error>> {
        CaptureRequest::for_plan(
            plan,
            layers,
            hidden,
            self.positions.clone(),
            sequence_len,
            self.budget_bytes,
        )
        .and_then(SiteCapture::new)
        .map_err(|fault| self.refuse(fault))
    }

    /// Write `capture.result.json` for a refused capture and return the
    /// fault as the run's error.
    pub(super) fn refuse(&self, fault: CaptureFault) -> Box<dyn std::error::Error> {
        let (status, detail) = match &fault {
            CaptureFault::ResourceLimit { required, budget } => (
                "resource_limit",
                json!({"required_bytes": required, "budget_bytes": budget}),
            ),
            _ => ("fault", json!(null)),
        };
        if let Err(e) = self.set_status(status, detail, &fault.to_string()) {
            return format!("{fault}; and writing {RESULT} failed: {e}").into();
        }
        Box::new(fault)
    }

    /// Persist a finished capture, or record its fault. A failure while
    /// writing is recorded as `fault`; `complete` is recorded last.
    pub(super) fn write(
        &self,
        finished: Result<CapturedSites, CaptureFault>,
        logit_dump: &Path,
        projection_plans: &[(String, u64)],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let sites = finished.map_err(|fault| self.refuse(fault))?;
        match self.write_planes(&sites, logit_dump, projection_plans) {
            Ok(manifest_sha256) => {
                self.set_status(
                    COMPLETE,
                    json!({"manifest_sha256": manifest_sha256, "bytes": sites.request.required_bytes()}),
                    "every requested slot was captured and written",
                )?;
                println!(
                    "wrote {} capture planes ({} bytes), {MANIFEST} and {RESULT} to {}",
                    sites.planes.len(),
                    sites.request.required_bytes(),
                    self.dir.display()
                );
                Ok(())
            }
            Err(e) => {
                let _ = self.set_status(
                    "fault",
                    json!(null),
                    &format!("writing the capture failed: {e}"),
                );
                Err(e)
            }
        }
    }

    /// Every plane, then the manifest; the manifest's sha256.
    fn write_planes(
        &self,
        sites: &CapturedSites,
        logit_dump: &Path,
        projection_plans: &[(String, u64)],
    ) -> Result<String, Box<dyn std::error::Error>> {
        let rows = sites.request.positions.len();
        let mut files = Vec::with_capacity(sites.planes.len());
        for plane in &sites.planes {
            let name = match plane.layer {
                Some(layer) => format!("{}.L{layer:03}.f32", plane.site.name()),
                None => format!("{}.f32", plane.site.name()),
            };
            let bytes: Vec<u8> = plane.values.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(self.dir.join(&name), &bytes)?;
            files.push(json!({
                "site": plane.site.name(),
                "layer": plane.layer,
                "rows": rows,
                "width": plane.width,
                "file": name,
                "sha256": hex(&Sha256::digest(&bytes)),
            }));
        }
        let layers: Vec<_> = sites
            .request
            .layers
            .iter()
            .map(|l| {
                json!({
                    "layer": l.layer,
                    "num_q_heads": l.num_q_heads,
                    "num_kv_heads": l.num_kv_heads,
                    "head_dim": l.head_dim,
                })
            })
            .collect();
        let logit_bytes = std::fs::read(logit_dump)?;
        let manifest = json!({
            "schema": SCHEMA,
            "identity": self.identity.json(),
            "logit_dump": logit_dump.display().to_string(),
            "logit_dump_sha256": hex(&Sha256::digest(&logit_bytes)),
            "projection_plans": projection_plans.iter()
                .map(|(plan, calls)| json!({"plan": plan, "calls": calls})).collect::<Vec<_>>(),
            "path": "cached decode (--logit-dump), one token per step",
            "dtype": "f32 little-endian, [positions, width] row-major",
            "positions": sites.request.positions,
            "sequence_len": sites.request.sequence_len,
            "hidden": sites.request.hidden,
            "layers": layers,
            "budget_bytes": sites.request.budget_bytes,
            "bytes": sites.request.required_bytes(),
            "files": files,
        });
        let bytes = serde_json::to_vec_pretty(&manifest)?;
        write_atomic(&self.dir.join(MANIFEST), &bytes)?;
        Ok(hex(&Sha256::digest(&bytes)))
    }
}

/// Write `bytes` to a temporary sibling, then rename over `path`, so a
/// reader sees the old file or the new one, never a partial one.
fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = PathBuf::from(temporary);
    std::fs::write(&temporary, bytes)?;
    std::fs::rename(&temporary, path)
}

fn write_json_atomic(path: &Path, value: &serde_json::Value) -> std::io::Result<()> {
    write_atomic(path, &serde_json::to_vec_pretty(value)?)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
