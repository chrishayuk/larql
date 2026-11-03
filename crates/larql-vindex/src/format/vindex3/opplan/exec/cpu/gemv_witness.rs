//! A witness of what the CPU Q8_0 kernel was given and what it returned.
//!
//! An outside verifier can recompute a matrix-vector product, but it can't
//! see one being run. This records, at the production call site, the stored
//! block bytes, the activation and the result, as raw bits, so that something
//! outside this repository's engine can judge them. The format and what it
//! does and doesn't show are in `docs/verification/gemv-witness.md`.
//!
//! **It only observes.** It runs after the kernel has returned, reads what the
//! kernel read and wrote, and computes no expected value: judging the result is
//! the verifier's job, and a check written here would be the engine marking
//! its own work.
//!
//! **Off unless asked for.** The idle cost is one atomic load per kernel call.
//! While capturing, it digests the matrix on every call and copies it on the first,
//! so don't compare timings from a capture run with others.
//!
//! **Process-wide**, like [`super::replay`]'s capture, because projections are
//! issued from pool workers.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// The witness format's name.
pub const SCHEMA: &str = "larql-gemv-witness/1";
/// The name of the witness's manifest, written last, so it marks a complete witness.
pub const MANIFEST: &str = "witness.json";
/// The kernel a record's output came from.
pub const KERNEL: &str = "cpu.FusedKQuant.q8_0_gemv";

const ENCODING: &str = "Q8_0";

/// One kernel call as it happened.
#[derive(Debug)]
pub struct Record {
    /// Which call this was, counting every Q8_0 call observed, from 0.
    pub call: u64,
    /// The operand, as named when its bytes were loaded.
    pub operand: Option<Operand>,
    pub rows: usize,
    pub blocks: Vec<u8>,
    pub x: Vec<f32>,
    pub y: Vec<f32>,
}

/// An operand's name. A tensor name is only unique within its object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Operand {
    pub object: String,
    pub tensor: String,
}

/// What a capture observed.
#[derive(Debug, Default)]
pub struct Capture {
    pub records: Vec<Record>,
    /// Every Q8_0 call observed, including those past the limit.
    pub calls_observed: u64,
    /// How many Q8_0 operands were loaded while the capture was open.
    pub operands_noted: usize,
    /// The limit asked for, so that "limited to N" can be told from "fewer were reached".
    pub max_operands: usize,
}

/// What identifies an operand's bytes. The address alone isn't enough: an allocator can reuse it for other bytes.
type Key = (usize, [u8; 32]);

struct Open {
    max_operands: usize,
    names: HashMap<Key, Operand>,
    /// Operands already recorded: the first call on each is the one kept.
    recorded: HashSet<Key>,
    capture: Capture,
}

static OPEN: Mutex<Option<Open>> = Mutex::new(None);
/// Whether a capture is open, readable without the lock.
static ACTIVE: AtomicBool = AtomicBool::new(false);

/// Begin capturing the first call reached on each of up to `max_operands` Q8_0 operands.
/// Must come before the weights load, so that their names are noted.
pub fn start(max_operands: usize) {
    *OPEN.lock().expect("witness lock") = Some(Open {
        max_operands,
        names: HashMap::new(),
        recorded: HashSet::new(),
        capture: Capture {
            max_operands,
            ..Capture::default()
        },
    });
    ACTIVE.store(true, Ordering::Release);
}

/// Stop capturing and take what was observed.
pub fn finish() -> Capture {
    ACTIVE.store(false, Ordering::Release);
    OPEN.lock()
        .expect("witness lock")
        .take()
        .map(|open| open.capture)
        .unwrap_or_default()
}

fn key(blocks: &[u8]) -> Key {
    (blocks.as_ptr() as usize, Sha256::digest(blocks).into())
}

/// Remember which operand these bytes are. Their heap address doesn't change when the owning
/// `Vec` moves into its `LoadedWeight`, and the digest refuses other bytes at the same address.
pub(in super::super) fn note_operand(object: &str, tensor: &str, encoding: &str, blocks: &[u8]) {
    if !ACTIVE.load(Ordering::Acquire) || encoding != ENCODING {
        return;
    }
    let key = key(blocks);
    if let Some(open) = OPEN.lock().expect("witness lock").as_mut() {
        let operand = Operand {
            object: object.to_owned(),
            tensor: tensor.to_owned(),
        };
        open.names.insert(key, operand);
        open.capture.operands_noted = open.names.len();
    }
}

/// Record a finished call of the Q8_0 kernel. Called from inside that kernel, with what it read
/// and wrote. Other encodings go through the same kernel type and aren't covered.
pub(super) fn record(blocks: &[u8], encoding: &str, x: &[f32], y: &[f32]) {
    if !ACTIVE.load(Ordering::Acquire) || encoding != ENCODING {
        return;
    }
    // Outside the lock: it reads the whole matrix, on every call while capturing.
    let key = key(blocks);
    let mut guard = OPEN.lock().expect("witness lock");
    let Some(open) = guard.as_mut() else {
        return;
    };
    let call = open.capture.calls_observed;
    open.capture.calls_observed += 1;
    if open.recorded.len() >= open.max_operands || !open.recorded.insert(key) {
        return;
    }
    let operand = open.names.get(&key).cloned();
    open.capture.records.push(Record {
        call,
        operand,
        rows: y.len(),
        blocks: blocks.to_vec(),
        x: x.to_vec(),
        y: y.to_vec(),
    });
}

/// What the run is, as the caller knows it. Reported, not established: see the
/// format's documentation.
pub struct Binding<'a> {
    /// The commit declared at build time. `None` when the build declared none,
    /// and never guessed from a working tree.
    pub build_commit: Option<&'a str>,
    /// Hashed when the run started, because the file could be replaced during a run.
    pub binary_sha256: &'a str,
    /// The container's absolute path.
    pub container: &'a Path,
    pub component: &'a str,
    pub backend: &'a str,
    pub representation: &'a str,
    pub execution_arm: &'a str,
    pub tokens: &'a [u32],
    pub draft_depth: Option<usize>,
    pub logit_dump: &'a Path,
    pub logit_dump_sha256: &'a str,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect()
}

/// Write a file and force it to disk.
fn write_synced(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

fn write_stream(dir: &Path, name: String, bytes: &[u8]) -> std::io::Result<Value> {
    write_synced(&dir.join(&name), bytes)?;
    Ok(json!({"file": name, "bytes": bytes.len(), "sha256": sha256_hex(bytes)}))
}

/// Refuse a witness directory that couldn't be written, before a long run rather than after it.
pub fn check_target(dir: &Path) -> Result<(), String> {
    if dir.exists() {
        return Err(format!(
            "{} exists: a witness is never overwritten or merged into another",
            dir.display()
        ));
    }
    let parent = dir.parent().filter(|p| !p.as_os_str().is_empty());
    if parent.is_some_and(|p| !p.is_dir()) {
        return Err(format!(
            "{} has no parent directory to be created in",
            dir.display()
        ));
    }
    Ok(())
}

/// Write a witness to a new directory. Refuses an empty capture, and a record whose
/// operand has no name, rather than writing a witness nobody could bind to a tensor.
pub fn write(capture: &Capture, binding: &Binding<'_>, dir: &Path) -> Result<(), String> {
    if capture.records.is_empty() {
        return Err(format!(
            "no {KERNEL} call was observed, so there's nothing to witness: this run didn't \
             execute stored {ENCODING} blocks against an f32 activation on the CPU"
        ));
    }
    if let Some(r) = capture.records.iter().find(|r| r.operand.is_none()) {
        return Err(format!(
            "call {} read an operand with no noted tensor",
            r.call
        ));
    }
    let io = |e: std::io::Error| format!("writing the witness to {}: {e}", dir.display());
    // Fails if it exists: a witness is never overwritten or merged into another.
    std::fs::create_dir(dir).map_err(io)?;
    let mut records = Vec::with_capacity(capture.records.len());
    for (i, r) in capture.records.iter().enumerate() {
        let operand = r.operand.as_ref().expect("checked above");
        records.push(json!({
            "call": r.call,
            "object": operand.object,
            "tensor": operand.tensor,
            "encoding": ENCODING,
            "rows": r.rows,
            "columns": r.x.len(),
            "blocks": write_stream(dir, format!("{i:06}.blocks"), &r.blocks).map_err(io)?,
            "x": write_stream(dir, format!("{i:06}.x"), &f32_le_bytes(&r.x)).map_err(io)?,
            "y": write_stream(dir, format!("{i:06}.y"), &f32_le_bytes(&r.y)).map_err(io)?,
        }));
    }
    let manifest = json!({
        "schema": SCHEMA,
        "kernel": KERNEL,
        "activation": "f32",
        "execution_arm": binding.execution_arm,
        "build_commit": binding.build_commit,
        "binary_sha256": binding.binary_sha256,
        "container": binding.container,
        "component": binding.component,
        "backend": binding.backend,
        "representation": binding.representation,
        "tokens": binding.tokens,
        "draft_depth": binding.draft_depth,
        "logit_dump": {"path": binding.logit_dump, "sha256": binding.logit_dump_sha256},
        "calls_observed": capture.calls_observed,
        "operands_noted": capture.operands_noted,
        "max_operands": capture.max_operands,
        "records": records,
    });
    let text = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
    // Renamed into place, then the directory synced: a witness.json that exists is whole, and it
    // can't outlive the streams it describes.
    let partial = dir.join(format!("{MANIFEST}.partial"));
    write_synced(&partial, &text).map_err(io)?;
    std::fs::rename(&partial, dir.join(MANIFEST)).map_err(io)?;
    std::fs::File::open(dir)
        .and_then(|d| d.sync_all())
        .map_err(io)
}
