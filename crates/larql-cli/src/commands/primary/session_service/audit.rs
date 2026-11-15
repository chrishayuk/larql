//! The audit log: one JSON object per line, appended (`O_APPEND`) and synced as each request is answered.
//!
//! Every record carries `seq` (monotonically increasing, from 1, across resumes), `time_unix_ms`, `event` and
//! `budget_used` after it. A fresh log is created exclusively: an existing file is refused unless the policy
//! says `resume`, and a resumed log must parse whole, end in a record with a `seq` and a `budget_used`, and
//! name the same container (by its index digest) in its last `start` record.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

use serde_json::{json, Value};

/// The event names a record carries.
pub const EVENT_START: &str = "start";
pub const EVENT_STOP: &str = "stop";
pub const EVENT_REQUEST: &str = "request";
pub const EVENT_CONNECT: &str = "connect";
pub const EVENT_DISCONNECT: &str = "disconnect";
pub const EVENT_REJECTED: &str = "connection_rejected";

pub struct AuditLog {
    file: File,
    next_seq: u64,
}

/// Where a log starts: its next sequence number and the budget already used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resumed {
    pub next_seq: u64,
    pub budget_used: u64,
}

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

impl AuditLog {
    /// Open the log at `path`: created, or continued when `resume`. `container_index_sha256` is checked
    /// against a resumed log's last start record.
    pub fn open(
        path: &Path,
        resume: bool,
        container_index_sha256: &str,
    ) -> Result<(Self, Resumed), String> {
        let exists = path.symlink_metadata().is_ok();
        if exists && !resume {
            return Err(format!(
                "the audit log {} already exists; set \"resume\": true in the policy to continue it",
                path.display()
            ));
        }
        let resumed = if exists {
            Self::read_existing(path, container_index_sha256)?
        } else {
            Resumed {
                next_seq: 1,
                budget_used: 0,
            }
        };
        let mut options = OpenOptions::new();
        options.append(true);
        if exists {
            // A resumed log is a regular file, never a link the service would write through.
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NOFOLLOW);
        } else {
            options.create_new(true);
        }
        let file = options
            .open(path)
            .map_err(|e| format!("open the audit log {}: {e}", path.display()))?;
        Ok((
            Self {
                file,
                next_seq: resumed.next_seq,
            },
            resumed,
        ))
    }

    fn read_existing(path: &Path, container_index_sha256: &str) -> Result<Resumed, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("read the audit log {}: {e}", path.display()))?;
        let mut last: Option<Value> = None;
        let mut last_start: Option<Value> = None;
        for (i, line) in text.lines().enumerate() {
            let record: Value = serde_json::from_str(line).map_err(|e| {
                format!(
                    "the audit log {} is not JSONL at line {}: {e}; refusing to resume it",
                    path.display(),
                    i + 1
                )
            })?;
            if record["event"] == EVENT_START {
                last_start = Some(record.clone());
            }
            last = Some(record);
        }
        let Some(last) = last else {
            return Ok(Resumed {
                next_seq: 1,
                budget_used: 0,
            });
        };
        let (Some(seq), Some(used)) = (last["seq"].as_u64(), last["budget_used"].as_u64()) else {
            return Err(format!(
                "the audit log {} ends in a record without `seq` and `budget_used`; refusing to resume it",
                path.display()
            ));
        };
        if let Some(start) = last_start {
            let logged = start["container_index_sha256"].as_str().unwrap_or_default();
            if logged != container_index_sha256 {
                return Err(format!(
                    "the audit log {} was written for container index {logged}; this policy's container \
                     is {container_index_sha256}",
                    path.display()
                ));
            }
        }
        Ok(Resumed {
            next_seq: seq + 1,
            budget_used: used,
        })
    }

    /// The sequence number the next record gets.
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    /// Append `fields` as one record, with `seq`, `time_unix_ms`, `event` and `budget_used`, and sync it.
    /// Returns the record's sequence number.
    pub fn append(&mut self, event: &str, budget_used: u64, fields: Value) -> std::io::Result<u64> {
        let seq = self.next_seq;
        let mut record = json!({
            "seq": seq,
            "time_unix_ms": now_unix_ms(),
            "event": event,
            "budget_used": budget_used,
        });
        if let (Some(out), Value::Object(extra)) = (record.as_object_mut(), fields) {
            for (k, v) in extra {
                out.entry(k).or_insert(v);
            }
        }
        let mut line = serde_json::to_vec(&record).map_err(std::io::Error::other)?;
        line.push(b'\n');
        self.file.write_all(&line)?;
        self.file.sync_data()?;
        self.next_seq += 1;
        Ok(seq)
    }
}
