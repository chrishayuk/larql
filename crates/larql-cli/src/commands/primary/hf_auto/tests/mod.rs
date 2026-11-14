//! Gates for the `hf://` first-run encode — every branch without a
//! network or the user's caches.

mod classify;
mod encode;
mod refresh;
mod refusal;
mod resolve;
mod store;

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};

use larql_vindex::format::filenames::INDEX_JSON;

use super::classify::RepoProbe;
use super::resolve::CheckpointEncoder;

type BoxErr = Box<dyn std::error::Error>;

/// A full-length commit id for fixtures.
const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

/// A repo id for fixtures.
const REPO: &str = "acme/tiny-model";

/// A hub that carries `files` and resolves every revision to `commit`.
/// `fail` makes every question an error, as an offline hub would.
#[derive(Default)]
struct FakeProbe {
    files: BTreeSet<&'static str>,
    commit: Option<String>,
    fail: bool,
    asked: RefCell<Vec<String>>,
}

impl FakeProbe {
    fn carrying(files: &[&'static str]) -> Self {
        Self {
            files: files.iter().copied().collect(),
            commit: Some(COMMIT.to_string()),
            ..Self::default()
        }
    }

    fn offline() -> Self {
        Self {
            fail: true,
            ..Self::default()
        }
    }

    fn asked(&self) -> Vec<String> {
        self.asked.borrow().clone()
    }
}

impl RepoProbe for FakeProbe {
    fn has_file(&self, _repo: &str, _revision: &str, file: &str) -> Result<bool, BoxErr> {
        self.asked.borrow_mut().push(file.to_string());
        if self.fail {
            return Err("offline".into());
        }
        Ok(self.files.contains(file))
    }

    fn commit(&self, _repo: &str, _revision: &str) -> Result<Option<String>, BoxErr> {
        self.asked.borrow_mut().push("<commit>".to_string());
        if self.fail {
            return Err("offline".into());
        }
        Ok(self.commit.clone())
    }
}

/// An encoder that writes a minimal container, or fails half-way after
/// writing one — the shape an interrupted real encode leaves.
#[derive(Default)]
struct FakeEncoder {
    fail: bool,
    calls: RefCell<Vec<(String, PathBuf)>>,
}

impl FakeEncoder {
    fn failing() -> Self {
        Self {
            fail: true,
            ..Self::default()
        }
    }

    fn calls(&self) -> Vec<(String, PathBuf)> {
        self.calls.borrow().clone()
    }
}

impl CheckpointEncoder for FakeEncoder {
    fn encode(&self, spec: &str, out: &Path, status: &mut dyn Write) -> Result<(), BoxErr> {
        self.calls
            .borrow_mut()
            .push((spec.to_string(), out.to_path_buf()));
        std::fs::create_dir_all(out)?;
        std::fs::write(out.join(INDEX_JSON), b"{}")?;
        writeln!(status, "  fake encode")?;
        if self.fail {
            return Err("encode refused".into());
        }
        Ok(())
    }
}

/// Write a hub-layout vindex snapshot for `repo` under `hub`.
fn fake_hub_vindex(hub: &Path, repo: &str) -> PathBuf {
    let (owner, name) = repo.split_once('/').unwrap();
    let snap = hub.join(format!("models--{owner}--{name}/snapshots/{COMMIT}"));
    std::fs::create_dir_all(&snap).unwrap();
    std::fs::write(snap.join(INDEX_JSON), b"{}").unwrap();
    snap
}
