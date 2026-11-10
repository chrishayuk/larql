//! What a run was made of: exact model, tokenizer, source, bank and harness.

use serde_json::{json, Value};
use std::path::Path;
use std::process::Command;

use crate::bank::{hex, sha256_file};
use sha2::{Digest, Sha256};

/// The harness's own source, so a result names the code that produced it.
const SOURCES: &[(&str, &str)] = &[
    ("main.rs", include_str!("main.rs")),
    ("args.rs", include_str!("args.rs")),
    ("bank.rs", include_str!("bank.rs")),
    ("consts.rs", include_str!("consts.rs")),
    ("controls.rs", include_str!("controls.rs")),
    ("fact.rs", include_str!("fact.rs")),
    ("faults.rs", include_str!("faults.rs")),
    ("guard.rs", include_str!("guard.rs")),
    ("infer.rs", include_str!("infer.rs")),
    ("observer.rs", include_str!("observer.rs")),
    ("provenance.rs", include_str!("provenance.rs")),
];

fn git(args: &[&str], dir: &str) -> Option<String> {
    let out = Command::new("git")
        .args(["-C", dir])
        .args(args)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

pub fn collect(
    container: &Path,
    bank_path: &Path,
    bank_sha256: &str,
    arm: &str,
    tag: &str,
    command: &str,
) -> Result<Value, String> {
    let mut sources = Sha256::new();
    for (name, text) in SOURCES {
        sources.update(name.as_bytes());
        sources.update(text.as_bytes());
    }
    // The directory this binary was compiled from: its commit and tracked changes say which
    // library the harness linked against.
    let built_in = env!("CARGO_MANIFEST_DIR");
    let commit = git(&["rev-parse", "HEAD"], built_in);
    let tracked_changes = git(&["status", "--porcelain", "--untracked-files=no"], built_in)
        .map(|s| s.lines().count());
    Ok(json!({
        "arm": arm,
        "model_tag": tag,
        "command": command,
        "harness": {"sources_sha256": hex(&sources.finalize()), "files": SOURCES.iter().map(|(n, _)| *n).collect::<Vec<_>>()},
        "source": {"built_in": built_in, "commit": commit, "tracked_changes": tracked_changes},
        "container": {
            "path": container.display().to_string(),
            "index_sha256": sha256_file(&container.join("index.json"))?,
            "tokenizer_sha256": sha256_file(&container.join("tokenizer.json"))?,
        },
        "bank": {"path": bank_path.display().to_string(), "sha256": bank_sha256},
        "build": {"debug_assertions": cfg!(debug_assertions)},
    }))
}
