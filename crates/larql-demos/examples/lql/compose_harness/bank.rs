//! The prompt bank, loaded from `bank.json`. Which sets a run may read is decided in `guard`.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Fact {
    pub relation: String,
    pub entity: String,
    pub target: String,
    pub canonical_prompt: String,
    pub same_template: Vec<String>,
    pub related: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct Bank {
    pub development: Vec<Fact>,
    pub facts: Vec<Fact>,
    pub unrelated: Vec<String>,
    /// The prompts whose predictions an atomicity check compares before and after a failed install.
    pub atomicity_probe: Vec<String>,
}

pub struct LoadedBank {
    pub bank: Bank,
    pub sha256: String,
}

pub fn load(path: &Path) -> Result<LoadedBank, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let bank = serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(LoadedBank {
        bank,
        sha256: hex(&Sha256::digest(&bytes)),
    })
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn sha256_file(path: &Path) -> Result<String, String> {
    std::fs::read(path)
        .map(|b| hex(&Sha256::digest(&b)))
        .map_err(|e| format!("{}: {e}", path.display()))
}
