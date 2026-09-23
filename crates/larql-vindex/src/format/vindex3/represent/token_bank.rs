//! **A sealed token-id bank** — the corpus of MEASURE-PLAN-1
//! (`docs/measure-plan-1.md`), schema `teacher-forced-token-bank/v1`.
//!
//! The Kimi procedure's bank is pre-embedded rows, so it belongs to one
//! model's embedding table. A plan-level measurement feeds token ids to
//! whatever the plan executes, so its bank is ids, tokenised once by the
//! model's own tokenizer and sealed:
//!
//! ```text
//! <bank>/manifest.json   schema, bank id, prompt source, tokenizer digest,
//!                        tokenisation policy, one entry per sample
//! <bank>/seq-NNN.u32     one sample's ids, little-endian u32
//! ```
//!
//! Three facts are sealed, and each has its own refusal:
//!
//! - **the payloads:** every sample carries the sha256 of its bytes, and a
//!   read that does not match is [`TokenBankError::SealMismatch`];
//! - **the bank:** its id is derived from everything else in the manifest,
//!   so an edited manifest no longer names itself
//!   ([`TokenBankError::BankIdMismatch`]);
//! - **the model:** ids mean nothing under another tokenizer, so a bank is
//!   checked against the container's tokenizer digest before use
//!   ([`TokenBankError::CorpusNotForThisModel`]).
//!
//! The exporter reproduces `bench/prompts/quality-bank-1/run_bank.py`'s
//! tokenisation exactly: raw text, the tokenizer's own special tokens,
//! truncated to a cap, and a prompt shorter than [`MIN_SAMPLE_TOKENS`]
//! dropped. So the two paths see identical streams when `run_bank.py`
//! becomes a client of the procedure.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The manifest schema this module reads and writes.
pub const TOKEN_BANK_SCHEMA: &str = "teacher-forced-token-bank/v1";

/// What a payload file is. Checked on open, so a directory of some other
/// kind of `.u32` file is refused rather than read.
pub const TOKEN_BANK_PAYLOAD_AUTHORITY: &str = "teacher-forced-token-bank-payload/v1";

/// The manifest's file name inside a bank directory.
pub const MANIFEST_FILE: &str = "manifest.json";

/// A container's tokenizer, whose digest a bank is checked against.
pub const TOKENIZER_FILE: &str = "tokenizer.json";

/// Fewest ids a sample may have: two positions are the minimum that yield
/// one scored transition, and `run_bank.py` keeps `len(ids) >= 3`.
pub const MIN_SAMPLE_TOKENS: usize = 3;

/// Bytes per stored token id.
const TOKEN_BYTES: usize = 4;

/// How prompt text became ids. One variant today, named so a templated
/// bank is a different bank rather than an unmarked one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemplatePolicy {
    /// The prompt text as written, with the tokenizer's own special tokens.
    Raw,
}

/// Where the prompts came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptSource {
    /// The prompt file's own `bank` name, e.g. `quality-bank-1`.
    pub bank: String,
    /// sha256 of the prompt file's bytes.
    pub sha256: String,
}

/// One sample: which prompt it is, and the seal on its ids.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenBankSample {
    /// `seq-NNN`, in bank order from `seq-000`.
    pub id: String,
    /// The prompt's id in the prompt file.
    pub prompt_id: String,
    /// The prompt's category in the prompt file.
    pub category: String,
    /// Ids in the payload.
    pub tokens: usize,
    /// sha256 of the payload's bytes.
    pub sha256: String,
}

/// A bank's manifest. Field order is the serialised order, and the bank id
/// is derived from the serialisation, so it is part of the format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenBankManifest {
    pub schema: String,
    /// Derived: sha256 of this manifest serialised with `bank_id` empty.
    pub bank_id: String,
    pub prompts: PromptSource,
    /// sha256 of the tokenizer file the ids came from.
    pub tokenizer_sha256: String,
    pub template: TemplatePolicy,
    /// Whether the tokenizer's special tokens were added.
    pub add_special_tokens: bool,
    /// The truncation cap, in ids.
    pub max_tokens: usize,
    pub payload_authority: String,
    pub samples: Vec<TokenBankSample>,
}

/// Why a bank cannot be read, or cannot be used for this model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenBankError {
    /// A file could not be read or written.
    Io { path: PathBuf, detail: String },
    /// A file is not what it should be (unparseable, wrong size, no prompts).
    Malformed { detail: String },
    /// The manifest names a schema this module does not read.
    UnknownSchema { found: String },
    /// The manifest's payload authority is not this module's.
    UnknownPayloadAuthority { found: String },
    /// Samples are not `seq-000`, `seq-001`, … in order.
    SampleOrder { index: usize, found: String },
    /// The manifest's bank id is not the id its contents derive.
    BankIdMismatch { recorded: String, derived: String },
    /// A payload's bytes are not the bytes the manifest sealed.
    SealMismatch {
        sample: String,
        expected: String,
        found: String,
    },
    /// The bank was tokenised by a different tokenizer than the model's.
    CorpusNotForThisModel { bank: String, model: String },
    /// An export would write over an existing directory.
    OutputExists { path: PathBuf },
}

impl std::fmt::Display for TokenBankError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for TokenBankError {}

/// An opened bank: a manifest whose schema, authority, order and id have
/// been checked. Payloads are checked when read.
#[derive(Debug, Clone)]
pub struct TokenBank {
    dir: PathBuf,
    manifest: TokenBankManifest,
}

impl TokenBank {
    /// Open and check a bank directory.
    pub fn open(dir: &Path) -> Result<Self, TokenBankError> {
        let _ = dir;
        todo!("MEASURE-PLAN-1 PR 1")
    }

    /// The checked manifest.
    pub fn manifest(&self) -> &TokenBankManifest {
        &self.manifest
    }

    /// The directory the bank was opened from.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Samples in the bank.
    pub fn sample_count(&self) -> usize {
        self.manifest.samples.len()
    }

    /// Refuse unless the bank was tokenised by the tokenizer whose digest
    /// is `model_tokenizer_sha256`.
    pub fn check_tokenizer(&self, model_tokenizer_sha256: &str) -> Result<(), TokenBankError> {
        let _ = model_tokenizer_sha256;
        todo!("MEASURE-PLAN-1 PR 1")
    }

    /// Sample `index`'s ids, after checking its bytes against the seal.
    pub fn read(&self, index: usize) -> Result<Vec<u32>, TokenBankError> {
        let _ = index;
        todo!("MEASURE-PLAN-1 PR 1")
    }
}

/// sha256 of a container's tokenizer file, the digest a bank is checked
/// against.
pub fn container_tokenizer_sha256(container: &Path) -> Result<String, TokenBankError> {
    let _ = container;
    todo!("MEASURE-PLAN-1 PR 1")
}

/// Tokenise a prompt file into a new bank at `out`, which must not exist.
pub fn export(
    prompts: &Path,
    tokenizer: &Path,
    max_tokens: usize,
    out: &Path,
) -> Result<TokenBankManifest, TokenBankError> {
    let _ = (prompts, tokenizer, max_tokens, out, TOKEN_BYTES);
    todo!("MEASURE-PLAN-1 PR 1")
}

#[cfg(test)]
#[path = "token_bank_tests.rs"]
mod tests;
