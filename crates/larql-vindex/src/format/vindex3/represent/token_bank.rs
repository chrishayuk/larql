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
use sha2::{Digest, Sha256};

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
        let path = dir.join(MANIFEST_FILE);
        let bytes = read_file(&path)?;
        let manifest: TokenBankManifest =
            serde_json::from_slice(&bytes).map_err(|e| TokenBankError::Malformed {
                detail: format!("{}: {e}", path.display()),
            })?;
        if manifest.schema != TOKEN_BANK_SCHEMA {
            return Err(TokenBankError::UnknownSchema {
                found: manifest.schema,
            });
        }
        if manifest.payload_authority != TOKEN_BANK_PAYLOAD_AUTHORITY {
            return Err(TokenBankError::UnknownPayloadAuthority {
                found: manifest.payload_authority,
            });
        }
        for (index, sample) in manifest.samples.iter().enumerate() {
            if sample.id != sample_id(index) {
                return Err(TokenBankError::SampleOrder {
                    index,
                    found: sample.id.clone(),
                });
            }
        }
        let derived = bank_id(&manifest)?;
        if manifest.bank_id != derived {
            return Err(TokenBankError::BankIdMismatch {
                recorded: manifest.bank_id,
                derived,
            });
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            manifest,
        })
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
        if self.manifest.tokenizer_sha256 == model_tokenizer_sha256 {
            return Ok(());
        }
        Err(TokenBankError::CorpusNotForThisModel {
            bank: self.manifest.tokenizer_sha256.clone(),
            model: model_tokenizer_sha256.to_string(),
        })
    }

    /// Sample `index`'s ids, after checking its bytes against the seal.
    pub fn read(&self, index: usize) -> Result<Vec<u32>, TokenBankError> {
        let sample = self
            .manifest
            .samples
            .get(index)
            .ok_or_else(|| TokenBankError::Malformed {
                detail: format!("no sample {index}; the bank has {}", self.sample_count()),
            })?;
        let bytes = read_file(&self.dir.join(payload_file(&sample.id)))?;
        let found = sha256_hex(&bytes);
        if found != sample.sha256 {
            return Err(TokenBankError::SealMismatch {
                sample: sample.id.clone(),
                expected: sample.sha256.clone(),
                found,
            });
        }
        let ids: Vec<u32> = bytes
            .chunks_exact(TOKEN_BYTES)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if ids.len() != sample.tokens || bytes.len() % TOKEN_BYTES != 0 {
            return Err(TokenBankError::Malformed {
                detail: format!(
                    "{}: {} bytes for {} ids",
                    sample.id,
                    bytes.len(),
                    sample.tokens
                ),
            });
        }
        Ok(ids)
    }
}

/// sha256 of a container's tokenizer file, the digest a bank is checked
/// against.
pub fn container_tokenizer_sha256(container: &Path) -> Result<String, TokenBankError> {
    let path = container.join(TOKENIZER_FILE);
    crate::format::checksums::sha256_file(&path).map_err(|e| TokenBankError::Io {
        path,
        detail: e.to_string(),
    })
}

/// Tokenise a prompt file into a new bank at `out`, which must not exist.
pub fn export(
    prompts: &Path,
    tokenizer: &Path,
    max_tokens: usize,
    out: &Path,
) -> Result<TokenBankManifest, TokenBankError> {
    if out.exists() {
        return Err(TokenBankError::OutputExists {
            path: out.to_path_buf(),
        });
    }
    let prompt_bytes = read_file(prompts)?;
    let prompt_file: PromptFile =
        serde_json::from_slice(&prompt_bytes).map_err(|e| TokenBankError::Malformed {
            detail: format!("{}: {e}", prompts.display()),
        })?;
    if prompt_file.prompts.is_empty() {
        return Err(TokenBankError::Malformed {
            detail: format!("{}: no prompts", prompts.display()),
        });
    }
    let tk = crate::tokenizers::Tokenizer::from_file(tokenizer).map_err(|e| {
        TokenBankError::Malformed {
            detail: format!("{}: {e}", tokenizer.display()),
        }
    })?;
    let tokenizer_sha256 = sha256_hex(&read_file(tokenizer)?);

    let mut samples = Vec::new();
    let mut payloads = Vec::new();
    for prompt in &prompt_file.prompts {
        let encoded = tk
            .encode(prompt.text.as_str(), ADD_SPECIAL_TOKENS)
            .map_err(|e| TokenBankError::Malformed {
                detail: format!("prompt {}: {e}", prompt.id),
            })?;
        let mut ids = encoded.get_ids().to_vec();
        ids.truncate(max_tokens);
        if ids.len() < MIN_SAMPLE_TOKENS {
            continue;
        }
        let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let id = sample_id(samples.len());
        samples.push(TokenBankSample {
            id: id.clone(),
            prompt_id: prompt.id.clone(),
            category: prompt.category.clone(),
            tokens: ids.len(),
            sha256: sha256_hex(&bytes),
        });
        payloads.push((id, bytes));
    }
    let mut manifest = TokenBankManifest {
        schema: TOKEN_BANK_SCHEMA.to_string(),
        bank_id: String::new(),
        prompts: PromptSource {
            bank: prompt_file.bank,
            sha256: sha256_hex(&prompt_bytes),
        },
        tokenizer_sha256,
        template: TemplatePolicy::Raw,
        add_special_tokens: ADD_SPECIAL_TOKENS,
        max_tokens,
        payload_authority: TOKEN_BANK_PAYLOAD_AUTHORITY.to_string(),
        samples,
    };
    manifest.bank_id = bank_id(&manifest)?;

    std::fs::create_dir_all(out).map_err(|e| io_error(out, e))?;
    for (id, bytes) in &payloads {
        let path = out.join(payload_file(id));
        std::fs::write(&path, bytes).map_err(|e| io_error(&path, e))?;
    }
    // The manifest last: a bank directory without one is visibly
    // incomplete, never a bank that names payloads it does not have.
    let path = out.join(MANIFEST_FILE);
    let json = serde_json::to_vec_pretty(&manifest).map_err(|e| TokenBankError::Malformed {
        detail: e.to_string(),
    })?;
    std::fs::write(&path, json).map_err(|e| io_error(&path, e))?;
    Ok(manifest)
}

/// Whether the exporter adds the tokenizer's special tokens, as
/// `run_bank.py` does (`tokenizers`' `encode` default).
const ADD_SPECIAL_TOKENS: bool = true;

/// The prompt file the exporter reads: Q-BANK-1's `prompts.json` shape.
#[derive(Deserialize)]
struct PromptFile {
    bank: String,
    prompts: Vec<PromptEntry>,
}

#[derive(Deserialize)]
struct PromptEntry {
    id: String,
    category: String,
    text: String,
}

/// `seq-NNN` for sample `index`.
fn sample_id(index: usize) -> String {
    format!("seq-{index:03}")
}

/// A sample's payload file name.
fn payload_file(id: &str) -> String {
    format!("{id}.u32")
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// The id a manifest's contents derive: sha256 of its serialisation with
/// the id field empty.
fn bank_id(manifest: &TokenBankManifest) -> Result<String, TokenBankError> {
    let unnamed = TokenBankManifest {
        bank_id: String::new(),
        ..manifest.clone()
    };
    let bytes = serde_json::to_vec(&unnamed).map_err(|e| TokenBankError::Malformed {
        detail: e.to_string(),
    })?;
    Ok(sha256_hex(&bytes))
}

fn read_file(path: &Path) -> Result<Vec<u8>, TokenBankError> {
    std::fs::read(path).map_err(|e| io_error(path, e))
}

fn io_error(path: &Path, e: std::io::Error) -> TokenBankError {
    TokenBankError::Io {
        path: path.to_path_buf(),
        detail: e.to_string(),
    }
}

#[cfg(test)]
#[path = "token_bank_tests.rs"]
mod tests;
