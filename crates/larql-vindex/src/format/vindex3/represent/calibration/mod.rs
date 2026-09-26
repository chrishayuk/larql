//! CAL-1.1: content-bound calibration statistics from the actual decode path.
//!
//! This is capture authority, not encoder dispatch or quality admission. A
//! prepared capture owns the image it observes; callers cannot label a canonical
//! session with a candidate prefix. Statistics are raw f64 sums (X^T X), with
//! sequence resets and explicit position masks. One site is retained at a time.
//! Dense softmax single-stream prefixes are the initial supported scope.

mod artifact;
mod capture;
mod identity;
mod statistic;

pub use artifact::{CalibrationArtifact, CalibrationKey, CalibrationManifest};
pub use capture::PreparedCalibration;
pub use identity::{Boundary, CalibrationSite, Projection};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::VindexError;

pub const SCHEMA: &str = "represent-calibration/v1";
pub const CAPTURE_REVISION: &str = "decode-inputs-f64-sum/v1";

fn refused(message: impl std::fmt::Display) -> VindexError {
    VindexError::Parse(format!("calibration refused: {message}"))
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_digest(value: &impl Serialize) -> Result<String, VindexError> {
    Ok(digest(&serde_json::to_vec(value).map_err(refused)?))
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

/// Reconstruction validation may select N; final admission is deliberately not
/// an encodable population here, so it cannot accidentally train a recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Population {
    Calibration,
    ReconstructionValidation,
}

/// Both forms are raw, uncentered sums; sample count is carried separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatisticKind {
    DenseGram,
    DiagonalSecondMoment,
}

impl StatisticKind {
    fn elements(self, width: usize) -> Result<usize, VindexError> {
        if width == 0 {
            return Err(refused("zero input width"));
        }
        match self {
            Self::DenseGram => width
                .checked_mul(width)
                .ok_or_else(|| refused("Gram size overflow")),
            Self::DiagonalSecondMoment => Ok(width),
        }
    }
}

/// Every token executes, including masked positions. The mask selects which
/// input rows contribute; it never removes context. KV resets between sequences.
#[derive(Debug, Clone, Serialize)]
pub struct CalibrationSequence {
    pub tokens: Vec<u32>,
    pub include: Vec<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CalibrationBank {
    name: String,
    tokenizer_sha256: String,
    population: Population,
    token_bank_id: Option<String>,
    sequences: Vec<CalibrationSequence>,
}

impl CalibrationBank {
    pub fn new(
        name: String,
        tokenizer_sha256: String,
        population: Population,
        sequences: Vec<CalibrationSequence>,
    ) -> Result<Self, VindexError> {
        if name.trim().is_empty() || !valid_digest(&tokenizer_sha256) {
            return Err(refused("bank name and tokenizer SHA-256 are required"));
        }
        if sequences.is_empty()
            || sequences
                .iter()
                .any(|s| s.tokens.is_empty() || s.tokens.len() != s.include.len())
            || !sequences.iter().any(|s| s.include.iter().any(|v| *v))
        {
            return Err(refused(
                "bank requires nonempty sequences, exact masks and selected positions",
            ));
        }
        Ok(Self {
            name,
            tokenizer_sha256,
            population,
            token_bank_id: None,
            sequences,
        })
    }

    /// Reuse the existing sealed bank authority. Every payload is verified by
    /// TokenBank::read; calibration adds explicit masks and population without
    /// replacing the bank's original identity or tokenization policy.
    pub fn from_token_bank(
        bank: &super::token_bank::TokenBank,
        masks: Vec<Vec<bool>>,
        population: Population,
    ) -> Result<Self, VindexError> {
        if masks.len() != bank.sample_count() {
            return Err(refused(
                "mask sequence count differs from sealed token bank",
            ));
        }
        let sequences = masks
            .into_iter()
            .enumerate()
            .map(|(i, include)| {
                Ok(CalibrationSequence {
                    tokens: bank.read(i).map_err(refused)?,
                    include,
                })
            })
            .collect::<Result<Vec<_>, VindexError>>()?;
        let mut capture = Self::new(
            bank.manifest().prompts.bank.clone(),
            bank.manifest().tokenizer_sha256.clone(),
            population,
            sequences,
        )?;
        capture.token_bank_id = Some(bank.manifest().bank_id.clone());
        Ok(capture)
    }

    /// Domain-separated digest includes token order, sequence boundaries, masks,
    /// tokenizer and population. It is distinct from the frozen R4 token digest;
    /// this envelope does not replace R4's disjointness or N-selection gates.
    pub fn sha256(&self) -> Result<String, VindexError> {
        json_digest(&(SCHEMA, "bank", self))
    }

    pub fn sample_count(&self) -> u64 {
        self.sequences
            .iter()
            .map(|s| s.include.iter().filter(|v| **v).count() as u64)
            .sum()
    }
}

#[cfg(test)]
mod tests;
