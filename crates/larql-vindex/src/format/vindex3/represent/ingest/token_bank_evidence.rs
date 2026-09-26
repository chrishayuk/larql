//! **A token bank, re-established at ingestion** (MEASURE-PLAN-2).
//!
//! The Kimi verifier checks fixed-length pre-embedded rows. A
//! `teacher-forced-token-bank/v1` bank holds token ids of varying length
//! per sample, so its positions are derived from the manifest's own token
//! counts and never from a fixed per-sample length. Everything the
//! procedure read is re-read here: the manifest's schema, payload
//! authority, sample order and derived `bank_id` (by `TokenBank::open`),
//! and every declared sample's payload against its seal.

use std::path::Path;

use super::super::state::EvidenceBank;
use super::super::token_bank::{TokenBank, TOKEN_BANK_SCHEMA};
use super::validation::{agrees, invalid, IngestionRefusal};

/// Verify the bank the record declares, returning the positions its
/// declared samples hold.
pub(super) fn verify(bank: &EvidenceBank, root: &Path) -> Result<u64, IngestionRefusal> {
    agrees("bank schema", TOKEN_BANK_SCHEMA, bank.schema.as_str())?;
    // A variable-length bank declares no per-sample length; a record
    // declaring one is describing some other bank.
    agrees("bank positions per sample", 0, bank.positions_per_sample)?;
    let opened = TokenBank::open(root).map_err(|e| invalid(format!("token bank: {e}")))?;
    let samples = &opened.manifest().samples;
    if bank.samples.len() > samples.len() {
        return Err(invalid(format!(
            "the record declares {} samples and the bank holds {}",
            bank.samples.len(),
            samples.len()
        )));
    }
    let mut positions = 0u64;
    for (index, declared) in bank.samples.iter().enumerate() {
        agrees(
            "bank sample order executed by procedure",
            samples[index].id.as_str(),
            declared.as_str(),
        )?;
        // `read` checks the payload against its seal before returning it.
        let ids = opened
            .read(index)
            .map_err(|e| invalid(format!("token bank sample {declared}: {e}")))?;
        agrees("bank sample token count", samples[index].tokens, ids.len())?;
        positions += ids.len() as u64;
    }
    Ok(positions)
}

#[cfg(test)]
#[path = "token_bank_evidence_tests.rs"]
mod tests;
