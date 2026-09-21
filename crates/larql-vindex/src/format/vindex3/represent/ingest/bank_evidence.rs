//! Bind a declared teacher-forced bank to the physical rows its procedure reads.
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

use super::super::{candidate_authority::digest_range, state::EvidenceBank};
use super::validation::{agrees, invalid, IngestionRefusal};

const PAYLOAD_AUTHORITY: &str = "teacher-forced-bank-payload/v1";

#[derive(Deserialize)]
struct Manifest {
    payload_authority: String,
    sequences: u64,
    positions: u64,
    hidden: u64,
    regime: String,
    token_ids: Vec<Vec<u32>>,
    payloads: BTreeMap<String, Payload>,
}

#[derive(Deserialize)]
struct Payload {
    len: u64,
    sha256: String,
}

/// The caller has already matched these exact manifest bytes to the bank ID.
/// A manifest without payload authority is insufficient even if its hash matches.
pub(super) fn verify(
    bank: &EvidenceBank,
    root: &Path,
    bytes: &[u8],
) -> Result<(), IngestionRefusal> {
    agrees(
        "bank schema",
        "kimi-teacher-forced/v1",
        bank.schema.as_str(),
    )?;
    let manifest: Manifest = serde_json::from_slice(bytes)
        .map_err(|e| invalid(format!("bank payload authority missing or malformed: {e}")))?;
    agrees(
        "bank payload authority version",
        PAYLOAD_AUTHORITY,
        manifest.payload_authority.as_str(),
    )?;
    agrees("bank regime", "teacher-forced", manifest.regime.as_str())?;
    agrees(
        "bank positions per sample",
        u64::from(bank.positions_per_sample),
        manifest.positions,
    )?;
    if manifest.hidden == 0
        || manifest.sequences < bank.samples.len() as u64
        || manifest.token_ids.len() as u64 != manifest.sequences
        || manifest
            .token_ids
            .iter()
            .any(|s| s.len() as u64 != manifest.positions)
    {
        return Err(invalid(
            "bank dimensions or token inventory disagree with its declaration",
        ));
    }
    let expected_len = manifest
        .positions
        .checked_mul(manifest.hidden)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| invalid("bank row dimensions overflow"))?;
    for (seq, sample) in bank.samples.iter().enumerate() {
        // This is the exact prefix the shipped procedure currently consumes.
        // Accepting another subset/order would invent actuation support.
        agrees(
            "bank sample order executed by procedure",
            format!("seq-{seq:03}"),
            sample.as_str(),
        )?;
        let file = format!("seq_{seq}.f32");
        let payload = manifest
            .payloads
            .get(&file)
            .ok_or_else(|| invalid(format!("bank payload {file} has no seal")))?;
        agrees("bank payload dimensions", expected_len, payload.len)?;
        let path = root.join(&file);
        let observed_len = std::fs::metadata(&path)
            .map_err(|e| invalid(format!("bank payload {file}: {e}")))?
            .len();
        agrees("bank payload length", payload.len, observed_len)?;
        let observed = digest_range(&path, 0, observed_len)
            .map_err(|e| invalid(format!("bank payload {file}: {e}")))?;
        agrees(&format!("bank payload {file}"), &payload.sha256, &observed)?;
    }
    Ok(())
}
