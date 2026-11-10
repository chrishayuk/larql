//! The registered facts are unreachable until the pre-registration is frozen, and every attempt
//! to reach them is written down, allowed or not.

use std::io::Write;
use std::path::Path;

use crate::consts::{FROZEN_MARKER, REGISTERED_LEDGER};

/// Appends the attempt to the ledger beside the bank, then allows it only if the `FROZEN` marker
/// holds exactly the bank's hash.
pub fn admit_registered(bank_dir: &Path, bank_sha256: &str, command: &str) -> Result<(), String> {
    let marker = std::fs::read_to_string(bank_dir.join(FROZEN_MARKER)).ok();
    let allowed = marker.as_deref().map(str::trim) == Some(bank_sha256);
    let entry = serde_json::json!({
        "command": command,
        "bank_sha256": bank_sha256,
        "frozen_marker_present": marker.is_some(),
        "allowed": allowed,
    });
    let mut ledger = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(bank_dir.join(REGISTERED_LEDGER))
        .map_err(|e| format!("cannot open the registered-attempt ledger: {e}"))?;
    writeln!(ledger, "{entry}").map_err(|e| format!("cannot write the ledger: {e}"))?;
    if allowed {
        Ok(())
    } else {
        Err(format!(
            "registered facts are not reachable: {FROZEN_MARKER} is missing or does not hold this bank's hash \
             ({bank_sha256}). The attempt was recorded in {REGISTERED_LEDGER}."
        ))
    }
}
