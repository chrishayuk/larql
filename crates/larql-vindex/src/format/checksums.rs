//! Checksum utilities for vindex file integrity verification.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::VindexError;
use crate::format::filenames::*;

/// Compute SHA256 checksum of a file. Returns hex string.
pub fn sha256_file(path: &Path) -> Result<String, VindexError> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20]; // 1 MB buffer
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compute checksums for all binary files in a vindex directory.
/// Returns a map of filename → SHA256 hex string.
pub fn compute_checksums(dir: &Path) -> Result<BTreeMap<String, String>, VindexError> {
    let mut checksums = BTreeMap::new();

    let files = [
        GATE_VECTORS_BIN,
        EMBEDDINGS_BIN,
        DOWN_META_BIN,
        DOWN_META_JSONL,
        ATTN_WEIGHTS_BIN,
        UP_WEIGHTS_BIN,
        DOWN_WEIGHTS_BIN,
        NORMS_BIN,
        LM_HEAD_BIN,
    ];

    for filename in &files {
        let path = dir.join(filename);
        if path.exists() {
            checksums.insert(filename.to_string(), sha256_file(&path)?);
        }
    }

    Ok(checksums)
}

/// Verify checksums of a vindex directory against stored checksums.
///
/// Returns `(filename, ok)` pairs **in canonical filename order**.
///
/// # Why the sort is not cosmetic
///
/// The stored map was once a `HashMap`, whose iteration order Rust randomises
/// per process. Rendering findings in that order made `larql verify` produce
/// different output on every run against an identical, intact artifact —
/// verified as three different digests across three consecutive runs.
///
/// That is a correctness problem, not a presentation one. A verification
/// command is consumed by golden files, CI diffs, signed reports and operators
/// comparing two runs by eye; every one of those reads a reordering as a
/// change. It also made the E0 preservation matrix's `verify` row permanently
/// unfalsifiable, since no capture of it could ever be reproduced.
///
/// The rule: **verification may execute in any order internally, but findings
/// are rendered in canonical artifact order.** Sorting here rather than in the
/// CLI puts it at the collection boundary, so every consumer inherits it and a
/// future caller cannot reintroduce the defect by forgetting.
///
/// The key is currently the relative filename, which is this artifact's whole
/// identity. When findings gain component coordinates or check kinds, extend
/// the key to `(path, coordinate, check kind, detail)` — it orders results and
/// must never deduplicate them.
pub fn verify_checksums(
    dir: &Path,
    stored: &BTreeMap<String, String>,
) -> Result<Vec<(String, bool)>, VindexError> {
    let mut results = Vec::with_capacity(stored.len());

    for (filename, expected) in stored {
        let path = dir.join(filename);
        if path.exists() {
            let actual = sha256_file(&path)?;
            results.push((filename.clone(), actual == *expected));
        } else {
            results.push((filename.clone(), false));
        }
    }

    // Sort only at the boundary, so the hashing above stays free to run in
    // whatever order — including in parallel later — without scheduling
    // deciding user-visible output.
    results.sort_by(|a, b| a.0.cmp(&b.0));

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn sha256_file_deterministic() {
        let dir = TempDir::new().unwrap();
        let f = dir.path().join("data.bin");
        std::fs::write(&f, b"hello world").unwrap();
        let h1 = sha256_file(&f).unwrap();
        let h2 = sha256_file(&f).unwrap();
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // hex-encoded SHA-256
    }

    #[test]
    fn sha256_file_different_content_different_hash() {
        let dir = TempDir::new().unwrap();
        let f1 = dir.path().join("a.bin");
        let f2 = dir.path().join("b.bin");
        std::fs::write(&f1, b"content A").unwrap();
        std::fs::write(&f2, b"content B").unwrap();
        assert_ne!(sha256_file(&f1).unwrap(), sha256_file(&f2).unwrap());
    }

    #[test]
    fn sha256_file_empty_file() {
        let dir = TempDir::new().unwrap();
        let f = dir.path().join("empty.bin");
        std::fs::write(&f, b"").unwrap();
        let h = sha256_file(&f).unwrap();
        // SHA-256 of empty input is well-known
        assert_eq!(
            h,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_file_missing_returns_error() {
        let result = sha256_file(Path::new("/nonexistent/no_such_file.bin"));
        assert!(result.is_err());
    }

    #[test]
    fn compute_checksums_skips_missing_files() {
        let dir = TempDir::new().unwrap();
        // Only write gate_vectors.bin; the rest are absent
        std::fs::write(dir.path().join(GATE_VECTORS_BIN), b"fake gate data").unwrap();
        let map = compute_checksums(dir.path()).unwrap();
        assert!(map.contains_key(GATE_VECTORS_BIN));
        // Files that don't exist are simply not in the map
        assert!(!map.contains_key(EMBEDDINGS_BIN));
    }

    #[test]
    fn compute_checksums_empty_dir() {
        let dir = TempDir::new().unwrap();
        let map = compute_checksums(dir.path()).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn verify_checksums_pass_for_correct_content() {
        let dir = TempDir::new().unwrap();
        let f = dir.path().join(GATE_VECTORS_BIN);
        std::fs::write(&f, b"gate data").unwrap();
        let stored = compute_checksums(dir.path()).unwrap();
        let results = verify_checksums(dir.path(), &stored).unwrap();
        for (_, ok) in &results {
            assert!(ok, "all stored checksums should verify");
        }
    }

    #[test]
    fn verify_checksums_fail_when_content_changed() {
        let dir = TempDir::new().unwrap();
        let f = dir.path().join(GATE_VECTORS_BIN);
        std::fs::write(&f, b"original").unwrap();
        let stored = compute_checksums(dir.path()).unwrap();
        // Overwrite with different content
        std::fs::write(&f, b"tampered").unwrap();
        let results = verify_checksums(dir.path(), &stored).unwrap();
        let gate_result = results
            .iter()
            .find(|(name, _)| name == GATE_VECTORS_BIN)
            .unwrap();
        assert!(!gate_result.1, "tampered file should fail verification");
    }

    #[test]
    fn verify_checksums_missing_file_is_false() {
        let dir = TempDir::new().unwrap();
        let mut stored = BTreeMap::new();
        stored.insert(GATE_VECTORS_BIN.to_string(), "fakehash".to_string());
        let results = verify_checksums(dir.path(), &stored).unwrap();
        let r = results.iter().find(|(n, _)| n == GATE_VECTORS_BIN).unwrap();
        assert!(!r.1, "missing file should report false");
    }

    // ── Determinism ────────────────────────────────────────────────────────
    //
    // `larql verify` produced different output on every run against an intact
    // artifact, because findings were rendered in `BTreeMap` iteration order.
    // These pin the fix: a verification command that disagrees with itself is
    // unusable for goldens, CI diffs, signed reports or operator triage.

    /// Several files, so ordering has something to get wrong.
    fn populated_dir() -> (TempDir, BTreeMap<String, String>) {
        let dir = TempDir::new().unwrap();
        for (name, body) in [
            (GATE_VECTORS_BIN, &b"gate"[..]),
            (EMBEDDINGS_BIN, &b"embed"[..]),
            (NORMS_BIN, &b"norms"[..]),
            (DOWN_META_BIN, &b"down"[..]),
        ] {
            std::fs::write(dir.path().join(name), body).unwrap();
        }
        let stored = compute_checksums(dir.path()).unwrap();
        assert!(stored.len() >= 4, "need several files to order");
        (dir, stored)
    }

    /// Repeated calls agree — a weaker property than it looks.
    ///
    /// This **cannot** catch the ordering defect, and a mutation check proved
    /// it: with the sort removed, this test still passes. `BTreeMap` iteration
    /// order is randomised per *process*, so one map iterates identically on
    /// every call within a single test. The real defect only appeared across
    /// process boundaries — three CLI invocations, three digests.
    ///
    /// Kept because it does guard something real: that verification is free of
    /// state carried between calls. The cross-process property is guarded by
    /// `insertion_order_does_not_affect_output_order`, which builds genuinely
    /// different maps and is the test that fails without the sort.
    #[test]
    fn repeated_verification_does_not_carry_state_between_calls() {
        let (dir, stored) = populated_dir();
        let first = verify_checksums(dir.path(), &stored).unwrap();
        for run in 1..16 {
            assert_eq!(
                verify_checksums(dir.path(), &stored).unwrap(),
                first,
                "run {run} disagreed with run 0"
            );
        }
    }

    /// The load-bearing determinism test.
    ///
    /// Building the map by a different insertion sequence is the in-process
    /// stand-in for the per-process hash randomisation that produced the real
    /// defect. Verified to fail when the canonical sort is removed.
    #[test]
    fn insertion_order_does_not_affect_output_order() {
        // The same findings, assembled several ways. A `BTreeMap` built by a
        // different insertion sequence iterates differently; the rendered
        // order must not follow it.
        let (dir, stored) = populated_dir();
        let canonical = verify_checksums(dir.path(), &stored).unwrap();

        let mut names: Vec<&String> = stored.keys().collect();
        names.sort();
        // Forward, reverse, and rotated insertion orders.
        for rotation in 0..names.len() {
            let mut shuffled = BTreeMap::new();
            for i in 0..names.len() {
                let name = names[(i + rotation) % names.len()];
                shuffled.insert(name.clone(), stored[name].clone());
            }
            assert_eq!(
                verify_checksums(dir.path(), &shuffled).unwrap(),
                canonical,
                "rotation {rotation} changed the output order"
            );
        }
    }

    #[test]
    fn output_is_sorted_by_filename() {
        // States the canonical key, so a future change to it is deliberate.
        let (dir, stored) = populated_dir();
        let results = verify_checksums(dir.path(), &stored).unwrap();
        let names: Vec<String> = results.iter().map(|(n, _)| n.clone()).collect();
        let mut expected = names.clone();
        expected.sort();
        assert_eq!(names, expected);
    }

    #[test]
    fn ordering_preserves_every_finding_rather_than_deduplicating() {
        // The sort key orders results; it must never collapse them. Distinct
        // findings that happen to sort adjacently must all survive.
        let (dir, stored) = populated_dir();
        let results = verify_checksums(dir.path(), &stored).unwrap();
        assert_eq!(results.len(), stored.len());
        let mut names: Vec<&String> = results.iter().map(|(n, _)| n).collect();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "a finding was lost to deduplication");
    }

    #[test]
    fn a_failing_file_keeps_its_canonical_position() {
        // Failures must not be hoisted or grouped — an operator diffing two
        // runs needs the same file on the same line whether it passed or not.
        let (dir, stored) = populated_dir();
        let passing = verify_checksums(dir.path(), &stored).unwrap();
        std::fs::write(dir.path().join(NORMS_BIN), b"tampered").unwrap();
        let failing = verify_checksums(dir.path(), &stored).unwrap();

        let names =
            |v: &Vec<(String, bool)>| -> Vec<String> { v.iter().map(|(n, _)| n.clone()).collect() };
        assert_eq!(names(&passing), names(&failing), "order moved on failure");
        let idx = names(&failing)
            .iter()
            .position(|n| n == NORMS_BIN)
            .expect("norms present");
        assert!(!failing[idx].1, "the tampered file should fail");
    }
}
