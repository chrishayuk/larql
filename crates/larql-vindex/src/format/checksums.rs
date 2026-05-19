//! Checksum utilities for vindex file integrity verification.

use std::collections::HashMap;
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
pub fn compute_checksums(dir: &Path) -> Result<HashMap<String, String>, VindexError> {
    let mut checksums = HashMap::new();

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
/// Returns a list of (filename, status) pairs.
pub fn verify_checksums(
    dir: &Path,
    stored: &HashMap<String, String>,
) -> Result<Vec<(String, bool)>, VindexError> {
    let mut results = Vec::new();

    for (filename, expected) in stored {
        let path = dir.join(filename);
        if path.exists() {
            let actual = sha256_file(&path)?;
            results.push((filename.clone(), actual == *expected));
        } else {
            results.push((filename.clone(), false));
        }
    }

    Ok(results)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::collections::HashMap;
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
        let mut stored = HashMap::new();
        stored.insert(GATE_VECTORS_BIN.to_string(), "fakehash".to_string());
        let results = verify_checksums(dir.path(), &stored).unwrap();
        let r = results.iter().find(|(n, _)| n == GATE_VECTORS_BIN).unwrap();
        assert!(!r.1, "missing file should report false");
    }
}
