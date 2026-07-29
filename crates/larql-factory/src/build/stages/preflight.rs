//! PREFLIGHT — check the scratch directory is usable before spending
//! any time on FETCH. Deliberately doesn't check *available* disk
//! space against `budget.requires.disk_gib`: nothing in Rust's std
//! exposes free-space querying cross-platform, and this workspace has
//! no dependency that does either (confirmed — no `fs2`/`sysinfo` in
//! `Cargo.lock`). This catches the most common real failure (bad
//! permissions, a missing parent path) rather than a partial,
//! platform-conditional space check.

use std::path::Path;

const PREFLIGHT_PROBE_FILE: &str = ".larql-build-preflight-probe";

/// Verify `scratch_dir` exists (creating it if needed) and is writable.
pub fn check_scratch_dir_writable(scratch_dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(scratch_dir)
        .map_err(|e| format!("creating scratch dir {}: {e}", scratch_dir.display()))?;
    let probe = scratch_dir.join(PREFLIGHT_PROBE_FILE);
    std::fs::write(&probe, b"preflight")
        .map_err(|e| format!("scratch dir {} is not writable: {e}", scratch_dir.display()))?;
    std::fs::remove_file(&probe).ok();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_and_accepts_a_fresh_nested_directory() {
        let dir = tempfile::tempdir().unwrap();
        let scratch = dir.path().join("nested/scratch");
        check_scratch_dir_writable(&scratch).unwrap();
        assert!(scratch.is_dir());
    }

    #[test]
    fn accepts_an_already_existing_directory() {
        let dir = tempfile::tempdir().unwrap();
        check_scratch_dir_writable(dir.path()).unwrap();
        check_scratch_dir_writable(dir.path()).unwrap();
    }

    #[test]
    fn probe_file_is_cleaned_up() {
        let dir = tempfile::tempdir().unwrap();
        check_scratch_dir_writable(dir.path()).unwrap();
        assert!(!dir.path().join(PREFLIGHT_PROBE_FILE).exists());
    }

    #[test]
    #[cfg(unix)]
    fn read_only_directory_is_rejected() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let scratch = dir.path().join("readonly");
        std::fs::create_dir(&scratch).unwrap();
        std::fs::set_permissions(&scratch, std::fs::Permissions::from_mode(0o500)).unwrap();

        let result = check_scratch_dir_writable(&scratch);

        // Restore permissions so the tempdir's own Drop cleanup can delete it.
        std::fs::set_permissions(&scratch, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert!(result.is_err());
    }
}
