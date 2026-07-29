//! MANIFEST — `index.json` itself is already written by EXTRACT/SLICE
//! (that's larql-vindex's job, not this driver's); this stage just
//! measures what came out, producing the [`crate::SliceSummary`] list
//! the card generator's slice table needs.

use std::path::Path;

use crate::SliceSummary;

/// Sum the byte size of every file under `dir`, recursively. Symlinks
/// aren't followed (a vindex directory doesn't contain any).
fn dir_size_bytes(dir: &Path) -> Result<u64, String> {
    let mut total = 0u64;
    let entries = std::fs::read_dir(dir).map_err(|e| format!("reading {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("reading entry in {}: {e}", dir.display()))?;
        let metadata = entry
            .metadata()
            .map_err(|e| format!("stat {}: {e}", entry.path().display()))?;
        if metadata.is_dir() {
            total += dir_size_bytes(&entry.path())?;
        } else if metadata.is_file() {
            total += metadata.len();
        }
    }
    Ok(total)
}

/// Measure one output directory into a [`SliceSummary`].
pub fn summarise(preset: &str, dir: &Path) -> Result<SliceSummary, String> {
    Ok(SliceSummary {
        preset: preset.to_string(),
        size_bytes: dir_size_bytes(dir)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_files_at_the_top_level() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.bin"), vec![0u8; 100]).unwrap();
        std::fs::write(dir.path().join("b.bin"), vec![0u8; 250]).unwrap();
        assert_eq!(dir_size_bytes(dir.path()).unwrap(), 350);
    }

    #[test]
    fn sums_files_in_nested_subdirectories() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.bin"), vec![0u8; 100]).unwrap();
        let sub = dir.path().join("layers");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("layer_00.bin"), vec![0u8; 50]).unwrap();
        std::fs::write(sub.join("layer_01.bin"), vec![0u8; 50]).unwrap();
        assert_eq!(dir_size_bytes(dir.path()).unwrap(), 200);
    }

    #[test]
    fn empty_directory_sums_to_zero() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(dir_size_bytes(dir.path()).unwrap(), 0);
    }

    #[test]
    fn missing_directory_errors() {
        let err = dir_size_bytes(Path::new("/nonexistent/does/not/exist")).unwrap_err();
        assert!(err.contains("reading"), "{err}");
    }

    #[test]
    fn summarise_wraps_preset_and_size() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.bin"), vec![0u8; 42]).unwrap();
        let summary = summarise("client", dir.path()).unwrap();
        assert_eq!(summary.preset, "client");
        assert_eq!(summary.size_bytes, 42);
    }
}
