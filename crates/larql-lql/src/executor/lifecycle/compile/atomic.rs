//! Staged, guarded writes for the legacy `COMPILE INTO {VINDEX, MODEL}`.
//!
//! Both targets produce a directory of files. Writing it in place leaves
//! a half-baked vindex behind a SIGKILL; promoting over whatever is
//! already at the output means deleting it. Neither is this verb's call
//! to make, so the destination check, the staging directory and the
//! promotion all belong to `larql_vindex::format::fresh_output` — the
//! same boundary the VINDEX3 bake and compaction write through:
//!
//!   1. the output is refused before any write if it is the source,
//!      contains it, sits inside it, is a symlink, or already holds
//!      anything (a previously compiled vindex included);
//!   2. all writes land in an exclusive sibling staging directory;
//!   3. on success the staging directory is renamed to the output — the
//!      only step that touches the user's destination;
//!   4. on error the staging directory is removed and the destination is
//!      exactly as it was.

use std::path::Path;

use crate::error::LqlError;
use larql_vindex::format::fresh_output::write_fresh_directory;

/// Run `work(staging_dir)` and promote the staging directory to
/// `final_dir` on success.
///
/// `source` is the directory the compile reads from.
pub(super) fn run_atomic_compile<F>(
    final_dir: &Path,
    source: &Path,
    work: F,
) -> Result<Vec<String>, LqlError>
where
    F: FnOnce(&Path) -> Result<Vec<String>, LqlError>,
{
    write_fresh_directory(source, final_dir, work)
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_vindex::format::fresh_output::STAGING_INFIX;

    const STALE_FILE: &str = "stale.bin";
    const STALE_BYTES: &[u8] = b"old";

    /// `<tmp>/source` and a not-yet-created `<tmp>/out`.
    fn dirs() -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source");
        std::fs::create_dir_all(&source).unwrap();
        let out = tmp.path().join("out");
        (tmp, source, out)
    }

    fn staging_leftovers(parent: &Path) -> usize {
        std::fs::read_dir(parent)
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .contains(STAGING_INFIX)
            })
            .count()
    }

    #[test]
    fn promotes_staging_on_success() {
        let (tmp, source, out) = dirs();

        let result = run_atomic_compile(&out, &source, |staging| {
            assert!(staging.exists(), "work runs against existing staging dir");
            std::fs::write(staging.join("a.bin"), b"hello").unwrap();
            Ok(vec!["wrote a.bin".into()])
        });

        assert_eq!(result.unwrap(), vec!["wrote a.bin".to_string()]);
        assert!(out.join("a.bin").exists(), "file promoted");
        assert_eq!(staging_leftovers(tmp.path()), 0, "staging removed");
    }

    #[test]
    fn cleans_up_staging_and_propagates_the_work_error() {
        let (tmp, source, out) = dirs();

        let err = run_atomic_compile(&out, &source, |staging| {
            std::fs::write(staging.join("partial.bin"), b"oops").unwrap();
            Err::<Vec<String>, _>(LqlError::Execution(
                "specific failure that callers will recognise".into(),
            ))
        })
        .unwrap_err();

        assert!(err.to_string().contains("specific failure"));
        assert_eq!(staging_leftovers(tmp.path()), 0, "staging removed");
        assert!(!out.exists(), "no partial output at final destination");
    }

    #[test]
    fn rejects_output_equal_to_source() {
        let (_tmp, source, _out) = dirs();

        let err = run_atomic_compile(&source, &source, |_staging| {
            unreachable!("work must not run for a refused destination")
        })
        .unwrap_err();

        assert!(
            err.to_string().contains("resolves to the source vindex"),
            "unexpected error: {err}"
        );
        assert!(source.exists(), "source untouched");
    }

    #[test]
    fn refuses_an_existing_non_empty_final_dir_and_keeps_its_contents() {
        // Replacing an existing directory used to be silent: the old
        // contents were recursively deleted on promotion.
        let (_tmp, source, out) = dirs();
        std::fs::create_dir_all(&out).unwrap();
        std::fs::write(out.join(STALE_FILE), STALE_BYTES).unwrap();

        let err = run_atomic_compile(&out, &source, |_staging| {
            unreachable!("work must not run for a refused destination")
        })
        .unwrap_err();

        assert!(
            err.to_string().contains("refusing to write"),
            "unexpected error: {err}"
        );
        assert_eq!(std::fs::read(out.join(STALE_FILE)).unwrap(), STALE_BYTES);
    }

    #[test]
    fn errors_when_the_output_parent_is_a_file() {
        let (tmp, source, _out) = dirs();
        let parent_as_file = tmp.path().join("plain-file");
        std::fs::write(&parent_as_file, b"definitely not a directory").unwrap();

        let result = run_atomic_compile(&parent_as_file.join("never_created"), &source, |_| {
            unreachable!("work must not run when the destination is unusable")
        });

        assert!(result.is_err());
        assert!(parent_as_file.is_file(), "the file is untouched");
    }
}
