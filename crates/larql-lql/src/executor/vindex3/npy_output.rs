//! What every statement that writes a NumPy array shares: the output's name and the rule that nothing is
//! overwritten, the sidecar beside it, the container identity it records, and cleaning up after a failed write.

use std::path::{Path, PathBuf};

use larql_vindex::format::filenames::INDEX_JSON;
use larql_vindex::format::vindex3::Vindex3Index;
use sha2::{Digest, Sha256};

use crate::error::LqlError;

/// The extension an output must carry.
pub(super) const NPY_SUFFIX: &str = ".npy";
/// Appended to the output's name to name the sidecar.
pub(super) const SIDECAR_SUFFIX: &str = ".json";
/// The LARQL version that wrote an array (the workspace version).
pub(super) const LARQL_VERSION: &str = env!("CARGO_PKG_VERSION");

/// `<statement> refused: <reason>`.
pub(super) fn refused(statement: &str, e: impl std::fmt::Display) -> LqlError {
    LqlError::Execution(format!("{statement} refused: {e}"))
}

/// A string as an LQL literal.
pub(super) fn literal(text: &str) -> String {
    format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\""))
}

/// `a, b, c`.
pub(super) fn index_list<T: ToString>(indices: &[T]) -> String {
    indices
        .iter()
        .map(T::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// The `.npy` path and its sidecar's path, refused unless the output is named `<file>.npy` and neither exists.
pub(super) fn output_paths(statement: &str, output: &str) -> Result<(PathBuf, PathBuf), LqlError> {
    let named_npy = Path::new(output)
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.len() > NPY_SUFFIX.len() && name.ends_with(NPY_SUFFIX));
    if !named_npy {
        return Err(refused(
            statement,
            format!(
                "the output `{output}` is not a {NPY_SUFFIX} file; name it `<file>{NPY_SUFFIX}`"
            ),
        ));
    }
    let npy_path = PathBuf::from(output);
    let sidecar_path = PathBuf::from(format!("{output}{SIDECAR_SUFFIX}"));
    for path in [&npy_path, &sidecar_path] {
        // `symlink_metadata` sees a dangling symlink too: nothing at the path may be replaced.
        if path.symlink_metadata().is_ok() {
            return Err(refused(
                statement,
                format!(
                    "refusing to write `{}`: it already exists, and an export never overwrites",
                    path.display()
                ),
            ));
        }
    }
    Ok((npy_path, sidecar_path))
}

/// The container's index and the SHA-256 of its bytes, as read now.
pub(super) fn container_index(root: &Path) -> Result<(Vindex3Index, String), LqlError> {
    let bytes = std::fs::read(root.join(INDEX_JSON))
        .map_err(|e| LqlError::exec("read container index", e))?;
    let index =
        serde_json::from_slice(&bytes).map_err(|e| LqlError::exec("parse container index", e))?;
    Ok((index, format!("{:x}", Sha256::digest(&bytes))))
}

/// The container's path as the sidecar records it: canonical where it can be resolved.
pub(super) fn container_path(root: &Path) -> String {
    std::fs::canonicalize(root)
        .unwrap_or_else(|_| root.to_path_buf())
        .display()
        .to_string()
}

/// Write the sidecar, which must not exist yet. On failure everything in `written` is removed too.
pub(super) fn write_sidecar(
    sidecar_path: &Path,
    sidecar: &serde_json::Value,
    written: &[&Path],
) -> Result<(), LqlError> {
    let result = serde_json::to_vec_pretty(sidecar)
        .map_err(std::io::Error::other)
        .and_then(|text| {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(sidecar_path)?;
            file.write_all(&text)?;
            file.sync_all()
        });
    result.map_err(|e| {
        let cause = LqlError::exec(format!("write {}", sidecar_path.display()), e);
        let mut paths = written.to_vec();
        paths.push(sidecar_path);
        remove_partial(&paths, cause)
    })
}

/// Remove what a failed write wrote. A file that cannot be removed is reported, never ignored silently.
pub(super) fn remove_partial(paths: &[&Path], cause: LqlError) -> LqlError {
    let leftovers: Vec<String> = paths
        .iter()
        .filter(|p| p.exists() && std::fs::remove_file(p).is_err())
        .map(|p| p.display().to_string())
        .collect();
    if leftovers.is_empty() {
        cause
    } else {
        LqlError::Execution(format!(
            "{cause}; could not remove the partial output {}",
            leftovers.join(", ")
        ))
    }
}
