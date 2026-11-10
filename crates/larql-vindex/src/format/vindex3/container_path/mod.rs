//! The one rule for every filename a VINDEX3 container names.
//!
//! `index.json` is the container's sole root authority (spec §12): it
//! names segments, manifests, sidecar tables and segment keys, and every
//! reader composes a filesystem path from those names. A container is
//! also something people download, so those names are untrusted input.
//! `Path::join` replaces the base outright when handed an absolute path,
//! and walks out of it on `..`; a name that does either turns "open the
//! container's segment" into "open, create or overwrite any file the
//! user can reach" — the remote hydrator and `compile` both *write* to
//! paths composed this way.
//!
//! The rule is lexical: a container name is a sequence of ordinary
//! components. No root, no drive prefix, no `.` or `..`, no backslash (a
//! separator on Windows, so the same index would mean different files on
//! different hosts), no NUL. It is enforced where the
//! name enters — [`serde_rules`] on the index's own fields — so no
//! reader can compose a path from an unchecked name, and [`check`] for
//! the few places that read a name out of raw JSON.
//!
//! An EMPTY name is not this rule's business: `root.join("")` is the root
//! itself, which escapes nothing. Whether a field may be empty is a
//! completeness question, and the layers that ask it (source identity,
//! inspection, candidate authority) already refuse an empty name naming
//! the field and the entry — a better refusal than a parse error at a
//! line and column.
//!
//! **What this guarantees, and what it does not.** [`check`] and
//! [`resolve`] guarantee LEXICAL containment: the name cannot climb out
//! of the root. They do not follow symlinks, and do not guarantee that
//! the resolved file is inside the root. That is deliberate for readers:
//! the HuggingFace cache lays every snapshot out as symlinks into
//! `blobs/`, so a resolved-containment rule would refuse every `hf://`
//! container. The trust assumption for reading is that the directory a
//! container sits in — its symlinks included — is the user's filesystem,
//! not the artifact's word; neither download path creates a symlink from
//! artifact data.
//!
//! Writers get more. [`prepare_write`] is the one route from a name to a
//! file LARQL creates: on top of the lexical rule it creates missing
//! parents one level at a time, refuses an existing ancestor that is a
//! symlink resolving outside the root, and refuses a destination that is
//! itself a symlink (which a write would follow). None of the roots
//! LARQL writes into (an encode or compile output, a hydration staging
//! root) is an HF snapshot, so this costs no compatibility. It is a
//! check, not a lock: a process that can rewrite the destination
//! directory between the check and the write is already trusted with
//! that directory.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use crate::error::VindexError;

/// Why a name was refused. One per rule, so a refusal says which rule.
const NUL: &str = "it contains a NUL byte";
const BACKSLASH: &str = "it contains `\\`, a path separator on some hosts";
const NOT_RELATIVE: &str = "it is absolute, or has a `.` or `..` component";
const RESOLVES_OUTSIDE: &str = "a directory on its way is a symlink that leaves the root";
const DESTINATION_IS_SYMLINK: &str = "the file it names is a symlink a write would follow";

/// The reason `name` is not a container-relative path, if it is not one.
fn refusal(name: &str) -> Option<&'static str> {
    if name.contains('\0') {
        Some(NUL)
    } else if name.contains('\\') {
        Some(BACKSLASH)
    } else if Path::new(name)
        .components()
        .any(|c| !matches!(c, Component::Normal(_)))
    {
        Some(NOT_RELATIVE)
    } else {
        None
    }
}

/// Refuse a name that could resolve outside the container root.
pub fn check(name: &str) -> Result<(), VindexError> {
    match refusal(name) {
        None => Ok(()),
        Some(reason) => Err(VindexError::UnsafeContainerPath {
            name: name.to_string(),
            reason,
        }),
    }
}

/// `root.join(name)`, only when `name` stays inside `root`.
pub fn resolve(root: &Path, name: &str) -> Result<PathBuf, VindexError> {
    check(name)?;
    Ok(root.join(name))
}

/// The path to write `name` at under `root`, with its parents created
/// and no symlink between `root` and the file leading outside `root`.
///
/// `root` itself must exist. See the module notes for what this does and
/// does not promise.
///
/// The returned path is a snapshot, not a capability: it is safe at the
/// moment of the call, and only for as long as nothing else rewrites the
/// directories on its way. Call it immediately before the create, copy,
/// link or rename it guards, and never hold the result across other work.
/// A staging file derived from it (`with_extension`) sits in the directory
/// this call checked, but must itself be removed, not written through, if
/// it already exists.
pub fn prepare_write(root: &Path, name: &str) -> Result<PathBuf, VindexError> {
    check(name)?;
    let refuse = |reason| VindexError::UnsafeContainerPath {
        name: name.to_string(),
        reason,
    };
    let root = root.canonicalize()?;
    let relative = Path::new(name);
    let mut dir = root.clone();
    for part in relative.parent().into_iter().flat_map(Path::components) {
        dir.push(part);
        match std::fs::symlink_metadata(&dir) {
            Ok(meta) if meta.file_type().is_symlink() => {
                if !dir.canonicalize()?.starts_with(&root) {
                    return Err(refuse(RESOLVES_OUTSIDE));
                }
            }
            Ok(_) => {}
            // `create_dir`, not `create_dir_all`: each level is checked
            // before the next is created through it.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => std::fs::create_dir(&dir)?,
            Err(e) => return Err(e.into()),
        }
    }
    let path = root.join(relative);
    if std::fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink()) {
        return Err(refuse(DESTINATION_IS_SYMLINK));
    }
    Ok(path)
}

/// `deserialize_with` adapters: the index refuses an unsafe name while
/// it is being parsed, so every caller of `Vindex3Index` inherits the
/// rule without knowing it exists.
pub(crate) mod serde_rules {
    use super::*;
    use serde::{de::Error, Deserialize, Deserializer};

    fn checked<E: Error>(name: String) -> Result<String, E> {
        check(&name).map_err(E::custom)?;
        Ok(name)
    }

    pub fn relative<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
        checked(String::deserialize(d)?)
    }

    pub fn optional_relative<'de, D: Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
        Option::<String>::deserialize(d)?.map(checked).transpose()
    }

    pub fn relative_keys<'de, D, V>(d: D) -> Result<BTreeMap<String, V>, D::Error>
    where
        D: Deserializer<'de>,
        V: Deserialize<'de>,
    {
        let map = BTreeMap::<String, V>::deserialize(d)?;
        for key in map.keys() {
            check(key).map_err(D::Error::custom)?;
        }
        Ok(map)
    }
}

#[cfg(test)]
mod tests;
