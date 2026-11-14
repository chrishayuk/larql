//! The production VINDEX3 registry's actual data — `registry/index.json` +
//! `registry/models/*.json` at the repo root, baked into the binary at
//! compile time.
//!
//! # Why compile-time embed, not a runtime file read
//!
//! `docs/vindex3-registry-design.md` §7 left this open ("embedded? a
//! file? fetched?"). A runtime read relative to the process's working
//! directory or executable path only works from a source checkout —
//! release binaries (ADR-0026) ship standalone, with no repo nearby. A
//! remote fetch would make every `larql pull <registry-name>` depend on
//! a second service beyond HF, which §7 already ruled out ("no website,
//! no remote registry service"). `include_str!` matches the precedent
//! [`super::fixtures`] already set — "static, in-process, no network" —
//! while still sourcing from real, git-reviewed JSON files rather than a
//! Rust literal, so a registry change is an ordinary reviewable diff.
//!
//! The real consequence, accepted deliberately: a new or updated entry
//! reaches users only via a new binary release, not instantly on merge.
//! That is not a regression — "official status conferred by PR merge"
//! (the promotion design, `docs/vindex3-registry-publishing-design.md`)
//! was already going to require a release to actually ship the entry;
//! this just means the release, not the merge, is the activation point.
//!
//! # Why one embed per model, not a glob
//!
//! `include_str!` needs a compile-time literal path — it cannot walk
//! `registry/models/` at build time without a build script. R3A is
//! deliberately one entry (`docs/vindex3-registry-design.md`'s own
//! scoping note: prove the representation works before generalising
//! it). [`embedded_model_json`] is the one place a second entry's
//! `include_str!` line joins this match — a small, explicit list, not
//! speculative glob machinery for a count that has been one.

use std::collections::BTreeMap;

use serde::Deserialize;

use super::error::RegistryError;
use super::manifest::{RegistryManifest, RegistryModel};

/// `registry/index.json` — the table of contents: which models exist,
/// under which manifest schema.
const INDEX_JSON: &str = include_str!("../../../../registry/index.json");

/// `registry/models/granite-4.1-3b.json` — see [`embedded_model_json`].
const GRANITE_4_1_3B_JSON: &str = include_str!("../../../../registry/models/granite-4.1-3b.json");

/// `registry/models/gemma-3-4b-it.json` — see [`embedded_model_json`].
const GEMMA_3_4B_IT_JSON: &str = include_str!("../../../../registry/models/gemma-3-4b-it.json");

/// `registry/models/gemma-3-12b-it.json` — see [`embedded_model_json`].
const GEMMA_3_12B_IT_JSON: &str = include_str!("../../../../registry/models/gemma-3-12b-it.json");

/// `registry/models/gemma-4-12b-it.json` — see [`embedded_model_json`].
const GEMMA_4_12B_IT_JSON: &str = include_str!("../../../../registry/models/gemma-4-12b-it.json");

/// `registry/models/gemma-4-31b-it.json` — see [`embedded_model_json`].
const GEMMA_4_31B_IT_JSON: &str = include_str!("../../../../registry/models/gemma-4-31b-it.json");

/// `registry/index.json`'s own shape: schema version plus the list of
/// model names this registry claims. Not [`RegistryManifest`] itself —
/// that type's `models` map holds full bodies; the index only names
/// them, mirroring the two-file split on disk.
#[derive(Debug, Deserialize)]
pub(super) struct RegistryIndex {
    pub(super) schema_version: u32,
    pub(super) models: Vec<String>,
}

/// Parse `registry/index.json`'s text alone — shared by [`assemble_registry`]
/// and [`super::check::load_registry_from_dir`], which both need the
/// model-name list before they can even know what to read next
/// (the embedded per-model consts, or `models/<name>.json` on disk).
pub(super) fn parse_index(index_json: &str) -> Result<RegistryIndex, RegistryError> {
    serde_json::from_str(index_json).map_err(|e| RegistryError::MalformedManifest {
        reason: format!("registry/index.json: {e}"),
    })
}

/// The embedded JSON text for one `registry/models/<name>.json`, or
/// `None` if `name` isn't one this binary was built with. Distinct from
/// "not a registered model" (that's [`RegistryError::UnknownModel`],
/// raised once resolution runs) — this is "index.json names a model
/// file this binary has no `include_str!` for", a build-time skew
/// between the two files that a real registry entry should never reach
/// (R3B's CI conformance gate is the intended place to catch it before
/// merge); [`assemble_registry`] still refuses it cleanly rather than
/// panicking here, since a corrupted or hand-edited index is exactly
/// the case this should fail loudly, not silently, on.
fn embedded_model_json(name: &str) -> Option<&'static str> {
    match name {
        "granite-4.1-3b" => Some(GRANITE_4_1_3B_JSON),
        "gemma-3-4b-it" => Some(GEMMA_3_4B_IT_JSON),
        "gemma-3-12b-it" => Some(GEMMA_3_12B_IT_JSON),
        "gemma-4-12b-it" => Some(GEMMA_4_12B_IT_JSON),
        "gemma-4-31b-it" => Some(GEMMA_4_31B_IT_JSON),
        _ => None,
    }
}

/// Parse, assemble, and validate the checked-in `registry/` files into a
/// [`RegistryManifest`]. The one function [`super::production::production_registry`]
/// trusts to have already done all of this by the time a caller sees a
/// plain, infallible `RegistryManifest`.
///
/// Also the non-panicking half of `larql registry check` (R3B) with no
/// path argument: validates exactly what THIS binary was built with —
/// for a CI build, that's the PR's own `registry/` files, since
/// `include_str!` embeds whatever was checked out at compile time. No
/// separate "is the registry valid" definition to drift from
/// [`super::production::production_registry`]'s own.
pub fn load_production_registry() -> Result<RegistryManifest, RegistryError> {
    assemble_registry(INDEX_JSON, embedded_model_json)
}

/// Testable core of [`load_production_registry`]. `lookup` is injected
/// (same convention as [`super::production::resolve_claimed_with`]) so
/// every branch — malformed index, an index/embed name mismatch,
/// malformed per-model JSON, a manifest that parses but fails
/// `validate()` — is provable without depending on the real,
/// compile-time-fixed `registry/` contents.
pub(super) fn assemble_registry<'a>(
    index_json: &str,
    lookup: impl Fn(&str) -> Option<&'a str>,
) -> Result<RegistryManifest, RegistryError> {
    let index = parse_index(index_json)?;

    let mut models = BTreeMap::new();
    for name in index.models {
        let json = lookup(&name).ok_or_else(|| RegistryError::MalformedManifest {
            reason: format!(
                "registry/index.json names model '{name}', which has no matching \
                 registry/models/{name}.json embedded in this binary"
            ),
        })?;
        let model: RegistryModel =
            serde_json::from_str(json).map_err(|e| RegistryError::MalformedManifest {
                reason: format!("registry/models/{name}.json: {e}"),
            })?;
        models.insert(name, model);
    }

    let manifest = RegistryManifest {
        schema_version: index.schema_version,
        models,
    };
    manifest.validate()?;
    Ok(manifest)
}
