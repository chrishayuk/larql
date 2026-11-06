//! The production wiring: the hub as the probe, the user's caches as the
//! store, stderr as the status stream.
//!
//! Thin by design — every decision lives in `resolve.rs`, `classify.rs`,
//! `store.rs` and `encode.rs`, which take these collaborators as
//! parameters and are tested without a network.

use larql_vindex::format::huggingface::metadata_checkpoint::resolve_commit;
use larql_vindex::format::huggingface::range::HfRangeClient;

use super::super::cache;
use super::classify::RepoProbe;
use super::encode::PlanGatedEncoder;
use super::resolve::{HfAuto, Resolution, VindexFetch};
use super::store::EncodedStore;

type BoxErr = Box<dyn std::error::Error>;

/// Asks the hub through the same range client header staging uses.
pub struct HubProbe;

impl RepoProbe for HubProbe {
    fn has_file(&self, repo: &str, revision: &str, file: &str) -> Result<bool, BoxErr> {
        Ok(HfRangeClient::new(repo, revision)?.fetch(file)?.is_some())
    }

    fn commit(&self, repo: &str, revision: &str) -> Result<Option<String>, BoxErr> {
        Ok(resolve_commit(&HfRangeClient::new(repo, revision)?)?)
    }
}

/// Resolve `spec` against the user's caches and the hub, encoding a raw
/// checkpoint on first use. `fetch_vindex` handles published vindexes.
pub fn resolve_hf_model(
    spec: &str,
    fetch_vindex: &mut VindexFetch<'_>,
) -> Result<Resolution, BoxErr> {
    let store = EncodedStore::open_default()?;
    let hub = cache::hf_hub_dir()?;
    HfAuto {
        store: &store,
        hub: &hub,
        probe: &HubProbe,
        encoder: &PlanGatedEncoder::for_run(),
    }
    .resolve(spec, fetch_vindex, &mut std::io::stderr())
}
