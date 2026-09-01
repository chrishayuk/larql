//! A [`TensorSource`] backed by byte ranges of a HuggingFace repo.
//!
//! # Where the offsets come from
//!
//! Not from the network. Admission has already staged a header-only
//! checkpoint ([`crate::format::huggingface::metadata_checkpoint`]) —
//! every shard's safetensors header, verbatim, because inventory and plan
//! needed them before anything decided to encode. A safetensors header
//! states every payload's offset and length, so the stub *is* the payload
//! manifest, and this source indexes it with the same
//! [`super::index_shard_header`] the local source uses.
//!
//! What crosses the network is therefore exactly the spans the operation
//! plan asked for, and nothing else — no shard is downloaded to reach a
//! tensor inside it, and a span larger than one chunk is fetched in
//! bounded pieces rather than buffered whole.
//!
//! # The ledger is the claim
//!
//! "You never need the canonical checkpoint on disk" is an assertion about
//! bytes moved, and an assertion about bytes moved that is only ever
//! checked by watching a progress bar regresses quietly. [`Self::fetched`]
//! counts what actually came over the wire, so a caller can state the
//! number rather than the intention.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::{index_shard_header, locate_in, read_shard_header, PayloadLocation, TensorSource};
use crate::error::VindexError;
use crate::format::huggingface::metadata_checkpoint::StagedCheckpoint;
use crate::format::huggingface::range::HfRangeClient;

/// Payload locations for a repo, resolved from its staged headers.
pub struct RemoteArtifactSource {
    /// Keyed by tensor name; each location's `shard` is repo-relative.
    locations: BTreeMap<String, PayloadLocation>,
    client: HfRangeClient,
    fetched: AtomicU64,
    tensors: AtomicU64,
}

impl RemoteArtifactSource {
    /// Index a staged checkpoint's headers and bind them to `client`.
    ///
    /// `client` should be pinned to the commit `staged` recorded, not to
    /// the branch it was named by: the headers were read at one revision
    /// and the payloads must be read at the same one, or the offsets
    /// address a different checkpoint.
    pub fn open(client: HfRangeClient, staged: &StagedCheckpoint) -> Result<Self, VindexError> {
        let locations = index_staged_shards(staged)?;
        if locations.is_empty() {
            return Err(VindexError::Parse(format!(
                "hf://{} staged {} shard(s) but no tensors — the headers are empty",
                client.repo(),
                staged.shards.len()
            )));
        }
        Ok(Self {
            locations,
            client,
            fetched: AtomicU64::new(0),
            tensors: AtomicU64::new(0),
        })
    }

    /// Location of one tensor by its full source name.
    pub fn locate(&self, name: &str) -> Result<&PayloadLocation, VindexError> {
        locate_in(&self.locations, name)
    }

    /// Bytes pulled over the network by this source so far.
    pub fn fetched(&self) -> u64 {
        self.fetched.load(Ordering::Relaxed)
    }

    /// Tensors streamed by this source so far.
    pub fn tensors(&self) -> u64 {
        self.tensors.load(Ordering::Relaxed)
    }

    /// Total payload the indexed headers declare — what a whole download
    /// would have cost.
    pub fn declared_bytes(&self) -> u64 {
        self.locations.values().map(|l| l.len).sum()
    }

    /// The shard files this source addresses.
    pub fn shards(&self) -> Vec<PathBuf> {
        let mut shards: Vec<PathBuf> = self
            .locations
            .values()
            .map(|l| l.shard.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        shards.sort();
        shards
    }
}

/// Payload locations for every tensor a staged checkpoint's headers
/// declare, keyed by tensor name, addressed repo-relative.
///
/// Separate from [`RemoteArtifactSource::open`] because the census is
/// wanted before anything decides to encode: "what would this transfer?"
/// is the question `plan` exists to answer cheaply, and answering it from
/// a second walk of the headers would be a second authority.
pub fn index_staged_shards(
    staged: &StagedCheckpoint,
) -> Result<BTreeMap<String, PayloadLocation>, VindexError> {
    let mut locations = BTreeMap::new();
    for shard in &staged.shards {
        let stub = staged.dir.join(shard);
        let (header, payload_base) = read_shard_header(&stub)?;
        index_shard_header(&header, payload_base, Path::new(shard), &mut locations)?;
    }
    Ok(locations)
}

/// Total payload the staged headers declare — what a whole download would
/// have cost, and what a range-read encode will transfer if the plan binds
/// everything.
///
/// This is the authority, not the shard index's `metadata.total_size`.
/// The two can legitimately differ: HF computes `total_size` from
/// *deduplicated parameter storage*, so a checkpoint whose embedding and
/// output head were tied in the source model declares one of them and
/// serialises both. granite-4.2-3b declares 6,805,672,960 bytes and its
/// own headers sum to 7,319,475,200 — short by exactly one 513,802,240-byte
/// tied member. The headers are what the file actually holds.
pub fn staged_payload_bytes(staged: &StagedCheckpoint) -> Result<u64, VindexError> {
    Ok(index_staged_shards(staged)?.values().map(|l| l.len).sum())
}

impl TensorSource for RemoteArtifactSource {
    fn stream_payload(
        &self,
        name: &str,
        write: &mut dyn std::io::Write,
        observe: &mut dyn FnMut(&[u8]),
    ) -> Result<u64, VindexError> {
        let location = self.locate(name)?;
        let shard = location.shard.to_str().ok_or_else(|| {
            VindexError::Parse(format!(
                "shard name `{}` is not valid UTF-8 — cannot form a repo URL",
                location.shard.display()
            ))
        })?;
        let copied = self
            .client
            .stream_range(shard, location.offset, location.len, write, observe)
            .map_err(|e| VindexError::Parse(format!("streaming `{name}`: {e}")))?;
        self.fetched.fetch_add(copied, Ordering::Relaxed);
        self.tensors.fetch_add(1, Ordering::Relaxed);
        Ok(copied)
    }
}

/// Unix-only: the subject is a filename that is not valid UTF-8, which
/// Windows paths cannot represent. The gate is on the module rather than
/// the test, because a `#[cfg(unix)]` test leaves its imports unused
/// everywhere else and `-D warnings` is a build failure there.
#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::format::vindex3::encode::source::TensorSource;

    /// The struct's own contract, which `open` cannot exercise.
    ///
    /// `PayloadLocation::shard` is a `PathBuf` because the local source
    /// needs a filesystem path. A repo-relative shard has to become a URL
    /// segment, and not every `PathBuf` can: today `index_staged_shards`
    /// builds them from `StagedCheckpoint::shards`, which are `String`s,
    /// so the conversion is total *by construction of that one caller*.
    /// That is an invariant of the caller and not of the type, and this
    /// arm is what stops a second caller from silently forming a mangled
    /// URL instead of saying it cannot form one at all.
    #[test]
    fn a_shard_name_that_cannot_be_a_url_is_refused_by_name() {
        use std::os::unix::ffi::OsStrExt;

        // Lone continuation bytes: a valid POSIX filename, not UTF-8.
        let shard = PathBuf::from(std::ffi::OsStr::from_bytes(b"model-\xff\xfe.safetensors"));
        let subject = RemoteArtifactSource {
            locations: [(
                "weight".to_string(),
                PayloadLocation {
                    shard,
                    offset: 0,
                    len: 16,
                },
            )]
            .into_iter()
            .collect(),
            client: HfRangeClient::new("larql-test/fixture", "main").unwrap(),
            fetched: AtomicU64::new(0),
            tensors: AtomicU64::new(0),
        };

        let mut sink = Vec::new();
        let err = subject
            .stream_payload("weight", &mut sink, &mut |_| {})
            .expect_err("a shard name that is not UTF-8 cannot address a repo");
        assert!(
            err.to_string().contains("not valid UTF-8"),
            "the refusal must say why the name cannot be used, got: {err}"
        );
        assert!(sink.is_empty(), "nothing may be written on a refusal");
        assert_eq!(subject.fetched(), 0, "and nothing counted as fetched");
    }
}
