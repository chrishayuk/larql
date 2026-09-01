//! The remote container, its allow-list and its seal.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::error::VindexError;
use crate::format::filenames::INDEX_JSON;
use crate::format::huggingface::range::HfRangeClient;
use crate::format::vindex3::encode::{SEGMENTS_DIR, SEGMENT_BIN_EXT, SYSTEM_GRAPH_JSON};

/// Upper bound on a segment header read, mirroring the checkpoint
/// staging bound. A header claiming more than this did not come from a
/// segment.
const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;

/// Length of a segment's length prefix.
const LENGTH_PREFIX_BYTES: u64 = 8;

/// Subdirectory holding the planning view: index, graph and header-only
/// segment stubs.
const HEADERS_DIR: &str = "headers";

/// Subdirectory holding the execution view: index, graph and the fully
/// hydrated segments of the execution set, and nothing else.
const CONTAINER_DIR: &str = "container";

/// Whether the network is still open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkPhase {
    /// Reads permitted, subject to the allow-list.
    Hydrating,
    /// No read may occur, for any reason.
    Sealed,
}

/// What one hydration moved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HydrationReport {
    /// Objects whose payload was fetched.
    pub hydrated: BTreeSet<String>,
    /// Objects the container describes that were deliberately left
    /// remote — the evidence that hydration was selective at all.
    pub left_remote: BTreeSet<String>,
    /// Bytes fetched during the payload phase.
    pub payload_bytes: u64,
    /// Bytes fetched during the metadata phase (index, graph, headers).
    pub metadata_bytes: u64,
}

/// A remote VINDEX3 container being hydrated into a local root.
pub struct RemoteContainer {
    client: HfRangeClient,
    root: PathBuf,
    /// Objects hydration is permitted to fetch. Empty until
    /// [`Self::allow`] — deny by default.
    allowed: BTreeSet<String>,
    /// Every object the container describes, from its index.
    described: BTreeSet<String>,
    phase: Mutex<NetworkPhase>,
    metadata_bytes: AtomicU64,
    payload_bytes: AtomicU64,
    /// Every read this container refused, and why — evidence rather than
    /// a bare failure.
    refusals: Mutex<Vec<String>>,
}

impl RemoteContainer {
    /// Fetch the container's description and every segment header into
    /// `root/headers`, so a plan can be built without any payload.
    pub fn open(client: HfRangeClient, root: &Path) -> Result<Self, VindexError> {
        let headers = root.join(HEADERS_DIR);
        std::fs::create_dir_all(headers.join(SEGMENTS_DIR))?;
        std::fs::create_dir_all(root.join(CONTAINER_DIR).join(SEGMENTS_DIR))?;

        let mut container = Self {
            client,
            root: root.to_path_buf(),
            allowed: BTreeSet::new(),
            described: BTreeSet::new(),
            phase: Mutex::new(NetworkPhase::Hydrating),
            metadata_bytes: AtomicU64::new(0),
            payload_bytes: AtomicU64::new(0),
            refusals: Mutex::new(Vec::new()),
        };

        // The description, into BOTH views: planning reads it from the
        // header root, execution from the container root, and they must
        // be the same bytes.
        for name in [INDEX_JSON, SYSTEM_GRAPH_JSON] {
            let body = container.fetch_metadata(name)?;
            std::fs::write(headers.join(name), &body)?;
            std::fs::write(root.join(CONTAINER_DIR).join(name), &body)?;
        }

        let index: serde_json::Value =
            serde_json::from_slice(&std::fs::read(headers.join(INDEX_JSON))?)
                .map_err(|e| VindexError::Parse(format!("remote index.json: {e}")))?;
        for segment in segments_of(&index)? {
            container.described.insert(segment.object.clone());
            container.fetch_header(&segment)?;
        }
        if container.described.is_empty() {
            return Err(VindexError::Parse(
                "the remote index describes no objects".to_string(),
            ));
        }
        Ok(container)
    }

    /// The planning view: index, graph and header-only segment stubs.
    ///
    /// Never open an operand store over this — its segments carry tensor
    /// tables and no payload.
    pub fn headers_root(&self) -> PathBuf {
        self.root.join(HEADERS_DIR)
    }

    /// The execution view: index, graph and the hydrated segments.
    pub fn container_root(&self) -> PathBuf {
        self.root.join(CONTAINER_DIR)
    }

    /// Objects the remote container describes.
    pub fn described_objects(&self) -> &BTreeSet<String> {
        &self.described
    }

    /// Permit hydration of exactly these objects.
    ///
    /// Refuses an object the container does not describe: a hydration set
    /// naming something that does not exist is a planning bug, and
    /// discovering it here rather than as a 404 keeps the failure
    /// attributable.
    pub fn allow(&mut self, objects: BTreeSet<String>) -> Result<(), VindexError> {
        let unknown: Vec<&String> = objects.difference(&self.described).collect();
        if !unknown.is_empty() {
            return Err(VindexError::Parse(format!(
                "hydration set names {unknown:?}, which this container does not describe"
            )));
        }
        self.allowed = objects;
        Ok(())
    }

    /// Fetch the payload of every allowed object into the execution view.
    pub fn hydrate(&self) -> Result<HydrationReport, VindexError> {
        let index: serde_json::Value =
            serde_json::from_slice(&std::fs::read(self.headers_root().join(INDEX_JSON))?)
                .map_err(|e| VindexError::Parse(format!("staged index.json: {e}")))?;
        let before = self.payload_bytes.load(Ordering::Relaxed);
        for segment in segments_of(&index)? {
            if !self.allowed.contains(&segment.object) {
                continue;
            }
            let body = self.fetch_payload(&segment)?;
            let dest = self.container_root().join(&segment.path);
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)?;
            }
            // Write through a temporary: a half-written segment is a file
            // that exists, and an existing segment reads as resident.
            let staging = dest.with_extension("incomplete");
            std::fs::write(&staging, &body)?;
            std::fs::rename(&staging, &dest)?;
        }
        Ok(HydrationReport {
            hydrated: self.allowed.clone(),
            left_remote: self.described.difference(&self.allowed).cloned().collect(),
            payload_bytes: self.payload_bytes.load(Ordering::Relaxed) - before,
            metadata_bytes: self.metadata_bytes.load(Ordering::Relaxed),
        })
    }

    /// Close the network. Every subsequent read fails at the point of
    /// violation.
    pub fn seal(&self) {
        *self.phase.lock().unwrap() = NetworkPhase::Sealed;
    }

    pub fn phase(&self) -> NetworkPhase {
        *self.phase.lock().unwrap()
    }

    /// Bytes fetched, by phase.
    pub fn metadata_bytes(&self) -> u64 {
        self.metadata_bytes.load(Ordering::Relaxed)
    }

    pub fn payload_bytes(&self) -> u64 {
        self.payload_bytes.load(Ordering::Relaxed)
    }

    /// Every read this container refused.
    pub fn refusals(&self) -> Vec<String> {
        self.refusals.lock().unwrap().clone()
    }

    /// Refuse unless the network is open, recording why.
    fn admit(&self, what: &str) -> Result<(), VindexError> {
        if self.phase() == NetworkPhase::Sealed {
            let reason = format!(
                "`{what}` was requested after seal — PREPARE and RUN must have no \
                 remote dependency"
            );
            self.refusals.lock().unwrap().push(reason.clone());
            return Err(VindexError::Parse(reason));
        }
        Ok(())
    }

    fn fetch_metadata(&self, name: &str) -> Result<Vec<u8>, VindexError> {
        self.admit(name)?;
        let body = self
            .client
            .fetch(name)?
            .ok_or_else(|| VindexError::Parse(format!("remote container has no `{name}`")))?;
        self.metadata_bytes
            .fetch_add(body.len() as u64, Ordering::Relaxed);
        Ok(body)
    }

    /// One segment's header stub: the length prefix and the tensor table
    /// it announces, and nothing after.
    fn fetch_header(&self, segment: &SegmentRef) -> Result<(), VindexError> {
        self.admit(&segment.path)?;
        let prefix = self
            .client
            .fetch_range(&segment.path, 0, LENGTH_PREFIX_BYTES)?;
        let header_len = u64::from_le_bytes(
            prefix
                .as_slice()
                .try_into()
                .map_err(|_| VindexError::Parse(format!("{}: short prefix", segment.path)))?,
        );
        if header_len > MAX_HEADER_BYTES {
            return Err(VindexError::Parse(format!(
                "{}: segment header claims {header_len} bytes — not a segment",
                segment.path
            )));
        }
        let header = self
            .client
            .fetch_range(&segment.path, LENGTH_PREFIX_BYTES, header_len)?;
        self.metadata_bytes
            .fetch_add(prefix.len() as u64 + header.len() as u64, Ordering::Relaxed);

        let dest = self.headers_root().join(&segment.path);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut bytes = prefix;
        bytes.extend_from_slice(&header);
        std::fs::write(dest, bytes)?;
        Ok(())
    }

    /// One object's whole segment — allow-listed.
    fn fetch_payload(&self, segment: &SegmentRef) -> Result<Vec<u8>, VindexError> {
        if !self.allowed.contains(&segment.object) {
            let reason = format!(
                "`{}` is not in the hydration set — hydration is deny-by-default",
                segment.object
            );
            self.refusals.lock().unwrap().push(reason.clone());
            return Err(VindexError::Parse(reason));
        }
        self.admit(&segment.path)?;
        let body = self.client.fetch(&segment.path)?.ok_or_else(|| {
            VindexError::Parse(format!("remote container has no `{}`", segment.path))
        })?;
        self.payload_bytes
            .fetch_add(body.len() as u64, Ordering::Relaxed);
        Ok(body)
    }
}

/// One object's segment, as the index names it.
struct SegmentRef {
    object: String,
    /// Container-relative path, e.g. `segments/target.embedding.bin`.
    path: String,
}

/// Every distinct segment the index's representations point at.
fn segments_of(index: &serde_json::Value) -> Result<Vec<SegmentRef>, VindexError> {
    let representations = index["representations"].as_object().ok_or_else(|| {
        VindexError::Parse("remote index.json carries no representations".to_string())
    })?;
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for entry in representations.values() {
        let (Some(object), Some(path)) = (entry["object"].as_str(), entry["segment"].as_str())
        else {
            continue;
        };
        if seen.insert(path.to_string()) {
            out.push(SegmentRef {
                object: object.to_string(),
                path: path.to_string(),
            });
        }
    }
    if out.is_empty() {
        return Err(VindexError::Parse(format!(
            "remote index.json names no `{SEGMENTS_DIR}/*.{SEGMENT_BIN_EXT}` segments"
        )));
    }
    Ok(out)
}
