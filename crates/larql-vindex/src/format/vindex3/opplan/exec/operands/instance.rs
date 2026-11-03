//! MEM-002 I2b: which opened file a store's weights came from.
//!
//! A store opens each selected segment once and keeps that descriptor for
//! its lifetime. The segment's header, every copied read and every mapping
//! go through it, and its identity is taken from it — so the identity
//! recorded is the identity of the file execution actually read, never of
//! whatever a path names by then.
//!
//! **This detects change. It does not prove content.** A descriptor binds
//! an inode, not its bytes: an in-place write shows through a live mapping,
//! and metadata can be restored. Only hashing the payload, outside the
//! timed experiment and by another tool, speaks to content.
//!
//! **What this does not bind.** The plan, `index.json`, the system graph
//! and every header read before the store exists were read by path, and
//! nothing here shows they came from the same bytes. The store's own header
//! is re-read through the retained descriptor, which says nothing about
//! what the planner read earlier.

use std::fs::File;
use std::path::PathBuf;

use super::OperandStore;

/// Where a volume UUID was read from: the descriptor itself.
pub const VOLUME_UUID_FROM_DESCRIPTOR: &str = "fgetattrlist";
/// Where a volume UUID was read from: the mount the descriptor is on. Used
/// only if the descriptor refuses, as its documentation says it should.
pub const VOLUME_UUID_FROM_MOUNT: &str = "fstatfs_mount_getattrlist";

#[cfg(unix)]
const NANOS_PER_SECOND: i64 = 1_000_000_000;

/// One opened file's metadata. A change detector: equal values do not
/// establish equal content, and some harmless events (an extended attribute,
/// a hard link) change the ctime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileIdentity {
    /// The OS this was captured on, because what can be read differs by OS.
    pub platform: &'static str,
    /// The volume's UUID. `None` where no such reading exists: never on
    /// Linux today, and on macOS only if every source failed.
    pub volume_uuid: Option<String>,
    /// Which source supplied [`Self::volume_uuid`].
    pub volume_uuid_source: Option<&'static str>,
    pub inode: u64,
    pub size: u64,
    pub mtime_ns: i64,
    pub ctime_ns: i64,
    /// Links to this inode. A file replaced by rename drops to 0 on the
    /// descriptor that still holds it.
    pub nlink: u64,
    /// Diagnostic only: a device id can change across a reboot.
    pub device: u64,
}

impl FileIdentity {
    /// The identity of `file`, read from that descriptor alone.
    #[cfg(unix)]
    pub fn capture(file: &File) -> Result<Self, String> {
        use std::os::unix::fs::MetadataExt;
        let metadata = file
            .metadata()
            .map_err(|e| format!("cannot read the descriptor's metadata: {e}"))?;
        let (volume_uuid, volume_uuid_source) = volume_uuid(file);
        Ok(Self {
            platform: std::env::consts::OS,
            volume_uuid,
            volume_uuid_source,
            inode: metadata.ino(),
            size: metadata.size(),
            mtime_ns: nanos(metadata.mtime(), metadata.mtime_nsec())?,
            ctime_ns: nanos(metadata.ctime(), metadata.ctime_nsec())?,
            nlink: metadata.nlink(),
            device: metadata.dev(),
        })
    }

    /// No descriptor identity exists on this platform: a failure, so a
    /// campaign attempt cannot proceed on it.
    #[cfg(not(unix))]
    pub fn capture(_file: &File) -> Result<Self, String> {
        Err(format!(
            "no descriptor identity is available on {}",
            std::env::consts::OS
        ))
    }

    /// The fields that differ from `earlier`. The device id is excluded
    /// because it can differ with no change to the file.
    pub fn changed_fields(&self, earlier: &Self) -> Vec<&'static str> {
        [
            (self.volume_uuid != earlier.volume_uuid, "volume_uuid"),
            (self.inode != earlier.inode, "inode"),
            (self.size != earlier.size, "size"),
            (self.mtime_ns != earlier.mtime_ns, "mtime_ns"),
            (self.ctime_ns != earlier.ctime_ns, "ctime_ns"),
            (self.nlink != earlier.nlink, "nlink"),
        ]
        .into_iter()
        .filter_map(|(changed, name)| changed.then_some(name))
        .collect()
    }
}

#[cfg(unix)]
fn nanos(seconds: i64, nanoseconds: i64) -> Result<i64, String> {
    seconds
        .checked_mul(NANOS_PER_SECOND)
        .and_then(|n| n.checked_add(nanoseconds))
        .ok_or_else(|| format!("timestamp {seconds}.{nanoseconds} does not fit nanoseconds"))
}

#[cfg(target_os = "macos")]
fn volume_uuid(file: &File) -> (Option<String>, Option<&'static str>) {
    if let Some(uuid) = volume_uuid_from_descriptor(file) {
        return (Some(uuid), Some(VOLUME_UUID_FROM_DESCRIPTOR));
    }
    match volume_uuid_from_mount(file) {
        Some(uuid) => (Some(uuid), Some(VOLUME_UUID_FROM_MOUNT)),
        None => (None, None),
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
fn volume_uuid(_file: &File) -> (Option<String>, Option<&'static str>) {
    (None, None)
}

/// What `getattrlist` writes for a volume UUID: its length, then the UUID.
#[cfg(target_os = "macos")]
#[repr(C, packed(4))]
struct VolumeUuidReply {
    length: u32,
    uuid: [u8; 16],
}

#[cfg(target_os = "macos")]
fn volume_uuid_request() -> libc::attrlist {
    // SAFETY: `attrlist` is plain integers, for which zero is valid.
    let mut request: libc::attrlist = unsafe { std::mem::zeroed() };
    request.bitmapcount = libc::ATTR_BIT_MAP_COUNT;
    request.volattr = libc::ATTR_VOL_INFO | libc::ATTR_VOL_UUID;
    request
}

#[cfg(target_os = "macos")]
fn format_volume_uuid(reply: &VolumeUuidReply) -> Option<String> {
    let uuid = reply.uuid;
    if (reply.length as usize) < std::mem::size_of::<VolumeUuidReply>() || uuid == [0u8; 16] {
        return None;
    }
    let hex = |bytes: &[u8]| bytes.iter().map(|b| format!("{b:02X}")).collect::<String>();
    Some(format!(
        "{}-{}-{}-{}-{}",
        hex(&uuid[0..4]),
        hex(&uuid[4..6]),
        hex(&uuid[6..8]),
        hex(&uuid[8..10]),
        hex(&uuid[10..16])
    ))
}

/// The volume UUID as the descriptor reports it. The man page says this
/// needs the volume root and `EINVAL` otherwise; it works on a regular file
/// on the systems tested, which a test pins, and [`volume_uuid_from_mount`]
/// is the fallback.
#[cfg(target_os = "macos")]
pub fn volume_uuid_from_descriptor(file: &File) -> Option<String> {
    use std::os::fd::AsRawFd;
    let mut request = volume_uuid_request();
    let mut reply = VolumeUuidReply {
        length: 0,
        uuid: [0; 16],
    };
    // SAFETY: the request and the reply are valid for the sizes given, and
    // the descriptor is open for the call.
    let status = unsafe {
        libc::fgetattrlist(
            file.as_raw_fd(),
            (&raw mut request).cast(),
            (&raw mut reply).cast(),
            std::mem::size_of::<VolumeUuidReply>(),
            0,
        )
    };
    (status == 0).then(|| format_volume_uuid(&reply)).flatten()
}

/// The volume UUID of the mount this descriptor's file is on: still taken
/// from the descriptor, by way of `fstatfs` on it.
#[cfg(target_os = "macos")]
pub fn volume_uuid_from_mount(file: &File) -> Option<String> {
    use std::os::fd::AsRawFd;
    // SAFETY: `statfs` is plain data, written by `fstatfs` and read only
    // when it returns 0.
    let mut mount: libc::statfs = unsafe { std::mem::zeroed() };
    // SAFETY: `mount` is writable and the descriptor is open for the call.
    if unsafe { libc::fstatfs(file.as_raw_fd(), &mut mount) } != 0 {
        return None;
    }
    let mut request = volume_uuid_request();
    let mut reply = VolumeUuidReply {
        length: 0,
        uuid: [0; 16],
    };
    // SAFETY: `f_mntonname` is NUL-terminated by the kernel; the request
    // and the reply are valid for the sizes given.
    let status = unsafe {
        libc::getattrlist(
            mount.f_mntonname.as_ptr(),
            (&raw mut request).cast(),
            (&raw mut reply).cast(),
            std::mem::size_of::<VolumeUuidReply>(),
            0,
        )
    };
    (status == 0).then(|| format_volume_uuid(&reply)).flatten()
}

/// One selected segment: what the store's retained descriptor said when it
/// was opened and what it says now, and how execution has used it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentInstance {
    pub object: String,
    /// The path the container's index declares, relative to its root.
    pub declared: String,
    /// The SHA-256 the container's index declares for this segment: what a
    /// manifest's verified hash is compared with. Read from `index.json` by
    /// path, so it says what the index claims and binds nothing.
    pub index_sha256: String,
    /// Diagnostic only: identity never comes from this path.
    pub path: PathBuf,
    /// Whether a region of this segment was mapped.
    pub mapped: bool,
    /// Whether bytes were copied from this segment.
    pub copied: bool,
    /// The identity captured from the descriptor when it was opened.
    pub at_open: Result<FileIdentity, String>,
    /// The identity read from the same descriptor just now.
    pub now: Result<FileIdentity, String>,
}

impl SegmentInstance {
    /// A capture that failed, at open or now.
    pub fn capture_failure(&self) -> Option<&str> {
        self.at_open
            .as_ref()
            .err()
            .or(self.now.as_ref().err())
            .map(String::as_str)
    }

    /// What differs between the two captures. Empty if either one failed:
    /// that is a failure, not a comparison.
    pub fn changed_fields(&self) -> Vec<&'static str> {
        match (&self.at_open, &self.now) {
            (Ok(at_open), Ok(now)) => now.changed_fields(at_open),
            _ => Vec::new(),
        }
    }
}

/// Every selected segment's identity at one moment, with the counters that
/// would show a later read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreInstance {
    /// In object order. Every segment the store selected, used or not.
    pub segments: Vec<SegmentInstance>,
    /// Objects described by the container whose segment was not on disk.
    pub absent_objects: Vec<String>,
    pub load_count: u64,
    pub bytes_read: u64,
    pub mapped_regions: usize,
    pub mapped_objects: usize,
}

impl OperandStore {
    /// Read every retained descriptor's identity now. One `fstat` per
    /// segment (and one volume query): it touches no payload page. It is
    /// for the points around a run, never between steps.
    pub fn instance_report(&self) -> StoreInstance {
        let mapped = self.mapped.lock().unwrap();
        let segments = self
            .segments
            .iter()
            .map(|(object, segment)| SegmentInstance {
                object: object.clone(),
                declared: segment.declared.clone(),
                index_sha256: segment.declared_sha256.clone(),
                path: segment.path.clone(),
                mapped: mapped.contains_key(object),
                copied: segment.copied.load(std::sync::atomic::Ordering::Relaxed),
                at_open: segment.identity_at_open.clone(),
                now: FileIdentity::capture(&segment.file),
            })
            .collect();
        StoreInstance {
            segments,
            absent_objects: self.absent.iter().cloned().collect(),
            load_count: self.load_count(),
            bytes_read: self.bytes_read(),
            mapped_regions: self.mapped_regions(),
            mapped_objects: mapped.len(),
        }
    }
}
