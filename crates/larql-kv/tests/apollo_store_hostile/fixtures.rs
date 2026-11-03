//! Well-formed stores and the building blocks for spoiling them.

use std::io::Write;
use std::path::Path;
use std::time::Duration;

use larql_kv::apollo::{ApolloStore, StoreLimits, StoreLoadError};
use tempfile::TempDir;
use zip::write::SimpleFileOptions;
use zip::ZipWriter;

use crate::instrument::{watched, Measured};

pub const MANIFEST: &str = "manifest.json";
pub const BOUNDARY: &str = "boundaries/window_000.npy";
pub const RESIDUAL: &str = "boundary_residual.npy";
pub const TOKENS_NPZ: &str = "window_token_lists.npz";
pub const ENTRIES_NPZ: &str = "entries.npz";
/// Every path a load opens in a one-window store.
pub const STORE_FILES: [&str; 5] = [MANIFEST, BOUNDARY, RESIDUAL, TOKENS_NPZ, ENTRIES_NPZ];

/// Generous for a load that does almost nothing, and far short of forever.
pub const DEADLINE: Duration = Duration::from_secs(20);
/// A refusal reads headers and at most the 64 KiB ZIP tail.
pub const REFUSAL_PEAK_BYTES: usize = 1 << 20;

const WINDOW_TOKENS: [u32; 3] = [0, 1, 2];
const HIDDEN: usize = 4;
const ENTRY_ROW: [u8; 14] = [7, 0, 0, 0, 0, 0, 0x80, 0x3f, 0, 0, 0, 0, 0, 0];
const ENTRIES_DESCR: &str = "[('token_id', '<u4'), ('coefficient', '<f4'), ('window_id', '<u2'), \
                             ('position_in_window', '<u2'), ('fact_id', '<u2')]";

pub fn npy(descr: &str, shape: &str, data: &[u8]) -> Vec<u8> {
    let header = format!("{{'descr': {descr}, 'fortran_order': False, 'shape': {shape}, }}\n");
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&u16::try_from(header.len()).unwrap().to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    out
}

pub fn f32_npy(values: usize) -> Vec<u8> {
    npy(
        "'<f4'",
        &format!("({values},)"),
        &vec![0; values * size_of::<f32>()],
    )
}

pub fn u32_npy(values: &[u32]) -> Vec<u8> {
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    npy("'<u4'", &format!("({},)", values.len()), &data)
}

pub fn entries_npy_with_token(token_id: u32) -> Vec<u8> {
    let mut row = ENTRY_ROW;
    row[..size_of::<u32>()].copy_from_slice(&token_id.to_le_bytes());
    npy(ENTRIES_DESCR, "(1,)", &row)
}

pub fn entries_npy(rows: usize) -> Vec<u8> {
    npy(
        ENTRIES_DESCR,
        &format!("({rows},)"),
        &ENTRY_ROW.repeat(rows),
    )
}

/// An honest stored archive, as `np.savez` makes.
pub fn npz(members: &[(&str, &[u8])]) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut writer = ZipWriter::new(std::io::Cursor::new(&mut buf));
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for (name, data) in members {
        writer.start_file(*name, options).unwrap();
        writer.write_all(data).unwrap();
    }
    writer.finish().unwrap();
    buf
}

pub fn manifest(num_windows: usize, num_entries: usize) -> Vec<u8> {
    format!(
        r#"{{"version":1,"num_entries":{num_entries},"num_windows":{num_windows},"num_tokens":3,
            "entries_per_window":1,"crystal_layer":1,"window_size":3,
            "arch_config":{{"retrieval_layer":1,"query_head":1,"injection_layer":1,
            "inject_coefficient":1.0}},"has_residuals":true}}"#
    )
    .into_bytes()
}

/// A well-formed one-window store, residual included.
pub fn control_store() -> TempDir {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    std::fs::create_dir(root.join("boundaries")).unwrap();
    std::fs::write(root.join(MANIFEST), manifest(1, 1)).unwrap();
    std::fs::write(root.join(BOUNDARY), f32_npy(HIDDEN)).unwrap();
    std::fs::write(root.join(RESIDUAL), f32_npy(HIDDEN)).unwrap();
    std::fs::write(
        root.join(TOKENS_NPZ),
        npz(&[("0.npy", &u32_npy(&WINDOW_TOKENS))]),
    )
    .unwrap();
    std::fs::write(
        root.join(ENTRIES_NPZ),
        npz(&[("entries.npy", &entries_npy(1))]),
    )
    .unwrap();
    dir
}

pub fn load_watched(
    root: &Path,
    limits: StoreLimits,
) -> Measured<Result<ApolloStore, StoreLoadError>> {
    let root = root.to_path_buf();
    watched(DEADLINE, None, move || {
        ApolloStore::load_with_limits(&root, &limits)
    })
}

/// A load under the default limits that must refuse cheaply and quickly.
pub fn cheap_refusal(root: &Path) -> StoreLoadError {
    let measured = load_watched(root, StoreLimits::default());
    assert!(
        measured.peak_bytes < REFUSAL_PEAK_BYTES,
        "refusal cost {} bytes at peak",
        measured.peak_bytes
    );
    measured.value.expect_err("store must be refused")
}

/// Replace `file` in the store with a hole of `len` logical bytes after `head`.
pub fn make_sparse(root: &Path, file: &str, head: &[u8], len: u64) {
    let path = root.join(file);
    std::fs::write(&path, head).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(len)
        .unwrap();
}

/// A raw stored record pair, written by hand because `ZipWriter` refuses duplicates
/// and can't make an archive lie. `crc` is over `member_len` bytes of `data` then zeros.
pub struct RawRecord<'a> {
    pub name: &'a [u8],
    pub crc: u32,
    pub member_len: u32,
    pub local_offset: u32,
}

const LOCAL_SIGNATURE: &[u8] = b"PK\x03\x04";
const CENTRAL_SIGNATURE: &[u8] = b"PK\x01\x02";
const END_SIGNATURE: &[u8] = b"PK\x05\x06";
const VERSION: [u8; 2] = [20, 0];
const NO_FLAGS_OR_METHOD_OR_TIME: [u8; 6] = [0; 6];
const DOS_DATE_EPOCH: [u8; 2] = [0x21, 0];
const NO_EXTRA: [u8; 2] = [0; 2];
/// Comment length, disk number, internal and external attributes.
const CENTRAL_ZEROED_FIELDS: [u8; 10] = [0; 10];

fn sizes_and_name_len(record: &RawRecord) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&VERSION);
    out.extend_from_slice(&NO_FLAGS_OR_METHOD_OR_TIME);
    out.extend_from_slice(&DOS_DATE_EPOCH);
    out.extend_from_slice(&record.crc.to_le_bytes());
    out.extend_from_slice(&record.member_len.to_le_bytes());
    out.extend_from_slice(&record.member_len.to_le_bytes());
    out.extend_from_slice(&u16::try_from(record.name.len()).unwrap().to_le_bytes());
    out.extend_from_slice(&NO_EXTRA);
    out
}

pub fn local_record(record: &RawRecord) -> Vec<u8> {
    let mut out = LOCAL_SIGNATURE.to_vec();
    out.extend_from_slice(&sizes_and_name_len(record));
    out.extend_from_slice(record.name);
    out
}

pub fn central_record(record: &RawRecord) -> Vec<u8> {
    let mut out = CENTRAL_SIGNATURE.to_vec();
    out.extend_from_slice(&VERSION);
    out.extend_from_slice(&sizes_and_name_len(record));
    out.extend_from_slice(&CENTRAL_ZEROED_FIELDS);
    out.extend_from_slice(&record.local_offset.to_le_bytes());
    out.extend_from_slice(record.name);
    out
}

/// A plain end record, or ZIP64 records with it past what 16 bits can count.
pub fn end_records(records: u64, directory_len: u64, directory_at: u64) -> Vec<u8> {
    const ZIP32_MAX_RECORDS: u64 = u16::MAX as u64 - 1;
    const NO_DISKS: [u8; 4] = [0; 4];
    const ZIP64_END_FIXED_SIZE: u64 = 44;
    const VERSION_ZIP64: [u8; 2] = [45, 0];
    let mut out = Vec::new();
    if records > ZIP32_MAX_RECORDS {
        out.extend_from_slice(b"PK\x06\x06");
        out.extend_from_slice(&ZIP64_END_FIXED_SIZE.to_le_bytes());
        out.extend_from_slice(&VERSION_ZIP64);
        out.extend_from_slice(&VERSION_ZIP64);
        out.extend_from_slice(&[0; 8]);
        out.extend_from_slice(&records.to_le_bytes());
        out.extend_from_slice(&records.to_le_bytes());
        out.extend_from_slice(&directory_len.to_le_bytes());
        out.extend_from_slice(&directory_at.to_le_bytes());
        out.extend_from_slice(b"PK\x06\x07");
        out.extend_from_slice(&NO_DISKS);
        out.extend_from_slice(&(directory_at + directory_len).to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(END_SIGNATURE);
        out.extend_from_slice(&NO_DISKS);
        out.extend_from_slice(&[0xFF; 12]);
    } else {
        let count = u16::try_from(records).unwrap().to_le_bytes();
        out.extend_from_slice(END_SIGNATURE);
        out.extend_from_slice(&NO_DISKS);
        out.extend_from_slice(&count);
        out.extend_from_slice(&count);
        out.extend_from_slice(&u32::try_from(directory_len).unwrap().to_le_bytes());
        out.extend_from_slice(&u32::try_from(directory_at).unwrap().to_le_bytes());
    }
    out.extend_from_slice(&[0; 2]);
    out
}

pub fn crc32_of(bytes: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

/// The members laid out honestly, by hand. Names may repeat.
pub fn raw_npz(members: &[(&[u8], &[u8])]) -> Vec<u8> {
    let mut body = Vec::new();
    let mut directory = Vec::new();
    for (name, data) in members {
        let record = RawRecord {
            name,
            crc: crc32_of(data),
            member_len: u32::try_from(data.len()).unwrap(),
            local_offset: u32::try_from(body.len()).unwrap(),
        };
        body.extend_from_slice(&local_record(&record));
        body.extend_from_slice(data);
        directory.extend_from_slice(&central_record(&record));
    }
    let end = end_records(
        members.len() as u64,
        directory.len() as u64,
        body.len() as u64,
    );
    [body, directory, end].concat()
}

/// A store whose manifest claims no windows, for loads aimed only at an archive.
pub fn store_with_no_windows() -> TempDir {
    let dir = control_store();
    std::fs::write(dir.path().join(MANIFEST), manifest(0, 1)).unwrap();
    dir
}

#[test]
fn control_store_loads_with_its_residual() {
    let store = ApolloStore::load(control_store().path()).expect("control loads");
    assert_eq!(store.window_tokens, vec![WINDOW_TOKENS.to_vec()]);
    assert_eq!(store.entries.len(), 1);
    assert_eq!(store.boundary_residual.map(|r| r.len()), Some(HIDDEN));
}
