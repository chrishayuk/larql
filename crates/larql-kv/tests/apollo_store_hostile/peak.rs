//! The peak stated in `store_limits.rs`, held to a measurement. The worst store for
//! the formula is sparse, within every limit, and has its entries archive at the
//! archive limit: those bytes are still held while the entries decode.

use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use larql_kv::apollo::{ApolloStore, StoreLimits};

use crate::fixtures::*;
use crate::instrument::watched;

const MIB: u64 = 1 << 20;
const ENTRY_ROW_BYTES: u64 = 14;
const ENTRY_ROW_IN_MEMORY: u64 = 16;
const CHUNK: usize = 1 << 20;
/// Directory records, window list and array headers, per record, from the formula.
const RECORD_OVERHEAD: u64 = 200;
const FIXED_OVERHEAD: u64 = 2 * MIB;
/// A worst-case store at these scaled limits, against which to check the formula.
const LIMITS: StoreLimits = StoreLimits {
    max_manifest_bytes: MIB,
    max_array_file_bytes: 8 * MIB,
    max_archive_bytes: 8 * MIB,
    max_store_bytes: 32 * MIB,
    max_records: 1 << 17,
};

fn crc_of_header_then_zeros(header: &[u8], zeros: u64) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(header);
    let chunk = vec![0u8; CHUNK];
    let mut left = zeros;
    while left > 0 {
        let step = left.min(CHUNK as u64);
        hasher.update(&chunk[..step as usize]);
        left -= step;
    }
    hasher.finalize()
}

/// A stored archive whose members are each a real header and then a hole.
fn write_sparse_archive(path: &Path, members: &[(&[u8], &[u8], u64)]) {
    let mut file = std::fs::File::create(path).unwrap();
    let mut directory = Vec::new();
    let mut at = 0u64;
    for (name, header, member_len) in members {
        let zeros = member_len - header.len() as u64;
        let record = RawRecord {
            name,
            crc: crc_of_header_then_zeros(header, zeros),
            member_len: u32::try_from(*member_len).unwrap(),
            local_offset: u32::try_from(at).unwrap(),
        };
        let local = local_record(&record);
        file.seek(SeekFrom::Start(at)).unwrap();
        file.write_all(&local).unwrap();
        file.write_all(header).unwrap();
        at += local.len() as u64 + member_len;
        directory.extend_from_slice(&central_record(&record));
    }
    file.seek(SeekFrom::Start(at)).unwrap();
    file.write_all(&directory).unwrap();
    file.write_all(&end_records(
        members.len() as u64,
        directory.len() as u64,
        at,
    ))
    .unwrap();
}

/// Everything but `len` bytes of a sparse `.npy`: the data is a hole.
fn write_sparse_array(path: &Path, values: u64) {
    let header = npy("'<f4'", &format!("({values},)"), &[]);
    std::fs::write(path, &header).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_len(header.len() as u64 + values * size_of::<f32>() as u64)
        .unwrap();
}

/// Holes need sparse-file support to be cheap; NTFS would allocate these.
#[cfg_attr(windows, ignore = "NTFS allocates a file not marked sparse")]
#[test]
fn sparse_store_within_the_limits_peaks_at_the_stated_bound() {
    const OVERHEAD_FOR_ARCHIVE_RECORDS: u64 = 256;
    let dir = control_store();
    let root = dir.path();
    let archive = LIMITS.max_archive_bytes - OVERHEAD_FOR_ARCHIVE_RECORDS;
    let rows = (archive - MIB) / ENTRY_ROW_BYTES;
    let entries_header = npy(
        "[('token_id', '<u4'), ('coefficient', '<f4'), ('window_id', '<u2'), \
         ('position_in_window', '<u2'), ('fact_id', '<u2')]",
        &format!("({rows},)"),
        &[],
    );
    let entries_member = entries_header.len() as u64 + rows * ENTRY_ROW_BYTES;
    write_sparse_archive(
        &root.join(ENTRIES_NPZ),
        &[(b"entries.npy", &entries_header, entries_member)],
    );

    // Boundaries fill what the archives leave of the store limit.
    let per_file = LIMITS.max_array_file_bytes - MIB;
    let spare = LIMITS.max_store_bytes - 2 * LIMITS.max_archive_bytes - MIB;
    let windows = spare / per_file;
    let window_values = per_file / size_of::<f32>() as u64;
    for window in 0..windows {
        write_sparse_array(
            &root.join(format!("boundaries/window_{window:03}.npy")),
            window_values,
        );
    }

    let tokens = (archive - MIB) / windows / size_of::<u32>() as u64;
    let tokens_header = npy("'<u4'", &format!("({tokens},)"), &[]);
    let tokens_member = tokens_header.len() as u64 + tokens * size_of::<u32>() as u64;
    let names: Vec<String> = (0..windows).map(|window| format!("{window}.npy")).collect();
    let members: Vec<(&[u8], &[u8], u64)> = names
        .iter()
        .map(|name| (name.as_bytes(), tokens_header.as_slice(), tokens_member))
        .collect();
    write_sparse_archive(&root.join(TOKENS_NPZ), &members);
    let used = entries_member + windows * tokens_member;
    std::fs::remove_file(root.join(RESIDUAL)).unwrap();
    std::fs::write(
        root.join(MANIFEST),
        manifest(
            usize::try_from(windows).unwrap(),
            usize::try_from(rows).unwrap(),
        ),
    )
    .unwrap();

    let root = root.to_path_buf();
    let measured = watched(DEADLINE, None, move || {
        ApolloStore::load_with_limits(&root, &LIMITS)
    });
    let store = measured.value.expect("a hole decodes as zeros");
    assert_eq!(store.entries.len() as u64, rows);

    let peak = measured.peak_bytes as u64;
    let bound = LIMITS.max_store_bytes
        + LIMITS.max_archive_bytes * ENTRY_ROW_IN_MEMORY / ENTRY_ROW_BYTES
        + LIMITS.max_records * RECORD_OVERHEAD
        + FIXED_OVERHEAD;
    let formula_says_at_least = used + windows * per_file + rows * ENTRY_ROW_IN_MEMORY;
    assert!(peak <= bound, "peak {peak} over the stated bound {bound}");
    assert!(
        peak >= formula_says_at_least * 9 / 10,
        "peak {peak} far under {formula_says_at_least}: the store isn't the worst case"
    );
    println!(
        "peak measured={peak} bound={bound} store_limit={}",
        LIMITS.max_store_bytes
    );
}
