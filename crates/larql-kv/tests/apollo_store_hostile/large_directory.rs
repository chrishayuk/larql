//! Archives with many directory records, the class the byte and filesystem fuzzing
//! never generated. Their failure modes scale with the record count rather than the
//! file's length: sort buffers, per-record allocations, members sharing a payload.
//!
//! `cargo test` runs a short smoke. For a campaign:
//! `LARQL_APOLLO_FUZZ_ITERS=1000 LARQL_APOLLO_FUZZ_MAX_RECORDS=2000000 cargo test \
//!   --release -p larql-kv --test apollo_store_hostile large_directory -- --nocapture`

use std::collections::BTreeMap;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use std::time::Duration;

use larql_kv::apollo::{ApolloStore, ArchiveError, StoreLimits, StoreLoadError};

use crate::fixtures::*;
use crate::instrument::watched_under_cap;

const SMOKE_ITERS: u64 = 12;
const SMOKE_MAX_RECORDS: u64 = 1 << 14;
const ITERS_VAR: &str = "LARQL_APOLLO_FUZZ_LARGE_ITERS";
const MAX_RECORDS_VAR: &str = "LARQL_APOLLO_FUZZ_MAX_RECORDS";
const SEED_VAR: &str = "LARQL_APOLLO_FUZZ_SEED";
const DEFAULT_SEED: u64 = 684;
const LOAD_DEADLINE: Duration = Duration::from_secs(120);
const MIB: usize = 1 << 20;
/// Enough to tell a 22-28 MiB sort buffer from a fallible refusal, and nothing at all.
const CAPS: [usize; 7] = [
    2 * MIB,
    4 * MIB,
    8 * MIB,
    16 * MIB,
    24 * MIB,
    64 * MIB,
    usize::MAX,
];
/// Directory records in memory, at most, per the module doc in `npz.rs`.
const DIRECTORY_PEAK_PER_RECORD: usize = 200;
const FIXED_PEAK_ALLOWANCE: usize = 2 * MIB;
/// Hole sizes after the payload, so that summed claims would fit the file.
const HOLE_PER_RECORD: u64 = 128;

#[derive(Debug, Clone, Copy)]
enum Shape {
    /// Every record is a window pointing at one payload.
    SharedPayload,
    /// The same, with a hole so that the sum of their sizes fits in the file.
    SharedPayloadWithHole,
    /// Every record has the same window name, over one payload.
    DuplicateNames,
    /// Every record a distinct member with its own local header: a legitimate layout.
    Disjoint,
    /// Records named so that the load ignores them, each with its own local header.
    IgnoredNames,
}

const SHAPES: [Shape; 5] = [
    Shape::SharedPayload,
    Shape::SharedPayloadWithHole,
    Shape::DuplicateNames,
    Shape::Disjoint,
    Shape::IgnoredNames,
];

fn name_for(shape: Shape, index: u64) -> Vec<u8> {
    match shape {
        Shape::DuplicateNames => b"0.npy".to_vec(),
        Shape::IgnoredNames => format!("a{index}").into_bytes(),
        _ => format!("{index}.npy").into_bytes(),
    }
}

/// Write `records` directory records straight to `path`, returning its length. A
/// `u64` offset can't be carried by the 32-bit fields here, which is fine for files
/// far smaller than 4 GiB.
fn write_directory(path: &Path, shape: Shape, records: u64) -> u64 {
    let payload = u32_npy(&[]);
    let crc = crc32_of(&payload);
    let member_len = u32::try_from(payload.len()).unwrap();
    let mut file = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    let mut directory = Vec::new();
    let mut at = 0u64;
    let disjoint = matches!(shape, Shape::Disjoint | Shape::IgnoredNames);
    for index in 0..records {
        let name = name_for(shape, index);
        let first_of_shared = index == 0;
        let record = RawRecord {
            name: &name,
            crc,
            member_len,
            local_offset: if disjoint {
                u32::try_from(at).unwrap()
            } else {
                0
            },
        };
        if disjoint || first_of_shared {
            let local = local_record(&record);
            file.write_all(&local).unwrap();
            file.write_all(&payload).unwrap();
            at += (local.len() + payload.len()) as u64;
        }
        directory.extend_from_slice(&central_record(&record));
    }
    if matches!(shape, Shape::SharedPayloadWithHole) {
        at += records * HOLE_PER_RECORD;
        file.seek(SeekFrom::Start(at)).unwrap();
    }
    file.write_all(&directory).unwrap();
    file.write_all(&end_records(records, directory.len() as u64, at))
        .unwrap();
    file.flush().unwrap();
    std::fs::metadata(path).unwrap().len()
}

fn no_limit_but_records(records: u64) -> StoreLimits {
    StoreLimits {
        max_records: records,
        ..StoreLimits::default()
    }
}

/// Different names can't all match the one local header they point at; the same name
/// can, and then the extents collide.
fn is_shared_payload_refusal(err: &StoreLoadError) -> bool {
    matches!(
        err,
        StoreLoadError::Archive {
            source: ArchiveError::OverlappingMembers { .. }
                | ArchiveError::BadLocalHeader("name differs from the directory"),
            ..
        }
    )
}

fn is_expected_refusal(shape: Shape, err: &StoreLoadError) -> bool {
    let overlap = is_shared_payload_refusal(err);
    let allocation = matches!(err, StoreLoadError::AllocationRefused { .. })
        || matches!(
            err,
            StoreLoadError::Npy {
                source: larql_kv::apollo::npy::NpyError::AllocationRefused { .. },
                ..
            }
        );
    match shape {
        Shape::SharedPayload | Shape::SharedPayloadWithHole | Shape::DuplicateNames => {
            overlap || allocation
        }
        // Everything here reads fine, and the manifest claims no windows.
        Shape::Disjoint => allocation || matches!(err, StoreLoadError::ManifestMismatch(_)),
        Shape::IgnoredNames => allocation,
    }
}

/// Load a directory of this shape under a cap. An abort fails the whole test process;
/// that's what the sort buffer used to do. Anything else must be a typed refusal this
/// shape can produce, and memory must stay within what the record count explains.
fn check_load(shape: Shape, records: u64, cap: usize, seen: &mut BTreeMap<String, u64>) {
    let dir = store_with_no_windows();
    let path = dir.path().join(TOKENS_NPZ);
    write_directory(&path, shape, records);
    let root = dir.path().to_path_buf();
    let limits = no_limit_but_records(records);
    let measured = watched_under_cap(LOAD_DEADLINE, cap, move || {
        ApolloStore::load_with_limits(&root, &limits)
    });
    let bound = records as usize * DIRECTORY_PEAK_PER_RECORD + FIXED_PEAK_ALLOWANCE;
    assert!(
        measured.peak_bytes <= bound,
        "{shape:?} x{records}: peak {} over {bound}",
        measured.peak_bytes
    );
    match &measured.value {
        Ok(_) if matches!(shape, Shape::IgnoredNames) => {
            *seen.entry("Ok".to_string()).or_default() += 1;
        }
        Ok(_) => panic!("{shape:?} x{records} cap {cap}: must not load"),
        Err(err) => {
            assert!(
                is_expected_refusal(shape, err),
                "{shape:?} x{records} cap {cap}: {err:?}"
            );
            let name = format!("{err:?}");
            let name = name
                .split(|c: char| !c.is_alphanumeric())
                .next()
                .unwrap_or_default()
                .to_string();
            *seen.entry(name).or_default() += 1;
        }
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// The fuzz entry point: random shape, record count and cap.
#[test]
fn fuzz_large_directories_hold_the_invariants() {
    let iterations = env_u64(ITERS_VAR, SMOKE_ITERS);
    let max_records = env_u64(MAX_RECORDS_VAR, SMOKE_MAX_RECORDS);
    let mut state = env_u64(SEED_VAR, DEFAULT_SEED) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut seen = BTreeMap::new();
    for _ in 0..iterations {
        let shape = SHAPES[next() as usize % SHAPES.len()];
        let cap = CAPS[next() as usize % CAPS.len()];
        // Mostly two to the power of something, so large counts actually come up.
        let bits = u64::BITS - max_records.leading_zeros();
        let records = (1 << (next() % u64::from(bits))).clamp(2, max_records);
        check_load(shape, records, cap, &mut seen);
    }
    println!("large_directory iters={iterations} max_records={max_records} outcomes={seen:?}");
}

/// The reported case: the sort's merge buffer scaled with records and aborted under a
/// cap between the file's cost and its own. Every cap must now be typed, or a load.
#[test]
fn large_directory_under_any_cap_never_aborts() {
    const RECORDS: u64 = 1 << 13;
    let mut seen = BTreeMap::new();
    for cap in CAPS {
        check_load(Shape::Disjoint, RECORDS, cap, &mut seen);
    }
    assert!(seen.contains_key("ManifestMismatch"), "{seen:?}");
}

#[test]
fn shared_payload_is_refused_with_or_without_a_hole() {
    const RECORDS: u64 = 1 << 10;
    for shape in [
        Shape::SharedPayload,
        Shape::SharedPayloadWithHole,
        Shape::DuplicateNames,
    ] {
        let dir = store_with_no_windows();
        write_directory(&dir.path().join(TOKENS_NPZ), shape, RECORDS);
        let err = load_watched(dir.path(), StoreLimits::default())
            .value
            .expect_err("shared payload");
        assert!(is_shared_payload_refusal(&err), "{shape:?}: {err:?}");
    }
}

/// The record limit's default holds in practice: 2^17 + 1 records is refused without
/// paying for any of them, and at the limit the directory costs what the docs say.
#[test]
fn record_limit_bounds_the_directory_whatever_the_file_claims() {
    const LIMIT: u64 = larql_kv::apollo::store_limits::DEFAULT_MAX_RECORDS;
    let dir = store_with_no_windows();
    let path = dir.path().join(TOKENS_NPZ);
    write_directory(&path, Shape::IgnoredNames, LIMIT + 1);
    match cheap_refusal(dir.path()) {
        StoreLoadError::TooManyRecords { records, limit, .. } => {
            assert_eq!((records, limit), (LIMIT + 1, LIMIT));
        }
        other => panic!("expected TooManyRecords, got {other:?}"),
    }
    write_directory(&path, Shape::Disjoint, LIMIT);
    let measured = load_watched(dir.path(), StoreLimits::default());
    assert!(measured.value.is_err(), "the manifest claims no windows");
    let bound = LIMIT as usize * DIRECTORY_PEAK_PER_RECORD + FIXED_PEAK_ALLOWANCE;
    assert!(
        measured.peak_bytes <= bound,
        "peak {} over {bound}",
        measured.peak_bytes
    );
}

/// Failure and success both, under the cap harness: a legitimate store with thousands
/// of windows loads with room, and a refused allocation is typed without it.
#[test]
fn legitimate_store_loads_under_a_generous_cap_and_refuses_under_a_tight_one() {
    const WINDOWS: u64 = 4096;
    const GENEROUS: usize = 16 * MIB;
    const TIGHT: usize = MIB / 4;
    let dir = control_store();
    let tokens: Vec<(String, Vec<u8>)> = (0..WINDOWS)
        .map(|window| (format!("{window}.npy"), u32_npy(&[0; 3])))
        .collect();
    let members: Vec<(&[u8], &[u8])> = tokens
        .iter()
        .map(|(name, data)| (name.as_bytes(), data.as_slice()))
        .collect();
    std::fs::write(dir.path().join(TOKENS_NPZ), raw_npz(&members)).unwrap();
    std::fs::write(dir.path().join(MANIFEST), manifest(WINDOWS as usize, 1)).unwrap();
    for window in 1..WINDOWS {
        std::fs::write(
            dir.path()
                .join(format!("boundaries/window_{window:03}.npy")),
            f32_npy(1),
        )
        .unwrap();
    }
    let root = dir.path().to_path_buf();
    let store = watched_under_cap(DEADLINE, GENEROUS, move || ApolloStore::load(&root))
        .value
        .expect("loads with room");
    assert_eq!(store.window_tokens.len() as u64, WINDOWS);

    let root = dir.path().to_path_buf();
    let err = watched_under_cap(DEADLINE, TIGHT, move || ApolloStore::load(&root))
        .value
        .expect_err("refused without room");
    assert!(
        matches!(err, StoreLoadError::AllocationRefused { .. }),
        "{err:?}"
    );
}
