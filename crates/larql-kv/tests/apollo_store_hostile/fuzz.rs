//! A deterministic fuzz harness over store directories, not just store bytes.
//!
//! Earlier byte fuzzing ran about 1.08 million iterations and never found the
//! sparse abort, because it only ever changed file contents. Here each iteration
//! spoils a store through bytes and through the filesystem: lengths that leave
//! holes, symlinks to devices, FIFOs, directories, aliases and removals.
//!
//! Invariants on every load: it returns within the deadline, does not panic, and
//! stays within a memory bound derived from its limits. Separately, a differential
//! run checks the first-party archive reader against the `zip` crate, and
//! `large_directory.rs` covers archives with many records.
//!
//! `cargo test` runs a short smoke. For a campaign:
//! `LARQL_APOLLO_FUZZ_ITERS=1000000 LARQL_APOLLO_FUZZ_SEED=7 cargo test --release \
//!   -p larql-kv --test apollo_store_hostile fuzz -- --nocapture`

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;
use std::time::Duration;

use larql_kv::apollo::{ApolloStore, StoreLimits, StoreLoadError};

use crate::fixtures::*;
use crate::instrument::watched;

const SMOKE_ITERS: u64 = 1_500;
const DEFAULT_SEED: u64 = 679_684;
const ITERS_VAR: &str = "LARQL_APOLLO_FUZZ_ITERS";
const SEED_VAR: &str = "LARQL_APOLLO_FUZZ_SEED";
const ITER_DEADLINE: Duration = Duration::from_secs(30);
const MAX_MUTATIONS: u64 = 4;
/// Tight so that a hole inside the limits costs little per iteration.
const FUZZ_LIMITS: StoreLimits = StoreLimits {
    max_manifest_bytes: 1 << 16,
    max_array_file_bytes: 1 << 20,
    max_archive_bytes: 8 << 20,
    max_store_bytes: 16 << 20,
    max_records: 1 << 12,
};
/// Every array read once as bytes and once decoded, plus small bookkeeping.
const PEAK_BOUND: usize = 2 * FUZZ_LIMITS.max_store_bytes as usize + (1 << 20);
/// Lengths for holes: within each limit, over each limit, and the abort's size.
#[cfg(not(windows))]
const HOLE_LENGTHS: [u64; 7] = [1, 4096, 1 << 16, 1 << 20, 1 << 30, 1 << 40, 1 << 44];
/// NTFS allocates a file not marked sparse.
#[cfg(windows)]
const HOLE_LENGTHS: [u64; 4] = [1, 4096, 1 << 16, 1 << 20];
/// Values that sit on arithmetic edges of ZIP and `.npy` fields.
const EDGE_VALUES: [u64; 12] = [
    0,
    1,
    0xFF,
    0xFFFF,
    0xFFFE,
    0x7FFF,
    0xFFFF_FFFF,
    0xFFFF_FFFE,
    1 << 31,
    1 << 40,
    1 << 62,
    u64::MAX,
];
const FIELD_WIDTHS: [usize; 4] = [1, 2, 4, 8];
/// ZIP records end near the file's end, so mutations favour the tail.
const TAIL_FOCUS: u64 = 120;
const MAX_HARNESS_READ: u64 = 1 << 20;
const WINDOW_LEN: usize = 3;
const ENTRIES_MEMBER: &str = "entries.npy";

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u64) as usize]
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Overwrite one field with an edge value, leaving the length unchanged.
fn mutate_bytes(rng: &mut Rng, bytes: &mut [u8]) {
    let len = bytes.len() as u64;
    if len == 0 {
        return;
    }
    let at = if rng.below(3) == 0 && len > TAIL_FOCUS {
        len - 1 - rng.below(TAIL_FOCUS)
    } else {
        rng.below(len)
    } as usize;
    let value = rng.pick(&EDGE_VALUES).to_le_bytes();
    let width = rng.pick(&FIELD_WIDTHS).min(bytes.len() - at);
    bytes[at..at + width].copy_from_slice(&value[..width]);
}

fn remove_path(path: &Path) {
    if path.is_dir() && !path.is_symlink() {
        let _ = std::fs::remove_dir_all(path);
    } else {
        let _ = std::fs::remove_file(path);
    }
}

/// The harness's own reads need the guards the loader has: an earlier mutation may
/// have made this path a device, a FIFO or a 16 TiB hole.
fn read_if_small_regular(path: &Path) -> Option<Vec<u8>> {
    let metadata = std::fs::symlink_metadata(path).ok()?;
    (metadata.is_file() && metadata.len() <= MAX_HARNESS_READ)
        .then(|| std::fs::read(path).ok())
        .flatten()
}

fn spoil_bytes(rng: &mut Rng, path: &Path) {
    if let Some(mut bytes) = read_if_small_regular(path) {
        mutate_bytes(rng, &mut bytes);
        let _ = std::fs::write(path, bytes);
    }
}

fn spoil_truncate(rng: &mut Rng, path: &Path) {
    if let Some(mut bytes) = read_if_small_regular(path) {
        let keep = rng.below(bytes.len() as u64 + 1) as usize;
        bytes.truncate(keep);
        let _ = std::fs::write(path, bytes);
    }
}

fn spoil_length(rng: &mut Rng, path: &Path) {
    let len = rng.pick(&HOLE_LENGTHS);
    // Opening a FIFO for writing would block the harness itself.
    if std::fs::symlink_metadata(path).is_ok_and(|metadata| metadata.is_file()) {
        if let Ok(file) = std::fs::OpenOptions::new().write(true).open(path) {
            let _ = file.set_len(len);
        }
    }
}

#[cfg(unix)]
fn spoil_kind(rng: &mut Rng, root: &Path, path: &Path) -> &'static str {
    const DEVICES: [&str; 4] = ["/dev/zero", "/dev/null", "/dev/urandom", "/dev/tty"];
    remove_path(path);
    match rng.below(6) {
        0 => {
            let _ = std::os::unix::fs::symlink(rng.pick(&DEVICES), path);
            "symlink_device"
        }
        1 => {
            if let Some(text) = path.to_str().and_then(|p| std::ffi::CString::new(p).ok()) {
                // SAFETY: `text` is a valid NUL-terminated string for the call.
                unsafe { libc::mkfifo(text.as_ptr(), 0o600) };
            }
            "fifo"
        }
        2 => {
            let _ = std::fs::create_dir(path);
            "directory"
        }
        3 => {
            let other = root.join(rng.pick(&STORE_FILES));
            let _ = std::os::unix::fs::symlink(other, path);
            "alias"
        }
        4 => {
            let _ = std::os::unix::fs::symlink(path, path);
            "loop"
        }
        _ => "removed",
    }
}

#[cfg(not(unix))]
fn spoil_kind(rng: &mut Rng, _root: &Path, path: &Path) -> &'static str {
    remove_path(path);
    if rng.below(2) == 0 {
        let _ = std::fs::create_dir(path);
        "directory"
    } else {
        "removed"
    }
}

/// Spoil the store once; returns the name of what was done, for the tally.
fn spoil(rng: &mut Rng, root: &Path) -> &'static str {
    let path = root.join(rng.pick(&STORE_FILES));
    match rng.below(5) {
        0 | 1 => {
            spoil_bytes(rng, &path);
            "bytes"
        }
        2 => {
            spoil_length(rng, &path);
            "length"
        }
        3 => spoil_kind(rng, root, &path),
        _ => {
            spoil_truncate(rng, &path);
            "truncate"
        }
    }
}

fn variant_name(err: &StoreLoadError) -> String {
    let text = format!("{err:?}");
    text.split(|c: char| !c.is_alphanumeric())
        .next()
        .unwrap_or_default()
        .to_string()
}

#[test]
fn fuzz_store_directories_hold_the_invariants() {
    let iterations = env_u64(ITERS_VAR, SMOKE_ITERS);
    let mut rng = Rng(env_u64(SEED_VAR, DEFAULT_SEED) | 1);
    let mut outcomes = BTreeMap::<String, u64>::new();
    let mut spoils = BTreeMap::<&'static str, u64>::new();
    let mut slowest = Duration::ZERO;
    let mut highest_peak = 0;
    for _ in 0..iterations {
        let dir = control_store();
        for _ in 0..=rng.below(MAX_MUTATIONS) {
            *spoils.entry(spoil(&mut rng, dir.path())).or_default() += 1;
        }
        let root = dir.path().to_path_buf();
        let measured = watched(ITER_DEADLINE, None, move || {
            ApolloStore::load_with_limits(&root, &FUZZ_LIMITS)
        });
        assert!(
            measured.peak_bytes <= PEAK_BOUND,
            "peak {} over {PEAK_BOUND}: {:?}",
            measured.peak_bytes,
            measured.value.as_ref().err()
        );
        slowest = slowest.max(measured.elapsed);
        highest_peak = highest_peak.max(measured.peak_bytes);
        let outcome = measured
            .value
            .as_ref()
            .map_or_else(variant_name, |_| "Ok".to_string());
        *outcomes.entry(outcome).or_default() += 1;
    }
    println!(
        "fuzz iters={iterations} slowest={slowest:?} highest_peak={highest_peak} \
         spoils={spoils:?} outcomes={outcomes:?}"
    );
}

/// The members this load would use, as the `zip` crate reads them, or `None` if it
/// refuses one or the archive. `zip` resolves a duplicate name to the last record.
fn read_with_zip(bytes: &[u8]) -> Option<BTreeMap<String, Vec<u8>>> {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes)).ok()?;
    let mut members = BTreeMap::new();
    let names: Vec<String> = archive.file_names().map(str::to_string).collect();
    for name in names {
        let is_used = name == ENTRIES_MEMBER
            || name
                .strip_suffix(".npy")
                .is_some_and(|stem| stem.parse::<usize>().is_ok());
        if !is_used {
            continue;
        }
        let mut member = archive.by_name(&name).ok()?;
        let mut data = Vec::new();
        member.read_to_end(&mut data).ok()?;
        members.insert(name, data);
    }
    Some(members)
}

/// The array bytes at the end of a one-dimensional `.npy`, given how long they are.
fn array_bytes(npy_bytes: &[u8], len: usize) -> Option<&[u8]> {
    npy_bytes.get(npy_bytes.len().checked_sub(len)?..)
}

/// Whether `store` holds what `zip` read from `archive_file`, for what both accepted.
fn check_same_data(
    store: &ApolloStore,
    archive_file: &str,
    zip_members: &BTreeMap<String, Vec<u8>>,
) {
    if archive_file == TOKENS_NPZ {
        for (window, tokens) in store.window_tokens.iter().enumerate() {
            let actual: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
            let member = zip_members.get(&format!("{window}.npy"));
            let expected = member.and_then(|data| array_bytes(data, actual.len()));
            assert_eq!(Some(actual.as_slice()), expected, "window {window} differs");
        }
    } else {
        let actual: Vec<u8> = store
            .entries
            .iter()
            .flat_map(|entry| {
                [
                    entry.token_id.to_le_bytes().as_slice(),
                    entry.coefficient.to_le_bytes().as_slice(),
                    entry.window_id.to_le_bytes().as_slice(),
                    entry.position_in_window.to_le_bytes().as_slice(),
                    entry.fact_id.to_le_bytes().as_slice(),
                ]
                .concat()
            })
            .collect();
        let member = zip_members.get(ENTRIES_MEMBER);
        let expected = member.and_then(|data| array_bytes(data, actual.len()));
        assert_eq!(Some(actual.as_slice()), expected, "entries differ");
    }
}

/// The honest archive and hand-made ones that test which member gets read.
fn differential_seeds(archive_file: &str) -> Vec<Vec<u8>> {
    const FIRST: u32 = 111;
    const LAST: u32 = 222;
    if archive_file == TOKENS_NPZ {
        let token_member = |token: u32| u32_npy(&[token; WINDOW_LEN]);
        vec![
            npz(&[
                ("0.npy", &token_member(FIRST)),
                ("1.npy", &token_member(LAST)),
            ]),
            raw_npz(&[
                (b"0.npy", &token_member(FIRST)),
                (b"1.npy", &token_member(LAST)),
                (b"1.npy", &token_member(FIRST)),
            ]),
            raw_npz(&[
                (b"0.npy", &token_member(FIRST)),
                (b"1.npy", &token_member(LAST)),
                (b"1.npy\0x", &token_member(FIRST)),
            ]),
            raw_npz(&[
                (b"00.npy", &token_member(FIRST)),
                (b"0.npy", &token_member(LAST)),
                (b"1.npy", &token_member(LAST)),
            ]),
        ]
    } else {
        let (first, last) = (entries_npy_with_token(FIRST), entries_npy_with_token(LAST));
        let entries = ENTRIES_MEMBER.as_bytes();
        vec![
            npz(&[(ENTRIES_MEMBER, &last)]),
            raw_npz(&[(entries, &first), (entries, &last)]),
            raw_npz(&[(b"entries_old.npy", &first), (entries, &last)]),
            raw_npz(&[(b"entries", &first), (entries, &last)]),
            raw_npz(&[(entries, &last), (b"entries.npy\0x", &first)]),
            raw_npz(&[
                (b"./entries.npy", &first),
                (b"ENTRIES.NPY", &first),
                (entries, &last),
            ]),
        ]
    }
}

/// Compare the two readers on crafted and mutated archives of both kinds. They may
/// reasonably disagree on whether to refuse, and the counts are printed for review.
/// What must never happen is both accepting an archive and returning different data.
#[test]
fn fuzz_archive_reader_agrees_with_zip_on_what_both_accept() {
    const WINDOWS: usize = 2;
    let iterations = env_u64(ITERS_VAR, SMOKE_ITERS);
    let mut rng = Rng(env_u64(SEED_VAR, DEFAULT_SEED) | 1);
    let dir = control_store();
    std::fs::write(dir.path().join(MANIFEST), manifest(WINDOWS, 1)).unwrap();
    std::fs::write(dir.path().join("boundaries/window_001.npy"), f32_npy(1)).unwrap();
    let good_tokens = differential_seeds(TOKENS_NPZ).swap_remove(0);
    let good_entries = differential_seeds(ENTRIES_NPZ).swap_remove(0);
    let (mut both_ok, mut both_refuse, mut only_ours) = (0u64, 0u64, 0u64);
    let mut only_zip = BTreeMap::<String, u64>::new();
    for _ in 0..iterations {
        let archive_file = rng.pick(&[TOKENS_NPZ, ENTRIES_NPZ]);
        let seeds = differential_seeds(archive_file);
        let mut bytes = seeds[rng.below(seeds.len() as u64) as usize].clone();
        for _ in 0..rng.below(MAX_MUTATIONS + 1) {
            mutate_bytes(&mut rng, &mut bytes);
        }
        std::fs::write(dir.path().join(TOKENS_NPZ), &good_tokens).unwrap();
        std::fs::write(dir.path().join(ENTRIES_NPZ), &good_entries).unwrap();
        std::fs::write(dir.path().join(archive_file), &bytes).unwrap();
        let ours = ApolloStore::load(dir.path());
        let theirs = read_with_zip(&bytes);
        match (&ours, &theirs) {
            (Ok(store), Some(zip_members)) => {
                both_ok += 1;
                check_same_data(store, archive_file, zip_members);
            }
            (Err(_), None) => both_refuse += 1,
            (Err(refusal), Some(_)) => *only_zip.entry(variant_name(refusal)).or_default() += 1,
            (Ok(_), None) => only_ours += 1,
        }
    }
    println!(
        "differential iters={iterations} both_ok={both_ok} both_refuse={both_refuse} \
         only_zip={only_zip:?} only_ours={only_ours}"
    );
}
