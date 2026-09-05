//! What staging must preserve, and what it must not do quietly.
//!
//! Every test here compares BIT PATTERNS rather than values. An f32
//! comparison would call two different NaNs equal and would call `-0.0`
//! equal to `0.0`, so a staging path that normalised either would pass a
//! value-equality test and change a model's arithmetic. The claim this
//! module makes is that the bytes come back unchanged, and that is the
//! claim the tests check.

use super::staged::{
    staged_bytes, staged_images, StagedF32, DEFAULT_STAGE_MIN_BYTES, STAGE_ENV,
    STAGE_MIN_BYTES_ENV, STAGE_OFF,
};
use serial_test::serial;

/// The environment is process-global, so every test sets what it needs
/// and puts it back. `min` of `Some(n)` stages anything at or above `n`.
fn with_env<T>(stage_off: bool, min: Option<usize>, f: impl FnOnce() -> T) -> T {
    let prev_off = std::env::var(STAGE_ENV).ok();
    let prev_min = std::env::var(STAGE_MIN_BYTES_ENV).ok();
    if stage_off {
        std::env::set_var(STAGE_ENV, STAGE_OFF);
    } else {
        std::env::remove_var(STAGE_ENV);
    }
    match min {
        Some(n) => std::env::set_var(STAGE_MIN_BYTES_ENV, n.to_string()),
        None => std::env::remove_var(STAGE_MIN_BYTES_ENV),
    }
    let out = f();
    match prev_off {
        Some(v) => std::env::set_var(STAGE_ENV, v),
        None => std::env::remove_var(STAGE_ENV),
    }
    match prev_min {
        Some(v) => std::env::set_var(STAGE_MIN_BYTES_ENV, v),
        None => std::env::remove_var(STAGE_MIN_BYTES_ENV),
    }
    out
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The values a float format can lose if anything reinterprets it:
/// signed zeroes, both infinities, a signalling and a quiet NaN, and the
/// smallest subnormal.
fn adversarial(n: usize) -> Vec<f32> {
    let seeds = [
        0.0f32,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_0001),
        f32::from_bits(0x7f80_0001),
        f32::from_bits(0x0000_0001),
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        1.0,
        -1.0,
    ];
    (0..n)
        .map(|i| {
            if i < seeds.len() {
                seeds[i]
            } else {
                // Deterministic filler that spans exponents.
                f32::from_bits(0x3f80_0000u32.wrapping_add((i as u32).wrapping_mul(0x0001_1111)))
            }
        })
        .collect()
}

#[test]
#[serial]
fn an_owned_image_reads_back_exactly() {
    let values = adversarial(64);
    let staged = StagedF32::from(values.clone());
    assert!(!staged.is_mapped());
    assert_eq!(bits(&staged), bits(&values));
}

#[test]
#[serial]
fn a_mapped_image_reads_back_bit_identically() {
    // 4 KB of f32 against a 1 KB threshold: comfortably staged, and
    // small enough that the test costs nothing.
    let values = adversarial(1024);
    let staged = with_env(false, Some(1024), || StagedF32::stage(values.clone()))
        .expect("staging a 4 KB image");
    assert!(
        staged.is_mapped(),
        "a 4 KB image against a 1 KB threshold must be mapped, or this test proves nothing"
    );
    assert_eq!(
        bits(&staged),
        bits(&values),
        "staged bytes differ from the values written"
    );
}

#[test]
#[serial]
fn the_control_that_proves_the_comparison_can_fail() {
    // The bit comparison above is only evidence if it can say no.
    let a = adversarial(1024);
    let mut b = a.clone();
    b[7] = f32::from_bits(a[7].to_bits() ^ 1);
    assert_ne!(bits(&a), bits(&b));
}

#[test]
#[serial]
fn negative_zero_and_nan_survive_the_round_trip() {
    let values = vec![
        -0.0f32,
        0.0,
        f32::from_bits(0x7fc0_0001),
        f32::from_bits(0xffc0_0002),
    ]
    .into_iter()
    .cycle()
    .take(1024)
    .collect::<Vec<_>>();
    let staged = with_env(false, Some(1024), || StagedF32::stage(values.clone())).expect("staging");
    assert!(staged.is_mapped());
    // Value equality would pass here even if -0.0 became 0.0.
    assert_eq!(bits(&staged), bits(&values));
    assert!(staged[0].is_sign_negative(), "-0.0 lost its sign");
    assert!(staged[2].is_nan(), "a NaN stopped being a NaN");
}

#[test]
#[serial]
fn an_image_below_the_threshold_stays_owned() {
    let values = adversarial(16);
    let staged = with_env(false, Some(DEFAULT_STAGE_MIN_BYTES), || {
        StagedF32::stage(values.clone())
    })
    .expect("staging");
    assert!(!staged.is_mapped());
    assert_eq!(bits(&staged), bits(&values));
}

#[test]
#[serial]
fn the_environment_can_turn_staging_off_entirely() {
    let values = adversarial(1024);
    let staged =
        with_env(true, Some(1), || StagedF32::stage(values.clone())).expect("staging disabled");
    assert!(
        !staged.is_mapped(),
        "`{STAGE_ENV}={STAGE_OFF}` must keep the image anonymous — the v1/v2 equivalence \
         control depends on this arm existing in one binary"
    );
    assert_eq!(bits(&staged), bits(&values));
}

#[test]
#[serial]
fn two_staged_images_do_not_overlap_in_the_arena() {
    // The offset arithmetic is the one place a bug would silently serve
    // one weight's bytes for another's.
    let a = adversarial(1024);
    let b: Vec<f32> = a
        .iter()
        .map(|x| f32::from_bits(x.to_bits() ^ 0x00ff_00ff))
        .collect();
    let (sa, sb) = with_env(false, Some(1024), || {
        (
            StagedF32::stage(a.clone()).expect("stage a"),
            StagedF32::stage(b.clone()).expect("stage b"),
        )
    });
    assert!(sa.is_mapped() && sb.is_mapped());
    assert_eq!(bits(&sa), bits(&a));
    assert_eq!(bits(&sb), bits(&b));
    assert_ne!(
        bits(&sa),
        bits(&sb),
        "the two images must be distinguishable"
    );
}

#[test]
#[serial]
fn an_odd_length_image_keeps_its_last_element() {
    // A length that is not a multiple of the page, so the mapping's tail
    // is inside the padding.
    let values = adversarial(1031);
    let staged = with_env(false, Some(1024), || StagedF32::stage(values.clone())).expect("staging");
    assert!(staged.is_mapped());
    assert_eq!(staged.len(), 1031);
    assert_eq!(bits(&staged), bits(&values));
}

#[test]
#[serial]
fn the_counters_only_move_for_mapped_images() {
    let before_bytes = staged_bytes();
    let before_images = staged_images();
    let small = with_env(false, Some(DEFAULT_STAGE_MIN_BYTES), || {
        StagedF32::stage(adversarial(4))
    })
    .expect("staging");
    assert!(!small.is_mapped());
    assert_eq!(staged_bytes(), before_bytes);
    assert_eq!(staged_images(), before_images);

    let big = with_env(false, Some(1024), || StagedF32::stage(adversarial(1024))).expect("staging");
    assert!(big.is_mapped());
    assert_eq!(staged_bytes(), before_bytes + 4096);
    assert_eq!(staged_images(), before_images + 1);
}

#[test]
#[serial]
fn an_empty_image_is_owned_rather_than_mapped() {
    // Zero bytes is below any threshold, and a zero-length mapping is an
    // error on every platform — so the branch must not be taken.
    let staged = with_env(false, Some(0), || StagedF32::stage(Vec::new())).expect("staging");
    assert!(!staged.is_mapped());
    assert!(staged.is_empty());
}

/// Only the word `off`, in any case, turns staging off. `0`, `false` and
/// `on` are not spellings of it and leave staging on — a value that did
/// not take effect would otherwise be recorded as an arm that ran.
#[test]
fn only_the_off_word_disables_staging_in_any_case() {
    use super::staged::staging_disabled_by;
    assert!(!staging_disabled_by(None));
    for v in ["off", "OFF", " Off\n"] {
        assert!(staging_disabled_by(Some(v)), "{v:?}");
    }
    for v in ["", "0", "false", "on", "off please"] {
        assert!(!staging_disabled_by(Some(v)), "{v:?}");
    }
}

/// The threshold is a number or the default — never zero from a typo,
/// which would stage every norm vector, and never "everything owned".
#[test]
fn the_threshold_parses_a_number_and_falls_back_on_anything_else() {
    use super::staged::{min_bytes_from, DEFAULT_STAGE_MIN_BYTES};
    assert_eq!(min_bytes_from(None), DEFAULT_STAGE_MIN_BYTES);
    assert_eq!(min_bytes_from(Some("1024")), 1024);
    assert_eq!(min_bytes_from(Some(" 7 ")), 7);
    assert_eq!(min_bytes_from(Some("0")), 0);
    for v in ["", "lots", "-1", "1e6"] {
        assert_eq!(min_bytes_from(Some(v)), DEFAULT_STAGE_MIN_BYTES, "{v:?}");
    }
}

/// The arena is unlinked the moment it exists, so nothing outlives the
/// process — and the unlinked descriptor is still a writable file.
#[test]
fn the_arena_file_is_unlinked_the_moment_it_exists() {
    use super::staged::open_arena;
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let mut arena = open_arena(dir.path()).expect("a writable directory holds an arena");
    assert_eq!(
        std::fs::read_dir(dir.path()).unwrap().count(),
        0,
        "the arena must not be visible in the directory"
    );
    arena
        .file
        .write_all(&[0u8; 64])
        .expect("the unlinked descriptor is still writable");
    assert_eq!(arena.file.metadata().unwrap().len(), 64);
}

/// A directory that cannot hold the arena is a refusal that names the
/// two ways out, never a silent fall back to anonymous memory — the
/// exact failure this module exists to make impossible.
#[test]
fn an_arena_that_cannot_be_created_is_refused_naming_the_remedy() {
    use super::staged::{open_arena, STAGE_DIR_ENV, STAGE_ENV};
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("no-such-dir");
    let err = open_arena(&missing)
        .expect_err("a missing directory cannot hold the arena")
        .to_string();
    assert!(err.contains("could not be created"), "{err}");
    assert!(
        err.contains(STAGE_DIR_ENV) && err.contains(STAGE_ENV),
        "the refusal must name both ways out: {err}"
    );
    assert!(err.contains("no-such-dir"), "{err}");
}

/// Debug says WHERE an image lives and how big it is, never its values —
/// a 17408 x 5120 matrix printed elementwise is not a diagnostic.
#[test]
#[serial]
fn debug_names_the_residency_and_the_count_not_the_values() {
    let owned = StagedF32::from(vec![1.0f32, 2.0]);
    assert_eq!(format!("{owned:?}"), "StagedF32(Owned(2 f32))");
    let mapped = with_env(false, Some(0), || {
        StagedF32::stage(vec![1.0f32; 4]).unwrap()
    });
    assert!(mapped.is_mapped());
    assert_eq!(format!("{mapped:?}"), "StagedF32(Mapped(4 f32))");
}

/// Two arenas opened by ONE process have their own names: the pid is
/// shared, the counter is not. This is the naming contract that stops
/// concurrent opens colliding on `create_new`.
#[test]
fn arena_names_are_unique_within_one_process() {
    use super::staged::arena_path;
    let dir = tempfile::tempdir().unwrap();
    let first = arena_path(dir.path());
    let second = arena_path(dir.path());
    assert_ne!(
        first, second,
        "two arenas in one process must not share a file"
    );
    let pid = std::process::id().to_string();
    for path in [&first, &second] {
        let name = path.file_name().unwrap().to_string_lossy();
        assert!(name.starts_with("larql-f32-stage-"), "{name}");
        assert!(
            name.contains(&pid),
            "{name} names the process it belongs to"
        );
        assert!(name.ends_with(".bin"), "{name}");
    }
}

/// Eight opens released at once in one process all succeed and coexist,
/// each on its own unlinked inode. With a pid-only name most of these
/// found the sibling's file inside its create-to-unlink window and
/// refused with "File exists" — the failure two test suites running at
/// once reproduced on 2026-09-05.
#[test]
fn concurrent_arenas_in_one_process_coexist() {
    use super::staged::open_arena;
    use std::sync::{Arc, Barrier};
    const OPENS: usize = 8;
    let dir = Arc::new(tempfile::tempdir().unwrap());
    let go = Arc::new(Barrier::new(OPENS));
    let handles: Vec<_> = (0..OPENS)
        .map(|_| {
            let dir = Arc::clone(&dir);
            let go = Arc::clone(&go);
            std::thread::spawn(move || {
                go.wait();
                open_arena(dir.path()).map(|arena| arena.file.metadata().is_ok())
            })
        })
        .collect();
    let mut opened = 0;
    for handle in handles {
        match handle.join().expect("the thread ran") {
            Ok(alive) => {
                assert!(alive, "an opened arena keeps a live descriptor");
                opened += 1;
            }
            Err(e) => panic!("a concurrent open must not collide: {e}"),
        }
    }
    assert_eq!(opened, OPENS);
    // Every one unlinked itself: nothing named after this process is
    // left behind for a later process to trip on.
    let leftovers = std::fs::read_dir(dir.path()).unwrap().count();
    assert_eq!(leftovers, 0, "arenas leave no file behind");
}
