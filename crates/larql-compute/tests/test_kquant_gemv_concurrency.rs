//! Concurrency and geometry stress for `q8_0_gemv`.
//!
//! The kernel's numerical contract is exact: every row is `acc += ((q as f32) * d) * x`
//! in ascending column order. These tests hold it to that contract under the conditions
//! the schedule changes: pool sizes from one thread to far more than the machine has,
//! concurrent callers (which take the fork-join fallback), callers nested inside rayon
//! tasks, calls back to back and calls spaced out, and a saturated pool. A watchdog aborts
//! the process if any test stops making progress, so a deadlock fails instead of hanging.

use larql_compute::cpu::kquant_gemv::{q8_0_gemv, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS};
use larql_models::quant::half::f16_to_f32;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Duration;

/// A test that makes no progress for this long is a deadlock.
const WATCHDOG: Duration = Duration::from_secs(180);

/// Pool sizes: serial, small, odd, a typical machine, and heavily oversubscribed.
const POOLS: [usize; 6] = [1, 2, 3, 5, 16, 64];

/// `(n, k)` shapes that reach the parallel path (more rows than one chunk) with ragged
/// tails, plus shapes that stay serial or leave a remainder outside the 16-row groups.
const SHAPES: [(usize, usize); 8] = [
    (1, 32),
    (15, 64),
    (17, 4096),
    (203, 4096),
    (1000, 1024),
    (3001, 96),
    (5000, 32),
    (4099, 32),
];

/// The saturated-pool test occupies every global-pool worker with a spinner. A gemv call from
/// another test that holds the dispatch slot would send this test's caller to the fork-join
/// fallback, which needs a free global worker, so the caller would wait on its own spinners and
/// the spinners would wait on the caller. Every test here takes this lock shared, except the
/// saturated one, which takes it exclusively and so has the process to itself.
static PROCESS: RwLock<()> = RwLock::new(());

fn shared() -> RwLockReadGuard<'static, ()> {
    PROCESS.read().unwrap_or_else(|e| e.into_inner())
}

fn exclusive() -> RwLockWriteGuard<'static, ()> {
    PROCESS.write().unwrap_or_else(|e| e.into_inner())
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Random but finite inputs: any code byte (including -128), finite f16 scales (including
/// subnormals and signed zero), and activations that include subnormals and signed zeros.
fn pack(n: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let mut rng = Rng(seed | 1);
    let mut blocks = Vec::with_capacity(n * (k / Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES);
    for _ in 0..n * (k / Q8_0_BLOCK_ELEMS) {
        let mut h = (rng.next() & 0xffff) as u16;
        if (h >> 10) & 0x1f == 0x1f {
            h &= !(1 << 10); // not infinity or NaN: outside the kernel's contract
        }
        blocks.extend_from_slice(&h.to_le_bytes());
        blocks.extend((0..Q8_0_BLOCK_ELEMS).map(|_| rng.next() as u8));
    }
    let x = (0..k)
        .map(|i| match rng.next() % 16 {
            0 => 0.0,
            1 => -0.0,
            2 => f32::from_bits(1 + (rng.next() as u32 & 0x7f_ffff)), // subnormal
            _ => ((rng.next() % 2001) as f32 - 1000.0) / 7.0 + (i % 3) as f32,
        })
        .collect();
    (blocks, x)
}

/// The declared program, a row at a time, serially.
fn reference(blocks: &[u8], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    let per_row = k / Q8_0_BLOCK_ELEMS;
    (0..n)
        .map(|row| {
            let mut acc = 0.0f32;
            for b in 0..per_row {
                let base = (row * per_row + b) * Q8_0_BLOCK_BYTES;
                let d = f16_to_f32(u16::from_le_bytes([blocks[base], blocks[base + 1]]));
                for j in 0..Q8_0_BLOCK_ELEMS {
                    acc += ((blocks[base + 2 + j] as i8 as f32) * d) * x[b * Q8_0_BLOCK_ELEMS + j];
                }
            }
            acc
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

fn assert_exact(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = (0..got.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
        panic!(
            "{what}: row {i} differs: {:#010x} vs {:#010x}",
            got[i].to_bits(),
            want[i].to_bits()
        );
    }
}

/// Aborts the process if the test does not finish. Dropping the guard disarms it.
struct Watchdog(Arc<AtomicBool>);

fn watchdog(name: &'static str) -> Watchdog {
    let done = Arc::new(AtomicBool::new(false));
    let flag = done.clone();
    std::thread::spawn(move || {
        let start = std::time::Instant::now();
        while start.elapsed() < WATCHDOG {
            if flag.load(Ordering::Relaxed) {
                return;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        eprintln!("DEADLOCK: {name} made no progress in {WATCHDOG:?}");
        std::process::abort();
    });
    Watchdog(done)
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

#[test]
fn matches_the_declared_program_at_every_pool_size_and_shape() {
    let _g = shared();
    let _w = watchdog("pool sweep");
    for &(n, k) in &SHAPES {
        let (blocks, x) = pack(n, k, (n * 31 + k) as u64);
        let want = reference(&blocks, &x, n, k);
        for &threads in &POOLS {
            let got = pool(threads)
                .install(|| q8_0_gemv(&blocks, &x, n, k))
                .unwrap();
            assert_exact(&got, &want, &format!("n={n} k={k} threads={threads}"));
        }
    }
}

#[test]
fn concurrent_callers_agree_with_the_reference() {
    let _g = shared();
    let _w = watchdog("concurrent callers");
    let (n, k) = (4099, 96);
    let (blocks, x) = pack(n, k, 7);
    let want = Arc::new(reference(&blocks, &x, n, k));
    let (blocks, x) = (Arc::new(blocks), Arc::new(x));
    for threads in [2, 4, 16] {
        let p = Arc::new(pool(threads));
        let callers: Vec<_> = (0..8)
            .map(|_| {
                let (p, blocks, x, want) = (p.clone(), blocks.clone(), x.clone(), want.clone());
                std::thread::spawn(move || {
                    for i in 0..60 {
                        let got = p.install(|| q8_0_gemv(&blocks, &x, n, k)).unwrap();
                        assert_exact(
                            &got,
                            &want,
                            &format!("concurrent threads={threads} call={i}"),
                        );
                    }
                })
            })
            .collect();
        for c in callers {
            c.join().unwrap();
        }
    }
}

#[test]
fn calls_nested_inside_rayon_tasks_complete_and_agree() {
    let _g = shared();
    let _w = watchdog("nested");
    let (n, k) = (3001, 96);
    let (blocks, x) = pack(n, k, 11);
    let want = reference(&blocks, &x, n, k);
    for threads in [2, 4, 16, 64] {
        pool(threads).install(|| {
            (0..96).into_par_iter().for_each(|i| {
                let got = q8_0_gemv(&blocks, &x, n, k).unwrap();
                assert_exact(&got, &want, &format!("nested threads={threads} task={i}"));
            });
        });
    }
}

#[test]
fn back_to_back_and_spaced_calls_agree() {
    let _g = shared();
    let _w = watchdog("back to back");
    let (n, k) = (5000, 32);
    let (blocks, x) = pack(n, k, 13);
    let want = reference(&blocks, &x, n, k);
    pool(8).install(|| {
        for i in 0..3000 {
            let got = q8_0_gemv(&blocks, &x, n, k).unwrap();
            assert_exact(&got, &want, &format!("back to back call={i}"));
        }
        // Spaced past the helpers' linger window, so each call wakes the pool again.
        for i in 0..150 {
            std::thread::sleep(Duration::from_micros(300));
            let got = q8_0_gemv(&blocks, &x, n, k).unwrap();
            assert_exact(&got, &want, &format!("spaced call={i}"));
        }
    });
}

#[test]
fn a_saturated_pool_does_not_stop_the_caller() {
    let _g = exclusive();
    let _w = watchdog("saturated pool");
    let (n, k) = (5000, 32);
    let (blocks, x) = pack(n, k, 17);
    let want = reference(&blocks, &x, n, k);
    let stop = Arc::new(AtomicBool::new(false));
    // Every global-pool thread is busy with unrelated work, so no helper can arrive. The caller is a plain thread,
    // outside the pool (`install` would itself wait for a free worker), and must finish alone.
    for _ in 0..rayon::current_num_threads() * 2 {
        let stop = stop.clone();
        rayon::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                std::hint::spin_loop();
            }
        });
    }
    for i in 0..50 {
        let got = q8_0_gemv(&blocks, &x, n, k).unwrap();
        assert_exact(&got, &want, &format!("saturated call={i}"));
    }
    stop.store(true, Ordering::Relaxed);
}

#[test]
fn a_geometry_that_does_not_describe_the_pack_is_refused() {
    let _g = shared();
    let (blocks, x) = pack(40, 64, 19);
    assert!(q8_0_gemv(&blocks, &x, 40, 64).is_some());
    assert!(
        q8_0_gemv(&blocks[1..], &x, 40, 64).is_none(),
        "short stream"
    );
    assert!(q8_0_gemv(&blocks, &x, 41, 64).is_none(), "too many rows");
    assert!(
        q8_0_gemv(&blocks, &x[1..], 40, 64).is_none(),
        "short activation"
    );
    assert!(q8_0_gemv(&blocks, &x, 40, 0).is_none(), "k = 0");
    assert!(
        q8_0_gemv(&blocks, &x, 40, 63).is_none(),
        "k off the block grid"
    );
    assert!(
        q8_0_gemv(&[], &x, usize::MAX / 2, 64).is_none(),
        "row count whose byte size overflows"
    );
    assert_eq!(
        q8_0_gemv(&[], &x, 0, 64).map(|v| v.len()),
        Some(0),
        "no rows"
    );
    assert_eq!(
        bits(&q8_0_gemv(&blocks, &x, 40, 64).unwrap()),
        bits(&reference(&blocks, &x, 40, 64))
    );
}
