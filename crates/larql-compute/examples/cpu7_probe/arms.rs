//! The measured arms, and the controls that gate them.
//!
//! Every arm here is a RATIO measurement — arm C against arm one, `N`
//! against `N = 1` — which is why the controls are not optional. A ratio
//! is exactly the shape of measurement that survives a broken instrument
//! looking plausible, because both halves move together.

use std::hint::black_box;
use std::sync::Barrier;
use std::time::Instant;

use super::fixture::{activation_bytes_per_row, sdots_per_row, Bank, QuantAct};
use super::kernel::{row_reference, row_stationary, ActVectors, WEIGHT_BLOCK};

/// Repeats per cell. Best is the attainable estimate; the best/median
/// spread is the contention tell, and it is printed rather than hidden
/// so a run on a busy box is visible instead of silently wrong.
pub const REPEATS: usize = 5;

/// One measured cell.
pub struct Cell {
    pub label: String,
    pub best_ms: f64,
    pub median_ms: f64,
    /// Weight bytes streamed per repeat. DRAM class — the only quantity
    /// the 127 GB/s wall is comparable to.
    pub weight_bytes: u64,
    /// Activation and output bytes re-read or written per repeat. L1
    /// class, scales with `N`, and never summed with the above.
    pub transient_bytes: u64,
    /// `SDOT`s issued per repeat that scale with `N`, and those that do
    /// not. The second is the `w . 1` term a stationary kernel hoists.
    pub sdots_scaling: u64,
    pub sdots_fixed: u64,
}

impl Cell {
    /// Weight-only rate. Deliberately not "the" rate: see `weight_bytes`.
    pub fn weight_gbs(&self) -> f64 {
        self.weight_bytes as f64 / (self.best_ms / 1e3) / 1e9
    }
}

fn summarise(
    label: String,
    mut ms: Vec<f64>,
    weight_bytes: u64,
    transient_bytes: u64,
    sdots_scaling: u64,
    sdots_fixed: u64,
) -> Cell {
    ms.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    Cell {
        label,
        best_ms: ms[0],
        median_ms: ms[ms.len() / 2],
        weight_bytes,
        transient_bytes,
        sdots_scaling,
        sdots_fixed,
    }
}

/// Borrow the first `N` activations in the layout the kernel consumes.
fn view<const N: usize>(acts: &[QuantAct]) -> ActVectors<'_, N> {
    assert!(acts.len() >= N, "fixture holds fewer than N activations");
    ActVectors {
        codes: std::array::from_fn(|i| acts[i].codes.as_slice()),
        scales: std::array::from_fn(|i| acts[i].scales.as_slice()),
        mids: std::array::from_fn(|i| acts[i].mids.as_slice()),
    }
}

/// Contiguous row range worker `w` of `workers` owns in a `rows`-row matrix.
fn share(rows: usize, workers: usize, w: usize) -> (usize, usize) {
    let per = rows.div_ceil(workers);
    let start = (w * per).min(rows);
    (start, (start + per).min(rows) - start)
}

/// **CPU-7B.** One projection against `N` activations, weight-stationary.
///
/// `passes` holds STREAMED BYTES roughly constant across working-set
/// sizes. Without it the cache-resident control finishes in ~10 us — far
/// below the cost of the threads that run it — and reports thread spawn
/// rather than bandwidth, which would fire the discriminator against a
/// probe artifact instead of against a real issue-rate limit.
pub fn sweep_7b<const N: usize>(
    label: &str,
    bank: &Bank,
    acts: &[QuantAct],
    workers: usize,
    passes: usize,
) -> Cell {
    let act = view::<N>(acts);
    let in_dim = bank.mats[0].in_dim;
    let rows = bank.mats[0].rows;
    let per_row = in_dim / WEIGHT_BLOCK;
    let (sdot_per_vec, sdot_fixed) = sdots_per_row(in_dim);

    // Allocated and first-touched outside every timed region.
    let mut outs: Vec<Vec<f32>> = (0..workers)
        .map(|w| vec![0.0f32; share(rows, workers, w).1 * N])
        .collect();

    let mut samples = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let barrier = Barrier::new(workers + 1);
        let elapsed = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            for (w, out) in outs.iter_mut().enumerate() {
                let (barrier, act, bank) = (&barrier, &act, bank);
                handles.push(scope.spawn(move || {
                    let (start, count) = share(rows, workers, w);
                    barrier.wait();
                    for _ in 0..passes {
                        for m in &bank.mats {
                            let (codes, scales) = m.slab(start, count);
                            for r in 0..count {
                                let row = &codes[r * in_dim..(r + 1) * in_dim];
                                let ws = &scales[r * per_row..(r + 1) * per_row];
                                let mut o = [0.0f32; N];
                                // SAFETY: `dotprod` is checked once in `main`,
                                // and every slice is cut to this row's geometry.
                                unsafe { row_stationary::<N>(row, ws, act, in_dim, &mut o) };
                                out[r * N..(r + 1) * N].copy_from_slice(&o);
                            }
                        }
                    }
                    black_box(out.as_ptr());
                }));
            }
            barrier.wait();
            let start = Instant::now();
            for h in handles {
                h.join().expect("worker panicked");
            }
            start.elapsed()
        });
        samples.push(elapsed.as_secs_f64() * 1e3);
    }

    let sweep_rows = (bank.mats.len() * rows * passes) as u64;
    summarise(
        format!("{label} N={N}"),
        samples,
        (bank.bytes() * passes) as u64,
        sweep_rows * (activation_bytes_per_row(in_dim) * N + N * 4) as u64,
        sweep_rows * (sdot_per_vec * N) as u64,
        sweep_rows * sdot_fixed as u64,
    )
}

/// How arm 7A spreads its workers over two independent matrices.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Spread {
    /// One matrix at a time, every worker on it. The single-traversal
    /// reference the bands are read against.
    One,
    /// Both matrices, but strictly one after the other, with a phase
    /// barrier between them.
    ///
    /// The barrier is load-bearing. Without it worker 5 could still be on
    /// `gate` while worker 0 has moved to `up`, which is arm `Conc` wearing
    /// arm `Seq`'s label — the confound this arm exists to exclude.
    Sequential,
    /// Half the workers on each matrix, all live at once.
    Concurrent,
}

/// **CPU-7A.** Two independent projections sharing one activation.
///
/// Note what this deliberately does NOT use: the shipped `CpuExecutor`.
/// It collapses a projection reached from inside a worker to a single
/// thread, which is right for the shipped path and fatal here — the
/// obvious implementation of `Concurrent` would compare six workers
/// against two and report an artifact as a catastrophic loss. This builds
/// one flat list of row ranges and never nests.
pub fn arm_7a(
    label: &str,
    bank: &Bank,
    acts: &[QuantAct],
    workers: usize,
    passes: usize,
    spread: Spread,
) -> Cell {
    let act = view::<1>(acts);
    let in_dim = bank.mats[0].in_dim;
    let rows = bank.mats[0].rows;
    let per_row = in_dim / WEIGHT_BLOCK;
    let (sdot_per_vec, sdot_fixed) = sdots_per_row(in_dim);
    let pairs = bank.mats.len() / 2;
    // `One` traverses the first of each pair; the others traverse both.
    let mats_touched = if spread == Spread::One {
        pairs
    } else {
        pairs * 2
    };
    let half = workers / 2;

    let mut outs: Vec<Vec<f32>> = (0..workers).map(|_| vec![0.0f32; rows]).collect();
    let mut samples = Vec::with_capacity(REPEATS);

    for _ in 0..REPEATS {
        let start_gate = Barrier::new(workers + 1);
        let phase = Barrier::new(workers);
        let elapsed = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            for (w, out) in outs.iter_mut().enumerate() {
                let (start_gate, phase, act, bank) = (&start_gate, &phase, &act, bank);
                handles.push(scope.spawn(move || {
                    // Each arm's assignment, decided once, outside the clock.
                    let plan: Vec<(usize, usize, usize)> = (0..pairs)
                        .flat_map(|p| match spread {
                            Spread::One => {
                                let (s, c) = share(rows, workers, w);
                                vec![(2 * p, s, c)]
                            }
                            Spread::Sequential => {
                                let (s, c) = share(rows, workers, w);
                                vec![(2 * p, s, c), (2 * p + 1, s, c)]
                            }
                            Spread::Concurrent => {
                                // Workers `0..half` take `gate`, the rest `up`.
                                // Each side divides its matrix by ITS OWN count,
                                // not by a shared one: at an odd worker budget a
                                // shared denominator leaves a row range that no
                                // worker owns, and the arm would report a rate for
                                // bytes it never read.
                                let (m, k, denom) = if w < half {
                                    (2 * p, w, half)
                                } else {
                                    (2 * p + 1, w - half, workers - half)
                                };
                                let (s, c) = share(rows, denom, k);
                                vec![(m, s, c)]
                            }
                        })
                        .collect();
                    start_gate.wait();
                    for _ in 0..passes {
                        for (step, &(m, s, c)) in plan.iter().enumerate() {
                            let mat = &bank.mats[m];
                            let (codes, scales) = mat.slab(s, c);
                            for r in 0..c {
                                let row = &codes[r * in_dim..(r + 1) * in_dim];
                                let ws = &scales[r * per_row..(r + 1) * per_row];
                                let mut o = [0.0f32; 1];
                                // SAFETY: as in `sweep_7b`.
                                unsafe { row_stationary::<1>(row, ws, act, in_dim, &mut o) };
                                out[r] = o[0];
                            }
                            // Only `Sequential` synchronises between matrices.
                            if spread == Spread::Sequential && step + 1 < plan.len() {
                                phase.wait();
                            }
                        }
                    }
                    black_box(out.as_ptr());
                }));
            }
            start_gate.wait();
            let t = Instant::now();
            for h in handles {
                h.join().expect("worker panicked");
            }
            t.elapsed()
        });
        samples.push(elapsed.as_secs_f64() * 1e3);
    }

    let bytes = bank.mats[..mats_touched.min(bank.mats.len())]
        .iter()
        .map(|m| m.bytes())
        .sum::<usize>();
    let swept = (mats_touched * rows * passes) as u64;
    summarise(
        label.to_string(),
        samples,
        (bytes * passes) as u64,
        swept * (activation_bytes_per_row(in_dim) + 4) as u64,
        swept * sdot_per_vec as u64,
        swept * sdot_fixed as u64,
    )
}

/// The outcome of one control, with the evidence it was actually able to
/// fail. A control that has never fired is decoration.
pub struct Control {
    pub name: &'static str,
    pub passed: bool,
    pub detail: String,
}

/// **C3.** The stationary kernel against its own definition, and against
/// itself at `N = 1`.
///
/// Three separate claims, because two of them can pass while the third
/// fails and the failures mean different things:
///
///   - `N = 1` matches an `f64` reference decoded from the FORMAT, so a
///     transcription that dropped a term cannot pass;
///   - `N = 8` is BIT-IDENTICAL to eight `N = 1` calls, so the arms being
///     timed compute the same thing;
///   - perturbing one activation MAKES THAT CHECK FAIL, so the identity
///     above is evidence rather than a tautology about a check that
///     cannot distinguish anything.
pub fn control_c3(bank: &Bank, acts: &[QuantAct]) -> Vec<Control> {
    const N: usize = 8;
    let m = &bank.mats[0];
    let in_dim = m.in_dim;
    let per_row = in_dim / WEIGHT_BLOCK;
    let rows_checked = 64.min(m.rows);
    let act = view::<N>(acts);

    let mut worst_rel = 0.0f64;
    let mut identical = true;
    for r in 0..rows_checked {
        let row = &m.codes[r * in_dim..(r + 1) * in_dim];
        let ws = &m.scales[r * per_row..(r + 1) * per_row];
        let mut wide = [0.0f32; N];
        // SAFETY: `dotprod` is checked in `main`.
        unsafe { row_stationary::<N>(row, ws, &act, in_dim, &mut wide) };
        for (n, a) in acts.iter().enumerate().take(N) {
            let one = view::<1>(&acts[n..n + 1]);
            let mut narrow = [0.0f32; 1];
            // SAFETY: as above.
            unsafe { row_stationary::<1>(row, ws, &one, in_dim, &mut narrow) };
            identical &= narrow[0].to_bits() == wide[n].to_bits();
            let reference = row_reference(row, ws, &a.codes, &a.scales, &a.mids, in_dim);
            if reference.abs() > 0.0 {
                worst_rel = worst_rel.max(((narrow[0] as f64 - reference) / reference).abs());
            }
        }
    }

    // The planted violation. One code changed in one vector must move that
    // vector's result and no other.
    let mut tampered = acts[3].codes.clone();
    tampered[in_dim - 1] = tampered[in_dim - 1].wrapping_add(17);
    let mut codes: [&[i8]; N] = std::array::from_fn(|i| acts[i].codes.as_slice());
    codes[3] = &tampered;
    let planted = ActVectors::<N> {
        codes,
        scales: std::array::from_fn(|i| acts[i].scales.as_slice()),
        mids: std::array::from_fn(|i| acts[i].mids.as_slice()),
    };
    let row = &m.codes[..in_dim];
    let ws = &m.scales[..per_row];
    let (mut clean, mut dirty) = ([0.0f32; N], [0.0f32; N]);
    // SAFETY: as above.
    unsafe {
        row_stationary::<N>(row, ws, &act, in_dim, &mut clean);
        row_stationary::<N>(row, ws, &planted, in_dim, &mut dirty);
    }
    let moved = clean[3].to_bits() != dirty[3].to_bits();
    let others_still = (0..N)
        .filter(|n| *n != 3)
        .all(|n| clean[n].to_bits() == dirty[n].to_bits());

    const REF_TOLERANCE: f64 = 1e-4;
    vec![
        Control {
            name: "C3a  N=1 vs f64 reference decoded from the format",
            passed: worst_rel < REF_TOLERANCE,
            detail: format!("worst relative error {worst_rel:.3e} against {REF_TOLERANCE:.0e}"),
        },
        Control {
            name: "C3b  N=8 bit-identical to eight N=1 calls",
            passed: identical,
            detail: format!("{rows_checked} rows x {N} vectors compared by bit pattern"),
        },
        Control {
            name: "C3c  planted violation fires, and only where planted",
            passed: moved && others_still,
            detail: format!(
                "perturbed vector moved: {moved}; other seven unchanged: {others_still}"
            ),
        },
    ]
}
