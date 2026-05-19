//! Auto-calibration: benchmark CPU vs Metal to find the FLOP threshold
//! where GPU dispatch starts beating CPU BLAS.

use ndarray::Array2;
use std::time::Instant;

use super::buffers::BufferCache;
use super::f32_ops::F32Ops;
use metal::CommandQueue;

/// Conservative default before calibration runs.
pub const DEFAULT_FLOP_THRESHOLD: usize = 500_000_000;

/// Absolute floor: never dispatch to GPU below this.
pub const MIN_FLOP_FLOOR: usize = 100_000;

/// Run calibration and return the optimal FLOP threshold.
pub fn calibrate(f32_ops: &F32Ops, queue: &CommandQueue, bufs: &BufferCache) -> usize {
    let test_cases: &[(usize, usize, usize)] = &[
        (6, 256, 256),    // ~800K FLOPs
        (6, 2560, 512),   // ~15M FLOPs
        (6, 2560, 2560),  // ~79M FLOPs — attention projection
        (6, 10240, 2560), // ~315M FLOPs — FFN gate/up
    ];

    let mut best = DEFAULT_FLOP_THRESHOLD;

    for &(m, n, k) in test_cases {
        let flops = 2 * m * n * k;
        let a = synth_matrix(m, k, 42);
        let b = synth_matrix(n, k, 43);

        let a_slice = a.as_slice().unwrap();
        let b_slice = b.as_slice().unwrap();

        // Warm Metal buffer cache
        let _ = f32_ops.dispatch_transb(queue, bufs, a_slice, b_slice, m, n, k);

        let cpu_us = bench_median(5, || {
            let _ = a.dot(&b.t());
        });
        let metal_us = bench_median(5, || {
            let _ = f32_ops.dispatch_transb(queue, bufs, a_slice, b_slice, m, n, k);
        });

        if metal_us < cpu_us {
            best = best.min(flops);
        }
    }

    best
}

fn synth_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut state = seed;
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    Array2::from_shape_vec((rows, cols), data)
        .expect("synth_matrix: data length matches (rows, cols) by construction")
}

fn bench_median<F: FnMut()>(n: usize, mut f: F) -> u64 {
    let mut times = Vec::with_capacity(n);
    for _ in 0..n {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed().as_micros() as u64);
    }
    times.sort_unstable();
    times[n / 2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalBackend;

    /// `calibrate()` returns a threshold inside the legal envelope:
    /// `[MIN_FLOP_FLOOR, DEFAULT_FLOP_THRESHOLD]` (inclusive on the
    /// upper bound — `best` starts at default and only goes down via
    /// `best.min(flops)`, so the worst case is "Metal never beats CPU"
    /// and we keep the conservative default).
    #[test]
    fn calibrate_returns_threshold_in_legal_envelope() {
        let Some(metal) = MetalBackend::new() else {
            return;
        };
        // Use the inherent helpers to access the private fields.
        // `f32_ops` and the buffer cache are the only inputs `calibrate()` needs.
        // Rather than reach into private state, just call `metal.calibrate()`
        // and read back via the public `flop_threshold()` accessor.
        metal.calibrate();
        let t = metal.flop_threshold();
        assert!(
            t >= MIN_FLOP_FLOOR,
            "calibrated threshold {t} below MIN_FLOP_FLOOR={MIN_FLOP_FLOOR}"
        );
        assert!(
            t <= DEFAULT_FLOP_THRESHOLD,
            "calibrated threshold {t} above DEFAULT_FLOP_THRESHOLD={DEFAULT_FLOP_THRESHOLD}"
        );
    }

    /// `set_flop_threshold` clamps to `MIN_FLOP_FLOOR`. Pin the
    /// invariant that "no caller can set a threshold below the floor"
    /// — small dispatches dominated by Metal command-buffer overhead
    /// would benchmark slower than CPU and the auto-router would
    /// thrash.
    #[test]
    fn set_flop_threshold_clamps_to_min_floor() {
        let Some(metal) = MetalBackend::new() else {
            return;
        };
        metal.set_flop_threshold(0);
        assert_eq!(metal.flop_threshold(), MIN_FLOP_FLOOR);
        metal.set_flop_threshold(MIN_FLOP_FLOOR / 2);
        assert_eq!(metal.flop_threshold(), MIN_FLOP_FLOOR);
        metal.set_flop_threshold(MIN_FLOP_FLOOR * 100);
        assert_eq!(metal.flop_threshold(), MIN_FLOP_FLOOR * 100);
    }

    // Note: calibration isn't deterministic across runs — at small
    // shapes Metal can win one run and lose the next (timing noise on
    // shared-system CPU/GPU contention). Repeatability *isn't* a
    // contract of `calibrate()`. The legal-envelope test above is
    // enough to catch real regressions; the worst case is the
    // conservative default kicks in.
}
