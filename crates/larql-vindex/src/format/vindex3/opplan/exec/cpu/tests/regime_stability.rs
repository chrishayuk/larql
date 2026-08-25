//! CPU-PERF-2: why does real Q8 x F32 drift when BF16 does not?
//!
//! CPU-PERF-1 found two things that want one explanation. The isolated
//! harness reproduces real BF16 projection to +0.7% and under-predicts
//! real Q8 by 7.9%; and across three quiet, warm, interleaved runs BF16
//! stays flat (453 / 459 / 449 ms) while Q8 climbs monotonically (379 /
//! 401 / 425).
//!
//! The hypothesis: **it is machine STATE, and only the conversion-bound
//! kernel can see it.** A real Q8 run quantises tens of GB at load and
//! then immediately runs a kernel whose limit is integer-to-float
//! conversion throughput. If core frequency is depressed even modestly,
//! that kernel slows nearly in proportion. BF16 sits at the ~120 GB/s
//! memory wall, where a lower clock changes very little.
//!
//! Three kernels now exist that occupy three regimes, which makes them
//! three thermometers rather than three benchmarks:
//!
//! ```text
//!   bf16 x f32     memory-bound       121.7 GB/s
//!   q8   x f32     conversion-bound    83.4 GB/s
//!   q8   x q8      memory-bound       118.0 GB/s   (SDOT)
//! ```
//!
//! and the signatures separate the candidates cleanly:
//!
//! ```text
//!   frequency / thermal   only q8 x f32 moves, and recovers when idle
//!   memory or system      all three move together
//!   neither               nothing moves here, and the problem is in
//!                         real execution integration rather than the
//!                         machine
//! ```
//!
//! No reboot first: a reboot would erase the state being explained.
//!
//! ```text
//! QW_REGIME=1 cargo test --release exec::cpu::tests::regime_stability -- --nocapture
//! ```

use std::time::{Duration, Instant};

use super::super::executor::CpuExecutor;
use super::super::kernels::{FusedBf16, FusedQ8};
use super::super::projector::WeightRows;
use crate::format::vindex3::fixtures::lcg_values;

const BLOCK: usize = 64;
/// `mlp.gate_proj` — the shape a token runs 128 times, and big enough
/// that all three kernels are in their steady regime on it.
const OUT: usize = 17408;
const IN: usize = 5120;

/// Seconds of idle between the hot and the recovered reading.
///
/// Long enough for a depressed clock to come back, short enough that the
/// probe stays runnable. If the recovery is real but slower than this,
/// the third row reads part-way and the shape of the result still shows
/// it.
const COOLDOWN: Duration = Duration::from_secs(45);

/// Bytes of f32 quantised in the burst, standing in for the load path.
///
/// A real Q8 open quantises ~51 GB. Four is enough to occupy every core
/// for seconds doing exactly that arithmetic, without making the probe
/// a memory experiment in its own right.
const BURST_BYTES: usize = 4 << 30;

fn narrow(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn quantise(weights: &[f32], in_dim: usize) -> (Vec<i8>, Vec<f32>) {
    let per_row = in_dim.div_ceil(BLOCK);
    let rows = weights.len() / in_dim;
    let mut codes = vec![0i8; weights.len()];
    let mut scales = vec![0.0f32; rows * per_row];
    for r in 0..rows {
        for b in 0..per_row {
            let lo = r * in_dim + b * BLOCK;
            let hi = (lo + BLOCK).min((r + 1) * in_dim);
            let peak = weights[lo..hi].iter().fold(0.0f32, |m, w| m.max(w.abs()));
            let scale = if peak > 0.0 { peak / 127.0 } else { 1.0 };
            scales[r * per_row + b] = scale;
            for i in lo..hi {
                codes[i] = (weights[i] / scale).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    (codes, scales)
}

fn quantise_activation(x: &[f32], out: &mut [i8]) -> f32 {
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = if peak > 0.0 { peak / 127.0 } else { 1.0 };
    let inv = 1.0 / scale;
    for (dst, v) in out.iter_mut().zip(x) {
        *dst = (v * inv).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn sdot_row(codes: &[i8], scales: &[f32], qx: &[i8]) -> f32 {
    use std::arch::aarch64::*;
    let mut acc = 0.0f32;
    for (b, scale) in scales.iter().enumerate() {
        let (lo, hi) = (b * BLOCK, ((b + 1) * BLOCK).min(IN));
        let mut lanes = vdupq_n_s32(0);
        let mut i = lo;
        while i + 16 <= hi {
            lanes = vdotq_s32(
                lanes,
                vld1q_s8(codes.as_ptr().add(i)),
                vld1q_s8(qx.as_ptr().add(i)),
            );
            i += 16;
        }
        acc += scale * vaddvq_s32(lanes) as f32;
    }
    acc
}

/// The three thermometers, each read `rounds` times, best kept.
///
/// Best rather than mean: this asks what the machine is CAPABLE of at
/// this moment, and a mean would fold in whatever else the scheduler did
/// during the sample.
fn read_all(exec: &CpuExecutor, w: &Fixtures, rounds: usize) -> [f64; 3] {
    let mut best = [f64::INFINITY; 3];
    for _ in 0..rounds {
        let t = Instant::now();
        std::hint::black_box(exec.project(&FusedBf16, WeightRows::Bf16(&w.bf16), &w.x, OUT)[0]);
        best[0] = best[0].min(t.elapsed().as_secs_f64());

        let t = Instant::now();
        std::hint::black_box(
            exec.project(
                &FusedQ8,
                WeightRows::Q8 {
                    codes: &w.codes,
                    scales: &w.scales,
                    sums: &[],
                    block: BLOCK,
                },
                &w.x,
                OUT,
            )[0],
        );
        best[1] = best[1].min(t.elapsed().as_secs_f64());

        let t = Instant::now();
        w.sdot_once();
        best[2] = best[2].min(t.elapsed().as_secs_f64());
    }
    best
}

struct Fixtures {
    bf16: Vec<u16>,
    codes: Vec<i8>,
    scales: Vec<f32>,
    x: Vec<f32>,
    // Read only by `sdot_once`'s `#[cfg(target_arch = "aarch64")]` arm
    // below — real usage on the architecture this bench actually
    // measures, not dead weight; the non-aarch64 arm is a `*cell = 0.0`
    // no-op by design (this whole bench measures nothing off-target).
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    qx: Vec<i8>,
    workers: usize,
    pool: rayon::ThreadPool,
}

impl Fixtures {
    fn sdot_once(&self) {
        use rayon::prelude::*;
        let per = OUT.div_ceil(self.workers);
        // Same story as the `qx` field above: read only on aarch64.
        #[cfg_attr(not(target_arch = "aarch64"), allow(unused_variables))]
        let per_row = IN.div_ceil(BLOCK);
        let mut out = vec![0.0f32; OUT];
        self.pool.install(|| {
            out.par_chunks_mut(per).enumerate().for_each(|(i, slot)| {
                let start = i * per;
                for (o, cell) in slot.iter_mut().enumerate() {
                    // Read only by the aarch64 arm below.
                    #[cfg_attr(not(target_arch = "aarch64"), allow(unused_variables))]
                    let r = start + o;
                    #[cfg(target_arch = "aarch64")]
                    {
                        // SAFETY: `dotprod` is checked by the caller
                        // before this probe runs at all.
                        *cell = unsafe {
                            sdot_row(
                                &self.codes[r * IN..(r + 1) * IN],
                                &self.scales[r * per_row..(r + 1) * per_row],
                                &self.qx,
                            )
                        };
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        *cell = 0.0;
                    }
                }
            });
        });
        std::hint::black_box(&out);
    }
}

/// Quantise a large slab, which is what a Q8 model open spends its time
/// doing — and the suspected cause of the state the kernels then run in.
fn load_burst() -> Duration {
    let values = BURST_BYTES / 4;
    let chunk = OUT * IN;
    let t = Instant::now();
    let mut done = 0;
    while done < values {
        let src = lcg_values(chunk, 7);
        std::hint::black_box(quantise(&src, IN));
        done += chunk;
    }
    t.elapsed()
}

#[test]
fn which_regime_moves_when_the_machine_is_worked() {
    if std::env::var("QW_REGIME").is_err() {
        eprintln!("SKIP regime_stability: set QW_REGIME=1");
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if !std::arch::is_aarch64_feature_detected!("dotprod") {
        println!("\n  SDOT unavailable; the third thermometer would be a scalar loop.\n");
        return;
    }
    let exec = CpuExecutor::new().unwrap();
    let workers = exec.workers();
    let f32w = lcg_values(OUT * IN, 11);
    let (codes, scales) = quantise(&f32w, IN);
    let x = lcg_values(IN, 22);
    let mut qx = vec![0i8; IN];
    quantise_activation(&x, &mut qx);
    let w = Fixtures {
        bf16: f32w.iter().map(|v| narrow(*v)).collect(),
        codes,
        scales,
        x,
        qx,
        workers,
        pool: rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap(),
    };

    println!("\n  Three regimes as thermometers, {workers} workers, {OUT}x{IN}.\n");
    let names = [
        "bf16 x f32  (memory)",
        "q8 x f32    (convert)",
        "q8 x q8     (memory)",
    ];

    let cold = read_all(&exec, &w, 12);
    let burst = load_burst();
    let hot = read_all(&exec, &w, 12);
    println!(
        "  quantised {} GB in {:.1} s to make the state\n",
        BURST_BYTES >> 30,
        burst.as_secs_f64()
    );
    std::thread::sleep(COOLDOWN);
    let rested = read_all(&exec, &w, 12);

    println!(
        "  {:<24} {:>9} {:>9} {:>9} {:>10} {:>10}",
        "kernel", "cold ms", "hot ms", "rested", "hot/cold", "rested/cold"
    );
    for i in 0..3 {
        println!(
            "  {:<24} {:>9.3} {:>9.3} {:>9.3} {:>9.2}x {:>10.2}x",
            names[i],
            cold[i] * 1e3,
            hot[i] * 1e3,
            rested[i] * 1e3,
            hot[i] / cold[i],
            rested[i] / cold[i],
        );
    }

    let moved = |i: usize| (hot[i] / cold[i] - 1.0).abs() > 0.04;
    println!(
        "\n  reading: {}",
        match (moved(0), moved(1), moved(2)) {
            (false, true, false) =>
                "ONLY the conversion-bound kernel moved — machine STATE, and the \
                 memory-bound kernels cannot see it",
            (true, true, true) => "ALL THREE moved — a system-wide effect, not a frequency one",
            (false, false, false) =>
                "NOTHING moved — the drift is in real execution integration, not the machine",
            _ => "a mixed signature; read the rows rather than this line",
        }
    );
    println!();
}
