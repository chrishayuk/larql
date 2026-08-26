//! Weight-stationary projection: ONE weight traversal, `N` activations.
//!
//! CPU-7B measured, on a 5120x5120 Q8[64] x asym-Q8[16] sweep at the
//! executor's own worker count:
//!
//! ```text
//! N=2  1.02x   N=4  1.27x   N=8  2.41x     (total cost, against N=1)
//! ```
//!
//! and showed the mechanism rather than only the outcome: the DRAM and
//! cache-resident arms CONVERGE at N=4 (98.8 vs 98.6 GB/s), so below that
//! knee a batch-1 projection leaves arithmetic capacity idle waiting on
//! memory. This module makes that available to a layer.
//!
//! # The invariant, which is the whole point
//!
//! > **Same numerical computation, different physical schedule.**
//!
//! Only the WEIGHT LOAD is shared. Each position keeps
//!
//!   - its own activation codes, scales and midpoints, quantised
//!     INDEPENDENTLY from its own block statistics;
//!   - its own accumulator;
//!   - the same `SDOT` sequence, in the same order, that
//!     [`super::integer::Q8xQ8`] would have issued for it alone.
//!
//! Two ways to break that, both of which would make this a new
//! REPRESENTATION rather than a new schedule:
//!
//!   - **a joint activation scale** across the `N` vectors. Convenient,
//!     cheaper, and a different quantisation — the numbers would move and
//!     the reason would be unattributable to either change.
//!   - **a cross-position reduction.** The accumulators are independent;
//!     nothing is summed across positions.
//!
//! Neither happens, so every column here is BIT-IDENTICAL to the
//! single-vector kernel. The parity tests assert that rather than this
//! paragraph arguing for it.
//!
//! The `w . 1` term IS hoisted out of the position loop, because it
//! depends only on the weights and is the same `f32` every position would
//! have computed for itself. CPU-7B priced it at 0.1 G SDOTs against
//! 0.1 -> 0.9 G that scale with `N`, so it is not where the amortisation
//! comes from and must not be reported as if it were.

use std::sync::atomic::{AtomicU8, Ordering};

use super::arithmetic::ScaleSpan;
use super::integer::{activation_code, activation_scaling, bit_identical_only, ActivationCode};
use super::ledger::{current_site, Site};
use super::projector::WeightRows;

// `has_dotprod` and the quantisers the sweep uses are reached only from
// the NEON arm. An un-gated import of an `aarch64`-only symbol is how this
// crate has broken on x86 twice, and it is invisible on the machine that
// writes it — so the cross-target check is the gate, not review.
#[cfg(target_arch = "aarch64")]
use super::integer::{
    has_dotprod, quantise_activation_asymmetric, quantise_activation_blocked, PER_WEIGHT_B16,
    SDOT_LANES,
};

/// Position counts with an unrolled sweep.
///
/// Two and four because those are the counts CPU-7B priced and CPU-7C
/// tests; `N = 8` amortised barely better than four (0.30x against 0.32x
/// per vector) and a real layer has no use for it yet. An unlisted `N`
/// falls to the looping default, which is correct and is not this.
#[cfg(target_arch = "aarch64")]
const UNROLLED: [usize; 2] = [2, 4];

/// Disables the stationary sweep, leaving everything else untouched.
pub const STATIONARY_ENV: &str = "LARQL_CPU_STATIONARY";

/// Tri-state, so an explicit override and the environment default can
/// both exist without one silently winning.
const UNSET: u8 = 0;
const OFF: u8 = 1;
const ON: u8 = 2;
static SWITCH: AtomicU8 = AtomicU8::new(UNSET);

/// Whether this process may share weight traversals across positions.
///
/// **CPU-7C's arm switch, and it exists because nothing else isolates the
/// variable.** Turning it off leaves the arithmetic, the kernel, the
/// worker policy and the activation quantisation exactly as they were and
/// changes only WHERE the weight is read from — so a timing difference
/// between the two settings is attributable to the schedule and to
/// nothing else. Comparing against a different arithmetic arm, or against
/// `bit_identical_only`, would move two things at once.
///
/// Default ON: the sweep is bit-identical to the loop it replaces, so
/// there is no correctness reason to make it opt-in, and an experiment
/// switch that also changed the default would make every other
/// measurement in the repo a different measurement. Only `0` and `off`
/// disable it; a typo leaves the sweep in force rather than silently
/// disabling it and being read as a regression.
///
/// Deliberately NOT a `OnceLock`. Arms B and C differ only by this flag,
/// and a once-per-process read would force them into separate processes —
/// separate 51 GB loads, and a time gap between the two halves of a
/// RATIO. CPU-7B showed absolute rates on this box moving 5% between runs
/// while ratios held to 0.04; one process, interleaved, is how that stays
/// true here.
pub fn enabled() -> bool {
    match SWITCH.load(Ordering::Relaxed) {
        OFF => false,
        ON => true,
        _ => {
            let on = !matches!(
                std::env::var(STATIONARY_ENV).ok().as_deref().map(str::trim),
                Some("0") | Some("off")
            );
            SWITCH.store(if on { ON } else { OFF }, Ordering::Relaxed);
            on
        }
    }
}

/// Which operator classes may share traversals. All, by default.
static CLASSES: AtomicU8 = AtomicU8::new(u8::MAX);

/// Whether the class the calling thread is inside may group.
///
/// CPU-7C2 arm `E`: raising the FFN surface makes TWO tranches groupable
/// at once — the recurrent one C1 already proved, and the new FFN one. A
/// single on/off switch can only measure their sum, leaving the FFN
/// contribution to be INFERRED from a mechanism established on a
/// different (fan-out-collapsed) schedule. Selecting by class measures it
/// directly instead, and costs one atomic load.
fn class_enabled() -> bool {
    CLASSES.load(Ordering::Relaxed) & current_site().bit() != 0
}

/// Whether `site`'s bit is set in the class mask.
///
/// Exposed so a harness can PROVE what an arm resolved to before printing
/// a number beside its letter. Several plausible timings in this
/// programme would have been entirely wrong if an arm had silently fallen
/// back, and an arm that cannot state its own physical configuration is
/// one nobody can adjudicate.
pub fn class_enabled_for(site: Site) -> bool {
    CLASSES.load(Ordering::Relaxed) & site.bit() != 0
}

/// Restrict grouping to a set of operator classes.
///
/// Harness-only, for the same reason [`set_enabled`] is.
pub fn set_enabled_for(site: Site, on: bool) {
    let bit = site.bit();
    let mut current = CLASSES.load(Ordering::Relaxed);
    loop {
        let next = if on { current | bit } else { current & !bit };
        match CLASSES.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(seen) => current = seen,
        }
    }
}

/// Every class groups again.
pub fn enable_all_classes() {
    CLASSES.store(u8::MAX, Ordering::Relaxed);
}

/// Set the arm explicitly, for a harness running both in one process.
///
/// Not for production code: a projection's schedule belongs to the plan,
/// not to a global anyone can flip mid-decode. Public because the CPU-7C
/// harness lives outside this crate's test tree, and the alternative —
/// two processes and two 51 GB model loads per data point — would put a
/// worse instrument in the way of a cleaner API.
pub fn set_enabled(on: bool) {
    SWITCH.store(if on { ON } else { OFF }, Ordering::Relaxed);
}

/// One activation, quantised on its own block statistics.
#[cfg(target_arch = "aarch64")]
struct Positioned {
    codes: Vec<i8>,
    scales: Vec<f32>,
    /// `None` under symmetric coding — an absent term, not a zero vector.
    mids: Option<Vec<f32>>,
}

/// The activation's scale geometry and coding, as a value.
///
/// Built on every target — `geometry()` reads the process's arithmetic
/// arm regardless — but only READ by the NEON sweep, so off `aarch64` the
/// fields are legitimately unused rather than accidentally so.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(super) struct Geometry {
    pub ablock: usize,
    pub asym: bool,
}

/// Whether there is a stationary path for this geometry and position
/// count, in this process.
///
/// The geometry conditions are deliberately the SAME ones
/// [`super::integer::Q8xQ8`] uses to reach its K5 register row. A
/// stationary path that engaged where the single-vector path does not
/// would be comparing two kernels rather than two schedules.
///
/// **Callers must consult this before calling a measurement stationary.**
/// Falling back to the looping default is correct arithmetic and the
/// WRONG EXPERIMENT — CPU-5's K1 measured exactly such a silent fallback
/// and the reading stood for two rungs.
pub(super) fn supports(weight: WeightRows<'_>, in_dim: usize, n: usize) -> bool {
    // `enabled` and `bit_identical_only` are PROCESS modes, not
    // geometries, so they live here and not in `supports_with`: a run
    // pinned to the reference path must stay on it, and a test asking a
    // geometry question should not have to know that.
    enabled()
        && class_enabled()
        && !bit_identical_only()
        && geometry().is_some_and(|geo| supports_with(weight, in_dim, n, geo))
}

/// The activation geometry this process is running.
///
/// Separated from the read so a test can exercise the sweep WITHOUT
/// setting a process-wide variable the rest of the suite shares — the
/// same separation `parse_workers` makes for the pool size, and for the
/// same reason: these selectors are `OnceLock`-cached, so the first test
/// to touch one fixes it for every test after it.
pub(super) fn geometry() -> Option<Geometry> {
    match activation_scaling() {
        ScaleSpan::Block(ablock) => Some(Geometry {
            ablock,
            asym: matches!(activation_code(), ActivationCode::Asymmetric),
        }),
        ScaleSpan::Tensor => None,
    }
}

/// Whether the GEOMETRY admits a stationary sweep, independent of any
/// process mode.
#[cfg(target_arch = "aarch64")]
pub(super) fn supports_with(
    weight: WeightRows<'_>,
    in_dim: usize,
    n: usize,
    geo: Geometry,
) -> bool {
    let WeightRows::Q8 { block, .. } = weight else {
        return false;
    };
    UNROLLED.contains(&n)
        && has_dotprod()
        && geo.ablock == SDOT_LANES
        && block.checked_div(geo.ablock) == Some(PER_WEIGHT_B16)
        && in_dim.is_multiple_of(SDOT_LANES * PER_WEIGHT_B16)
}

/// No stationary sweep exists off `aarch64`, so no geometry admits one.
///
/// `false` rather than a panic: the caller's contract is to ASK, and the
/// looping default it falls back to is correct arithmetic everywhere.
#[cfg(not(target_arch = "aarch64"))]
pub(super) fn supports_with(
    _weight: WeightRows<'_>,
    _in_dim: usize,
    _n: usize,
    _geo: Geometry,
) -> bool {
    false
}

/// `out` holds `rows * n` results, POSITION-MINOR: row `r`'s `n` values
/// are contiguous at `out[r*n .. (r+1)*n]`.
///
/// Position-minor because the weight ROW is what is being kept resident.
/// The opposite layout would make each position's output stride the whole
/// matrix, which is the traffic pattern this module exists to remove.
///
/// # Panics
/// If [`supports`] does not hold. The caller has to have asked, and a
/// kernel that quietly did something else here is the failure mode the
/// whole module guards against.
pub(super) fn project_rows_many(weight: WeightRows<'_>, xs: &[&[f32]], out: &mut [f32], n: usize) {
    let geo = geometry().expect("the caller consulted `supports` first");
    project_rows_many_with(weight, xs, out, n, geo);
}

/// The sweep at an explicit geometry. See [`geometry`] for why the
/// environment read is a separate step.
#[cfg(target_arch = "aarch64")]
pub(super) fn project_rows_many_with(
    weight: WeightRows<'_>,
    xs: &[&[f32]],
    out: &mut [f32],
    n: usize,
    geo: Geometry,
) {
    let WeightRows::Q8 {
        codes,
        scales,
        block,
        ..
    } = weight
    else {
        panic!("the stationary kernel consumes q8 weights only");
    };
    let in_dim = xs[0].len();
    assert!(
        supports_with(weight, in_dim, n, geo) && xs.len() == n,
        "the stationary kernel was called for a geometry it does not serve; the caller \
         must consult `supports` first rather than receive silently non-stationary timings"
    );
    let per_row = in_dim.div_ceil(block);

    // Quantised per position, from that position's OWN block statistics.
    // A joint scale here would be a different representation.
    let acts: Vec<Positioned> = xs
        .iter()
        .map(|x| {
            if geo.asym {
                let (codes, scales, mids) = quantise_activation_asymmetric(x, geo.ablock);
                Positioned {
                    codes,
                    scales,
                    mids: Some(mids),
                }
            } else {
                let (codes, scales) = quantise_activation_blocked(x, geo.ablock);
                Positioned {
                    codes,
                    scales,
                    mids: None,
                }
            }
        })
        .collect();

    match n {
        2 => sweep::<2>(codes, scales, &acts, in_dim, per_row, out),
        4 => sweep::<4>(codes, scales, &acts, in_dim, per_row, out),
        _ => unreachable!("`supports_with` admits only the unrolled counts"),
    }
}

/// The unrolled sweep. `N` is a compile-time constant so the position
/// loop inside a block unrolls and the `N` accumulators stay in registers
/// — the difference between sharing a weight LOAD and merely sharing a
/// weight ADDRESS.
#[cfg(target_arch = "aarch64")]
fn sweep<const N: usize>(
    codes: &[i8],
    scales: &[f32],
    acts: &[Positioned],
    in_dim: usize,
    per_row: usize,
    out: &mut [f32],
) {
    let code: [&[i8]; N] = std::array::from_fn(|i| acts[i].codes.as_slice());
    let scale: [&[f32]; N] = std::array::from_fn(|i| acts[i].scales.as_slice());
    let mids: Option<[&[f32]; N]> = acts[0]
        .mids
        .is_some()
        .then(|| std::array::from_fn(|i| acts[i].mids.as_deref().expect("one code for all")));
    for (r, slot) in out.chunks_mut(N).enumerate() {
        let row = &codes[r * in_dim..(r + 1) * in_dim];
        let ws = &scales[r * per_row..(r + 1) * per_row];
        let mut acc = [0.0f32; N];
        // SAFETY: `supports_with` established `dotprod` and the block
        // geometry; every slice above is cut to this row's own geometry.
        unsafe { row_many::<N>(row, ws, &code, &scale, mids.as_ref(), in_dim, &mut acc) };
        slot.copy_from_slice(&acc);
    }
}

/// One Q8 row against `N` quantised activations.
///
/// The frozen K5 row (`q8_row_b16_register_sdot`) with exactly one change:
/// the four weight vectors are loaded once and the per-position work is
/// nested inside. The two fused multiply-adds per position keep their
/// order — the `dv` term, then the `mid` term — because that order is
/// which `f32` comes out.
///
/// # Safety
/// Requires `dotprod` and `in_dim` a multiple of `SDOT_LANES *
/// PER_WEIGHT_B16`, both established by [`supports_with`].
#[cfg(target_arch = "aarch64")]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "dotprod")]
unsafe fn row_many<const N: usize>(
    codes: &[i8],
    wscales: &[f32],
    act_codes: &[&[i8]; N],
    act_scales: &[&[f32]; N],
    act_mids: Option<&[&[f32]; N]>,
    in_dim: usize,
    out: &mut [f32; N],
) {
    use std::arch::aarch64::*;
    let ones = vdupq_n_s8(1);
    let z = vdupq_n_s32(0);
    let blocks = in_dim / SDOT_LANES;
    let mut acc = [vdupq_n_f32(0.0); N];
    let mut b = 0usize;
    while b + PER_WEIGHT_B16 <= blocks {
        let i0 = b * SDOT_LANES;
        // Loaded ONCE. Every position below reuses these four registers.
        let w0 = vld1q_s8(codes.as_ptr().add(i0));
        let w1 = vld1q_s8(codes.as_ptr().add(i0 + SDOT_LANES));
        let w2 = vld1q_s8(codes.as_ptr().add(i0 + 2 * SDOT_LANES));
        let w3 = vld1q_s8(codes.as_ptr().add(i0 + 3 * SDOT_LANES));
        let ws = *wscales.get_unchecked(b / PER_WEIGHT_B16);
        // Weight-only, so hoisted. The same `f32` each position would
        // have computed for itself.
        let svf = act_mids.map(|_| {
            let s0 = vdotq_s32(z, w0, ones);
            let s1 = vdotq_s32(z, w1, ones);
            let s2 = vdotq_s32(z, w2, ones);
            let s3 = vdotq_s32(z, w3, ones);
            vcvtq_f32_s32(vpaddq_s32(vpaddq_s32(s0, s1), vpaddq_s32(s2, s3)))
        });
        for n in 0..N {
            let qx = act_codes.get_unchecked(n).as_ptr();
            let d0 = vdotq_s32(z, w0, vld1q_s8(qx.add(i0)));
            let d1 = vdotq_s32(z, w1, vld1q_s8(qx.add(i0 + SDOT_LANES)));
            let d2 = vdotq_s32(z, w2, vld1q_s8(qx.add(i0 + 2 * SDOT_LANES)));
            let d3 = vdotq_s32(z, w3, vld1q_s8(qx.add(i0 + 3 * SDOT_LANES)));
            let dv = vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));
            let scale_v = vmulq_n_f32(vld1q_f32(act_scales.get_unchecked(n).as_ptr().add(b)), ws);
            let mut a = vfmaq_f32(*acc.get_unchecked(n), scale_v, vcvtq_f32_s32(dv));
            if let (Some(svf), Some(mids)) = (svf, act_mids) {
                let mid_v = vmulq_n_f32(vld1q_f32(mids.get_unchecked(n).as_ptr().add(b)), ws);
                a = vfmaq_f32(a, mid_v, svf);
            }
            *acc.get_unchecked_mut(n) = a;
        }
        b += PER_WEIGHT_B16;
    }
    for n in 0..N {
        *out.get_unchecked_mut(n) = vaddvq_f32(*acc.get_unchecked(n));
    }
}

/// Non-aarch64 stub for the sweep entry. Unreachable: `supports_with` is
/// `false` on every target that compiles this.
#[cfg(not(target_arch = "aarch64"))]
pub(super) fn project_rows_many_with(
    _weight: WeightRows<'_>,
    _xs: &[&[f32]],
    _out: &mut [f32],
    _n: usize,
    _geo: Geometry,
) {
    unreachable!("the stationary kernel requires aarch64 dotprod")
}
