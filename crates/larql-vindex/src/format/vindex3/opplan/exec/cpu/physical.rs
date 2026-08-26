//! One decision — format, kernel, and threading together.
//!
//! The hazard this file exists to remove is a loader that chooses BF16
//! while an executor separately guesses which kernel to run. Those are
//! two derivations of one fact, and two derivations drift. With two
//! formats the drift is a bug; with Q8 and Q4 as well it is a state
//! space nobody can hold in their head.
//!
//! So there is exactly one value, [`PhysicalProjectionPlan`], and both
//! halves are read off it. The loader asks [`PhysicalProjectionPlan::choose`]
//! what to make resident; the executor asks
//! [`PhysicalProjectionPlan::for_resident`] what is resident. The second
//! is not a second decision — it is an OBSERVATION of the first, total
//! over the representations a CPU kernel can consume, so the two cannot
//! disagree about a matrix even in principle.
//!
//! [`project_matrix`] and [`ExecutorProjections`] live here rather than
//! beside the backend for the same reason: they are the only readers of
//! the observation, and a projection helper that sat somewhere else would
//! be one refactor away from choosing its own kernel again.

use super::arithmetic::{AccumulatorRep, ActivationRep, Arithmetic, WeightRep};
use super::integer::{activation_scaling, Bf16xQ8, Q4xQ8, Q8xQ8};
use super::kernels::{BlasF32, FusedBf16, FusedQ4, FusedQ8, ScalarF32};
use super::projector::{DenseProjector, WeightRows};
use crate::error::VindexError;
use crate::format::vindex3::opplan::exec::backend::{MatrixClass, WeightFormat, WeightSlice};

/// Default performance-cluster L2, used where the machine does not
/// report one. The value this rung measured against (Apple M3 Max).
const DEFAULT_L2_BYTES: usize = 16 * 1024 * 1024;

/// How a dense projection is physically realised on the CPU.
///
/// A single enum rather than a `(format, kernel)` pair because the
/// pairing is not free: `FusedBf16` consumes [`WeightRows::Bf16`] and
/// nothing else, and a pair type would let a caller build the one
/// combination that cannot run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PhysicalProjectionPlan {
    /// The literal scalar transcription over f32. The oracle: chosen by
    /// the reference backend, never by the policy.
    ScalarF32,
    /// Q8 resident, widened and scaled in registers, executor-threaded.
    ///
    /// The first LOSSY plan: the values it decodes are not the values the
    /// checkpoint stores. Worth 1.28x on the projections a token runs —
    /// half the bytes returning a third of the time, because at 8.5 bits
    /// the kernel stops waiting for memory and starts waiting for the
    /// widen.
    FusedQ8,
    /// Q4 resident, unpacked and scaled in registers.
    ///
    /// Reachable by OBSERVATION and not by [`Self::choose`]: CPU-4A asks
    /// only whether Q4 x f32 is worth making a model representation, and
    /// no `WeightFormat` names it, so a policy answering Q4 would refuse
    /// at load. Listed so `for_resident` stays total.
    FusedQ4,
    /// f32 resident, BLAS `sgemv`, threaded by the library.
    ///
    /// The right answer for a matrix whose widened image still fits
    /// cache — see [`compact_threshold_bytes`].
    BlasF32,
    /// bf16 resident, widened in registers, threaded by the executor.
    ///
    /// Halves the bytes a decoded token streams AND halves what the
    /// model occupies, because they are the same bytes.
    FusedBf16,
    /// Q8 resident, **int8 activation, i32 accumulator**, rescaled once
    /// per row. 224.75 ms/token at 118.0 GB/s.
    Q8xQ8,
    /// Q4 resident, int8 activation, i32 accumulator. 135.10 ms/token at
    /// 106.6 GB/s — the CPU-4Y frontier.
    Q4xQ8,
    /// bf16 resident and EXACT, int8 activation, f32 dot. The control
    /// arm: it isolates activation quantisation from weight
    /// quantisation, and is never chosen for speed.
    Bf16xQ8,
}

/// **Which arithmetic the projections run in**, for the whole process.
///
/// A weight representation does NOT determine this. Q8 bytes can be
/// consumed by a widening f32 kernel or by `SDOT`, and the two are the
/// same residency with different numerics — which is exactly why
/// [`PhysicalProjectionPlan::for_resident`] cannot answer from the bytes
/// alone any more and has to consult the same policy the loader did.
///
/// Process-wide and read ONCE, because it is the arm of an experiment:
/// a bank run is one process per arm, and a value that could change
/// mid-decode would make the resulting distribution describe no single
/// representation.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ArithmeticArm {
    /// Compact weights against an f32 activation — what ships today.
    #[default]
    FloatActivation,
    /// Exact bf16 weights against an int8 activation. CPU-5 arm A1.
    Bf16TimesQ8,
    /// Q8 weights against an int8 activation. CPU-5 arm A3.
    Q8TimesQ8,
    /// Q4 weights against an int8 activation. CPU-5 arm A4.
    Q4TimesQ8,
}

/// **Which matrix classes a Q4 arm is permitted to reach.**
///
/// Blanket Q4 is a hypothesis, not a plan. If it fails, the question
/// becomes the smallest set of operands that must be RESTORED to a
/// higher precision to recover quality — and the only axis today's seam
/// can express is the matrix class, because
/// [`super::super::prepared`] deliberately refuses to hand the policy an
/// `OperandRef`: resolving operands by name is the one thing the seam
/// forbids, and a name-based exception set would be a per-model recipe
/// rather than a policy.
///
/// Class is therefore a FIRST CUT, not the final axis. It cannot say
/// "the last five layers' FFN", which is where a 4-bit knee has already
/// been found once on another model. Saying so is a scope limit, not a
/// result.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Q4Classes {
    pub attention: bool,
    pub ffn: bool,
    pub head: bool,
}

impl Q4Classes {
    /// Blanket Q4 — every eligible operand, which is arm R0.
    pub const ALL: Self = Self {
        attention: true,
        ffn: true,
        head: true,
    };

    /// Whether this class may go to Q4. A class that may not falls back
    /// to Q8 **in the same arithmetic domain**, never to an f32
    /// activation: a rescue that also changed the activation treatment
    /// would move two things at once, which is the mistake CPU-4A made.
    pub fn admits(self, class: MatrixClass) -> bool {
        match class {
            MatrixClass::AttentionProjection => self.attention,
            MatrixClass::FfnProjection => self.ffn,
            MatrixClass::OutputHead => self.head,
            // The bank is widened to f32 on the way in; no compact bytes
            // remain by the time a format could apply.
            MatrixClass::RoutedExpertBank => false,
        }
    }
}

/// Names which classes a Q4 arm reaches: a comma-separated subset of
/// `attn`, `ffn`, `head`, or `all`.
pub const Q4_CLASSES_ENV: &str = "LARQL_CPU_Q4_CLASSES";

/// The Q4 class set, resolved once per process.
///
/// Unset means [`Q4Classes::ALL`] — the blanket arm — so an arm run
/// without this variable is the hypothesis rather than a silent
/// exception set.
pub fn q4_classes() -> Q4Classes {
    static CLASSES: std::sync::OnceLock<Q4Classes> = std::sync::OnceLock::new();
    *CLASSES.get_or_init(|| match std::env::var(Q4_CLASSES_ENV).ok() {
        None => Q4Classes::ALL,
        Some(v) if v.trim().is_empty() || v.trim() == "all" => Q4Classes::ALL,
        Some(v) => {
            let named = |k: &str| v.split(',').any(|t| t.trim() == k);
            Q4Classes {
                attention: named("attn"),
                ffn: named("ffn"),
                head: named("head"),
            }
        }
    })
}

/// Names the arithmetic arm. See [`ArithmeticArm`].
pub const ARITHMETIC_ARM_ENV: &str = "LARQL_CPU_ARITHMETIC";

/// The arm, resolved once per process.
///
/// An unrecognised value is the DEFAULT rather than an error, matching
/// [`MAX_FORMAT_ENV`]: a typo must not silently invent a fourth
/// numerical regime that then gets reported as a measurement.
pub fn arithmetic_arm() -> ArithmeticArm {
    static ARM: std::sync::OnceLock<ArithmeticArm> = std::sync::OnceLock::new();
    *ARM.get_or_init(|| {
        match std::env::var(ARITHMETIC_ARM_ENV)
            .ok()
            .as_deref()
            .map(str::trim)
        {
            // The `b` suffix selects a per-BLOCK activation scale; the
            // arm itself is the same arithmetic either way, which is why
            // the scale geometry is a separate value and not a fourth arm.
            Some("bf16xq8") | Some("bf16xq8b") => ArithmeticArm::Bf16TimesQ8,
            Some("q8xq8") | Some("q8xq8b") => ArithmeticArm::Q8TimesQ8,
            Some("q4xq8") | Some("q4xq8b") => ArithmeticArm::Q4TimesQ8,
            _ => ArithmeticArm::FloatActivation,
        }
    })
}

impl PhysicalProjectionPlan {
    /// The representation the loader must make resident for this plan.
    pub fn format(self) -> WeightFormat {
        match self {
            Self::ScalarF32 | Self::BlasF32 => WeightFormat::F32,
            Self::FusedBf16 | Self::Bf16xQ8 => WeightFormat::Bf16,
            Self::FusedQ8 | Self::Q8xQ8 => WeightFormat::Q8,
            Self::FusedQ4 | Self::Q4xQ8 => WeightFormat::Q4,
        }
    }

    /// **What this plan actually computes**, as a value.
    ///
    /// The weight half is a property of the variant. The activation half
    /// is a property of the PROCESS — one arm per run — which is why it
    /// is read through [`activation_scaling`] rather than stored per
    /// operand. When blocking becomes per-operand (a hardware or
    /// architecture choice rather than an experiment's arm) this is the
    /// accessor that has to start taking the operand.
    pub fn arithmetic(self) -> Arithmetic {
        let q8_block = crate::format::vindex3::opplan::exec::quantise::Q8_BLOCK;
        let q4_block = crate::format::vindex3::opplan::exec::quantise::Q4_BLOCK;
        let int8_act = ActivationRep::Q8 {
            span: activation_scaling(),
        };
        match self {
            Self::ScalarF32 | Self::BlasF32 => Arithmetic {
                weight: WeightRep::F32,
                activation: ActivationRep::F32,
                accumulator: AccumulatorRep::F32,
            },
            Self::FusedBf16 => Arithmetic {
                weight: WeightRep::Bf16,
                activation: ActivationRep::F32,
                accumulator: AccumulatorRep::F32,
            },
            Self::FusedQ8 => Arithmetic {
                weight: WeightRep::Q8 { block: q8_block },
                activation: ActivationRep::F32,
                accumulator: AccumulatorRep::F32,
            },
            Self::FusedQ4 => Arithmetic {
                weight: WeightRep::Q4 { block: q4_block },
                activation: ActivationRep::F32,
                accumulator: AccumulatorRep::F32,
            },
            // The control holds EXACT weights and an f32 dot; only the
            // activation is quantised, which is the whole of its job.
            Self::Bf16xQ8 => Arithmetic {
                weight: WeightRep::Bf16,
                activation: int8_act,
                accumulator: AccumulatorRep::F32,
            },
            Self::Q8xQ8 => Arithmetic {
                weight: WeightRep::Q8 { block: q8_block },
                activation: int8_act,
                accumulator: AccumulatorRep::I32,
            },
            Self::Q4xQ8 => Arithmetic {
                weight: WeightRep::Q4 { block: q4_block },
                activation: int8_act,
                accumulator: AccumulatorRep::I32,
            },
        }
    }

    /// The weight representation this plan is resident as.
    pub fn weight_rep(self) -> WeightRep {
        self.arithmetic().weight
    }

    /// The kernel that consumes it, which in turn declares its threading.
    pub fn kernel(self) -> &'static dyn DenseProjector {
        match self {
            Self::ScalarF32 => &ScalarF32,
            Self::BlasF32 => &BlasF32,
            Self::FusedBf16 => &FusedBf16,
            Self::FusedQ8 => &FusedQ8,
            Self::FusedQ4 => &FusedQ4,
            Self::Q8xQ8 => &Q8xQ8,
            Self::Q4xQ8 => &Q4xQ8,
            Self::Bf16xQ8 => &Bf16xQ8,
        }
    }

    /// **The policy.** What to make one matrix resident as.
    ///
    /// `elements` is the matrix's element count — `out_dim * in_dim` —
    /// so the question is asked per MATRIX, not per matrix class. That
    /// distinction is the whole point: Qwen3.8's `48 x 5120` delta gates
    /// and its `10240 x 5120` fused projection are both attention-class
    /// operands, and they want opposite answers.
    ///
    /// `stored_bf16` is a physical fact about the checkpoint, not a
    /// preference. A container holding f32 has no compact bytes to keep,
    /// and narrowing them here would ROUND — bf16 residency promises the
    /// stored bytes are the resident bytes, and a policy that quietly
    /// quantised to hit its own threshold would make that a lie.
    pub fn choose(elements: usize, stored_bf16: bool) -> Self {
        // Class-agnostic: every class admitted, which is what a caller
        // asking without one means.
        Self::choose_for(None, elements, stored_bf16)
    }

    /// The same policy, told which class the operand belongs to.
    ///
    /// The class changes nothing except whether a Q4 arm may reach this
    /// operand — the cache thresholds are physical and identical for
    /// every class.
    pub fn choose_for(class: Option<MatrixClass>, elements: usize, stored_bf16: bool) -> Self {
        if !stored_bf16 || elements * F32_BYTES < compact_threshold_bytes() {
            return Self::BlasF32;
        }
        // **The same cache argument, one format further down.**
        //
        // BF16 beats BLAS f32 once the F32 image stops fitting L2; Q8
        // beats BF16 once the BF16 image does. Measured on the real
        // shapes: `1024 x 5120` is 10.5 MB as bf16, still L2-resident,
        // and runs 0.81x through Q8 — no traffic to halve and the extra
        // unpacking is pure cost. `5120 x 6144` is 62.9 MB, streams, and
        // wins 1.16x. Every measured shape falls on the side this
        // predicts.
        if elements * BF16_BYTES >= compact_threshold_bytes() && q8_permitted() {
            // **The arm applies to exactly this population** — the
            // streaming operands, and no others. The tiny f32 ones and
            // the cache-resident bf16 ones are identical across every
            // arm, so a comparison between arms is a comparison of one
            // representation over one operand set rather than of two
            // differently-composed models.
            match arithmetic_arm() {
                ArithmeticArm::FloatActivation => Self::FusedQ8,
                ArithmeticArm::Bf16TimesQ8 => Self::Bf16xQ8,
                ArithmeticArm::Q8TimesQ8 => Self::Q8xQ8,
                // A class the exception set has RESTORED falls back to Q8
                // in the same integer domain — same activation, same
                // accumulator, only the weight bits change. That is what
                // makes a rescue rung a one-variable experiment.
                ArithmeticArm::Q4TimesQ8 => match class {
                    Some(c) if !q4_classes().admits(c) => Self::Q8xQ8,
                    _ => Self::Q4xQ8,
                },
            }
        } else {
            Self::FusedBf16
        }
    }

    /// **The observation.** What IS resident, read off the bytes.
    ///
    /// Deliberately not a second call to [`Self::choose`]: an executor
    /// that re-derived the policy could be handed a matrix the loader
    /// decided differently about — a fallback, a checkpoint that stores
    /// something else, a threshold read on a machine reporting a
    /// different cache — and would then run the wrong kernel over the
    /// right bytes. Reading the representation cannot be wrong about it.
    pub fn for_resident(rows: WeightRows<'_>, in_dim: usize) -> Self {
        // The bytes still decide the FORMAT — that half is an observation
        // and cannot be wrong. They no longer decide the ARITHMETIC,
        // because Q8 bytes are consumable by a widening f32 kernel and by
        // `SDOT` alike, so the arm is read from the same policy the
        // loader read. Two readers of one value, not two derivations.
        let arm = arithmetic_arm();
        match rows {
            WeightRows::F32(_) => Self::BlasF32,
            // **The control has to cover the SAME operands the arms do.**
            //
            // bf16 bytes are ambiguous in a way Q8 and Q4 bytes are not:
            // an operand is resident as bf16 either because its image
            // fits L2 and the policy kept it exact, or because the A1
            // control swapped a streaming Q8 operand back to exact
            // weights. The bytes cannot tell those apart, so the size
            // rule is re-applied here — the same rule, off the same
            // geometry, not a second policy.
            //
            // Measured cost of getting this wrong: A1 quantised the
            // activation on the cache-resident operands as well, which
            // made the control a LARGER perturbation than the arm it
            // exists to explain, and it read worse than Q8 x Q8 despite
            // holding exact weights.
            WeightRows::Bf16(_) => match arm {
                ArithmeticArm::Bf16TimesQ8 if streams(rows, in_dim) => Self::Bf16xQ8,
                _ => Self::FusedBf16,
            },
            // Under EITHER integer arm, Q8 bytes are consumed by SDOT.
            // Under the Q4 arm they exist only because an exception set
            // restored them, and restoring precision must not also drop
            // the operand back to an f32 activation.
            WeightRows::Q8 { .. } => match arm {
                ArithmeticArm::Q8TimesQ8 | ArithmeticArm::Q4TimesQ8 => Self::Q8xQ8,
                _ => Self::FusedQ8,
            },
            WeightRows::Q4 { .. } => match arm {
                ArithmeticArm::Q4TimesQ8 => Self::Q4xQ8,
                _ => Self::FusedQ4,
            },
        }
    }
}

/// One projection through LARQL's own CPU executor.
///
/// **The kernel is OBSERVED, not chosen again.**
/// [`PhysicalProjectionPlan::for_resident`] reads back the decision
/// `weight_format` already made at load, off the bytes themselves — so a
/// matrix the loader kept compact cannot reach a kernel that expects f32,
/// and a fallback (an f32 checkpoint, an overlaid operand, a machine
/// reporting a different cache) needs no second rule to stay consistent.
///
/// NOT bit-identical to the scalar transcription: both kernels reassociate
/// the sum, measured at rel_rms ~1.3e-6 for BLAS and 3.6e-7 for the fused
/// bf16 kernel. No weight VALUE changes — bf16 widens exactly — so the
/// only difference either kernel introduces is summation order, and the
/// parity gates are what judge it.
pub fn project_rows(
    weight: WeightRows<'_>,
    x: &[f32],
    out_dim: usize,
) -> Result<Vec<f32>, VindexError> {
    let plan = PhysicalProjectionPlan::for_resident(weight, x.len());
    Ok(super::shared()?.project(plan.kernel(), weight, x, out_dim))
}

/// The same, from a declared operand slice plus the geometry that says
/// how much of it is the matrix.
pub fn project_matrix(
    weight: &WeightSlice<'_>,
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>, VindexError> {
    project_rows(weight.rows(out_dim, in_dim)?, x, out_dim)
}

/// The same, for several activations against one weight traversal.
pub fn project_matrix_many(
    weight: &WeightSlice<'_>,
    xs: &[&[f32]],
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<Vec<f32>>, VindexError> {
    project_rows_many(weight.rows(out_dim, in_dim)?, xs, out_dim)
}

/// Gated DeltaNet's five dense projections, through the same executor
/// and the same observation as every other production matrix.
pub struct ExecutorProjections;

impl crate::format::vindex3::opplan::exec::gated_delta::DenseProjections for ExecutorProjections {
    fn project(&self, weight: WeightRows<'_>, x: &[f32], out_dim: usize) -> Vec<f32> {
        project_rows(weight, x, out_dim)
            .expect("the CPU executor pool is unavailable, so no projection can run")
    }

    fn project_many(&self, weight: WeightRows<'_>, xs: &[&[f32]], out_dim: usize) -> Vec<Vec<f32>> {
        project_rows_many(weight, xs, out_dim)
            .expect("the CPU executor pool is unavailable, so no projection can run")
    }

    fn is_weight_stationary(&self, weight: WeightRows<'_>, in_dim: usize, n: usize) -> bool {
        PhysicalProjectionPlan::for_resident(weight, in_dim)
            .kernel()
            .is_weight_stationary(weight, in_dim, n)
    }
}

/// The same observation as [`project_rows`], for `n` positions at once.
///
/// The kernel is still OBSERVED and not chosen again: one plan, read off
/// the resident bytes, and whether that plan's kernel has a stationary
/// path is the kernel's own answer rather than a second policy here.
pub fn project_rows_many(
    weight: WeightRows<'_>,
    xs: &[&[f32]],
    out_dim: usize,
) -> Result<Vec<Vec<f32>>, VindexError> {
    let plan = PhysicalProjectionPlan::for_resident(weight, xs[0].len());
    Ok(super::shared()?.project_many(plan.kernel(), weight, xs, out_dim))
}

/// Caps the policy at a representation, for A/B'ing FORMATS in one
/// binary.
pub const MAX_FORMAT_ENV: &str = "LARQL_CPU_MAX_FORMAT";

/// Whether the policy may reach Q8.
///
/// Exists so a lossy format can be compared against the exact one it
/// replaces WITHOUT rebuilding. Q8 changes logits, so the comparison has
/// to be against the same binary's own bf16 answer — a rebuild moved an
/// untouched function 14% in CPU-2D, and a numerical A/B across builds
/// would be arguing with a compiler as much as with a format.
///
/// Only `bf16` caps anything; every other value (and no value) leaves the
/// measured policy in force, so a typo cannot silently disable Q8 in
/// production and be mistaken for a regression.
fn q8_permitted() -> bool {
    !matches!(
        std::env::var(MAX_FORMAT_ENV).ok().as_deref().map(str::trim),
        Some("bf16")
    )
}

/// Whether this operand is one the policy would have made Q8 —
/// i.e. one whose BF16 image does not fit L2 and therefore STREAMS.
///
/// The same predicate [`PhysicalProjectionPlan::choose`] applies, read
/// off the slab's own geometry so the two cannot describe different
/// populations.
fn streams(rows: WeightRows<'_>, in_dim: usize) -> bool {
    in_dim > 0 && rows.rows(in_dim) * in_dim * BF16_BYTES >= compact_threshold_bytes()
}

/// f32 bytes per element — what the BLAS alternative must read.
pub(crate) const F32_BYTES: usize = 4;

/// bf16 bytes per element — what the Q8 alternative must read.
pub(crate) const BF16_BYTES: usize = 2;

/// The f32 footprint at or above which compact-to-registers wins.
///
/// **Not a fitted constant: it is the performance cluster's L2 size.**
/// BLAS `sgemv` reads its weights from cache while the widened matrix
/// fits (measured 291 GB/s at 7.9 MB) and from RAM once it does not (117
/// GB/s at 21 MB), so the crossover between the two kernels IS the cache
/// boundary — the fused kernel has no such cliff because it streams
/// either way.
///
/// Swept at Qwen3.8's `in_dim` of 5120, the two cross at 832 rows =
/// 17.04 MB against this machine's 16 MiB L2: fused loses 0.60x at 768
/// rows below the boundary and wins 1.82x at 896 rows above it. The
/// transition is a cliff, not a slope, which is why the constant is read
/// from the hardware rather than tuned.
///
/// Every real Qwen3.8 matrix sits far from it — the nearest below is
/// `48 x 5120` at 0.98 MB (BLAS by 3.8x) and the nearest above is
/// `1024 x 5120` at 20.97 MB (fused by 1.99x), a factor of 21 apart —
/// so this model would decode identically under any threshold inside
/// that bracket. A future model with a matrix in the gap is what the
/// boundary is for.
pub(crate) fn compact_threshold_bytes() -> usize {
    #[cfg(target_os = "macos")]
    {
        if let Some(bytes) = super::executor::sysctl_usize("hw.perflevel0.l2cachesize") {
            return bytes.max(1);
        }
    }
    DEFAULT_L2_BYTES
}
