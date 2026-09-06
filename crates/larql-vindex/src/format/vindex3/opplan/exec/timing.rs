//! Where a decode token's milliseconds actually go.
//!
//! The projection ledger prices weight traffic; this prices TIME, at the
//! same call sites, so the two describe one execution rather than two
//! stories about it. Together they replace the arithmetic that produced
//! the old "159 ms residue": token wall minus an assumed bandwidth,
//! which is not a measurement of anything and turned out to be wrong by
//! more than a factor of two.
//!
//! **The leaves are disjoint and nothing nests.** A class that wrapped
//! another would double-count, and a sum that double-counts can be made
//! to equal anything. So a timer covers exactly its own arithmetic — the
//! FFN's activation but not the three projections around it, the
//! recurrence but not the five projections beside it — and the
//! reconciliation is:
//!
//! ```text
//! sum(leaf classes) + unattributed = steady token wall
//! ```
//!
//! **`unattributed` is a failing diagnostic, not a bucket.** Above a few
//! percent it means a boundary is missing, and the answer is to find it,
//! not to name it. The moment it becomes somewhere to put the
//! unexplained, this file has recreated the thing it was built to
//! delete.
//!
//! Reconciliation holds for the DECODE path, which runs one position on
//! the caller's thread. The batched driver runs positions in parallel, so
//! its leaves sum across threads and exceed the wall by design; the
//! report says which path it measured.
//!
//! [`OpClass::KdaBranchFanout`] briefly broke "nothing nests" on purpose
//! at P4c-2a (KDA's independent branches dispatched concurrently, a
//! wall-clock question the branches' own per-thread timers couldn't
//! answer) — that rung found branch concurrency cost more than it saved
//! and the class is unwired again at P4c-4, so the exception no longer
//! applies; kept here as history, not a live design constraint. The
//! REAL near-miss P4c-4 caught: `CpuExecutor::project_as` exists
//! specifically so a caller with its own named boundary (KDA's
//! q/k/v/o_proj) can use the executor's row-parallel path WITHOUT
//! nesting its generic `OpClass::Projection` timer inside the caller's
//! own — `project()` still does that generic timing for every OTHER
//! caller, `project_as` is the one to reach for from inside a class
//! that already has a name.
//!
//! Cost is two `Instant` reads per leaf against roughly 1200 leaves and
//! 480 ms per token — about 60 microseconds, 0.01%. Always on rather than
//! behind a flag that would be off exactly when a number needed
//! explaining.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// One kind of work a decode token is made of.
///
/// Deliberately operator-shaped rather than layer-shaped: "layer 7 cost
/// 8 ms" cannot be acted on, and "the recurrence cost 8 ms over 48 calls"
/// can.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OpClass {
    /// Every dense `y = Wx`, whatever kernel ran it. Timed inside the
    /// executor, at the same site the byte ledger is written, so the two
    /// cannot describe different call sets.
    Projection,
    Embed,
    /// RMS/LayerNorm at every site, including Q/K normalisation.
    Norm,
    Rope,
    /// Softmax over the K/V cache and the weighted sum of V.
    AttentionCore,
    /// The gate's sigmoid and elementwise multiply — NOT its projection.
    OutputGate,
    /// Gated DeltaNet's depthwise causal convolution and the SiLU after
    /// it.
    DeltaConv,
    /// `repeat_interleave` of q/k across value heads.
    DeltaHeadExpand,
    /// beta, softplus, and the decay term.
    DeltaGates,
    /// The delta rule itself — the state update.
    DeltaRecurrence,
    /// Per-head RMS over `Dv`, the norm weight, and the SiLU'd gate.
    DeltaGatedNorm,
    /// GeGLU/SwiGLU/GELU between the FFN's projections.
    FfnActivation,
    /// Residual adds and layer scaling.
    Residual,
    /// Logit multiplier and softcapping over the vocabulary — NOT the
    /// head's projection.
    Logits,
    /// Kimi Linear's KDA attention, whole — projections, convolution,
    /// gates and recurrence together, as ONE leaf. Coarse by design at
    /// P3d-n ("KDA total", not a sub-breakdown); SUPERSEDED at P4c by
    /// the eleven `Kda*` classes below, once "KDA total" itself became
    /// the largest bucket and needed splitting. No longer wired to any
    /// call site — kept as a variant rather than removed, since a
    /// future coarse read (one number, not eleven) is still a
    /// legitimate question to ask.
    Kda,
    /// Kimi Linear's MLA attention, whole — q/kv_a/kv_b projections,
    /// the per-position decompression read and the softmax/combine,
    /// together as ONE leaf, same coarseness reasoning as [`OpClass::Kda`].
    Mla,
    /// `KimiMoEGate` — sigmoid, bias-corrected top-k selection, gather
    /// and (when `top_k>1`) renormalisation. NOT the experts it selects.
    MoeRouter,
    /// Every SELECTED routed expert's `w2(silu(w1x)*w3x)`, PER-JOB — one
    /// leaf per expert, on whatever thread ran it. Since P4d these run
    /// concurrently with the shared branch (same `parallel_map` fan-out,
    /// see [`OpClass::MoeFanout`]), so the SUM of these eight is total
    /// CPU-seconds across threads, not wall-clock — a diagnostic
    /// ("did any expert's own cost move"), not a `MoeFanout` summand.
    /// Also covers layer 0's dense `KimiMLP` (the same shape, one
    /// expert, always selected, never concurrent with anything since a
    /// dense layer has no routed branch to share the fan-out with).
    MoeRoutedExpert,
    /// The shared expert's `w2(silu(w1x)*w3x)`, PER-JOB — same
    /// diagnostic-only posture as [`OpClass::MoeRoutedExpert`] since
    /// P4d: this now runs INSIDE the same fan-out as the eight routed
    /// experts, not sequentially after them, so it is not summed into
    /// `MoeFanout` either.
    MoeSharedExpert,
    /// Wall-clock latency, measured on the CALLING thread, of dispatching
    /// the routed branch's up-to-eight experts AND the shared branch
    /// TOGETHER as one `parallel_map` fan-out (P4d) — the routed 8-way
    /// fan-out `parallel_map` already proved at P4b-1, with the shared
    /// branch added as a ninth independent job rather than run
    /// sequentially after. This is the number `generate_baseline.rs`
    /// sums for "MoE total", same reasoning
    /// [`OpClass::KdaBranchFanout`] once used: [`OpClass::MoeRoutedExpert`]/
    /// [`OpClass::MoeSharedExpert`] keep running INSIDE this fan-out on
    /// their own threads and would double-count against it if summed
    /// alongside.
    MoeFanout,
    /// `lm_head`'s matvec over the full vocabulary — NOT the final norm
    /// before it (that is [`OpClass::Norm`]).
    LmHead,
    /// KDA's `q_proj` matvec alone.
    KdaQProj,
    /// KDA's `k_proj` matvec alone.
    KdaKProj,
    /// KDA's `v_proj` matvec alone.
    KdaVProj,
    /// All three of KDA's depthwise causal convolutions (q, k, v) plus
    /// their SiLU — combined into one class because they are the SAME
    /// operation on three streams, not because they share data.
    KdaConv,
    /// KDA's q AND k L2 normalisation, combined — the easiest operation
    /// in the whole block to omit by accident (see this file's own
    /// history), now with its own cost so an "optimisation" that drops
    /// it is visible in more than correctness.
    KdaQkNorm,
    /// `f_a_proj → f_b_proj`, THROUGH the decay-gate value itself
    /// (`-a_log.exp() * softplus(f_low + dt_bias)`) — the projection and
    /// the nonlinearity that turns it into a per-head decay, together.
    KdaDecayGate,
    /// The output gate's PROJECTION only — `g_a_proj → g_b_proj` in the
    /// low-rank form, one `g_proj` matvec in Kimi-K3's declared full-rank
    /// form; its sigmoid and elementwise apply happen later, fused into
    /// [`OpClass::KdaGatedNorm`], not here.
    KdaOutputGate,
    /// `b_proj` through the sigmoid that turns it into `beta`.
    KdaBProj,
    /// The delta-rule state update and readout, per head — KDA's own
    /// proven recurrence (see this file's own doc comment on why this
    /// rung profiles around it rather than through it).
    KdaRecurrence,
    /// The gated RMSNorm: per-head RMS, the norm weight, and
    /// `sigmoid(o_gate)` applied together — NOT `g_a_proj → g_b_proj`
    /// itself (that is [`OpClass::KdaOutputGate`]).
    KdaGatedNorm,
    /// KDA's `o_proj` matvec alone.
    KdaOProj,
    /// Wall-clock latency of dispatching KDA's six input-only branches
    /// (q/k/v projection+conv+norm, decay-gate, output-gate, b_proj)
    /// concurrently onto the CPU pool via `CpuExecutor::parallel6`.
    /// P4c-2a built this to answer "did concurrency shrink end-to-end
    /// latency for this segment" — it did, barely (~12%), while the
    /// branches' own per-boundary timers showed core contention roughly
    /// TRIPLING their individual costs, so P4c-4 reverted `kda::step` to
    /// sequential dispatch and this class is UNWIRED again, same
    /// coarse-superseded posture as [`OpClass::Kda`] above. Kept as a
    /// variant, not deleted, since the finding it measured is real and a
    /// future machine/kernel combination might make the question worth
    /// re-asking.
    KdaBranchFanout,
}

impl OpClass {
    /// Every class, so a reader enumerates rather than remembers.
    pub const ALL: [OpClass; 33] = [
        OpClass::Projection,
        OpClass::Embed,
        OpClass::Norm,
        OpClass::Rope,
        OpClass::AttentionCore,
        OpClass::OutputGate,
        OpClass::DeltaConv,
        OpClass::DeltaHeadExpand,
        OpClass::DeltaGates,
        OpClass::DeltaRecurrence,
        OpClass::DeltaGatedNorm,
        OpClass::FfnActivation,
        OpClass::Residual,
        OpClass::Logits,
        OpClass::Kda,
        OpClass::Mla,
        OpClass::MoeRouter,
        OpClass::MoeRoutedExpert,
        OpClass::MoeSharedExpert,
        OpClass::MoeFanout,
        OpClass::LmHead,
        OpClass::KdaQProj,
        OpClass::KdaKProj,
        OpClass::KdaVProj,
        OpClass::KdaConv,
        OpClass::KdaQkNorm,
        OpClass::KdaDecayGate,
        OpClass::KdaOutputGate,
        OpClass::KdaBProj,
        OpClass::KdaRecurrence,
        OpClass::KdaGatedNorm,
        OpClass::KdaOProj,
        OpClass::KdaBranchFanout,
    ];

    pub fn name(self) -> &'static str {
        match self {
            OpClass::Projection => "Projection",
            OpClass::Embed => "Embed",
            OpClass::Norm => "Norm",
            OpClass::Rope => "RoPE",
            OpClass::AttentionCore => "AttentionCore",
            OpClass::OutputGate => "OutputGate",
            OpClass::DeltaConv => "DeltaConv",
            OpClass::DeltaHeadExpand => "DeltaHeadExpand",
            OpClass::DeltaGates => "DeltaGates",
            OpClass::DeltaRecurrence => "DeltaRecurrence",
            OpClass::DeltaGatedNorm => "DeltaGatedNorm",
            OpClass::FfnActivation => "FfnActivation",
            OpClass::Residual => "Residual",
            OpClass::Logits => "Logits",
            OpClass::Kda => "Kda",
            OpClass::Mla => "Mla",
            OpClass::MoeRouter => "MoeRouter",
            OpClass::MoeRoutedExpert => "MoeRoutedExpert",
            OpClass::MoeSharedExpert => "MoeSharedExpert",
            OpClass::MoeFanout => "MoeFanout",
            OpClass::LmHead => "LmHead",
            OpClass::KdaQProj => "KdaQProj",
            OpClass::KdaKProj => "KdaKProj",
            OpClass::KdaVProj => "KdaVProj",
            OpClass::KdaConv => "KdaConv",
            OpClass::KdaQkNorm => "KdaQkNorm",
            OpClass::KdaDecayGate => "KdaDecayGate",
            OpClass::KdaOutputGate => "KdaOutputGate",
            OpClass::KdaBProj => "KdaBProj",
            OpClass::KdaRecurrence => "KdaRecurrence",
            OpClass::KdaGatedNorm => "KdaGatedNorm",
            OpClass::KdaOProj => "KdaOProj",
            OpClass::KdaBranchFanout => "KdaBranchFanout",
        }
    }

    fn index(self) -> usize {
        OpClass::ALL
            .iter()
            .position(|c| *c == self)
            .expect("ALL covers every class")
    }
}

/// One class's tally, read out.
///
/// Calls alongside nanos because they are different problems: 12 ms over
/// 48 calls is an arithmetic cost and 12 ms over 5000 is a dispatch cost,
/// and only one of them is fixed by a faster kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClassTally {
    pub calls: u64,
    pub nanos: u64,
}

impl ClassTally {
    pub fn nanos_per_call(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.nanos as f64 / self.calls as f64
        }
    }
}

#[derive(Default)]
struct Slot {
    calls: AtomicU64,
    nanos: AtomicU64,
}

/// Every leaf the executor has timed, by class.
pub struct TimingLedger {
    slots: [Slot; 33],
    /// Timers that started while another was already running ON THE SAME
    /// THREAD.
    ///
    /// Counted rather than fatal: a panic here would take down a decode
    /// over an instrumentation mistake. But any overlap at all voids the
    /// reconciliation — the classes would double-count — so the report
    /// must refuse to add up rather than quietly present a total that
    /// is too large.
    nested: AtomicU64,
}

thread_local! {
    /// Whether this thread is already inside a timed leaf.
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
}

impl TimingLedger {
    /// A ledger with nothing in it. `const` so the process ledger is a
    /// static rather than a lazily-initialised one, and so a test can
    /// hold its own instead of racing the shared counters.
    pub(super) const fn new() -> Self {
        #[allow(clippy::declare_interior_mutable_const)]
        const ZERO: Slot = Slot {
            calls: AtomicU64::new(0),
            nanos: AtomicU64::new(0),
        };
        Self {
            slots: [ZERO; 33],
            nested: AtomicU64::new(0),
        }
    }

    pub(super) fn record(&self, class: OpClass, nanos: u64) {
        let slot = &self.slots[class.index()];
        slot.calls.fetch_add(1, Ordering::Relaxed);
        slot.nanos.fetch_add(nanos, Ordering::Relaxed);
    }

    pub fn get(&self, class: OpClass) -> ClassTally {
        let slot = &self.slots[class.index()];
        ClassTally {
            calls: slot.calls.load(Ordering::Relaxed),
            nanos: slot.nanos.load(Ordering::Relaxed),
        }
    }

    pub fn all(&self) -> [(OpClass, ClassTally); 33] {
        OpClass::ALL.map(|c| (c, self.get(c)))
    }

    /// Total timed nanoseconds across every class.
    pub fn total_nanos(&self) -> u64 {
        OpClass::ALL.iter().map(|c| self.get(*c).nanos).sum()
    }

    /// Overlapping timers seen. Non-zero invalidates the reconciliation.
    pub fn nested(&self) -> u64 {
        self.nested.load(Ordering::Relaxed)
    }

    /// Zero everything, so a caller can price ONE step.
    pub fn reset(&self) {
        for slot in &self.slots {
            slot.calls.store(0, Ordering::Relaxed);
            slot.nanos.store(0, Ordering::Relaxed);
        }
        self.nested.store(0, Ordering::Relaxed);
    }
}

/// A running leaf timer. Records on drop.
pub struct Timed {
    class: OpClass,
    started: Instant,
    /// Whether this timer owns the thread's active flag. A nested timer
    /// does not, so it must not clear the flag its parent set.
    outermost: bool,
}

impl Drop for Timed {
    fn drop(&mut self) {
        let nanos = self.started.elapsed().as_nanos() as u64;
        ledger().record(self.class, nanos);
        if self.outermost {
            ACTIVE.with(|a| a.set(false));
        }
    }
}

/// Start timing one leaf. The value must be held for the leaf's extent.
///
/// `let _t = timed(OpClass::Norm);` — binding to `_` instead would drop
/// it immediately and time nothing, which is the one mistake this API
/// makes easy, so bind it to a name.
pub fn timed(class: OpClass) -> Timed {
    let outermost = ACTIVE.with(|a| {
        let was = a.get();
        a.set(true);
        !was
    });
    if !outermost {
        // Counted, never fatal, and identically in every build profile.
        // A `debug_assert` here would make the executor behave one way
        // under test and another in release — the failure mode being
        // guarded against is a silently wrong total, and a panic that
        // only happens in debug does not prevent it.
        ledger().nested.fetch_add(1, Ordering::Relaxed);
    }
    Timed {
        class,
        started: Instant::now(),
        outermost,
    }
}

static LEDGER: TimingLedger = TimingLedger::new();

/// The process's timing ledger.
pub fn ledger() -> &'static TimingLedger {
    &LEDGER
}
