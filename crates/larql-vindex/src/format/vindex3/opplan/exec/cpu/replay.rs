//! CPU-PERF-3B: replay a real token's projections against the real
//! resident model.
//!
//! The synthetic shape harness predicts real BF16 projection to +0.7% and
//! misses real Q8 by +7.9%. Machine state, allocation alignment and the
//! code/scale stream split have each been falsified, and one structural
//! difference is left:
//!
//! ```text
//!   harness   one 178 MB matrix, exercised in a tight loop
//!   decode    369 distinct operands over 27 GB, each touched once
//! ```
//!
//! This closes that gap by removing everything else. It records the
//! projections one steady step actually issued — the real resident
//! operands, the real geometry, the real activations, in the real order —
//! and replays exactly those, with no norm, no recurrence, no attention,
//! no activation function in between.
//!
//! ```text
//!   synthetic shape harness     ~326 ms
//!   this replay                    X
//!   full decode Projection      ~352 ms
//! ```
//!
//! `X` near 352 says the synthetic harness fails because it cannot
//! reproduce full-model residency. `X` near 326 says residency is
//! innocent and the cost comes from interleaving projections with the
//! rest of decode.
//!
//! **The ordering arms are diagnostic, not proposals.** Replaying the
//! same calls grouped by operand family, or shuffled, separates a
//! locality effect from a cost intrinsic to traversing hundreds of
//! distinct allocations.

use std::sync::Mutex;

use super::executor::CpuExecutor;
use super::physical::PhysicalProjectionPlan;
use super::projector::WeightRows;

/// Which representation a captured operand was resident as.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    F32,
    Bf16,
    Q8,
    Q4,
}

/// One projection exactly as the decode issued it.
///
/// Addresses rather than slices: the point is to replay against the
/// operands the model is ALREADY holding, so copying them would destroy
/// the residency this measures. The session outlives every replay — see
/// [`replay`]'s safety note.
pub struct Captured {
    kind: Kind,
    /// Primary stream: address and element count.
    primary: (usize, usize),
    /// Scales, where the format has them.
    secondary: (usize, usize),
    /// The asymmetric path's code-sum index, where an arm built one.
    /// Captured rather than dropped: a replay that reconstructed the
    /// operand WITHOUT its index would silently take the symmetric code
    /// path and price a kernel the decode never ran.
    tertiary: (usize, usize),
    block: usize,
    out_dim: usize,
    /// The activation the decode actually projected. Kept by value
    /// because it is tens of KB against tens of MB of weight, and
    /// substituting a dummy would leave "the activations differ" as a
    /// live alternative explanation.
    x: Vec<f32>,
}

impl Captured {
    /// Rebuild the row view.
    ///
    /// # Safety
    /// The operands must still be resident and unmoved. Every caller
    /// holds the `DecodeSession` that owns them for the whole replay.
    unsafe fn rows(&self) -> WeightRows<'_> {
        let (p, n) = self.primary;
        let (s, m) = self.secondary;
        match self.kind {
            Kind::F32 => WeightRows::F32(std::slice::from_raw_parts(p as *const f32, n)),
            Kind::Bf16 => WeightRows::Bf16(std::slice::from_raw_parts(p as *const u16, n)),
            Kind::Q8 => WeightRows::Q8 {
                codes: std::slice::from_raw_parts(p as *const i8, n),
                scales: std::slice::from_raw_parts(s as *const f32, m),
                sums: match self.tertiary {
                    (0, _) | (_, 0) => &[],
                    (a, k) => std::slice::from_raw_parts(a as *const i16, k),
                },
                block: self.block,
            },
            Kind::Q4 => WeightRows::Q4 {
                packed: std::slice::from_raw_parts(p as *const u8, n),
                scales: std::slice::from_raw_parts(s as *const f32, m),
                block: self.block,
            },
        }
    }

    /// A stable name for the operand family, for the grouped arm.
    fn family(&self) -> (usize, usize) {
        (self.out_dim, self.x.len())
    }
}

static CAPTURE: Mutex<Option<Vec<Captured>>> = Mutex::new(None);

/// Begin recording projections. Any previous recording is discarded.
pub fn start_capture() {
    *CAPTURE.lock().expect("capture lock") = Some(Vec::new());
}

/// Stop recording and take what was captured.
pub fn take_capture() -> Vec<Captured> {
    CAPTURE
        .lock()
        .expect("capture lock")
        .take()
        .unwrap_or_default()
}

/// Record one projection, if recording is on.
///
/// Called from the executor's own `project`, so a captured call is the
/// call the model made and not a reconstruction of it. Costs one
/// uncontended lock check per projection while idle.
pub(super) fn record(weight: WeightRows<'_>, x: &[f32], out_dim: usize) {
    let mut slot = match CAPTURE.try_lock() {
        Ok(slot) => slot,
        // A worker thread racing the capture would only ever be inside a
        // projection this call already recorded; skipping is correct and
        // cheaper than blocking a decode.
        Err(_) => return,
    };
    let Some(log) = slot.as_mut() else {
        return;
    };
    let (kind, primary, secondary, tertiary, block) = match weight {
        WeightRows::F32(w) => (Kind::F32, (w.as_ptr() as usize, w.len()), (0, 0), (0, 0), 0),
        WeightRows::Bf16(w) => (
            Kind::Bf16,
            (w.as_ptr() as usize, w.len()),
            (0, 0),
            (0, 0),
            0,
        ),
        WeightRows::Q8 {
            codes,
            scales,
            sums,
            block,
        } => (
            Kind::Q8,
            (codes.as_ptr() as usize, codes.len()),
            (scales.as_ptr() as usize, scales.len()),
            (sums.as_ptr() as usize, sums.len()),
            block,
        ),
        WeightRows::Q4 {
            packed,
            scales,
            block,
        } => (
            Kind::Q4,
            (packed.as_ptr() as usize, packed.len()),
            (scales.as_ptr() as usize, scales.len()),
            (0, 0),
            block,
        ),
    };
    log.push(Captured {
        kind,
        primary,
        secondary,
        tertiary,
        block,
        out_dim,
        x: x.to_vec(),
    });
}

/// The order a replay issues the captured calls in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReplayOrder {
    /// Exactly as the decode issued them.
    Captured,
    /// All calls on one operand family together. Diagnostic: if this is
    /// faster, the penalty is temporal locality rather than something
    /// intrinsic to the number of distinct allocations.
    Grouped,
    /// Deterministically shuffled. Diagnostic from the other side: if
    /// this is no worse than captured order, the traversal was never
    /// benefiting from order in the first place.
    Shuffled,
}

impl ReplayOrder {
    pub const ALL: [ReplayOrder; 3] = [Self::Captured, Self::Grouped, Self::Shuffled];

    pub fn name(self) -> &'static str {
        match self {
            Self::Captured => "captured order",
            Self::Grouped => "grouped by family",
            Self::Shuffled => "shuffled",
        }
    }

    fn indices(self, calls: &[Captured]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..calls.len()).collect();
        match self {
            Self::Captured => {}
            Self::Grouped => order.sort_by_key(|i| calls[*i].family()),
            // A fixed multiplier rather than a random source: the shuffle
            // must be the same on every run, or this arm would add noise
            // instead of removing an explanation.
            Self::Shuffled => order.sort_by_key(|i| i.wrapping_mul(2_654_435_761) & 0xffff_ffff),
        }
        order
    }
}

/// Replay every captured projection once, in `order`, and return the
/// seconds it took.
///
/// # Safety
/// The model whose operands were captured must still be resident and
/// unmoved.
pub unsafe fn replay(exec: &CpuExecutor, calls: &[Captured], order: ReplayOrder) -> f64 {
    let indices = order.indices(calls);
    let started = std::time::Instant::now();
    for i in indices {
        let call = &calls[i];
        let rows = call.rows();
        let plan = PhysicalProjectionPlan::for_resident(rows, call.x.len());
        std::hint::black_box(exec.project(plan.kernel(), rows, &call.x, call.out_dim)[0]);
    }
    started.elapsed().as_secs_f64()
}

/// Bytes the captured calls read, so a replay can be priced against the
/// decode's own ledger rather than against an assumption.
pub fn captured_bytes(calls: &[Captured]) -> usize {
    // SAFETY: reading only the recorded lengths, no dereference.
    calls.iter().map(|c| unsafe { c.rows() }.bytes()).sum()
}
