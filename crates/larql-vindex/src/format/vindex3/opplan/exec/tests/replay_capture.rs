//! **Every representation the executor can project must survive capture
//! and replay** — the contract the replay instrument depends on, and the
//! one that quietly widens.
//!
//! `replay` exists to answer a measurement question: the synthetic shape
//! harness predicts real BF16 projection to +0.7% and misses real Q8 by
//! +7.9%, and the remaining structural difference is residency — one
//! large matrix in a loop against hundreds of distinct operands touched
//! once each. It answers that by recording the projections a real decode
//! issued and replaying exactly those.
//!
//! The failure mode is specific and silent. `Captured` stores ADDRESSES,
//! not slices, so that a replay reads the operands the model is already
//! holding rather than copies of them — and a representation with more
//! than one stream (Q8's codes, scales and code-sums; NVFP4's packed
//! codes and its two scale levels) is only reconstructible if capture
//! recorded every stream. Add a variant to `WeightRows` without widening
//! the capture and the instrument does not fail loudly: it either
//! reconstructs a shorter slice and measures the wrong thing, or reads
//! past a stream that was never recorded. Both produce a number.
//!
//! So this pins the property rather than the lines: what goes in comes
//! back, for every representation, in every replay order.
//!
//! **One test, deliberately.** The recorder is a process-wide static —
//! it has to be, because it is written from inside the executor's own
//! `project` and read by a benchmark that owns neither. Splitting these
//! assertions across parallel tests would have them steal each other's
//! recordings, and a suite that passes on scheduling luck is worse than
//! no suite at all.

use crate::format::vindex3::opplan::exec::cpu::executor::shared;
use crate::format::vindex3::opplan::exec::cpu::physical::PhysicalProjectionPlan;
use crate::format::vindex3::opplan::exec::cpu::projector::WeightRows;
use crate::format::vindex3::opplan::exec::cpu::replay::{
    captured_bytes, replay, start_capture, take_capture, Captured, ReplayOrder,
};
use crate::format::vindex3::opplan::exec::quantise::{Q4_BLOCK, Q8_BLOCK};

/// Serialises the tests that OPEN a capture.
///
/// A capture is process-wide and singular: `start_capture` replaces any
/// recording in progress, and `take_capture` removes it. Two tests
/// holding one open at the same time therefore take each other's calls,
/// no matter how carefully each selects its own operands afterwards —
/// so opening one is mutually exclusive, and every test below that calls
/// `start_capture` takes THIS lock, not a lock of its own.
///
/// Concurrency with tests that merely PROJECT is not excluded here: that
/// is the process-wide behaviour the capture is specified to have, and
/// it is handled by selecting on [`WeightRows::primary_addr`].
static CAPTURE_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

const ROWS: usize = 3;
/// One Q8 block wide, so the blocked formats carry real per-block
/// scales rather than a single degenerate one.
const K: usize = Q8_BLOCK;

/// The operands, owned by the caller for the whole test — `replay` reads
/// them by address, so they must outlive every replay.
struct Operands {
    f32_rows: Vec<f32>,
    bf16: Vec<u16>,
    q8_codes: Vec<i8>,
    q8_scales: Vec<f32>,
    q8_sums: Vec<i16>,
    q4_packed: Vec<u8>,
    q4_scales: Vec<f32>,
    nvfp4: larql_models::quant::nvfp4::Nvfp4Matrix,
    x: Vec<f32>,
}

impl Operands {
    fn build() -> Self {
        let values: Vec<f32> = (0..ROWS * K)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.125)
            .collect();
        let groups = ROWS * K / Q8_BLOCK;
        Self {
            bf16: values.iter().map(|v| (v.to_bits() >> 16) as u16).collect(),
            q8_codes: values.iter().map(|v| (v * 8.0) as i8).collect(),
            q8_scales: vec![0.125; groups],
            // Empty: the asymmetric-activation path is the only consumer
            // of the code sums, and an EMPTY index is a state capture
            // must carry as faithfully as a populated one.
            q8_sums: Vec::new(),
            q4_packed: vec![0x42; ROWS * K / 2],
            q4_scales: vec![0.25; ROWS * K / Q4_BLOCK],
            nvfp4: larql_models::quant::nvfp4::quantize(&values, ROWS, K).expect("quantise"),
            x: (0..K).map(|i| ((i % 5) as f32 - 2.0) * 0.5).collect(),
            f32_rows: values,
        }
    }

    /// One of each representation, in a fixed order.
    fn rows(&self) -> Vec<WeightRows<'_>> {
        vec![
            WeightRows::F32(&self.f32_rows),
            WeightRows::Bf16(&self.bf16),
            WeightRows::Q8 {
                codes: &self.q8_codes,
                scales: &self.q8_scales,
                sums: &self.q8_sums,
                block: Q8_BLOCK,
            },
            WeightRows::Q4 {
                packed: &self.q4_packed,
                scales: &self.q4_scales,
                block: Q4_BLOCK,
            },
            WeightRows::Nvfp4 {
                packed: &self.nvfp4.packed,
                scales: &self.nvfp4.scales,
                tensor_scale: self.nvfp4.tensor_scale,
            },
        ]
    }
}

/// What the executor projected is what the capture holds — for every
/// representation, byte for byte of declared traffic — and replay
/// reconstructs all of it in every diagnostic order.
#[test]
fn capture_and_replay_carry_every_representation_the_executor_projects() {
    let _exclusive = CAPTURE_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    let operands = Operands::build();
    let exec = shared().expect("the CPU executor pool");
    let rows = operands.rows();
    let project_all = || {
        for w in &rows {
            let plan = PhysicalProjectionPlan::for_resident(*w, K);
            exec.project(plan.kernel(), *w, &operands.x, ROWS);
        }
    };

    // A capture is PROCESS-WIDE: it records every projection the process
    // issues while it is open, which under `cargo test` includes calls
    // from whatever sibling test is running on another thread. Counting
    // those would make this assertion a statement about the scheduler.
    // So select this test's own calls by operand address, which is
    // unique per live allocation.
    let mine = |calls: Vec<Captured>| -> Vec<Captured> {
        let ours: std::collections::BTreeSet<usize> =
            rows.iter().map(WeightRows::primary_addr).collect();
        calls
            .into_iter()
            .filter(|c| ours.contains(&c.operand_addr()))
            .collect()
    };

    // Off by default: the projection runs, nothing is kept. A recorder
    // left on would grow behind every later measurement.
    project_all();
    assert!(
        mine(take_capture()).is_empty(),
        "capture is off until started"
    );

    start_capture();
    project_all();
    let calls = mine(take_capture());
    assert_eq!(
        calls.len(),
        rows.len(),
        "one captured call per projection, in issue order"
    );
    assert!(
        mine(take_capture()).is_empty(),
        "taking a capture ends it — a second take must not re-serve the same calls"
    );

    // The widening contract, stated as arithmetic: a capture that
    // recorded only the primary stream would under-count exactly the
    // multi-stream formats — Q8's scales and code-sums, NVFP4's two
    // scale levels.
    let expected: usize = rows.iter().map(WeightRows::bytes).sum();
    assert_eq!(
        captured_bytes(&calls),
        expected,
        "captured traffic must include every stream of every format"
    );

    // Replay reconstructs each call from its recorded addresses. A
    // format whose streams were never recorded panics or mis-slices
    // here; a wrong ORDER would not, which is why the orders are
    // checked for permutation rather than for cost.
    for order in ReplayOrder::ALL {
        // SAFETY: `operands` is alive for the whole test and has not
        // moved — the condition `replay` documents.
        let seconds = unsafe { replay(exec, &calls, order) };
        assert!(
            seconds.is_finite() && seconds >= 0.0,
            "{} replayed to {seconds}",
            order.name()
        );
        // Replaying does not consume or reorder the captured set: the
        // arms must price the SAME work, or a cheaper arm would be
        // cheaper for a reason that is not locality.
        assert_eq!(calls.len(), rows.len());
        assert_eq!(captured_bytes(&calls), expected, "{}", order.name());
    }
    let names: std::collections::BTreeSet<&str> =
        ReplayOrder::ALL.iter().map(|o| o.name()).collect();
    assert_eq!(names.len(), 3, "each arm names itself distinctly");

    // A new capture discards the previous recording rather than
    // appending to it — one left over would be replayed against
    // operands that had since moved.
    start_capture();
    project_all();
    start_capture();
    project_all();
    assert_eq!(
        mine(take_capture()).len(),
        rows.len(),
        "a new capture starts empty"
    );
}

/// A concurrent projection is captured too, and does not disturb the
/// caller's own count.
///
/// The regression for the Windows CI failure at `8ac0b7a7`, where
/// `capture_and_replay_carry_every_representation_the_executor_projects`
/// counted SIX calls for five issued: a capture is process-wide, so a
/// sibling test projecting on another thread landed inside its window.
///
/// Two properties have to hold together, and each fails a different old
/// defect:
///
/// ```text
///   the foreign call IS recorded      record() no longer drops a call
///                                     that loses a lock probe — latent
///                                     today (parallel6 is unwired), a
///                                     silent under-count the moment it
///                                     is not
///   the caller's own count is exact   selecting by operand address
///                                     isolates it from the scheduler
/// ```
#[test]
fn a_concurrent_projection_is_captured_without_disturbing_the_owner_count() {
    let _exclusive = CAPTURE_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    let operands = Operands::build();
    let foreign = Operands::build();
    let exec = shared().expect("the CPU executor pool");
    let rows = operands.rows();

    start_capture();
    std::thread::scope(|s| {
        // One projection from ANOTHER thread, guaranteed inside the
        // window because the scope joins before the capture is taken.
        s.spawn(|| {
            let w = foreign.rows()[0];
            let plan = PhysicalProjectionPlan::for_resident(w, K);
            exec.project(plan.kernel(), w, &foreign.x, ROWS);
        });
        for w in &rows {
            let plan = PhysicalProjectionPlan::for_resident(*w, K);
            exec.project(plan.kernel(), *w, &operands.x, ROWS);
        }
    });
    let calls = take_capture();

    let ours: std::collections::BTreeSet<usize> =
        rows.iter().map(WeightRows::primary_addr).collect();
    let (mine, theirs): (Vec<Captured>, Vec<Captured>) = calls
        .into_iter()
        .partition(|c| ours.contains(&c.operand_addr()));

    assert_eq!(
        mine.len(),
        rows.len(),
        "every projection the owner issued is recorded, none dropped"
    );
    // Identify the concurrent call by ITS operand, not by counting.
    // `theirs` also collects whatever sibling tests projected during the
    // window — a capture is process-wide, which is this test's own
    // premise, so asserting `theirs.len() == 1` asserted an exclusivity
    // that does not exist. It held under `cargo test` and broke under
    // `cargo llvm-cov`, where the slower run let 19 sibling calls land.
    let foreign_addr = foreign.rows()[0].primary_addr();
    assert!(
        theirs.iter().any(|c| c.operand_addr() == foreign_addr),
        "the concurrent call is recorded too — a capture is process-wide"
    );
}
