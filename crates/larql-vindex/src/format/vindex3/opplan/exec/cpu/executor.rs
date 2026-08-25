//! The persistent worker pool and the partitioning policy.

use std::sync::OnceLock;

use super::super::timing::{timed, OpClass};
use super::ledger::ledger;
use super::physical::PhysicalProjectionPlan;
use super::projector::{CpuParallelism, DenseProjector, WeightRows};
use crate::error::VindexError;

/// The process-wide pool.
///
/// One pool for the machine, not one per backend. `ProductionBackend` is
/// a zero-sized value that dozens of call sites construct freely, and a
/// pool per instance would put twelve worker threads behind each of them
/// — the oversubscription this module exists to forbid, arrived at by
/// OWNERSHIP rather than by nesting. Threads are a property of the
/// machine, so the machine holds them.
static SHARED: OnceLock<Result<CpuExecutor, String>> = OnceLock::new();

/// The machine's executor, built once.
///
/// Fallible rather than `expect`: the only way this fails is that the OS
/// refused to spawn threads, and a decode that then ran silently on one
/// core would report a throughput number that means nothing.
pub fn shared() -> Result<&'static CpuExecutor, VindexError> {
    match SHARED.get_or_init(CpuExecutor::new) {
        Ok(exec) => Ok(exec),
        Err(e) => Err(VindexError::Parse(format!(
            "the CPU executor pool is unavailable: {e}"
        ))),
    }
}

/// LARQL's CPU worker pool.
///
/// Persistent on purpose: decode runs hundreds of projections per token
/// (64 layers x 3 FFN + 5 delta or 4 attention matrices), and rebuilding
/// a task graph for each would cost more than some of them.
pub struct CpuExecutor {
    pool: rayon::ThreadPool,
    workers: usize,
}

/// Below this many RESIDENT bytes a projection is not worth splitting.
///
/// Measured, not guessed: the `48 x 5120` delta projections are ~1 MB,
/// fit in cache, and ran at 262 GB/s as a single call — there is no
/// streaming to parallelise, only fan-out to add.
///
/// Subordinate to the format decision, not a second version of it.
/// `PhysicalProjectionPlan::choose` only reaches `FusedBf16` at 16 MiB of
/// f32 — 8 MiB resident — so every compact matrix already clears this by
/// 2x. It governs what is left: an `ExternalPool` kernel handed a small
/// matrix, which today means only a hand-built one in a test.
const MIN_SPLIT_BYTES: usize = 4 * 1024 * 1024;

impl CpuExecutor {
    /// A pool sized to SATURATE the memory system, not to fill the
    /// machine — see [`workers_from`].
    ///
    /// Performance cores are the input because the fused BF16 kernel is a
    /// streaming load, and efficiency cores contribute little to memory
    /// throughput while still taking a share of the rows and finishing
    /// late. Falls back to the total core count where the split is not
    /// reported.
    pub fn new() -> Result<Self, String> {
        let workers =
            requested_workers().unwrap_or_else(|| workers_from(reported_performance_cores()));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|i| format!("larql-cpu-{i}"))
            .build()
            .map_err(|e| format!("could not build the CPU executor pool: {e}"))?;
        Ok(Self { pool, workers })
    }

    pub fn workers(&self) -> usize {
        self.workers
    }

    /// How many workers to cut this projection across.
    ///
    /// A policy, deliberately shaped by measurement rather than fixed at
    /// the core count: CPU-1B's large `10240 x 5120` kernel kept scaling
    /// to twelve, but small projections lose to the split, and a future
    /// Q4 kernel with more compute per byte will have its own curve. This
    /// is the one place that judgement lives.
    fn workers_for(&self, kind: CpuParallelism, bytes: usize) -> usize {
        match kind {
            // Already threaded — calling it once is the measured optimum.
            CpuParallelism::LibraryOwned | CpuParallelism::Serial => 1,
            CpuParallelism::ExternalPool => {
                if bytes < MIN_SPLIT_BYTES {
                    1
                } else {
                    self.workers
                }
            }
        }
    }

    /// Run `y = W x` under this executor's threading policy.
    pub fn project(
        &self,
        kernel: &dyn DenseProjector,
        weight: WeightRows<'_>,
        x: &[f32],
        out_dim: usize,
    ) -> Vec<f32> {
        // The SAME site the byte ledger is written, so time and traffic
        // describe one call set rather than two.
        let _t = timed(OpClass::Projection);
        super::replay::record(weight, x, out_dim);
        let in_dim = x.len();
        let mut out = vec![0.0f32; out_dim];
        let workers = if caller_owns_the_machine() {
            1
        } else {
            self.workers_for(kernel.parallelism(), weight.bytes())
        };
        if workers <= 1 || out_dim < workers {
            ledger().record(
                PhysicalProjectionPlan::for_resident(weight, in_dim),
                weight.bytes(),
                1,
            );
            kernel.project_rows(weight, x, &mut out);
            return out;
        }
        // Row-contiguous partitions: each worker streams one unbroken
        // slab of weight, which is what the memory system wants.
        let rows = out_dim.div_ceil(workers);
        ledger().record(
            PhysicalProjectionPlan::for_resident(weight, in_dim),
            weight.bytes(),
            out_dim.div_ceil(rows),
        );
        self.pool.install(|| {
            use rayon::prelude::*;
            out.par_chunks_mut(rows).enumerate().for_each(|(i, slot)| {
                let slab = weight.slice_rows(in_dim, i * rows, slot.len());
                kernel.project_rows(slab, x, slot);
            });
        });
        out
    }
}

/// Whether some enclosing loop is already parallel over this work.
///
/// The batched driver runs whole POSITIONS in parallel and calls the
/// backend from inside that loop; the decode driver runs one position on
/// the caller's own thread. Same backend method, opposite ownership — so
/// the executor asks rather than assumes, and a projection reached from
/// inside a parallel region runs as one call.
///
/// The rule in [`super`] is "at most one layer of parallelism owns the
/// machine for a primitive". Without this the batched path would nest a
/// twelve-worker fan-out inside every position of an already-parallel
/// loop, which is that rule broken from the other end: the seam cannot
/// stop a caller being parallel, only decline to add a second layer.
fn caller_owns_the_machine() -> bool {
    rayon::current_thread_index().is_some()
}

/// Environment variable pinning the pool size.
pub const WORKERS_ENV: &str = "LARQL_CPU_WORKERS";

/// An explicit pool size, when one is asked for.
///
/// Exists so the worker POLICY can be priced on a real decode rather than
/// on a transcription of it. A standalone probe can sweep worker counts
/// all it likes, but the shipped executor is the authority on what the
/// shipped executor does, and rebuilding the binary per point would make
/// the sweep a comparison across builds.
///
/// Ignored unless it parses to at least one — a typo must not silently
/// produce a pool of no workers, which would look like a hang rather than
/// a misconfiguration.
fn requested_workers() -> Option<usize> {
    parse_workers(std::env::var(WORKERS_ENV).ok().as_deref())
}

/// The parse, separated from the environment read.
///
/// Separated so a test can exercise it without setting a process-wide
/// variable the rest of the suite shares — and so the test calls THIS
/// rather than restating the same `trim().parse().filter()` chain, which
/// would pass whatever the real one did.
pub(super) fn parse_workers(raw: Option<&str>) -> Option<usize> {
    raw?.trim().parse::<usize>().ok().filter(|n| *n >= 1)
}

/// Performance-core count where the machine reports one.
///
/// `None` on a machine that does not draw the distinction, which is most
/// of them — the split is an Apple-silicon fact, not a universal one.
fn reported_performance_cores() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        if let Some(n) = sysctl_usize("hw.perflevel0.logicalcpu") {
            return Some(n);
        }
    }
    None
}

/// Fewest workers below which the memory system is no longer saturated.
///
/// Measured on the real decode: 3 workers cost 589 ms/token and 2 cost
/// 836, against 484-490 anywhere from 4 to 8. The cliff is sharp on this
/// side, so the floor is not a taste.
const MIN_STREAMING_WORKERS: usize = 4;

/// Turn a reported performance-core count into a pool size.
///
/// **Half the performance cores, because a streaming kernel saturates the
/// memory system long before it runs out of cores.** Past saturation an
/// extra worker adds a dispatch to every projection and no bandwidth, and
/// the executor was paying that on every one of the 417 compact
/// projections a Qwen3.8 token runs.
///
/// Measured through this executor on the real model, interleaved, two
/// passes agreeing within 0.5%:
///
/// ```text
///  workers   ms/token   tok/s
///       12        518    1.93
///       10        494    2.02
///        8        489    2.05
///        6        484    2.06
///        4        487    2.05
///        3        589    1.70
/// ```
///
/// The leaf ledger says the whole difference is inside the projection
/// class — 488.7 ms at twelve against 440.3 ms at six, with the
/// recurrence and the elementwise glue unchanged to 0.5%. So this is
/// bandwidth saturation and dispatch cost, NOT the pool competing with
/// the caller's own work, and the policy is about the memory system
/// rather than about leaving a core free.
///
/// The basin is flat from 4 to 8 (within 1.3%), so being wrong by two
/// either way costs about one percent — the same robustness the L2
/// threshold has, and the reason half-the-cores is defensible as an
/// extrapolation from one machine rather than a fitted constant.
/// [`WORKERS_ENV`] re-calibrates it where that extrapolation does not
/// hold.
///
/// Split out from the query so BOTH answers are reachable from a test on
/// one machine. The fallback is the branch a macOS-only test can never
/// take: every non-Apple target reaches it, and a `0` from a machine that
/// reports nonsense must not produce a pool of no workers.
pub(super) fn workers_from(reported: Option<usize>) -> usize {
    let cores = match reported {
        Some(n) if n >= 1 => n,
        // Reported zero, or not reported at all: the total core count is
        // the honest answer, and one is the floor.
        _ => std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    };
    (cores / 2)
        .clamp(1, cores)
        .max(MIN_STREAMING_WORKERS.min(cores))
}

/// One integer `sysctl`, or `None` where the machine does not report it.
///
/// Shared with [`super::physical`] so the worker count and the format
/// threshold read the machine through one path — they are two questions
/// about the same hardware, and two spellings of the query is how they
/// start disagreeing about which machine they are on.
#[cfg(target_os = "macos")]
pub(super) fn sysctl_usize(name: &str) -> Option<usize> {
    std::process::Command::new("sysctl")
        .args(["-n", name])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse().ok())
}
