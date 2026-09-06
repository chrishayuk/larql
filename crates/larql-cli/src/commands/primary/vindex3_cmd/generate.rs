//! `larql vindex3 exec --generate N` — greedy autoregressive decode
//! from the container's own program, with the research report around it.
//!
//! Runs on a [`DecodeSession`]: every operand is loaded once (in the
//! backend's declared weight format, so a device buffer cache can keep
//! the model resident) and each token advances one position against the
//! session's KV cache. The loop itself is [`greedy_decode`]; this file
//! is what the exec verb wraps around it — the per-token trace, the
//! phase timings (weight load, prompt ingestion, first generated token,
//! steady decode, reported separately because they are different costs
//! and conflating them is how a decode number lies), the residency and
//! allocation censuses, the weight-traffic ledger and the optional
//! projection replay.
//!
//! Token ids go in and come out as ids — a tokenizer is part of the
//! fixture and lives outside this verb. `larql run` is the wrapper that
//! owns one.

use std::time::Instant;

use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::cpu::{self, PhysicalProjectionPlan, PlanTally};
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prepared::{AllocationCensus, ResidencyCensus};
use larql_vindex::format::vindex3::opplan::exec::timing;
use larql_vindex::format::vindex3::opplan::ComponentOpPlan;

use super::decode::{greedy_decode, DecodeReport, Flow};

/// Greedy decode with the exec verb's report. Returns the generated ids
/// (the prompt excluded), so a caller has them as data and not only as
/// lines on the console.
pub(super) fn run_generate<B: PlanBackend>(
    backend: &B,
    engine: &str,
    prompt: &[u32],
    new_tokens: usize,
    plan: &ComponentOpPlan,
    store: &OperandStore,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    // Admission BEFORE any work: this is the only point at which the
    // load average is about the machine rather than about us.
    let replaying = std::env::var("LARQL_REPLAY_PROJECTIONS").is_ok();
    if replaying && !admitted(cpu::environment::Phase::BeforeWork)? {
        return Ok(Vec::new());
    }

    let loading = Instant::now();
    let mut session = DecodeSession::new(plan, store, backend)?;
    let load_seconds = loading.elapsed().as_secs_f64();
    eprintln!("weights resident in {load_seconds:.1} s");
    report_residency(&session.residency_census());
    report_allocations(&session.allocation_census());
    // What was pinned, from the structured records; the rendering is
    // presentation only.
    eprint!(
        "{}",
        larql_vindex::format::vindex3::opplan::exec::accounting::render_selection_summary(
            session.realizations()
        )
    );

    let mut emitted = 0usize;
    let decoded = greedy_decode(&mut session, prompt, new_tokens, &mut |id, value| {
        emitted += 1;
        eprintln!(
            "token {emitted:>3}/{new_tokens}  id {id:<8} ({value:+.3})  context {}",
            prompt.len() + emitted,
        );
        Ok(Flow::Continue)
    })?;
    let generated = decoded.generated;
    let sequence: Vec<u32> = prompt.iter().chain(&generated).copied().collect();

    println!("engine: {engine}");
    println!("prompt tokens: {}", prompt.len());
    println!("generated ids: {}", join_ids(&generated));
    println!("sequence ids: {}", join_ids(&sequence));
    println!("weights loaded: {load_seconds:.1} s");
    println!(
        "prompt: {} tokens in {:.1} s ({:.0} ms/token) — first new token ready",
        prompt.len(),
        decoded.prompt_seconds,
        decoded.prompt_seconds * 1e3 / prompt.len().max(1) as f64,
    );
    if let Some(report) = DecodeReport::from_steps(&decoded.step_seconds) {
        println!("decode tokens: {}", report.decode_tokens);
        println!("decode elapsed: {:.1} s", report.decode_seconds);
        println!(
            "mean: {:.0} ms/token ({:.3} tok/s)",
            report.mean_seconds_per_token * 1e3,
            report.mean_seconds_per_token.recip(),
        );
        println!(
            "steady (last half): {:.0} ms/token ({:.3} tok/s)",
            report.steady_seconds_per_token * 1e3,
            report.steady_seconds_per_token.recip(),
        );
        // Split the token between device dispatch and everything else.
        // "Everything else" is the interpreter's elementwise glue —
        // norms, RoPE, softmax over the KV cache, activations,
        // residuals — which is a fixed per-token cost just as
        // submission is, and which a bytes-vs-time fit cannot separate
        // from it.
        if let Some(stats) = backend.dispatch_stats() {
            let device_s = stats.device_nanos as f64 / 1e9;
            let per_token = device_s / (report.decode_tokens + prompt.len()) as f64;
            println!(
                "device: {:.0} ms/token in {} submissions/token ({:.0} us each)",
                per_token * 1e3,
                stats.submissions / (report.decode_tokens + prompt.len()) as u64,
                per_token * 1e6
                    / (stats.submissions as f64 / (report.decode_tokens + prompt.len()) as f64),
            );
            println!(
                "glue:   {:.0} ms/token (everything not inside a device call)",
                (report.mean_seconds_per_token - per_token) * 1e3,
            );
        }
    }
    if let Some((seconds, tallies)) = decoded.priced_step {
        report_projections(seconds, &tallies);
        report_leaves(seconds);
    }
    if replaying {
        replay_projections(&mut session)?;
    }
    Ok(generated)
}

/// The prepared image's bytes, site by site.
///
/// Site by site because a single total cannot fail usefully: "the model
/// is smaller" is satisfied just as well by a stack that halved its FFN
/// and left 11 GB of recurrence widened.
fn report_residency(census: &ResidencyCensus) {
    println!(
        "residency: {:.2} GB total — {:.2} GB compact, {:.2} GB widened f32",
        census.total() as f64 / 1e9,
        census.compact() as f64 / 1e9,
        census.widened_f32() as f64 / 1e9,
    );
    for (site, bytes) in census.sites() {
        if bytes.total() == 0 {
            continue;
        }
        println!(
            "  {site:<10} {:>8.2} GB  ({:.2} compact / {:.2} widened f32)",
            bytes.total() as f64 / 1e9,
            bytes.compact as f64 / 1e9,
            bytes.widened_f32 as f64 / 1e9,
        );
    }
}

/// What the CPU executor actually ran for one steady step.
///
/// The counterpart to the residency census, and not a restatement of it:
/// residency is what the loader decided, this is what the kernels read.
/// A path that kept bf16 resident and widened a scratch tile before
/// computing would satisfy the census and show up here as `blas-f32` at
/// twice the bytes.
/// Whether this machine may be measured on, reporting either way.
///
/// Refuses rather than warns. Three contamination events in one session
/// were each caught only because a number happened to be absurd, and a
/// caveat attached to a plausible one does not survive contact with the
/// table the number ends up in.
fn admitted(phase: cpu::environment::Phase) -> Result<bool, Box<dyn std::error::Error>> {
    let environment = cpu::Environment::read();
    let refusals = environment.disqualifiers(phase);
    println!("  machine ({phase:?}): {}", environment.describe());
    if refusals.is_empty() {
        return Ok(true);
    }
    println!("  REFUSING to measure — this machine is not quiet:");
    for reason in &refusals {
        println!("    - {reason}");
    }
    println!("  Nothing is reported: a contaminated replay would calibrate the cost");
    println!("  model against whatever else was running.");
    Ok(false)
}

/// **CPU-PERF-3B.** Replay one steady token's projections against the
/// operands the model is already holding.
///
/// Everything else is removed — no norm, no recurrence, no attention, no
/// activation — so the only difference from the synthetic shape harness
/// is that these are the REAL resident operands, 369 of them spanning 27
/// GB at Q8, rather than one matrix exercised in a loop.
///
/// The ordering arms are diagnostic and not proposals: grouped separates
/// a temporal-locality effect from a cost intrinsic to traversing
/// hundreds of distinct allocations, and shuffled checks the same thing
/// from the other side.
fn replay_projections<B: PlanBackend>(
    session: &mut DecodeSession<'_, B>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Capture one more step, which the report above has already priced,
    // so the replay and the ledger describe the same call set.
    // Sanity AFTER our own load phase: external signals only. Opening a
    // 51 GB model is sixteen seconds of every core, so a raised
    // one-minute average at this point is US, and refusing for it would
    // be refusing LARQL for being LARQL.
    if !admitted(cpu::environment::Phase::AfterWork)? {
        return Ok(());
    }

    cpu::start_capture();
    session.step(0)?;
    let calls = cpu::take_capture();
    if calls.is_empty() {
        println!("\nreplay: nothing captured — the step issued no projections");
        return Ok(());
    }
    let bytes = cpu::replay::captured_bytes(&calls);
    let exec = cpu::shared()?;
    println!(
        "\n  projection replay: {} calls, {:.2} GB, against the resident model\n",
        calls.len(),
        bytes as f64 / 1e9
    );
    // INTERLEAVED, not one arm after another. Run sequentially, a
    // machine that loads up during the measurement puts its entire drift
    // on the later arms and manufactures an ordering effect — which is
    // exactly what a contaminated first attempt produced (532 / 703 / 703
    // ms, monotonic in run order rather than in arm).
    let mut best = [f64::INFINITY; cpu::ReplayOrder::ALL.len()];
    for _ in 0..3 {
        for (i, order) in cpu::ReplayOrder::ALL.into_iter().enumerate() {
            // SAFETY: `session` owns the operands for this whole scope,
            // so every captured address is still resident and unmoved.
            best[i] = best[i].min(unsafe { cpu::replay(exec, &calls, order) });
        }
    }
    for (i, order) in cpu::ReplayOrder::ALL.into_iter().enumerate() {
        println!(
            "  {:<20} {:>8.1} ms   {:>6.1} GB/s",
            order.name(),
            best[i] * 1e3,
            bytes as f64 / best[i] / 1e9
        );
    }
    println!();
    Ok(())
}

/// Where the operand allocations landed, as distinct from how big they
/// are.
///
/// Printed beside the byte census because the two answer different
/// questions and one of them is currently unexplained: an isolated kernel
/// harness predicts real bf16 projection to +0.7% and misses real Q8 by
/// 7.9%, and the formats differ in allocation COUNT and ALIGNMENT as well
/// as in bytes.
fn report_allocations(census: &AllocationCensus) {
    println!(
        "allocations: {} holding {:.2} GB — {} page-aligned ({:.0}%), common alignment {} B",
        census.allocations,
        census.bytes as f64 / 1e9,
        census.page_aligned,
        census.page_aligned as f64 / census.allocations.max(1) as f64 * 100.0,
        census.common_alignment,
    );
}

fn report_projections(seconds: f64, tallies: &[(PhysicalProjectionPlan, PlanTally)]) {
    let total: u64 = tallies.iter().map(|(_, t)| t.bytes).sum();
    println!(
        "projections (one steady step): {:.2} GB over {} calls in {:.0} ms — {:.0} GB/s",
        total as f64 / 1e9,
        tallies.iter().map(|(_, t)| t.calls).sum::<u64>(),
        seconds * 1e3,
        total as f64 / seconds / 1e9,
    );
    for (plan, t) in tallies {
        if t.calls == 0 {
            continue;
        }
        println!(
            "  {:<12} {:>8.2} GB over {:>4} calls, {:>5} worker slabs   {}",
            format!("{plan:?}"),
            t.bytes as f64 / 1e9,
            t.calls,
            t.slabs,
            plan.arithmetic(),
        );
    }
    report_budget(tallies);
}

/// **The bridge from a byte census to a decode claim.**
///
/// The measured time above is what THIS run did. This is what the same
/// plan is PREDICTED to cost from its bytes alone, at the rates CPU-4Y
/// measured — which is the only way an exception-set search is
/// affordable, since a recipe that had to be benchmarked end to end
/// would cost about an hour per candidate.
///
/// Printed beside the measurement on purpose: a predicted number next to
/// the measurement it is meant to reproduce is auditable, and one
/// printed alone is a claim.
fn report_budget(tallies: &[(PhysicalProjectionPlan, PlanTally)]) {
    let budget = cpu::budget(tallies);
    if budget.rows.is_empty() {
        return;
    }
    println!(
        "predicted from bytes: {:.1} ms synthetic, {:.1} ms real (x{:.3}, measured CPU-PERF-3B)",
        budget.synthetic_ms,
        budget.predicted_ms,
        cpu::SYNTHETIC_TO_REAL,
    );
    for row in &budget.rows {
        println!(
            "  {:<12} {:>8.2} GB at {:>6.2} GB/s = {:>7.1} ms",
            format!("{:?}", row.plan),
            row.bytes as f64 / 1e9,
            row.rate_gbps,
            row.synthetic_ms,
        );
    }
    // The floor is the term quantisation does not touch, so a decode
    // prediction that omitted it would improve without limit as the
    // weights shrank.
    for floor in NON_PROJECTION_FLOOR_MS {
        println!(
            "  + {floor:>4.0} ms non-projection floor -> {:>6.1} ms/token, {:.2} tok/s",
            budget.predicted_ms + floor,
            cpu::predicted_tokens_per_second(&budget, floor),
        );
    }
}

/// **Where the token's milliseconds went.**
///
/// The counterpart to the byte ledger, at the same call sites. Ends with
/// the reconciliation rather than the classes, because the classes alone
/// invite reading a table and skipping the part that says whether the
/// table is complete.
///
/// `unattributed` is a FAILING DIAGNOSTIC, not a bucket. Above
/// `UNATTRIBUTED_LIMIT` the instrumentation is incomplete and the right
/// response is to find the missing boundary — not to optimise the
/// largest named class, and not to name the gap and move on.
fn report_leaves(seconds: f64) {
    let l = timing::ledger();
    let nested = l.nested();
    let mut rows: Vec<_> = l.all().into_iter().filter(|(_, t)| t.calls > 0).collect();
    rows.sort_by_key(|(_, t)| std::cmp::Reverse(t.nanos));

    println!("\n  where the token went (one steady step):");
    println!(
        "  {:<18} {:>7} {:>10} {:>10} {:>8}",
        "class", "calls", "total ms", "us/call", "% token"
    );
    let wall_ns = seconds * 1e9;
    for (class, t) in &rows {
        println!(
            "  {:<18} {:>7} {:>9.2}  {:>9.2} {:>7.1}%",
            class.name(),
            t.calls,
            t.nanos as f64 / 1e6,
            t.nanos_per_call() / 1e3,
            t.nanos as f64 / wall_ns * 100.0,
        );
    }

    let timed_ns = l.total_nanos() as f64;
    let unattributed = wall_ns - timed_ns;
    let share = unattributed / wall_ns * 100.0;
    println!("  {:-<58}", "");
    println!("  {:<18} {:>28.2} ms", "timed leaves", timed_ns / 1e6);
    println!(
        "  {:<18} {:>28.2} ms  {:>6.1}%",
        "unattributed",
        unattributed / 1e6,
        share
    );
    println!("  {:<18} {:>28.2} ms", "steady token wall", wall_ns / 1e6);
    if nested > 0 {
        println!(
            "\n  REFUSING TO RECONCILE: {nested} overlapping timers. Leaves that nest \
             double-count, so the total above is not a sum of disjoint work."
        );
    } else if share.abs() > UNATTRIBUTED_LIMIT {
        println!(
            "\n  INCOMPLETE: {share:.1}% unattributed exceeds {UNATTRIBUTED_LIMIT:.0}%. A \
             boundary is missing — find it before optimising any class above."
        );
    }
}

/// Above this share of the token, the ledger is reporting its own gaps
/// rather than the model's costs.
const UNATTRIBUTED_LIMIT: f64 = 5.0;

/// The measured non-projection cost of one token on this build, as a
/// RANGE rather than a point.
///
/// Everything that is not a dense projection: the Gated DeltaNet
/// recurrence (13.3 ms after CPU-2D1), the convolution, the norms, the
/// attention core and the glue. Quantisation does not touch any of it —
/// at Q8 the projections halve and this does not move — so a decode
/// prediction that omitted it would improve without bound as the weights
/// shrank.
const NON_PROJECTION_FLOOR_MS: [f64; 2] = [17.0, 24.0];

/// Comma-separated ids, the same shape `--tokens` accepts, so a run's
/// output can be fed straight back in as a prompt.
fn join_ids(ids: &[u32]) -> String {
    ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
}

// ── The residency curve: predicted before a byte, observed between tokens ──

/// What the process has been charged so far, from `getrusage`: peak RSS
/// and page faults. Deltas between two readings are what one token
/// cost; the absolute peak is a fact about the whole run.
#[derive(Debug, Clone, Copy, Default)]
struct ProcessResources {
    max_rss: u64,
    minor_faults: u64,
    major_faults: u64,
}

fn process_resources() -> ProcessResources {
    #[cfg(unix)]
    {
        // SAFETY: `usage` is read only after `getrusage` returns 0, which
        // is documented to fully populate the struct.
        let usage = unsafe {
            let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
            if libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) != 0 {
                return ProcessResources::default();
            }
            usage.assume_init()
        };
        // macOS reports ru_maxrss in bytes; other POSIX targets in KiB.
        let unit: u64 = if cfg!(target_os = "macos") { 1 } else { 1024 };
        ProcessResources {
            max_rss: usage.ru_maxrss.max(0) as u64 * unit,
            minor_faults: usage.ru_minflt.max(0) as u64,
            major_faults: usage.ru_majflt.max(0) as u64,
        }
    }
    #[cfg(not(unix))]
    {
        ProcessResources::default()
    }
}

const GB: f64 = 1e9;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The K3 vertical's rung 7: one prepared image, `repeat` passes of the
/// same prompt and greedy decode, each pass with fresh continuation
/// state, and between every token the OBSERVED residency beside what the
/// plan PREDICTED before any byte was read — the mapping's address space
/// against the pages of it resident, page faults, peak RSS, and where the
/// token's time went by timing class. The first pass is cold; the later
/// passes are what the page cache kept.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_residency_curve<B: PlanBackend>(
    backend: &B,
    engine: &str,
    prompt: &[u32],
    new_tokens: usize,
    plan: &ComponentOpPlan,
    store: &OperandStore,
    repeat: usize,
    warmup: usize,
    unquiet_ok: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use larql_vindex::format::vindex3::opplan::exec::accounting::{
        expectations, BlockGeometry, ResourceLedger,
    };
    use larql_vindex::format::vindex3::opplan::exec::kv::RowKvState;
    use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
    use larql_vindex::format::vindex3::opplan::exec::timing::OpClass;
    use larql_vindex::format::vindex3::opplan::exec::{routing_trace, stages};

    if prompt.is_empty() {
        return Err("prompt holds no tokens — nothing to condition on".into());
    }
    if warmup >= repeat.max(1) {
        return Err(format!("warmup {warmup} leaves no counted pass out of {repeat}").into());
    }
    // Scheduling, declared before anything is timed: the pool the
    // production backend joins inside every step, and the machine the
    // probe sees. The boundary is the step's return — the CPU backend
    // is synchronous and nothing of a token runs past it.
    println!(
        "scheduling: cpu pool {} workers of {} available; timing boundary = step return (synchronous CPU backend)",
        cpu::shared().map(|e| e.workers()).unwrap_or(0),
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
    );
    let environment = cpu::Environment::read();
    let disqualifiers = environment.disqualifiers(cpu::environment::Phase::BeforeWork);
    println!("machine: {}", environment.describe());
    let qualification = if disqualifiers.is_empty() {
        "QUALIFIED".to_string()
    } else {
        let reasons: Vec<String> = disqualifiers.iter().map(|d| d.to_string()).collect();
        if !unquiet_ok {
            return Err(format!(
                "this machine is not quiet — {} — so the curve would calibrate against whatever \
                 else is running; pass --unquiet-ok to run anyway, labelled",
                reasons.join("; ")
            )
            .into());
        }
        format!("UNQUALIFIED ({})", reasons.join("; "))
    };
    println!("qualification: {qualification}");
    let loading = Instant::now();
    let ops = PreparedOperands::load(plan, store, backend, ExecutionSlice::Full)?;
    let load_seconds = loading.elapsed().as_secs_f64();
    println!("engine: {engine}");
    println!("weights bound in {load_seconds:.1} s (once, for {repeat} pass(es))");

    // PREDICTED, from the pins and the tensor tables.
    let priced = expectations(
        ops.realizations(),
        |op| store.stored_len(op),
        BlockGeometry::executor(),
    );
    let ledger = ResourceLedger::aggregate(&priced);
    println!("predicted, before any payload byte:");
    println!(
        "  mapped address space  {:>10.2} GB",
        ledger.mapped as f64 / GB
    );
    println!(
        "  persistent resident   {:>10.2} GB",
        ledger.resident as f64 / GB
    );
    println!(
        "  transient peak        {:>10.2} GB",
        ledger.transient_peak as f64 / GB
    );
    println!(
        "  touch per token       {:>10.3} GB",
        ledger.touch_per_token as f64 / GB
    );
    println!(
        "  cold page-in / token  {:>10.3} GB",
        ledger.page_in_per_token as f64 / GB
    );
    println!(
        "  physical working set  {:>10.2} GiB",
        ledger.physical_working_set() as f64 / GIB
    );

    // OBSERVED, at binding.
    let census = ops.residency_census();
    let mapped = ops.mapped_residency();
    let base = process_resources();
    println!("observed, after binding:");
    println!(
        "  committed             {:>10.2} GB",
        census.total() as f64 / GB
    );
    println!(
        "  mapped                {:>10.2} GB address space over {} regions, {:.3} GB resident",
        mapped.mapped_bytes as f64 / GB,
        mapped.regions,
        mapped.resident_bytes as f64 / GB
    );
    println!(
        "  peak rss              {:>10.2} GB   faults minor {} major {}",
        base.max_rss as f64 / GB,
        base.minor_faults,
        base.major_faults
    );

    let counted_from = warmup + 1;
    let mut samples: Vec<PassSample> = Vec::new();
    let mut first_logits: Option<Vec<f32>> = None;
    let mut first_generated: Option<Vec<u32>> = None;
    for pass in 1..=repeat.max(1) {
        let mut kv = RowKvState::default();
        let mut session = DecodeSession::over_prepared(plan, &ops, backend, &mut kv)?;
        let label = if pass < counted_from {
            "warmup"
        } else if pass == 1 {
            "cold"
        } else {
            "counted"
        };
        println!(
            "pass {pass} ({label}): machine {}",
            cpu::Environment::read().describe()
        );
        println!(
            "  {:<8} {:>9} {:>12} {:>12} {:>10} {:>10} {:>9}  stages (ms)",
            "step", "ms", "mapped res", "Δ res (GB)", "minor Δ", "major Δ", "rss GB"
        );
        let mut before_res = session.mapped_residency();
        let mut before_proc = process_resources();
        let observe = |name: String,
                       elapsed: f64,
                       session: &DecodeSession<'_, B>,
                       before_res: &mut larql_vindex::format::vindex3::opplan::exec::prepared::MappedResidency,
                       before_proc: &mut ProcessResources|
         -> StepObservation {
            let now_res = session.mapped_residency();
            let now_proc = process_resources();
            let stage_ledger = stages::ledger();
            let by_stage: Vec<String> = stage_ledger
                .all()
                .iter()
                .filter(|(_, t)| t.calls > 0)
                .map(|(st, t)| format!("{}={:.1}", st.name(), t.nanos as f64 / 1e6))
                .collect();
            let leaves = timing::ledger();
            let leaf_note: Vec<String> = [OpClass::Projection, OpClass::Norm, OpClass::Logits]
                .iter()
                .map(|c| (c, leaves.get(*c)))
                .filter(|(_, t)| t.calls > 0)
                .map(|(c, t)| format!("{}={:.0}", c.name(), t.nanos as f64 / 1e6))
                .collect();
            let obs = StepObservation {
                elapsed_ms: elapsed * 1e3,
                resident_delta: now_res.resident_bytes.saturating_sub(before_res.resident_bytes),
                minor_faults: now_proc.minor_faults.saturating_sub(before_proc.minor_faults),
                major_faults: now_proc.major_faults.saturating_sub(before_proc.major_faults),
                stage_ms: stages::Stage::ALL
                    .map(|st| stage_ledger.get(st).nanos as f64 / 1e6),
                stages_nested: stage_ledger.nested(),
            };
            println!(
                "  {:<8} {:>9.0} {:>9.3} GB {:>12.3} {:>10} {:>10} {:>9.2}  {}  [leaves {}]",
                name,
                obs.elapsed_ms,
                now_res.resident_bytes as f64 / GB,
                obs.resident_delta as f64 / GB,
                obs.minor_faults,
                obs.major_faults,
                now_proc.max_rss as f64 / GB,
                by_stage.join(" "),
                leaf_note.join(" ")
            );
            *before_res = now_res;
            *before_proc = now_proc;
            obs
        };
        // Prompt: every position through the stack, timed as one step.
        timing::ledger().reset();
        stages::ledger().reset();
        let started = Instant::now();
        let mut logits = None;
        for &token in prompt {
            logits = session.step(token)?.logits;
        }
        observe(
            format!("prompt×{}", prompt.len()),
            started.elapsed().as_secs_f64(),
            &session,
            &mut before_res,
            &mut before_proc,
        );
        let mut logits = logits.ok_or("plan carries no output head — cannot generate")?;
        let mut generated = Vec::with_capacity(new_tokens);
        let mut token_sample: Option<PassSample> = None;
        for step in 0..new_tokens {
            let (next, _) = argmax(&logits).ok_or("output head produced no logits")?;
            let id = u32::try_from(next)?;
            generated.push(id);
            if step + 1 == new_tokens {
                break;
            }
            timing::ledger().reset();
            stages::ledger().reset();
            routing_trace::start_capture();
            let started = Instant::now();
            logits = session
                .step(id)?
                .logits
                .ok_or("plan carries no output head — cannot generate")?;
            let elapsed = started.elapsed().as_secs_f64();
            let routing = routing_trace::take_capture();
            let obs = observe(
                format!("token {}", step + 1),
                elapsed,
                &session,
                &mut before_res,
                &mut before_proc,
            );
            if step == 0 {
                // The characterised token: the first generated one, whose
                // state is the prompt's and therefore the same in every pass.
                let logits_delta = match &first_logits {
                    Some(first) => first
                        .iter()
                        .zip(&logits)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max),
                    None => 0.0,
                };
                if first_logits.is_none() {
                    first_logits = Some(logits.clone());
                }
                token_sample = Some(PassSample {
                    pass,
                    observation: obs,
                    routing_fingerprint: routing_trace::fingerprint(&routing),
                    routed_layers: routing.len(),
                    first_layer_experts: routing.first().cloned().unwrap_or_default(),
                    logits_delta,
                });
            }
        }
        println!("  generated ids: {}", join_ids(&generated));
        if let Some(sample) = token_sample {
            println!(
                "  token 1: routing {:016x} over {} routed layers, layer-1 experts {:?}, logits max|Δ| vs pass 1 {:.2e}",
                sample.routing_fingerprint, sample.routed_layers, sample.first_layer_experts, sample.logits_delta
            );
            if first_generated.is_none() {
                first_generated = Some(generated.clone());
            } else if first_generated.as_deref() != Some(generated.as_slice()) {
                println!(
                    "  GENERATED IDS DIFFER from pass 1: {}",
                    join_ids(&generated)
                );
            }
            if pass >= counted_from {
                samples.push(sample);
            }
        }
    }
    summarise_passes(&samples, repeat.max(1), warmup, &qualification);
    Ok(())
}

/// One step's observation, as the curve reads it between tokens.
#[derive(Debug, Clone, Copy)]
struct StepObservation {
    elapsed_ms: f64,
    resident_delta: u64,
    minor_faults: u64,
    major_faults: u64,
    /// Milliseconds per [`stages::Stage::ALL`] entry, in that order.
    stage_ms: [f64; 4],
    stages_nested: u64,
}

/// The characterised token of one counted pass.
#[derive(Debug, Clone)]
struct PassSample {
    pass: usize,
    observation: StepObservation,
    routing_fingerprint: u64,
    routed_layers: usize,
    first_layer_experts: Vec<usize>,
    logits_delta: f32,
}

/// Nearest-rank percentile of a sorted sample.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

fn stats(values: impl Iterator<Item = f64>) -> (f64, f64, f64, f64) {
    let mut v: Vec<f64> = values.collect();
    v.sort_by(|a, b| a.total_cmp(b));
    (
        percentile(&v, 50.0),
        percentile(&v, 95.0),
        v.first().copied().unwrap_or(f64::NAN),
        v.last().copied().unwrap_or(f64::NAN),
    )
}

/// The counted passes, reduced: percentiles of the token and of every
/// stage, and the determinism witnesses — every counted pass routed the
/// same experts, or the statistics are over different work and say so.
fn summarise_passes(samples: &[PassSample], repeat: usize, warmup: usize, qualification: &str) {
    use larql_vindex::format::vindex3::opplan::exec::stages::Stage;
    println!(
        "\ncharacterised token over {} counted pass(es) ({} warmup of {}), machine {}:",
        samples.len(),
        warmup,
        repeat,
        qualification
    );
    if samples.is_empty() {
        println!("  nothing counted");
        return;
    }
    let fingerprints: std::collections::BTreeSet<u64> =
        samples.iter().map(|s| s.routing_fingerprint).collect();
    if fingerprints.len() == 1 {
        println!(
            "  routing: IDENTICAL in every counted pass ({:016x}, {} routed layers)",
            samples[0].routing_fingerprint, samples[0].routed_layers
        );
    } else {
        println!(
            "  routing: DIFFERS across passes — {} distinct selections; the statistics below are over different work",
            fingerprints.len()
        );
        for s in samples {
            println!("    pass {} {:016x}", s.pass, s.routing_fingerprint);
        }
    }
    let nested: u64 = samples.iter().map(|s| s.observation.stages_nested).sum();
    if nested > 0 {
        println!("  REFUSING TO RECONCILE STAGES: {nested} nested stage timers across the counted passes");
    }
    let worst_logits = samples
        .iter()
        .map(|s| s.logits_delta)
        .fold(0.0f32, f32::max);
    println!("  logits max|Δ| vs pass 1 across counted passes: {worst_logits:.2e}");
    let major: u64 = samples.iter().map(|s| s.observation.major_faults).sum();
    let minor_max = samples
        .iter()
        .map(|s| s.observation.minor_faults)
        .max()
        .unwrap_or(0);
    let resident: u64 = samples.iter().map(|s| s.observation.resident_delta).sum();
    println!(
        "  faults: major {major} total, minor ≤ {minor_max} per pass; resident-page delta {} GB total",
        resident as f64 / GB
    );
    println!(
        "  {:<16} {:>9} {:>9} {:>9} {:>9} {:>8}",
        "ms", "p50", "p95", "min", "max", "p95/p50"
    );
    let (p50, p95, min, max) = stats(samples.iter().map(|s| s.observation.elapsed_ms));
    println!(
        "  {:<16} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>8.2}",
        "token",
        p50,
        p95,
        min,
        max,
        p95 / p50
    );
    let token_p50 = p50;
    let mut staged = 0.0;
    for (i, st) in Stage::ALL.iter().enumerate() {
        let (p50, p95, min, max) = stats(samples.iter().map(|s| s.observation.stage_ms[i]));
        staged += p50;
        println!(
            "  {:<16} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>8.2}   {:>5.1}% of token p50",
            st.name(),
            p50,
            p95,
            min,
            max,
            p95 / p50,
            p50 / token_p50 * 100.0
        );
    }
    println!(
        "  {:<16} {:>9.1}   ({:.1}% of token p50: norms, residual, embedding, head, glue)",
        "other",
        token_p50 - staged,
        (token_p50 - staged) / token_p50 * 100.0
    );
}

fn argmax(v: &[f32]) -> Option<(usize, f32)> {
    v.iter()
        .copied()
        .enumerate()
        .fold(None, |best, (i, x)| match best {
            Some((_, b)) if b >= x => best,
            _ => Some((i, x)),
        })
}
