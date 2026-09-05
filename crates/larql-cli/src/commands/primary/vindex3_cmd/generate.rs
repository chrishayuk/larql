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
