//! `--residency-curve`: per-pass residency and process-resource sampling.

use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::cpu::{self};
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prepared::{ResidencyDelta, ResidencyObservation};
use larql_vindex::format::vindex3::opplan::exec::stages;
use larql_vindex::format::vindex3::opplan::exec::timing;
use larql_vindex::format::vindex3::opplan::ComponentOpPlan;
use std::path::Path;
use std::time::Instant;

use super::residency_record::{self, RunIdentity};

#[allow(unused_imports)]
use super::*;

/// Above this share of the token, the ledger is reporting its own gaps
/// rather than the model's costs.
pub(super) const UNATTRIBUTED_LIMIT: f64 = 5.0;

/// The measured non-projection cost of one token on this build, as a
/// RANGE rather than a point.
///
/// Everything that is not a dense projection: the Gated DeltaNet
/// recurrence (13.3 ms after CPU-2D1), the convolution, the norms, the
/// attention core and the glue. Quantisation does not touch any of it —
/// at Q8 the projections halve and this does not move — so a decode
/// prediction that omitted it would improve without bound as the weights
/// shrank.
pub(super) const NON_PROJECTION_FLOOR_MS: [f64; 2] = [17.0, 24.0];

/// Comma-separated ids, the same shape `--tokens` accepts, so a run's
/// output can be fed straight back in as a prompt.
pub(super) fn join_ids(ids: &[u32]) -> String {
    ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
}

// ── The residency curve: predicted before a byte, observed between tokens ──

/// What the process has been charged so far, from `getrusage`: peak RSS
/// and page faults. Deltas between two readings are what one token
/// cost; the absolute peak is a fact about the whole run.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct ProcessResources {
    pub(super) max_rss: u64,
    pub(super) minor_faults: u64,
    pub(super) major_faults: u64,
}

pub(super) fn process_resources() -> ProcessResources {
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

pub(super) const GB: f64 = 1e9;
pub(super) const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The K3 vertical's rung 7: one prepared image, `repeat` passes of the
/// same prompt and greedy decode, each pass with fresh continuation
/// state, and between every token the OBSERVED residency beside what the
/// plan PREDICTED before any byte was read — the mapping's address space
/// against the pages of it resident, page faults, peak RSS, and where the
/// token's time went by timing class. The first pass is cold; the later
/// passes are what the page cache kept.
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn run_residency_curve<B: PlanBackend>(
    backend: &B,
    engine: &str,
    prompt: &[u32],
    new_tokens: usize,
    plan: &ComponentOpPlan,
    store: &OperandStore,
    repeat: usize,
    warmup: usize,
    unquiet_ok: bool,
    expert_access: &str,
    container: &Path,
    regions_out: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    use larql_vindex::format::vindex3::opplan::exec::accounting::{
        expectations, BlockGeometry, ResourceLedger,
    };
    use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
    use larql_vindex::format::vindex3::opplan::exec::routing_trace;
    use larql_vindex::format::vindex3::opplan::exec::timing::OpClass;

    if prompt.is_empty() {
        return Err("prompt holds no tokens — nothing to condition on".into());
    }
    let access = larql_vindex::format::vindex3::opplan::exec::realization::MappedAccess::parse(
        expert_access,
    )?;
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
    let budget =
        larql_vindex::format::vindex3::opplan::exec::accounting::ResidencyBudget::UNBOUNDED
            .with_expert_access(access);
    let ops = PreparedOperands::load_within(plan, store, backend, ExecutionSlice::Full, &budget)?;
    let load_seconds = loading.elapsed().as_secs_f64();
    println!("engine: {engine}");
    println!(
        "weights bound in {load_seconds:.1} s (once, for {repeat} pass(es)); expert access: {}",
        access.name()
    );

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
    let binding = ops.observe_residency();
    let mapped = binding.summary();
    let base = process_resources();
    // The per-region record, written before any pass runs so it adds no
    // time to a timed region.
    let run = RunIdentity::for_container(container);
    let mut regions_file = match regions_out {
        Some(path) => {
            let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
            residency_record::write_observation(&mut file, &run, 0, "binding", &binding)?;
            Some(file)
        }
        None => None,
    };
    println!("observed, after binding:");
    println!(
        "  committed             {:>10.2} GB",
        census.total() as f64 / GB
    );
    println!("{}", residency_record::binding_line(&mapped));
    if let Some(line) = residency_record::incomplete_line(&binding) {
        println!("{line}");
    }
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
        let mut kv = crate::commands::primary::continuation::select_for(plan, None)?.build();
        let mut session = DecodeSession::over_prepared(plan, &ops, backend, &mut *kv)?;
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
            "  {:<8} {:>9} {:>12} {:>12} {:>9} {:>10} {:>10} {:>9}  stages (ms)",
            "step", "ms", "read res", "Δ res (GB)", "excluded", "minor Δ", "major Δ", "rss GB"
        );
        let mut before_res = session.observe_residency();
        let mut before_proc = process_resources();
        let observe = |name: String,
                       elapsed: f64,
                       session: &DecodeSession<'_, B>,
                       before_res: &mut ResidencyObservation,
                       before_proc: &mut ProcessResources|
         -> StepObservation {
            let now_res = session.observe_residency();
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
                resident_delta: ResidencyDelta::between(before_res, &now_res),
                minor_faults: now_proc
                    .minor_faults
                    .saturating_sub(before_proc.minor_faults),
                major_faults: now_proc
                    .major_faults
                    .saturating_sub(before_proc.major_faults),
                stage_ms: stages::Stage::ALL.map(|st| stage_ledger.get(st).nanos as f64 / 1e6),
                stages_nested: stage_ledger.nested(),
            };
            println!(
                "  {:<8} {:>9.0} {:>9.3} GB {:>12} {:>9} {:>10} {:>10} {:>9.2}  {}  [leaves {}]",
                name,
                obs.elapsed_ms,
                now_res.summary().resident_bytes as f64 / GB,
                residency_record::delta_cell(&obs.resident_delta),
                obs.resident_delta.excluded,
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
    if let Some(file) = regions_file.as_mut() {
        let after = ops.observe_residency();
        residency_record::write_observation(file, &run, 1, "after_run", &after)?;
        std::io::Write::flush(file)?;
    }
    summarise_passes(&samples, repeat.max(1), warmup, &qualification);
    Ok(())
}

/// One step's observation, as the curve reads it between tokens.
#[derive(Debug, Clone, Copy)]
pub(super) struct StepObservation {
    pub(super) elapsed_ms: f64,
    /// Per-region change over the regions read in both observations.
    pub(super) resident_delta: ResidencyDelta,
    pub(super) minor_faults: u64,
    pub(super) major_faults: u64,
    /// Milliseconds per [`stages::Stage::ALL`] entry, in that order.
    pub(super) stage_ms: [f64; stages::Stage::ALL.len()],
    pub(super) stages_nested: u64,
}

/// The characterised token of one counted pass.
#[derive(Debug, Clone)]
pub(super) struct PassSample {
    pub(super) pass: usize,
    pub(super) observation: StepObservation,
    pub(super) routing_fingerprint: u64,
    pub(super) routed_layers: usize,
    pub(super) first_layer_experts: Vec<usize>,
    pub(super) logits_delta: f32,
}

/// Nearest-rank percentile of a sorted sample.
pub(super) fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

pub(super) fn stats(values: impl Iterator<Item = f64>) -> (f64, f64, f64, f64) {
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
pub(super) fn summarise_passes(
    samples: &[PassSample],
    repeat: usize,
    warmup: usize,
    qualification: &str,
) {
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
    let measured: Vec<i64> = samples
        .iter()
        .filter_map(|s| s.observation.resident_delta.signed_total())
        .collect();
    let unmeasured = samples.len() - measured.len();
    let resident = if measured.is_empty() {
        "n/a".to_string()
    } else {
        format!("{:.3}", measured.iter().sum::<i64>() as f64 / GB)
    };
    println!(
        "  faults: major {major} total, minor ≤ {minor_max} per pass; resident-page delta {resident} GB total over {} pass(es){}",
        measured.len(),
        if unmeasured > 0 {
            format!(", {unmeasured} with no comparable region")
        } else {
            String::new()
        }
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

pub(super) fn argmax(v: &[f32]) -> Option<(usize, f32)> {
    v.iter()
        .copied()
        .enumerate()
        .fold(None, |best, (i, x)| match best {
            Some((_, b)) if b >= x => best,
            _ => Some((i, x)),
        })
}
