//! `--residency-curve`: per-pass residency and process-resource sampling.

use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::cpu::{self};
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prepared::{ResidencyDelta, ResidencyObservation};
use larql_vindex::format::vindex3::opplan::exec::stages;
use larql_vindex::format::vindex3::opplan::exec::timing;
use larql_vindex::format::vindex3::opplan::ComponentOpPlan;
use serde_json::json;
use std::io::BufWriter;
use std::path::Path;
use std::time::Instant;

use super::admission::{
    admission_line, commit_admission, create_evidence_file, decide, plan_evidence,
    AdmissionCommitted, EvidenceRequest, EvidenceSink,
};
use super::execution_evidence::{
    after_run_execution, class_with_execution, effective_execution, execution_line,
    selection_block, selection_problems,
};
use super::instance_evidence::{
    after_run_record, before_binding_record, instance_block, instance_line, judge,
    post_admission_reads, EvidenceClass, InstanceJudgement, VerificationManifest,
};
use super::residency_record::{self, AttemptFacts, RunIdentity};
use super::sealed_latency::SealedLatency;

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
pub(in super::super) struct ProcessResources {
    pub(in super::super) max_rss: u64,
    pub(in super::super) minor_faults: u64,
    pub(in super::super) major_faults: u64,
}

pub(in super::super) fn process_resources() -> ProcessResources {
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

/// One measured step: its value, its wall time, and the process's
/// counters read immediately before and immediately after it — the same
/// window the timer covers. The counters are PROCESS-wide: any thread
/// active in the window is counted too, so a count bounds this window's
/// faults rather than attributing them to the step's weights.
pub(in super::super) struct Bracketed<T> {
    pub(in super::super) value: T,
    pub(in super::super) elapsed_seconds: f64,
    pub(in super::super) before: ProcessResources,
    pub(in super::super) after: ProcessResources,
}

impl<T> Bracketed<T> {
    pub(in super::super) fn minor_faults(&self) -> u64 {
        self.after
            .minor_faults
            .saturating_sub(self.before.minor_faults)
    }

    pub(in super::super) fn major_faults(&self) -> u64 {
        self.after
            .major_faults
            .saturating_sub(self.before.major_faults)
    }
}

/// Run `step` with the fault counters and the timer around it. Whatever
/// the observer does — residency scans, deltas, printing — happens outside
/// this window, so its own work is never charged to the model's step,
/// this one or the next. Other threads of the process are not excluded:
/// see [`Bracketed`].
pub(in super::super) fn bracket<T>(step: impl FnOnce() -> T) -> Bracketed<T> {
    let before = process_resources();
    let started = Instant::now();
    let value = step();
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let after = process_resources();
    Bracketed {
        value,
        elapsed_seconds,
        before,
        after,
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
///
/// MEM-002 I2a: after the binding observation and before any pass, one
/// admission decision is made from that observation alone, written and
/// flushed to the residency stream; the passes cannot start without it on
/// record. Every evidence file is created exclusively before any weight
/// is bound, and while `--sealed-latency` is active no timing reaches the
/// console.
///
/// MEM-002 I2b: the store's retained descriptors are judged before binding,
/// at admission and after the run. A campaign attempt whose identity failed
/// stops on the record before the next timed region; found after the run,
/// the attempt's record is kept whole, marked invalid, and the run exits
/// non-zero. Diagnostic execution records the same facts and carries on.
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
    evidence: EvidenceRequest<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    use larql_vindex::format::vindex3::opplan::exec::accounting::{
        expectations, BlockGeometry, ResourceLedger,
    };
    use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};

    if prompt.is_empty() {
        return Err("prompt holds no tokens — nothing to condition on".into());
    }
    let access = larql_vindex::format::vindex3::opplan::exec::realization::MappedAccess::parse(
        expert_access,
    )?;
    if warmup >= repeat.max(1) {
        return Err(format!("warmup {warmup} leaves no counted pass out of {repeat}").into());
    }
    let evidence = plan_evidence(evidence)?;
    // The evidence files, created exclusively before the quiet gate and
    // before any weight is bound: an existing file is refused, never
    // truncated, and from here on every way out leaves a record.
    let run = RunIdentity::for_container(container);
    // How an attempt ended before its first identity judgement is classed:
    // a campaign's evidence cannot be used, and diagnostic execution's never
    // could. From that judgement on, an attempt carries the judgement's class.
    let unjudged = match evidence.mode {
        super::admission::RunMode::Campaign => EvidenceClass::Invalid,
        super::admission::RunMode::Diagnostic => EvidenceClass::Diagnostic,
    };
    let mut regions_file = evidence
        .regions
        .as_deref()
        .map(|path| create_evidence_file(path, "--residency-regions").map(BufWriter::new))
        .transpose()?;
    let mut sealed = evidence
        .sealed_latency
        .as_deref()
        .map(|path| SealedLatency::create(path, &run))
        .transpose()
        .map_err(|e| end_before_passes(&mut regions_file, &run, unjudged, e))?;
    let sealing = sealed.is_some();
    let manifest = evidence
        .manifest
        .as_deref()
        .map(VerificationManifest::read)
        .transpose()
        .map_err(|e| stop_attempt(&mut regions_file, &run, unjudged, e, &mut sealed))?;
    // Identity before binding, so an attempt on the wrong files never starts
    // the bind timer or touches the page cache for its weights.
    let index_digest = run.index_json_sha256.as_deref();
    let opening = store.instance_report();
    let judged = judge(evidence.mode, &opening, manifest.as_ref(), index_digest);
    println!("{}", instance_line("before binding", &judged));
    let wanted = evidence.execution.wanted_encoding.as_deref();
    let requested = evidence.execution.block();
    let selection_faults = selection_problems(wanted, store.selection(), &opening.absent_objects);
    println!("{}", execution_line("before binding", &selection_faults));
    let judged_early = class_with_execution(evidence.mode, judged.class, &selection_faults);
    if let Some(file) = regions_file.as_mut() {
        let mut record = before_binding_record(
            &run,
            instance_block(&opening, manifest.as_ref(), &judged),
            judged_early,
        );
        record["execution_identity"] = json!({
            "requested": requested,
            "objects": selection_block(store.selection()),
            "problems": selection_faults,
        });
        residency_record::write_record(file, &record)
            .and_then(|()| file.sync())
            .map_err(|e| {
                stop_attempt(
                    &mut regions_file,
                    &run,
                    judged_early,
                    format!("the identity record could not be written: {e}"),
                    &mut sealed,
                )
            })?;
    }
    if judged_early == EvidenceClass::Invalid {
        let reason = stop_reason(&judged, &selection_faults);
        return Err(
            stop_attempt(&mut regions_file, &run, judged_early, reason, &mut sealed).into(),
        );
    }
    let judged_class = judged_early;
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
    let disqualifiers: Vec<String> = environment
        .disqualifiers(cpu::environment::Phase::BeforeWork)
        .iter()
        .map(|d| d.to_string())
        .collect();
    println!("machine: {}", environment.describe());
    let qualification = if disqualifiers.is_empty() {
        "QUALIFIED".to_string()
    } else {
        if !unquiet_ok {
            let refusal = format!(
                "this machine is not quiet — {} — so the curve would calibrate against whatever \
                 else is running; pass --unquiet-ok to run anyway, labelled",
                disqualifiers.join("; ")
            );
            return Err(
                stop_attempt(&mut regions_file, &run, judged_class, refusal, &mut sealed).into(),
            );
        }
        format!("UNQUALIFIED ({})", disqualifiers.join("; "))
    };
    println!("qualification: {qualification}");
    println!("mode: {}", evidence.mode.name());

    let loading = Instant::now();
    let budget =
        larql_vindex::format::vindex3::opplan::exec::accounting::ResidencyBudget::UNBOUNDED
            .with_expert_access(access);
    let ops = PreparedOperands::load_within(plan, store, backend, ExecutionSlice::Full, &budget)
        .map_err(|e| {
            stop_attempt(
                &mut regions_file,
                &run,
                judged_class,
                e.to_string(),
                &mut sealed,
            )
        })?;
    let load_seconds = loading.elapsed().as_secs_f64();
    let bound = sealed.as_mut().map_or(Ok(()), |s| s.bind(load_seconds));
    bound.map_err(|e| {
        stop_attempt(
            &mut regions_file,
            &run,
            judged_class,
            e.to_string(),
            &mut sealed,
        )
    })?;
    println!("engine: {engine}");
    println!(
        "{}",
        bind_line(load_seconds, repeat, access.name(), sealing)
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
    let written = match regions_file.as_mut() {
        Some(file) => residency_record::write_observation(
            file,
            &run,
            BINDING_OBSERVATION,
            "binding",
            &binding,
        ),
        None => Ok(()),
    };
    written.map_err(|e| {
        stop_attempt(
            &mut regions_file,
            &run,
            judged_class,
            e.to_string(),
            &mut sealed,
        )
    })?;
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

    // ADMISSION, from the binding observation alone, on record before any
    // pass. It never stops the run: a refused attempt is executed too.
    let decision = decide(&mapped, evidence.threshold.as_ref());
    let at_admission = store.instance_report();
    let judged = judge(
        evidence.mode,
        &at_admission,
        manifest.as_ref(),
        index_digest,
    );
    let effective = effective_execution(backend, &ops, store, wanted);
    println!("{}", execution_line("admission", &effective.problems));
    let admission_class = class_with_execution(evidence.mode, judged.class, &effective.problems);
    let mut execution = effective.block.clone();
    execution["requested"] = requested;
    let facts = AttemptFacts {
        mode: evidence.mode,
        warmup,
        repeat: repeat.max(1),
        qualification: qualification.clone(),
        disqualifiers,
        unquiet_ok,
        expert_access: access.name().to_string(),
        representation_source: representation_source_name(store),
        sealed_latency: sealed.as_ref().map(|s| {
            std::path::absolute(s.path())
                .unwrap_or_else(|_| s.path().to_path_buf())
                .display()
                .to_string()
        }),
        instance: instance_block(&at_admission, manifest.as_ref(), &judged),
        execution,
        evidence_class: admission_class,
    };
    let record = residency_record::admission_record(&run, BINDING_OBSERVATION, &decision, &facts);
    let committed = commit_admission(
        evidence.mode,
        regions_file.as_mut().map(|f| f as &mut dyn EvidenceSink),
        &record,
    )
    .map_err(|e| stop_attempt(&mut regions_file, &run, admission_class, e, &mut sealed))?;
    println!("{}", instance_line("admission", &judged));
    // Invalid evidence stops here, recorded, before any timed pass. A
    // threshold rejection did not reach this: it is not invalid.
    if admission_class == EvidenceClass::Invalid {
        let reason = stop_reason(&judged, &effective.problems);
        return Err(stop_attempt(
            &mut regions_file,
            &run,
            admission_class,
            reason,
            &mut sealed,
        )
        .into());
    }
    let admitted_class = admission_class;
    let synced = sealed.as_mut().map_or(Ok(()), SealedLatency::sync);
    synced.map_err(|e| {
        stop_attempt(
            &mut regions_file,
            &run,
            admitted_class,
            e.to_string(),
            &mut sealed,
        )
    })?;
    // Printed only now, so the console cannot announce an unrecorded decision.
    println!("{}", admission_line(&decision));

    let passes = run_passes(
        committed,
        PassInputs {
            backend,
            plan,
            ops: &ops,
            prompt,
            new_tokens,
            repeat,
            warmup,
        },
        sealed.as_mut(),
    );

    // Close the attempt whatever the passes returned: the sealed file's
    // digest, the after-run observation, and the attempt's end. The first
    // failure is the attempt's error; a later one never replaces it.
    let sealed = sealed.map(SealedLatency::finish).transpose();
    let mut failures: Vec<String> = Vec::new();
    if let Err(e) = &passes {
        failures.push(e.to_string());
    }
    if let Err(e) = &sealed {
        failures.push(e.clone());
    }
    // Identity after the run, from the same descriptors. A change found now
    // cannot stop anything: the attempt's record is kept whole and marked.
    let finishing = store.instance_report();
    let judged = judge(evidence.mode, &finishing, manifest.as_ref(), index_digest);
    println!("{}", instance_line("after run", &judged));
    let ran = after_run_execution(&effective);
    println!("{}", execution_line("after run", &ran.problems));
    let finishing_class = class_with_execution(evidence.mode, judged.class, &ran.problems);
    let final_class = if finishing_class == EvidenceClass::Invalid {
        failures.push(stop_reason(&judged, &ran.problems));
        finishing_class
    } else {
        admitted_class
    };
    if let Some(file) = regions_file.as_mut() {
        let mut block = instance_block(&finishing, manifest.as_ref(), &judged);
        block["post_admission_reads"] = post_admission_reads(&at_admission, &finishing);
        let mut record = after_run_record(&run, block, finishing_class);
        record["execution_identity"] = ran.block;
        if let Err(e) = residency_record::write_record(file, &record) {
            failures.push(format!(
                "the after-run identity record was not written: {e}"
            ));
        }
        let after = ops.observe_residency();
        let fields = match &sealed {
            Ok(Some(digest)) => residency_record::sealed_header_fields(digest),
            _ => serde_json::Map::new(),
        };
        let observed = residency_record::write_observation_with(
            file,
            &run,
            AFTER_RUN_OBSERVATION,
            "after_run",
            &after,
            fields,
        )
        .and_then(|()| file.sync());
        if let Err(e) = observed {
            failures.push(format!("the after-run observation was not written: {e}"));
        }
        let error = (!failures.is_empty()).then(|| failures.join("; "));
        let ended = residency_record::write_record(
            file,
            &residency_record::attempt_end_record(
                &run,
                error.as_deref(),
                final_class,
                sealed.as_ref().ok().and_then(Option::as_ref),
            ),
        )
        .and_then(|()| file.sync());
        if let Err(e) = ended {
            failures.push(format!(
                "the attempt's end may not have reached the disk, whatever its status says: {e}"
            ));
        }
    }
    // With no stream, or a failed one, the digest goes to the console: a
    // digest is not a latency, and nothing else could later show the file
    // untouched.
    if let Ok(Some(digest)) = &sealed {
        if regions_file.is_none() || !failures.is_empty() {
            println!(
                "sealed latency: {} sha256 {}",
                digest.path.display(),
                digest.sha256
            );
        }
    }
    if !failures.is_empty() {
        return Err(failures.join("; ").into());
    }
    let samples = passes?;
    let sealed = sealed?;
    for line in summary_lines(
        &samples,
        repeat.max(1),
        warmup,
        &qualification,
        sealed.as_ref().map(|d| d.path.as_path()),
    ) {
        println!("{line}");
    }
    Ok(())
}

/// Close an attempt that ended before it could be admitted, returning
/// its reason. The record is best effort: the attempt has already failed,
/// and a stream that cannot take this record must not hide the reason. It
/// says so on stderr, since the stream then has no closing record.
pub(in super::super) fn end_before_passes<S: EvidenceSink>(
    stream: &mut Option<S>,
    run: &RunIdentity,
    class: EvidenceClass,
    reason: String,
) -> String {
    end_with_digest(stream, run, class, reason, None)
}

fn end_with_digest<S: EvidenceSink>(
    stream: &mut Option<S>,
    run: &RunIdentity,
    class: EvidenceClass,
    reason: String,
    sealed: Option<&super::sealed_latency::SealedDigest>,
) -> String {
    if let Some(out) = stream.as_mut() {
        let record = residency_record::attempt_end_record(run, Some(&reason), class, sealed);
        if let Err(e) = residency_record::write_record(out, &record).and_then(|()| out.sync()) {
            eprintln!("the attempt's closing record was not written: {e}");
        }
    }
    reason
}

/// [`end_before_passes`] for an attempt with a sealed file: finish it and put
/// its digest on the closing record, and on the console, which is not a
/// latency, for a run whose stream cannot take it. Nothing else could show
/// the file untouched: this attempt has no `after_run` header.
fn stop_attempt<S: EvidenceSink>(
    stream: &mut Option<S>,
    run: &RunIdentity,
    class: EvidenceClass,
    reason: String,
    sealed: &mut Option<SealedLatency>,
) -> String {
    let digest = match sealed.take().map(SealedLatency::finish).transpose() {
        Ok(digest) => digest,
        Err(e) => {
            // On stderr, and in the reason below: the record says why the
            // attempt has no digest.
            eprintln!("the sealed latency file could not be finished: {e}");
            let reason = format!("{reason}; the sealed latency file was not finished: {e}");
            return end_with_digest(stream, run, class, reason, None);
        }
    };
    if let Some(digest) = &digest {
        println!(
            "sealed latency: {} sha256 {}",
            digest.path.display(),
            digest.sha256
        );
    }
    end_with_digest(stream, run, class, reason, digest.as_ref())
}

/// Why an attempt stops: its identity judgement if that failed, and whatever
/// its execution's judgement found.
fn stop_reason(judged: &InstanceJudgement, execution: &[String]) -> String {
    let identity = judged.invalid().then(|| judged.reason());
    identity
        .into_iter()
        .chain(execution.iter().cloned())
        .collect::<Vec<_>>()
        .join("; ")
}

/// The store's `--representation-source`, as the flag spells it.
fn representation_source_name(store: &OperandStore) -> String {
    use larql_vindex::format::vindex3::opplan::exec::operands::RepresentationSource;
    match store.representation_source() {
        RepresentationSource::Auto => "auto",
        RepresentationSource::Stored => "stored",
        RepresentationSource::Transient => "transient",
    }
    .to_string()
}

/// The binding observation's id in the residency stream.
const BINDING_OBSERVATION: u64 = 0;
/// The after-run observation's id.
const AFTER_RUN_OBSERVATION: u64 = 1;

/// The console line for the binding: its time, unless sealed.
pub(in super::super) fn bind_line(
    seconds: f64,
    repeat: usize,
    access: &str,
    sealing: bool,
) -> String {
    let time = if sealing {
        "(bind time sealed)".to_string()
    } else {
        format!("in {seconds:.1} s")
    };
    format!("weights bound {time} (once, for {repeat} pass(es)); expert access: {access}")
}

/// A step's time in the curve's table — or `sealed`, never a number,
/// while sealing is active.
pub(in super::super) fn timing_cell(elapsed_ms: f64, sealing: bool) -> String {
    if sealing {
        "sealed".to_string()
    } else {
        format!("{elapsed_ms:.0}")
    }
}

/// The table's per-stage and per-leaf times, or nothing while sealed.
pub(in super::super) fn timing_tail(
    by_stage: &[String],
    leaves: &[String],
    sealing: bool,
) -> String {
    if sealing {
        String::new()
    } else {
        format!("  {}  [leaves {}]", by_stage.join(" "), leaves.join(" "))
    }
}

/// The leaf timing classes the curve's table names.
const LEAF_CLASSES: [timing::OpClass; 3] = [
    timing::OpClass::Projection,
    timing::OpClass::Norm,
    timing::OpClass::Logits,
];

/// What every pass needs, apart from the proof of admission.
struct PassInputs<'a, B: PlanBackend> {
    backend: &'a B,
    plan: &'a ComponentOpPlan,
    ops: &'a larql_vindex::format::vindex3::opplan::exec::prepared::PreparedOperands,
    prompt: &'a [u32],
    new_tokens: usize,
    repeat: usize,
    warmup: usize,
}

/// The passes. The `AdmissionCommitted` argument is the only way in: no
/// timed region can run before the decision is on record.
fn run_passes<B: PlanBackend>(
    committed: AdmissionCommitted,
    inputs: PassInputs<'_, B>,
    mut sealed: Option<&mut SealedLatency>,
) -> Result<Vec<PassSample>, Box<dyn std::error::Error>> {
    use larql_vindex::format::vindex3::opplan::exec::routing_trace;

    let PassInputs {
        backend,
        plan,
        ops,
        prompt,
        new_tokens,
        repeat,
        warmup,
    } = inputs;
    if !committed.recorded() {
        println!("admission: decided, not recorded (diagnostic run with no --residency-regions)");
    }
    let sealing = sealed.is_some();
    let counted_from = warmup + 1;
    let mut samples: Vec<PassSample> = Vec::new();
    let mut first_logits: Option<Vec<f32>> = None;
    let mut first_generated: Option<Vec<u32>> = None;
    for pass in 1..=repeat.max(1) {
        let mut kv = crate::commands::primary::continuation::select_for(plan, None)?.build();
        let mut session = DecodeSession::over_prepared(plan, ops, backend, &mut *kv)?;
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
            "  {:<8} {:>9} {:>12} {:>12} {:>9} {:>10} {:>10} {:>9}{}",
            "step",
            "ms",
            "read res",
            "Δ res (GB)",
            "excluded",
            "minor Δ",
            "major Δ",
            "rss GB",
            if sealing { "" } else { "  stages (ms)" }
        );
        let mut before_res = session.observe_residency();
        // The fault counts come from the step's own bracket: this
        // observer never reads the counters, so nothing it allocates is
        // charged to a step.
        let observe = |name: &str,
                       elapsed: f64,
                       faults: (u64, u64),
                       rss_after: u64,
                       session: &DecodeSession<'_, B>,
                       before_res: &mut ResidencyObservation|
         -> StepObservation {
            let now_res = session.observe_residency();
            let stage_ledger = stages::ledger();
            let by_stage: Vec<String> = stage_ledger
                .all()
                .iter()
                .filter(|(_, t)| t.calls > 0)
                .map(|(st, t)| format!("{}={:.1}", st.name(), t.nanos as f64 / 1e6))
                .collect();
            let leaves = timing::ledger();
            let leaf_note: Vec<String> = LEAF_CLASSES
                .iter()
                .map(|c| (c, leaves.get(*c)))
                .filter(|(_, t)| t.calls > 0)
                .map(|(c, t)| format!("{}={:.0}", c.name(), t.nanos as f64 / 1e6))
                .collect();
            let obs = StepObservation {
                elapsed_ms: elapsed * 1e3,
                resident_delta: ResidencyDelta::between(before_res, &now_res),
                minor_faults: faults.0,
                major_faults: faults.1,
                stage_ms: stages::Stage::ALL.map(|st| stage_ledger.get(st).nanos as f64 / 1e6),
                stages_nested: stage_ledger.nested(),
                leaf_ms: LEAF_CLASSES.map(|c| leaves.get(c).nanos as f64 / 1e6),
            };
            println!(
                "  {:<8} {:>9} {:>9.3} GB {:>12} {:>9} {:>10} {:>10} {:>9.2}{}",
                name,
                timing_cell(obs.elapsed_ms, sealing),
                now_res.summary().resident_bytes as f64 / GB,
                residency_record::delta_cell(&obs.resident_delta),
                obs.resident_delta.excluded,
                obs.minor_faults,
                obs.major_faults,
                rss_after as f64 / GB,
                timing_tail(&by_stage, &leaf_note, sealing)
            );
            *before_res = now_res;
            obs
        };
        // Prompt: every position through the stack, timed as one step.
        timing::ledger().reset();
        stages::ledger().reset();
        let prompted = bracket(|| -> Result<_, larql_vindex::error::VindexError> {
            let mut logits = None;
            for &token in prompt {
                logits = session.step(token)?.logits;
            }
            Ok(logits)
        });
        let faults = (prompted.minor_faults(), prompted.major_faults());
        let (elapsed, rss_after) = (prompted.elapsed_seconds, prompted.after.max_rss);
        let name = format!("prompt×{}", prompt.len());
        seal_if_failed(&mut sealed, pass, label, &name, elapsed, &prompted.value)?;
        let logits = prompted.value?;
        let obs = observe(&name, elapsed, faults, rss_after, &session, &mut before_res);
        seal_step(&mut sealed, pass, label, &name, &obs)?;
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
            let stepped = bracket(|| session.step(id));
            let routing = routing_trace::take_capture();
            let faults = (stepped.minor_faults(), stepped.major_faults());
            let (elapsed, rss_after) = (stepped.elapsed_seconds, stepped.after.max_rss);
            let name = format!("token {}", step + 1);
            seal_if_failed(&mut sealed, pass, label, &name, elapsed, &stepped.value)?;
            logits = stepped
                .value?
                .logits
                .ok_or("plan carries no output head — cannot generate")?;
            let obs = observe(&name, elapsed, faults, rss_after, &session, &mut before_res);
            seal_step(&mut sealed, pass, label, &name, &obs)?;
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
    Ok(samples)
}

/// One step's latency into the sealed file, when sealing: its time and
/// its breakdowns, outside the step's timed window.
fn seal_step(
    sealed: &mut Option<&mut SealedLatency>,
    pass: usize,
    label: &str,
    step: &str,
    obs: &StepObservation,
) -> std::io::Result<()> {
    let Some(sealed) = sealed.as_deref_mut() else {
        return Ok(());
    };
    let stages_ms: serde_json::Map<String, serde_json::Value> = stages::Stage::ALL
        .iter()
        .zip(obs.stage_ms)
        .map(|(st, ms)| (st.name().to_string(), serde_json::json!(ms)))
        .collect();
    let leaves_ms: serde_json::Map<String, serde_json::Value> = LEAF_CLASSES
        .iter()
        .zip(obs.leaf_ms)
        .map(|(c, ms)| (c.name().to_string(), serde_json::json!(ms)))
        .collect();
    let breakdown = serde_json::json!({
        "stages_ms": stages_ms,
        "stages_nested": obs.stages_nested,
        "leaves_ms": leaves_ms,
    });
    sealed.step(pass, label, step, obs.elapsed_ms, breakdown)
}

/// A step that returned an error: its time into the sealed file. If that
/// fails too, both errors are reported, the step's first.
fn seal_if_failed<T, E: std::fmt::Display>(
    sealed: &mut Option<&mut SealedLatency>,
    pass: usize,
    label: &str,
    step: &str,
    elapsed_seconds: f64,
    outcome: &Result<T, E>,
) -> Result<(), String> {
    let (Err(error), Some(sealed)) = (outcome, sealed.as_deref_mut()) else {
        return Ok(());
    };
    sealed
        .failed_step(pass, label, step, elapsed_seconds * 1e3)
        .map_err(|e| format!("{error}; and its time was not sealed: {e}"))
}

/// One step's observation, as the curve reads it between tokens.
#[derive(Debug, Clone, Copy)]
pub(in super::super) struct StepObservation {
    pub(in super::super) elapsed_ms: f64,
    /// Per-region change over the regions read in both observations.
    pub(in super::super) resident_delta: ResidencyDelta,
    pub(in super::super) minor_faults: u64,
    pub(in super::super) major_faults: u64,
    /// Milliseconds per [`stages::Stage::ALL`] entry, in that order.
    pub(in super::super) stage_ms: [f64; stages::Stage::ALL.len()],
    pub(in super::super) stages_nested: u64,
    /// Milliseconds per [`LEAF_CLASSES`] entry, in that order.
    pub(in super::super) leaf_ms: [f64; 3],
}

/// The characterised token of one counted pass.
#[derive(Debug, Clone)]
pub(in super::super) struct PassSample {
    pub(in super::super) pass: usize,
    pub(in super::super) observation: StepObservation,
    pub(in super::super) routing_fingerprint: u64,
    pub(in super::super) routed_layers: usize,
    pub(in super::super) first_layer_experts: Vec<usize>,
    pub(in super::super) logits_delta: f32,
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
///
/// While sealing is active (`sealed` names the file) the lines carry no
/// timing: routing, logits and faults only.
pub(in super::super) fn summary_lines(
    samples: &[PassSample],
    repeat: usize,
    warmup: usize,
    qualification: &str,
    sealed: Option<&Path>,
) -> Vec<String> {
    use larql_vindex::format::vindex3::opplan::exec::stages::Stage;
    let mut lines = vec![format!(
        "\ncharacterised token over {} counted pass(es) ({} warmup of {}), machine {}:",
        samples.len(),
        warmup,
        repeat,
        qualification
    )];
    if samples.is_empty() {
        lines.push("  nothing counted".to_string());
        return lines;
    }
    let fingerprints: std::collections::BTreeSet<u64> =
        samples.iter().map(|s| s.routing_fingerprint).collect();
    if fingerprints.len() == 1 {
        lines.push(format!(
            "  routing: IDENTICAL in every counted pass ({:016x}, {} routed layers)",
            samples[0].routing_fingerprint, samples[0].routed_layers
        ));
    } else {
        lines.push(format!(
            "  routing: DIFFERS across passes — {} distinct selections; the statistics below are over different work",
            fingerprints.len()
        ));
        for s in samples {
            lines.push(format!(
                "    pass {} {:016x}",
                s.pass, s.routing_fingerprint
            ));
        }
    }
    let nested: u64 = samples.iter().map(|s| s.observation.stages_nested).sum();
    if nested > 0 {
        lines.push(format!(
            "  REFUSING TO RECONCILE STAGES: {nested} nested stage timers across the counted passes"
        ));
    }
    let worst_logits = samples
        .iter()
        .map(|s| s.logits_delta)
        .fold(0.0f32, f32::max);
    lines.push(format!(
        "  logits max|Δ| vs pass 1 across counted passes: {worst_logits:.2e}"
    ));
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
    lines.push(format!(
        "  faults: major {major} total, minor ≤ {minor_max} per pass; resident-page delta {resident} GB total over {} pass(es){}",
        measured.len(),
        if unmeasured > 0 {
            format!(", {unmeasured} with no comparable region")
        } else {
            String::new()
        }
    ));
    if let Some(path) = sealed {
        lines.push(format!(
            "  timings: SEALED in {} — no latency is printed while sealing is active",
            path.display()
        ));
        return lines;
    }
    lines.push(format!(
        "  {:<16} {:>9} {:>9} {:>9} {:>9} {:>8}",
        "ms", "p50", "p95", "min", "max", "p95/p50"
    ));
    let (p50, p95, min, max) = stats(samples.iter().map(|s| s.observation.elapsed_ms));
    lines.push(format!(
        "  {:<16} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>8.2}",
        "token",
        p50,
        p95,
        min,
        max,
        p95 / p50
    ));
    let token_p50 = p50;
    let mut staged = 0.0;
    for (i, st) in Stage::ALL.iter().enumerate() {
        let (p50, p95, min, max) = stats(samples.iter().map(|s| s.observation.stage_ms[i]));
        staged += p50;
        lines.push(format!(
            "  {:<16} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>8.2}   {:>5.1}% of token p50",
            st.name(),
            p50,
            p95,
            min,
            max,
            p95 / p50,
            p50 / token_p50 * 100.0
        ));
    }
    lines.push(format!(
        "  {:<16} {:>9.1}   ({:.1}% of token p50: norms, residual, embedding, head, glue)",
        "other",
        token_p50 - staged,
        (token_p50 - staged) / token_p50 * 100.0
    ));
    lines
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
