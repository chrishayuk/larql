//! `vindex3 measure` — MEASURE-PLAN-1's procedure
//! (`teacher-forced-two-arm/plan-v1`, `docs/measure-plan-1.md`) from the
//! command line.
//!
//! Two arms, each a container, a `vindex3 exec --backend` name and a
//! representation source, teacher-forced over a sealed token bank
//! (`vindex3 token-bank export`). The procedure, its proofs and every
//! refusal live in `larql_vindex::format::vindex3::represent::measure::plan`.
//! This file opens the containers, builds the arms, and prints what the
//! procedure returned. It computes no number of its own.
//!
//! An interpreter backend becomes an interpreter arm over the registered
//! provider `exec` would use. A lowered Metal backend becomes a lowered arm
//! over a `LoweredSession`, sized to the bank's longest sample.

use std::path::PathBuf;

use clap::Args;
use larql_inference::vindex3::OpenedComponent;
use larql_vindex::format::vindex3::opplan::exec::operands::RepresentationSource;
use larql_vindex::format::vindex3::represent::measure::plan::arm::{
    InterpreterArm, TeacherForcedArm,
};
use larql_vindex::format::vindex3::represent::measure::plan::metrics::Aggregate;
use larql_vindex::format::vindex3::represent::measure::plan::{
    run as run_procedure, PlanMeasureRequest, PlanReceipt, PlanRefusal,
};

use super::plugins::{PluginArgs, Plugins};
use super::prepare::{lowerings_with, parse_representation_source, prepare, DEFAULT_COMPONENT};
use super::ExecBackend;

type BoxErr = Box<dyn std::error::Error>;

/// The reference reads canonical bytes unless told otherwise.
const DEFAULT_REFERENCE_SOURCE: &str = "auto";

/// The candidate may not manufacture its representation: a pack arm must
/// read the pack it names, or the procedure refuses it.
const DEFAULT_CANDIDATE_SOURCE: &str = "stored";

#[derive(Args)]
pub struct MeasureArgs {
    /// Reference container.
    #[arg(long)]
    pub reference: PathBuf,
    /// Reference execution arm, as `vindex3 exec --backend` spells it.
    #[arg(long, value_enum)]
    pub reference_backend: ExecBackend,
    /// Reference representation source: auto, stored or transient.
    #[arg(long, default_value = DEFAULT_REFERENCE_SOURCE)]
    pub reference_source: String,
    /// Candidate container.
    #[arg(long)]
    pub candidate: PathBuf,
    /// Candidate execution arm.
    #[arg(long, value_enum)]
    pub candidate_backend: ExecBackend,
    /// Candidate representation source.
    #[arg(long, default_value = DEFAULT_CANDIDATE_SOURCE)]
    pub candidate_source: String,
    /// Token bank directory.
    #[arg(long)]
    pub bank: PathBuf,
    /// Samples to measure, from `seq-000`.
    #[arg(long)]
    pub sequences: usize,
    /// A name for the run.
    #[arg(long)]
    pub label: String,
    /// Directory for report.json, positions.jsonl and receipt.json.
    #[arg(long)]
    pub output: PathBuf,
    /// Component to measure.
    #[arg(long, default_value = DEFAULT_COMPONENT)]
    pub component: String,
    /// Extra provenance recorded verbatim in the report, as `key=value`.
    /// Repeat for several. `larql_version` is always recorded.
    #[arg(long = "provenance", value_parser = parse_provenance)]
    pub provenance: Vec<(String, String)>,
    /// Load a larql plugin for both arms, as `vindex3 exec --plugin` does.
    /// Repeatable.
    #[arg(long = "plugin", value_name = "PATH")]
    pub plugins: Vec<PathBuf>,
    /// Execute the reference on the lowering provider with this identity
    /// (`family/vN`) instead of the one `--reference-backend` names.
    #[arg(long, value_name = "FAMILY/vN")]
    pub reference_lowering: Option<String>,
    /// Execute the candidate on the lowering provider with this identity
    /// (`family/vN`) instead of the one `--candidate-backend` names.
    #[arg(long, value_name = "FAMILY/vN")]
    pub candidate_lowering: Option<String>,
    /// Bind the reference to the stored representation with this encoding
    /// instead of the one `--reference-backend` names.
    #[arg(long, value_name = "ENCODING")]
    pub reference_representation: Option<String>,
    /// Bind the candidate to the stored representation with this encoding
    /// (e.g. a pack a `--plugin`'s encoder compiled) instead of the one
    /// `--candidate-backend` names.
    #[arg(long, value_name = "ENCODING")]
    pub candidate_representation: Option<String>,
}

/// The report key for this binary's version.
const VERSION_KEY: &str = "larql_version";

/// Parse one `key=value` provenance entry.
fn parse_provenance(entry: &str) -> Result<(String, String), String> {
    match entry.split_once('=') {
        Some((key, value)) if !key.is_empty() => Ok((key.to_string(), value.to_string())),
        _ => Err(format!("`{entry}` is not key=value")),
    }
}

/// One arm's inputs, resolved.
struct ArmSpec<'a> {
    container: &'a PathBuf,
    backend: ExecBackend,
    source: RepresentationSource,
    plugins: Plugins,
}

pub fn run(args: MeasureArgs) -> Result<(), BoxErr> {
    // Every `--plugin` is loaded once; each arm asks it for its own provider.
    let loaded = Plugins::load(&PluginArgs {
        plugins: args.plugins.clone(),
        lowering: None,
        representation: None,
    })?;
    let reference = ArmSpec {
        container: &args.reference,
        backend: args.reference_backend,
        source: parse_representation_source(&args.reference_source)?,
        plugins: loaded.selecting(
            args.reference_lowering.as_deref(),
            args.reference_representation.as_deref(),
        )?,
    };
    let candidate = ArmSpec {
        container: &args.candidate,
        backend: args.candidate_backend,
        source: parse_representation_source(&args.candidate_source)?,
        plugins: loaded.selecting(
            args.candidate_lowering.as_deref(),
            args.candidate_representation.as_deref(),
        )?,
    };
    let reference_opened = prepare(
        reference.container,
        &args.component,
        reference.backend,
        reference.source,
        &reference.plugins,
    )?;
    let candidate_opened = prepare(
        candidate.container,
        &args.component,
        candidate.backend,
        candidate.source,
        &candidate.plugins,
    )?;
    let request = PlanMeasureRequest {
        bank: args.bank.clone(),
        sequences: args.sequences,
        label: args.label.clone(),
        output: args.output.clone(),
        provenance: std::iter::once((
            VERSION_KEY.to_string(),
            env!("CARGO_PKG_VERSION").to_string(),
        ))
        .chain(args.provenance.iter().cloned())
        .collect(),
    };
    let outcome = with_arms(
        (&reference, &reference_opened),
        (&candidate, &candidate_opened),
        &request,
    )?;
    report(&request, outcome)
}

/// Build both arms and run the procedure while they are alive. A lowered
/// arm borrows its device and loaded weights, so they are owned here.
#[cfg(all(feature = "gpu", target_os = "macos"))]
fn with_arms(
    reference: (&ArmSpec<'_>, &OpenedComponent),
    candidate: (&ArmSpec<'_>, &OpenedComponent),
    request: &PlanMeasureRequest,
) -> Result<Result<PlanReceipt, PlanRefusal>, BoxErr> {
    {
        use larql_vindex::format::vindex3::represent::token_bank::TokenBank;
        let max_positions = TokenBank::open(&request.bank)?
            .manifest()
            .samples
            .iter()
            .map(|s| s.tokens)
            .max()
            .unwrap_or(1);
        let gpus = [
            lowered::device_for(reference.0.backend)?,
            lowered::device_for(candidate.0.backend)?,
        ];
        let mut keep = [Vec::new(), Vec::new()];
        let [keep_r, keep_c] = &mut keep;
        let mut r = build_arm(reference, gpus[0].as_ref(), max_positions, keep_r)?;
        let mut c = build_arm(candidate, gpus[1].as_ref(), max_positions, keep_c)?;
        Ok(run_procedure(request, r.as_mut(), c.as_mut()))
    }
}

/// Without a Metal build every arm is an interpreter arm.
#[cfg(not(all(feature = "gpu", target_os = "macos")))]
fn with_arms(
    reference: (&ArmSpec<'_>, &OpenedComponent),
    candidate: (&ArmSpec<'_>, &OpenedComponent),
    request: &PlanMeasureRequest,
) -> Result<Result<PlanReceipt, PlanRefusal>, BoxErr> {
    let mut r = interpreter_arm(reference)?;
    let mut c = interpreter_arm(candidate)?;
    Ok(run_procedure(request, r.as_mut(), c.as_mut()))
}

/// An interpreter arm over the provider `exec` would use for `backend`.
fn interpreter_arm(
    (spec, opened): (&ArmSpec<'_>, &OpenedComponent),
) -> Result<Box<dyn TeacherForcedArm>, BoxErr> {
    let (lowerings, identity) = lowerings_with(spec.backend, &spec.plugins)?;
    let provider = lowerings.provider_shared(&identity)?;
    Ok(Box::new(InterpreterArm::prepare(
        &arm_name(spec),
        spec.container.clone(),
        opened.want.clone(),
        spec.source == RepresentationSource::Stored,
        opened.plan.clone(),
        &opened.store,
        provider,
    )?))
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
fn build_arm<'a>(
    arm: (&ArmSpec<'_>, &'a OpenedComponent),
    gpu: Option<&'a larql_compute_metal::MetalBackend>,
    max_positions: usize,
    keep: &mut Vec<larql_vindex::format::vindex3::opplan::exec::weights::LoadedWeight>,
) -> Result<Box<dyn TeacherForcedArm + 'a>, BoxErr> {
    let (spec, opened) = arm;
    let Some(formats) = super::prepare::lowered_formats(spec.backend).map(|(f, _)| f) else {
        return interpreter_arm(arm);
    };
    if spec.plugins.select.is_some() || spec.plugins.want.is_some() {
        return Err(format!(
            "lowering and representation overrides do not apply to `{:?}`: a lowered arm \
             executes its own formats, not through a lowering provider",
            spec.backend
        )
        .into());
    }
    let gpu = gpu.ok_or("a lowered arm needs a Metal device")?;
    let session = super::lowered::LoweredSession::new(
        gpu,
        &opened.plan,
        &opened.store,
        formats,
        max_positions,
        keep,
    )?;
    Ok(Box::new(super::lowered::measure_arm::LoweredArm::new(
        session,
        &opened.store,
        &arm_name(spec),
        spec.container.clone(),
        opened.want.clone(),
        spec.source == RepresentationSource::Stored,
    )))
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
mod lowered {
    use super::{BoxErr, ExecBackend};
    use larql_compute_metal::MetalBackend;

    /// A Metal device for a lowered backend, `None` for any other.
    pub(super) fn device_for(backend: ExecBackend) -> Result<Option<MetalBackend>, BoxErr> {
        if super::super::prepare::lowered_formats(backend).is_none() {
            return Ok(None);
        }
        Ok(Some(
            MetalBackend::new().ok_or("no Metal device available for a lowered arm")?,
        ))
    }
}

/// The arm name the procedure records: the backend's CLI spelling, and the
/// provider identity when a lowering override replaced the backend's own.
fn arm_name(spec: &ArmSpec<'_>) -> String {
    use clap::ValueEnum;
    let backend = spec
        .backend
        .to_possible_value()
        .map(|v| v.get_name().to_string())
        .unwrap_or_else(|| format!("{:?}", spec.backend));
    match &spec.plugins.select {
        Some(identity) => format!("{backend}@{identity}"),
        None => backend,
    }
}

/// Print what the procedure returned: the summary if admissible, the typed
/// refusal otherwise. A refusal is an error exit, because it is not evidence.
fn report(
    request: &PlanMeasureRequest,
    outcome: Result<PlanReceipt, PlanRefusal>,
) -> Result<(), BoxErr> {
    let receipt = match outcome {
        Ok(receipt) => receipt,
        Err(refusal) => {
            return Err(format!(
                "{} refused: {refusal:?} (receipt in {})",
                request.label,
                request.output.display()
            )
            .into())
        }
    };
    println!(
        "{}: admissible — {} positions over {} samples, bank {}",
        receipt.label, receipt.facts.positions, receipt.facts.bank_samples_read, receipt.bank_id
    );
    println!(
        "reference {}  candidate {}  changed: {}",
        receipt.reference.arm,
        receipt.candidate.arm,
        if receipt.facts.changed_representations.is_empty() {
            "the arm only".to_string()
        } else {
            receipt.facts.changed_representations.join(", ")
        }
    );
    print_row("all", &receipt.summary.all);
    for (category, aggregate) in &receipt.summary.by_category {
        print_row(category, aggregate);
    }
    for ((lo, hi), aggregate) in &receipt.summary.by_margin_band {
        print_row(&format!("margin {lo:.1}-{hi:.1}"), aggregate);
    }
    println!("-> {}", request.output.display());
    Ok(())
}

fn print_row(name: &str, a: &Aggregate) {
    println!(
        "  {name:<18} n={:<6} KL mean {:.3e}  p50 {:.3e}  p99 {:.3e}  max {:.3e}  top-1 {:.2}%",
        a.positions,
        a.kl_mean,
        a.kl_p50,
        a.kl_p99,
        a.kl_max,
        a.top1_agreement * 100.0
    );
}
