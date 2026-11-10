//! `vindex3 auto-rep` — AUTO-REP over a plan-v1 record (MEASURE-PLAN-2,
//! amendment A2).
//!
//! `init` produces a plan-v1 record from a container and a token bank,
//! optionally over a declared group vocabulary. It is characterisation-only
//! unless `--gate` arms it with a gate a slice-3 freeze pre-registered;
//! choosing that gate is the freeze's decision, not this verb's. `run`
//! advances a record through AUTO-REP-1b's loop with the same arms
//! `vindex3 measure` builds (lowered on Metal, interpreter otherwise), and
//! refuses a record without a gate. The record format, the producer and the loop live in
//! `larql_vindex::format::vindex3::represent`; this file names arguments.

use std::path::{Path, PathBuf};

use clap::{Args, Subcommand};
use larql_vindex::format::vindex3::represent::actuate::executor::{
    ExecutorRegistry, ExperimentExecutor,
};
use larql_vindex::format::vindex3::represent::auto_rep::{
    self, CampaignRecord, CampaignSetup, CandidateCompiler,
};
use larql_vindex::format::vindex3::represent::codec::EncoderRegistry;
use larql_vindex::format::vindex3::represent::measurement::TailSupportPolicy;
use larql_vindex::format::vindex3::represent::produce::{produce, Arming, ProduceInputs};
use larql_vindex::format::vindex3::represent::reading::gate_of_any_kind;
use larql_vindex::format::vindex3::represent::state::snapshot::SearchSnapshot;
use larql_vindex::format::vindex3::represent::state::ActionVocabulary;
use larql_vindex::format::vindex3::represent::{compile_representation_with, RepresentSpec};
use larql_vindex::VindexError;

use super::measure::{SketchArgs, VerbPlanExecutor};
use super::plugins::{PluginArgs, Plugins};
use super::ExecBackend;

#[derive(Args)]
pub struct AutoRepArgs {
    #[command(subcommand)]
    pub command: AutoRepCommand,
}

#[derive(Subcommand)]
pub enum AutoRepCommand {
    /// Produce a characterisation-only plan-v1 record from a container and
    /// a token bank.
    Init(InitArgs),
    /// Advance a record through the propose, compile, measure, cut loop.
    Run(RunArgs),
}

#[derive(Args)]
pub struct InitArgs {
    /// Source container every candidate is compiled from: a directory, `hf://owner/name`, `owner/name` or a cache shorthand such as `Qwen3-0.6B`; never downloaded.
    pub container: PathBuf,
    /// Token bank exported with this container's tokenizer.
    #[arg(long)]
    pub bank: PathBuf,
    /// Samples each measurement reads, from `seq-000`.
    #[arg(long)]
    pub sequences: usize,
    /// Encoding a compiled group takes.
    #[arg(long, default_value = "NVFP4")]
    pub encoding: String,
    /// The record to write. Refused if it exists.
    #[arg(long)]
    pub output: PathBuf,
    /// A declared group vocabulary (JSON: `{"edits": [{"name", "exceptions":
    /// [{"projection", "layers": [lo, hi]}]}]}`). Without it the search has
    /// one group per (projection, layer).
    #[arg(long, value_name = "JSON")]
    pub vocabulary: Option<PathBuf>,
    /// Arm the record with this pre-registered gate id. Without it the
    /// record is characterisation-only.
    #[arg(long, value_name = "ID", requires_all = ["min_tail_observations", "tail_provenance"])]
    pub gate: Option<String>,
    /// Expected tail observations a percentile criterion needs.
    #[arg(long, value_name = "N", requires = "gate")]
    pub min_tail_observations: Option<f64>,
    /// Where the tail policy was pre-registered.
    #[arg(long, value_name = "TEXT", requires = "gate")]
    pub tail_provenance: Option<String>,
}

impl InitArgs {
    /// The gate and tail policy to arm with, or `None` for a
    /// characterisation-only record.
    fn arming(&self) -> Result<Option<Arming>, Box<dyn std::error::Error>> {
        match (
            &self.gate,
            self.min_tail_observations,
            &self.tail_provenance,
        ) {
            (None, None, None) => Ok(None),
            (Some(id), Some(min_tail_observations), Some(provenance)) => Ok(Some(Arming {
                gate: gate_of_any_kind(id)?,
                tail_support: TailSupportPolicy {
                    min_tail_observations,
                    provenance: provenance.clone(),
                },
            })),
            _ => Err("--gate, --min-tail-observations and --tail-provenance go together".into()),
        }
    }
}

#[derive(Args)]
pub struct RunArgs {
    /// The record to advance. Not read with `--resume`.
    #[arg(long, required_unless_present = "resume")]
    pub snapshot: Option<PathBuf>,
    /// Continue the campaign checkpointed in `--output` and `--campaign`.
    /// Both are rewritten after every proposal, so a stopped campaign loses
    /// at most the measurement it was running.
    #[arg(long, conflicts_with = "snapshot")]
    pub resume: bool,
    /// Where the advanced record is written. Refused if it exists: the
    /// input record is never overwritten.
    #[arg(long)]
    pub output: PathBuf,
    /// Where the campaign record is written. Refused if it exists.
    #[arg(long)]
    pub campaign: PathBuf,
    /// The source container the record describes: a directory, `hf://owner/name`, `owner/name` or a cache shorthand such as `Qwen3-0.6B`; never downloaded.
    #[arg(long)]
    pub source: PathBuf,
    /// The token bank the record's protocol was sealed from.
    #[arg(long)]
    pub bank: PathBuf,
    /// Candidates are compiled here, one directory per state.
    #[arg(long)]
    pub workdir: PathBuf,
    /// Each measurement's report is written here.
    #[arg(long)]
    pub runs: PathBuf,
    /// Measurements this campaign may spend.
    #[arg(long, default_value_t = 3)]
    pub budget: usize,
    /// Solver node budget per proposal.
    #[arg(long, default_value_t = 1_000_000)]
    pub node_limit: u64,
    /// Component both arms execute.
    #[arg(long, default_value = "target")]
    pub component: String,
    /// The reference arm's `vindex3 exec` backend, over the source's
    /// canonical bytes. A lowered Metal backend is a lowered arm.
    #[arg(long, value_enum, default_value = "production")]
    pub reference_backend: ExecBackend,
    /// The candidate arm's backend. It must read the stored pack in the
    /// record's encoding, or the run is refused before it starts.
    #[arg(long, value_enum, default_value = "production-nvfp4")]
    pub candidate_backend: ExecBackend,
    #[command(flatten)]
    pub plugins: PluginArgs,
    #[command(flatten)]
    pub sketch: SketchArgs,
}

pub fn run(args: AutoRepArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        AutoRepCommand::Init(a) => run_init(&a),
        AutoRepCommand::Run(a) => run_campaign(&a),
    }
}

fn refuse_existing(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if path.exists() {
        return Err(format!("{} exists; refusing to overwrite it", path.display()).into());
    }
    Ok(())
}

fn run_init(args: &InitArgs) -> Result<(), Box<dyn std::error::Error>> {
    refuse_existing(&args.output)?;
    let spec = RepresentSpec {
        encoding: args.encoding.clone(),
        ..RepresentSpec::nvfp4()
    };
    let vocabulary: Option<ActionVocabulary> = match &args.vocabulary {
        Some(path) => Some(serde_json::from_slice(&std::fs::read(path)?)?),
        None => None,
    };
    let snapshot = produce(&ProduceInputs {
        source: &args.container,
        spec: &spec,
        bank: &args.bank,
        sequences: args.sequences,
        vocabulary: vocabulary.as_ref(),
        arming: args.arming()?,
    })?;
    std::fs::write(&args.output, serde_json::to_vec_pretty(&snapshot)?)?;
    println!("auto-rep init: {}", args.output.display());
    println!("  source   : {}", args.container.display());
    println!(
        "  bank     : {} ({} sequences)",
        args.bank.display(),
        args.sequences
    );
    println!("  surface  : {} tensors", snapshot.space().surface.len());
    println!("  groups   : {}", snapshot.space().vocabulary.len());
    match snapshot.gate() {
        Some(gate) => println!(
            "  gate     : {} (tail ≥ {} observations)",
            gate.id(),
            snapshot.config().tail_support.min_tail_observations
        ),
        None => println!("  gate     : none (characterisation-only)"),
    }
    Ok(())
}

/// Replace `path` with `value` in one rename, so a reader never sees half a
/// file.
fn write_atomically(path: &Path, value: &impl serde::Serialize) -> Result<(), VindexError> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| VindexError::Parse(e.to_string()))?;
    let partial = path.with_extension("partial");
    std::fs::write(&partial, bytes)?;
    std::fs::rename(&partial, path)?;
    Ok(())
}

/// [`compile_representation_with`] over the plugins' encoders.
struct PluginCompiler<'a> {
    source: &'a Path,
    encoders: &'a EncoderRegistry,
}

impl CandidateCompiler for PluginCompiler<'_> {
    fn compile(&self, spec: &RepresentSpec, out: &Path) -> Result<(), VindexError> {
        compile_representation_with(self.source, out, spec, self.encoders).map(drop)
    }
}

fn run_campaign(args: &RunArgs) -> Result<(), Box<dyn std::error::Error>> {
    let (mut snapshot, prior): (SearchSnapshot, Option<CampaignRecord>) = if args.resume {
        (
            serde_json::from_slice(&std::fs::read(&args.output)?)?,
            Some(serde_json::from_slice(&std::fs::read(&args.campaign)?)?),
        )
    } else {
        refuse_existing(&args.output)?;
        refuse_existing(&args.campaign)?;
        let path = args
            .snapshot
            .as_ref()
            .ok_or("--snapshot is required unless --resume")?;
        (serde_json::from_slice(&std::fs::read(path)?)?, None)
    };
    snapshot.check_schema()?;
    let plugins = Plugins::load(&args.plugins)?;
    let spec = RepresentSpec {
        encoding: snapshot.space().base_map.encoding.clone(),
        ..RepresentSpec::nvfp4()
    };
    std::fs::create_dir_all(&args.workdir)?;
    std::fs::create_dir_all(&args.runs)?;
    let executor = VerbPlanExecutor {
        reference_backend: args.reference_backend,
        candidate_backend: args.candidate_backend,
        component: args.component.clone(),
        plugins: args.plugins.plugins.clone(),
        output_root: args.runs.clone(),
        sketch: args.sketch.spec()?,
    };
    let registry = ExecutorRegistry::new([&executor as &dyn ExperimentExecutor])?;
    let compiler = PluginCompiler {
        source: &args.source,
        encoders: plugins.encoders,
    };
    // The record first, then the campaign: interrupted between the two, the
    // campaign lags the record, which a resume accepts (the extra reading is
    // already a cut); the other order could list a reading the record lacks.
    let persist = |snapshot: &SearchSnapshot, record: &CampaignRecord| {
        write_atomically(&args.output, snapshot)?;
        write_atomically(&args.campaign, record)
    };
    let record = auto_rep::run_resuming(
        &mut snapshot,
        &CampaignSetup {
            source: &args.source,
            corpus: &args.bank,
            workdir: &args.workdir,
            spec: &spec,
            compiler: &compiler,
            executors: &registry,
            budget: args.budget,
            node_limit: args.node_limit,
            pins: Default::default(),
            checkpoint: Some(&persist),
        },
        prior,
    )?;
    persist(&snapshot, &record)?;
    println!("auto-rep run: {:?}", record.outcome);
    println!("  measurements spent: {}", record.measurements_spent);
    for entry in &record.entries {
        println!(
            "  rank-1 {:>12} bytes  protects {:>3} group(s)  {}  {:?}",
            entry.proposal.bytes,
            entry.proposal.protected.len(),
            if entry.reused { "reused  " } else { "measured" },
            entry.verdict
        );
    }
    println!("  record  : {}", args.output.display());
    println!("  campaign: {}", args.campaign.display());
    Ok(())
}
