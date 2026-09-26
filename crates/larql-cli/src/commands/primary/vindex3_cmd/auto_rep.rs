//! `vindex3 auto-rep` — AUTO-REP over a plan-v1 record (MEASURE-PLAN-2,
//! amendment A2).
//!
//! `init` produces a characterisation-only record from a container and a
//! token bank. `run` advances a record through AUTO-REP-1b's loop with the
//! same arms `vindex3 measure` builds (lowered on Metal, interpreter
//! otherwise). A record without a gate is refused by
//! the loop itself: arming one is slice 3's pre-registration, not this
//! verb's. The record format, the producer and the loop live in
//! `larql_vindex::format::vindex3::represent`; this file names arguments.

use std::path::{Path, PathBuf};

use clap::{Args, Subcommand};
use larql_vindex::format::vindex3::represent::actuate::executor::{
    ExecutorRegistry, ExperimentExecutor,
};
use larql_vindex::format::vindex3::represent::auto_rep::{self, CampaignSetup, CandidateCompiler};
use larql_vindex::format::vindex3::represent::codec::EncoderRegistry;
use larql_vindex::format::vindex3::represent::produce::{produce, ProduceInputs};
use larql_vindex::format::vindex3::represent::state::snapshot::SearchSnapshot;
use larql_vindex::format::vindex3::represent::{compile_representation_with, RepresentSpec};
use larql_vindex::VindexError;

use super::measure::VerbPlanExecutor;
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
    /// Source container every candidate is compiled from.
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
}

#[derive(Args)]
pub struct RunArgs {
    /// The record to advance.
    #[arg(long)]
    pub snapshot: PathBuf,
    /// Where the advanced record is written. Refused if it exists: the
    /// input record is never overwritten.
    #[arg(long)]
    pub output: PathBuf,
    /// Where the campaign record is written. Refused if it exists.
    #[arg(long)]
    pub campaign: PathBuf,
    /// The source container the record describes.
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
    let snapshot = produce(&ProduceInputs {
        source: &args.container,
        spec: &spec,
        bank: &args.bank,
        sequences: args.sequences,
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
    println!("  gate     : none (characterisation-only; slice 3 arms it)");
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
    refuse_existing(&args.output)?;
    refuse_existing(&args.campaign)?;
    let mut snapshot: SearchSnapshot = serde_json::from_slice(&std::fs::read(&args.snapshot)?)?;
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
    };
    let registry = ExecutorRegistry::new([&executor as &dyn ExperimentExecutor])?;
    let compiler = PluginCompiler {
        source: &args.source,
        encoders: plugins.encoders,
    };
    let record = auto_rep::run(
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
        },
    )?;
    std::fs::write(&args.output, serde_json::to_vec_pretty(&snapshot)?)?;
    std::fs::write(&args.campaign, serde_json::to_vec_pretty(&record)?)?;
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
