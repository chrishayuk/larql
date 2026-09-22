//! MAD-V3-KNN oracle tooling.
//!
//! kNN is deliberately outside inference: this command reads completed
//! capture artifacts and asks whether a residual is a useful historical key
//! for semantic identity and future model-object access. It never changes the
//! model's execution or routes a live token.

mod capture;
mod census;
mod evaluate;
mod format;

use std::path::PathBuf;

use clap::{Args, Subcommand, ValueEnum};

pub use census::run_census;
pub use evaluate::run_evaluate;

#[derive(Args)]
pub struct MadKnnArgs {
    #[command(subcommand)]
    command: MadKnnCommand,
}

#[derive(Subcommand)]
enum MadKnnCommand {
    /// Capture residual keys and exact raw FFN-block contribution masses.
    Capture(CaptureArgs),
    /// Score canonical-key purity and future-object contribution coverage.
    Evaluate(EvaluateArgs),
    /// Census per-layer FFN sparsity and oracle headroom before fitting kNN.
    Census(CensusArgs),
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum CaptureBackend {
    /// Naive f32 semantic anchor.
    Reference,
    /// Production CPU kernels with f32 resident operands.
    Production,
    /// Metal projections with f16 resident operands. This uses the generic
    /// VINDEX3 device executor so diagnostic taps remain visible.
    #[cfg(feature = "gpu")]
    Metal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ContributionEngine {
    /// Scalar CPU authority: f16/f32 weights are visited in block/output/
    /// channel order and block norms accumulate through f64.
    Exact,
    /// Accelerator observer primitive. It measures the same raw block output
    /// but uses device-parallel reduction and must pass the exact parity gate.
    Fast,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ValueEnum)]
pub enum SeamKind {
    /// Normalised vector consumed by the attention projections.
    AttentionInput,
    /// Attention branch output immediately before its residual addition.
    AttentionOutput,
    /// Residual after the attention branch has been added.
    PostAttention,
    /// Normalised vector consumed by the FFN projections.
    FfnInput,
    /// FFN branch output immediately before its residual addition.
    FfnOutput,
}

impl SeamKind {
    const ORDERED: [Self; 5] = [
        Self::AttentionInput,
        Self::AttentionOutput,
        Self::PostAttention,
        Self::FfnInput,
        Self::FfnOutput,
    ];

    fn as_str(self) -> &'static str {
        match self {
            Self::AttentionInput => "attention_input",
            Self::AttentionOutput => "attention_output",
            Self::PostAttention => "post_attention",
            Self::FfnInput => "ffn_input",
            Self::FfnOutput => "ffn_output",
        }
    }
}

#[derive(Args)]
pub struct CaptureArgs {
    /// VINDEX3 container directory.
    pub container: PathBuf,

    /// JSONL query fixture. Each row carries id, semantic_id, wording_id,
    /// split, optional group_id, and token_ids.
    #[arg(long)]
    pub queries: PathBuf,

    /// New capture directory. A manifest is written last as the completion
    /// marker; a non-empty existing directory is refused.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Component in the VINDEX3 system graph.
    #[arg(long, default_value = "target")]
    pub component: String,

    /// Residuals entering these layers become kNN keys.
    #[arg(long, default_value = "1,8,16,24,32,40,48,51")]
    pub residual_layers: String,

    /// Layers whose FFN channel-block contributions are recorded. Defaults
    /// to every layer so arbitrary future horizons remain measurable.
    #[arg(long)]
    pub contribution_layers: Option<String>,

    /// Intermediate channels per logical model object.
    #[arg(long, default_value_t = 128)]
    pub block_channels: usize,

    /// Capture several exact contiguous page schemas in the same forward.
    /// When present this overrides `--block-channels`; comma-separated values
    /// are accepted. The FFN activation is computed once and each observer
    /// reduction remains read-only.
    #[arg(long, value_delimiter = ',')]
    pub page_channels: Vec<usize>,

    /// Layers whose hidden-width execution seams are recorded for ADDR-1.
    /// Requires `--seam-kinds`.
    #[arg(long)]
    pub seam_layers: Option<String>,

    /// Read-only execution seams captured at every selected seam layer.
    /// Requires `--seam-layers`; comma-separated values are accepted.
    #[arg(long, value_enum, value_delimiter = ',')]
    pub seam_kinds: Vec<SeamKind>,

    /// Arithmetic used only for contribution observation. This never changes
    /// the model's forward execution.
    #[arg(long, value_enum, default_value_t = ContributionEngine::Exact)]
    pub contribution_engine: ContributionEngine,

    /// Numerical realisation used for the unchanged forward pass.
    #[arg(long, value_enum, default_value_t = CaptureBackend::Production)]
    pub backend: CaptureBackend,

    /// Capture only the first N fixture rows (smoke runs).
    #[arg(long)]
    pub limit: Option<usize>,

    /// Resume from the last per-sample commit in an interrupted capture.
    #[arg(long)]
    pub resume: bool,
}

#[derive(Args)]
pub struct CensusArgs {
    /// Capture directory containing all FFN contribution layers.
    pub capture: PathBuf,

    /// Comma-separated fractions of each layer's object bytes.
    #[arg(long, default_value = "0.05,0.10,0.20")]
    pub byte_budgets: String,

    /// Score only query rows carrying this regime label. History rows remain
    /// the popularity authority.
    #[arg(long)]
    pub query_regime: Option<String>,

    /// Write the JSON report here instead of stdout.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum SearchKind {
    /// Exhaustive cosine search. Exact, but quadratic in capture size.
    Exact,
    /// Projected HNSW traversal followed by exact cosine re-ranking.
    /// The report includes a sampled recall audit against exhaustive search.
    Hnsw,
}

#[derive(Args)]
pub struct EvaluateArgs {
    /// Capture directory containing manifest.json and its binary planes.
    pub capture: PathBuf,

    /// Comma-separated neighbour counts.
    #[arg(long, default_value = "1,4,8,16")]
    pub neighbors: String,

    /// Comma-separated fractions of the target layer's object bytes.
    #[arg(long, default_value = "0.01,0.05,0.10,0.20")]
    pub byte_budgets: String,

    /// Comma-separated future layer offsets.
    #[arg(long, default_value = "1,4,8")]
    pub horizons: String,

    /// Search guarantee. Exact cosine is the scientific default and headline;
    /// HNSW is an explicitly approximate scalability arm.
    #[arg(long, value_enum, default_value_t = SearchKind::Exact)]
    pub search: SearchKind,

    /// HNSW connections per node.
    #[arg(long, default_value_t = 16)]
    pub hnsw_m: usize,

    /// HNSW construction beam width.
    #[arg(long, default_value_t = 100)]
    pub ef_construction: usize,

    /// HNSW query beam width.
    #[arg(long, default_value_t = 128)]
    pub ef_search: usize,

    /// Number of query rows per layer audited against exact search.
    #[arg(long, default_value_t = 32)]
    pub audit_queries: usize,

    /// Do not admit historical rows with the query's group_id. This is the
    /// held-out-graph control; rows without group_id are unaffected.
    #[arg(long)]
    pub exclude_same_group: bool,

    /// Score only query rows carrying this regime label. History rows remain
    /// the common index for all arms.
    #[arg(long)]
    pub query_regime: Option<String>,

    /// Write the JSON report here instead of stdout.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

pub fn run(args: MadKnnArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        MadKnnCommand::Capture(args) => capture::run_capture(args),
        MadKnnCommand::Evaluate(args) => run_evaluate(args),
        MadKnnCommand::Census(args) => run_census(args),
    }
}
