//! The `shannon` command surface: subcommand enum and argument structs.

use super::*;

#[derive(Subcommand)]
pub enum ShannonCommand {
    /// Score a corpus as model next-token bits.
    Score(ScoreArgs),

    /// Score an answer slot after a prefix, e.g. "The capital of France is " + "Paris".
    Slot(SlotArgs),

    /// Score repeated occurrences of a needle in a passage.
    Repeat(RepeatArgs),

    /// Per-layer Shannon bits via the final-norm logit lens.
    /// At every layer L (embed plus each post-block residual), project through
    /// `final_norm + lm_head` and report bits/token, KL-to-final, and the
    /// adjacent `bits_saved[L] = bits_via_lens[L-1] - bits_via_lens[L]` deltas.
    Layers(LayersArgs),

    /// Encode a short text file with model-driven arithmetic coding.
    Encode(EncodeArgs),

    /// Decode a file produced by `larql shannon encode`.
    Decode(DecodeArgs),

    /// Cross-engine bits/char comparison. Orchestrates `shannon score` (LARQL
    /// Rust, in-process) plus optional MLX and HF/PyTorch reference scorers
    /// (subprocesses); prints a delta table and exits non-zero if any pair-wise
    /// delta exceeds `--threshold`. See `scripts/README_shannon_score.md`.
    Verify(VerifyArgs),

    /// Dump every end-of-layer residual of one forward pass as raw f32 planes.
    /// The *where* half of Gate B — `verify` says two engines disagree, this
    /// says at which layer. See [`crate::commands::primary::shannon_trace`].
    LayerDump(crate::commands::primary::shannon_trace::LayerDumpArgs),

    /// BW12-0's static half: dump every (layer, expert)'s down-projection
    /// column norms through `LARQL_MOE_BLOCK_CONTRIB_TRACE`. See
    /// [`crate::commands::primary::shannon_trace::block_norms`].
    BlockNorms(crate::commands::primary::shannon_trace::BlockNormsArgs),

    /// Compare two `layer-dump` directories layer by layer and name the first
    /// capture that drifts.
    LayerDiff(crate::commands::primary::shannon_trace::LayerDiffArgs),

    /// CPU-vs-Metal parity across the prefill→decode seam. `layer-diff`
    /// compares this engine to an external reference over a prefill and so
    /// cannot see a decode-only defect; this one can.
    DecodeDiff(crate::commands::primary::shannon_trace::DecodeDiffArgs),
}

#[derive(Args)]
pub struct ScoreArgs {
    /// Model path or HuggingFace model ID.
    pub(super) model: String,

    /// UTF-8 corpus file to score.
    #[arg(long, value_name = "FILE")]
    pub(super) corpus: PathBuf,

    /// Limit input to the first N bytes, truncated on a UTF-8 boundary.
    #[arg(long)]
    pub(super) bytes: Option<usize>,

    /// Maximum tokens in each scoring forward window.
    #[arg(long, default_value_t = DEFAULT_CONTEXT)]
    pub(super) context: usize,

    /// Newly-scored target tokens per forward window.
    #[arg(long, default_value_t = DEFAULT_STRIDE)]
    pub(super) stride: usize,
}

#[derive(Args)]
pub struct SlotArgs {
    /// Model path or HuggingFace model ID.
    pub(super) model: String,

    /// Prefix before the answer slot. Include boundary whitespace if needed.
    #[arg(long)]
    pub(super) prefix: String,

    /// Slot text to score.
    #[arg(long)]
    pub(super) answer: String,

    /// Maximum tokens in the scoring forward window.
    #[arg(long, default_value_t = DEFAULT_CONTEXT)]
    pub(super) context: usize,

    /// Show top-k predictions before the first answer token.
    #[arg(long, default_value_t = 5)]
    pub(super) top_k: usize,
}

#[derive(Args)]
pub struct RepeatArgs {
    /// Model path or HuggingFace model ID.
    pub(super) model: String,

    /// UTF-8 passage file.
    #[arg(long, value_name = "FILE")]
    pub(super) text: PathBuf,

    /// String whose occurrences should be scored in context.
    #[arg(long)]
    pub(super) needle: String,

    /// Limit input to the first N bytes, truncated on a UTF-8 boundary.
    #[arg(long)]
    pub(super) bytes: Option<usize>,

    /// Maximum tokens in the scoring forward window.
    #[arg(long, default_value_t = DEFAULT_CONTEXT)]
    pub(super) context: usize,
}

#[derive(Args)]
pub struct LayersArgs {
    /// Model path or HuggingFace model ID.
    pub(super) model: String,

    /// UTF-8 corpus file to score.
    #[arg(long, value_name = "FILE")]
    pub(super) corpus: PathBuf,

    /// Limit input to the first N bytes, truncated on a UTF-8 boundary.
    #[arg(long)]
    pub(super) bytes: Option<usize>,

    /// Maximum tokens in each scoring forward window.
    #[arg(long, default_value_t = DEFAULT_CONTEXT)]
    pub(super) context: usize,

    /// Newly-scored target tokens per forward window.
    #[arg(long, default_value_t = DEFAULT_STRIDE)]
    pub(super) stride: usize,
}

#[derive(Args)]
pub struct EncodeArgs {
    /// Model path or HuggingFace model ID.
    pub(super) model: String,

    /// UTF-8 input text.
    #[arg(long = "in", value_name = "FILE")]
    pub(super) input: PathBuf,

    /// Compressed output file.
    #[arg(long, value_name = "FILE")]
    pub(super) out: PathBuf,

    /// Limit input to the first N bytes, truncated on a UTF-8 boundary.
    #[arg(long)]
    pub(super) bytes: Option<usize>,

    /// Previous tokens visible to the model for each arithmetic-code step.
    /// Ignored when --vindex is used; the KV-cache path uses 512-token blocks.
    #[arg(long, default_value_t = 256)]
    pub(super) context: usize,

    /// Use a Q4K vindex for KV-cached forced-token scoring instead of raw HF weights.
    #[arg(long, value_name = "DIR")]
    pub(super) vindex: Option<PathBuf>,

    /// Use the best GPU backend for the vindex path. Required for the fast Q4K path.
    #[arg(long)]
    pub(super) metal: bool,
}

#[derive(Args)]
pub struct DecodeArgs {
    /// Model path or HuggingFace model ID. Must match the encoder model.
    pub(super) model: String,

    /// File produced by `larql shannon encode`.
    #[arg(long = "in", value_name = "FILE")]
    pub(super) input: PathBuf,

    /// Recovered UTF-8 text output.
    #[arg(long, value_name = "FILE")]
    pub(super) out: PathBuf,

    /// Use a Q4K vindex for KV-cached forced-token scoring instead of raw HF weights.
    #[arg(long, value_name = "DIR")]
    pub(super) vindex: Option<PathBuf>,

    /// Use the best GPU backend for the vindex path. Required for the fast Q4K path.
    #[arg(long)]
    pub(super) metal: bool,
}

#[derive(Args)]
pub struct VerifyArgs {
    /// Model path or HuggingFace model ID.
    pub(super) model: String,

    /// UTF-8 corpus file to score. CRLF is normalized to LF before scoring so
    /// the three engines agree on tokenization (Python text I/O strips \r,
    /// LARQL Rust doesn't — see scripts/README_shannon_score.md).
    #[arg(long, value_name = "FILE")]
    pub(super) corpus: PathBuf,

    /// Limit input to the first N bytes, truncated on a UTF-8 boundary.
    #[arg(long)]
    pub(super) bytes: Option<usize>,

    /// Maximum tokens in each scoring forward window.
    #[arg(long, default_value_t = DEFAULT_CONTEXT)]
    pub(super) context: usize,

    /// Newly-scored target tokens per forward window.
    #[arg(long, default_value_t = DEFAULT_STRIDE)]
    pub(super) stride: usize,

    /// Comma-separated reference engines to run alongside LARQL Rust.
    /// Available: `mlx`, `hf`. Default: both.
    #[arg(long, default_value = "mlx,hf", value_name = "LIST")]
    pub(super) engines: String,

    /// Maximum acceptable pair-wise delta in percent. Exits non-zero if any
    /// pair of engines disagrees by more than this on total bits.
    #[arg(long, default_value_t = 0.5)]
    pub(super) threshold: f64,

    /// Python interpreter used to invoke the MLX and HF reference scorers.
    #[arg(long, default_value = ".venv/bin/python")]
    pub(super) python: PathBuf,

    /// Override the MLX scorer script location.
    #[arg(
        long,
        default_value = "scripts/shannon_score_mlx.py",
        value_name = "FILE"
    )]
    pub(super) mlx_script: PathBuf,

    /// Override the HF scorer script location.
    #[arg(
        long,
        default_value = "scripts/shannon_score_hf.py",
        value_name = "FILE"
    )]
    pub(super) hf_script: PathBuf,

    /// Device passed to the HF scorer. `cpu` is deterministic; `mps` is faster.
    #[arg(long, default_value = "cpu")]
    pub(super) hf_device: String,

    /// Emit a final `RESULT {...}` JSON line on stdout in addition to the
    /// human-readable delta table. Mirrors the `--json` flag on the Python
    /// reference scorers and is what `scripts/diagnose_models.py` consumes
    /// when sweeping multiple architectures, so the multi-arch driver
    /// doesn't have to regex-parse the formatted table.
    #[arg(long)]
    pub(super) json: bool,
}
