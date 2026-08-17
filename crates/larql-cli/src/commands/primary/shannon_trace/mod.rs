//! `larql shannon layer-dump` / `layer-diff` — the Gate-B instrument.
//!
//! [`docs/k3-funnel.md`] §4 defines Gate B at rungs R1/R2 as "a layer-by-layer
//! f32 diff against the local reference implementation, **then** `larql shannon
//! verify` ≤ 0.5 % bits/char". Only the second half existed. `shannon verify`
//! compares one scalar at the end of a forward pass, so it can say *that* two
//! engines disagree and never *where* — and §4.7's six-defect MoE hunt was run
//! without it, by transcribing one block into a test rather than diffing the
//! pass.
//!
//! These two subcommands are the first half. `layer-dump` writes each
//! end-of-layer residual as a raw f32 plane plus a manifest; `layer-diff`
//! compares two dumps and names the first capture that drifts. The reference
//! side is `scripts/dump_layers_hf.py`, which takes its **token ids from the
//! manifest** rather than re-tokenising — §4.7.7's method note stated in code:
//! both sides of a comparison must use the same fixture, and the tokenizer is
//! part of the fixture.

use std::path::PathBuf;

use clap::Args;

pub mod compare;
pub mod decode_diff;
pub mod dump;

/// Cosine similarity below which a capture is reported as drifted.
///
/// Matches `examples/residual_diff.rs`'s CPU-vs-Metal threshold. Two f32
/// forwards of the same arithmetic in a different order sit far above this;
/// below it is a structural difference, not accumulation.
pub const DEFAULT_DRIFT_COS: f64 = 0.9999;

/// Relative-RMS error above which a capture counts as drifted.
///
/// Cosine alone cannot see a magnitude error: for `a = alpha * b` the
/// cosine is exactly 1.0 for every positive `alpha`. Two real Glimmer
/// defects were of precisely that form locally — a missing weightless
/// embedding norm and a flattened centred-norm offset — and a
/// cosine-only gate scored the guilty capture 1.000000000 and blamed
/// the layer after it.
///
/// `rel_rms` is `||a - b|| / ||b||`, so for a pure rescale it reads
/// `|1 - alpha|` directly: the missing embedding norm showed 0.9375,
/// exactly `|1 - 1/16|`. The default sits well above genuine bf16-vs-f32
/// accumulation (~0.008 measured over 52 layers) and well below any
/// scale defect worth naming.
///
/// **This is a localisation threshold, not a correctness tolerance.**
/// It exists to name the shallowest capture worth investigating. It does
/// *not* assert that 0.049 is semantically safe — nothing here licenses
/// a claim about agreement below it. The raw table stays authoritative:
/// read the columns, not the verdict line.
pub const DEFAULT_DRIFT_REL_RMS: f64 = 0.05;

#[derive(Args)]
pub struct LayerDumpArgs {
    /// Model path or HuggingFace model ID.
    pub model: String,

    /// UTF-8 corpus file supplying the token window.
    #[arg(
        long,
        value_name = "FILE",
        required_unless_present = "tokens",
        conflicts_with = "tokens"
    )]
    pub corpus: Option<PathBuf>,

    /// Comma-separated token ids to run instead of tokenising a corpus —
    /// given, never tokenised here, so this dump can be diffed against a
    /// `vindex3 exec --dump-layers` run of the *same* ids: the two engines
    /// then cannot silently score different token sequences.
    #[arg(long, value_name = "IDS", conflicts_with = "corpus")]
    pub tokens: Option<String>,

    /// Limit input to the first N bytes, truncated on a UTF-8 boundary.
    #[arg(long)]
    pub bytes: Option<usize>,

    /// Truncate the token window to at most N tokens. One window, no sliding:
    /// a diff wants a single deterministic pass, not an aggregate over windows.
    #[arg(long, default_value_t = super::shannon_cmd::DEFAULT_CONTEXT)]
    pub context: usize,

    /// Output directory for the planes and manifest.
    #[arg(long, value_name = "DIR")]
    pub out: PathBuf,
}

#[derive(Args)]
pub struct LayerDiffArgs {
    /// First dump directory (conventionally the engine under test).
    #[arg(long, value_name = "DIR")]
    pub a: PathBuf,

    /// Second dump directory (conventionally the reference).
    #[arg(long, value_name = "DIR")]
    pub b: PathBuf,

    /// Cosine similarity below which a capture counts as drifted.
    #[arg(long, default_value_t = DEFAULT_DRIFT_COS)]
    pub threshold_cos: f64,

    /// Relative RMS error above which a capture counts as drifted.
    /// Catches magnitude errors cosine is blind to.
    #[arg(long, default_value_t = DEFAULT_DRIFT_REL_RMS)]
    pub threshold_rel_rms: f64,

    /// Emit a `RESULT {...}` JSON line after the table.
    #[arg(long)]
    pub json: bool,
}

/// `decode-diff` — CPU-vs-Metal parity across the prefill→decode seam.
///
/// A separate axis from `layer-diff`: that one compares this engine to an
/// external reference over a prefill, which by construction cannot see a
/// decode defect. See [`decode_diff`] for why that distinction cost a
/// week once already.
#[derive(Args)]
pub struct DecodeDiffArgs {
    /// Vindex directory to load.
    #[arg(value_name = "VINDEX")]
    pub vindex: PathBuf,

    /// Prompt supplying the token window.
    #[arg(long, default_value = "The capital of France is")]
    pub prompt: String,

    /// Decode this many trailing tokens instead of prefilling them. More
    /// than one also exercises the decode→decode KV hand-off, which a
    /// single step does not reach.
    #[arg(long, default_value_t = 1)]
    pub steps: usize,

    /// Cosine similarity below which a layer counts as drifted.
    #[arg(long, default_value_t = DEFAULT_DRIFT_COS)]
    pub threshold_cos: f64,

    /// Emit a `RESULT {...}` JSON line after the table.
    #[arg(long)]
    pub json: bool,
}
