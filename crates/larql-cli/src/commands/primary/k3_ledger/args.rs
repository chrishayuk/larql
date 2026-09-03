//! `larql k3-ledger` clap surface.

use clap::{Args, Subcommand};

#[derive(Debug, Args)]
pub struct K3LedgerArgs {
    #[command(subcommand)]
    pub cmd: K3LedgerCmd,

    /// Hugging Face repo to read geometry from.
    #[arg(long, global = true, default_value = "moonshotai/Kimi-K3")]
    pub repo: String,

    /// Shard holding a KDA (linear-attention) layer.
    #[arg(
        long,
        global = true,
        default_value = "model-00050-of-000096.safetensors"
    )]
    pub kda_shard: String,

    /// Shard holding a full-attention (MLA) layer. NOTE the config's layer lists
    /// are 1-indexed against 0-indexed tensor names — config layer 4 is tensor
    /// layer 3. Picking wrong yields another KDA layer; the loader rejects it.
    #[arg(
        long,
        global = true,
        default_value = "model-00004-of-000096.safetensors"
    )]
    pub mla_shard: String,

    /// Attainable memory bandwidth, GB/s (measured, not spec sheet).
    #[arg(long, global = true, default_value_t = 367.0)]
    pub bandwidth_gb_s: f64,

    /// Fraction of that bandwidth the kernel realises. Owner: the MXFP4 bench.
    #[arg(long, global = true, default_value_t = 1.0)]
    pub dequant_efficiency: f64,

    /// Price the dense side at a HYPOTHETICAL width instead of the one the
    /// checkpoint stores.
    ///
    /// K3-L1-F1: this used to be a hardcoded 4.25 with no way to ask for the
    /// truth. It is now opt-in, and every row produced under it is labelled
    /// HYPOTHETICAL. Omit it to price the model as it actually is.
    #[arg(long, global = true)]
    pub hypothetical_dense_bits: Option<f64>,

    /// Emit the full record as JSON instead of a table.
    #[arg(long, global = true)]
    pub json: bool,
}

#[derive(Debug, Subcommand)]
pub enum K3LedgerCmd {
    /// DEC-8.0 — per-token fetch budget when experts live on external storage.
    Budget(BudgetArgs),
    /// DEC-8.6 — weight touch once resident, split reducible vs irreducible.
    Touch(TouchArgs),
    /// DEC-8.7a — dense-precision frontier; the target generator.
    Frontier(FrontierArgs),
    /// DEC-9.2 — speculative block economics, work-width separate from commit.
    Block(BlockArgs),
    /// Can MXFP4 transcode to Q6_K exactly? Scans e8m0 exponent spread.
    TranscodeScan(TranscodeScanArgs),
    /// Composed per-class ceiling, and the R4 best case for each proposed lever.
    Ceilings(CeilingsArgs),
    /// Witness that one measured layer represents its whole family, by
    /// reading EVERY layer's safetensors header. Turns the census's
    /// "x69" from an assumption into a structural fact.
    Homogeneity(HomogeneityArgs),
    /// K3-ACTIONS-1: the weight-free physical action catalogue.
    Actions(ActionsArgs),
    /// Which containers hold MXFP4 exactly, and at what width. Decided by
    /// counting the source alphabet — takes no checkpoint and no network.
    Formats,
    /// Measured MXFP4 symbol distribution: is an exact VARIABLE-rate code below
    /// 4 payload bits available? Reads real packed nibbles by range request.
    SymbolCensus(SymbolCensusArgs),
    /// ETC-0A — is an expert cheaper to send GIVEN another expert? Conditional
    /// entropy and mutual information on the untouched MXFP4 symbol streams,
    /// against a marginal-preserving shuffled null, priced straight into
    /// effective transport bpw and tok/s on both arms.
    CondCensus(CondCensusArgs),
    /// Trace which KDA projections read the same hidden state, and what a shared
    /// input rotation would have to be folded into. Reads the real tensor table.
    KdaGraph(KdaGraphArgs),
    /// DEC-9A — retention oracle gate. How much external traffic is avoidable
    /// by perfect RETENTION decisions alone? Belady/MIN bounds every predictor
    /// that could ever be built; a best-fixed-set arm isolates how much of that
    /// is merely popularity. Reads a local pool — no checkpoint, no network.
    Retention(RetentionArgs),
    /// Frequency-mass coverage from a captured routing trace: what a resident
    /// set of C symbols per stratum actually serves, static vs adaptive vs
    /// oracle vs null. Reads a local pool — no checkpoint, no network.
    FreqMass(FreqMassArgs),
}

#[derive(Debug, Args)]
pub struct RetentionArgs {
    /// DEC residual capture pool captured with `--routing`.
    #[arg(long)]
    pub pool: std::path::PathBuf,

    /// Cache capacities to sweep, as a fraction of the layer-expert bank. A
    /// capacity below one stratum's simultaneous demand can only thrash, and
    /// the report flags those rows rather than ranking policies inside them.
    #[arg(long, num_args = 1.., default_values_t = [0.01f64, 0.02, 0.04, 0.0625, 0.08, 0.125, 0.25])]
    pub capacity_fracs: Vec<f64>,

    /// Reset the cache at every session boundary. Default is warm (kept
    /// across sessions), which is the serving case.
    #[arg(long)]
    pub cold: bool,

    /// Seed for the random-eviction control.
    #[arg(long, default_value_t = 20_260_810)]
    pub seed: u64,

    /// Bytes per cached expert. Defaults to K3's routed expert, which on any
    /// other model's trace is a scaled projection for readability and not a
    /// measurement of that model (R2).
    #[arg(long)]
    pub expert_bytes: Option<u64>,
}

#[derive(Debug, Args)]
pub struct FreqMassArgs {
    /// DEC residual capture pool captured with `--routing`.
    #[arg(long)]
    pub pool: std::path::PathBuf,

    /// Resident-set sizes to score, in symbols per stratum.
    #[arg(long, num_args = 1.., default_values_t = [1usize, 2, 4, 8, 16, 32, 64])]
    pub cache_sizes: Vec<usize>,

    /// Shrinkage weights on the pooled prior. 1.0 = static slice, 0.0 = causal
    /// adaptive cache; an interior winner is a shrinkage cache.
    #[arg(long, num_args = 1.., default_values_t = [0.0f64, 0.25, 0.5, 0.75, 0.9, 1.0])]
    pub lambdas: Vec<f64>,

    /// Fraction of each session's steps used to fit the adaptive key; the rest
    /// is scored. A cache cannot rank on events it has not seen.
    #[arg(long, default_value_t = super::freqmass::DEFAULT_FIT_FRACTION)]
    pub fit_fraction: f64,

    /// Seed for the marginal-preserving null.
    #[arg(long, default_value_t = super::freqmass::DEFAULT_NULL_SEED)]
    pub seed: u64,

    /// Step counts at which to report support (union) growth.
    #[arg(long, num_args = 1.., default_values_t = [1usize, 2, 4, 8, 16])]
    pub support_steps: Vec<usize>,

    /// Residency fraction to grade the operating point at. K3's proposed
    /// 55-65 GB expert cache is 5-7% of its bank, so the default is 6.25%.
    #[arg(long, default_value_t = 0.0625)]
    pub operating_residency: f64,

    /// Also emit the measured per-symbol mass vector (counts and probabilities,
    /// per stratum) — DEC-8.4's precision allocator consumes this rather than
    /// differencing the coverage curve. JSON only; one row per selected symbol,
    /// so a 92-layer 896-expert bank emits tens of thousands.
    #[arg(long)]
    pub per_symbol: bool,
}

#[derive(Debug, Args)]
pub struct KdaGraphArgs {
    /// Tensor-name layer index to trace (0-indexed, as tensor names are).
    #[arg(long, default_value_t = 49)]
    pub layer: usize,
}

#[derive(Debug, Args)]
pub struct SymbolCensusArgs {
    /// Expert indices to sample from the KDA shard.
    #[arg(long, num_args = 1.., default_values_t = [0usize, 1, 7])]
    pub experts: Vec<usize>,
    /// Rows read from each tensor. A row is one weight vector; tiles never
    /// straddle one, so this bounds traffic without biasing the census.
    #[arg(long, default_value_t = 256)]
    pub rows: usize,
    /// Tile sizes to report cardinality and local entropy at.
    #[arg(long, num_args = 1.., default_values_t = [32usize, 64, 256, 512])]
    pub blocks: Vec<usize>,
}

#[derive(Debug, Args)]
pub struct CondCensusArgs {
    /// Experts to sample, taken from index 0 upward. The leave-one-out
    /// prototype is built from exactly this set, so a wider bank is a better
    /// prototype *and* more candidate parents — but also quadratic selection
    /// cost. Traffic is `experts * rows * row_bytes`, a few MB at the default.
    #[arg(long, default_value_t = 16)]
    pub experts: usize,

    /// Rows read from each expert tensor. A row is one weight vector; the
    /// census aligns by coordinate, so every expert must read the same rows.
    #[arg(long, default_value_t = 256)]
    pub rows: usize,

    /// Which GLU branch to census. `w1` is the gate, `w3` the up, `w2` the down
    /// — they are equal to the byte, but they are not the same matrices, and a
    /// result on one is not a result on the others.
    #[arg(long, default_value = "w1")]
    pub branch: String,

    /// Fraction of coordinates used to CHOOSE coding parents. The rest are
    /// scored. Selecting and scoring on the same coordinates turns the parent
    /// arm into an oracle.
    #[arg(long, default_value_t = super::cond_arms::DEFAULT_FIT_FRACTION)]
    pub fit_fraction: f64,

    /// Seed for the marginal-preserving shuffled nulls.
    #[arg(long, default_value_t = super::cond_arms::DEFAULT_NULL_SEED)]
    pub seed: u64,

    /// Conditional payload rates to price in the kill table, in bits per weight.
    #[arg(long, num_args = 1.., default_values_t = [3.0f64, 2.5, 2.0, 1.5, 1.0, 0.75])]
    pub payload_grid: Vec<f64>,

    /// Link throughput for the external-expert arm, GB/s.
    #[arg(long, default_value_t = 3.5)]
    pub link_gb_s: f64,
}

#[derive(Debug, Args)]
pub struct BudgetArgs {
    /// Link throughput per port, GB/s.
    #[arg(long, default_value_t = 3.5)]
    pub link_gb_s: f64,
    #[arg(long, default_value_t = 20.0)]
    pub target_tok_s: f64,
    #[arg(long, default_value_t = 1)]
    pub ports: u32,
    /// Feature fraction to price row-granular access at.
    #[arg(long, default_value_t = 0.064)]
    pub feature_fraction: f64,
    /// Measured routing-locality discount on the request rate.
    #[arg(long, default_value_t = 1.5)]
    pub locality: f64,
}

#[derive(Debug, Args)]
pub struct TouchArgs {
    #[arg(long, default_value_t = 20.0)]
    pub target_tok_s: f64,
    /// Demo-tier slice sizes to compose, in GB.
    #[arg(long, value_delimiter = ',', default_value = "55,60,65")]
    pub slice_gb: Vec<f64>,
}

#[derive(Debug, Args)]
pub struct FrontierArgs {
    #[arg(long, value_delimiter = ',', default_value = "6.6,8,10,12,14,20")]
    pub targets: Vec<f64>,
    /// Routed bytes still read. Defaults to the banked up-fold ceiling (2/3);
    /// anything lower is row sparsity, which is not yet measured.
    #[arg(long)]
    pub routed_retention: Option<f64>,
    /// Also run the R4 zero-out check: what the dense half alone permits.
    #[arg(long, default_value_t = true)]
    pub zero_out_check: bool,
}

#[derive(Debug, Args)]
pub struct BlockArgs {
    /// Positions the target evaluates per pass.
    #[arg(long, default_value_t = 7)]
    pub proposal_width: u32,
    /// Positions that commit, at that width. Must differ from the width unless
    /// perfect acceptance is genuinely intended (R5).
    #[arg(long, default_value_t = 3.85)]
    pub mean_accepted: f64,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "1,2,3,4,5,6,7,8,9,10,11,12"
    )]
    pub widths: Vec<u32>,
    #[arg(long)]
    pub routed_retention: Option<f64>,
    #[arg(long, default_value_t = 20.0)]
    pub target_tok_s: f64,
    /// KDA state and transaction bytes per pass. Owner M1-B; zero means
    /// unmeasured, and the record says so rather than omitting the term.
    #[arg(long, default_value_t = 0.0)]
    pub state_bytes_per_pass: f64,
    /// KDA state bytes per *proposed* position (prefix-state retention). This is
    /// the term that grows with width while acceptance saturates, so omitting it
    /// inverts the sign of state traffic's effect on the optimum.
    #[arg(long, default_value_t = 0.0)]
    pub state_bytes_per_position: f64,
    /// Derive per-position state traffic from K3's prefix-state ladder: one
    /// [96,128,128] KDA recurrent state per proposed position, at this dtype
    /// width in bytes (2 = bf16, 4 = fp32). Overrides --state-bytes-per-position.
    #[arg(long)]
    pub state_prefix_ladder_bytes: Option<u64>,
    /// Model execution as a naive token loop instead of assuming ideal grouping.
    #[arg(long)]
    pub token_loop: bool,
}

#[derive(Debug, Args)]
pub struct TranscodeScanArgs {
    /// How many expert `weight_scale` tensors to sample from the shard.
    #[arg(long, default_value_t = 24)]
    pub tensors: usize,
}

#[derive(Debug, Args)]
pub struct CeilingsArgs {
    /// Serving format for the dense path, in all-in bits per weight.
    #[arg(long, default_value_t = 4.25)]
    pub dense_bits: f64,
    /// Serving format for the routed experts, in all-in bits per weight.
    #[arg(long, default_value_t = 4.25)]
    pub routed_bits: f64,
}

#[derive(Debug, Args)]
pub struct HomogeneityArgs {
    /// Shard-name template; `{}` is the 1-based shard number, zero-padded.
    #[arg(long, default_value = "model-{:05}-of-000096.safetensors")]
    pub shard_template: String,
    /// Stop after this many layers. 0 means every layer.
    #[arg(long, default_value_t = 0)]
    pub limit: usize,
}

#[derive(Debug, Args)]
pub struct ActionsArgs {
    /// Show only actions at or above this activated saving, in GB/token.
    #[arg(long, default_value_t = 0.5)]
    pub min_gb: f64,
    /// Include family-wide CEILING rows alongside per-layer candidates.
    #[arg(long, default_value_t = true)]
    pub ceilings: bool,
}
