//! Arguments for every `larql vindex3` verb.

use clap::{Args, Subcommand};
use std::path::PathBuf;

#[allow(unused_imports)]
use super::*;

#[derive(Subcommand)]
pub enum Vindex3Command {
    /// Semantic representability plan over HF checkpoint dirs and/or saved
    /// `inspect-hf` inventory JSONs, treated together as one model system.
    /// Exits non-zero when the plan is inadmissible.
    Plan(PlanArgs),
    /// Encode the system into a self-contained container (G3). Refuses an
    /// inadmissible plan; consumes the built graph, never re-interprets
    /// the checkpoint.
    Encode(EncodeArgs),
    /// Reconstruct and check a container solely from its own contents —
    /// no source checkpoint, no architecture registry (the G3 gate).
    Inspect(InspectArgs),
    /// The dependencies a container declares between its own tensors —
    /// today, the `scales` grid of every fine-grained FP8 weight. With
    /// `--declare`, derive them from the segment headers and write the
    /// table, for a container encoded before the encoder declared them;
    /// the rule is the encoder's own, applied late.
    References(ReferencesArgs),
    /// Prove source ≡ encoded (the G4 gate): four-authority semantic
    /// comparison plus per-representation byte equivalence, both ends
    /// re-hashed now. Exits non-zero on any disagreement.
    Verify(VerifyArgs),
    /// Emit the generic operation plan of one component, solely from the
    /// container (G5b-1). Operand closure is the gate: every stack tensor
    /// must classify into a role a declared op consumes, with the
    /// geometry the surface states. Exits non-zero on any closure defect.
    Ops(OpsArgs),
    /// Execute one component's own program from the container alone
    /// (G5b-3c), optionally dumping per-layer hidden states in the
    /// `shannon layer-dump` format so `layer-diff` can compare it
    /// against an upstream trace with no new comparator.
    Exec(ExecArgs),
    /// Compile a physical representation of the container's objects and
    /// persist it beside the canonical bytes, so execution reads a
    /// compiled pack instead of quantising every operand at load.
    ///
    /// The canonical representation is never replaced: the pack is added
    /// and marked approximate, and a profile then selects between
    /// representations that exist.
    Represent(RepresentArgs),
    /// Observe a prompt through the canonical decode session and write a
    /// lossless run record (V3-OBS-1 + V3-STREAM-1): carrier writes with
    /// norms and a fixed projection, provenance, a verified receipt, and
    /// the final position's top candidates on stdout.
    Observe(observe::ObserveArgs),
    /// SENSITIVITY-1A: score every eligible tensor by the relative error
    /// quantising it introduces, from the weights alone and with no forward
    /// pass. One screen scores every candidate precision map.
    Sensitivity(sensitivity::SensitivityArgs),

    /// SENSITIVITY-1B': per-tensor activation-weighted consequence from a
    /// frozen capture. Emits numbers only — aggregation and the bar live in
    /// `bench/prompts/quality-bank-1/`.
    Consequence(consequence::ConsequenceArgs),

    /// MEASURE-PLAN-1's sealed corpus: `export` tokenises a prompt file with
    /// a container's tokenizer into a bank; `import` seals ids another
    /// harness already tokenised; `check` reads every sample against its
    /// seal and the container's tokenizer.
    TokenBank(token_bank::TokenBankArgs),

    /// MEASURE-PLAN-1: teacher-force a candidate realization against a
    /// reference over a token bank, with every validity proof, and write a
    /// report, per-position records and a receipt.
    Measure(measure::MeasureArgs),
    /// AUTO-REP over a plan-v1 record: `init` produces a characterisation-only
    /// record, `run` advances one through the propose, compile, measure, cut loop.
    AutoRep(auto_rep::AutoRepArgs),
}

/// Which numerical realisation runs the plan. Both execute the *same*
/// program through the same interpreter; only the arithmetic differs.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum ExecBackend {
    /// Naive f32, sharing no arithmetic with `larql-compute`.
    Reference,
    /// The `larql-compute` kernels.
    Production,
    /// The `larql-compute` kernels, asking for a compiled NVFP4 pack.
    ///
    /// The CPU sibling of the `metal-nvfp4*` arms, and the only way to
    /// execute an NVFP4 representation of a model the device cannot run.
    /// Qwen3.8 is the case that forced it: 48 Gated DeltaNet layers with
    /// no Metal kernel, so before this arm its NVFP4 pack could be
    /// compiled and verified and then executed nowhere, and its
    /// behavioural fidelity was unmeasurable in principle.
    ///
    /// Separate from [`Self::Production`] rather than a flag on it: the
    /// backend is what declares which representation execution wants, and
    /// silently changing that for every existing `production` run would
    /// reinterpret every result already taken with it.
    ProductionNvfp4,
    /// The `larql-compute` kernels, asking for a compiled Q8_0 pack.
    ///
    /// These three exist for the same reason `ProductionNvfp4` does: the
    /// BACKEND declares which representation execution wants, so a
    /// container's compiled K-quant pack is bound instead of its
    /// canonical bytes. One arm per encoding rather than one arm plus a
    /// flag, because `wanted_representation` is exhaustive on purpose —
    /// a new arm is a compile error until someone states what it runs.
    ProductionQ8,
    /// The `larql-compute` kernels, asking for a compiled Q6_K pack.
    ProductionQ6k,
    /// The `larql-compute` kernels, asking for a compiled Q4_K pack.
    ProductionQ4k,
    /// The same compiled Q4_K pack, executed against a **Q8_K
    /// activation** by the integer-dot kernel V2's CPU decode uses
    /// (Q8K-ACT-1, `docs/q8k-act-1.md`). Lossy in the activation by
    /// declaration; [`Self::ProductionQ4k`] keeps its f32-activation
    /// meaning.
    ProductionQ4kQ8k,
    /// The same compiled NVFP4 pack as [`Self::ProductionNvfp4`], executed
    /// against a **Q8 activation** with integer dot products (NVFP4-Q8-1,
    /// `docs/nvfp4-q8-1.md`). Lossy in the activation by declaration;
    /// [`Self::ProductionNvfp4`] keeps its f32-activation meaning.
    ProductionNvfp4Q8,
    /// GPU matmuls via `larql-compute-metal` (rung 1: matrix work on
    /// the device, elementwise glue on the CPU).
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    Metal,
    /// Metal with every matrix operand quantised to MXFP4 at load —
    /// the compressed-execution realisation (VINDEX3-Q1). Lossy;
    /// judged by the parity gates, not assumed.
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    MetalMxfp4,
    /// Every matrix operand MXFP4 — the preset Q1's 6-token gate
    /// *falsified*. Kept as the control arm: a Q2 result showing NVFP4
    /// holds the prediction means nothing unless the same harness is
    /// shown to break on the format Q1 broke on.
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    MetalMxfp4All,
    /// VINDEX3-Q2 arm A: every matrix operand NVFP4 — e2m1 elements,
    /// 16-element groups, E4M3 scales. 4.5 bpw everywhere.
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    MetalNvfp4,
    /// Q2 arm B: attention and FFN NVFP4, head f16.
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    MetalNvfp4NoHead,
    /// Q2 arm C: FFN NVFP4 only — Q1's passing partition under the new
    /// scale geometry, so the formats are compared at one class split.
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    MetalNvfp4Ffn,
    /// VINDEX3-G6d: the plan lowered onto GPU-resident execution — the
    /// whole stack and head in one command buffer per token, KV resident,
    /// host out of the dependency chain. All-NVFP4.
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    MetalLowered,
    /// Lowered execution, NVFP4 FFN with attention and head f16 — the
    /// quality end of the Q2 frontier, under identical scheduling.
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    MetalLoweredFfn,
    /// Lowered execution, NVFP4 attention and FFN with an f16 head.
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    MetalLoweredNoHead,
    /// Lowered execution, all-MXFP4 — the format bakeoff arm, so the two
    /// representations are priced under one schedule.
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    MetalLoweredMxfp4,
    /// Lowered execution, MXFP4 FFN (dense AND expert banks) with f16
    /// attention and head — the same representation as the interpreter's
    /// `metal-mxfp4` arm, so the two certify each other at f32 noise.
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    MetalLoweredMxfp4Ffn,
    /// Lowered execution, f16 everywhere the lowering can hold f16 —
    /// attention, dense FFN, head. Expert banks go through the descriptor
    /// MoE path, which serves Q6_K and MXFP4 only, so a bf16 bank is
    /// MXFP4 here: this arm isolates the expert representation's cost.
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    MetalLoweredF16,
}

#[derive(Args)]
pub struct ExecArgs {
    /// Container directory.
    pub container: PathBuf,

    /// Component to execute.
    #[arg(long, default_value = "target")]
    pub component: String,

    /// Comma-separated token ids. Given, never tokenised here: a
    /// tokenizer is part of the fixture and only one side may choose it.
    #[arg(long)]
    pub tokens: String,

    /// Write per-layer planes + manifest here instead of a summary.
    /// Planes are written as each layer completes, so an interrupted
    /// run leaves everything it finished.
    #[arg(long)]
    pub dump_layers: Option<PathBuf>,

    /// Continue an interrupted `--dump-layers` run from its last
    /// complete plane. The dump's recorded fixture (tokens, container,
    /// engine) must match, or the resume refuses rather than splice
    /// two different runs.
    #[arg(long, requires = "dump_layers")]
    pub resume: bool,

    /// Numerical realisation to run the plan on.
    #[arg(long, value_enum, default_value_t = ExecBackend::Reference)]
    pub backend: ExecBackend,

    #[command(flatten)]
    pub plugin: plugins::PluginArgs,

    /// Where an execution representation may come from.
    ///
    /// Separate from `--backend` on purpose: the backend says *what*
    /// representation execution wants, this says whether the runtime may
    /// manufacture it now.
    ///
    /// `auto` uses a compiled pack when present and quantises at load
    /// otherwise. `stored` forbids manufacturing — the run fails naming any
    /// tensor that would be quantised, so "no runtime quantisation" is an
    /// invariant rather than a timing to infer. `transient` ignores any
    /// pack and quantises at load; it is the oracle the representation
    /// compiler is checked against, and is retained permanently.
    #[arg(long, value_name = "auto|stored|transient", default_value = "auto")]
    pub representation_source: String,

    /// Greedy-decode this many new tokens after the prompt, printing
    /// per-step timing and a decode report instead of a single-forward
    /// summary. Runs on a `DecodeSession`: operands are loaded once and
    /// each token advances one position against the session's
    /// continuation state (KV cache, or recurrent state), so the report
    /// prices weight load, prompt ingestion and steady decode separately.
    #[arg(long, conflicts_with_all = ["dump_layers", "resume"])]
    pub generate: Option<usize>,

    /// With `--generate` on a `metal-lowered*` backend: prompt-lookup
    /// speculative decoding. When the context's trailing n-gram (n >= 2)
    /// occurred earlier, the `N-1` tokens that followed it are verified
    /// together with the last token in one N-position forward, and the
    /// longest agreeing prefix is kept. Output ids are the greedy ids
    /// exactly; only the schedule changes. No match → an ordinary step.
    #[arg(long, requires = "generate", value_parser = clap::value_parser!(u16).range(2..=8))]
    pub speculate: Option<u16>,

    /// With `--generate`: stop at the container's EOS ids
    /// (`generation_config.json`), in both the greedy and `--speculate`
    /// loops, instead of decoding the full count past end-of-turn.
    #[arg(long, requires = "generate")]
    pub stop_at_eos: bool,

    /// With `--generate`: observe the prepared image's residency BETWEEN
    /// tokens — mapped address space against the pages of it physically
    /// resident, page faults, peak RSS, and where each token's time went
    /// (attention against FFN) — beside what the plan predicted before a
    /// byte was read. The residency curve of the K3 vertical's rung 7.
    #[arg(long, requires = "generate")]
    pub residency_curve: bool,

    /// With `--residency-curve`: run the same prompt and decode this many
    /// times on ONE prepared image, each pass with fresh continuation
    /// state. The first pass is cold; every later pass is what the page
    /// cache kept — the warm number the cold one is compared against.
    #[arg(long, default_value_t = 1, requires = "residency_curve")]
    pub repeat: usize,
    /// Passes excluded from the residency curve's statistics (they still
    /// run and print); the counted passes are `repeat - warmup`.
    #[arg(long, default_value_t = 0, requires = "residency_curve")]
    pub warmup: usize,
    /// Run the residency curve even when the machine probe disqualifies
    /// the machine (battery, load, background CPU); the report is then
    /// labelled UNQUALIFIED and says why. Without it, a disqualified
    /// machine is refused before any weight is bound.
    #[arg(long, requires = "residency_curve")]
    pub unquiet_ok: bool,
    /// With `--residency-curve`: write every mapped region's residency to
    /// this file as JSON Lines (`larql.residency-observation/1`) — one
    /// observation at binding, before any pass runs, and one after the
    /// run. A region whose residency could not be read is recorded as
    /// failed, with the reason, never as zero.
    #[arg(long, requires = "residency_curve")]
    pub residency_regions: Option<PathBuf>,
    /// How a mapped expert bank's selected experts are brought in per
    /// token: `demand` (the loop faults each page), `advise` (the kernel
    /// is told ahead), `touch` (pages are faulted concurrently ahead of
    /// the loop). The same lossless bytes under every policy.
    #[arg(long, default_value = "demand", requires = "residency_curve")]
    pub expert_access: String,

    /// Teacher-force a whole quality bank through ONE resident model,
    /// writing `<--dump-dir>/<id>.f32` per entry.
    ///
    /// The file is JSON lines: `{"id": "...", "ids": [1,2,3]}`. Each entry
    /// gets a brand-new continuation state, and the run fails if a session
    /// does not start at position 0 or does not end at the entry's length
    /// — a leak between entries would silently score later prompts against
    /// a context no reference ever saw.
    #[arg(long, value_name = "JSONL")]
    pub bank: Option<PathBuf>,

    /// Where `--bank` writes its per-entry logit dumps.
    #[arg(long, value_name = "DIR")]
    pub dump_dir: Option<PathBuf>,

    /// Step the given tokens through the plan one position at a time and
    /// write every position's logits here as `[positions, vocab]` f32.
    ///
    /// Teacher forcing, so two realisations scored this way see identical
    /// context at every position and a per-position divergence is
    /// attributable to the representation rather than to the two arms
    /// having generated different text. This is what a KL/NLL gate needs;
    /// `--generate` cannot supply it.
    #[arg(long, conflicts_with_all = ["dump_layers", "resume", "generate"])]
    pub logit_dump: Option<PathBuf>,

    /// Execute only the first N layers, then the component's own final
    /// norm and output head — a reduced-depth *model*, not a shard.
    ///
    /// One semantic effect: `ExecutionSlice::Draft { end: N }`. Nothing
    /// else about the run changes, which is the point — a draft measured
    /// through a different code path than its target would confound
    /// depth with harness.
    ///
    /// Omitted, or equal to the layer count, is the whole stack and is
    /// bit-identical to leaving the flag off.
    #[arg(long)]
    pub draft_depth: Option<usize>,

    /// Lowered backends only: attribute each decode token's GPU time to
    /// its stage classes (stage-boundary timestamp counters) and print
    /// the ledger against the bytes each class reads. Sampling drains the
    /// pipeline at every stage boundary, so judge throughput from an
    /// unprofiled run and attribution from this one.
    #[arg(long, requires = "generate")]
    pub profile: bool,
}

#[derive(Args)]
pub struct OpsArgs {
    /// Container directory.
    pub container: PathBuf,

    /// Component to plan.
    #[arg(long, default_value = "target")]
    pub component: String,

    /// Print one layer's full program instead of the per-layer summary.
    #[arg(long)]
    pub layer: Option<usize>,

    /// Print the full plan as JSON instead of the summary.
    #[arg(long)]
    pub json: bool,

    /// Prepare the plan against the production CPU backend WITHOUT
    /// reading a payload byte: select and pin a realization per planned
    /// operand, or print every refusal; then price the pins — declared
    /// resident bytes, staging, stored footprint, execution touch — from
    /// the container's tensor tables alone.
    #[arg(long)]
    pub realizations: bool,

    /// A memory budget in GiB to hold the declared working set against
    /// (with `--realizations`). Omitted: the machine's physical memory.
    #[arg(long)]
    pub budget_gib: Option<f64>,

    /// With `--realizations`, and only when the plan fits the budget:
    /// PREPARE the plan — bind every pin to its object — and reconcile
    /// what the loader bound against what the pins declared, reporting
    /// what was mapped, what is physically resident, and what was read.
    #[arg(long)]
    pub bind: bool,

    /// Host bandwidth in GB/s the plan may stream per second (with
    /// `--target-tok-s`, a per-token touch budget). Omitted: no
    /// throughput constraint.
    #[arg(long)]
    pub bandwidth_gbs: Option<f64>,

    /// The token rate the throughput budget is held at.
    #[arg(long, default_value_t = 20.0)]
    pub target_tok_s: f64,
}

#[derive(Args)]
pub struct VerifyArgs {
    /// Checkpoint directories or inventory JSON files — the same artifact
    /// set the container was encoded from.
    #[arg(required = true)]
    pub artifacts: Vec<PathBuf>,

    /// Container directory to verify against.
    #[arg(long)]
    pub container: PathBuf,

    /// Print the full report as JSON instead of the summary.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args)]
pub struct EncodeArgs {
    /// Checkpoint directories, inventory JSON files, or `hf://` repos
    /// (one per artifact).
    ///
    /// An `hf://org/name[@revision]` artifact is admitted from its
    /// safetensors headers alone — a few MB staged locally, standing in
    /// for a checkpoint that is never downloaded.
    #[arg(required = true)]
    pub artifacts: Vec<PathBuf>,

    /// Container directory to write.
    #[arg(long)]
    pub output: PathBuf,

    /// Gate admission on ONE capability's execution closure instead of
    /// whole-model completeness.
    ///
    /// Without this, encode requires every declared execution-semantic
    /// fact in the checkpoint to be understood — the right bar for
    /// "we understand this model", and too strong a bar for "we can run
    /// text generation on it". Qwen3.8-27B is admissible for text with
    /// 16 whole-model findings outstanding, none of them reachable from
    /// a text forward pass.
    ///
    /// The container written is identical either way; only the gate
    /// changes.
    #[arg(long, value_enum)]
    pub capability: Option<EncodeCapability>,
}

/// Capabilities `--capability` accepts. A subset of
/// [`larql_vindex::format::vindex3::plan::capability::Capability`]: only
/// those this build can execute are offerable, because encoding for a
/// capability with no executor would be a promise the runtime cannot
/// keep.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum EncodeCapability {
    /// Text in, text out.
    TextGeneration,
}

impl From<EncodeCapability> for larql_vindex::format::vindex3::plan::capability::Capability {
    fn from(value: EncodeCapability) -> Self {
        match value {
            EncodeCapability::TextGeneration => Self::TextGeneration,
        }
    }
}

#[derive(Args)]
pub struct RepresentArgs {
    /// Container directory to compile from.
    pub container: PathBuf,

    /// Container directory to write. The canonical segments are
    /// hard-linked where the filesystem allows, so the new container costs
    /// the compiled pack's bytes rather than the whole model's.
    #[arg(long)]
    pub output: PathBuf,

    /// Target encoding: `NVFP4`, a K-quant, or the label of an encoder a
    /// `--plugin` registered.
    #[arg(long, default_value = "NVFP4")]
    pub encoding: String,

    /// Load a larql plugin whose encoders `--encoding` may name, as
    /// `vindex3 exec --plugin` loads codecs. Repeatable.
    #[arg(long = "plugin", value_name = "PATH")]
    pub plugins: Vec<PathBuf>,

    /// Encode under input-feature weights: `E[x^2]` per input feature from a
    /// `sensitivity --calibration` capture of this container (the mapping
    /// `consequence` uses; `down_proj` reconstructed and gated, `o_proj`
    /// unweighted for want of a site). The encoder must accept weights
    /// (plugin encoders may; shipped compilers refuse), and the pack's
    /// recipe records the capture's digest.
    #[arg(long, value_name = "JSON")]
    pub moments: Option<PathBuf>,

    /// Objects to compile. Repeat the flag to name several; omit to
    /// compile every object carrying an eligible tensor.
    #[arg(long = "object")]
    pub objects: Vec<String>,

    /// Compile a role the conservative default preserves. Repeat to
    /// name several.
    ///
    /// The default compiles the parameter mass — the decoder's and a
    /// recurrence's bulk projections, and routed experts — and
    /// preserves the surfaces where 4-bit is known to be delicate or
    /// where error compounds. This flag is how a profile becomes more
    /// aggressive deliberately rather than by accident.
    ///
    /// The role names are not listed here on purpose: this comment
    /// enumerated them once, and was silently wrong the moment a role
    /// was added. Pass any name to be refused by one that names the
    /// current set.
    #[arg(long = "include-role")]
    pub include_roles: Vec<String>,

    /// Write a deployment image instead of an archival container.
    ///
    /// The image carries the compiled representation plus every surface the
    /// precision policy protected — the BF16 embedding and norms have to
    /// travel or it will not execute — and drops the source bytes it
    /// replaced. It names the digests it derives from, so the authority can
    /// be found again; it cannot recompile itself.
    ///
    /// Nothing is destroyed: the container this was compiled from is
    /// untouched.
    #[arg(long)]
    pub deployment: bool,

    /// Hold a projection at source precision despite its role being
    /// eligible, e.g. `--protect v_proj`. Repeat to name several.
    ///
    /// Append `@LO-HI` to protect it only within a depth range —
    /// `--protect gate_proj@30-39`. That intersection is a different
    /// policy from `--protect gate_proj --protect-layers 30-39`, which is
    /// their union and protects far more.
    ///
    /// This is how a precision map is expressed: role eligibility says the
    /// encoding applies to a kind of weight, and this says which of them to
    /// actually spend it on.
    #[arg(long = "protect")]
    pub protect: Vec<String>,

    /// Hold an inclusive range of layer depths at source precision, e.g.
    /// `--protect-layers 0-7`. Repeat to name several.
    #[arg(long = "protect-layers", value_name = "LO-HI")]
    pub protect_layers: Vec<String>,
}

#[derive(Args)]
pub struct InspectArgs {
    /// Container directory.
    pub container: PathBuf,

    /// Additionally re-hash every segment against the directory.
    #[arg(long)]
    pub verify: bool,

    /// Additionally require execution completeness (the G5a gate): every
    /// component with executable objects must carry the surface those
    /// operations read. Exits non-zero when incomplete.
    #[arg(long)]
    pub execution_complete: bool,

    /// Print the full reconstruction as JSON instead of the summary.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args)]
pub struct ReferencesArgs {
    /// Container directory.
    pub container: PathBuf,

    /// Derive the table from the segment headers and write it. Refuses a
    /// container that already declares one.
    #[arg(long)]
    pub declare: bool,
}

#[derive(Args)]
pub struct PlanArgs {
    /// Checkpoint directories, inventory JSON files, or `hf://` repos
    /// (one per artifact).
    ///
    /// Planning an `hf://` repo costs its safetensors headers and
    /// nothing else — the admission verdict for a 328 GB checkpoint,
    /// before deciding whether to spend the download.
    #[arg(required = true)]
    pub artifacts: Vec<PathBuf>,

    /// Write the plan JSON here instead of stdout.
    #[arg(long)]
    pub output: Option<PathBuf>,
}
