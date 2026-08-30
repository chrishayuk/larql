//! `larql vindex3` — VINDEX3 container programme verbs.
//!
//! `plan` is the G1 gate: a semantic representability check over one or
//! more artifacts, run *before* any conversion. It prints the full
//! [`larql_vindex::format::vindex3::plan::SystemPlan`] as JSON and exits
//! non-zero when the plan is inadmissible, so scripts can gate on it.

use std::path::{Path, PathBuf};

use clap::{Args, Subcommand};

use larql_models::inventory::{build_inventory, ArchitectureInventory};
use larql_vindex::format::vindex3::plan::plan_system;

/// Extension distinguishing a saved inventory JSON from a checkpoint dir.
const INVENTORY_EXT: &str = "json";

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
    /// SENSITIVITY-1A: score every eligible tensor by the relative error
    /// quantising it introduces, from the weights alone and with no forward
    /// pass. One screen scores every candidate precision map.
    Sensitivity(sensitivity::SensitivityArgs),

    /// SENSITIVITY-1B': per-tensor activation-weighted consequence from a
    /// frozen capture. Emits numbers only — aggregation and the bar live in
    /// `bench/prompts/quality-bank-1/`.
    Consequence(consequence::ConsequenceArgs),
}

/// Which numerical realisation runs the plan. Both execute the *same*
/// program through the same interpreter; only the arithmetic differs.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum ExecBackend {
    /// Naive f32, sharing no arithmetic with `larql-compute`.
    Reference,
    /// The `larql-compute` kernels.
    Production,
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
    /// summary. Every step re-runs the full forward — the interpreter
    /// has no KV cache yet, and the report says so.
    #[arg(long, conflicts_with_all = ["dump_layers", "resume"])]
    pub generate: Option<usize>,

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
    /// Checkpoint directories or inventory JSON files (one per artifact).
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

    /// Target encoding. `NVFP4` is the only compiler today.
    #[arg(long, default_value = "NVFP4")]
    pub encoding: String,

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
pub struct PlanArgs {
    /// Checkpoint directories or inventory JSON files (one per artifact).
    #[arg(required = true)]
    pub artifacts: Vec<PathBuf>,

    /// Write the plan JSON here instead of stdout.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

pub fn run(cmd: Vindex3Command) -> Result<(), Box<dyn std::error::Error>> {
    match cmd {
        Vindex3Command::Plan(args) => run_plan(args),
        Vindex3Command::Encode(args) => run_encode(args),
        Vindex3Command::Inspect(args) => run_inspect(args),
        Vindex3Command::Verify(args) => run_verify(args),
        Vindex3Command::Ops(args) => run_ops(args),
        Vindex3Command::Exec(args) => run_exec(args),
        Vindex3Command::Represent(args) => run_represent(args),
        Vindex3Command::Sensitivity(args) => sensitivity::run(args),
        Vindex3Command::Consequence(args) => consequence::run(args),
    }
}

mod bank;
mod consequence;
mod exec;
mod generate;
#[cfg(all(feature = "gpu", target_os = "macos"))]
mod lowered;
mod ops;
mod optional_op;
mod sensitivity;
mod teacher_force;
use exec::run_exec;
use ops::run_ops;

fn run_verify(args: VerifyArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut named = Vec::new();
    for path in &args.artifacts {
        named.push((artifact_name(path), load_artifact(path)?));
    }
    let verification =
        larql_vindex::format::vindex3::verify_system::verify_system(&named, &args.container)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&verification)?);
    } else {
        for defect in &verification.container_defects {
            println!("container defect: {defect}");
        }
        let semantic_pass = verification.semantic.iter().filter(|c| c.pass).count();
        println!(
            "semantic: {semantic_pass}/{} authority checks pass",
            verification.semantic.len()
        );
        for failure in verification.semantic_failures() {
            println!(
                "  FAIL {} ≡ {}  {}: {}",
                failure.left, failure.right, failure.subject, failure.detail
            );
        }
        println!("payloads:");
        for check in &verification.payloads {
            println!(
                "  {} {:40} {:8.3} GB  {}",
                if check.pass { "PASS" } else { "FAIL" },
                check.representation,
                check.payload_bytes as f64 / 1e9,
                if check.pass {
                    format!(
                        "sha256 {}…",
                        &check.recorded_sha256[..12.min(check.recorded_sha256.len())]
                    )
                } else {
                    check.detail.clone()
                },
            );
        }
    }
    if verification.verified {
        eprintln!("verified: Declared ≡ Resolved ≡ Graph ≡ Encoded; payloads byte-equal");
        Ok(())
    } else {
        Err(format!(
            "NOT verified: {} semantic failure(s), {} payload failure(s), {} container defect(s)",
            verification.semantic_failures().count(),
            verification.payload_failures().count(),
            verification.container_defects.len(),
        )
        .into())
    }
}

fn run_encode(args: EncodeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut named = Vec::new();
    for path in &args.artifacts {
        named.push((artifact_name(path), load_artifact(path)?));
    }
    let outcome = match args.capability {
        Some(capability) => {
            eprintln!(
                "admission scoped to {:?}; whole-model completeness is NOT asserted",
                capability
            );
            larql_vindex::format::vindex3::encode::encode_system_for_capability(
                &named,
                &args.output,
                capability.into(),
            )?
        }
        None => larql_vindex::format::vindex3::encode::encode_system(&named, &args.output)?,
    };
    // Capability snapshot: tokenizer + HF metadata from the first
    // artifact directory that carries them (the inventory records its
    // source dir, so this covers both checkpoint-dir and saved-inventory
    // inputs). A container without them binds with token-id capability
    // only — which is why the granite smoke needed a manual copy before
    // this existed.
    for (_, inventory) in &named {
        let copied =
            larql_vindex::format::vindex3::encode::checkpoint::snapshot_checkpoint_capabilities(
                std::path::Path::new(&inventory.path),
                &args.output,
            )?;
        if !copied.is_empty() {
            eprintln!("capabilities: {}", copied.join(", "));
            break;
        }
    }
    eprintln!(
        "encoded {} representation(s), {:.2} GB payload → {}",
        outcome.representations,
        outcome.total_payload_bytes as f64 / 1e9,
        outcome.container.display(),
    );
    Ok(())
}

/// `larql vindex3 represent` — compile a physical representation.
///
/// Prints what each object cost before and after, because the whole point
/// of the operation is a number: the pack is only worth persisting if it is
/// materially smaller than the bytes it was compiled from.
fn run_represent(args: RepresentArgs) -> Result<(), Box<dyn std::error::Error>> {
    use larql_vindex::format::vindex3::represent::{compile_representation, RepresentSpec};

    let mut roles = larql_vindex::format::vindex3::represent::policy::RolePolicy::default();
    for name in &args.include_roles {
        let role = larql_vindex::format::vindex3::represent::policy::Role::parse(name).ok_or_else(
            || {
                // Derived, never restated: a hand-written list is wrong
                // one commit after a role is added, and this message is
                // the only place the caller learns what is valid.
                let known: Vec<&str> = larql_vindex::format::vindex3::represent::policy::Role::ALL
                    .iter()
                    .map(|r| r.name())
                    .collect();
                format!(
                    "unknown role `{name}` — the roles are: {}",
                    known.join(", ")
                )
            },
        )?;
        roles = roles.including(role);
    }
    let mut protect = larql_vindex::format::vindex3::represent::policy::Protections::default();
    for p in &args.protect {
        protect = match p.split_once('@') {
            Some((name, range)) => {
                let (lo, hi) = range
                    .split_once('-')
                    .ok_or_else(|| format!("--protect {p}: expected PROJ@LO-HI"))?;
                protect.projection_in(
                    name,
                    lo.trim().parse::<u32>().map_err(|e| format!("{p}: {e}"))?,
                    hi.trim().parse::<u32>().map_err(|e| format!("{p}: {e}"))?,
                )
            }
            None => protect.projection(p),
        };
    }
    for r in &args.protect_layers {
        let (lo, hi) = r
            .split_once('-')
            .ok_or_else(|| format!("--protect-layers expects LO-HI, got `{r}`"))?;
        protect = protect.layers(
            lo.trim().parse::<u32>().map_err(|e| format!("{r}: {e}"))?,
            hi.trim().parse::<u32>().map_err(|e| format!("{r}: {e}"))?,
        );
    }
    if !protect.is_empty() {
        println!("  protect: {}", protect.describe());
    }
    let spec = RepresentSpec {
        encoding: args.encoding.clone(),
        objects: args.objects.clone(),
        roles,
        deployment: args.deployment,
        protect,
    };
    println!(
        "== represent {} ({}) ==",
        args.encoding,
        if args.deployment {
            "deployment image"
        } else {
            "archival container"
        }
    );
    println!("  in     : {}", args.container.display());
    println!("  out    : {}", args.output.display());
    if !args.objects.is_empty() {
        println!("  objects: {}", args.objects.join(", "));
    }

    let started = std::time::Instant::now();
    let report = compile_representation(&args.container, &args.output, &spec)?;

    println!("\n── compiled ──");
    println!(
        "  {:<34} {:>12} {:>12} {:>8} {:>9}",
        "object", "source", "compiled", "ratio", "tensors"
    );
    println!("  {}", "-".repeat(80));
    let mut src_total = 0u64;
    let mut out_total = 0u64;
    for c in &report.compiled_objects {
        src_total += c.source_bytes;
        out_total += c.compiled_bytes;
        println!(
            "  {:<34} {:>12} {:>12} {:>7.2}x {:>4} +{:<4}",
            c.object,
            human_bytes(c.source_bytes),
            human_bytes(c.compiled_bytes),
            c.compression(),
            c.compiled_tensors,
            c.carried_tensors,
        );
    }
    println!("  {}", "-".repeat(80));
    let ratio = if out_total == 0 {
        0.0
    } else {
        src_total as f64 / out_total as f64
    };
    println!(
        "  {:<34} {:>12} {:>12} {:>7.2}x",
        "TOTAL",
        human_bytes(src_total),
        human_bytes(out_total),
        ratio
    );
    let mut protected: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for c in &report.compiled_objects {
        for (role, n) in &c.preserved {
            *protected.entry(role.to_string()).or_insert(0) += n;
        }
    }
    for po in &report.preserved_objects {
        let roles: Vec<String> = po.roles.iter().map(|(r, n)| format!("{r} x{n}")).collect();
        println!(
            "  {:<34} {:>12} {:>12}   preserved whole [{}]",
            po.object,
            human_bytes(po.bytes),
            po.encoding,
            roles.join(", ")
        );
    }
    if !protected.is_empty() {
        println!("\n  preserved at source precision (conservative default):");
        for (role, n) in &protected {
            println!("    {role:<16} {n} tensor(s)");
        }
    }
    println!(
        "\n  {} segment(s) carried unchanged; canonical bytes are untouched.",
        report.linked_segments
    );
    println!("  wall time: {:.1}s", started.elapsed().as_secs_f64());
    println!("\n→ {}", args.output.display());
    Ok(())
}

fn human_bytes(bytes: u64) -> String {
    const K: u64 = 1024;
    const M: u64 = K * 1024;
    const G: u64 = M * 1024;
    if bytes >= G {
        format!("{:.2} GB", bytes as f64 / G as f64)
    } else if bytes >= M {
        format!("{:.1} MB", bytes as f64 / M as f64)
    } else if bytes >= K {
        format!("{:.1} KB", bytes as f64 / K as f64)
    } else {
        format!("{bytes} B")
    }
}

fn run_inspect(args: InspectArgs) -> Result<(), Box<dyn std::error::Error>> {
    let inspection =
        larql_vindex::format::vindex3::inspect::inspect_container(&args.container, args.verify)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&inspection)?);
    } else {
        println!("components:");
        for c in &inspection.components {
            let policy = match (c.sliding_layers, c.full_layers, c.nope_layers) {
                (Some(s), Some(f), Some(n)) => {
                    format!(
                        ", {s} sliding / {f} full{}, {n} NoPE, window {:?}",
                        match c.recurrent_layers {
                            Some(r) if r > 0 => format!(" / {r} gated-delta recurrent"),
                            _ => String::new(),
                        },
                        c.window
                    )
                }
                _ => ", (no per-layer table)".to_string(),
            };
            println!(
                "  {:8} {:12} {} layers, hidden {}{policy}",
                c.id, c.role, c.num_layers, c.hidden_size
            );
        }
        println!("objects:");
        for (id, entry) in &inspection.index.representations {
            println!(
                "  {:40} {:8.3} GB  {} tensors  sha256 {}…",
                id,
                entry.payload_bytes as f64 / 1e9,
                entry.tensor_count,
                &entry.payload_sha256[..12],
            );
        }
        println!("edges:");
        for e in &inspection.graph.edges {
            println!(
                "  {}.hidden{:?} -> {} via {} (block {:?})",
                e.producer_component,
                e.producer_layers,
                e.consumer_component,
                e.consumer_object,
                e.block_size,
            );
        }
    }
    let surface_defects = if args.execution_complete {
        let defects = inspection.execution_completeness();
        if !args.json {
            println!("execution surfaces:");
            for component in &inspection.graph.components {
                match &component.execution {
                    Some(surface) => {
                        // Presence follows the program (schema 6): each
                        // group prints only when the component runs it,
                        // and absence is said in words — a pure-SSM
                        // component has no attention/FFN to describe.
                        let attention = match &surface.attention {
                            Some(a) => format!(
                                "attention {}q/{}kv head {} q-scale {:.4} s-scale {:.4}{}",
                                a.num_q_heads,
                                a.num_kv_heads,
                                a.head_dim,
                                optional_op::scalar(a.query_scale),
                                a.score_scale,
                                if a.output_gate.is_some() {
                                    " gated"
                                } else {
                                    ""
                                },
                            ),
                            None => "attention absent".to_string(),
                        };
                        let ffn = match &surface.ffn {
                            Some(f) => format!(
                                "ffn {:?} {:?} {}",
                                f.activation, f.ffn_type, f.intermediate_size
                            ),
                            None => "ffn absent".to_string(),
                        };
                        let mixer = match &surface.mamba2 {
                            Some(m) => format!(
                                ", mamba2 mixer {}h×{}×{} conv {}",
                                m.geometry.num_heads,
                                m.geometry.head_dim,
                                m.geometry.state_size,
                                m.geometry.conv_kernel
                            ),
                            None => String::new(),
                        };
                        println!(
                            "  {:8} {attention}, {ffn}{mixer}, norm {:?} eps {:e}{}",
                            component.id,
                            surface.norm.pre.kind,
                            surface.norm.pre.eps,
                            match &surface.head {
                                Some(head) => format!(", head vocab {}", head.vocab_size),
                                None => String::new(),
                            },
                        )
                    }
                    None => println!("  {:8} (no execution surface)", component.id),
                }
            }
            for defect in &defects {
                println!("  defect: {defect}");
            }
            println!(
                "executable: {}",
                if defects.is_empty() { "yes" } else { "NO" }
            );
        }
        defects
    } else {
        Vec::new()
    };

    if !surface_defects.is_empty() {
        return Err(format!(
            "container not executable: {} execution-completeness defect(s)",
            surface_defects.len()
        )
        .into());
    }
    if inspection.is_coherent() {
        eprintln!(
            "container coherent{}",
            if args.verify {
                " (payloads verified)"
            } else {
                ""
            }
        );
        Ok(())
    } else {
        for defect in &inspection.defects {
            eprintln!("defect: {defect:?}");
        }
        Err(format!(
            "container incoherent: {} defect(s)",
            inspection.defects.len()
        )
        .into())
    }
}

/// Artifact display name: the file/directory stem.
fn artifact_name(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Load one artifact: a `.json` file deserialises as a saved inventory;
/// anything else is inspected as a checkpoint directory.
fn load_artifact(path: &Path) -> Result<ArchitectureInventory, Box<dyn std::error::Error>> {
    if path.extension().is_some_and(|ext| ext == INVENTORY_EXT) {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    } else {
        Ok(build_inventory(path)?)
    }
}

fn run_plan(args: PlanArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut named = Vec::new();
    for path in &args.artifacts {
        named.push((artifact_name(path), load_artifact(path)?));
    }
    let plan = plan_system(&named);
    let json = serde_json::to_string_pretty(&plan)?;
    match &args.output {
        Some(path) => {
            std::fs::write(path, &json)?;
            eprintln!("plan written to {}", path.display());
        }
        None => println!("{json}"),
    }
    let summary = &plan.summary;
    eprintln!(
        "plan: {} representable, {} mismatched, {} unrepresented, {} interfaces — {} blocking",
        summary.representable,
        summary.mismatched,
        summary.unrepresented,
        summary.interfaces,
        summary.blocking,
    );
    if plan.admissible {
        Ok(())
    } else {
        Err(format!(
            "plan not admissible: {} blocking finding(s); \
             every one is a schema or resolution gap to close before conversion",
            summary.blocking
        )
        .into())
    }
}

#[cfg(test)]
mod tests;
