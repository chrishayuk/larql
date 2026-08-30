//! `larql vindex3 exec` — run a container's own program (V3-G5b-3c).
//!
//! Research-oriented on purpose. The first useful mode is not chat: it is
//! a layer-by-layer hidden-state dump in exactly the format
//! `larql shannon layer-dump` writes, so `larql shannon layer-diff`
//! compares a VINDEX3 execution against an upstream `transformers` trace
//! with **no new comparator**. A divergence localises to a layer before
//! anyone asks what the model said.
//!
//! Token ids are given explicitly rather than tokenised here. A tokenizer
//! is part of the fixture, and only one side of a parity comparison may
//! choose it — `scripts/capture_glimmer_oracle.py` already recorded the
//! ids this reads back.
//!
//! The backend is a flag over the same plan. That is the point of the
//! seam: `--backend reference` and `--backend production` execute one
//! program through two numerical realisations, and their dumps are
//! directly diffable against each other as well as against upstream.
//!
//! Dumped runs are resumable. Each plane is written the moment its layer
//! completes, and plane `k` is exactly the residual entering layer `k`,
//! so the dump directory *is* the checkpoint — `--resume` reloads the
//! last complete plane and continues bit-identically. The manifest is
//! written only at the end and therefore doubles as the completion
//! marker; a directory without one is an interrupted run.

use std::path::{Path, PathBuf};
use std::time::Instant;

use larql_vindex::error::VindexError;
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::operands::{OperandStore, RepresentationSource};
use larql_vindex::format::vindex3::opplan::exec::prepared::ExecutionSlice;
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use larql_vindex::format::vindex3::opplan::exec::{
    execute_plan_streaming, execute_slice, ExecutionTrace, PlaneEvent, ResumePoint,
};
use larql_vindex::format::vindex3::opplan::plan_component_ops;
use larql_vindex::format::vindex3::opplan::ComponentOpPlan;
use ndarray::Array2;

use super::super::shannon_trace::dump::{
    plane_name, write_plane, LayerDumpManifest, MANIFEST_NAME, PLANE_DTYPE,
};
use super::{ExecArgs, ExecBackend};

/// Extra planes beyond the layer table, matching
/// `scripts/capture_glimmer_oracle.py`.
const FINAL_NORM_PLANE: &str = "final_norm.f32";
const LOGITS_PLANE: &str = "logits.f32";

/// Engine tag prefix; the backend name completes it so a dump can never
/// be mistaken for one produced by the other realisation.
const ENGINE_PREFIX: &str = "vindex3";

/// Sidecar recording what fixture an interrupted dump was running, so
/// `--resume` can refuse to splice two different runs. Written at start;
/// the manifest (written at completion) is deliberately a different file.
pub(super) const RESUME_NAME: &str = "exec_resume.json";

/// Raw plane files are little-endian f32, per `PLANE_DTYPE`.
const BYTES_PER_VALUE: usize = std::mem::size_of::<f32>();

/// Everything that must match for a resume to be the *same* run.
#[derive(serde::Serialize, serde::Deserialize, PartialEq)]
pub(super) struct ResumeSidecar {
    pub(super) engine: String,
    pub(super) container: String,
    pub(super) component: String,
    pub(super) token_ids: Vec<u32>,
}

pub fn run_exec(args: ExecArgs) -> Result<(), Box<dyn std::error::Error>> {
    let tokens = parse_tokens(&args.tokens)?;
    let inspection = inspect_container(&args.container, false)?;
    let outcome = plan_component_ops(&inspection, &args.container, &args.component)?;
    if !outcome.defects.is_empty() {
        for defect in &outcome.defects {
            eprintln!("defect: {defect}");
        }
        return Err(format!(
            "component `{}` does not close: {} defect(s)",
            args.component,
            outcome.defects.len()
        )
        .into());
    }
    let plan = outcome
        .plan
        .ok_or_else(|| format!("component `{}` produced no plan", args.component))?;
    // What execution wants, and whether it may be manufactured now, are
    // separate questions — see `--representation-source`.
    let source = match args.representation_source.as_str() {
        "auto" => RepresentationSource::Auto,
        "stored" => RepresentationSource::Stored,
        "transient" => RepresentationSource::Transient,
        other => {
            return Err(format!(
                "unknown --representation-source `{other}`; expected auto, stored or transient"
            )
            .into())
        }
    };
    // The encoding a compiled pack would have to carry for this backend.
    // `None` on arms that execute the canonical bytes directly, which then
    // never look for a pack.
    let want = wanted_representation(args.backend);
    let store = OperandStore::open_for(&args.container, &inspection, want, source)?;

    let from_pack = store.selection().values().filter(|s| s.stored).count();
    if want.is_some() {
        println!(
            "representation: {}  source: {}  objects from a compiled pack: {}/{}",
            want.unwrap_or("-"),
            args.representation_source,
            from_pack,
            store.selection().len()
        );
    }

    #[cfg(all(feature = "gpu", target_os = "macos"))]
    {
        use larql_vindex::format::vindex3::opplan::exec::backend::{WeightFormat, WeightFormats};
        // The lowered path's per-class policy. Same scheduling for every
        // arm — one command buffer per token — so a comparison between
        // them prices the *representation*, which the pre-lowering
        // numbers could not (they mixed kernel families and starvation).
        let lowered = match args.backend {
            ExecBackend::MetalLowered => {
                Some((WeightFormats::uniform(WeightFormat::Nvfp4), "nvfp4-all"))
            }
            ExecBackend::MetalLoweredFfn => Some((
                WeightFormats {
                    attention: WeightFormat::F16,
                    ffn: WeightFormat::Nvfp4,
                    head: WeightFormat::F16,
                },
                "nvfp4-ffn",
            )),
            ExecBackend::MetalLoweredNoHead => Some((
                WeightFormats {
                    attention: WeightFormat::Nvfp4,
                    ffn: WeightFormat::Nvfp4,
                    head: WeightFormat::F16,
                },
                "nvfp4-no-head",
            )),
            ExecBackend::MetalLoweredMxfp4 => {
                Some((WeightFormats::uniform(WeightFormat::Mxfp4), "mxfp4-all"))
            }
            ExecBackend::MetalLoweredF16 => {
                Some((WeightFormats::uniform(WeightFormat::F16), "f16-all"))
            }
            ExecBackend::MetalLoweredMxfp4Ffn => Some((
                WeightFormats {
                    attention: WeightFormat::F16,
                    ffn: WeightFormat::Mxfp4,
                    head: WeightFormat::F16,
                },
                "mxfp4-ffn",
            )),
            _ => None,
        };
        if let Some((formats, label)) = lowered {
            let r = super::lowered::run_lowered(&args, &tokens, &plan, &store, formats, label);
            report_representation_work(&store, want, r.is_ok());
            return r;
        }
    }
    let outcome = match args.backend {
        ExecBackend::Reference => run_on(&ReferenceBackend::new(), &args, &tokens, &plan, &store),
        ExecBackend::Production => run_on(&ProductionBackend::new(), &args, &tokens, &plan, &store),
        // Same kernels as `production`; the difference is upstream, in
        // `wanted_representation`, which makes the store bind the
        // compiled NVFP4 pack instead of the canonical bytes. The
        // projector then dispatches `FusedNvfp4` off the resident
        // representation, exactly as it dispatches every other arm.
        ExecBackend::ProductionNvfp4 => {
            run_on(&ProductionBackend::new(), &args, &tokens, &plan, &store)
        }
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        ExecBackend::MetalMxfp4 => {
            let gpu = larql_compute_metal::MetalBackend::new()
                .ok_or("no Metal device available for --backend metal-mxfp4")?;
            // FFN-only MXFP4 — the gpt-oss precedent. The gates
            // falsified the wider presets on the 6-token fixture:
            // all-MXFP4 flipped the argmax (top-2 gap 0.08 vs
            // upstream's 1.13) and an f16 head alone did not recover
            // it (gap 0.01) — 4-bit attention projections accumulate
            // ~14% rel_rms across 52 layers. Attention and head stay
            // f16; the FFN bulk (~3/4 of the bytes) is quantised.
            use larql_vindex::format::vindex3::opplan::exec::backend::{
                WeightFormat, WeightFormats,
            };
            let backend =
                larql_vindex::format::vindex3::opplan::exec::device::DevicePlanBackend::with_formats(
                    gpu,
                    "metal-q1-mxfp4-ffn",
                    WeightFormats {
                        attention: WeightFormat::F16,
                        ffn: WeightFormat::Mxfp4,
                        head: WeightFormat::F16,
                    },
                );
            run_on(&backend, &args, &tokens, &plan, &store)
        }
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        ExecBackend::MetalLowered
        | ExecBackend::MetalLoweredFfn
        | ExecBackend::MetalLoweredNoHead
        | ExecBackend::MetalLoweredMxfp4
        | ExecBackend::MetalLoweredMxfp4Ffn
        | ExecBackend::MetalLoweredF16 => unreachable!("handled above"),
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        ExecBackend::MetalMxfp4All => {
            let gpu = larql_compute_metal::MetalBackend::new()
                .ok_or("no Metal device available for --backend metal-mxfp4-all")?;
            // The control arm: the preset Q1 falsified. Its job is to
            // fail, so that a Q2 arm holding the prediction is evidence
            // about the format rather than about the harness.
            use larql_vindex::format::vindex3::opplan::exec::backend::{
                WeightFormat, WeightFormats,
            };
            let backend =
                larql_vindex::format::vindex3::opplan::exec::device::DevicePlanBackend::with_formats(
                    gpu,
                    "metal-q1-mxfp4-all",
                    WeightFormats::uniform(WeightFormat::Mxfp4),
                );
            run_on(&backend, &args, &tokens, &plan, &store)
        }
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        ExecBackend::MetalNvfp4 | ExecBackend::MetalNvfp4Ffn | ExecBackend::MetalNvfp4NoHead => {
            let gpu = larql_compute_metal::MetalBackend::new()
                .ok_or("no Metal device available for the nvfp4 backends")?;
            // The VINDEX3-Q2 ladder. Q1 established that *this model's*
            // attention does not survive MXFP4; NVFP4 keeps the same
            // e2m1 elements and changes only the scale geometry, which a
            // weight-reconstruction sweep with an equal-bit-budget
            // control (E8M0 at group 16) isolated as the whole source of
            // the difference. Arm A is the one that matters — it is the
            // ~17 GB regime — and B and C exist so a failure says which
            // class it came from rather than only that it failed.
            use larql_vindex::format::vindex3::opplan::exec::backend::{
                WeightFormat, WeightFormats,
            };
            let (name, formats) = match args.backend {
                // A — everything 4-bit.
                ExecBackend::MetalNvfp4 => (
                    "metal-q2-nvfp4-all",
                    WeightFormats::uniform(WeightFormat::Nvfp4),
                ),
                // B — attention and FFN 4-bit, head wide. Isolates the
                // head, which Q1's second rung showed was not the whole
                // story under MXFP4.
                ExecBackend::MetalNvfp4NoHead => (
                    "metal-q2-nvfp4-no-head",
                    WeightFormats {
                        attention: WeightFormat::Nvfp4,
                        ffn: WeightFormat::Nvfp4,
                        head: WeightFormat::F16,
                    },
                ),
                // C — the Q1-passing partition, re-run under NVFP4, so
                // the two formats are compared at the same class split.
                _ => (
                    "metal-q2-nvfp4-ffn",
                    WeightFormats {
                        attention: WeightFormat::F16,
                        ffn: WeightFormat::Nvfp4,
                        head: WeightFormat::F16,
                    },
                ),
            };
            let backend =
                larql_vindex::format::vindex3::opplan::exec::device::DevicePlanBackend::with_formats(
                    gpu, name, formats,
                );
            run_on(&backend, &args, &tokens, &plan, &store)
        }
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        ExecBackend::Metal => {
            // vindex never links Metal: the CLI injects the concrete
            // device through larql-compute's MatMul seam. f16 weights so
            // the Metal buffer cache keeps the model resident (r2); the
            // engine tag names the realisation so a dump can never be
            // mistaken for the f32 r1 lowering.
            let gpu = larql_compute_metal::MetalBackend::new()
                .ok_or("no Metal device available for --backend metal")?;
            let backend =
                larql_vindex::format::vindex3::opplan::exec::device::DevicePlanBackend::new(
                    gpu,
                    "metal-r3-f16",
                    larql_vindex::format::vindex3::opplan::exec::backend::WeightFormat::F16,
                );
            run_on(&backend, &args, &tokens, &plan, &store)
        }
    };
    report_representation_work(&store, want, outcome.is_ok());
    outcome
}

/// Say how much of the representation the runtime had to manufacture.
///
/// The number that matters is not how long a load took but whether the
/// quantisation phase happened at all: a compiled representation is only
/// doing its job when this reads zero.
fn report_representation_work(store: &OperandStore, want: Option<&str>, ok: bool) {
    // A refused run has quantised nothing, but saying "served entirely
    // from stored bytes" over a failure would read as success.
    if want.is_none() || !ok {
        return;
    }
    let n = store.runtime_quantised();
    println!(
        "runtime compile: {n} tensor(s){}",
        if n == 0 {
            "  — served entirely from stored bytes"
        } else {
            ""
        }
    );
    let held = store.bound_at_stored_precision();
    if held > 0 {
        // Honouring a precision map means running higher precision than the
        // arm asked for. Never silent: a size that does not match the arm's
        // name should be explicable from the run's own output.
        println!(
            "stored precision: {held} tensor(s) ran above the requested format \
             (the pack's precision map)"
        );
    }
}

/// One monomorphised run: the backend is chosen exactly once, above.
fn run_on<B: PlanBackend>(
    backend: &B,
    args: &ExecArgs,
    tokens: &[u32],
    plan: &ComponentOpPlan,
    store: &OperandStore,
) -> Result<(), Box<dyn std::error::Error>> {
    let engine = format!("{ENGINE_PREFIX}-{}", backend.name());
    // A whole bank through one resident model. Checked before the
    // single-prompt paths because `--bank` supplies its own ids and the
    // `--tokens` argument is unused by it.
    if let Some(path) = &args.bank {
        let dump = args.dump_dir.clone().ok_or("--bank requires --dump-dir")?;
        let text = std::fs::read_to_string(path)?;
        let entries: Vec<super::bank::BankEntry> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?;
        return super::bank::run_bank(backend, &engine, plan, store, &entries, &dump);
    }
    // One flag, one meaning. A depth deeper than the model is refused by
    // the slice itself rather than clamped: a run that silently served a
    // different depth than it was asked for would poison a ladder.
    let slice = match args.draft_depth {
        Some(end) => ExecutionSlice::Draft { end },
        None => ExecutionSlice::Full,
    };
    if let Some(out) = &args.logit_dump {
        return super::teacher_force::run_teacher_force(
            backend, &engine, tokens, plan, store, out, slice,
        );
    }
    match (&args.dump_layers, args.generate) {
        (Some(dir), _) => run_dump(dir, &engine, args, tokens, plan, store, backend),
        (None, Some(new_tokens)) => {
            super::generate::run_generate(backend, &engine, tokens, new_tokens, plan, store)
        }
        (None, None) => {
            let trace = execute_slice(plan, store, tokens, backend, slice)?;
            summarise(&engine, &trace);
            Ok(())
        }
    }
}

/// The dumped (and resumable) execution path.
#[allow(clippy::too_many_arguments)]
fn run_dump<B: PlanBackend>(
    dir: &PathBuf,
    engine: &str,
    args: &ExecArgs,
    tokens: &[u32],
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dir)?;
    let hidden = plan
        .embedding
        .as_ref()
        .map(|e| e.table.shape[1])
        .ok_or("plan carries no embedding op")?;
    let seq = tokens.len();
    let total_layers = plan.layers.len();
    let sidecar = ResumeSidecar {
        engine: engine.to_string(),
        container: args.container.display().to_string(),
        component: args.component.clone(),
        token_ids: tokens.to_vec(),
    };

    let resume = if args.resume {
        prepare_resume(dir, &sidecar, seq, hidden, total_layers)?
    } else {
        // A fresh dump must start from a clean slate: planes left by an
        // earlier, longer run would otherwise be indistinguishable from
        // this run's own progress the next time `--resume` scans.
        clear_dump(dir, total_layers)?;
        std::fs::write(
            dir.join(RESUME_NAME),
            serde_json::to_string_pretty(&sidecar)?,
        )?;
        None
    };

    let started = Instant::now();
    let mut layer_started = Instant::now();
    let out = execute_plan_streaming(plan, store, tokens, backend, resume, &mut |event| {
        match event {
            PlaneEvent::Embedded(rows) => {
                write_rows(&dir.join(plane_name(0)), rows)?;
                eprintln!(
                    "plane 000 (embedding)  {:.1}s",
                    started.elapsed().as_secs_f64()
                );
            }
            PlaneEvent::Layer { index, trace } => {
                write_rows(&dir.join(plane_name(index + 1)), &trace.post_layer)?;
                eprintln!(
                    "layer {:>3}/{}  {:.1}s  (elapsed {:.0}s)",
                    index + 1,
                    total_layers,
                    layer_started.elapsed().as_secs_f64(),
                    started.elapsed().as_secs_f64(),
                );
            }
        }
        layer_started = Instant::now();
        Ok(())
    })?;

    write_rows(
        &dir.join(FINAL_NORM_PLANE),
        std::slice::from_ref(&out.final_hidden),
    )?;
    if let Some(logits) = &out.logits {
        write_rows(&dir.join(LOGITS_PLANE), std::slice::from_ref(logits))?;
    }

    let manifest = LayerDumpManifest {
        engine: engine.to_string(),
        model: args.container.display().to_string(),
        num_layers: total_layers,
        seq_len: seq,
        hidden_size: hidden,
        token_ids: tokens.to_vec(),
        planes: (0..=total_layers).map(plane_name).collect(),
        dtype: PLANE_DTYPE.to_string(),
    };
    std::fs::write(
        dir.join(MANIFEST_NAME),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    eprintln!(
        "wrote {} planes + final norm + logits to {}",
        total_layers + 1,
        dir.display()
    );
    Ok(())
}

/// Validate a `--resume` request and build the interpreter's entry state.
///
/// Returns `None` (start from the embedding) when no complete plane
/// survived — that is still a valid resume of a run killed before plane
/// 000 landed.
pub(super) fn prepare_resume(
    dir: &Path,
    sidecar: &ResumeSidecar,
    seq: usize,
    hidden: usize,
    total_layers: usize,
) -> Result<Option<ResumePoint>, Box<dyn std::error::Error>> {
    if dir.join(MANIFEST_NAME).exists() {
        return Err("dump is already complete (manifest present) — nothing to resume".into());
    }
    let recorded = std::fs::read_to_string(dir.join(RESUME_NAME))
        .map_err(|_| "no resume record in the dump directory — was a dump ever started here?")?;
    let recorded: ResumeSidecar = serde_json::from_str(&recorded)?;
    if &recorded != sidecar {
        return Err(
            "resume record does not match this invocation (tokens, container, component, \
             or backend differ) — refusing to splice two different runs"
                .into(),
        );
    }
    match last_complete_plane(dir, seq, hidden, total_layers) {
        Some(plane) => {
            let rows = read_plane(&dir.join(plane_name(plane)), seq, hidden)?;
            eprintln!(
                "resuming from plane {plane:03}: layers {}..{} still to run",
                plane, total_layers
            );
            Ok(Some(ResumePoint {
                next_layer: plane,
                hidden: rows,
            }))
        }
        None => Ok(None),
    }
}

/// Highest plane index `p` such that planes `0..=p` all exist with the
/// right byte length. A truncated file (killed mid-write) ends the scan
/// *before* itself, so resume re-executes the layer that was cut off.
pub(super) fn last_complete_plane(
    dir: &Path,
    seq: usize,
    hidden: usize,
    total_layers: usize,
) -> Option<usize> {
    let expected = (seq * hidden * BYTES_PER_VALUE) as u64;
    let mut last = None;
    for plane in 0..=total_layers {
        match std::fs::metadata(dir.join(plane_name(plane))) {
            Ok(meta) if meta.len() == expected => last = Some(plane),
            _ => break,
        }
    }
    last
}

/// Remove every file a previous dump could have left, so a fresh run's
/// directory contains only its own progress.
fn clear_dump(dir: &Path, total_layers: usize) -> std::io::Result<()> {
    let mut names: Vec<String> = (0..=total_layers).map(plane_name).collect();
    names.push(FINAL_NORM_PLANE.to_string());
    names.push(LOGITS_PLANE.to_string());
    names.push(MANIFEST_NAME.to_string());
    names.push(RESUME_NAME.to_string());
    for name in names {
        match std::fs::remove_file(dir.join(name)) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// One raw little-endian f32 plane back into per-position rows.
pub(super) fn read_plane(
    path: &Path,
    seq: usize,
    hidden: usize,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let expected = seq * hidden * BYTES_PER_VALUE;
    if bytes.len() != expected {
        return Err(format!(
            "plane {} is {} bytes, expected {expected}",
            path.display(),
            bytes.len()
        )
        .into());
    }
    let values: Vec<f32> = bytes
        .chunks_exact(BYTES_PER_VALUE)
        .map(|c| f32::from_le_bytes(c.try_into().expect("chunks_exact yields 4-byte chunks")))
        .collect();
    Ok(values.chunks(hidden).map(<[f32]>::to_vec).collect())
}

/// Write rows as one plane, converting IO failure into the interpreter's
/// error type so the sink can abort the run.
fn write_rows(path: &Path, rows: &[Vec<f32>]) -> Result<(), VindexError> {
    let plane = plane_of(rows)
        .map_err(|e| VindexError::Parse(format!("plane shape for {}: {e}", path.display())))?;
    write_plane(path, &plane)
        .map_err(|e| VindexError::Parse(format!("writing {}: {e}", path.display())))
}

/// Parse a comma-separated token list.
fn parse_tokens(spec: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let tokens: Result<Vec<u32>, _> = spec
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::parse::<u32>)
        .collect();
    let tokens = tokens.map_err(|e| format!("--tokens must be comma-separated ids: {e}"))?;
    if tokens.is_empty() {
        return Err("--tokens is empty".into());
    }
    Ok(tokens)
}

/// One `[seq, hidden]` plane from a per-position row list.
fn plane_of(rows: &[Vec<f32>]) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let seq = rows.len();
    let hidden = rows.first().map(Vec::len).unwrap_or(0);
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    Ok(Array2::from_shape_vec((seq, hidden), flat)?)
}

/// Without `--dump-layers`, print enough to see the forward ran.
fn summarise(engine: &str, trace: &ExecutionTrace) {
    println!("engine: {engine}");
    println!(
        "layers: {}  seq: {}  hidden: {}",
        trace.layers.len(),
        trace.embedded.len(),
        trace.embedded.first().map(Vec::len).unwrap_or(0),
    );
    match &trace.logits {
        Some(logits) => match super::generate::argmax(logits) {
            Some((best, value)) => {
                println!("logits: {}, argmax {best} ({value:+.4})", logits.len());
            }
            None => println!("logits: empty"),
        },
        None => println!("logits: none (plan carries no output head)"),
    }
}

/// The stored encoding a backend could be served from, if one is compiled.
///
/// Only the NVFP4 arms have a compiled counterpart today. Arms that run
/// the canonical bytes return `None` and never look for a pack, so adding
/// packs to a container cannot change what they execute.
fn wanted_representation(backend: ExecBackend) -> Option<&'static str> {
    use larql_vindex::format::vindex3::represent::nvfp4_pack::DTYPE_NVFP4;
    // Exhaustive, with no wildcard arm, and that is the point: `_ =>
    // None` stood here and a newly added NVFP4 backend silently
    // inherited "wants nothing", bound the canonical bytes and produced
    // logits bit-identical to BF16. A run that looks like perfect
    // fidelity is the exact failure this file must not be able to
    // express, so a new backend is now a compile error until someone
    // states which representation it executes.
    match backend {
        ExecBackend::Reference | ExecBackend::Production => None,
        ExecBackend::ProductionNvfp4 => Some(DTYPE_NVFP4),
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        ExecBackend::Metal
        | ExecBackend::MetalMxfp4
        | ExecBackend::MetalMxfp4All
        | ExecBackend::MetalLoweredMxfp4
        | ExecBackend::MetalLoweredMxfp4Ffn
        | ExecBackend::MetalLoweredF16 => None,
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        ExecBackend::MetalNvfp4
        | ExecBackend::MetalNvfp4Ffn
        | ExecBackend::MetalNvfp4NoHead
        | ExecBackend::MetalLowered
        | ExecBackend::MetalLoweredFfn
        | ExecBackend::MetalLoweredNoHead => Some(DTYPE_NVFP4),
    }
}
