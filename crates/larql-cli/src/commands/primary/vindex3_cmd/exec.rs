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
//! ids this reads back. `larql run <container> [prompt]` is the text
//! shell over the same preparation (`prepare`) and the same interpreter;
//! its `--emit-ids` prints the ids this verb would be given.
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
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prepared::ExecutionSlice;
use larql_vindex::format::vindex3::opplan::exec::{
    execute_plan_streaming, execute_slice, ExecutionTrace, Plane, PlaneEvent, ResumePoint,
};
use larql_vindex::format::vindex3::opplan::ComponentOpPlan;
use ndarray::Array2;

use super::super::shannon_trace::dump::{
    plane_name, write_plane, LayerDumpManifest, MANIFEST_NAME, PLANE_DTYPE,
};
use larql_inference::vindex3::OpenedComponent;

use super::prepare::{
    parse_representation_source, prepare, with_plan_backend, BackendVisitor, ENGINE_PREFIX,
};
use super::ExecArgs;

/// Extra planes beyond the layer table, matching
/// `scripts/capture_glimmer_oracle.py`.
const FINAL_NORM_PLANE: &str = "final_norm.f32";
const LOGITS_PLANE: &str = "logits.f32";

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
    let source = parse_representation_source(&args.representation_source)?;
    let OpenedComponent {
        plan, store, want, ..
    } = prepare(&args.container, &args.component, args.backend, source)?;

    let from_pack = store.selection().values().filter(|s| s.stored).count();
    if let Some(want) = &want {
        println!(
            "representation: {want}  source: {}  objects from a compiled pack: {}/{}",
            args.representation_source,
            from_pack,
            store.selection().len()
        );
    }

    #[cfg(all(feature = "gpu", target_os = "macos"))]
    {
        if let Some((formats, label)) = super::prepare::lowered_formats(args.backend) {
            let r = super::lowered::run_lowered(&args, &tokens, &plan, &store, formats, label);
            report_representation_work(&store, want.as_deref(), r.is_ok());
            return r;
        }
    }
    let outcome = with_plan_backend(
        args.backend,
        ExecVisitor {
            args: &args,
            tokens: &tokens,
            plan: &plan,
            store: &store,
        },
    );
    report_representation_work(&store, want.as_deref(), outcome.is_ok());
    outcome
}

/// The exec verb's work, once the backend is a concrete type.
struct ExecVisitor<'a> {
    args: &'a ExecArgs,
    tokens: &'a [u32],
    plan: &'a ComponentOpPlan,
    store: &'a OperandStore,
}

impl BackendVisitor for ExecVisitor<'_> {
    type Out = ();

    fn visit<B: PlanBackend>(self, backend: &B) -> Result<(), Box<dyn std::error::Error>> {
        run_on(backend, self.args, self.tokens, self.plan, self.store)
    }
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
    report_projection_plans();
}

/// Bytes per gigabyte, as the projection ledger reports traffic.
const BYTES_PER_GB: f64 = 1e9;

/// Which projection plans actually ran, from the executor's own ledger.
///
/// The representation line says what the backend ASKED for; this says
/// what the bytes were consumed by. PARETO-1 needs the second: a stored
/// K-quant pack can be executed in place (`FusedKQuant`) or decoded and
/// run as f32 (`BlasF32`), and both bind the same pack, both report
/// `runtime compile: 0`, and they differ in every logit. A guard that
/// read only the first line could not tell them apart — which is how the
/// format-cap confound went unseen.
fn report_projection_plans() {
    let rows: Vec<String> = larql_vindex::format::vindex3::opplan::exec::cpu::ledger()
        .all()
        .iter()
        .filter(|(_, t)| t.calls > 0)
        .map(|(plan, t)| {
            format!(
                "{plan:?} {} calls {:.2} GB",
                t.calls,
                t.bytes as f64 / BYTES_PER_GB
            )
        })
        .collect();
    if !rows.is_empty() {
        println!("projection plans: {}", rows.join(", "));
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
            if args.residency_curve {
                return super::generate::run_residency_curve(
                    backend,
                    &engine,
                    tokens,
                    new_tokens,
                    plan,
                    store,
                    args.repeat,
                    args.warmup,
                    args.unquiet_ok,
                    &args.expert_access,
                );
            }
            super::generate::run_generate(backend, &engine, tokens, new_tokens, plan, store)
                .map(|_generated| ())
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
            PlaneEvent::Embedded(plane) => {
                write_rows(&dir.join(plane_name(0)), plane.try_rows()?)?;
                eprintln!(
                    "plane 000 (embedding)  {:.1}s",
                    started.elapsed().as_secs_f64()
                );
            }
            PlaneEvent::Layer { index, trace } => {
                write_rows(
                    &dir.join(plane_name(index + 1)),
                    trace.post_layer.try_rows()?,
                )?;
                eprintln!(
                    "layer {:>3}/{}  {:.1}s  (elapsed {:.0}s)",
                    index + 1,
                    total_layers,
                    layer_started.elapsed().as_secs_f64(),
                    started.elapsed().as_secs_f64(),
                );
            }
            PlaneEvent::HyperConnectionSite(_)
            | PlaneEvent::AttentionResidualSite(_)
            | PlaneEvent::AttentionResidualBoundary(_) => {}
        }
        layer_started = Instant::now();
        Ok(())
    })?;
    // (A site's state and a block-boundary event are the witness's taps,
    // not planes; the dump persists layer boundaries only. A dump of an
    // attention-residual component does not reach here anyway — its
    // planes hold histories, and `try_rows` refuses them by name above
    // rather than flattening a prefix-plus-snapshots state into a file
    // whose format nothing could read back.)

    write_rows(
        &dir.join(FINAL_NORM_PLANE),
        &[out.exit.try_hidden()?.to_vec()],
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
                hidden: Plane::Rows(rows),
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
        trace.embedded.positions(),
        match &trace.embedded {
            Plane::Rows(rows) => rows.first().map(Vec::len).unwrap_or(0),
            Plane::Bundles(bundles) => bundles.first().map(|b| b.hidden()).unwrap_or(0),
            // A residual history's width is its prefix's; the snapshot
            // count is depth, not width, and reporting it here would
            // call a history a wider hidden state than it is.
            Plane::Histories(histories) => {
                histories.first().map(|h| h.hidden()).unwrap_or(0)
            }
        },
    );
    match &trace.logits {
        Some(logits) => match super::decode::argmax(logits) {
            Some((best, value)) => {
                println!("logits: {}, argmax {best} ({value:+.4})", logits.len());
            }
            None => println!("logits: empty"),
        },
        None => println!("logits: none (plan carries no output head)"),
    }
}
