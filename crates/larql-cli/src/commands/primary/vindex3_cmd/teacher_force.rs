//! Teacher-forced logit capture — the predictive-units gate (VINDEX3-Q2-R3).
//!
//! The 6-token oracle answers "does this realisation predict the same
//! token". It cannot answer "does this realisation predict the same
//! *distribution*", and the Q2 arms made that gap concrete: all-NVFP4
//! restores the correct argmax but **over-separates** the top two logits
//! (gap 1.49 against the f16 anchor's 1.07). Argmax parity is therefore
//! stronger than distribution parity, and a serving claim needs the
//! latter.
//!
//! **Why teacher forcing.** A greedy generation comparison is not a
//! paired measurement: each arm picks its own next token, the contexts
//! diverge after the first disagreement, and the resulting KL mixes
//! representation error with the consequences of having read different
//! text. Stepping both arms through **one fixed token sequence** makes
//! every position's context identical by construction, so a per-position
//! divergence is attributable to the representation alone.
//!
//! Writes `[positions, vocab]` f32 — position `i` holds the distribution
//! predicting token `i+1`, which is what an NLL over the sequence needs.

use larql_vindex::format::vindex3::opplan::exec::cpu::gemv_witness;
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use larql_vindex::format::vindex3::opplan::ComponentOpPlan;

/// The sidecar beside a logit dump: `<out stem>.meta.json`.
pub(super) const META_EXTENSION: &str = "meta.json";

/// Step `tokens` through the plan one position at a time, writing every
/// position's logits to `out`.
///
/// Every position is stepped — including the last, whose prediction has
/// no target in the sequence — because the caller decides what to score;
/// silently dropping it here would make the plane's row count disagree
/// with the token count for no reason a reader could see.
pub(super) fn run_teacher_force<B: PlanBackend>(
    backend: &B,
    engine: &str,
    tokens: &[u32],
    plan: &ComponentOpPlan,
    store: &OperandStore,
    out: &Path,
    slice: ExecutionSlice,
    capture: Option<&super::site_capture_dump::CaptureTarget>,
    rope_angles: larql_vindex::format::vindex3::opplan::exec::rope_angles::RopeAngles,
) -> Result<(), Box<dyn std::error::Error>> {
    let loading = Instant::now();
    // Prepared explicitly rather than through `DecodeSession::new`, so
    // the slice reaches the operand loader. A `Draft` prepares only its
    // own layers — measured at 4 + ~13.4 reads per layer on the 27B
    // container — which is what makes a shallow draft affordable at all.
    let mut ops = PreparedOperands::load(plan, store, backend, slice.clone())?;
    ops.set_rope_angles(rope_angles)?;
    let mut kv = crate::commands::primary::continuation::select_for(plan, None)?.build();
    let mut session = DecodeSession::over_prepared(plan, &ops, backend, &mut *kv)?;
    let prepare_s = loading.elapsed().as_secs_f64();
    let executed = ops.executed_layers();
    eprintln!(
        "weights resident in {prepare_s:.1} s — layers {}..{} of {}",
        executed.start,
        executed.end,
        plan.layers.len()
    );

    // PREC-DIAG-1 E0a: the capture is built (and its budget judged)
    // before the first step; an observer the step only borrows.
    let mut observer = match capture {
        Some(target) => {
            Some(target.observer(plan, executed.clone(), ops.hidden(), tokens.len())?)
        }
        None => None,
    };

    let started = Instant::now();
    // Appears at `out` only once every position is written (#744).
    let mut file = super::pending_output::PendingOutput::create(out)?;
    let mut vocab = 0usize;
    // As the bytes are written, so that a witness describes this run's dump, not whatever is at
    // that path later.
    let mut witness_digest = gemv_witness::is_active().then(|| (Sha256::new(), 0u64));
    for (i, &token) in tokens.iter().enumerate() {
        let stepped = match observer.as_mut() {
            Some(observer) => session.step_observed(token, observer)?,
            None => session.step(token)?,
        };
        let logits = stepped
            .logits
            .ok_or("plan carries no output head — cannot score")?;
        if vocab == 0 {
            vocab = logits.len();
        } else if logits.len() != vocab {
            return Err(format!(
                "position {i}: vocabulary changed mid-sequence ({} vs {vocab})",
                logits.len()
            )
            .into());
        }
        for value in &logits {
            let bytes = value.to_le_bytes();
            file.write_all(&bytes)?;
            if let Some((digest, written)) = &mut witness_digest {
                digest.update(bytes);
                *written += bytes.len() as u64;
            }
        }
        if (i + 1) % 32 == 0 || i + 1 == tokens.len() {
            eprintln!(
                "  position {:>4}/{}  ({:.1} s)",
                i + 1,
                tokens.len(),
                started.elapsed().as_secs_f64()
            );
        }
    }
    file.commit()?;
    if let Some((digest, written)) = witness_digest {
        let hex = digest
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        gemv_witness::note_logit_dump(hex, written);
    }

    println!("engine: {engine}");
    println!("rope angles: {}", rope_angles.name());
    println!("positions: {}  vocab: {vocab}", tokens.len());
    println!(
        "teacher-forced {} positions in {:.1} s ({:.0} ms/position)",
        tokens.len(),
        started.elapsed().as_secs_f64(),
        started.elapsed().as_secs_f64() * 1000.0 / tokens.len() as f64,
    );
    println!("wrote [{}, {vocab}] f32 to {}", tokens.len(), out.display());

    // A sidecar beside the plane, so a comparison can ASSERT the run's
    // identity instead of trusting a filename. A depth result is only a
    // depth result if the arms agree on backend, format policy and
    // container; this project has manufactured a "depth effect" that was
    // a backend effect before, and a bare .f32 carries no way to catch it.
    let steps_s = started.elapsed().as_secs_f64();
    let meta = format!(
        "{{\n  \"engine\": \"{engine}\",\n  \"draft_depth\": {},\n  \
         \"executed_layers\": [{}, {}],\n  \"model_layers\": {},\n  \
         \"positions\": {},\n  \"vocab\": {vocab},\n  \
         \"prepare_s\": {prepare_s:.3},\n  \"steps_s\": {steps_s:.3},\n  \
         \"ms_per_position\": {:.1},\n  \
         \"max_format_env\": \"{}\",\n  \"rope_angles\": \"{}\"\n}}\n",
        match &slice {
            ExecutionSlice::Draft { end } => end.to_string(),
            _ => "null".to_string(),
        },
        executed.start,
        executed.end,
        plan.layers.len(),
        tokens.len(),
        steps_s * 1000.0 / tokens.len() as f64,
        std::env::var("LARQL_CPU_MAX_FORMAT").unwrap_or_default(),
        rope_angles.name(),
    );
    let meta_path = out.with_extension(META_EXTENSION);
    std::fs::write(&meta_path, meta)?;
    println!("wrote provenance to {}", meta_path.display());
    if let (Some(target), Some(observer)) = (capture, observer) {
        // What the CPU projections actually ran, for the capture's record.
        let projection_plans: Vec<(String, u64)> =
            larql_vindex::format::vindex3::opplan::exec::cpu::ledger()
                .all()
                .iter()
                .filter(|(_, t)| t.calls > 0)
                .map(|(plan, t)| (format!("{plan:?}"), t.calls))
                .collect();
        target.write(observer.finish(), out, &projection_plans)?;
    }
    Ok(())
}
