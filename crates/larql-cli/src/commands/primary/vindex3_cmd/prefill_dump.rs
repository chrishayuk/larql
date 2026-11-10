//! Fresh-prefill logit capture — the independent arm to teacher forcing.
//!
//! [`super::teacher_force`] steps the CACHED decode path one position at
//! a time. This runs the batch PREFILL from position 0 once per requested
//! prefix length, carrying no continuation state between rows, and keeps
//! only each prefill's final-position logits. Row `r` of the dump is the
//! distribution predicting the token after `tokens[..lengths[r]]` — the
//! same row the teacher-forced dump holds at index `lengths[r] - 1`.
//!
//! An equivalence harness compares BOTH against an outside reference at
//! the same positions, so a cache bug and a prefill bug cannot cancel by
//! agreeing with each other.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prepared::ExecutionSlice;
use larql_vindex::format::vindex3::opplan::exec::rope_angles::RopeAngles;
use larql_vindex::format::vindex3::opplan::exec::{
    execute_slice, execute_slice_in, execute_slice_position_logits_with_rope_angles,
};
use larql_vindex::format::vindex3::opplan::ComponentOpPlan;

use crate::commands::primary::continuation::one_shot_state;

/// The prefix lengths to prefill: the parsed list, or the whole sequence.
///
/// Every length must lie in `1..=sequence`: an empty prefix predicts
/// nothing, and one longer than the sequence names tokens that were
/// never given. Refused rather than clamped, so a dump's rows are always
/// exactly the lengths that were asked for.
pub(super) fn parse_lengths(
    spec: Option<&str>,
    sequence: usize,
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let Some(spec) = spec else {
        return Ok(vec![sequence]);
    };
    let lengths = spec
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("--prefill-lengths: {e}"))?;
    if let Some(bad) = lengths.iter().find(|&&n| n == 0 || n > sequence) {
        return Err(format!(
            "--prefill-lengths: {bad} is outside 1..={sequence} (the given token count)"
        )
        .into());
    }
    Ok(lengths)
}

/// Every row of one dump must have the first row's width: a reader
/// reshapes the file by it.
pub(super) fn same_vocabulary(
    vocab: &mut Option<usize>,
    width: usize,
    length: usize,
) -> Result<(), String> {
    match *vocab.get_or_insert(width) == width {
        true => Ok(()),
        false => Err(format!(
            "prefix {length}: vocabulary changed between prefills"
        )),
    }
}

/// How the fresh-prefill rows are computed. Both run a one-shot causal
/// prefill with no decode state; they differ only in how many prefills.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum PrefillMethod {
    /// ONE prefill over the whole sequence, the exit (final norm + head)
    /// applied to each requested position's final residual.
    Batched,
    /// One prefill per requested prefix, keeping only its last row — the
    /// original method, O(n²) over a sequence; kept as the measurement
    /// control the batched method is validated against.
    PerPrefix,
}

/// Write the logits predicting the token after `tokens[..n]`, for each
/// `n` in `lengths`, to `out` as f32 little-endian rows.
pub(super) fn run_prefill_dump<B: PlanBackend>(
    backend: &B,
    tokens: &[u32],
    lengths: &[usize],
    plan: &ComponentOpPlan,
    store: &OperandStore,
    out: &Path,
    slice: ExecutionSlice,
    method: PrefillMethod,
    rope_angles: RopeAngles,
) -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let rows = match method {
        PrefillMethod::Batched => {
            batched_rows(backend, tokens, lengths, plan, store, slice, rope_angles)?
        }
        // The measurement-method control runs the shipped construction
        // only; `exec` refuses `--rope-angles` with it before this.
        PrefillMethod::PerPrefix if rope_angles == RopeAngles::Larql => {
            per_prefix_rows(backend, tokens, lengths, plan, store, slice)?
        }
        PrefillMethod::PerPrefix => {
            return Err(
                "--rope-angles hf-f32 is not accepted with --prefill-method per-prefix".into(),
            )
        }
    };
    println!("rope angles: {}", rope_angles.name());
    // Appears at `out` only once every row is written (#744).
    let mut file = super::pending_output::PendingOutput::create(out)?;
    let mut vocab = None;
    for (row, &length) in rows.iter().zip(lengths) {
        same_vocabulary(&mut vocab, row.len(), length)?;
        for value in row {
            file.write_all(&value.to_le_bytes())?;
        }
    }
    file.commit()?;
    println!(
        "prefilled {} position(s) ({method:?}) in {:.1} s  vocab: {}",
        lengths.len(),
        started.elapsed().as_secs_f64(),
        vocab.unwrap_or(0)
    );
    println!("wrote {}", out.display());
    Ok(())
}

/// One causal prefill over `tokens[..max(lengths)]`, rows at each length.
fn batched_rows<B: PlanBackend>(
    backend: &B,
    tokens: &[u32],
    lengths: &[usize],
    plan: &ComponentOpPlan,
    store: &OperandStore,
    slice: ExecutionSlice,
    rope_angles: RopeAngles,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let longest = lengths.iter().copied().max().unwrap_or(0);
    let positions: Vec<usize> = lengths.iter().map(|n| n - 1).collect();
    Ok(execute_slice_position_logits_with_rope_angles(
        plan,
        store,
        &tokens[..longest],
        backend,
        slice,
        one_shot_state(plan)?,
        &positions,
        rope_angles,
    )?)
}

/// One causal prefill per prefix, its last row kept.
fn per_prefix_rows<B: PlanBackend>(
    backend: &B,
    tokens: &[u32],
    lengths: &[usize],
    plan: &ComponentOpPlan,
    store: &OperandStore,
    slice: ExecutionSlice,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let mut rows = Vec::with_capacity(lengths.len());
    for &length in lengths {
        let prefix = &tokens[..length];
        let trace = match one_shot_state(plan)? {
            Some(state) => execute_slice_in(plan, store, prefix, backend, slice.clone(), state)?,
            None => execute_slice(plan, store, prefix, backend, slice.clone())?,
        };
        rows.push(
            trace
                .logits
                .ok_or("plan carries no output head — cannot score")?,
        );
    }
    Ok(rows)
}
