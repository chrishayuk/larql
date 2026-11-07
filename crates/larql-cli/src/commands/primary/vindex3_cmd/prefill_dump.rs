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
use larql_vindex::format::vindex3::opplan::exec::{execute_slice, execute_slice_in};
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

/// Prefill `tokens[..n]` for each `n` in `lengths`, writing each final
/// position's logits to `out` as f32 little-endian rows.
pub(super) fn run_prefill_dump<B: PlanBackend>(
    backend: &B,
    tokens: &[u32],
    lengths: &[usize],
    plan: &ComponentOpPlan,
    store: &OperandStore,
    out: &Path,
    slice: ExecutionSlice,
) -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let mut file = std::io::BufWriter::new(std::fs::File::create(out)?);
    let mut vocab = None;
    for &length in lengths {
        let prefix = &tokens[..length];
        let trace = match one_shot_state(plan)? {
            Some(state) => execute_slice_in(plan, store, prefix, backend, slice.clone(), state)?,
            None => execute_slice(plan, store, prefix, backend, slice.clone())?,
        };
        let logits = trace
            .logits
            .ok_or("plan carries no output head — cannot score")?;
        same_vocabulary(&mut vocab, logits.len(), length)?;
        for value in &logits {
            file.write_all(&value.to_le_bytes())?;
        }
    }
    file.flush()?;
    println!(
        "prefilled {} prefix(es) in {:.1} s  vocab: {}",
        lengths.len(),
        started.elapsed().as_secs_f64(),
        vocab.unwrap_or(0)
    );
    println!("wrote {}", out.display());
    Ok(())
}
