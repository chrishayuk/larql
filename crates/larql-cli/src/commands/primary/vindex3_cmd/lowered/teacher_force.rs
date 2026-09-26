//! Teacher-forced logit capture on the lowered path — the lowered-arm
//! counterpart to `../teacher_force.rs`'s interpreter implementation.
//!
//! Before this existed, `--logit-dump` on `metal-lowered-*` silently wrote
//! only the *last* position's logits (whatever `step()` left resident in
//! the device's single logits slot), regardless of how many tokens were
//! passed — the file was always exactly `vocab * 4` bytes. A caller doing
//! per-position KL/NLL scoring (the shape `../teacher_force.rs`'s own
//! doc comment exists to serve) got silently truncated to one position
//! with no error, which is a worse failure than refusing outright.
//!
//! The fix costs nothing architecturally: prefill on the lowered path is
//! already a per-token step loop (`run.rs`'s `for &token in tokens {
//! session.step(token)?; }`), one command buffer per token, computing one
//! position's logits at a time into the same device slot `last_logits()`
//! reads. The lookahead optimisation that overwrites that slot for
//! position `t+1` is only *encoded* during prefill, never *committed* —
//! commit only happens once `begin_decode()` flips `decode_chain`, which
//! this path never calls — so reading `last_logits()` immediately after
//! each `step()` during prefill is safe with no synchronisation change.
//! This mirrors `dump_lowered`'s already-proven per-position capture
//! (`step_capturing` in a loop) applied to the plain non-capturing
//! `step()` path instead.
//!
//! Writes `[positions, vocab]` f32 — byte-identical layout to the
//! interpreter's `run_teacher_force`, so a scorer reading either doesn't
//! need to know which backend produced the dump.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use super::LoweredSession;

/// Step `tokens` through the lowering one position at a time, writing
/// every position's logits to `out`. Every position is stepped —
/// including the last, whose prediction has no target in the sequence —
/// for the same reason the interpreter path does: the caller decides
/// what to score, and silently dropping it here would make the plane's
/// row count disagree with the token count for no reason a reader could
/// see.
pub(super) fn run_teacher_force_lowered(
    session: &mut LoweredSession<'_>,
    engine: &str,
    tokens: &[u32],
    out: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let mut file = std::io::BufWriter::new(std::fs::File::create(out)?);
    let mut vocab = 0usize;
    for (i, &token) in tokens.iter().enumerate() {
        session.step(token)?;
        let logits = session
            .last_logits()
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
            file.write_all(&value.to_le_bytes())?;
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
    file.flush()?;

    println!("engine: {engine}");
    println!("positions: {}  vocab: {vocab}", tokens.len());
    println!(
        "teacher-forced {} positions in {:.1} s ({:.0} ms/position)",
        tokens.len(),
        started.elapsed().as_secs_f64(),
        started.elapsed().as_secs_f64() * 1000.0 / tokens.len() as f64,
    );
    println!("wrote [{}, {vocab}] f32 to {}", tokens.len(), out.display());
    Ok(())
}
