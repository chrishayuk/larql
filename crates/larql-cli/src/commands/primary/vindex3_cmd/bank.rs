//! `vindex3 exec --bank` — teacher-force many prompts through one
//! resident model.
//!
//! The statistics are the point of a quality bank; the harness is not. But
//! at 30B a process per prompt turned a 69-prompt sweep into hours of
//! model loading, which stops being an inefficiency and starts deciding
//! how many hypotheses can be afforded. Loading once is infrastructure
//! only, and it must be *provably* only that.
//!
//! ## The property that must survive
//!
//! Every position must see exactly the same teacher-forced context as its
//! reference did. One resident model with a long-lived session would leak
//! K/V and position state from one bank entry into the next, and the
//! damage would be invisible: later prompts would score against a context
//! no reference ever saw, and the numbers would still look like numbers.
//!
//! So state is not *reset* between entries — it is *replaced*. Operands
//! are prepared once (the expensive part, and immutable); every entry gets
//! a brand-new provider built from the selected continuation, which cannot
//! carry anything because it did not exist a moment ago. The session asserts it starts at position 0 and
//! ends at exactly the entry's length, so a leak is a failure rather than
//! a silent contamination.

use std::io::Write;
use std::path::Path;

use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use larql_vindex::format::vindex3::opplan::ComponentOpPlan;

/// One bank entry: an id and the token ids to teacher-force.
#[derive(serde::Deserialize)]
pub struct BankEntry {
    pub id: String,
    pub ids: Vec<u32>,
}

/// Run every entry through one resident model, writing `<dump>/<id>.f32`.
pub fn run_bank<B: PlanBackend>(
    backend: &B,
    engine: &str,
    plan: &ComponentOpPlan,
    store: &OperandStore,
    entries: &[BankEntry],
    dump_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dump_dir)?;

    let loading = std::time::Instant::now();
    // The expensive, immutable half: done once and shared by every entry.
    let source: larql_vindex::format::vindex3::opplan::exec::operands::OperandSource<'_> =
        store.into();
    let ops = PreparedOperands::load(plan, source, backend, ExecutionSlice::Full)?;
    let load_s = loading.elapsed().as_secs_f64();
    println!("engine: {engine}");
    println!(
        "weights resident in {load_s:.1} s (once, for {} entries)",
        entries.len()
    );

    let started = std::time::Instant::now();
    let mut positions = 0usize;
    for (n, entry) in entries.iter().enumerate() {
        // A brand-new continuation state per entry. Not a reset — a
        // replacement, so there is nothing that *could* carry over.
        let mut kv = crate::commands::primary::continuation::select_for(plan, None)?.build();
        let mut session = DecodeSession::over_prepared(plan, &ops, backend, &mut *kv)?;
        if session.position() != 0 {
            return Err(format!(
                "{}: session started at position {} — state leaked from the previous entry",
                entry.id,
                session.position()
            )
            .into());
        }

        let path = dump_dir.join(format!("{}.f32", entry.id));
        let mut file = std::io::BufWriter::new(std::fs::File::create(&path)?);
        let mut vocab = 0usize;
        for (i, &token) in entry.ids.iter().enumerate() {
            let logits = session
                .step(token)?
                .logits
                .ok_or("plan carries no output head — cannot score")?;
            if vocab == 0 {
                vocab = logits.len();
            } else if logits.len() != vocab {
                return Err(format!(
                    "{}: position {i}: vocabulary changed mid-sequence",
                    entry.id
                )
                .into());
            }
            for value in &logits {
                file.write_all(&value.to_le_bytes())?;
            }
        }
        file.flush()?;

        // The entry consumed exactly its own ids and nothing else.
        if session.position() != entry.ids.len() {
            return Err(format!(
                "{}: ended at position {} after {} ids",
                entry.id,
                session.position(),
                entry.ids.len()
            )
            .into());
        }
        positions += entry.ids.len();
        println!(
            "  {:>3}/{} {:<18} {:>4} positions",
            n + 1,
            entries.len(),
            entry.id,
            entry.ids.len()
        );
    }

    let secs = started.elapsed().as_secs_f64();
    println!(
        "banked {} entries, {positions} positions in {secs:.1} s ({:.0} ms/position)",
        entries.len(),
        secs * 1000.0 / positions.max(1) as f64
    );
    Ok(())
}
