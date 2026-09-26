//! CONTINUATION-VIEW-1 V2: the cost of reading rows through the view.
//!
//! V2 replaces the kernels' `step.keys[p]` slice index with one virtual
//! call per row fetch. This measures what that costs where it matters —
//! decode, where every step reads every held row — so the notes can say it
//! rather than assume it. Run the same test on main and on V2, alternating:
//!
//! ```text
//! LARQL_VIEW1_QWEN=~/chris-models/qwen3-0.6b.vindex3 \
//!   cargo test --release -p larql-vindex --test view_decode_cost -- --ignored --nocapture
//! ```

use std::time::Instant;

use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::kv::RowKvState;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prefill_prepared;
use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::plan_component_ops;

/// Positions held before decode starts: enough that every step's row
/// reads dominate the step.
const PREFILL: u32 = 512;
/// Decode steps timed.
const STEPS: u32 = 64;
/// Untimed decode steps first, so caches and pools are warm.
const WARMUP: u32 = 4;

#[test]
#[ignore = "real container: LARQL_VIEW1_QWEN"]
fn decode_ms_per_token_through_the_row_read_path() {
    let path = std::env::var("LARQL_VIEW1_QWEN").expect("set LARQL_VIEW1_QWEN");
    let root = std::path::Path::new(&path);
    let inspection = inspect_container(root, false).unwrap();
    let plan = plan_component_ops(&inspection, root, "target")
        .unwrap()
        .plan
        .unwrap();
    let store = OperandStore::open(root, &inspection).unwrap();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();

    let prompt: Vec<u32> = (1000..1000 + PREFILL).collect();
    let mut kv = RowKvState::default();
    prefill_prepared(&plan, &ops, &prompt, &backend, &mut kv).unwrap();
    let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv).unwrap();
    for token in 0..WARMUP {
        session.step(2000 + token).unwrap();
    }
    let start = Instant::now();
    let mut checksum = 0.0f64;
    for token in 0..STEPS {
        let logits = session.step(3000 + token).unwrap().logits.unwrap();
        checksum += logits.iter().map(|&x| f64::from(x)).sum::<f64>();
    }
    let per_token = start.elapsed().as_secs_f64() * 1e3 / f64::from(STEPS);
    // The checksum pins that both builds decoded the same thing.
    println!(
        "view_decode_cost: {per_token:.3} ms/token over {STEPS} steps after {PREFILL} positions; \
         logits checksum {checksum:.6e}"
    );
}
