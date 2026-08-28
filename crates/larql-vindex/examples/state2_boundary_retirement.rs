//! STATE-2: can a context chunk's boundary stand in for its interior?
//!
//! One prompt carries a distinctive chunk C1. All arms share ONE
//! prefilled continuation state (reference backend end to end — no
//! substrate mixing anywhere); each retiring arm then declares a
//! different part of C1's interior retired and decodes the same steps,
//! teacher-forced along the canonical arm's greedy trajectory. Reported
//! per arm: per-step KL (bits, f64 log-softmax) of the arm's next-token
//! distribution against the canonical arm's, greedy-token agreement, and
//! the rows the retirement would reclaim.
//!
//! Arms at MATCHED footprint are the point: `boundary-k` retains the
//! LAST k positions of C1, `first-k` the first k, `random-k` a seeded k
//! interior positions. If boundary-k does not beat its matched controls,
//! "boundary" is not the operative object no matter how small the KL.
//! `zero` (retire all of C1) is the carries-nothing floor.
//!
//! Usage:
//!   state2_boundary_retirement <container> <spec.json> [steps] [retire_at]
//!
//! The spec is written by the tokenizer side (`make_specs.py`): tokens,
//! `c1_start`, `c1_end`. Tokens are never produced here — only one side
//! may choose the tokenizer.
//!
//! `retire_at` (a token index, normally `c1_end`) moves the retirement
//! EARLIER than the last prompt token: the shared prefill stops there,
//! each arm retires, and the remaining prompt tokens are consumed as
//! decode steps UNDER the arm's retirement. Without it, retirement
//! happens after the whole prompt — which tests whether decode still
//! needs C1's rows; with it, the question itself must read C1 through
//! the boundary, which is the chuk-mlx claim proper.

use std::ops::Range;
use std::time::Instant;

use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::kv::RowKvState;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prefill_prepared;
use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use larql_vindex::format::vindex3::opplan::exec::retire::RetiringKvState;
use larql_vindex::format::vindex3::opplan::plan_component_ops;

const DEFAULT_STEPS: usize = 24;
/// Matched-footprint ladder shared by boundary/first/random arms.
const RETAINED_LADDER: [usize; 3] = [1, 4, 16];
/// Seed for the random-retention control. Fixed so the run is a fact.
const RANDOM_SEED: u64 = 0xB0;

struct Spec {
    label: String,
    tokens: Vec<u32>,
    c1: Range<usize>,
}

fn read_spec(path: &str) -> Spec {
    let raw = std::fs::read_to_string(path).expect("spec file");
    let json: serde_json::Value = serde_json::from_str(&raw).expect("spec json");
    Spec {
        label: json["label"].as_str().expect("label").to_string(),
        tokens: json["tokens"]
            .as_array()
            .expect("tokens")
            .iter()
            .map(|t| t.as_u64().expect("token id") as u32)
            .collect(),
        c1: json["c1_start"].as_u64().expect("c1_start") as usize
            ..json["c1_end"].as_u64().expect("c1_end") as usize,
    }
}

/// Retired spans that retain exactly `keep` inside the interior.
fn retire_all_but(interior: &Range<usize>, keep: &[usize]) -> Vec<Range<usize>> {
    let mut kept: Vec<usize> = keep.to_vec();
    kept.sort_unstable();
    kept.dedup();
    let mut spans = Vec::new();
    let mut cursor = interior.start;
    for &position in &kept {
        assert!(
            interior.contains(&position),
            "retained {position} outside C1"
        );
        if cursor < position {
            spans.push(cursor..position);
        }
        cursor = position + 1;
    }
    if cursor < interior.end {
        spans.push(cursor..interior.end);
    }
    spans
}

/// Seeded LCG draw of `k` distinct interior positions — deterministic,
/// dependency-free, and printed with the arm so the draw is auditable.
fn random_positions(interior: &Range<usize>, k: usize, seed: u64) -> Vec<usize> {
    let width = interior.end - interior.start;
    let mut state = seed;
    let mut picked = Vec::new();
    while picked.len() < k {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let candidate = interior.start + ((state >> 33) as usize % width);
        if !picked.contains(&candidate) {
            picked.push(candidate);
        }
    }
    picked.sort_unstable();
    picked
}

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let log_z = logits
        .iter()
        .map(|&l| (l as f64 - max).exp())
        .sum::<f64>()
        .ln()
        + max;
    logits.iter().map(|&l| l as f64 - log_z).collect()
}

/// KL(a || b) in BITS from two logit rows.
fn kl_bits(a: &[f32], b: &[f32]) -> f64 {
    let (la, lb) = (log_softmax(a), log_softmax(b));
    la.iter()
        .zip(&lb)
        .map(|(&pa, &pb)| pa.exp() * (pa - pb))
        .sum::<f64>()
        / std::f64::consts::LN_2
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("non-empty logits")
        .0 as u32
}

/// Decode `inputs` over `state`, returning each step's logits.
fn decode_steps(
    plan: &larql_vindex::format::vindex3::opplan::ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &ReferenceBackend,
    state: &mut RetiringKvState,
    inputs: &[u32],
) -> Vec<Vec<f32>> {
    let mut session =
        DecodeSession::over_prepared(plan, ops, backend, state).expect("decode session");
    inputs
        .iter()
        .map(|&token| {
            session
                .step(token)
                .expect("decode step")
                .logits
                .expect("output head")
        })
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let container = args.next().expect("usage: <container> <spec.json> [steps]");
    let spec_path = args.next().expect("usage: <container> <spec.json> [steps]");
    let steps: usize = args
        .next()
        .map(|s| s.parse().expect("steps"))
        .unwrap_or(DEFAULT_STEPS);
    let retire_at: Option<usize> = args.next().map(|s| s.parse().expect("retire_at"));
    let spec = read_spec(&spec_path);
    let interior = spec.c1.clone();
    if let Some(at) = retire_at {
        assert!(
            at >= interior.end && at < spec.tokens.len(),
            "retire_at {at} must lie between c1_end {} and the prompt's end {}",
            interior.end,
            spec.tokens.len()
        );
    }

    let inspection = inspect_container(container.as_ref(), false).expect("inspect");
    let outcome =
        plan_component_ops(&inspection, container.as_ref(), "target").expect("plan_component_ops");
    let plan = outcome.plan.expect("closed plan");
    let store = OperandStore::open(container.as_ref(), &inspection).expect("operand store");
    let backend = ReferenceBackend::new();
    eprintln!("loading operands (f32)…");
    let ops =
        PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).expect("prepared");

    // ONE shared prefill up to the retirement point (default: everything
    // but the last prompt token). Every arm consumes the rest of the
    // prompt as decode steps itself, so those positions — and the first
    // reported distribution — are computed UNDER the arm's retirement.
    let split = retire_at.unwrap_or(spec.tokens.len() - 1);
    let (prefix, rest) = spec.tokens.split_at(split);
    let (consume, last) = (&rest[..rest.len() - 1], rest[rest.len() - 1]);
    eprintln!(
        "prefilling {} positions ({} consumed under retirement)…",
        prefix.len(),
        consume.len() + 1
    );
    let started = Instant::now();
    let mut base = RowKvState::default();
    prefill_prepared(&plan, &ops, prefix, &backend, &mut base).expect("prefill");
    eprintln!("prefill took {:.1}s", started.elapsed().as_secs_f64());

    // Canonical arm: greedy trajectory + per-step logits.
    let started = Instant::now();
    let mut canonical_state = RetiringKvState::from_state(base.clone());
    let mut canonical: Vec<Vec<f32>> = Vec::with_capacity(steps);
    let mut trajectory: Vec<u32> = Vec::with_capacity(steps);
    {
        let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut canonical_state)
            .expect("canonical session");
        for &token in consume {
            session.step(token).expect("canonical consume");
        }
        let mut next = last;
        for _ in 0..steps {
            let logits = session
                .step(next)
                .expect("canonical step")
                .logits
                .expect("output head");
            next = argmax(&logits);
            trajectory.push(next);
            canonical.push(logits);
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "spec": spec.label,
            "arm": "canonical",
            "retire_at": retire_at,
            "retained": interior.end - interior.start,
            "retired_rows": 0,
            "greedy_tokens": trajectory,
            "seconds": started.elapsed().as_secs_f64(),
        })
    );

    // Teacher-forced inputs shared by every retiring arm: the prompt
    // remainder consumed under retirement, then the canonical trajectory
    // minus its final token. Only the trailing `steps` logits are
    // measured — the remainder's logits are conditioning, not outcome.
    let mut forced = consume.to_vec();
    forced.push(last);
    forced.extend_from_slice(&trajectory[..steps - 1]);

    let mut arms: Vec<(String, Vec<usize>)> = vec![("zero".into(), Vec::new())];
    for k in RETAINED_LADDER {
        arms.push((
            format!("boundary-{k}"),
            (interior.end - k..interior.end).collect(),
        ));
        arms.push((
            format!("first-{k}"),
            (interior.start..interior.start + k).collect(),
        ));
        arms.push((
            format!("random-{k}"),
            random_positions(&interior, k, RANDOM_SEED),
        ));
    }

    for (name, retained) in arms {
        let spans = retire_all_but(&interior, &retained);
        let started = Instant::now();
        let mut state = RetiringKvState::from_state(base.clone());
        for span in &spans {
            state.retire(span.clone()).expect("coherent retirement");
        }
        let retired_rows = state.retired_positions();
        let all_logits = decode_steps(&plan, &ops, &backend, &mut state, &forced);
        let logits = &all_logits[consume.len()..];
        let kl: Vec<f64> = logits
            .iter()
            .zip(&canonical)
            .map(|(arm, canon)| kl_bits(canon, arm))
            .collect();
        let matched = logits
            .iter()
            .zip(&trajectory)
            .filter(|(l, &t)| argmax(l) == t)
            .count();
        let first_divergence = logits
            .iter()
            .zip(&trajectory)
            .position(|(l, &t)| argmax(l) != t);
        println!(
            "{}",
            serde_json::json!({
                "spec": spec.label,
                "arm": name,
                "retire_at": retire_at,
                "retained": retained.len(),
                "retained_positions": retained,
                "retired_rows": retired_rows,
                "kl_bits": kl,
                "kl_sum_bits": kl.iter().sum::<f64>(),
                "kl_mean_bits": kl.iter().sum::<f64>() / kl.len() as f64,
                "tokens_matched": matched,
                "steps": steps,
                "first_divergence": first_divergence,
                "seconds": started.elapsed().as_secs_f64(),
            })
        );
    }
}
