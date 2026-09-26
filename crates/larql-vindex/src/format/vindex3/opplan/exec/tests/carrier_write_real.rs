//! V3-OBS-1 on a REAL container (`docs/v3-obs-1-carrier-observation.md`):
//! P1–P3 and the batch/decode witness through `carrier_write`'s chain on
//! each requested backend, the CROSS-BACKEND comparison of the chains
//! those backends produced, and the P5 capture-cost measurement on the
//! production backend.
//!
//! Ignored by default because it needs a container on disk:
//!
//! ```sh
//! LARQL_V3_CONTAINER=~/chris-models/granite-4.2-3b.s6.vindex3 \
//!   cargo test --release -p larql-vindex --lib carrier_write_real -- --ignored --nocapture
//! ```
//!
//! `LARQL_V3_BACKENDS` (default `production,reference`) picks the arms;
//! `LARQL_V3_TOKENS` (comma-separated ids) overrides the prompt;
//! `LARQL_V3_COST_TRIALS` (default 5) sets the interleaved pairs after
//! one warm-up pair. Release build only: a debug build prices the CPU
//! stages, not the observer. Operands are prepared ONCE per backend and
//! every arm on that backend runs over the same resident image.
//!
//! The cross-backend claim has two parts with two strengths. The
//! STRUCTURE — event sequence, write count, sites, positions, layer
//! scales — must be identical, and is asserted. The VALUES come from two
//! backends that share no arithmetic, so they are measured (max abs,
//! relative RMS, bit-identical fraction) and reported; the one
//! assertion on them is a bug tripwire far above any accumulation-order
//! drift, not a tolerance the claim rests on.

use std::time::{Duration, Instant};

use super::carrier_write::{
    assert_batch_matches_chain, ffn_layers, writes_declared_by, ChainWitness,
};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::kv::RowKvState;
use crate::format::vindex3::opplan::exec::observe::{CarrierForm, NoopObserver, StepObserver};
use crate::format::vindex3::opplan::exec::observe_stats::{FixedBasis, StatsObserver};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::provenance::ExecutionProvenance;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::{
    execute_prepared_streaming, ExecutionTrace, PlaneEvent,
};
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

/// The default prompt when `LARQL_V3_TOKENS` is unset: ids every
/// vocabulary has.
const DEFAULT_TOKENS: std::ops::RangeInclusive<u32> = 1..=8;
const DEFAULT_TRIALS: usize = 5;
const DEFAULT_BACKENDS: &str = "production,reference";
/// The projection width the P5 treatment computes per write.
const COST_BASIS_DIMS: usize = 3;
const COST_BASIS_SEED: u64 = 0x5EED;
/// Relative RMS between two backends' carrier states above this is a
/// bug, not drift: f32 accumulation-order differences on a real stack
/// sit orders of magnitude below it. The claim itself rests on the
/// REPORTED numbers, never on this bound.
const CROSS_BACKEND_REL_RMS_ALARM: f64 = 1e-3;
/// Layers whose per-layer agreement is printed (every n-th), so the
/// depth profile of the drift is visible without forty lines.
const DEPTH_PROFILE_STRIDE: usize = 5;

fn env_tokens() -> Vec<u32> {
    match std::env::var("LARQL_V3_TOKENS") {
        Ok(list) => list
            .split(',')
            .map(|t| {
                t.trim()
                    .parse()
                    .expect("LARQL_V3_TOKENS is a comma-separated list of ids")
            })
            .collect(),
        Err(_) => DEFAULT_TOKENS.collect(),
    }
}

fn env_backends() -> Vec<String> {
    std::env::var("LARQL_V3_BACKENDS")
        .unwrap_or_else(|_| DEFAULT_BACKENDS.to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Every position's logits, stepping a fresh session over the prepared
/// image with `observer`.
fn step_all<B: PlanBackend>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    tokens: &[u32],
    observer: &mut dyn StepObserver,
) -> Vec<Vec<f32>> {
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(plan, ops, backend, &mut kv).unwrap();
    tokens
        .iter()
        .map(|&t| {
            session
                .step_observed(t, observer)
                .unwrap()
                .logits
                .expect("the plan carries an output head")
        })
        .collect()
}

/// The batch traversal over the same prepared image, assembled into the
/// trace shape the chain comparison reads.
fn batch_trace<B: PlanBackend>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    tokens: &[u32],
) -> ExecutionTrace {
    let mut embedded = None;
    let mut layers = Vec::new();
    let out = execute_prepared_streaming(plan, ops, tokens, backend, None, &mut |event| {
        match event {
            PlaneEvent::Embedded(plane) => embedded = Some(plane.clone()),
            PlaneEvent::Layer { index, trace } => {
                assert_eq!(index, layers.len(), "layers arrive in plan order");
                layers.push(trace);
            }
            _ => {}
        }
        Ok(())
    })
    .unwrap();
    ExecutionTrace {
        embedded: embedded.expect("the batch traversal embeds before layer 0"),
        executed_layers: (0..layers.len()).collect(),
        layers,
        exit: out.exit,
        logits: out.logits,
    }
}

fn wall_per_token<B: PlanBackend>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    tokens: &[u32],
    observer: &mut dyn StepObserver,
) -> Duration {
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(plan, ops, backend, &mut kv).unwrap();
    let clock = Instant::now();
    for &t in tokens {
        session.step_observed(t, observer).unwrap();
    }
    clock.elapsed() / u32::try_from(tokens.len()).unwrap()
}

/// The executor's own residency census, printed so a large image's run
/// records what was mapped and what was actually resident — evidence
/// about the execution, never an acceptance criterion of this rung.
fn print_residency(name: &str, ops: &PreparedOperands, when: &str) {
    let mapped = ops.mapped_residency();
    let census = ops.allocation_census();
    println!(
        "[{name}] residency {when}: {} mapped regions, {:.2} GiB mapped, {:.2} GiB resident; \
         {} allocations, {:.2} GiB allocated",
        mapped.regions,
        mapped.mapped_bytes as f64 / GIB,
        mapped.resident_bytes as f64 / GIB,
        census.allocations,
        census.bytes as f64 / GIB
    );
}

const GIB: f64 = 1_073_741_824.0;

/// One backend's complete evidence: the chain it produced, the logits
/// at every position, and the realizations it pinned.
struct Arm {
    name: String,
    witness: ChainWitness,
    logits: Vec<Vec<f32>>,
    /// What this arm RAN: pinned forms and the process arm, as the
    /// executor records them. Values are comparable across two arms
    /// only when these fingerprint identically.
    provenance: ExecutionProvenance,
}

/// P1–P3 and batch/decode on one backend, then P5 when `trials > 0`.
fn witness_on<B: PlanBackend>(
    name: &str,
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
    tokens: &[u32],
    trials: usize,
) -> Arm {
    let clock = Instant::now();
    let ops = PreparedOperands::load(plan, store, backend, ExecutionSlice::Full).unwrap();
    println!("[{name}] operands prepared once in {:?}", clock.elapsed());
    print_residency(name, &ops, "after preparation");
    let provenance = ExecutionProvenance::of(&ops);
    for class in &provenance.realizations {
        println!(
            "[{name}] realization: {} operands {} codec {} -> {}",
            class.operands,
            class.representation,
            class.codec.as_deref().unwrap_or("-"),
            class.form
        );
    }
    println!(
        "[{name}] lowering {:?}; cpu arithmetic arm (process-global) {:?}; kquant {:?}; execution fingerprint {}",
        provenance.lowering,
        provenance.arithmetic_arm,
        provenance.kquant_execution,
        provenance.fingerprint()
    );
    let expected = writes_declared_by(plan);

    let plain = step_all(plan, &ops, backend, tokens, &mut NoopObserver);
    let mut witness = ChainWitness::default();
    let seen = step_all(plan, &ops, backend, tokens, &mut witness);
    assert_eq!(plain, seen, "[{name}] P1: observation changed the logits");
    assert_eq!(
        witness.writes.len(),
        expected * tokens.len(),
        "[{name}] P3: write count"
    );
    assert_eq!(
        witness.writes_of(CarrierForm::Single),
        expected * tokens.len()
    );
    assert_eq!(
        witness.boundaries(true),
        plan.layers.len() * tokens.len(),
        "[{name}] FfnDone closes every layer"
    );
    assert_eq!(
        witness.exact,
        expected * tokens.len(),
        "[{name}] P2 inexact: {:?}",
        witness.inexact
    );
    println!(
        "[{name}] P1 PASS (bit-identical logits at {} positions), P2 PASS ({} exact writes), \
         P3 PASS ({} writes = forecast)",
        tokens.len(),
        witness.exact,
        witness.writes.len()
    );

    let trace = batch_trace(plan, &ops, backend, tokens);
    assert_batch_matches_chain(&trace, &witness, tokens);
    println!(
        "[{name}] batch/decode PASS (bit-identical carrier states at every layer and position)"
    );

    if trials > 0 {
        capture_cost(name, plan, &ops, backend, tokens, trials, expected);
    }
    print_residency(name, &ops, "after the witnesses");
    Arm {
        name: name.to_string(),
        witness,
        logits: seen,
        provenance,
    }
}

/// P5: interleaved arms over the prepared image, one warm-up pair
/// discarded, three estimators reported.
fn capture_cost<B: PlanBackend>(
    name: &str,
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    tokens: &[u32],
    trials: usize,
    expected: usize,
) {
    // The carrier width, read off the executor rather than assumed.
    let hidden = {
        let mut probe = ChainWitness::default();
        step_all(plan, ops, backend, &tokens[..1], &mut probe);
        probe.entering[0].len()
    };
    let basis = FixedBasis::seeded(hidden, COST_BASIS_DIMS, COST_BASIS_SEED).unwrap();
    let mut control = Vec::with_capacity(trials);
    let mut treatment = Vec::with_capacity(trials);
    for trial in 0..=trials {
        let (noop, stats) = if trial % 2 == 0 {
            let n = wall_per_token(plan, ops, backend, tokens, &mut NoopObserver);
            let mut s = StatsObserver::new(basis.clone(), None);
            let t = wall_per_token(plan, ops, backend, tokens, &mut s);
            assert_eq!(s.rows.len(), expected * tokens.len());
            (n, t)
        } else {
            let mut s = StatsObserver::new(basis.clone(), None);
            let t = wall_per_token(plan, ops, backend, tokens, &mut s);
            let n = wall_per_token(plan, ops, backend, tokens, &mut NoopObserver);
            (n, t)
        };
        if trial == 0 {
            println!("[{name}] warm-up pair discarded: noop {noop:?}, stats {stats:?}");
            continue;
        }
        control.push(noop);
        treatment.push(stats);
    }
    let mean = |v: &[Duration]| v.iter().sum::<Duration>() / u32::try_from(v.len()).unwrap();
    let median = |v: &[Duration]| {
        let mut sorted = v.to_vec();
        sorted.sort_unstable();
        sorted[sorted.len() / 2]
    };
    let signed = |a: Duration, b: Duration| match a.checked_sub(b) {
        Some(d) => format!("+{d:?}"),
        None => format!("-{:?}", b - a),
    };
    let (c, t) = (mean(&control), mean(&treatment));
    let (cm, tm) = (median(&control), median(&treatment));
    let (cmin, tmin) = (
        *control.iter().min().unwrap(),
        *treatment.iter().min().unwrap(),
    );
    // Three estimators, because they disagree when the machine is not
    // quiet: the mean carries every outlier, the median discards them,
    // and min-vs-min is the floor the observer's own arithmetic cannot
    // be below. Report all three; a cost claim rests on the one whose
    // spread the reader can see.
    println!(
        "[{name}] P5 capture cost, {} interleaved pairs, per token:\n  noop  mean {c:?}  median {cm:?}  min {cmin:?}  max {:?}\n  stats mean {t:?}  median {tm:?}  min {tmin:?}  max {:?}\n  stats - noop: mean {} (ratio {:.4}); median {} (ratio {:.4}); min {} (ratio {:.4})\n  basis {:?}",
        control.len(),
        control.iter().max().unwrap(),
        treatment.iter().max().unwrap(),
        signed(t, c),
        t.as_secs_f64() / c.as_secs_f64(),
        signed(tm, cm),
        tm.as_secs_f64() / cm.as_secs_f64(),
        signed(tmin, cmin),
        tmin.as_secs_f64() / cmin.as_secs_f64(),
        basis.identity()
    );
}

/// How far two vectors are apart, in the three quantities the existing
/// parity tests report.
#[derive(Clone, Copy, Debug, Default)]
struct Divergence {
    max_abs: f32,
    rel_rms: f64,
    bit_identical: bool,
}

fn diverge(a: &[f32], b: &[f32]) -> Divergence {
    assert_eq!(a.len(), b.len(), "vectors of different width");
    let mut max_abs = 0.0f32;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq_a = 0.0f64;
    let mut bit_identical = true;
    for (x, y) in a.iter().zip(b) {
        if x.to_bits() != y.to_bits() {
            bit_identical = false;
        }
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        sum_sq_diff += f64::from(d) * f64::from(d);
        sum_sq_a += f64::from(*x) * f64::from(*x);
    }
    let rel_rms = if sum_sq_a > 0.0 {
        (sum_sq_diff / sum_sq_a).sqrt()
    } else if sum_sq_diff > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };
    Divergence {
        max_abs,
        rel_rms,
        bit_identical,
    }
}

fn worse(a: Divergence, b: Divergence) -> Divergence {
    Divergence {
        max_abs: a.max_abs.max(b.max_abs),
        rel_rms: a.rel_rms.max(b.rel_rms),
        bit_identical: a.bit_identical && b.bit_identical,
    }
}

/// The cross-backend witness: identical structure asserted, value
/// agreement measured and reported with its depth profile.
fn compare(a: &Arm, b: &Arm, layers: usize) {
    let pair = format!("{} vs {}", a.name, b.name);
    assert_eq!(
        a.witness.events, b.witness.events,
        "[{pair}] the structural event streams differ"
    );
    assert_eq!(a.witness.entering.len(), b.witness.entering.len());
    assert_eq!(a.witness.writes.len(), b.witness.writes.len());
    assert_eq!(a.logits.len(), b.logits.len());

    let mut entering = Divergence {
        bit_identical: true,
        ..Divergence::default()
    };
    for (x, y) in a.witness.entering.iter().zip(&b.witness.entering) {
        entering = worse(entering, diverge(x, y));
    }
    let mut per_layer = vec![
        Divergence {
            bit_identical: true,
            ..Divergence::default()
        };
        layers
    ];
    let mut identical_writes = 0usize;
    let mut first_drift: Option<(usize, usize)> = None;
    for (x, y) in a.witness.writes.iter().zip(&b.witness.writes) {
        assert_eq!(
            (x.layer, x.site, x.position, x.layer_scale),
            (y.layer, y.site, y.position, y.layer_scale),
            "[{pair}] write identity differs"
        );
        let d = diverge(&x.after, &y.after);
        if d.bit_identical {
            identical_writes += 1;
        } else if first_drift.is_none() {
            first_drift = Some((x.layer, x.position));
        }
        per_layer[x.layer] = worse(per_layer[x.layer], d);
    }
    let mut logits = Divergence {
        bit_identical: true,
        ..Divergence::default()
    };
    for (x, y) in a.logits.iter().zip(&b.logits) {
        logits = worse(logits, diverge(x, y));
    }
    let worst = per_layer.iter().copied().fold(Divergence::default(), worse);
    let profile: Vec<String> = per_layer
        .iter()
        .enumerate()
        .filter(|(l, _)| l % DEPTH_PROFILE_STRIDE == 0 || *l + 1 == layers)
        .map(|(l, d)| format!("L{l}:{:.2e}", d.rel_rms))
        .collect();
    println!(
        "[{pair}] CROSS-BACKEND: structure identical ({} events, {} writes, {} positions)\n  entering carrier: max_abs {:.3e} rel_rms {:.3e} bit_identical {}\n  carrier writes: {}/{} bit-identical; first drift at {:?} (layer, position)\n  worst layer: max_abs {:.3e} rel_rms {:.3e}\n  depth profile (rel_rms): {}\n  final logits: max_abs {:.3e} rel_rms {:.3e} bit_identical {}",
        a.witness.events.len(),
        a.witness.writes.len(),
        a.logits.len(),
        entering.max_abs,
        entering.rel_rms,
        entering.bit_identical,
        identical_writes,
        a.witness.writes.len(),
        first_drift,
        worst.max_abs,
        worst.rel_rms,
        profile.join(" "),
        logits.max_abs,
        logits.rel_rms,
        logits.bit_identical
    );
    if a.provenance.fingerprint() == b.provenance.fingerprint() {
        // Same realizations, same arm: the only remaining difference is
        // summation order, and the tripwire applies.
        assert!(
            worst.rel_rms <= CROSS_BACKEND_REL_RMS_ALARM
                && logits.rel_rms <= CROSS_BACKEND_REL_RMS_ALARM,
            "[{pair}] same execution fingerprint, yet disagreement is above the bug tripwire \
             ({CROSS_BACKEND_REL_RMS_ALARM:e})"
        );
        println!("[{pair}] execution fingerprints IDENTICAL; value agreement is asserted at the tripwire");
    } else {
        // Different realizations or arms: the executor DECLARED that
        // these two arms compute differently, so the numbers above are
        // the measurement of that declaration and nothing here may call
        // them a bug. The structure assertion stands regardless.
        println!(
            "[{pair}] execution fingerprints DIFFER ({} vs {}); value agreement is REPORTED at the \
             realizations' precision, not asserted",
            a.provenance.fingerprint(),
            b.provenance.fingerprint()
        );
    }
}

#[test]
#[ignore = "needs a real VINDEX3 container in LARQL_V3_CONTAINER"]
fn real_container_witnesses_and_capture_cost() {
    let Ok(path) = std::env::var("LARQL_V3_CONTAINER") else {
        panic!("set LARQL_V3_CONTAINER to a container directory");
    };
    let root = std::path::Path::new(&path);
    let inspection = inspect_container(root, false).unwrap();
    let outcome = plan_component_ops(&inspection, root, "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(root, &inspection).unwrap();
    let tokens = env_tokens();
    let trials: usize = std::env::var("LARQL_V3_COST_TRIALS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_TRIALS);
    let expected = writes_declared_by(&plan);
    let backends = env_backends();
    println!(
        "{path}: {} layers ({} with an FFN program), forecast {expected} carrier writes per \
         token, {} tokens, backends {backends:?}",
        plan.layers.len(),
        ffn_layers(&plan),
        tokens.len()
    );

    let mut arms: Vec<Arm> = Vec::new();
    for name in &backends {
        let arm = match name.as_str() {
            "production" => witness_on(
                "production",
                &plan,
                &store,
                &ProductionBackend::new(),
                &tokens,
                trials,
            ),
            // The reference backend is naive f32 that shares no
            // arithmetic with larql-compute; its timing is not a
            // measurement of anything, so P5 never runs on it.
            "reference" => witness_on(
                "reference",
                &plan,
                &store,
                &ReferenceBackend::new(),
                &tokens,
                0,
            ),
            other => panic!("unknown backend `{other}` in LARQL_V3_BACKENDS"),
        };
        arms.push(arm);
    }
    for pair in arms.windows(2) {
        compare(&pair[0], &pair[1], plan.layers.len());
    }
}
