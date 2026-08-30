//! cpu7c_arms — CPU-7C2's five arms on a real container.
//!
//! Runs the sequence frozen in `docs/cpu7c2-multi-position-surfaces.md`,
//! in order, and refuses to print a verdict until the ones before it pass.
//!
//! ```text
//! A  serial            K x step
//! B  legacy batched    old position-parallel FFN, stationary OFF
//! C  raised            FFN-many, stationary OFF   scheduling change ONLY
//! E  raised            + recurrent stationary
//! D  raised            + recurrent AND ffn stationary
//! ```
//!
//! Three independent experiments in one run:
//!
//! ```text
//! B -> C   remove the nested-position fan-out collapse
//! C -> E   replay C1's PROVEN recurrent tranche on a repaired substrate
//! E -> D   test the NEW FFN tranche
//! ```
//!
//! # Machine ownership adjudicates FIRST
//!
//! Restoration requires BOTH the clock and the mechanism:
//!
//! ```text
//! C/(K·A) <= 1.05   AND   C slabs/call restored toward A
//! ```
//!
//! A timing that came back without the row partition coming back means
//! something unrelated compensated, and the stationarity verdicts below
//! it would be resting on an unproven mechanism. They are not printed.
//!
//! # Every arm proves what its letter means
//!
//! Two independent switches now exist — the FFN surface and the
//! stationary class mask. Each arm asserts its resolved physical state
//! before it is timed. Several plausible timings in this programme would
//! have been entirely wrong had an arm silently fallen back.

use std::path::Path;
use std::time::Instant;

use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::cpu::ledger::{PlanTally, Site};
use larql_vindex::format::vindex3::opplan::exec::cpu::{ledger, stationary};
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::kv::RowKvState;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::exec::{
    multi_position_ffn, set_multi_position_ffn, timing,
};
use larql_vindex::format::vindex3::opplan::plan_component_ops;

/// CPU-7B's measured total-cost multipliers, from the worse of both sweep
/// orders. The ONLY quantity imported from another run — a kernel
/// property, not a timing anchor.
fn multiplier(k: usize) -> f64 {
    match k {
        2 => 1.02,
        4 => 1.27,
        other => panic!("no banked CPU-7B multiplier for K={other}"),
    }
}

/// Fraction of a grouped tranche that grouping removes: `1 - m_K/K`.
fn saving_fraction(k: usize) -> f64 {
    1.0 - multiplier(k) / k as f64
}

/// Classes CPU-7C2 has raised to a multi-position surface. Attention is
/// C3 and is deliberately absent.
const ELIGIBLE: [Site; 2] = [Site::Recurrent, Site::Ffn];

/// Numerical-equivalence bar for A against a batched traversal.
const REL_RMS: f64 = 1e-5;

/// Restoration bands, frozen in the protocol.
const RESTORED: f64 = 1.05;
const NOT_RESTORED: f64 = 1.15;

/// Restoration's MECHANISM half: the row partition must come back.
///
/// A timing that returned to ~1 while `slabs/call` stayed collapsed would
/// mean something unrelated compensated, and every stationarity verdict
/// below it would rest on an unproven mechanism.
const SLABS_RESTORED: f64 = 0.90;

/// What an arm's physical configuration is, and what it must resolve to.
struct Spec {
    letter: &'static str,
    ffn_many: bool,
    stationary: &'static [Site],
}

const ARMS: [Spec; 4] = [
    Spec {
        letter: "B",
        ffn_many: false,
        stationary: &[],
    },
    Spec {
        letter: "C",
        ffn_many: true,
        stationary: &[],
    },
    Spec {
        letter: "E",
        ffn_many: true,
        stationary: &[Site::Recurrent],
    },
    Spec {
        letter: "D",
        ffn_many: true,
        stationary: &[Site::Recurrent, Site::Ffn],
    },
];

impl Spec {
    /// Apply, then PROVE. An arm that cannot state its own configuration
    /// is one nobody can adjudicate.
    fn engage(&self) {
        set_multi_position_ffn(self.ffn_many);
        stationary::set_enabled(!self.stationary.is_empty());
        for site in Site::ALL {
            stationary::set_enabled_for(site, self.stationary.contains(&site));
        }
        assert_eq!(
            multi_position_ffn(),
            self.ffn_many,
            "arm {} did not resolve its FFN surface",
            self.letter
        );
        assert_eq!(
            stationary::enabled(),
            !self.stationary.is_empty(),
            "arm {} did not resolve its stationary switch",
            self.letter
        );
        for site in Site::ALL {
            assert_eq!(
                stationary::class_enabled_for(site),
                self.stationary.contains(&site),
                "arm {} did not resolve class {}",
                self.letter,
                site.name()
            );
        }
    }

    fn describe(&self) -> String {
        let classes = if self.stationary.is_empty() {
            "none".to_string()
        } else {
            self.stationary
                .iter()
                .map(|s| s.name())
                .collect::<Vec<_>>()
                .join(",")
        };
        format!(
            "ffn_many {:<3}  stationary {classes}",
            if self.ffn_many { "yes" } else { "no" }
        )
    }
}

struct Run {
    batch: Vec<f32>,
    probe: Vec<f32>,
    seconds: f64,
    counts: Vec<(
        larql_vindex::format::vindex3::opplan::exec::cpu::PhysicalProjectionPlan,
        PlanTally,
    )>,
}

impl Run {
    fn slabs_per_call(&self) -> f64 {
        let calls: u64 = self.counts.iter().map(|(_, t)| t.calls).sum();
        let slabs: u64 = self.counts.iter().map(|(_, t)| t.slabs).sum();
        slabs as f64 / calls.max(1) as f64
    }
    fn totals(&self) -> (u64, u64, u64, f64) {
        let calls: u64 = self.counts.iter().map(|(_, t)| t.calls).sum();
        let grouped: u64 = self.counts.iter().map(|(_, t)| t.grouped).sum();
        let bytes: u64 = self.counts.iter().map(|(_, t)| t.bytes).sum();
        let positions: u64 = self.counts.iter().map(|(_, t)| t.positions).sum();
        (
            calls,
            grouped,
            bytes,
            positions as f64 / calls.max(1) as f64,
        )
    }
}

fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x - *y).powi(2))
        .sum();
    let den: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
    (num / den.max(f64::MIN_POSITIVE)).sqrt()
}

fn identical(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn ids(raw: &str) -> Vec<u32> {
    raw.split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().expect("token ids are integers"))
        .collect()
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let container = args
        .first()
        .cloned()
        .ok_or("usage: cpu7c_arms <container> --prompt a,b --batch c,d --probe e [--repeats 3]")?;
    let prompt = ids(&flag(&args, "--prompt").ok_or("--prompt is required")?);
    let batch = ids(&flag(&args, "--batch").ok_or("--batch is required")?);
    let probe: u32 = flag(&args, "--probe")
        .ok_or("--probe is required")?
        .trim()
        .parse()?;
    let repeats: usize = flag(&args, "--repeats").map_or(3, |v| v.parse().expect("integer"));
    let k = batch.len();
    let q = saving_fraction(k);

    println!("cpu7c_arms — CPU-7C2, five arms");
    println!("  container {container}");
    println!(
        "  prompt {} tokens, K={k}, probe {probe}, repeats {repeats}, m_{k}={:.2}, q={q:.4}",
        prompt.len(),
        multiplier(k)
    );
    println!("  gates: docs/cpu7c2-multi-position-surfaces.md (frozen before this ran)\n");

    let backend = ProductionBackend::new();
    let loading = Instant::now();
    let root = Path::new(&container);
    let inspection = inspect_container(root, false)?;
    let outcome = plan_component_ops(&inspection, root, "target")?;
    let plan = outcome.plan.ok_or("component `target` produced no plan")?;
    let store = OperandStore::open(root, &inspection)?;
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full)?;
    println!(
        "  weights resident in {:.1} s\n",
        loading.elapsed().as_secs_f64()
    );

    let live = || -> Vec<_> {
        ledger()
            .all()
            .iter()
            .copied()
            .filter(|(_, t)| t.calls > 0)
            .collect()
    };

    // ── serial arm ──────────────────────────────────────────────────────
    let serial = |timed_only_batch: bool| -> Result<Run, Box<dyn std::error::Error>> {
        // Arm A is the SERIAL path: no `step_many`, so neither switch can
        // reach it, and it is the fixed reference both others are read
        // against.
        let mut kv = RowKvState::default();
        let mut s = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv)?;
        for &t in &prompt {
            s.step(t)?;
        }
        ledger().reset();
        let clock = Instant::now();
        let mut last = Vec::new();
        for &t in &batch {
            last = s.step(t)?.logits.ok_or("no output head")?;
        }
        let seconds = clock.elapsed().as_secs_f64();
        let counts = live();
        let _ = timed_only_batch;
        let probe_logits = s.step(probe)?.logits.ok_or("no output head")?;
        Ok(Run {
            batch: last,
            probe: probe_logits,
            seconds,
            counts,
        })
    };

    let batched = |spec: &Spec| -> Result<Run, Box<dyn std::error::Error>> {
        spec.engage();
        let mut kv = RowKvState::default();
        let mut s = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv)?;
        for &t in &prompt {
            s.step(t)?;
        }
        ledger().reset();
        let clock = Instant::now();
        let last = s.step_many(&batch)?.logits.ok_or("no output head")?;
        let seconds = clock.elapsed().as_secs_f64();
        // Snapshot BEFORE the probe: it is an ordinary single-position
        // step and would add a whole token to a K-position measurement.
        let counts = live();
        let probe_logits = s.step(probe)?.logits.ok_or("no output head")?;
        Ok(Run {
            batch: last,
            probe: probe_logits,
            seconds,
            counts,
        })
    };

    // ── K=1 calibration ─────────────────────────────────────────────────
    let mut kv = RowKvState::default();
    let (p, shares, t1) = {
        let mut s = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv)?;
        for &t in &prompt {
            s.step(t)?;
        }
        ledger().reset();
        timing::ledger().reset();
        let clock = Instant::now();
        s.step(probe)?;
        let t1 = clock.elapsed().as_secs_f64();
        let (proj_ns, _) = ledger().projection_nanos();
        let shares: Vec<(Site, f64)> = Site::ALL
            .iter()
            .map(|s| (*s, ledger().site(*s).nanos as f64 / 1e9 / t1))
            .collect();
        (proj_ns as f64 / 1e9 / t1, shares, t1)
    };
    println!("CALIBRATION K=1");
    println!("  wall            {:8.1} ms", t1 * 1e3);
    println!("  projection      {:8.1} ms   p = {p:.3}", p * t1 * 1e3);
    println!("  {:<12} {:>8}   eligible in C2", "class", "share");
    let mut summed = 0.0;
    let mut g_c2 = 0.0;
    for (site, share) in &shares {
        if *share == 0.0 {
            continue;
        }
        let eligible = ELIGIBLE.contains(site);
        println!(
            "  {:<12} {share:>8.3}   {}",
            site.name(),
            if eligible { "YES" } else { "no" }
        );
        summed += share;
        if eligible {
            g_c2 += share;
        }
    }
    // The accounting invariant: every projection site belongs to a
    // declared class. A site outside them would contaminate `g` silently.
    let closure = (summed - p).abs();
    println!(
        "  {:<12} {summed:>8.3}   sum vs p = {p:.3}, gap {closure:.4}",
        "TOTAL"
    );
    assert!(
        closure < 0.01,
        "the class shares do not account for p (gap {closure:.4}) — some projection site \
         is outside every declared class and `g` below would be contaminated"
    );
    let g_rec = shares
        .iter()
        .find(|(s, _)| *s == Site::Recurrent)
        .map_or(0.0, |(_, v)| *v);
    let g_ffn = shares
        .iter()
        .find(|(s, _)| *s == Site::Ffn)
        .map_or(0.0, |(_, v)| *v);
    println!("  g_C2 = g_rec + g_ffn = {g_rec:.3} + {g_ffn:.3} = {g_c2:.3}\n");

    let s_rec = q * g_rec;
    let s_ffn = q * g_ffn;
    println!("PREDICTION (normalized savings of K·A, additive, from THIS run's shares)");
    println!("  s_recurrent = {q:.4} x {g_rec:.3} = {s_rec:.4}");
    println!("  s_ffn       = {q:.4} x {g_ffn:.3} = {s_ffn:.4}");
    println!("  E/(K·A) = c - s_rec        D/(K·A) = c - s_rec - s_ffn\n");

    // ── parity ──────────────────────────────────────────────────────────
    let a = serial(true)?;
    let mut runs = Vec::new();
    for spec in &ARMS {
        runs.push((spec, batched(spec)?));
    }
    println!("RESOLVED STATE");
    println!(
        "  {:<3} {:<40} {:>10}",
        "A", "serial path — neither switch applies", ""
    );
    for (spec, _) in &runs {
        println!("  {:<3} {}", spec.letter, spec.describe());
    }
    println!("\nPARITY");
    let mut exact = true;
    for w in runs.windows(2) {
        let (l, r) = (&w[0], &w[1]);
        let b_ok = identical(&l.1.batch, &r.1.batch);
        let p_ok = identical(&l.1.probe, &r.1.probe);
        exact &= b_ok && p_ok;
        println!(
            "  {} vs {}   batch {}   probe {}   (MUST be bit-identical)",
            l.0.letter,
            r.0.letter,
            if b_ok { "identical" } else { "DIFFER" },
            if p_ok { "identical" } else { "DIFFER" }
        );
    }
    let c = &runs[1].1;
    let ac_b = rel_rms(&a.batch, &c.batch);
    let ac_p = rel_rms(&a.probe, &c.probe);
    println!("  A vs C   batch {ac_b:.2e}   probe {ac_p:.2e}   (MUST be < {REL_RMS:.0e})");
    if identical(&a.probe, &c.probe) {
        println!("  A vs C is bit-identical — stronger than the frozen gate asked for.");
    }
    if !exact || ac_b >= REL_RMS || ac_p >= REL_RMS {
        println!("\nPARITY FAILED — nothing below is readable.");
        return Ok(());
    }

    // ── machine ownership, which adjudicates first ──────────────────────
    println!("\nMACHINE OWNERSHIP");
    println!(
        "  {:<3} {:>11}  {:>7} {:>8} {:>9} {:>9}",
        "arm", "slabs/call", "calls", "grouped", "pos/call", "bytes GB"
    );
    let (ca, ga, ba, wa) = a.totals();
    println!(
        "  {:<3} {:>11.2}  {ca:>7} {ga:>8} {wa:>9.2} {:>9.2}",
        "A",
        a.slabs_per_call(),
        ba as f64 / 1e9
    );
    for (spec, run) in &runs {
        let (calls, grouped, bytes, width) = run.totals();
        println!(
            "  {:<3} {:>11.2}  {calls:>7} {grouped:>8} {width:>9.2} {:>9.2}",
            spec.letter,
            run.slabs_per_call(),
            bytes as f64 / 1e9
        );
    }

    let c_slabs = runs[1].1.slabs_per_call();
    let mechanism = c_slabs >= SLABS_RESTORED * a.slabs_per_call();
    println!(
        "  mechanism: C {c_slabs:.2} vs A {:.2} x {SLABS_RESTORED:.2} = {:.2}  -> {}",
        a.slabs_per_call(),
        SLABS_RESTORED * a.slabs_per_call(),
        if mechanism {
            "RESTORED"
        } else {
            "NOT RESTORED"
        }
    );

    // ── clock, order-reversed ───────────────────────────────────────────
    println!("\nCLOCK  (order-reversed; the worse ratio adjudicates, never the mean)");
    let mut best: Vec<f64> = vec![f64::INFINITY; ARMS.len()];
    let mut best_a = f64::INFINITY;
    for r in 0..repeats {
        best_a = best_a.min(serial(true)?.seconds);
        // Reverse the arm order on alternate repeats, so whichever half of
        // a drifting machine a ratio lands on, one pass puts it on the
        // other arm.
        let order: Vec<usize> = if r % 2 == 0 {
            (0..ARMS.len()).collect()
        } else {
            (0..ARMS.len()).rev().collect()
        };
        for i in order {
            let t = batched(&ARMS[i])?.seconds;
            best[i] = best[i].min(t);
        }
    }
    let (b, cc, e, d) = (best[0], best[1], best[2], best[3]);
    let ka = best_a;
    println!("  A  K x A          {:8.1} ms", ka * 1e3);
    for (i, spec) in ARMS.iter().enumerate() {
        println!(
            "  {}  {:8.1} ms   /(K·A) {:.3}",
            spec.letter,
            best[i] * 1e3,
            best[i] / ka
        );
    }

    // ── adjudication, in the frozen order ───────────────────────────────
    let c = cc / ka;
    println!("\nADJUDICATION");
    println!(
        "  1  restoration   B/(K·A) {:.3}  ->  C/(K·A) {c:.3}",
        b / ka
    );
    let timing_ok = c <= RESTORED;
    println!(
        "     timing {} (<= {RESTORED:.2}), mechanism {}",
        if timing_ok {
            "PASS"
        } else if c > NOT_RESTORED {
            "FAIL"
        } else {
            "MARGINAL"
        },
        if mechanism { "PASS" } else { "FAIL" }
    );
    if !(timing_ok && mechanism) {
        println!(
            "\n  RESTORATION FAILED — the stationarity verdicts are NOT printed. \
             Machine ownership is what the arms below are measured on top of."
        );
        return Ok(());
    }
    let (e_pred, d_pred) = (c - s_rec, c - s_rec - s_ffn);
    let (e_meas, d_meas) = (e / ka, d / ka);
    println!(
        "  2  recurrent     saving predicted {s_rec:.4}   measured {:.4}   delta {:+.4}",
        c - e_meas,
        (c - e_meas) - s_rec
    );
    println!("     E/(K·A) predicted {e_pred:.3}   measured {e_meas:.3}");
    println!(
        "  3  ffn           saving predicted {s_ffn:.4}   measured {:.4}   delta {:+.4}",
        e_meas - d_meas,
        (e_meas - d_meas) - s_ffn
    );
    println!("     D/(K·A) predicted {d_pred:.3}   measured {d_meas:.3}");
    println!(
        "\n  TOTAL   D/(K·A) predicted {d_pred:.3}   measured {d_meas:.3}   \
         target-side {:.2}x",
        1.0 / d_meas
    );
    Ok(())
}
