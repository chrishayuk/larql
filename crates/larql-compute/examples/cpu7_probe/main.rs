//! cpu7_probe — does the CPU have memory-level parallelism, and can one
//! weight traversal serve more than one token?
//!
//! Two pre-registered arms from `docs/cpu7-parallelism-protocol.md`, which
//! is the authority on every band quoted below. The gates in that file were
//! frozen before this probe produced a number and are restated here only so
//! the output is readable without it.
//!
//! ```text
//! CPU-7A   two independent projections, one activation, one worker budget
//!          -> is 127 GB/s a DRAM wall or a single-traversal wall?
//! CPU-7B   one projection, N activation vectors, weights loaded ONCE
//!          -> can a traversal serve more than one position?
//! ```
//!
//! **What a PASS here does and does not license.** This is a synthetic
//! probe on production-shaped data, not the real decode; the banked
//! synthetic->real correction on this programme is x1.047 and nothing here
//! supports a tok/s claim. A 7B pass says the kernel amortises on this
//! shape at this operating point. It does not say a hybrid architecture can
//! supply N independent positions — that is CPU-7C, and Qwen3.8's GatedDelta
//! recurrence is exactly why it is a separate rung.
//!
//! Usage:
//!   cargo run --release -p larql-compute --example cpu7_probe
//!
//! Run it on AC power on a quiet machine. Both arms are RATIO measurements,
//! where a thermal or contention artifact lands directly on the verdict
//! rather than on a number someone can sanity-check.

mod arms;
mod fixture;
mod kernel;

use arms::{arm_7a, control_c3, sweep_7b, Cell, Spread, REPEATS};
use fixture::{activations, Bank};

/// Qwen3.8's hidden size, and the input axis of every projection the
/// 27 GB/token figure is dominated by.
const IN_DIM: usize = 5120;

/// Output rows per matrix. Square, like the attention projections.
const ROWS: usize = 5120;

/// **CPU-7B2.** The geometries a real Qwen3.8 layer is actually made of,
/// with CPU-7B's original square shape first as the CONTROL.
///
/// CPU-7B measured `m_2 = 1.02` on ONE shape and CPU-7C2 then failed to
/// realise any saving on a layer dominated by these others. So the
/// question this sweep asks is whether the amortisation multiplier is a
/// property of the KERNEL or of the GEOMETRY — and the square row must
/// reproduce 1.02, or the probe changed rather than the shape.
const GEOMETRIES: [(&str, usize, usize); 4] = [
    ("control 5120x5120", 5120, 5120),
    ("ffn up/gate", 17408, 5120),
    ("ffn down", 5120, 17408),
    ("gd qkv 10240x5120", 10240, 5120),
];

/// Matrices in the DRAM bank. Eight 5120x5120 Q8 matrices is ~223 MB —
/// several times any cache on this part, which is the point: one matrix is
/// 26.2 MB and would sit largely in the system-level cache, letting a
/// weight-stationary kernel win against a cache decode never faces.
const DRAM_MATRICES: usize = 8;

/// Rows in the cache-resident control matrix. 512 x 5120 Q8 is 2.8 MB —
/// inside the 4 MB P-core L2, so this arm cannot be DRAM-bound.
const CACHE_ROWS: usize = 512;

/// Bytes each cell streams per repeat, held constant across working-set
/// sizes so both arms run for a comparable duration and amortise thread
/// spawn identically. Without this the cache arm measures its own setup.
const STREAMED_BYTES_TARGET: usize = 2 << 30;

/// The banked rate for this kernel, and the tolerance C1 allows.
const BANKED_Q8XQ8_GBS: f64 = 118.0;
const C1_TOLERANCE: f64 = 0.10;

/// How far the cache arm must outrun the DRAM arm before the DRAM number
/// may be quoted as attainable rather than as a floor.
const C2_MIN_RATIO: f64 = 1.5;

/// CPU-7B, frozen: cost at N relative to N=1.
const GATE_N2_AMORTISING: f64 = 1.35;
const GATE_N2_DEAD: f64 = 1.70;
const GATE_N4_STRONG: f64 = 1.90;
const GATE_N8_STRONG: f64 = 3.20;

/// C4, frozen: how far a multiplier may move between the forward and
/// reversed sweeps before the ordering artifact is live.
const C4_MAX_DRIFT: f64 = 0.15;

/// CPU-7A, frozen: aggregate GB/s bands for the concurrent arm.
/// MALFORMED — an absolute band against an anchor measured elsewhere.
/// Retained for historical completeness and never adjudicated; the
/// interpretation is the same-run ratio printed beside it.
const GATE_7A_NOTHING: f64 = 125.0;
const GATE_7A_MODEST: f64 = 140.0;
const GATE_7A_SIGNIFICANT: f64 = 160.0;

fn passes_for(bank: &Bank) -> usize {
    (STREAMED_BYTES_TARGET / bank.bytes()).max(1)
}

fn row(c: &Cell, baseline_ms: Option<f64>) {
    let mult = baseline_ms.map_or(String::from("     —"), |b| {
        format!("{:6.2}x", c.best_ms / b)
    });
    println!(
        "  {:<18} {:9.2} {:9.2} {:>8} {:9.1} {:11.2} {:9.1} {:8.1}",
        c.label,
        c.best_ms,
        c.median_ms,
        mult,
        c.weight_gbs(),
        c.transient_bytes as f64 / 1e9,
        c.sdots_scaling as f64 / 1e9,
        c.sdots_fixed as f64 / 1e9,
    );
}

fn main() {
    if !kernel::has_dotprod() {
        eprintln!(
            "cpu7_probe needs the aarch64 `dotprod` SDOT this programme is built on.\n\
             Refusing to run: a portable fallback would measure a different kernel \n\
             and the bands in docs/cpu7-parallelism-protocol.md would not apply to it."
        );
        std::process::exit(1);
    }
    let cores = std::thread::available_parallelism().map_or(8, |n| n.get());
    // Half the performance cores, matching the shipped executor's policy —
    // measured there as the flat basin, not chosen here.
    let workers = 6;

    println!("cpu7_probe — memory-level parallelism and weight-stationary execution");
    println!("  shape {IN_DIM}x{ROWS}  Q8[64] x asym-Q8[16]  workers={workers}  cores={cores}  repeats={REPEATS}");
    println!("  gates: docs/cpu7-parallelism-protocol.md (frozen before this ran)");
    println!();

    print!("building fixtures ... ");
    let dram = Bank::new(DRAM_MATRICES, ROWS, IN_DIM, 0x5EED_C7A1);
    let cache = Bank::new(1, CACHE_ROWS, IN_DIM, 0x5EED_C7A2);
    let acts = activations(8, IN_DIM, 0xAC71_0001);
    println!(
        "dram bank {:.1} MB, cache bank {:.1} MB, 8 activations",
        dram.bytes() as f64 / 1e6,
        cache.bytes() as f64 / 1e6
    );
    println!();

    // ── controls, before any arm is readable ────────────────────────────
    println!("CONTROLS");
    let mut controls = control_c3(&dram, &acts);
    let dram_passes = passes_for(&dram);
    let cache_passes = passes_for(&cache);

    let d1 = sweep_7b::<1>("dram", &dram, &acts, workers, dram_passes);
    let c1_rate = d1.weight_gbs();
    let c1_ok = (c1_rate - BANKED_Q8XQ8_GBS).abs() / BANKED_Q8XQ8_GBS <= C1_TOLERANCE;
    controls.insert(
        0,
        arms::Control {
            name: "C1   single-projection rate reproduces the banked kernel",
            passed: c1_ok,
            detail: format!(
                "{c1_rate:.1} GB/s against {BANKED_Q8XQ8_GBS:.1} banked, tolerance {:.0}%",
                C1_TOLERANCE * 100.0
            ),
        },
    );

    let k1 = sweep_7b::<1>("cache", &cache, &acts, workers, cache_passes);
    let ratio = k1.weight_gbs() / c1_rate;
    controls.insert(
        1,
        arms::Control {
            name: "C2   cache arm outruns DRAM arm (else DRAM is a floor)",
            passed: ratio >= C2_MIN_RATIO,
            detail: format!(
                "{:.1} GB/s cache / {c1_rate:.1} GB/s dram = {ratio:.2}x against {C2_MIN_RATIO:.1}x",
                k1.weight_gbs()
            ),
        },
    );

    let mut all_passed = true;
    for c in &controls {
        println!("  [{}] {}", if c.passed { "PASS" } else { "FAIL" }, c.name);
        println!("         {}", c.detail);
        all_passed &= c.passed;
    }
    println!();
    if !all_passed {
        println!("A control failed. Per the protocol, no arm below is readable — the");
        println!("probe is measuring itself. Arms are still printed so the failure can");
        println!("be diagnosed, and MUST NOT be quoted as results.");
        println!();
    }

    // ── CPU-7B ──────────────────────────────────────────────────────────
    println!("CPU-7B — weight-stationary, DRAM-resident bank");
    println!(
        "  {:<18} {:>9} {:>9} {:>8} {:>9} {:>11} {:>9} {:>8}",
        "arm", "best ms", "med ms", "vs N=1", "wt GB/s", "transient GB", "SDOT/vec", "SDOT w.1"
    );
    let d2 = sweep_7b::<2>("dram", &dram, &acts, workers, dram_passes);
    let d4 = sweep_7b::<4>("dram", &dram, &acts, workers, dram_passes);
    let d8 = sweep_7b::<8>("dram", &dram, &acts, workers, dram_passes);
    let base = d1.best_ms;
    row(&d1, None);
    for c in [&d2, &d4, &d8] {
        row(c, Some(base));
    }
    println!();

    // C4. The same sweep, largest N first. `N = 8` measuring last in the
    // forward pass is exactly where thermal or power drift would inflate a
    // multiplier and look like a real bend; running it FIRST puts the drift
    // on the other cell, so a curve that survives both orders is not one.
    println!("C4 — the same sweep, reversed (largest N first)");
    let r8 = sweep_7b::<8>("dram rev", &dram, &acts, workers, dram_passes);
    let r4 = sweep_7b::<4>("dram rev", &dram, &acts, workers, dram_passes);
    let r2 = sweep_7b::<2>("dram rev", &dram, &acts, workers, dram_passes);
    let r1 = sweep_7b::<1>("dram rev", &dram, &acts, workers, dram_passes);
    let rbase = r1.best_ms;
    row(&r1, None);
    for c in [&r2, &r4, &r8] {
        row(c, Some(rbase));
    }
    println!();

    println!(
        "  {:<8} {:>9} {:>9} {:>9}",
        "N", "forward", "reversed", "drift"
    );
    let mut worst_drift: f64 = 0.0;
    for (n, f, r) in [(2, &d2, &r2), (4, &d4, &r4), (8, &d8, &r8)] {
        let (mf, mr) = (f.best_ms / base, r.best_ms / rbase);
        worst_drift = worst_drift.max((mf - mr).abs());
        println!("  {n:<8} {mf:8.2}x {mr:8.2}x {:9.2}", (mf - mr).abs());
    }
    let c4_ok = worst_drift <= C4_MAX_DRIFT;
    println!(
        "  [{}] C4   worst drift {worst_drift:.2} against {C4_MAX_DRIFT:.2}",
        if c4_ok { "PASS" } else { "FAIL" }
    );
    all_passed &= c4_ok;
    println!();

    println!("CPU-7B — cache-resident control (same sweep, L2-resident)");
    let k2 = sweep_7b::<2>("cache", &cache, &acts, workers, cache_passes);
    let k4 = sweep_7b::<4>("cache", &cache, &acts, workers, cache_passes);
    let k8 = sweep_7b::<8>("cache", &cache, &acts, workers, cache_passes);
    let kbase = k1.best_ms;
    row(&k1, None);
    for c in [&k2, &k4, &k8] {
        row(c, Some(kbase));
    }
    println!();

    // ── CPU-7A ──────────────────────────────────────────────────────────
    // ── CPU-7B2: the same sweep, per production geometry ────────────────
    println!("CPU-7B2 — amortisation by matrix geometry (control first)");
    println!(
        "  {:<20} {:>9} {:>9} {:>9} {:>9}",
        "shape", "N=1 ms", "N=2 x", "N=4 x", "wt GB/s"
    );
    for (label, out_rows, in_dim) in GEOMETRIES {
        // Bank sized to the same total bytes whatever the shape, so every
        // row is DRAM-resident and none is measuring its own working set.
        let per_matrix = out_rows * in_dim + out_rows * (in_dim / 64) * 4;
        let count = (DRAM_MATRICES * ROWS * IN_DIM / per_matrix).max(2);
        let bank = Bank::new(count, out_rows, in_dim, 0x005E_EDB2);
        let acts = activations(4, in_dim, 0x00AC_71B2);
        let passes = (STREAMED_BYTES_TARGET / bank.bytes()).max(1);
        let g1 = sweep_7b::<1>(label, &bank, &acts, workers, passes);
        let g2 = sweep_7b::<2>(label, &bank, &acts, workers, passes);
        let g4 = sweep_7b::<4>(label, &bank, &acts, workers, passes);
        println!(
            "  {label:<20} {:9.2} {:9.3} {:9.3} {:9.1}",
            g1.best_ms,
            g2.best_ms / g1.best_ms,
            g4.best_ms / g1.best_ms,
            g1.weight_gbs()
        );
    }
    println!();

    println!("CPU-7A — independent projections, same worker budget");
    let one = arm_7a(
        "one matrix",
        &dram,
        &acts,
        workers,
        dram_passes,
        Spread::One,
    );
    let seq = arm_7a(
        "two, sequential",
        &dram,
        &acts,
        workers,
        dram_passes,
        Spread::Sequential,
    );
    let conc = arm_7a(
        "two, concurrent",
        &dram,
        &acts,
        workers,
        dram_passes,
        Spread::Concurrent,
    );
    println!(
        "  {:<18} {:>9} {:>9} {:>8} {:>9}",
        "arm", "best ms", "med ms", "vs one", "agg GB/s"
    );
    for c in [&one, &seq, &conc] {
        println!(
            "  {:<18} {:9.2} {:9.2} {:>8} {:9.1}",
            c.label,
            c.best_ms,
            c.median_ms,
            format!("{:6.2}x", c.best_ms / one.best_ms),
            c.weight_gbs()
        );
    }
    println!();

    // ── verdicts against the frozen bands ───────────────────────────────
    println!("VERDICTS (bands frozen in docs/cpu7-parallelism-protocol.md)");
    // The WORSE of the two orders, per C4: a bend that appears in either
    // pass is the one the programme has to plan around.
    let m2 = (d2.best_ms / base).max(r2.best_ms / rbase);
    let m4 = (d4.best_ms / base).max(r4.best_ms / rbase);
    let m8 = (d8.best_ms / base).max(r8.best_ms / rbase);
    println!(
        "  7B  N=2 {m2:.2}x   N=4 {m4:.2}x   N=8 {m8:.2}x   (per-vector {:.2}x / {:.2}x / {:.2}x)",
        m2 / 2.0,
        m4 / 4.0,
        m8 / 8.0
    );
    if !all_passed {
        println!("  NO VERDICT — a control failed, so the probe is measuring itself.");
        println!("  The multipliers above are diagnostic only and MUST NOT be quoted.");
        return;
    }
    let verdict_7b = if m2 > GATE_N2_DEAD {
        "DEAD — the kernel is not weight-bound; speculation is worth much less"
    } else if m4 <= GATE_N4_STRONG && m8 <= GATE_N8_STRONG {
        "STRONG — earns CPU-7C, the layer-shaped multi-position harness"
    } else if m2 <= GATE_N2_AMORTISING {
        "AMORTISING at N=2, bends earlier than predicted — report the bend"
    } else {
        "INCONCLUSIVE against the frozen bands — no rung is earned"
    };
    println!("      {verdict_7b}");

    let agg = conc.weight_gbs();
    let verdict_7a = if agg <= GATE_7A_NOTHING {
        "NOTHING AVAILABLE — 127 is a DRAM wall; close the concurrency lever"
    } else if agg <= GATE_7A_MODEST {
        "modest — worth a scheduling change only if it is free"
    } else if agg <= GATE_7A_SIGNIFICANT {
        "significant — schedule independent physical operators concurrently"
    } else {
        "the wall was single-GEMV concurrency — reopens executor scheduling"
    };
    println!("  7A  concurrent aggregate {agg:.1} GB/s");
    println!("      by the frozen band (MALFORMED, not adjudicated): {verdict_7a}");
    println!(
        "      same-run ratio to a single traversal: {:.2}x  <- the honest reading",
        conc.weight_gbs() / one.weight_gbs()
    );
}
