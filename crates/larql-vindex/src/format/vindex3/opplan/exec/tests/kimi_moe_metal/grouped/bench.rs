//! Measurement: what the grouped dispatch achieves, and the controls that decide whether those numbers mean anything.
//!
//! See [`super`] for the hypothesis this rung tests and what the
//! grouped dispatch does and does not change.

use larql_compute_metal::trait_impl::bf16_grouped::GroupedShape;
use larql_compute_metal::trait_impl::grouped_experts::{ExpertOffset, InputLayout};

use super::*;

/// The measurement this rung exists for: achieved bandwidth per stage,
/// grouped against the two rung-1 shapes.
///
/// Three arms per stage, identical bytes and identical arithmetic:
///   * **sequential** — one command buffer per expert (rung 1's naive
///     shape);
///   * **one CB** — N dispatches batched into a single command buffer
///     (rung 1's 4.68x result), which removes the submission tax but
///     leaves each dispatch at 128 threadgroups;
///   * **grouped** — one 2-D dispatch of `row_tiles x slots`, which is
///     the only arm that raises the threadgroup count.
///
/// The gap between `one CB` and `grouped` is the occupancy answer, with
/// submission cost already removed from both.
#[test]
fn report_achieved_bandwidth_per_stage() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let gbs = |bytes: f64, ms: f64| bytes / 1e6 / ms;
    let slots = fx.experts.len();
    let rows_per_tg = metal.bf16_grouped_experts_pipeline.rows_per_tg;

    // Every bank built up front and held for the whole test — see the
    // aliasing note on `Bank`. Building them inside the loop, where each
    // is dropped before the next is allocated, silently measured one
    // stage twice.
    let banks: Vec<(Stage, Bank)> = [Stage::Gate, Stage::Up, Stage::Down]
        .into_iter()
        .map(|s| (s, Bank::build(&fx, s)))
        .collect();
    let bank_for = |want: Stage| -> &Bank {
        &banks
            .iter()
            .find(|(s, _)| *s == want)
            .expect("built above")
            .1
    };

    // The submission floor, through the same entry point: a grouped
    // dispatch whose bytes cannot explain its cost. Both batched arms
    // pay it exactly once, so subtracting it turns a stack-level number
    // into a kernel-level one without trusting the GPU timer.
    //
    // Re-measured before each stage and kept as a RUNNING MINIMUM.
    // Submission cost is a property of the stack, not of the stage, so
    // more samples give a better estimate of the same quantity — and the
    // failure mode of a single sample is asymmetric: one unlucky high
    // floor over-subtracts and inflates the kernel figure it feeds. Both
    // tails were seen before this: a stale high floor printed `inf GB/s`
    // for a negative kernel time, and a 0.587 ms first-stage floor
    // inflated a 79 GB/s arm to 370.
    let tiny_w = vec![0u8; TINY_ROWS * TINY_COLS * 2];
    let tiny_off = [ExpertOffset(0)];
    let tiny_x = vec![1.0f32; TINY_COLS];
    let submission_floor = || {
        measure(BENCH_WARMUP, BENCH_ITERS, || {
            let (out, gpu) = metal
                .bf16_grouped_experts_profiled(
                    &tiny_w,
                    &tiny_off,
                    &tiny_x,
                    TINY_ROWS,
                    TINY_COLS,
                    InputLayout::Shared,
                )
                .expect("floor dispatch");
            std::hint::black_box(out);
            gpu
        })
        .wall_min_ms
    };

    // The down stage needs real per-slot activations, so form them the
    // way the FFN does rather than feeding it arbitrary numbers.
    let grouped_shared = |bank: &Bank, n: usize, k: usize| {
        metal
            .bf16_grouped_experts(&bank.bytes, &bank.offsets, &fx.x, n, k, InputLayout::Shared)
            .expect("seed dispatch")
    };
    let h = down_inputs(
        &fx,
        &grouped_shared(bank_for(Stage::Gate), fx.inter, fx.hidden),
        &grouped_shared(bank_for(Stage::Up), fx.inter, fx.hidden),
    );

    eprintln!(
        "[grouped] {slots} slots, hidden={} inter={}, rows_per_tg={rows_per_tg}",
        fx.hidden, fx.inter,
    );
    let mut floor_ms = f64::INFINITY;
    for (stage, bank) in &banks {
        let stage = *stage;
        floor_ms = floor_ms.min(submission_floor());
        let (n, k) = stage.shape(&fx);
        let x: &[f32] = match stage.layout() {
            InputLayout::Shared => &fx.x,
            InputLayout::PerSlot => &h,
        };
        let bytes = bank.bytes.len() as f64;
        let tiles = (n as u64).div_ceil(rows_per_tg);

        let batch: Vec<(&[u8], usize, usize)> =
            fx.experts.iter().map(|e| (stage.matrix(e), n, k)).collect();

        let seq_ms = min_ms(BENCH_WARMUP, BENCH_ITERS, || {
            for (slot, &(w, n, k)) in batch.iter().enumerate() {
                let xs = match stage.layout() {
                    InputLayout::Shared => &x[..k],
                    InputLayout::PerSlot => &x[slot * k..(slot + 1) * k],
                };
                std::hint::black_box(metal.bf16_gemv_force(w, xs, n, k));
            }
        });
        // The two batched arms, INTERLEAVED. Both pay exactly one
        // submission, so what separates them is the dispatch shape —
        // but only if they were measured on the same machine, which
        // blocked arms cannot promise here. `bf16_gemv_multi` shares one
        // input across every matrix, so it can only serve the
        // shared-input stages; the down stage has no one-CB arm short of
        // the grouped kernel itself, which is part of the point.
        let grouped_arm = || {
            let (out, gpu) = metal
                .bf16_grouped_experts_profiled(&bank.bytes, &bank.offsets, x, n, k, stage.layout())
                .expect("grouped dispatch");
            std::hint::black_box(out);
            gpu
        };
        eprintln!(
            "[grouped] {} [{n},{k}] {:.1} MiB  grid {tiles}x{slots} = {} TGs \
             (per-expert: {tiles} TGs)",
            stage.label(),
            bytes / (1024.0 * 1024.0),
            tiles * slots as u64,
        );
        eprintln!(
            "[grouped]   sequential  wall {seq_ms:.3} ms {:>6.1} GB/s  ({slots} submissions)",
            gbs(bytes, seq_ms)
        );
        match stage.layout() {
            InputLayout::Shared => {
                let r = interleave(
                    BENCH_WARMUP,
                    BENCH_ITERS,
                    || {
                        let (out, gpu) = metal
                            .encode_bf16_gemv_multi_profiled(&batch, &x[..k])
                            .expect("batched submission");
                        std::hint::black_box(out);
                        gpu
                    },
                    grouped_arm,
                );
                eprintln!(
                    "[grouped]   one CB    gpu {:.3} ms {:>6.1} GB/s   ({slots} dispatches \
                     of {tiles} TGs, one submission)",
                    r.a_median_ms,
                    gbs(bytes, r.a_median_ms),
                );
                eprintln!(
                    "[grouped]   GROUPED   gpu {:.3} ms {:>6.1} GB/s   (1 dispatch of {} TGs)",
                    r.b_median_ms,
                    gbs(bytes, r.b_median_ms),
                    tiles * slots as u64,
                );
                eprintln!(
                    "[grouped]   grouped vs one CB {:.2}x GPU-busy   [ramp {:.2}x — 1.00 \
                     means the machine held still]",
                    r.a_median_ms / r.b_median_ms,
                    r.ramp,
                );
            }
            InputLayout::PerSlot => {
                let t = measure(BENCH_WARMUP, BENCH_ITERS, grouped_arm);
                eprintln!("[grouped]   one CB    n/a (per-slot inputs)");
                eprintln!(
                    "[grouped]   GROUPED   gpu {:.3} ms {:>6.1} GB/s   wall {:.3} ms \
                     (spread {:.2}x)",
                    t.gpu_median_ms,
                    gbs(bytes, t.gpu_median_ms),
                    t.wall_min_ms,
                    t.gpu_spread(),
                );
            }
        }
        eprintln!(
            "[grouped]   bank staging (repack, once, not per token) {:.3} ms",
            bank.stage_ms,
        );
    }
    assert!(slots > 1, "grouping needs more than one expert");
}

/// **The control that decides whether the bandwidth numbers above mean
/// anything, and the cold reading of the occupancy trend.**
///
/// One stage's bank is 40.5 MiB and this machine's system level cache is
/// of that order, so a loop that dispatches the SAME bank fifteen times
/// may be measuring the cache. The hot arm has been seen bimodal —
/// ~172 GB/s in one run and ~285 GB/s in the next at identical geometry
/// — which is the signature of residency luck rather than of the memory
/// system.
///
/// So: build enough DISTINCT banks to exceed any plausible cache,
/// dispatch each exactly once per cycle, and report the mean over a full
/// cycle. Every dispatch has identical geometry to the production shape,
/// so the only thing that changed is whether the bytes could still be
/// resident. Production is the cold case: a token reads ~3 GB of active
/// expert weights across 27 layers and the next token routes elsewhere,
/// so no layer's experts are ever warm.
///
/// **Both shapes, because they differ in threadgroup count and that is
/// the variable under study.** gate is `[1024, 2304]` → 128 tiles × 9 =
/// 1152 threadgroups; down is `[2304, 1024]` → 288 × 9 = 2592. If
/// occupancy is still the limiter, the wider one is faster per byte —
/// and that comparison is only worth anything cold.
#[test]
fn report_cold_bandwidth_against_a_working_set_that_defeats_the_cache() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let gbs = |bytes: f64, ms: f64| bytes / 1e6 / ms;
    let rows_per_tg = metal.bf16_grouped_experts_pipeline.rows_per_tg;

    // Templates for every shape, and the per-slot activations the down
    // stage consumes, all built before any bank is cloned — see the
    // aliasing note on `Bank`, which bites hardest here because every
    // allocation is the same size.
    let templates: Vec<(Stage, Bank)> = [Stage::Gate, Stage::Up, Stage::Down]
        .into_iter()
        .map(|st| (st, Bank::build(&fx, st)))
        .collect();
    let grouped_shared = |bank: &Bank| {
        metal
            .bf16_grouped_experts(
                &bank.bytes,
                &bank.offsets,
                &fx.x,
                fx.inter,
                fx.hidden,
                InputLayout::Shared,
            )
            .expect("seed dispatch")
    };
    let h = down_inputs(
        &fx,
        &grouped_shared(&templates[0].1),
        &grouped_shared(&templates[1].1),
    );

    // Every clone of every shape held alive at once, for the same
    // reason. ~1 GB, which this machine has.
    let cloned: Vec<(Stage, Vec<Vec<u8>>)> = [Stage::Gate, Stage::Down]
        .into_iter()
        .map(|st| {
            let template = &templates.iter().find(|(s, _)| *s == st).unwrap().1;
            let copies = COLD_WORKING_SET_BYTES.div_ceil(template.bytes.len()).max(2);
            (st, (0..copies).map(|_| template.bytes.clone()).collect())
        })
        .collect();

    // **Both shapes interleaved**, not one then the other. Ramping the
    // GPU first removes most of the position effect, but not all of it:
    // a blocked cold control kept reporting the narrow shape at half the
    // wide one's bandwidth after ramp-up, while an interleaved probe put
    // them within ~13%. Whatever the residue is — clock, wired-page
    // state, allocation age — the fix is the same, so measure them
    // against each other rather than one after the other.
    /// One shape's whole measurement setup: which stage, the offsets
    /// that address it, the cold banks, and the input it consumes.
    struct Arm<'a> {
        stage: Stage,
        template: &'a Bank,
        banks: &'a [Vec<u8>],
        n: usize,
        k: usize,
        x: &'a [f32],
    }
    let plan: Vec<Arm<'_>> = cloned
        .iter()
        .map(|(stage, banks)| {
            let template = &templates.iter().find(|(s, _)| s == stage).unwrap().1;
            let (n, k) = stage.shape(&fx);
            Arm {
                stage: *stage,
                template,
                banks,
                n,
                k,
                x: match stage.layout() {
                    InputLayout::Shared => &fx.x,
                    InputLayout::PerSlot => &h,
                },
            }
        })
        .collect();

    let run = |a: &Arm<'_>, bank: &[u8]| -> f64 {
        let (out, gpu) = metal
            .bf16_grouped_experts_profiled(
                bank,
                &a.template.offsets,
                a.x,
                a.n,
                a.k,
                a.stage.layout(),
            )
            .expect("grouped dispatch");
        std::hint::black_box(out);
        gpu
    };

    // Warm every (shape, bank) pair once: the first touch of a fresh
    // mapping pays faults that are not memory bandwidth either.
    for a in &plan {
        for b in a.banks {
            run(a, b);
        }
    }
    let cycles = plan[0].banks.len();
    let mut cold = vec![0.0f64; plan.len()];
    let mut hot = vec![0.0f64; plan.len()];
    for c in 0..cycles {
        for (i, a) in plan.iter().enumerate() {
            cold[i] += run(a, &a.banks[c]);
            // The hot arm rides the same cycle so the two share a
            // machine: one bank, revisited, against a bank the previous
            // cycles did not touch.
            hot[i] += run(a, &a.banks[0]);
        }
    }

    for (i, a) in plan.iter().enumerate() {
        let (stage, template, banks, n, k) = (a.stage, a.template, a.banks, a.n, a.k);
        let per_bank = template.bytes.len() as f64;
        let (cold_ms, hot_ms) = (cold[i] / cycles as f64, hot[i] / cycles as f64);
        let tiles = (n as u64).div_ceil(rows_per_tg);
        eprintln!(
            "[cold] {} [{n},{k}]  {} banks x {:.1} MiB = {:.0} MiB working set  \
             grid {tiles}x{} = {} TGs",
            stage.label(),
            banks.len(),
            per_bank / (1024.0 * 1024.0),
            per_bank * cycles as f64 / (1024.0 * 1024.0),
            template.slots(),
            tiles * template.slots() as u64,
        );
        eprintln!(
            "[cold]   COLD (a bank not touched this cycle)  gpu {cold_ms:.3} ms {:>6.1} GB/s",
            gbs(per_bank, cold_ms),
        );
        eprintln!(
            "[cold]   HOT  (the same bank every cycle)      gpu {hot_ms:.3} ms {:>6.1} GB/s",
            gbs(per_bank, hot_ms),
        );
        eprintln!("[cold]   hot/cold on GPU-busy: {:.2}x", cold_ms / hot_ms);
        assert!(
            cold_ms.is_finite() && cold_ms > 0.0,
            "the cold arm must run"
        );
    }
}

/// **The geometry sweep, run cold.** How many rows a threadgroup covers,
/// against achieved bandwidth, at both real projection shapes.
///
/// Earned by the control above, which found 171 GB/s at 1152
/// threadgroups and 339 at 2592 — same bytes, same kernel body, twice
/// the memory system from more independent work. If that reading is
/// right, the fix for the skinny `[1024, 2304]` gate/up shape is not a
/// cleverer kernel but a finer tiling: `r=4` doubles its launch to 2304
/// threadgroups, which is where `down` already sits.
///
/// So this is a prediction under test, not an exploration. Every variant
/// computes bit-identical values (pinned in
/// `every_row_tiling_computes_the_same_values`), so anything that moves
/// here is scheduling.
///
/// Cold throughout — each dispatch reads a bank the previous one did not
/// touch. A hot sweep would measure how well each tiling exploits a
/// cache that production never gets to keep.
#[test]
fn sweep_row_tiling_cold() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let gbs = |bytes: f64, ms: f64| bytes / 1e6 / ms;

    let templates: Vec<(Stage, Bank)> = [Stage::Gate, Stage::Up, Stage::Down]
        .into_iter()
        .map(|st| (st, Bank::build(&fx, st)))
        .collect();
    let seed = |bank: &Bank| {
        metal
            .bf16_grouped_experts(
                &bank.bytes,
                &bank.offsets,
                &fx.x,
                fx.inter,
                fx.hidden,
                InputLayout::Shared,
            )
            .expect("seed dispatch")
    };
    let h = down_inputs(&fx, &seed(&templates[0].1), &seed(&templates[1].1));

    // Every clone of every shape alive at once — see the aliasing note
    // on `Bank`, which bites hardest here because every allocation is
    // the same size.
    let cloned: Vec<(Stage, Vec<Vec<u8>>)> = [Stage::Gate, Stage::Down]
        .into_iter()
        .map(|st| {
            let t = &templates.iter().find(|(s, _)| *s == st).unwrap().1;
            let copies = COLD_WORKING_SET_BYTES.div_ceil(t.bytes.len()).max(2);
            (st, (0..copies).map(|_| t.bytes.clone()).collect())
        })
        .collect();

    for (stage, banks) in &cloned {
        let stage = *stage;
        let template = &templates.iter().find(|(s, _)| *s == stage).unwrap().1;
        let (n, k) = stage.shape(&fx);
        let x: &[f32] = match stage.layout() {
            InputLayout::Shared => &fx.x,
            InputLayout::PerSlot => &h,
        };
        let per_bank = template.bytes.len() as f64;
        let slots = template.slots() as u64;
        eprintln!(
            "[sweep] {} [{n},{k}]  {} slots  cold over {} banks x {:.1} MiB",
            stage.label(),
            slots,
            banks.len(),
            per_bank / (1024.0 * 1024.0),
        );

        let mut best: Option<(u64, f64)> = None;
        for handle in &metal.bf16_grouped_variants {
            let run = |bank: &[u8]| -> f64 {
                let (out, gpu) = metal
                    .bf16_grouped_experts_tiled(
                        handle,
                        bank,
                        &template.offsets,
                        x,
                        GroupedShape {
                            n,
                            k,
                            layout: stage.layout(),
                        },
                    )
                    .expect("tiled dispatch");
                std::hint::black_box(out);
                gpu
            };
            for b in banks {
                run(b);
            }
            let gpu_mean: f64 = banks.iter().map(|b| run(b)).sum::<f64>() / banks.len() as f64;
            let tiles = (n as u64).div_ceil(handle.rows_per_tg);
            let bw = gbs(per_bank, gpu_mean);
            eprintln!(
                "[sweep]   r{:<2} ({:>3} threads/tg)  grid {:>4}x{slots} = {:>5} TGs   \
                 gpu {gpu_mean:.3} ms {:>6.1} GB/s",
                handle.rows_per_tg,
                handle.threads_per_tg,
                tiles,
                tiles * slots,
                bw,
            );
            if best.is_none_or(|(_, b)| bw > b) {
                best = Some((handle.rows_per_tg, bw));
            }
        }
        let (rows, bw) = best.expect("at least one variant");
        eprintln!("[sweep]   best: r{rows} at {bw:.1} GB/s");
    }
}

/// **What the sweep ruled out, and the probe that separates what is
/// left.**
///
/// The cold control found the two projection shapes 2x apart at
/// identical byte volume — `[1024, 2304]` at ~170 GB/s against
/// `[2304, 1024]` at ~340. The obvious reading was occupancy, so the
/// sweep tested it directly: `r=4` gives the narrow shape 2304
/// threadgroups, essentially what the wide one already has at `r=8`
/// (2592). It stayed at ~170. Tiling is roughly flat within each shape
/// and does not cross the gap. **Threadgroup count is not the
/// explanation.**
///
/// Two candidates survive, and the shapes confound them:
///   * the **geometry** — `K` is the reduction length each simdgroup
///     streams (2304 vs 1024 codes, so 4608- vs 2048-byte rows), and `N`
///     is how many such streams exist;
///   * the **input layout** — gate/up share one activation across every
///     slot (`XSTRIDE = 0`), down gives each slot its own.
///
/// So: run both banks under both layouts. `PerSlot` on the gate bank
/// computes numbers no model wants, which is fine — this measures bytes
/// per second, and the values only have to be real floats.
///
/// **Interleaved, not blocked.** This machine drifts within a session,
/// so the arms are cycled `A,B,C,D,A,B,C,D` and each arm's samples are
/// spread across the whole run. Blocked arms would attribute drift to
/// whichever ran last, which is exactly what the row sweep's own
/// r8-then-r4-then-r2-then-r1 ordering leaves open.
#[test]
fn separate_geometry_from_input_layout_cold() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let gbs = |bytes: f64, ms: f64| bytes / 1e6 / ms;

    let narrow = Bank::build(&fx, Stage::Gate); // [inter, hidden]
    let wide = Bank::build(&fx, Stage::Down); // [hidden, inter]
    let slots = narrow.slots();
    let per_bank = narrow.bytes.len();
    assert_eq!(per_bank, wide.bytes.len(), "arms must read equal bytes");

    let copies = COLD_WORKING_SET_BYTES.div_ceil(per_bank).max(2);
    let narrow_banks: Vec<Vec<u8>> = (0..copies).map(|_| narrow.bytes.clone()).collect();
    let wide_banks: Vec<Vec<u8>> = (0..copies).map(|_| wide.bytes.clone()).collect();

    // Inputs long enough for the per-slot arm of each geometry; the
    // shared arm reads the first K of the same buffer.
    let x_narrow: Vec<f32> = (0..slots * fx.hidden)
        .map(|i| ((i as f32) * 0.017).sin() * 0.5)
        .collect();
    let x_wide: Vec<f32> = (0..slots * fx.inter)
        .map(|i| ((i as f32) * 0.023).cos() * 0.5)
        .collect();

    struct Arm<'a> {
        label: &'a str,
        banks: &'a [Vec<u8>],
        offsets: &'a [ExpertOffset],
        x: &'a [f32],
        n: usize,
        k: usize,
        layout: InputLayout,
    }
    let arms = [
        Arm {
            label: "narrow [1024,2304] shared  ",
            banks: &narrow_banks,
            offsets: &narrow.offsets,
            x: &x_narrow,
            n: fx.inter,
            k: fx.hidden,
            layout: InputLayout::Shared,
        },
        Arm {
            label: "narrow [1024,2304] per-slot",
            banks: &narrow_banks,
            offsets: &narrow.offsets,
            x: &x_narrow,
            n: fx.inter,
            k: fx.hidden,
            layout: InputLayout::PerSlot,
        },
        Arm {
            label: "wide   [2304,1024] shared  ",
            banks: &wide_banks,
            offsets: &wide.offsets,
            x: &x_wide,
            n: fx.hidden,
            k: fx.inter,
            layout: InputLayout::Shared,
        },
        Arm {
            label: "wide   [2304,1024] per-slot",
            banks: &wide_banks,
            offsets: &wide.offsets,
            x: &x_wide,
            n: fx.hidden,
            k: fx.inter,
            layout: InputLayout::PerSlot,
        },
    ];

    let run = |a: &Arm, bank: &[u8]| -> f64 {
        let (out, gpu) = metal
            .bf16_grouped_experts_profiled(bank, a.offsets, a.x, a.n, a.k, a.layout)
            .expect("probe dispatch");
        std::hint::black_box(out);
        gpu
    };

    // Warm every (arm, bank) pair once so no arm pays first-touch.
    for a in &arms {
        for b in a.banks {
            run(a, b);
        }
    }
    let mut totals = [0.0f64; 4];
    for cycle in 0..copies {
        for (i, a) in arms.iter().enumerate() {
            totals[i] += run(a, &a.banks[cycle]);
        }
    }
    eprintln!(
        "[probe] {slots} slots, {copies} cold banks x {:.1} MiB each, arms interleaved",
        per_bank as f64 / (1024.0 * 1024.0)
    );
    for (i, a) in arms.iter().enumerate() {
        let mean = totals[i] / copies as f64;
        eprintln!(
            "[probe]   {}  gpu {mean:.3} ms {:>6.1} GB/s",
            a.label,
            gbs(per_bank as f64, mean),
        );
    }
    assert!(totals.iter().all(|t| t.is_finite() && *t > 0.0));
}
