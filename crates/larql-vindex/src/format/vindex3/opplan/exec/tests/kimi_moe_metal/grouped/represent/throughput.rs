//! The throughput half of Q1: does a smaller bank run faster?

use super::*;

/// **Q1 — the throughput half: does a smaller bank actually run faster?**
///
/// Fewer bytes only helps while the kernel stays bandwidth-bound. A
/// k-quant reads a fraction of BF16's bytes but does far more work per
/// byte — six-bit sub-block scales to unpack, mins to apply, nibbles to
/// split — so the whole quantisation thesis rests on an assumption that
/// has to be measured: that the dequantisation arithmetic stays cheaper
/// than the memory it saves.
///
/// Measured on the GPU window over a cache-defeating working set, on the
/// same banks the quality screen used, so bytes and answer come from one
/// experiment.
#[test]
fn report_expert_bank_representation_throughput() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    ramp_up(&metal);
    let (n, k) = Stage::Gate.shape(&fx);
    // Every bank built and HELD before any dispatch — see [`Arm`].
    let banks: Vec<(Format, Reencoded)> = Format::ALL
        .iter()
        .map(|&f| {
            let one = Reencoded::build(&fx, Stage::Gate, f);
            (f, one.replicated(THROUGHPUT_REPLICAS))
        })
        .collect();

    let mut rows = Vec::new();
    for (f, bank) in &banks {
        let slots = bank.offsets.len();
        let t = measure(BENCH_WARMUP, BENCH_ITERS, || {
            f.dispatch_profiled(
                &metal,
                &bank.bytes,
                &bank.offsets,
                &fx.x,
                n,
                k,
                InputLayout::Shared,
            )
            .1
        });
        // The simulated arm reads a bf16 carrier, so its GB/s would be
        // BF16's. Reported as unknown rather than as a number a reader
        // would take for a measurement of MXFP4.
        let effective = if f.native() { bank.bytes.len() } else { 0 };
        rows.push((*f, bank.bytes.len(), effective, slots, t));
    }

    let bf16 = rows[0].4.gpu_median_ms;
    eprintln!(
        "[q1-perf] gate projection, {} slots ({}x the real bank), n={n} k={k} — GPU window, \
         {BENCH_ITERS} iters",
        rows[0].3, THROUGHPUT_REPLICAS
    );
    eprintln!(
        "[q1-perf] {:<6} {:>9} {:>10} {:>7} {:>9} {:>8} {:>8}",
        "format", "MiB read", "gpu ms", "GB/s", "vs BF16", "spread", "bytes/x"
    );
    for (f, bank_bytes, effective, _, t) in &rows {
        eprintln!(
            "[q1-perf] {:<6} {:>9.1} {:>10.4} {:>7} {:>9.2} {:>8.2} {:>8.2}",
            f.label(),
            *bank_bytes as f64 / (1024.0 * 1024.0),
            t.gpu_median_ms,
            if *effective > 0 {
                format!(
                    "{:.0}",
                    *effective as f64 / (t.gpu_median_ms / 1000.0) / 1e9
                )
            } else {
                "n/a".to_string()
            },
            bf16 / t.gpu_median_ms,
            t.gpu_spread(),
            16.0 / f.bpw(),
        );
        assert!(
            t.gpu_spread() < 2.0,
            "{}: GPU timer spread {:.2} — the median and the minimum disagree enough that \
             neither is a floor of anything real",
            f.label(),
            t.gpu_spread()
        );
    }

    // The claim this test exists to test. A bandwidth-bound kernel gets
    // the full byte ratio; an arithmetic-bound one does not. Reported as
    // an efficiency rather than asserted, because which side of it Kimi's
    // shape lands on IS the result.
    for (f, _, _, _, t) in rows.iter().skip(1).filter(|(f, ..)| f.native()) {
        let byte_ratio = 16.0 / f.bpw();
        let speedup = bf16 / t.gpu_median_ms;
        eprintln!(
            "[q1-perf] {}: {:.2}x fewer bytes bought {:.2}x less GPU time — {:.0}% of the \
             bandwidth-bound ideal",
            f.label(),
            byte_ratio,
            speedup,
            100.0 * speedup / byte_ratio
        );
    }
}
