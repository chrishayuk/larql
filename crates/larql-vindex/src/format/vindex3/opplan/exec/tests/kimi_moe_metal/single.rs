//! Rung 1: one BF16 expert FFN, three separate dispatches.
//!
//! Controls first, then parity against the checkpoint's own output,
//! then what it costs. See [`super`] for what the gate does and does
//! not license.

use super::*;

/// **Control, before parity.** The same real expert bytes read through
/// the f16 kernel must not reproduce the oracle.
///
/// Without this, `REL_TOLERANCE` would be an untested number: a kernel
/// that decoded bf16 codes as `half` might still land inside a loose
/// ceiling on weights that happen to be small, and the gate would be
/// blind to the one confusion this whole rung exists to avoid.
#[test]
fn the_f16_decode_on_the_same_bytes_is_rejected() {
    let Some(dir) = fixture_dir() else {
        eprintln!("skipped: set {FIXTURE_ENV} to the exported fixture directory");
        return;
    };
    let Some(metal) = MetalBackend::new() else {
        eprintln!("skipped: no Metal device on this host");
        return;
    };
    let fx = load(&dir);
    let e = &fx.experts[0];

    let want = metal
        .bf16_gemv_force(&e.gate, &fx.x, fx.inter, fx.hidden)
        .expect("bf16 arm dispatches");
    let wrong = metal
        .f16_gemv_force(&e.gate, &fx.x, fx.inter, fx.hidden)
        .expect("f16 arm dispatches on the same bytes");

    let rel = rel_err(&wrong, &want);
    assert!(
        rel > 1e-2,
        "reading real bf16 expert weights as f16 scored rel {rel:e} — this gate \
         cannot tell the two decodes apart, so its tolerance means nothing"
    );
    eprintln!("[control] f16-on-bf16-bytes rel {rel:e} (tolerance {REL_TOLERANCE:e})");
}

/// **Control, before parity.** A different expert's weights must fail
/// the same comparison the parity test passes.
///
/// The failure this rules out is a gate satisfied by something other
/// than the weights — an all-zero dispatch, a stale buffer, an oracle
/// read from the wrong file.
#[test]
fn a_different_experts_weights_are_rejected() {
    let Some(dir) = fixture_dir() else {
        eprintln!("skipped: set {FIXTURE_ENV} to the exported fixture directory");
        return;
    };
    let Some(metal) = MetalBackend::new() else {
        eprintln!("skipped: no Metal device on this host");
        return;
    };
    let fx = load(&dir);
    let (a, b) = (&fx.experts[0], &fx.experts[1]);

    let mismatched = metal_expert_ffn(&metal, &fx.x, b.metal(), fx.hidden, fx.inter);
    let rel = rel_err(&mismatched, &a.oracle);
    assert!(
        rel > 1e-2,
        "expert {} scored rel {rel:e} against expert {}'s oracle — the gate is \
         not reading the weights it claims to",
        b.id,
        a.id
    );
    eprintln!("[control] wrong-expert rel {rel:e} (tolerance {REL_TOLERANCE:e})");
}

/// The rung. Every selected expert plus the shared branch: Metal
/// against `modeling_kimi.py`'s own output, and Metal against the CPU
/// kernel that arc already proved.
#[test]
fn the_metal_bf16_expert_ffn_matches_the_checkpoints_own_output() {
    let Some(dir) = fixture_dir() else {
        eprintln!("skipped: set {FIXTURE_ENV} to the exported fixture directory");
        return;
    };
    let Some(metal) = MetalBackend::new() else {
        eprintln!("skipped: no Metal device on this host");
        return;
    };
    let fx = load(&dir);
    assert_eq!(fx.experts.len(), 9, "8 routed + 1 shared");

    for e in &fx.experts {
        let gpu = metal_expert_ffn(&metal, &fx.x, e.metal(), fx.hidden, fx.inter);
        let cpu = expert_ffn(&fx.x, e.cpu(), fx.hidden, fx.inter);

        let vs_oracle = rel_err(&gpu, &e.oracle);
        let vs_cpu = rel_err(&gpu, &cpu);
        let cpu_vs_oracle = rel_err(&cpu, &e.oracle);
        eprintln!(
            "[{:>9}] metal-vs-hf {vs_oracle:.3e}  metal-vs-cpu {vs_cpu:.3e}  \
             cpu-vs-hf {cpu_vs_oracle:.3e}",
            e.id
        );
        assert!(
            vs_oracle < REL_TOLERANCE,
            "{}: metal vs the checkpoint's own output, rel {vs_oracle:e}",
            e.id
        );
        assert!(
            vs_cpu < REL_TOLERANCE,
            "{}: metal vs the proven CPU kernel, rel {vs_cpu:e}",
            e.id
        );
    }
}

/// What one expert's three projections cost on each side, at the real
/// geometry, and — the question that decides the next rung — whether the
/// GPU arm is limited by memory bandwidth or by per-submission latency.
///
/// Three measurements, each min-of-N after warmup:
///   * the **unfused** FFN, three separate command buffers, which is
///     what a naive port would ship;
///   * one gemv alone, and a deliberately tiny one, whose difference is
///     the fixed cost of a submission at this shape;
///   * nine DISTINCT expert gate matrices (~40 MiB, past any cache) run
///     sequentially and then as one command buffer. Same kernel, same
///     bytes, same arguments — only the submission batching differs, so
///     the gap between them is the submission cost and nothing else.
///
/// Asserts nothing about speed: a timing assertion is a flaky test, and
/// this machine's bench protocol wants bracketed, interleaved arms
/// rather than one in-test loop. It reports, so the next rung starts
/// from a measured number. The CPU arm is single-threaded on purpose —
/// `expert_ffn` calls `FusedBf16.project_rows` directly and the block's
/// parallelism is across experts, not within one.
#[test]
fn report_what_one_expert_ffn_costs_on_each_side() {
    let Some(dir) = fixture_dir() else {
        eprintln!("skipped: set {FIXTURE_ENV} to the exported fixture directory");
        return;
    };
    let Some(metal) = MetalBackend::new() else {
        eprintln!("skipped: no Metal device on this host");
        return;
    };
    let fx = load(&dir);
    let e = &fx.experts[0];
    let stored_bytes = (e.gate.len() + e.up.len() + e.down.len()) as f64;
    let gbs = |bytes: f64, ms: f64| bytes / 1e6 / ms;

    let gpu_ms = min_ms(BENCH_WARMUP, BENCH_ITERS, || {
        std::hint::black_box(metal_expert_ffn(
            &metal,
            &fx.x,
            e.metal(),
            fx.hidden,
            fx.inter,
        ));
    });
    let cpu_ms = min_ms(BENCH_WARMUP, BENCH_ITERS, || {
        std::hint::black_box(expert_ffn(&fx.x, e.cpu(), fx.hidden, fx.inter));
    });
    let one_gemv_ms = min_ms(BENCH_WARMUP, BENCH_ITERS, || {
        std::hint::black_box(metal.bf16_gemv_force(&e.gate, &fx.x, fx.inter, fx.hidden));
    });

    // A dispatch that reads almost nothing, through the identical entry
    // point: whatever this costs is not bandwidth.
    let (tiny_n, tiny_k) = (TINY_ROWS, TINY_COLS);
    let tiny_w = vec![0u8; tiny_n * tiny_k * BF16_BYTES];
    let tiny_x = vec![1.0f32; tiny_k];
    let floor_ms = min_ms(BENCH_WARMUP, BENCH_ITERS, || {
        std::hint::black_box(metal.bf16_gemv_force(&tiny_w, &tiny_x, tiny_n, tiny_k));
    });

    // Nine distinct expert gate matrices: sequential submissions vs one.
    let batch: Vec<(&[u8], usize, usize)> = fx
        .experts
        .iter()
        .map(|x| (x.gate.as_slice(), fx.inter, fx.hidden))
        .collect();
    let batch_bytes: f64 = fx.experts.iter().map(|x| x.gate.len() as f64).sum();
    let seq_ms = min_ms(BENCH_WARMUP, BENCH_ITERS, || {
        for &(w, n, k) in &batch {
            std::hint::black_box(metal.bf16_gemv_force(w, &fx.x, n, k));
        }
    });
    let batched_ms = min_ms(BENCH_WARMUP, BENCH_ITERS, || {
        std::hint::black_box(metal.bf16_gemv_multi(&batch, &fx.x));
    });

    eprintln!(
        "[bench] expert {} hidden={} inter={} stored={:.2} MiB (bf16)\n\
         [bench]   metal 3-gemv FFN   {gpu_ms:.3} ms  {:.1} GB/s\n\
         [bench]   cpu   3-gemv FFN   {cpu_ms:.3} ms  {:.1} GB/s (single thread)\n\
         [bench]   speedup metal/cpu  {:.2}x (unfused: one command buffer per projection)\n\
         [bench]   metal 1 gemv       {one_gemv_ms:.3} ms  {:.1} GB/s\n\
         [bench]   metal {tiny_n}x{tiny_k} gemv    {floor_ms:.3} ms  <- per-submission floor, not bandwidth\n\
         [bench]   9 experts, {:.1} MiB, sequential {seq_ms:.3} ms  {:.1} GB/s\n\
         [bench]   9 experts, {:.1} MiB, one CB     {batched_ms:.3} ms  {:.1} GB/s  ({:.2}x)",
        e.id,
        fx.hidden,
        fx.inter,
        stored_bytes / (1024.0 * 1024.0),
        gbs(stored_bytes, gpu_ms),
        gbs(stored_bytes, cpu_ms),
        cpu_ms / gpu_ms,
        gbs(e.gate.len() as f64, one_gemv_ms),
        batch_bytes / (1024.0 * 1024.0),
        gbs(batch_bytes, seq_ms),
        batch_bytes / (1024.0 * 1024.0),
        gbs(batch_bytes, batched_ms),
        seq_ms / batched_ms,
    );
    assert!(gpu_ms.is_finite() && cpu_ms.is_finite() && batched_ms.is_finite());
}
