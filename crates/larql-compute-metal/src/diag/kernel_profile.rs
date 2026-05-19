//! Per-kernel Metal GPU bandwidth profiler.
//!
//! Measures each production kernel at Gemma 3 4B shapes in two modes:
//!
//! **Isolated**: one commit+wait per kernel call. Includes ~20µs GPU spin-up
//! cost. Useful for comparing kernels against each other.
//!
//! **Batched**: `n_layers` (default 34) calls per command buffer, single
//! commit+wait. The GPU stays warm; this matches the real decode pipeline.
//! Use batched numbers for understanding actual tok/s impact.
//!
//! ## Key findings (2026-04-26, M3 Max, Gemma 3 4B)
//! | Kernel | Batched GB/s | ms/tok | Bottleneck |
//! |---|---|---|---|
//! | q6k_matvec (FFN down, K=10240) | 312 GB/s | 2.34ms | bandwidth-bound (LPDDR5X) |
//! | q4k_ffn_gate_up_8sg (gate+up, K=2560) | 272 GB/s | 3.68ms | compute-bound (Q4_K dequant) |
//! | lm_head f32_gemv (262K×2560) | 370 GB/s | — | bandwidth-bound (near peak) |
//!
//! Gate+up is compute-bound because Q4_K at K=2560 has low bytes-per-element
//! (0.5625 B/elem) — the GPU spends more cycles on nibble dequant than waiting
//! for memory. Closing the gap vs Ollama's ~414 GB/s effective rate requires
//! reducing the per-element compute overhead (vectorized accumulation).

use std::time::Instant;

const GEMMA3_4B_KV_DIM: usize = 4096;

/// Result for a single kernel profiling run.
#[derive(Debug, Clone)]
pub struct KernelResult {
    pub name: String,
    /// Megabytes of weight data read per kernel call.
    pub mb_per_call: f64,
    /// Mean isolated time per call (ms), including GPU spin-up.
    pub isolated_ms: f64,
    /// Stddev of isolated times.
    pub isolated_sd_ms: f64,
    /// Effective bandwidth from isolated measurement (GB/s).
    pub isolated_gbs: f64,
    /// Mean time per layer when batched n_layers in one command buffer (ms).
    pub batched_ms_per_layer: f64,
    /// Effective bandwidth from batched measurement (GB/s).
    pub batched_gbs: f64,
}

impl KernelResult {
    /// ms/token at `n_layers` layers using the batched rate.
    pub fn ms_per_token(&self, n_layers: usize) -> f64 {
        self.batched_ms_per_layer * n_layers as f64
    }

    /// Whether the kernel appears compute-bound (GB/s well below peak ~350).
    pub fn is_compute_bound(&self) -> bool {
        self.batched_gbs < 300.0
    }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}
fn stddev(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

fn synth_f32(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (seed + i as f32 * 0.007).sin() * 0.4)
        .collect()
}

fn measure_isolated(warmup: usize, iters: usize, f: &mut impl FnMut()) -> (f64, f64) {
    let mut times = Vec::with_capacity(iters);
    for i in 0..warmup + iters {
        let t = Instant::now();
        f();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            times.push(ms);
        }
    }
    (mean(&times), stddev(&times))
}

/// Measure batched throughput where each iteration runs `f()` `n_layers`
/// times. **`f()` is responsible for its own cmd-buffer + commit + wait.**
///
/// This MIS-measures throughput when used with closures that create one
/// cmd-buffer per call: each cmd-buffer costs ~10 µs of dispatch overhead
/// that gets billed against the kernel time. Real production runs all
/// `n_layers` dispatches in ONE cmd buffer with a single commit+wait —
/// see [`measure_single_cmdbuf_batched`] for that.
///
/// Kept for callers who genuinely want per-call cmd-buffer overhead in
/// the measurement (rare).
#[allow(dead_code)]
fn measure_batched(warmup: usize, iters: usize, n_layers: usize, f: &mut impl FnMut()) -> f64 {
    let mut times = Vec::with_capacity(iters);
    for i in 0..warmup + iters {
        let t = Instant::now();
        for _ in 0..n_layers {
            f();
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            times.push(ms / n_layers as f64);
        }
    }
    mean(&times)
}

/// Measure batched throughput with all `n_layers` dispatches in ONE cmd
/// buffer, single commit+wait. This is what production decode actually
/// does (all of a token's per-layer kernels live in one cmd buffer), so
/// the GB/s number reflects real per-kernel cost without dispatch
/// overhead pollution.
///
/// `encode` must NOT call `commit`/`wait_until_completed`/`end_encoding`
/// — this function owns the cmd-buffer lifecycle.
///
/// Discovered 2026-04-28: the older `measure_batched` was being used
/// with closures that did per-call commit+wait, undercounting q6k_matvec
/// throughput by 4× (74 vs real 315 GB/s). See ROADMAP P0 "Decode kernel
/// optimization → Track A" for the bisect.
fn measure_single_cmdbuf_batched(
    metal: &super::super::MetalBackend,
    warmup: usize,
    iters: usize,
    n_layers: usize,
    encode: &impl Fn(&metal::ComputeCommandEncoderRef),
) -> f64 {
    let mut times: Vec<f64> = Vec::with_capacity(iters);
    for i in 0..warmup + iters {
        let t = Instant::now();
        let cmd = metal.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        for _ in 0..n_layers {
            encode(enc);
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if i >= warmup {
            times.push(ms / n_layers as f64);
        }
    }
    mean(&times)
}

/// Profile all production kernels at Gemma 3 4B shapes.
///
/// Returns one `KernelResult` per kernel. Prints a formatted table to stdout.
/// Pass `n_layers=34` for Gemma 3 4B, `warmup=5`, `iters=50` for reliable numbers.
pub fn profile_all(n_layers: usize, warmup: usize, iters: usize) -> Vec<KernelResult> {
    use crate::MetalBackend;
    use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};
    use larql_compute::{MatMul, QuantMatVec};
    use metal::MTLSize;

    let metal = MetalBackend::new().expect("Metal backend required for profiling");

    // Gemma 3 4B production shapes
    let hidden = 2560usize;
    let inter = 10240usize;
    let q_dim = 8192usize;
    let kv_dim = GEMMA3_4B_KV_DIM;
    let sb = 256usize;
    let q4k_sb = 144usize;
    let q6k_sb = 210usize;

    let mut results = Vec::new();

    // Measure commit+wait overhead (empty command buffer).
    let commit_overhead_ms = {
        let mut times = Vec::new();
        for i in 0..warmup + iters {
            let t = Instant::now();
            let cmd = metal.queue().new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            if i >= warmup {
                times.push(ms);
            }
        }
        mean(&times)
    };

    println!("Commit+wait overhead: {commit_overhead_ms:.3}ms");
    println!();
    println!(
        "{:<44} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Kernel", "iso_ms", "iso_gbs", "bat_ms", "bat_gbs", "ms/tok"
    );
    println!("{}", "-".repeat(88));

    // ── q6k_matvec: FFN down (N=hidden, K=inter) ─────────────────────────
    {
        let n = hidden;
        let k = inter;
        let mb = (n * (k / sb * q6k_sb)) as f64 / 1e6;
        let w = quantize_q6_k(&synth_f32(n * k, 0.1));
        let x = synth_f32(k, 0.5);

        let (iso_ms, iso_sd) = measure_isolated(warmup, iters, &mut || {
            let _ = metal.q6k_matvec(&w, &x, n, k);
        });

        let wb = metal.bufs().get_bytes(&w);
        let xb = metal.bufs().transient_from_f32(&x);
        let ob = metal.bufs().output((n * 4) as u64);
        let kh = &metal.quant.q6k_matvec_pipeline;
        let n_tgs = (n as u64).div_ceil(kh.rows_per_tg);
        let n_val = n as u32;
        let k_val = k as u32;

        // TRUE batched (warm-cache): same weight buffer reused per call.
        let bat_ms = measure_single_cmdbuf_batched(&metal, warmup, iters, n_layers, &|enc| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&wb), 0);
            enc.set_buffer(1, Some(&xb), 0);
            enc.set_buffer(2, Some(&ob), 0);
            enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(n_tgs, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        });

        // COLD-cache: rotate through 8 distinct weight buffers (each
        // 21.5 MB, total 172 MB — far exceeds L2). Each kernel call
        // sees its weights fresh from DRAM, mirroring real decode
        // where each layer's down weights are evicted by the next.
        let cold_n = n_layers.min(8);
        let cold_ms = {
            let weights: Vec<_> = (0..cold_n)
                .map(|i| {
                    let w = quantize_q6_k(&synth_f32(n * k, 0.1 + i as f32 * 0.05));
                    metal.bufs().get_bytes(&w)
                })
                .collect();
            let mut times: Vec<f64> = Vec::with_capacity(iters);
            for i in 0..warmup + iters {
                let t = std::time::Instant::now();
                let cmd = metal.queue().new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                for layer in 0..n_layers {
                    let idx = layer % cold_n;
                    enc.set_compute_pipeline_state(&kh.state);
                    enc.set_buffer(0, Some(&weights[idx]), 0);
                    enc.set_buffer(1, Some(&xb), 0);
                    enc.set_buffer(2, Some(&ob), 0);
                    enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(4, 4, &k_val as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(n_tgs, 1, 1),
                        MTLSize::new(kh.threads_per_tg, 1, 1),
                    );
                }
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                if i >= warmup {
                    times.push(ms / n_layers as f64);
                }
            }
            mean(&times)
        };
        let cold_gbs = mb / cold_ms;

        let iso_kernel = (iso_ms - commit_overhead_ms).max(0.001);
        let r = KernelResult {
            name: "q6k_matvec (down, 2560×10240)".into(),
            mb_per_call: mb,
            isolated_ms: iso_ms,
            isolated_sd_ms: iso_sd,
            isolated_gbs: mb / iso_kernel,
            batched_ms_per_layer: bat_ms,
            batched_gbs: mb / bat_ms,
        };
        println!(
            "{:<44} {:>7.3}ms {:>7.1} {:>7.3}ms {:>7.1} {:>7.1}ms",
            r.name,
            r.isolated_ms,
            r.isolated_gbs,
            r.batched_ms_per_layer,
            r.batched_gbs,
            r.ms_per_token(n_layers)
        );
        println!(
            "  ↳ cold-cache (rotate {cold_n} weight buffers): {cold_ms:>7.3}ms/call  {cold_gbs:>7.1} GB/s  ({:.1}ms/tok)",
            cold_ms * n_layers as f64
        );
        results.push(r);
    }

    // ── q4k_ffn_gate_up_8sg: production fused gate+up (N=inter, K=hidden) ──
    {
        let n = inter;
        let k = hidden;
        let mb = 2.0 * (n * (k / sb * q4k_sb)) as f64 / 1e6;
        let gate_q4k = quantize_q4_k(&synth_f32(n * k, 0.2));
        let up_q4k = quantize_q4_k(&synth_f32(n * k, 0.3));
        let x = synth_f32(k, 0.5);

        // Isolated: use the trait method which handles dispatch internally.
        // We can't use trait method for gate+up (it's internal), so dispatch directly.
        let wg = metal.bufs().get_bytes(&gate_q4k);
        let wu = metal.bufs().get_bytes(&up_q4k);
        let xb = metal.bufs().transient_from_f32(&x);
        let go = metal.bufs().output((n * 4) as u64);
        let uo = metal.bufs().output((n * 4) as u64);
        let kh = &metal.ffn.q4k_ffn_gate_up_8sg_pipeline;
        let tgs = (n as u64).div_ceil(kh.rows_per_tg);
        let n_val = n as u32;
        let k_val = k as u32;

        let dispatch = |enc: &metal::ComputeCommandEncoderRef| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&wg), 0);
            enc.set_buffer(1, Some(&wu), 0);
            enc.set_buffer(2, Some(&xb), 0);
            enc.set_buffer(3, Some(&go), 0);
            enc.set_buffer(4, Some(&uo), 0);
            enc.set_bytes(5, 4, &n_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(tgs * 2, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        };

        let (iso_ms, iso_sd) = measure_isolated(warmup, iters, &mut || {
            let cmd = metal.queue().new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            dispatch(enc);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        });
        // TRUE batched (warm-cache): all n_layers dispatches reuse the
        // SAME weight buffers (wg/wu). After the first call, weights
        // stay hot in L2 for the next 33 calls — overstates production
        // because real decode walks 34 different layers' weights, only
        // 2-3 of which fit in L2 simultaneously.
        let bat_ms = measure_single_cmdbuf_batched(&metal, warmup, iters, n_layers, &dispatch);

        // COLD-cache batched: allocate n_layers distinct weight buffer
        // pairs, dispatch on each in sequence within ONE cmd buffer.
        // By the time the cmd buffer finishes, the GPU has touched
        // n_layers × 2 × 14.7 MB = ~1 GB of weight data — far beyond
        // L2's ~16-32 MB capacity, so each kernel call sees cold L2
        // for its specific weights. This is the production-realistic
        // throughput: in real decode, each layer's gate+up weights
        // are loaded fresh from DRAM, not reused from L2.
        //
        // Allocating n_layers buffers up front is heavy (~1 GB of
        // device-resident memory) so we cap at min(n_layers, 8) and
        // round-robin through them — 8 × 30 MB = 240 MB still
        // exceeds L2, guarantees eviction without exhausting GPU
        // memory. Eight is empirically enough on M3 Max.
        let cold_n = n_layers.min(8);
        let cold_ms = {
            let weights_g: Vec<_> = (0..cold_n)
                .map(|i| {
                    let w = quantize_q4_k(&synth_f32(n * k, 0.2 + i as f32 * 0.07));
                    metal.bufs().get_bytes(&w)
                })
                .collect();
            let weights_u: Vec<_> = (0..cold_n)
                .map(|i| {
                    let w = quantize_q4_k(&synth_f32(n * k, 0.3 + i as f32 * 0.11));
                    metal.bufs().get_bytes(&w)
                })
                .collect();

            let mut times: Vec<f64> = Vec::with_capacity(iters);
            for i in 0..warmup + iters {
                let t = std::time::Instant::now();
                let cmd = metal.queue().new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                for layer in 0..n_layers {
                    let idx = layer % cold_n;
                    enc.set_compute_pipeline_state(&kh.state);
                    enc.set_buffer(0, Some(&weights_g[idx]), 0);
                    enc.set_buffer(1, Some(&weights_u[idx]), 0);
                    enc.set_buffer(2, Some(&xb), 0);
                    enc.set_buffer(3, Some(&go), 0);
                    enc.set_buffer(4, Some(&uo), 0);
                    enc.set_bytes(5, 4, &n_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(6, 4, &k_val as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(tgs * 2, 1, 1),
                        MTLSize::new(kh.threads_per_tg, 1, 1),
                    );
                }
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                if i >= warmup {
                    times.push(ms / n_layers as f64);
                }
            }
            mean(&times)
        };
        let cold_gbs = mb / cold_ms;

        let iso_kernel = (iso_ms - commit_overhead_ms).max(0.001);
        let r = KernelResult {
            name: "q4k_ffn_gate_up_8sg (gate+up, 10240×2560)".into(),
            mb_per_call: mb,
            isolated_ms: iso_ms,
            isolated_sd_ms: iso_sd,
            isolated_gbs: mb / iso_kernel,
            batched_ms_per_layer: bat_ms,
            batched_gbs: mb / bat_ms,
        };
        println!(
            "{:<44} {:>7.3}ms {:>7.1} {:>7.3}ms {:>7.1} {:>7.1}ms",
            r.name,
            r.isolated_ms,
            r.isolated_gbs,
            r.batched_ms_per_layer,
            r.batched_gbs,
            r.ms_per_token(n_layers)
        );
        println!(
            "  ↳ cold-cache (rotate {cold_n} weight buffers): {cold_ms:>7.3}ms/call  {cold_gbs:>7.1} GB/s  ({:.1}ms/tok)",
            cold_ms * n_layers as f64
        );
        results.push(r);
    }

    // ── q4k_matvec: Wo O-projection (N=hidden, K=q_dim) ──────────────────
    {
        let n = hidden;
        let k = q_dim;
        let mb = (n * (k / sb * q4k_sb)) as f64 / 1e6;
        let w = quantize_q4_k(&synth_f32(n * k, 0.4));
        let x = synth_f32(k, 0.6);
        let (iso_ms, iso_sd) = measure_isolated(warmup, iters, &mut || {
            let _ = metal.q4k_matvec(&w, &x, n, k);
        });

        // Batched: same single-cmd-buffer pattern as gate+up. Was
        // missing here historically — Wo's "13.4 ms/tok" earlier
        // estimate was iso_ms × 34 which over-counts cmd-buffer
        // overhead.
        let wb = metal.bufs().get_bytes(&w);
        let xb = metal.bufs().transient_from_f32(&x);
        let ob = metal.bufs().output((n * 4) as u64);
        let kh = &metal.quant.q4k_matvec_pipeline;
        let n_tgs = (n as u64).div_ceil(kh.rows_per_tg);
        let n_val = n as u32;
        let k_val = k as u32;
        let bat_ms = measure_single_cmdbuf_batched(&metal, warmup, iters, n_layers, &|enc| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&wb), 0);
            enc.set_buffer(1, Some(&xb), 0);
            enc.set_buffer(2, Some(&ob), 0);
            enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(n_tgs, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        });

        let iso_kernel = (iso_ms - commit_overhead_ms).max(0.001);
        let r = KernelResult {
            name: "q4k_matvec (Wo, 2560×8192)".into(),
            mb_per_call: mb,
            isolated_ms: iso_ms,
            isolated_sd_ms: iso_sd,
            isolated_gbs: mb / iso_kernel,
            batched_ms_per_layer: bat_ms,
            batched_gbs: mb / bat_ms,
        };
        println!(
            "{:<44} {:>7.3}ms {:>7.1} {:>7.3}ms {:>7.1} {:>7.1}ms",
            r.name,
            r.isolated_ms,
            r.isolated_gbs,
            r.batched_ms_per_layer,
            r.batched_gbs,
            r.ms_per_token(n_layers)
        );
        results.push(r);
    }

    // ── q4k_qkv_proj: fused Q+K+V projection (production decode path) ────
    //
    // Three rectangles in one dispatch: Wq[q_dim × K], Wk[kv_dim × K],
    // Wv[kv_dim × K]. K = hidden = 2560 for Gemma 3 4B. Total bytes
    // moved per call: (q_dim + 2*kv_dim) × K × 0.5625. Lane utilisation
    // is poor at K=2560: kernel uses `sb += 32` lane stride but only
    // K/256 = 10 super-blocks per row, so 22/32 lanes idle inside each
    // simdgroup (auto-memory suggests this is the migration target —
    // q4k_matvec was rewritten to (ix, j, sh) decomposition that uses
    // all 32 lanes).
    {
        let q_rows = q_dim;
        let k_rows = kv_dim;
        let v_rows = kv_dim;
        let total_rows = q_rows + k_rows + v_rows;
        let k = hidden;
        let mb = ((q_rows + k_rows + v_rows) * (k / sb * q4k_sb)) as f64 / 1e6;
        let wq = quantize_q4_k(&synth_f32(q_rows * k, 0.5));
        let wk = quantize_q4_k(&synth_f32(k_rows * k, 0.6));
        let wv = quantize_q4_k(&synth_f32(v_rows * k, 0.7));
        let x = synth_f32(k, 0.4);

        let wqb = metal.bufs().get_bytes(&wq);
        let wkb = metal.bufs().get_bytes(&wk);
        let wvb = metal.bufs().get_bytes(&wv);
        let xb = metal.bufs().transient_from_f32(&x);
        let qo = metal.bufs().output((q_rows * 4) as u64);
        let ko = metal.bufs().output((k_rows * 4) as u64);
        let vo = metal.bufs().output((v_rows * 4) as u64);
        let kh = &metal.attention.q4k_qkv_proj_pipeline;
        let n_tgs = (total_rows as u64).div_ceil(kh.rows_per_tg);
        let q_val = q_rows as u32;
        let k_val_n = k_rows as u32;
        let v_val = v_rows as u32;
        let k_val = k as u32;

        let dispatch = |enc: &metal::ComputeCommandEncoderRef| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&wqb), 0);
            enc.set_buffer(1, Some(&wkb), 0);
            enc.set_buffer(2, Some(&wvb), 0);
            enc.set_buffer(3, Some(&xb), 0);
            enc.set_buffer(4, Some(&qo), 0);
            enc.set_buffer(5, Some(&ko), 0);
            enc.set_buffer(6, Some(&vo), 0);
            enc.set_bytes(7, 4, &q_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(8, 4, &k_val_n as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &v_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(10, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(n_tgs, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        };

        let (iso_ms, iso_sd) = measure_isolated(warmup, iters, &mut || {
            let cmd = metal.queue().new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            dispatch(enc);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        });
        let bat_ms = measure_single_cmdbuf_batched(&metal, warmup, iters, n_layers, &dispatch);

        let iso_kernel = (iso_ms - commit_overhead_ms).max(0.001);
        let r = KernelResult {
            name: format!(
                "q4k_qkv_proj (Q+K+V, {}+{}+{}×{})",
                q_rows, k_rows, v_rows, k
            ),
            mb_per_call: mb,
            isolated_ms: iso_ms,
            isolated_sd_ms: iso_sd,
            isolated_gbs: mb / iso_kernel,
            batched_ms_per_layer: bat_ms,
            batched_gbs: mb / bat_ms,
        };
        println!(
            "{:<44} {:>7.3}ms {:>7.1} {:>7.3}ms {:>7.1} {:>7.1}ms",
            r.name,
            r.isolated_ms,
            r.isolated_gbs,
            r.batched_ms_per_layer,
            r.batched_gbs,
            r.ms_per_token(n_layers)
        );
        results.push(r);
    }

    // ── q4k_q6k_qkv_proj_normed: production Gemma 3 QKV ─────────────────
    //
    // This is the actual Gemma 3 4B hot path: input RMS norm fused into a
    // mixed Q4_K Q/K + Q6_K V projection. Measure it separately from the
    // uniform-Q4_K synthetic q4k_qkv_proj above so QKV shows up correctly in
    // the decode gap diagnosis.
    {
        let q_rows = q_dim;
        let k_rows = kv_dim;
        let v_rows = kv_dim;
        let total_rows = q_rows + k_rows + v_rows;
        let k = hidden;
        let mb_q4 = ((q_rows + k_rows) * (k / sb * q4k_sb)) as f64 / 1e6;
        let mb_q6 = (v_rows * (k / sb * q6k_sb)) as f64 / 1e6;
        let mb = mb_q4 + mb_q6;

        let wq = quantize_q4_k(&synth_f32(q_rows * k, 0.5));
        let wk = quantize_q4_k(&synth_f32(k_rows * k, 0.6));
        let wv = quantize_q6_k(&synth_f32(v_rows * k, 0.7));
        let h = synth_f32(k, 0.4);
        let norm_w = vec![1.0f32; k];

        let wqb = metal.bufs().get_bytes(&wq);
        let wkb = metal.bufs().get_bytes(&wk);
        let wvb = metal.bufs().get_bytes(&wv);
        let hb = metal.bufs().transient_from_f32(&h);
        let nb = metal.bufs().get_f32(&norm_w);
        let qo = metal.bufs().output((q_rows * 4) as u64);
        let ko = metal.bufs().output((k_rows * 4) as u64);
        let vo = metal.bufs().output((v_rows * 4) as u64);
        let kh = &metal.attention.q4k_q6k_qkv_proj_normed_pipeline;
        let n_tgs = (total_rows as u64).div_ceil(kh.rows_per_tg);
        let q_val = q_rows as u32;
        let k_rows_val = k_rows as u32;
        let v_val = v_rows as u32;
        let k_val = k as u32;
        let eps = larql_compute::RMSNORM_EPSILON_DEFAULT;
        let offset = 1.0f32;

        let dispatch = |enc: &metal::ComputeCommandEncoderRef| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&wqb), 0);
            enc.set_buffer(1, Some(&wkb), 0);
            enc.set_buffer(2, Some(&wvb), 0);
            enc.set_buffer(3, Some(&hb), 0);
            enc.set_buffer(4, Some(&nb), 0);
            enc.set_buffer(5, Some(&qo), 0);
            enc.set_buffer(6, Some(&ko), 0);
            enc.set_buffer(7, Some(&vo), 0);
            enc.set_bytes(8, 4, &q_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &k_rows_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(10, 4, &v_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(11, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(12, 4, &eps as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(13, 4, &offset as *const f32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(n_tgs, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        };

        let (iso_ms, iso_sd) = measure_isolated(warmup, iters, &mut || {
            let cmd = metal.queue().new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            dispatch(enc);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        });
        let bat_ms = measure_single_cmdbuf_batched(&metal, warmup, iters, n_layers, &dispatch);

        let iso_kernel = (iso_ms - commit_overhead_ms).max(0.001);
        let r = KernelResult {
            name: format!(
                "q4k_q6k_qkv_normed (Q+K+V, {}+{}+{}×{})",
                q_rows, k_rows, v_rows, k
            ),
            mb_per_call: mb,
            isolated_ms: iso_ms,
            isolated_sd_ms: iso_sd,
            isolated_gbs: mb / iso_kernel,
            batched_ms_per_layer: bat_ms,
            batched_gbs: mb / bat_ms,
        };
        println!(
            "{:<44} {:>7.3}ms {:>7.1} {:>7.3}ms {:>7.1} {:>7.1}ms",
            r.name,
            r.isolated_ms,
            r.isolated_gbs,
            r.batched_ms_per_layer,
            r.batched_gbs,
            r.ms_per_token(n_layers)
        );
        println!(
            "  ↳ GB/s counts Q/K/V weight bytes only; normed kernel also rereads H+norm per TG"
        );
        results.push(r);
    }

    // ── f32_gemv: lm_head (N=vocab, K=hidden) ────────────────────────────
    {
        let n = 262_144usize;
        let k = hidden;
        let mb = (n * k * 4) as f64 / 1e6;
        let w = ndarray::Array2::from_shape_vec((n, k), synth_f32(n * k, 0.7)).unwrap();
        let x = synth_f32(k, 0.5);
        let (iso_ms, iso_sd) = measure_isolated(warmup, iters.min(20), &mut || {
            let _ = metal.f32_gemv_force(w.view(), &x);
        });
        let iso_kernel = (iso_ms - commit_overhead_ms).max(0.001);
        let r = KernelResult {
            name: "f32_gemv (lm_head, 262K×2560)".into(),
            mb_per_call: mb,
            isolated_ms: iso_ms,
            isolated_sd_ms: iso_sd,
            isolated_gbs: mb / iso_kernel,
            batched_ms_per_layer: iso_ms, // lm_head is one-per-token, not per-layer
            batched_gbs: mb / iso_kernel,
        };
        println!(
            "{:<44} {:>7.3}ms {:>7.1} {:>7}     {:>7}   (per token, not per layer)",
            r.name, r.isolated_ms, r.isolated_gbs, "—", "—"
        );
        results.push(r);
    }

    // ── Summary ───────────────────────────────────────────────────────────
    let down = results.iter().find(|r| r.name.contains("down")).unwrap();
    let gate = results.iter().find(|r| r.name.contains("gate")).unwrap();
    let total_ms = down.ms_per_token(n_layers) + gate.ms_per_token(n_layers);

    println!();
    println!("=== Bottleneck analysis ===");
    println!(
        "q6k_matvec (down)   {:.1} GB/s — {}",
        down.batched_gbs,
        if down.is_compute_bound() {
            "COMPUTE-BOUND"
        } else {
            "bandwidth-bound"
        }
    );
    println!(
        "q4k_ffn_gate_up     {:.1} GB/s — {}",
        gate.batched_gbs,
        if gate.is_compute_bound() {
            "COMPUTE-BOUND (K=2560 dequant dominates)"
        } else {
            "bandwidth-bound"
        }
    );
    println!(
        "These two: {total_ms:.2}ms/tok ({:.0}% of ~11.7ms GPU fwd)",
        total_ms / 11.7 * 100.0
    );
    println!(
        "At 350 GB/s: would take {:.1}ms/tok → need {:.0}% more throughput",
        3029.0 / 350.0,
        (3029.0 / 350.0 / (down.batched_ms_per_layer + gate.batched_ms_per_layer + 0.001) - 1.0)
            .abs()
            * 0.0
            + (350.0 / ((down.batched_gbs + gate.batched_gbs) / 2.0) - 1.0) * 100.0
    );

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_and_stddev_match_textbook_formulas() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&v) - 3.0).abs() < 1e-12);
        // Population stddev of 1..=5 is √2.
        assert!((stddev(&v) - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn synth_f32_returns_requested_length() {
        let v = synth_f32(7, 0.1);
        assert_eq!(v.len(), 7);
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn ms_per_token_scales_linearly_with_layer_count() {
        let r = KernelResult {
            name: "k".into(),
            mb_per_call: 1.0,
            isolated_ms: 0.5,
            isolated_sd_ms: 0.01,
            isolated_gbs: 2000.0,
            batched_ms_per_layer: 0.1,
            batched_gbs: 10_000.0,
        };
        assert!((r.ms_per_token(34) - 3.4).abs() < 1e-12);
        assert!((r.ms_per_token(0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn is_compute_bound_flags_low_throughput_kernels() {
        let mut r = KernelResult {
            name: "k".into(),
            mb_per_call: 1.0,
            isolated_ms: 0.5,
            isolated_sd_ms: 0.01,
            isolated_gbs: 2000.0,
            batched_ms_per_layer: 0.1,
            batched_gbs: 250.0,
        };
        assert!(r.is_compute_bound());
        r.batched_gbs = 320.0;
        assert!(!r.is_compute_bound());
    }

    /// Drive `profile_all` end-to-end at minimum params (n_layers=1
    /// warmup=0 iters=1) so the per-kernel `measure_isolated` +
    /// `measure_single_cmdbuf_batched` + cold-cache loops all run on
    /// real Metal. Skips on hosts without a Metal device.
    #[test]
    fn profile_all_smoke_runs_every_kernel_once() {
        if crate::MetalBackend::new().is_none() {
            return;
        }
        let results = profile_all(1, 0, 1);
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.batched_ms_per_layer.is_finite()
            && r.isolated_ms.is_finite()
            && r.batched_gbs.is_finite()));
    }
}
