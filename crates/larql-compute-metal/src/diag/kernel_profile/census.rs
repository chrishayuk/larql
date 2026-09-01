//! Per-shape census of the kernels a model actually dispatches.
//!
//! Split out of `kernel_profile.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::measure::{
    mean, measure_batched, measure_isolated, measure_single_cmdbuf_batched, stddev, synth_f32,
};
#[allow(unused_imports)]
use super::*;

/// One (kernel, shape) cell of the census.
#[derive(Debug, Clone)]
pub struct ShapeCell {
    pub kernel: &'static str,
    pub shape: &'static str,
    pub n: usize,
    pub k: usize,
    pub packed_mb: f64,
    pub cold_ms: f64,
    pub cold_gbs: f64,
    pub eta: f64,
}

/// Cross kernel against shape so the two explanations for a low eta separate.
///
/// The composed ledger assigns `q4k_matvec`'s 0.59 to K3's attention class, but
/// that figure was measured on a *Gemma* shape in *Q4K*, while a transcoded
/// image would run *Q6K* on a *K3* shape. Two hypotheses follow, and they imply
/// opposite plans:
///
/// - **Kernel-borne**: the inefficiency belongs to `q4k_matvec`. Transcoding to
///   Q6K sidesteps it, and the attention class inherits the better eta.
/// - **Shape-borne**: it belongs to the `[12288, 7168]` geometry — long
///   reduction, wide output, single vector. Then Q6K inherits it too, the
///   transcode ceiling stands, and every future attention kernel must be
///   designed against it.
///
/// Running both kernels over both shape families answers it directly. Gemma
/// rows are anchors: they must reproduce the banked 0.59 / 0.85, or the harness
/// is not comparable to the figures the ledger already cites.
///
/// Cold-rotating throughout — 8 distinct weight buffers per cell, batched
/// `n_layers` to one command buffer, matching `profile_all`'s protocol exactly.
pub fn profile_shape_census(n_layers: usize, warmup: usize, iters: usize) -> Vec<ShapeCell> {
    use crate::MetalBackend;
    use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};
    use metal::MTLSize;

    let metal = MetalBackend::new().expect("Metal backend required");
    const SB: usize = 256;
    const Q4K_SB: usize = 144;
    const Q6K_SB: usize = 210;
    const ROOFLINE_GB_S: f64 = 367.0;

    // (label, N = output rows, K = reduction)
    let shapes: [(&'static str, usize, usize); 6] = [
        ("gemma down  2560x10240", 2560, 10240),
        ("gemma Wo    2560x8192", 2560, 8192),
        ("K3 KDA attn 12288x7168", 12288, 7168),
        ("K3 shared exp 6144x7168", 6144, 7168),
        ("K3 latent up 7168x3584", 7168, 3584),
        ("K3 expert w2 3584x3072", 3584, 3072),
    ];

    let mut out = Vec::new();
    println!(
        "{:<24} {:<6} {:>9} {:>9} {:>9} {:>6}",
        "shape", "kernel", "packed MB", "cold ms", "cold GB/s", "eta"
    );
    println!("{}", "-".repeat(70));

    for (label, n, k) in shapes {
        for kernel in ["q4k", "q6k"] {
            let (sb_bytes, handle) = match kernel {
                "q4k" => (Q4K_SB, &metal.quant.q4k_matvec_pipeline),
                _ => (Q6K_SB, &metal.quant.q6k_matvec_pipeline),
            };
            let mb = (n * (k / SB * sb_bytes)) as f64 / 1e6;
            let x = synth_f32(k, 0.5);
            let xb = metal.bufs().transient_from_f32(&x);
            let ob = metal.bufs().output((n * 4) as u64);
            let n_tgs = (n as u64).div_ceil(handle.rows_per_tg);
            let n_val = n as u32;
            let k_val = k as u32;

            // 8 distinct weight buffers: the working set must exceed L2 so the
            // number is DRAM bandwidth, not cache.
            let cold_n = n_layers.min(8);
            let weights: Vec<_> = (0..cold_n)
                .map(|i| {
                    let f = synth_f32(n * k, 0.1 + i as f32 * 0.05);
                    let q = if kernel == "q4k" {
                        quantize_q4_k(&f)
                    } else {
                        quantize_q6_k(&f)
                    };
                    // See the note above: a temporary must not go through the
                    // address-keyed cache, or the rotation is not a rotation.
                    metal.bufs().uncached_bytes(&q)
                })
                .collect();

            let mut times: Vec<f64> = Vec::with_capacity(iters);
            for i in 0..warmup + iters {
                let t = std::time::Instant::now();
                let cmd = metal.queue().new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                for layer in 0..n_layers {
                    enc.set_compute_pipeline_state(&handle.state);
                    enc.set_buffer(0, Some(&weights[layer % cold_n]), 0);
                    enc.set_buffer(1, Some(&xb), 0);
                    enc.set_buffer(2, Some(&ob), 0);
                    enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(4, 4, &k_val as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(n_tgs, 1, 1),
                        MTLSize::new(handle.threads_per_tg, 1, 1),
                    );
                }
                enc.end_encoding();
                cmd.commit();
                crate::cb_status::wait_or_abort(
                    cmd,
                    "crates/larql-compute-metal/src/diag/kernel_profile/census.rs:123",
                );
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                if i >= warmup {
                    times.push(ms / n_layers as f64);
                }
            }
            let cold_ms = mean(&times);
            let cold_gbs = mb / cold_ms;
            let eta = cold_gbs / ROOFLINE_GB_S;
            println!("{label:<24} {kernel:<6} {mb:>9.1} {cold_ms:>9.3} {cold_gbs:>9.1} {eta:>6.2}");
            out.push(ShapeCell {
                kernel: if kernel == "q4k" { "q4k" } else { "q6k" },
                shape: label,
                n,
                k,
                packed_mb: mb,
                cold_ms,
                cold_gbs,
                eta,
            });
        }
    }
    out
}
