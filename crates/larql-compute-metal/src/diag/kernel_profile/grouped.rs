//! Grouped-expert kernel profiling.
//!
//! Split out of `kernel_profile.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::measure::{
    mean, measure_batched, measure_isolated, measure_single_cmdbuf_batched, stddev, synth_f32,
};
#[allow(unused_imports)]
use super::*;

/// Grouped vs ungrouped routed-expert dispatch, both under the SAME batched
/// protocol, so the comparison isolates occupancy from commit amortisation.
///
/// A naive bench commits once per expert in the ungrouped arm and once total in
/// the grouped arm; the resulting ratio then mixes the scheduling gain with the
/// removal of 15 commit+waits. Batching both arms into one command buffer
/// removes that confound, and matches how `profile_shape_census` measured the
/// 0.64 this experiment is trying to explain.
///
/// Returns `(ungrouped_gbs, grouped_gbs)`, both counting the same expert bytes.
pub fn profile_grouped_experts(
    n: usize,
    k: usize,
    top_k: usize,
    batch: usize,
    warmup: usize,
    iters: usize,
) -> (f64, f64) {
    use crate::MetalBackend;
    use larql_compute::cpu::ops::q4_common::quantize_q6_k;
    use metal::MTLSize;

    let metal = MetalBackend::new().expect("Metal backend required");

    let mut bank: Vec<u8> = Vec::new();
    let mut offsets: Vec<u32> = Vec::new();
    let mut per_expert = 0usize;
    for e in 0..top_k {
        let q = quantize_q6_k(&synth_f32(n * k, 0.1 + e as f32 * 0.03));
        per_expert = q.len();
        offsets.push(bank.len() as u32);
        bank.extend_from_slice(&q);
    }
    let total_mb = (per_expert * top_k) as f64 / 1e6;
    let x = synth_f32(k, 0.5);

    let wb = metal.bufs().uncached_bytes(&bank);
    let off_bytes: Vec<u8> = offsets.iter().flat_map(|o| o.to_ne_bytes()).collect();
    let offb = metal.bufs().uncached_bytes(&off_bytes);
    let xb = metal.bufs().transient_from_f32(&x);
    let out_single = metal.bufs().output((n * 4) as u64);
    let out_group = metal.bufs().output((top_k * n * 4) as u64);
    let n_val = n as u32;
    let k_val = k as u32;

    let solo = &metal.quant.q6k_matvec_pipeline;
    let grp = &metal.quant.q6k_grouped_experts_pipeline;
    let tiles_solo = (n as u64).div_ceil(solo.rows_per_tg);
    let tiles_grp = (n as u64).div_ceil(grp.rows_per_tg);

    let ungrouped_ms = {
        let mut times = Vec::new();
        for i in 0..warmup + iters {
            let t = std::time::Instant::now();
            let cmd = metal.queue().new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            for _ in 0..batch {
                for &off in &offsets {
                    enc.set_compute_pipeline_state(&solo.state);
                    enc.set_buffer(0, Some(&wb), off as u64);
                    enc.set_buffer(1, Some(&xb), 0);
                    enc.set_buffer(2, Some(&out_single), 0);
                    enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(4, 4, &k_val as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(tiles_solo, 1, 1),
                        MTLSize::new(solo.threads_per_tg, 1, 1),
                    );
                }
            }
            enc.end_encoding();
            cmd.commit();
            crate::cb_status::wait_or_abort(
                cmd,
                "crates/larql-compute-metal/src/diag/kernel_profile/grouped.rs:84",
            );
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            if i >= warmup {
                times.push(ms / batch as f64);
            }
        }
        mean(&times)
    };

    let grouped_ms = {
        let mut times = Vec::new();
        for i in 0..warmup + iters {
            let t = std::time::Instant::now();
            let cmd = metal.queue().new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            for _ in 0..batch {
                enc.set_compute_pipeline_state(&grp.state);
                enc.set_buffer(0, Some(&wb), 0);
                enc.set_buffer(1, Some(&offb), 0);
                enc.set_buffer(2, Some(&xb), 0);
                enc.set_buffer(3, Some(&out_group), 0);
                enc.set_bytes(4, 4, &n_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &k_val as *const u32 as *const std::ffi::c_void);
                let x_stride: u32 = 0; // shared-input regime for this bench
                enc.set_bytes(6, 4, &x_stride as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(tiles_grp, top_k as u64, 1),
                    MTLSize::new(grp.threads_per_tg, 1, 1),
                );
            }
            enc.end_encoding();
            cmd.commit();
            crate::cb_status::wait_or_abort(
                cmd,
                "crates/larql-compute-metal/src/diag/kernel_profile/grouped.rs:116",
            );
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            if i >= warmup {
                times.push(ms / batch as f64);
            }
        }
        mean(&times)
    };

    (total_mb / ungrouped_ms, total_mb / grouped_ms)
}
