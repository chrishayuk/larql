//! `MatMul` impl + private encoder helpers shared by `f32_gemv` and
//! `f16_gemv` (threshold-gated and force variants).

use ndarray::{Array2, ArrayView2};
use std::sync::atomic::Ordering;

use crate::MetalBackend;
use larql_compute::backend::{MatMul, MatMulOp};

impl MatMul for MetalBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        self.f32_ops.matmul(
            &self.queue,
            &self.bufs,
            a,
            b,
            self.flop_threshold.load(Ordering::Relaxed),
        )
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        self.f32_ops.matmul_transb(
            &self.queue,
            &self.bufs,
            a,
            b,
            self.flop_threshold.load(Ordering::Relaxed),
        )
    }

    fn f32_gemv(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k {
            return None;
        }
        // Fall back below the GPU threshold — small gemvs are dominated by
        // dispatch overhead.
        if 2 * n * k < self.flop_threshold.load(Ordering::Relaxed) {
            return None;
        }
        self.encode_f32_gemv(w, x)
    }

    fn f32_gemv_force(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (_n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k {
            return None;
        }
        self.encode_f32_gemv(w, x)
    }

    fn f16_gemv(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        if w_f16.len() < n * k * 2 || x.len() != k {
            return None;
        }
        if 2 * n * k < self.flop_threshold.load(Ordering::Relaxed) {
            return None;
        }
        self.encode_f16_gemv(w_f16, x, n, k)
    }

    fn f16_gemv_force(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        if w_f16.len() < n * k * 2 || x.len() != k {
            return None;
        }
        self.encode_f16_gemv(w_f16, x, n, k)
    }

    fn f32_gemv_topk1(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<(u32, f32)> {
        MetalBackend::f32_gemv_topk1(self, w, x)
    }

    fn f16_gemv_topk1(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<(u32, f32)> {
        MetalBackend::f16_gemv_topk1(self, w_f16, x, n, k)
    }

    fn f16_gemv_topk(
        &self,
        w_f16: &[u8],
        x: &[f32],
        n: usize,
        k: usize,
        top_k: usize,
    ) -> Option<Vec<(u32, f32)>> {
        MetalBackend::f16_gemv_topk(self, w_f16, x, n, k, top_k)
    }

    fn matmul_batch(&self, ops: &[MatMulOp]) -> Vec<Array2<f32>> {
        ops.iter()
            .map(|op| {
                if op.transpose_b {
                    self.matmul_transb(op.a.view(), op.b.view())
                } else {
                    self.matmul(op.a.view(), op.b.view())
                }
            })
            .collect()
    }
}

impl MetalBackend {
    /// Shared GPU dispatch body for `f32_gemv` (threshold-gated) and
    /// `f32_gemv_force` (direct). Kept inherent so the 30+ lines of
    /// Metal plumbing aren't duplicated.
    fn encode_f32_gemv(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k {
            return None;
        }
        let w_buf = match w.as_slice() {
            Some(s) => self.bufs.get_f32(s),
            None => {
                let owned = w.as_standard_layout().into_owned();
                self.bufs.transient_from_f32(owned.as_slice().unwrap())
            }
        };
        let x_buf = self.bufs.transient_from_f32(x);
        let out_buf = self.bufs.output((n * 4) as u64);

        // Geometry travels with the f32_gemv KernelHandle.
        let kernel = &self.f32_gemv_pipeline;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let num_tgs = (n as u64).div_ceil(kernel.rows_per_tg);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);
        enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs, 1, 1),
            metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // `try_read_buffer_f32` returns `None` if the GPU ran out of
        // memory and `contents()` is null — callers fall back to CPU
        // rather than crash. Hit by the lm-head 2.5 GB bench shape on
        // memory-constrained CI runners.
        crate::buffers::try_read_buffer_f32(&out_buf, n)
    }

    /// GPU gemv → GPU argmax, returning (token_id, score) without a 1MB readback.
    ///
    /// Replaces the three-step `f32_gemv` + read 262K floats + CPU argmax with:
    /// 1. f32_gemv kernel → scores buffer (stays on GPU)
    /// 2. f32_argmax_partial → 1024 (val, idx) partial results (8 KB)
    /// 3. Read back 8 KB, CPU reduces 1024 candidates (~1 µs)
    ///
    /// Saves ~0.33ms (1MB readback eliminated). Used by lm_head top-1 path.
    pub fn f32_gemv_topk1(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<(u32, f32)> {
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k || n == 0 {
            return None;
        }

        let w_buf = match w.as_slice() {
            Some(s) => self.bufs.get_f32(s),
            None => {
                let owned = w.as_standard_layout().into_owned();
                self.bufs.transient_from_f32(owned.as_slice().unwrap())
            }
        };
        let x_buf = self.bufs.transient_from_f32(x);
        let scores = self.bufs.output((n * 4) as u64);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        let kh = &self.f32_gemv_pipeline;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let gemv_tgs = (n as u64).div_ceil(kh.rows_per_tg);
        enc.set_compute_pipeline_state(&kh.state);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&scores), 0);
        enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(gemv_tgs, 1, 1),
            metal::MTLSize::new(kh.threads_per_tg, 1, 1),
        );

        let (partial_vals, partial_idxs, n_partials) = self.encode_argmax_partial(enc, &scores, n);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Self::reduce_argmax_partial(&partial_vals, &partial_idxs, n_partials)
    }

    /// f16 gemv + GPU argmax. Mirrors `f32_gemv_topk1` for the tied-embed
    /// lm_head path on Gemma 3/4 (mmap'd `embeddings.bin` reused as f16
    /// lm_head). Saves the 1MB readback + 262K-element CPU sort that
    /// `f16_gemv` + `top_k_sorted` would otherwise spend on each greedy
    /// decode step.
    pub fn f16_gemv_topk1(
        &self,
        w_f16: &[u8],
        x: &[f32],
        n: usize,
        k: usize,
    ) -> Option<(u32, f32)> {
        if w_f16.len() < n * k * 2 || x.len() != k || n == 0 {
            return None;
        }
        let w_buf = self.bufs.get_bytes(w_f16);
        let x_buf = self.bufs.transient_from_f32(x);
        let scores = self.bufs.output((n * 4) as u64);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        let kh = &self.f16_gemv_pipeline;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let gemv_tgs = (n as u64).div_ceil(kh.rows_per_tg);
        enc.set_compute_pipeline_state(&kh.state);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&scores), 0);
        enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(gemv_tgs, 1, 1),
            metal::MTLSize::new(kh.threads_per_tg, 1, 1),
        );

        let (partial_vals, partial_idxs, n_partials) = self.encode_argmax_partial(enc, &scores, n);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Self::reduce_argmax_partial(&partial_vals, &partial_idxs, n_partials)
    }

    /// Encode `f32_argmax_partial` over `scores[..n]` into `enc`. Returns
    /// the (vals_buf, idxs_buf, n_partials) needed for `reduce_argmax_partial`
    /// once the command buffer commits. The encoder is left active for any
    /// downstream dispatches the caller wants to add (none today).
    pub(crate) fn encode_argmax_partial(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        scores: &metal::Buffer,
        n: usize,
    ) -> (metal::Buffer, metal::Buffer, usize) {
        // Same TG width as `encode_topk_partial` — flows from the Rust
        // constant the templated MSL is built from.
        let tg_sz = crate::shaders::f32_gemv::PARTIAL_TG_SZ;
        let argmax_tgs = (n as u64).div_ceil(tg_sz);
        let partial_vals = self.bufs.output(argmax_tgs * 4);
        let partial_idxs = self.bufs.output(argmax_tgs * 4);
        let n_u32 = n as u32;
        enc.set_compute_pipeline_state(&self.f32_argmax_partial_pipeline);
        enc.set_buffer(0, Some(scores), 0);
        enc.set_buffer(1, Some(&partial_vals), 0);
        enc.set_buffer(2, Some(&partial_idxs), 0);
        enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(argmax_tgs, 1, 1),
            metal::MTLSize::new(tg_sz, 1, 1),
        );
        (partial_vals, partial_idxs, argmax_tgs as usize)
    }

    /// CPU side of the argmax_partial pipeline: read back ≤1024 partial
    /// (val, idx) pairs (≤8 KB) and pick the global maximum. The caller
    /// must have committed and waited on the command buffer that wrote
    /// `partial_vals` / `partial_idxs`.
    pub(crate) fn reduce_argmax_partial(
        partial_vals: &metal::Buffer,
        partial_idxs: &metal::Buffer,
        n_partials: usize,
    ) -> Option<(u32, f32)> {
        let vals = crate::buffers::read_buffer_f32(partial_vals, n_partials);
        let idxs_raw = unsafe {
            let ptr = partial_idxs.contents() as *const u32;
            std::slice::from_raw_parts(ptr, n_partials)
        };
        let (best_idx, best_val) = vals
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            });
        if best_val == f32::NEG_INFINITY {
            return None;
        }
        Some((idxs_raw[best_idx], best_val))
    }

    /// Encode `f32_topk_partial` over `scores[..n]`. Each TG of 256 threads
    /// emits `K_TOPK` (val, idx) pairs sorted descending; the caller merges
    /// `num_tgs × K_TOPK` candidates on CPU. Returns
    /// `(partial_vals, partial_idxs, num_tgs)`.
    pub(crate) fn encode_topk_partial(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        scores: &metal::Buffer,
        n: usize,
    ) -> (metal::Buffer, metal::Buffer, usize) {
        // TG width and per-TG K both flow from the same Rust constants the
        // MSL source is templated from; can't drift.
        let tg_sz = crate::shaders::f32_gemv::PARTIAL_TG_SZ;
        let k_topk = crate::shaders::f32_gemv::K_TOPK as u64;
        let topk_tgs = (n as u64).div_ceil(tg_sz);
        let partial_vals = self.bufs.output(topk_tgs * k_topk * 4);
        let partial_idxs = self.bufs.output(topk_tgs * k_topk * 4);
        let n_u32 = n as u32;
        enc.set_compute_pipeline_state(&self.f32_topk_partial_pipeline);
        enc.set_buffer(0, Some(scores), 0);
        enc.set_buffer(1, Some(&partial_vals), 0);
        enc.set_buffer(2, Some(&partial_idxs), 0);
        enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(topk_tgs, 1, 1),
            metal::MTLSize::new(tg_sz, 1, 1),
        );
        (partial_vals, partial_idxs, topk_tgs as usize)
    }

    /// CPU final reduction of `num_tgs × K_TOPK` partial top-K candidates
    /// into the caller's requested `top_k`. Uses a size-`top_k` min-heap.
    /// Returns sorted descending `(token_id, score)` pairs.
    pub(crate) fn reduce_topk_partial(
        partial_vals: &metal::Buffer,
        partial_idxs: &metal::Buffer,
        num_tgs: usize,
        top_k: usize,
    ) -> Vec<(u32, f32)> {
        let k_topk = crate::shaders::f32_gemv::K_TOPK;
        let total = num_tgs * k_topk;
        let vals = crate::buffers::read_buffer_f32(partial_vals, total);
        let idxs = unsafe {
            let ptr = partial_idxs.contents() as *const u32;
            std::slice::from_raw_parts(ptr, total)
        };

        let k = top_k.min(total);
        if k == 0 {
            return Vec::new();
        }

        let mut heap: Vec<(f32, u32)> = Vec::with_capacity(k + 1);

        fn sift_down(h: &mut [(f32, u32)], mut i: usize) {
            let n = h.len();
            loop {
                let mut smallest = i;
                let l = 2 * i + 1;
                let r = 2 * i + 2;
                if l < n && h[l].0 < h[smallest].0 {
                    smallest = l;
                }
                if r < n && h[r].0 < h[smallest].0 {
                    smallest = r;
                }
                if smallest == i {
                    break;
                }
                h.swap(i, smallest);
                i = smallest;
            }
        }

        for (i, &v) in vals.iter().enumerate() {
            if !v.is_finite() {
                continue;
            }
            // Skip the sentinel-index slots emitted by trailing TGs that
            // had nothing to rank (idx = ~0u from the masked-out lanes).
            if idxs[i] == u32::MAX {
                continue;
            }
            if heap.len() < k {
                heap.push((v, idxs[i]));
                if heap.len() == k {
                    for j in (0..k / 2).rev() {
                        sift_down(&mut heap, j);
                    }
                }
            } else if v > heap[0].0 {
                heap[0] = (v, idxs[i]);
                sift_down(&mut heap, 0);
            }
        }
        if heap.len() < k && heap.len() > 1 {
            for j in (0..heap.len() / 2).rev() {
                sift_down(&mut heap, j);
            }
        }
        heap.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        heap.into_iter().map(|(s, i)| (i, s)).collect()
    }

    /// f16 gemv + GPU partial top-K. Mirrors `f16_gemv_topk1` but produces
    /// the top `top_k` scores in one round-trip (top_k ≤ K_TOPK = 8).
    /// Returns `None` if `top_k` exceeds the per-TG capacity — the caller
    /// then falls back to `f16_gemv` + CPU sort.
    pub fn f16_gemv_topk(
        &self,
        w_f16: &[u8],
        x: &[f32],
        n: usize,
        k: usize,
        top_k: usize,
    ) -> Option<Vec<(u32, f32)>> {
        if top_k == 0 || top_k > crate::shaders::f32_gemv::K_TOPK {
            return None;
        }
        if w_f16.len() < n * k * 2 || x.len() != k || n == 0 {
            return None;
        }
        let w_buf = self.bufs.get_bytes(w_f16);
        let x_buf = self.bufs.transient_from_f32(x);
        let scores = self.bufs.output((n * 4) as u64);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        let kh = &self.f16_gemv_pipeline;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let gemv_tgs = (n as u64).div_ceil(kh.rows_per_tg);
        enc.set_compute_pipeline_state(&kh.state);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&scores), 0);
        enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(gemv_tgs, 1, 1),
            metal::MTLSize::new(kh.threads_per_tg, 1, 1),
        );

        let (partial_vals, partial_idxs, num_tgs) = self.encode_topk_partial(enc, &scores, n);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Some(Self::reduce_topk_partial(
            &partial_vals,
            &partial_idxs,
            num_tgs,
            top_k,
        ))
    }

    /// Q4_K stride-32 matvec → full f32 scores. Same Q4_K input format
    /// as `q4k_matvec`, but uses the shader at
    /// `shaders::q4k_matvec_stride32` whose 32-lane reduction matches
    /// `f16_gemv`'s tree (lane k accumulates stride-32 elements then
    /// `simd_sum`). Required for the LM head when the production
    /// `q4k_matvec`'s block-aware lane split drifts enough vs CPU to
    /// flip top-1 on close-call tokens.
    pub fn q4k_matvec_stride32(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        if hidden == 0 || !hidden.is_multiple_of(256) {
            return None;
        }
        let kh = &self.quant.q4k_matvec_stride32_pipeline;
        let buf_w = self.bufs.get_bytes(q4k_data);
        let buf_x = self.bufs.transient_from_f32(x);
        let buf_out = self.bufs.output((num_rows * 4) as u64);
        let n = num_rows as u32;
        let k = hidden as u32;
        let num_tgs = (num_rows as u64).div_ceil(kh.rows_per_tg);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&kh.state);
        enc.set_buffer(0, Some(&buf_w), 0);
        enc.set_buffer(1, Some(&buf_x), 0);
        enc.set_buffer(2, Some(&buf_out), 0);
        enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs, 1, 1),
            metal::MTLSize::new(kh.threads_per_tg, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        Some(crate::buffers::read_buffer_f32(&buf_out, num_rows))
    }

    /// Shared dispatch body for f16-weight gemv (behind both trait
    /// variants: threshold-gated `f16_gemv` and direct `f16_gemv_force`).
    fn encode_f16_gemv(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        let w_buf = self.bufs.get_bytes(w_f16);
        let x_buf = self.bufs.transient_from_f32(x);
        let out_buf = self.bufs.output((n * 4) as u64);

        // Geometry travels with the f16_gemv KernelHandle.
        let kernel = &self.f16_gemv_pipeline;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let num_tgs = (n as u64).div_ceil(kernel.rows_per_tg);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);
        enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs, 1, 1),
            metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        Some(crate::buffers::read_buffer_f32(&out_buf, n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `f32_topk_partial` correctness against synthetic scores. Exercises:
    ///   - the partial last TG (vocab not divisible by 256), which is the
    ///     case that broke `q4_matvec_topk` parity in development.
    ///   - vocab smaller than one TG (single partial TG only).
    ///
    /// The Q4/f16 integration tests cover the typical "full TGs" path; this
    /// pins the boundary cases that those don't reach.
    #[test]
    fn topk_partial_handles_partial_last_tg() {
        let metal = match MetalBackend::new() {
            Some(m) => m,
            None => return, // not on Metal-capable hardware
        };

        // 4 full TGs + 1 partial (1024 + 100 = 1124). Plant maxima at 700
        // (full TG) and 1100 (partial last TG) so both must be picked.
        let n = 1124usize;
        let mut scores = vec![0.0f32; n];
        for (i, s) in scores.iter_mut().enumerate() {
            *s = (i as f32) * 0.001;
        }
        scores[700] = 999.0;
        scores[1100] = 998.0;

        let scores_buf = metal.bufs.transient_from_f32(&scores);
        let cmd = metal.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let (vals, idxs, num_tgs) = metal.encode_topk_partial(enc, &scores_buf, n);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let hits = MetalBackend::reduce_topk_partial(&vals, &idxs, num_tgs, 5);
        assert_eq!(hits.len(), 5);
        let top_idxs: Vec<u32> = hits.iter().map(|(i, _)| *i).collect();
        assert!(
            top_idxs.contains(&700),
            "missing planted argmax 700: {:?}",
            top_idxs
        );
        assert!(
            top_idxs.contains(&1100),
            "missing planted second-max 1100 (in partial TG): {:?}",
            top_idxs
        );

        // vocab smaller than one TG (200 elements, single partial TG).
        let n = 200usize;
        let mut scores = vec![0.0f32; n];
        for (i, s) in scores.iter_mut().enumerate() {
            *s = -(i as f32);
        }
        scores[42] = 5.0;
        scores[99] = 4.0;
        let scores_buf = metal.bufs.transient_from_f32(&scores);
        let cmd = metal.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let (vals, idxs, num_tgs) = metal.encode_topk_partial(enc, &scores_buf, n);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let hits = MetalBackend::reduce_topk_partial(&vals, &idxs, num_tgs, 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, 42);
        assert_eq!(hits[1].0, 99);
    }

    /// `top_k > K_TOPK` is rejected at the public method (returns `None`)
    /// so the reducer is never called with mismatched K. Sanity-check the
    /// public-facing wrappers honour the `K_TOPK = 8` ceiling.
    #[test]
    fn topk_capacity_ceiling_enforced() {
        let metal = match MetalBackend::new() {
            Some(m) => m,
            None => return,
        };
        let n = 512;
        let k = 256;
        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.001).cos()).collect();
        let w_f16 = larql_models::quant::half::encode_f16(&vec![0.5f32; n * k]);
        // top_k = 0 and top_k > K_TOPK both yield None — caller falls back.
        assert!(metal.f16_gemv_topk(&w_f16, &x, n, k, 0).is_none());
        assert!(metal.f16_gemv_topk(&w_f16, &x, n, k, 9).is_none());
        // top_k within range produces a result.
        let hits = metal
            .f16_gemv_topk(&w_f16, &x, n, k, 8)
            .expect("top_k=8 is exactly K_TOPK and must be accepted");
        assert_eq!(hits.len(), 8);
    }

    // ─── End-to-end trait-method coverage for the gemv family ───

    fn backend() -> MetalBackend {
        MetalBackend::new().expect("Metal device available on test host")
    }

    /// Width sized above the calibration FLOP threshold so the
    /// `2*n*k < flop_threshold` guard doesn't short-circuit the
    /// dispatch. Apple Silicon calibrates to ~5K-50K depending on
    /// device, so 256×256 = 131K flops is safe.
    fn gemv_shapes() -> (usize, usize) {
        (256, 256)
    }

    /// `f32_gemv` returns `None` when the input vector length doesn't
    /// match the matrix `k` dim (line 34 early-out).
    #[test]
    fn f32_gemv_rejects_mismatched_x_length() {
        let m = backend();
        let w = ndarray::Array2::<f32>::zeros((16, 32));
        let x = vec![0.0f32; 31]; // wrong length
        assert!(m.f32_gemv(w.view(), &x).is_none());
    }

    /// `f32_gemv` returns `None` when the operation falls below the
    /// FLOP threshold (line 39 — CPU fallback, dispatch overhead would
    /// dominate).
    #[test]
    fn f32_gemv_falls_back_below_flop_threshold() {
        let m = backend();
        // 2×2 gemv → 8 FLOPs, well under any sane threshold.
        let w = ndarray::Array2::<f32>::zeros((2, 2));
        let x = vec![0.0f32; 2];
        assert!(m.f32_gemv(w.view(), &x).is_none());
    }

    /// `f32_gemv` runs the GPU path on large-enough shapes.  Force the
    /// threshold low so the test doesn't depend on the calibration
    /// number on whatever host is running.
    #[test]
    fn f32_gemv_dispatches_above_threshold() {
        let m = backend();
        m.set_flop_threshold(1);
        let (n, k) = gemv_shapes();
        let w_data: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.0001).collect();
        let w = ndarray::Array2::from_shape_vec((n, k), w_data).unwrap();
        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = m
            .f32_gemv(w.view(), &x)
            .expect("f32_gemv above threshold returns Some");
        assert_eq!(out.len(), n);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// `f32_gemv_force` bypasses the threshold guard and always
    /// dispatches when shapes agree (covers lines 44-50).
    #[test]
    fn f32_gemv_force_bypasses_threshold() {
        let m = backend();
        // Tiny shape that f32_gemv would short-circuit on.
        let w = ndarray::Array2::<f32>::from_shape_vec((4, 4), vec![1.0f32; 16]).unwrap();
        let x = vec![1.0f32; 4];
        let out = m
            .f32_gemv_force(w.view(), &x)
            .expect("f32_gemv_force dispatches regardless of threshold");
        assert_eq!(out.len(), 4);
    }

    /// `f32_gemv_force` still rejects shape mismatches.
    #[test]
    fn f32_gemv_force_rejects_mismatched_x_length() {
        let m = backend();
        let w = ndarray::Array2::<f32>::zeros((4, 8));
        let x = vec![0.0f32; 7];
        assert!(m.f32_gemv_force(w.view(), &x).is_none());
    }

    /// `encode_f32_gemv` handles non-contiguous `ArrayView2` inputs by
    /// materialising a standard-layout copy (lines 113-114).
    #[test]
    fn f32_gemv_force_handles_non_contiguous_input() {
        let m = backend();
        // Build a column-major view (non-standard layout) by transposing.
        let raw: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let w_t = ndarray::Array2::from_shape_vec((4, 4), raw)
            .unwrap()
            .reversed_axes();
        assert!(w_t.as_slice().is_none(), "test setup: non-standard layout");
        let x = vec![1.0f32; 4];
        let out = m
            .f32_gemv_force(w_t.view(), &x)
            .expect("non-contiguous gemv still dispatches");
        assert_eq!(out.len(), 4);
    }

    /// `f16_gemv` shape-mismatch early-outs (lines 53-54, 64).
    #[test]
    fn f16_gemv_rejects_invalid_shapes() {
        let m = backend();
        let w = vec![0u8; 7]; // too short for n*k*2
        let x = vec![0.0f32; 4];
        assert!(m.f16_gemv(&w, &x, 4, 4).is_none());
        assert!(m.f16_gemv_force(&w, &x, 4, 4).is_none());
        // x.len() != k — even on `force`.
        let w = larql_models::quant::half::encode_f16(&[0.0f32; 16]);
        let x = vec![0.0f32; 3];
        assert!(m.f16_gemv_force(&w, &x, 4, 4).is_none());
    }

    /// `f16_gemv` falls back below the FLOP threshold (line 57).
    #[test]
    fn f16_gemv_falls_back_below_flop_threshold() {
        let m = backend();
        let w_f16 = larql_models::quant::half::encode_f16(&[0.5f32; 4]);
        let x = vec![1.0f32; 2];
        assert!(m.f16_gemv(&w_f16, &x, 2, 2).is_none());
    }

    /// `f32_gemv_topk1` and `f16_gemv_topk1` wrap the inherent helpers
    /// — they're the trait surface (lines 69-77, 85).
    #[test]
    fn gemv_topk1_trait_wrappers_route_to_inherent_helpers() {
        let m = backend();
        let (n, k) = gemv_shapes();
        let w_data: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.0001).collect();
        let w = ndarray::Array2::from_shape_vec((n, k), w_data.clone()).unwrap();
        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).sin()).collect();

        let (idx, val) = m
            .f32_gemv_topk1(w.view(), &x)
            .expect("topk1 returns Some on valid shape");
        assert!((idx as usize) < n);
        assert!(val.is_finite());

        let w_f16 = larql_models::quant::half::encode_f16(&w_data);
        let (idx2, val2) = m
            .f16_gemv_topk1(&w_f16, &x, n, k)
            .expect("f16 topk1 returns Some on valid shape");
        assert!((idx2 as usize) < n);
        assert!(val2.is_finite());

        // f16_gemv_topk trait wrapper round-trips top_k=4.
        let hits = m
            .f16_gemv_topk(&w_f16, &x, n, k, 4)
            .expect("top_k=4 returns Some");
        assert_eq!(hits.len(), 4);
    }

    /// `f32_gemv_topk1` rejects shape mismatch + n=0 (lines 159-160).
    #[test]
    fn f32_gemv_topk1_rejects_invalid_shapes() {
        let m = backend();
        let w = ndarray::Array2::<f32>::zeros((4, 8));
        let x = vec![0.0f32; 7]; // wrong length
        assert!(m.f32_gemv_topk1(w.view(), &x).is_none());
        // n = 0
        let w_empty = ndarray::Array2::<f32>::zeros((0, 8));
        let x_ok = vec![0.0f32; 8];
        assert!(m.f32_gemv_topk1(w_empty.view(), &x_ok).is_none());
    }

    /// `matmul_batch` dispatches each op through `matmul` or
    /// `matmul_transb` based on `transpose_b` (lines 88-98).
    #[test]
    fn matmul_batch_dispatches_per_op_transpose() {
        let m = backend();
        let a = ndarray::Array2::from_shape_vec((2, 3), vec![1.0f32; 6]).unwrap();
        let b = ndarray::Array2::from_shape_vec((3, 4), vec![0.5f32; 12]).unwrap();
        let b_t = ndarray::Array2::from_shape_vec((4, 3), vec![0.5f32; 12]).unwrap();
        let ops = vec![
            MatMulOp {
                a: a.clone(),
                b: b.clone(),
                transpose_b: false,
            },
            MatMulOp {
                a: a.clone(),
                b: b_t.clone(),
                transpose_b: true,
            },
        ];
        let outs = m.matmul_batch(&ops);
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0].shape(), &[2, 4]);
        assert_eq!(outs[1].shape(), &[2, 4]);
    }

    /// `reduce_argmax_partial` returns `None` when every partial is
    /// `NaN`/`INF` (line 297-298 early-out).
    #[test]
    fn reduce_argmax_partial_returns_none_for_all_non_finite() {
        let m = backend();
        // Build vals = [NaN, NaN, ...] and idxs = [0, 1, ...].
        let vals: Vec<f32> = (0..4).map(|_| f32::NAN).collect();
        let idxs: Vec<u32> = (0..4u32).collect();
        let vals_buf = m.bufs.transient_from_f32(&vals);
        let idxs_buf = m.bufs.transient_from_bytes(unsafe {
            std::slice::from_raw_parts(idxs.as_ptr() as *const u8, idxs.len() * 4)
        });
        let out = MetalBackend::reduce_argmax_partial(&vals_buf, &idxs_buf, vals.len());
        assert!(out.is_none());
    }

    /// `reduce_topk_partial` with `k=0` returns an empty vec (line 351-352).
    /// Note: the reducer reads `num_tgs * K_TOPK` floats from the buffer
    /// before clamping `k`, so we still need to provide a correctly-sized
    /// vals/idxs buffer.
    #[test]
    fn reduce_topk_partial_zero_k_returns_empty() {
        let m = backend();
        let k_topk = crate::shaders::f32_gemv::K_TOPK;
        let vals = vec![1.0f32; k_topk];
        let idxs = vec![0u32; k_topk];
        let vals_buf = m.bufs.transient_from_f32(&vals);
        let idxs_bytes: Vec<u8> = idxs.iter().flat_map(|v| v.to_le_bytes()).collect();
        let idxs_buf = m.bufs.transient_from_bytes(&idxs_bytes);
        let out = MetalBackend::reduce_topk_partial(&vals_buf, &idxs_buf, 1, 0);
        assert!(out.is_empty());
    }

    /// `reduce_topk_partial` skips non-finite vals + `u32::MAX`
    /// sentinel indices (lines 378-385).
    #[test]
    fn reduce_topk_partial_skips_invalid_entries() {
        let m = backend();
        let k_topk = crate::shaders::f32_gemv::K_TOPK;
        // Two TGs worth of partials. Plant valid (idx=5, val=10) in
        // the first slot, and fill the rest with non-finite + sentinel
        // garbage.  reduce_topk_partial should pick only the valid one.
        let mut vals: Vec<f32> = Vec::with_capacity(2 * k_topk);
        let mut idxs: Vec<u32> = Vec::with_capacity(2 * k_topk);
        // First slot: valid.
        vals.push(10.0);
        idxs.push(5);
        // Remainder of first TG: non-finite + sentinel.
        for _ in 1..k_topk {
            vals.push(f32::NAN);
            idxs.push(u32::MAX);
        }
        // Second TG: all sentinels.
        for _ in 0..k_topk {
            vals.push(0.0); // finite but idx is sentinel
            idxs.push(u32::MAX);
        }

        let vals_buf = m.bufs.transient_from_f32(&vals);
        let idxs_bytes: Vec<u8> = idxs.iter().flat_map(|v| v.to_le_bytes()).collect();
        let idxs_buf = m.bufs.transient_from_bytes(&idxs_bytes);
        let out = MetalBackend::reduce_topk_partial(&vals_buf, &idxs_buf, 2, 3);
        // Only 1 valid candidate; reducer returns a 1-element vec.
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, 5);
        assert_eq!(out[0].1, 10.0);
    }

    /// `reduce_topk_partial` with `k > total` clamps k to total —
    /// covers the final-heapify branch (lines 398-400) when the heap
    /// fills incompletely.
    #[test]
    fn reduce_topk_partial_clamps_k_to_total_and_finalizes_heap() {
        let m = backend();
        let k_topk = crate::shaders::f32_gemv::K_TOPK;
        // One TG of partials. Plant 4 valid scores: (idx=0, val=1.0),
        // (idx=1, val=3.0), (idx=2, val=2.0), (idx=3, val=4.0).
        let mut vals: Vec<f32> = vec![1.0, 3.0, 2.0, 4.0];
        let mut idxs: Vec<u32> = vec![0, 1, 2, 3];
        while vals.len() < k_topk {
            vals.push(0.0);
            idxs.push(u32::MAX);
        }
        let vals_buf = m.bufs.transient_from_f32(&vals);
        let idxs_bytes: Vec<u8> = idxs.iter().flat_map(|v| v.to_le_bytes()).collect();
        let idxs_buf = m.bufs.transient_from_bytes(&idxs_bytes);

        // Request top-100, far more than the 4 finite entries available.
        // Reducer should clamp and return 4 sorted descending.
        let out = MetalBackend::reduce_topk_partial(&vals_buf, &idxs_buf, 1, 100);
        // total = num_tgs * K_TOPK = 8; clamped to 4 valid entries; sorted desc.
        // (Note: k is clamped to `total`, not to `valid_count`, so the
        // result may include up to `total` entries; here only 4 are valid.)
        assert!(out.len() <= 8 && !out.is_empty());
        // Top entry is (3, 4.0).
        assert_eq!(out[0], (3, 4.0));
    }
}
