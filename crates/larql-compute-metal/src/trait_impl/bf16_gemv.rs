//! The bf16-weight gemv dispatch, behind [`MatMul::bf16_gemv`].
//!
//! Split out of [`super::matmul`] rather than added to it: that file is
//! already past the workspace's per-file ceiling, and this kernel serves
//! a different weight representation from every gemv in it.
//!
//! The encoder body is deliberately the same shape as
//! `encode_f16_gemv` — pooled `x`/`out`, weights through the byte cache,
//! one command buffer, geometry read from the bound `KernelHandle`
//! rather than from the shader module's constants (the dispatch-geometry
//! mismatch class has cost this crate a 3.8x regression once already).
//!
//! [`MatMul::bf16_gemv`]: larql_compute::backend::MatMul::bf16_gemv

use crate::MetalBackend;

/// Bytes per bf16 code unit, on disk and in the device buffer alike.
const BF16_BYTES: usize = 2;

impl MetalBackend {
    /// Shared dispatch body for bf16-weight gemv (behind both trait
    /// variants: threshold-gated `bf16_gemv` and direct
    /// `bf16_gemv_force`).
    ///
    /// `w_bf16` is `n * k` little-endian `u16` codes, row-major — the
    /// checkpoint's own bytes, bound to the device without a widening
    /// pass. Returns `None` only when the pooled staging buffer cannot
    /// be mapped; shape validation belongs to the callers.
    pub(crate) fn encode_bf16_gemv(
        &self,
        w_bf16: &[u8],
        x: &[f32],
        n: usize,
        k: usize,
    ) -> Option<Vec<f32>> {
        let w_buf = self.bufs.get_bytes(w_bf16);
        // Pooled x and out, per the f16 path's own contract: a decode
        // step issues hundreds of gemvs and per-call device allocation
        // was a measurable share of its cost.
        let x_buf = self.bufs.output((x.len() * 4) as u64);
        let x_ptr = x_buf.contents() as *mut f32;
        if x_ptr.is_null() {
            return None;
        }
        // SAFETY: pooled buffer is at least x.len()*4 bytes, and it is
        // not bound to any encoder yet, so the GPU is not reading it.
        unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), x_ptr, x.len()) };
        let out_buf = self.bufs.output((n * 4) as u64);

        // Geometry travels with the bf16_gemv KernelHandle.
        let kernel = &self.bf16_gemv_pipeline;
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
        crate::cb_status::wait_or_abort(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/bf16_gemv.rs:71",
        );

        let result = crate::buffers::try_read_buffer_f32(&out_buf, n);
        self.bufs.recycle(out_buf);
        self.bufs.recycle(x_buf);
        result
    }

    /// One command buffer, one encoder, one input upload, N dispatches,
    /// one wait — same kernel and same per-dispatch arguments as the
    /// sequential path, so the results are bit-identical to N separate
    /// [`encode_bf16_gemv`](Self::encode_bf16_gemv) calls. The dispatches
    /// write distinct output buffers, so encoding them without barriers
    /// reorders nothing observable.
    ///
    /// Callers validate shapes; `weights` must be non-empty.
    pub(crate) fn encode_bf16_gemv_multi(
        &self,
        weights: &[(&[u8], usize, usize)],
        x: &[f32],
    ) -> Option<Vec<Vec<f32>>> {
        self.encode_bf16_gemv_multi_profiled(weights, x)
            .map(|(out, _gpu_ms)| out)
    }

    /// The batched submission, also returning the **GPU-side** window
    /// for the command buffer in ms. See
    /// [`bf16_grouped_experts_profiled`] for why a bandwidth number
    /// taken from wall time is a claim about the stack rather than the
    /// kernel.
    ///
    /// [`bf16_grouped_experts_profiled`]: Self::bf16_grouped_experts_profiled
    pub fn encode_bf16_gemv_multi_profiled(
        &self,
        weights: &[(&[u8], usize, usize)],
        x: &[f32],
    ) -> Option<(Vec<Vec<f32>>, f64)> {
        let x_buf = self.bufs.output((x.len() * 4) as u64);
        let x_ptr = x_buf.contents() as *mut f32;
        if x_ptr.is_null() {
            return None;
        }
        // SAFETY: pooled buffer is at least x.len()*4 bytes and has not
        // been bound to any encoder yet, so the GPU is not reading it.
        unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), x_ptr, x.len()) };

        let kernel = &self.bf16_gemv_pipeline;
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&kernel.state);
        // The weight buffers must outlive the wait; holding the cache
        // clones here guarantees that independent of encoder retention.
        let mut w_bufs = Vec::with_capacity(weights.len());
        let mut out_bufs = Vec::with_capacity(weights.len());
        for &(w, n, k) in weights {
            let w_buf = self.bufs.get_bytes(w);
            let out_buf = self.bufs.output((n * 4) as u64);
            let n_u32 = n as u32;
            let k_u32 = k as u32;
            enc.set_buffer(0, Some(&w_buf), 0);
            enc.set_buffer(1, Some(&x_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_bytes(3, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize::new((n as u64).div_ceil(kernel.rows_per_tg), 1, 1),
                metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
            );
            w_bufs.push(w_buf);
            out_bufs.push((out_buf, n));
        }
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_or_abort(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/bf16_gemv.rs:multi",
        );

        let gpu_ms = crate::decode::gpu_timing::gpu_elapsed_ms(cmd);
        let results: Option<Vec<Vec<f32>>> = out_bufs
            .iter()
            .map(|(buf, n)| crate::buffers::try_read_buffer_f32(buf, *n))
            .collect();
        for (buf, _) in out_bufs {
            self.bufs.recycle(buf);
        }
        self.bufs.recycle(x_buf);
        results.map(|r| (r, gpu_ms))
    }

    /// `w_bf16` holds at least `n * k` codes and `x` is exactly `k` long.
    ///
    /// Shared by both trait variants so the two cannot drift: a length
    /// check that passed on `force` but not on the threshold-gated entry
    /// would be an out-of-bounds device read reachable by one caller and
    /// not the other.
    pub(crate) fn bf16_gemv_shape_ok(w_bf16: &[u8], x: &[f32], n: usize, k: usize) -> bool {
        w_bf16.len() >= n * k * BF16_BYTES && x.len() == k
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::backend::MatMul;

    /// The exact widen the shader's `decode_bf16_metal` implements, and
    /// the same one `exec::cpu::kernels::FusedBf16` uses host-side.
    fn widen(code: u16) -> f32 {
        f32::from_bits((code as u32) << 16)
    }

    fn narrow(v: f32) -> u16 {
        (v.to_bits() >> 16) as u16
    }

    /// Row-major `[n, k]` bf16 codes as little-endian bytes, plus the
    /// exact f32 values they denote (never the pre-rounding values —
    /// the oracle must score the *stored* weights, not the intent).
    fn synth_weights(n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
        let values: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.017).sin() * 0.35 + ((i as f32) * 0.003).cos() * 0.15)
            .collect();
        let codes: Vec<u16> = values.iter().copied().map(narrow).collect();
        let exact: Vec<f32> = codes.iter().copied().map(widen).collect();
        let bytes: Vec<u8> = codes.iter().flat_map(|c| c.to_le_bytes()).collect();
        (bytes, exact)
    }

    /// Reduction-order noise ceiling, RELATIVE to the output vector's
    /// own scale. The GPU sums 32 lane partials in a tree with `fma`;
    /// the oracle sums serially with separate multiply and add. Nothing
    /// else differs — both read the same widened codes — so the only
    /// disagreement possible is reassociation.
    ///
    /// Measured on this host, not assumed: 6.8e-7 on the 64x512 case,
    /// 4.9e-7 on the ragged 13x100 one, i.e. a handful of f32 ULP. The
    /// ceiling keeps an order of magnitude of headroom over both. A
    /// wrong decode lands at rel ~1e0.
    const REL_NOISE_CEILING: f32 = 5e-6;

    /// Max absolute disagreement scaled by the oracle's own magnitude —
    /// an absolute bound would be a different test at every shape,
    /// because the dot's magnitude grows with K.
    fn rel_err(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(
            got.len(),
            want.len(),
            "length {} vs {}",
            got.len(),
            want.len()
        );
        let max_abs = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(scale > 0.0, "degenerate oracle: every output is zero");
        max_abs / scale
    }

    fn scalar_gemv(w: &[f32], x: &[f32], n: usize, k: usize) -> Vec<f32> {
        (0..n)
            .map(|row| {
                w[row * k..(row + 1) * k]
                    .iter()
                    .zip(x)
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect()
    }

    fn backend() -> MetalBackend {
        MetalBackend::new().expect("Metal device available on test host")
    }

    /// The load-bearing claim: the GPU computes what the bf16 codes
    /// denote. Scored against a scalar f32 dot over the *widened* codes,
    /// so any disagreement is the reduction order (~1e-6 at this K) and
    /// nothing else — a wrong decode (reading the codes as `half`, or
    /// shifting the wrong way) lands orders of magnitude past this.
    #[test]
    fn bf16_gemv_computes_what_the_codes_denote() {
        let m = backend();
        let (n, k) = (64, 512);
        let (bytes, exact) = synth_weights(n, k);
        let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.011).cos()).collect();

        let got = m
            .bf16_gemv_force(&bytes, &x, n, k)
            .expect("bf16 gemv dispatches on a valid shape");
        let want = scalar_gemv(&exact, &x, n, k);

        assert_eq!(got.len(), n);
        let rel = rel_err(&got, &want);
        assert!(
            rel < REL_NOISE_CEILING,
            "rel {rel:e} against the scalar oracle"
        );
    }

    /// A control for the test above: if the same bytes are read through
    /// the f16 kernel the answers must NOT agree. Without this, a shader
    /// that silently decoded as `half` could still pass a loose
    /// tolerance on a fixture whose values happened to be small.
    #[test]
    fn the_f16_kernel_on_the_same_bytes_disagrees() {
        let m = backend();
        let (n, k) = (64, 512);
        let (bytes, exact) = synth_weights(n, k);
        let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.011).cos()).collect();

        let as_bf16 = m.bf16_gemv_force(&bytes, &x, n, k).expect("bf16 arm");
        let as_f16 = m.f16_gemv_force(&bytes, &x, n, k).expect("f16 arm");
        let want = scalar_gemv(&exact, &x, n, k);

        let bf16_rel = rel_err(&as_bf16, &want);
        let f16_rel = rel_err(&as_f16, &want);
        assert!(
            bf16_rel < REL_NOISE_CEILING,
            "control setup: bf16 arm {bf16_rel:e}"
        );
        assert!(
            f16_rel > 1e-2,
            "the two decodes must not be interchangeable, yet reading the same \
             bytes as f16 scored rel {f16_rel:e} (bf16 arm: {bf16_rel:e})"
        );
    }

    /// A partial last threadgroup (`n` not a multiple of `ROWS_PER_TG`)
    /// and a `K` that does not divide the 4-way unrolled lane stride —
    /// the two geometry cases that have broken matvec parity in this
    /// crate before.
    #[test]
    fn bf16_gemv_handles_ragged_rows_and_a_ragged_tail() {
        let m = backend();
        let (n, k) = (13, 100);
        assert!(
            n % crate::shaders::bf16_gemv::ROWS_PER_TG as usize != 0 && k % (4 * 32) != 0,
            "test setup: this case exists to be ragged on both axes"
        );
        let (bytes, exact) = synth_weights(n, k);
        let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.031).sin()).collect();

        let got = m.bf16_gemv_force(&bytes, &x, n, k).expect("ragged shape");
        let want = scalar_gemv(&exact, &x, n, k);
        let rel = rel_err(&got, &want);
        assert!(rel < REL_NOISE_CEILING, "ragged rel {rel:e}");
    }

    /// Shape refusals, on both entry points — a short weight buffer
    /// would be an out-of-bounds device read, not a wrong answer.
    #[test]
    fn bf16_gemv_rejects_invalid_shapes() {
        let m = backend();
        let short = vec![0u8; 7]; // < n*k*2
        let x = vec![0.0f32; 4];
        assert!(m.bf16_gemv(&short, &x, 4, 4).is_none());
        assert!(m.bf16_gemv_force(&short, &x, 4, 4).is_none());

        let (bytes, _) = synth_weights(4, 4);
        let wrong_len = vec![0.0f32; 3];
        assert!(m.bf16_gemv_force(&bytes, &wrong_len, 4, 4).is_none());
    }

    /// The threshold-gated entry falls back below the FLOP threshold;
    /// `force` does not. Same contract as the f16 pair.
    #[test]
    fn bf16_gemv_falls_back_below_flop_threshold() {
        let m = backend();
        let (bytes, _) = synth_weights(2, 2);
        let x = vec![1.0f32; 2];
        assert!(m.bf16_gemv(&bytes, &x, 2, 2).is_none());
        assert!(m.bf16_gemv_force(&bytes, &x, 2, 2).is_some());
    }

    /// The batched submission changes only WHEN the work is submitted,
    /// never what it computes — so it must be bit-identical to the
    /// sequential calls, not merely close. A `_multi` that reordered a
    /// reduction or shared a scratch buffer would show up here.
    #[test]
    fn bf16_gemv_multi_is_bit_identical_to_the_sequential_calls() {
        let m = backend();
        let (n, k) = (48, 256);
        let (a, _) = synth_weights(n, k);
        let (b, _) = synth_weights(n * 2, k);
        let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.019).sin()).collect();

        let batched = m
            .bf16_gemv_multi(&[(&a, n, k), (&b, n * 2, k)], &x)
            .expect("batched submission");
        let sequential = vec![
            m.bf16_gemv_force(&a, &x, n, k).expect("first"),
            m.bf16_gemv_force(&b, &x, n * 2, k).expect("second"),
        ];

        assert_eq!(batched.len(), 2, "one result per matrix, in call order");
        assert_eq!(batched, sequential, "batching must not change any value");
    }

    /// Empty input is a no-op that succeeds; a bad shape anywhere in the
    /// batch refuses the whole batch rather than returning a short Vec
    /// the caller would index by position.
    #[test]
    fn bf16_gemv_multi_edge_cases() {
        let m = backend();
        let (n, k) = (16, 64);
        let (w, _) = synth_weights(n, k);
        let x = vec![0.5f32; k];

        assert_eq!(m.bf16_gemv_multi(&[], &x), Some(Vec::new()));
        let short = vec![0u8; 4];
        assert!(m
            .bf16_gemv_multi(&[(&w, n, k), (&short, n, k)], &x)
            .is_none());
    }

    /// Geometry is read from the bound pipeline, never from the shader
    /// module's constants — the mismatch class that cost this crate a
    /// 3.8x regression. Pins that the two agree at binding time.
    #[test]
    fn the_bound_pipeline_carries_the_shaders_own_geometry() {
        let m = backend();
        let kh = &m.bf16_gemv_pipeline;
        assert_eq!(kh.kernel_name, "bf16_gemv");
        assert_eq!(kh.rows_per_tg, crate::shaders::bf16_gemv::ROWS_PER_TG);
        assert_eq!(kh.threads_per_tg, crate::shaders::bf16_gemv::THREADS_PER_TG);
    }
}
