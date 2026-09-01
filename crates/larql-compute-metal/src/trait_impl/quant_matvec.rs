//! `QuantMatVec` impl for `MetalBackend`.
//!
//! Each per-format method delegates to the corresponding kernel
//! dispatcher in `metal::ops` or to a per-format dispatcher built
//! around the appropriate shader pipeline.

use crate::MetalBackend;
use larql_compute::backend::QuantMatVec;

impl QuantMatVec for MetalBackend {
    fn q4_matvec(
        &self,
        q4_data: &[u8],
        q8_x: &[i8],
        q8_scales: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(self.q4_matvec_direct(q4_data, q8_x, q8_scales, num_rows, hidden))
    }

    /// Q4 matvec → GPU argmax_partial, returning `(token_id, score)` for
    /// the top-1 element. Used by the lm_head greedy-decode path on models
    /// that have a Q4 lm_head (`lm_head_q4.bin` or synthesized from f16
    /// embeddings). Saves the 1MB readback + 262K-element CPU sort.
    fn q4_matvec_topk1(
        &self,
        q4_data: &[u8],
        q8_x: &[i8],
        q8_scales: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<(u32, f32)> {
        if num_rows == 0 || q8_x.len() != hidden {
            return None;
        }
        let buf_q4 = self.bufs.get_bytes(q4_data);
        let buf_q8 = self.bufs.transient_from_i8(q8_x);
        let buf_scales = self.bufs.transient_from_f32(q8_scales);
        let scores = self.bufs.output((num_rows * 4) as u64);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        crate::ops::q4_matvec::encode(
            enc,
            &self.q4.matvec,
            &buf_q4,
            &buf_q8,
            &buf_scales,
            &scores,
            num_rows as u32,
            hidden as u32,
            num_rows,
        );
        let (partial_vals, partial_idxs, n_partials) =
            self.encode_argmax_partial(enc, &scores, num_rows);
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_or_abort(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/quant_matvec.rs:59",
        );
        Self::reduce_argmax_partial(&partial_vals, &partial_idxs, n_partials)
    }

    /// Q4 matvec + GPU partial top-K. Returns up to `top_k` entries
    /// (`top_k ≤ K_TOPK = 8`) sorted descending. Caller falls back to
    /// `q4_matvec` + CPU sort when this returns `None`.
    fn q4_matvec_topk(
        &self,
        q4_data: &[u8],
        q8_x: &[i8],
        q8_scales: &[f32],
        num_rows: usize,
        hidden: usize,
        top_k: usize,
    ) -> Option<Vec<(u32, f32)>> {
        if top_k == 0 || top_k > crate::shaders::f32_gemv::K_TOPK {
            return None;
        }
        if num_rows == 0 || q8_x.len() != hidden {
            return None;
        }
        let buf_q4 = self.bufs.get_bytes(q4_data);
        let buf_q8 = self.bufs.transient_from_i8(q8_x);
        let buf_scales = self.bufs.transient_from_f32(q8_scales);
        let scores = self.bufs.output((num_rows * 4) as u64);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        crate::ops::q4_matvec::encode(
            enc,
            &self.q4.matvec,
            &buf_q4,
            &buf_q8,
            &buf_scales,
            &scores,
            num_rows as u32,
            hidden as u32,
            num_rows,
        );
        let (partial_vals, partial_idxs, num_tgs) =
            self.encode_topk_partial(enc, &scores, num_rows);
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_or_abort(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/quant_matvec.rs:103",
        );
        Some(MetalBackend::reduce_topk_partial(
            &partial_vals,
            &partial_idxs,
            num_tgs,
            top_k,
        ))
    }

    fn q4_vecmat(
        &self,
        activation: &[f32],
        q4_data: &[u8],
        intermediate: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(self.q4_vecmat_direct(activation, q4_data, intermediate, hidden))
    }

    fn q4_matvec_pair_batch(
        &self,
        gate_q4: &[u8],
        up_q4: &[u8],
        x_matrix: &[f32],
        seq_len: usize,
        num_rows: usize,
        hidden: usize,
    ) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        Some(self.q4_matvec_pair_batch_direct(gate_q4, up_q4, x_matrix, seq_len, num_rows, hidden))
    }

    fn q4k_matvec_stride32(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        MetalBackend::q4k_matvec_stride32(self, q4k_data, x, num_rows, hidden)
    }

    fn q4k_matvec(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        // Pull dispatch geometry from the actually-bound pipeline rather
        // than from `shaders::q4k_matvec`'s hard-coded constants. The
        // `q4k_matvec_pipeline` field is bound at startup to either
        // `q4k_matvec` (4 rows × 128 threads) or `q4k_matvec_8sg`
        // (8 rows × 256 threads) per `LARQL_Q4K_MATVEC_8SG`. Using the
        // 4sg constants here under-dispatches by 50% when 8sg is bound,
        // leaving simdgroups 4..7 unscheduled and half the rows in each
        // TG unwritten — same family of bug as the historical 077884b
        // "81–84 tok/s on broken Q4_K dispatch" (Q4_K bytes routed
        // through a kernel with mismatched threadgroup geometry).
        let buf_w = self.bufs.get_bytes(q4k_data);
        let buf_x = self.bufs.transient_from_f32(x);
        let buf_out = self.bufs.output((num_rows * 4) as u64);
        let n = num_rows as u32;
        let k = hidden as u32;
        let rows_per_tg = self.quant.q4k_matvec_pipeline.rows_per_tg;
        let threads_per_tg = self.quant.q4k_matvec_pipeline.threads_per_tg;
        let num_tgs = (num_rows as u64).div_ceil(rows_per_tg);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.quant.q4k_matvec_pipeline.state);
        enc.set_buffer(0, Some(&buf_w), 0);
        enc.set_buffer(1, Some(&buf_x), 0);
        enc.set_buffer(2, Some(&buf_out), 0);
        enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs, 1, 1),
            metal::MTLSize::new(threads_per_tg, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_or_abort(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/quant_matvec.rs:184",
        );

        Some(crate::buffers::read_buffer_f32(&buf_out, num_rows))
    }

    /// Q4_K matvec + GPU partial top-K in one command buffer. The scores
    /// vector never leaves the GPU: only `num_tgs × K_TOPK` (val, idx)
    /// pairs come back for a small CPU merge — on a 201K-row lm_head that
    /// replaces an ~800 KB readback and a full-vocab CPU heap scan.
    fn q4k_matvec_topk(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
        top_k: usize,
    ) -> Option<Vec<(u32, f32)>> {
        if top_k == 0 || top_k > crate::shaders::f32_gemv::K_TOPK {
            return None;
        }
        if num_rows == 0 || x.len() != hidden {
            return None;
        }
        let buf_w = self.bufs.get_bytes(q4k_data);
        let buf_x = self.bufs.transient_from_f32(x);
        let buf_out = self.bufs.output((num_rows * 4) as u64);
        let n = num_rows as u32;
        let k = hidden as u32;
        // Geometry from the bound pipeline, same as `q4k_matvec` — see the
        // dispatch-geometry note there.
        let rows_per_tg = self.quant.q4k_matvec_pipeline.rows_per_tg;
        let threads_per_tg = self.quant.q4k_matvec_pipeline.threads_per_tg;
        let num_tgs = (num_rows as u64).div_ceil(rows_per_tg);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.quant.q4k_matvec_pipeline.state);
        enc.set_buffer(0, Some(&buf_w), 0);
        enc.set_buffer(1, Some(&buf_x), 0);
        enc.set_buffer(2, Some(&buf_out), 0);
        enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs, 1, 1),
            metal::MTLSize::new(threads_per_tg, 1, 1),
        );
        let (partial_vals, partial_idxs, topk_tgs) =
            self.encode_topk_partial(enc, &buf_out, num_rows);
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_or_abort(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/quant_matvec.rs:234",
        );
        Some(MetalBackend::reduce_topk_partial(
            &partial_vals,
            &partial_idxs,
            topk_tgs,
            top_k,
        ))
    }

    /// Q4_K matrix-matrix multiply: `C[m, n] = sum_k W[n, k] * X[m, k]`.
    ///
    /// `W` is `[num_rows, hidden]` Q4_K row-major. `X` is `[seq_len,
    /// hidden]` f32 row-major. Output is `[seq_len, num_rows]` f32
    /// row-major (one row per input position, matching the convention
    /// downstream attention/FFN stages expect).
    ///
    /// Parity contract: the result of this call MUST equal stacking
    /// `q4k_matvec(W, X[m..m+1])` for `m=0..seq_len`. The matmul kernel
    /// just amortises the Q4_K dequant across `seq_len` positions —
    /// the per-element math is identical. Verified by
    /// `q4k_matmul_matches_stacked_matvec`.
    fn q4k_matmul(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
        seq_len: usize,
    ) -> Option<Vec<f32>> {
        use crate::shaders::q4k_matmul as q4k_mm;
        if seq_len == 0 || num_rows == 0 || hidden == 0 {
            return Some(Vec::new());
        }
        let buf_w = self.bufs.get_bytes(q4k_data);
        let buf_x = self.bufs.transient_from_f32(x);
        let buf_out = self.bufs.output((seq_len * num_rows * 4) as u64);
        let n = num_rows as u32;
        let k = hidden as u32;
        let m = seq_len as u32;
        let row_tgs = (num_rows as u64).div_ceil(q4k_mm::ROWS_PER_TG);
        let col_tgs = (seq_len as u64).div_ceil(q4k_mm::COLS_PER_TG);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.quant.q4k_matmul_pipeline.state);
        enc.set_buffer(0, Some(&buf_w), 0);
        enc.set_buffer(1, Some(&buf_x), 0);
        enc.set_buffer(2, Some(&buf_out), 0);
        enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &m as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(col_tgs, row_tgs, 1),
            metal::MTLSize::new(q4k_mm::THREADS_PER_TG, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_or_abort(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/quant_matvec.rs:291",
        );

        Some(crate::buffers::read_buffer_f32(
            &buf_out,
            seq_len * num_rows,
        ))
    }

    fn q6k_matvec(
        &self,
        q6k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        // Geometry from the bound `KernelHandle`, never from a shader
        // module's constants: `q6k_matvec_pipeline` resolves to the 8sg
        // variant (8 rows/TG, 256 threads) under `LARQL_Q6K_8SG=1`, and
        // dispatching it with the 4sg module constants launches half
        // the threadgroups — rows `8t+4..8t+7` are never written and
        // the output buffer serves stale pool data for them. The q4k
        // twin above documents the same hazard; this site was left
        // behind when that one was fixed. Capability audit F5.
        let kh = &self.quant.q6k_matvec_pipeline;
        let buf_w = self.bufs.get_bytes(q6k_data);
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
        crate::cb_status::wait_or_abort(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/quant_matvec.rs:336",
        );

        Some(crate::buffers::read_buffer_f32(&buf_out, num_rows))
    }

    fn supports_quant(&self, format: larql_compute::QuantFormat) -> bool {
        use larql_compute::QuantFormat;
        matches!(
            format,
            QuantFormat::Q4_0
                | QuantFormat::Q4_K
                | QuantFormat::Q4_KF
                | QuantFormat::Q6_K
                | QuantFormat::Q8_0
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::MetalBackend;
    use larql_compute::backend::QuantMatVec;

    fn backend() -> MetalBackend {
        MetalBackend::new().expect("Metal device available on test host")
    }

    /// `q4_matvec_topk1` returns `None` when `num_rows == 0`. Covers the
    /// early-exit branch at the top of the function.
    #[test]
    fn q4_matvec_topk1_returns_none_for_zero_rows() {
        let m = backend();
        let out = m.q4_matvec_topk1(&[], &[1i8; 4], &[1.0f32], 0, 4);
        assert!(out.is_none());
    }

    /// `q4_matvec_topk1` returns `None` when `q8_x.len() != hidden`.
    #[test]
    fn q4_matvec_topk1_returns_none_for_mismatched_q8x_length() {
        let m = backend();
        // hidden=64 but q8_x has 16 elements → mismatch.
        let out = m.q4_matvec_topk1(&[], &[1i8; 16], &[1.0f32], 4, 64);
        assert!(out.is_none());
    }

    /// `q4_matvec_topk` returns `None` for `top_k == 0` or `top_k >
    /// K_TOPK`. Pin both edges of the early-exit.
    #[test]
    fn q4_matvec_topk_returns_none_for_invalid_top_k() {
        let m = backend();
        // top_k = 0
        assert!(m
            .q4_matvec_topk(&[], &[1i8; 4], &[1.0f32], 4, 4, 0)
            .is_none());
        // top_k = K_TOPK + 1 (just over the cap).
        let too_big = crate::shaders::f32_gemv::K_TOPK + 1;
        assert!(m
            .q4_matvec_topk(&[], &[1i8; 4], &[1.0f32], 4, 4, too_big)
            .is_none());
    }

    /// `q4_matvec_topk` returns `None` when `num_rows == 0` or
    /// `q8_x.len() != hidden`.
    #[test]
    fn q4_matvec_topk_returns_none_for_zero_rows_or_mismatched_q8x() {
        let m = backend();
        assert!(m
            .q4_matvec_topk(&[], &[1i8; 4], &[1.0f32], 0, 4, 1)
            .is_none());
        assert!(m
            .q4_matvec_topk(&[], &[1i8; 16], &[1.0f32], 4, 64, 1)
            .is_none());
    }

    /// `q4_vecmat` trait wrapper delegates to `q4_vecmat_direct`.
    /// Call with a minimal shape to cover the wrapper body.
    #[test]
    fn q4_vecmat_wrapper_returns_some_vector() {
        let m = backend();
        let hidden = 32usize; // multiple of 32 → one Q4_0 block per row
        let inter = 32usize;
        // 32 floats per Q4_0 block. block = 2-byte scale + 16-byte quants = 18 bytes.
        let block_bytes = 18usize;
        // q4_vecmat reads `inter` rows of `hidden/32` blocks each.
        let q4_data = vec![0u8; inter * (hidden / 32) * block_bytes];
        let activation = vec![0.1f32; inter];
        let out = m
            .q4_vecmat(&activation, &q4_data, inter, hidden)
            .expect("q4_vecmat wrapper always returns Some");
        assert_eq!(out.len(), hidden);
    }

    /// `q4_matvec_pair_batch` trait wrapper delegates to
    /// `q4_matvec_pair_batch_direct`. Empty seq_len is the cheapest
    /// way to invoke it without staging real Q4_0 model weights.
    #[test]
    fn q4_matvec_pair_batch_wrapper_returns_some() {
        let m = backend();
        // seq_len = 0 → the underlying direct path returns
        // (vec![], vec![]) without doing any GPU work.
        let out = m.q4_matvec_pair_batch(&[], &[], &[], 0, 0, 0);
        assert!(out.is_some());
    }

    /// `q4k_matvec_stride32` trait wrapper. Use a hidden multiple of
    /// 256 (Q4_K super-block) so the dispatch shape is valid; q4k_data
    /// shape must equal num_rows × Q4_K_BLOCK_BYTES bytes per
    /// (hidden/256) blocks.
    #[test]
    fn q4k_matvec_stride32_wrapper_returns_some_vector() {
        let m = backend();
        let hidden = 256usize;
        let num_rows = 4usize;
        let block_bytes = larql_models::quant::ggml::Q4_K_BLOCK_BYTES;
        let q4k_data = vec![0u8; num_rows * (hidden / 256) * block_bytes];
        let x = vec![0.1f32; hidden];
        let out = m
            .q4k_matvec_stride32(&q4k_data, &x, num_rows, hidden)
            .expect("q4k_matvec_stride32 wrapper returns Some on valid shapes");
        assert_eq!(out.len(), num_rows);
    }

    /// `q4k_matmul` returns an empty vec when any of `seq_len`,
    /// `num_rows`, `hidden` is zero (line 211 early-out).
    #[test]
    fn q4k_matmul_zero_dims_returns_empty_vec() {
        let m = backend();
        let out = m
            .q4k_matmul(&[], &[1.0f32; 4], 0, 4, 1)
            .expect("zero-dim path returns Some(empty)");
        assert!(out.is_empty());

        let out = m
            .q4k_matmul(&[], &[1.0f32; 4], 1, 0, 1)
            .expect("zero-dim path returns Some(empty)");
        assert!(out.is_empty());

        let out = m
            .q4k_matmul(&[], &[], 1, 4, 0)
            .expect("zero-dim path returns Some(empty)");
        assert!(out.is_empty());
    }
}

#[cfg(test)]
mod q4k_topk_tests {
    use crate::MetalBackend;
    use larql_compute::backend::QuantMatVec;
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;

    fn backend() -> MetalBackend {
        MetalBackend::new().expect("Metal device available on test host")
    }

    /// Rows deliberately not a multiple of the tile so the remainder path
    /// runs; K a Q4_K super-block multiple.
    const N: usize = 517;
    const K: usize = 512;

    fn fixture() -> (Vec<u8>, Vec<f32>) {
        // Knuth-hash mix, NOT a short-period ramp: a pattern with period p
        // makes rows p apart byte-identical, and tied duplicate rows turn
        // rank order into a coin flip the parity assert would then fail on.
        let w: Vec<f32> = (0..N * K)
            .map(|i| {
                let h = (i as u32).wrapping_mul(2654435761) >> 16;
                ((h % 4001) as f32 - 2000.0) * 1e-3
            })
            .collect();
        let x: Vec<f32> = (0..K).map(|i| ((i % 89) as f32 - 44.0) * 0.02).collect();
        (quantize_q4_k(&w), x)
    }

    /// The one contract that lets the lm_head path take the small-K arm:
    /// the fused matvec+top-K submission returns exactly what the full
    /// readback + CPU reduce returns.
    #[test]
    fn q4k_matvec_topk_matches_full_readback_reduce() {
        let m = backend();
        let (q, x) = fixture();
        let scores = m.q4k_matvec(&q, &x, N, K).expect("full matvec");
        let mut want: Vec<(u32, f32)> = scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i as u32, s))
            .collect();
        want.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for top_k in [1usize, 5, 8] {
            let hits = m
                .q4k_matvec_topk(&q, &x, N, K, top_k)
                .expect("topk path fires for top_k <= K_TOPK");
            assert_eq!(hits.len(), top_k.min(N));
            for (rank, (id, score)) in hits.iter().enumerate() {
                // Exact ties order arbitrarily between the two reduces, so
                // the id contract is "an id whose score is this rank's
                // score", with score parity exact-tolerance either way.
                assert!(
                    (score - want[rank].1).abs() <= 1e-5 * want[rank].1.abs().max(1.0),
                    "rank {rank} score {score} vs {} (top_k={top_k})",
                    want[rank].1
                );
                assert!(
                    *id == want[rank].0 || scores[*id as usize] == want[rank].1,
                    "rank {rank}: id {id} is not the rank-{rank} id {} and its \
                     score {} is not tied with {} (top_k={top_k})",
                    want[rank].0,
                    scores[*id as usize],
                    want[rank].1
                );
            }
        }
    }

    /// The refusals that keep the caller's fallback honest.
    #[test]
    fn q4k_matvec_topk_refuses_out_of_capacity_and_bad_shapes() {
        let m = backend();
        let (q, x) = fixture();
        assert!(m.q4k_matvec_topk(&q, &x, N, K, 0).is_none());
        assert!(m
            .q4k_matvec_topk(&q, &x, N, K, crate::shaders::f32_gemv::K_TOPK + 1)
            .is_none());
        assert!(m.q4k_matvec_topk(&q, &x[..K - 1], N, K, 1).is_none());
        assert!(m.q4k_matvec_topk(&q, &x, 0, K, 1).is_none());
    }
}
