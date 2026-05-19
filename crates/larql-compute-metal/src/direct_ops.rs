use super::*;

impl MetalBackend {
    // ── Direct Q4 ops (for benchmarking outside the trait) ──

    pub fn q4_matvec_direct(
        &self,
        q4_data: &[u8],
        q8_x: &[i8],
        q8_scales: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Vec<f32> {
        ops::q4_matvec::dispatch(
            &self.queue,
            &self.bufs,
            &self.q4.matvec,
            q4_data,
            q8_x,
            q8_scales,
            num_rows,
            hidden,
        )
    }

    pub fn q4_vecmat_direct(
        &self,
        activation: &[f32],
        q4_data: &[u8],
        intermediate: usize,
        hidden: usize,
    ) -> Vec<f32> {
        ops::q4_vecmat::dispatch(
            &self.queue,
            &self.bufs,
            &self.q4.vecmat,
            activation,
            q4_data,
            intermediate,
            hidden,
        )
    }

    /// Q4 × f32 matvec (for transposed down projection).
    pub fn q4_f32_matvec_direct(
        &self,
        q4_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Vec<f32> {
        ops::q4_f32_matvec::dispatch(
            &self.queue,
            &self.bufs,
            &self.q4.f32_matvec,
            q4_data,
            x,
            num_rows,
            hidden,
        )
    }

    /// Full layer pipeline: attention + FFN in one Metal command buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn full_layer_direct(
        &self,
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        gate_q4: &[u8],
        up_q4: &[u8],
        down_t_q4: &[u8],
        x: &[f32],
        seq_len: usize,
        hidden: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        inter: usize,
        attn_scale: f32,
    ) -> Vec<f32> {
        ops::full_layer::dispatch(
            &self.queue,
            &self.bufs,
            &self.f32_ops.transb_pipeline,
            &self.attention.causal_attn_pipeline,
            &self.q4,
            w_q,
            w_k,
            w_v,
            w_o,
            gate_q4,
            up_q4,
            down_t_q4,
            x,
            seq_len,
            hidden,
            num_q_heads,
            num_kv_heads,
            head_dim,
            inter,
            attn_scale,
        )
    }

    pub fn q4_matvec_pair_batch_direct(
        &self,
        gate_q4: &[u8],
        up_q4: &[u8],
        x_matrix: &[f32],
        seq_len: usize,
        num_rows: usize,
        hidden: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        ops::q4_batched::pair_batch(
            &self.queue,
            &self.bufs,
            &self.q4,
            gate_q4,
            up_q4,
            x_matrix,
            seq_len,
            num_rows,
            hidden,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend() -> MetalBackend {
        MetalBackend::new().expect("Metal device available on test host")
    }

    /// `full_layer_direct` wraps `ops::full_layer::dispatch` with the
    /// backend's pipelines pre-bound.  Drive it with synthetic
    /// (all-zero) weights at a small shape — the test pins that the
    /// signature wiring is intact and the dispatch completes.
    /// Numerical correctness is checked at higher levels.
    #[test]
    fn full_layer_direct_dispatches_with_synthetic_weights() {
        let m = backend();
        let hidden = 32usize;
        let head_dim = 16usize;
        let num_q_heads = 2usize;
        let num_kv_heads = 2usize;
        let inter = 64usize;
        let seq_len = 1usize;

        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let w_q = vec![0.0f32; q_dim * hidden];
        let w_k = vec![0.0f32; kv_dim * hidden];
        let w_v = vec![0.0f32; kv_dim * hidden];
        let w_o = vec![0.0f32; hidden * q_dim];

        // Q4_0 super-block stride (18 bytes per 32 elements).  Synth
        // zeros: scale = f16(0) and nibbles = 0 means the layer
        // contributes nothing — but that's fine for a wiring test.
        let q4_blocks_per_row = hidden / 32;
        let gate_q4 = vec![0u8; inter * q4_blocks_per_row * 18];
        let up_q4 = vec![0u8; inter * q4_blocks_per_row * 18];
        let down_t_q4 = vec![0u8; inter * q4_blocks_per_row * 18];

        let x = vec![0.0f32; seq_len * hidden];

        let out = m.full_layer_direct(
            &w_q,
            &w_k,
            &w_v,
            &w_o,
            &gate_q4,
            &up_q4,
            &down_t_q4,
            &x,
            seq_len,
            hidden,
            num_q_heads,
            num_kv_heads,
            head_dim,
            inter,
            1.0 / (head_dim as f32).sqrt(),
        );
        assert_eq!(out.len(), seq_len * hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
