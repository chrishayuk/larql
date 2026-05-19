use super::*;

impl MetalBackend {
    /// Multi-layer Q4 FFN in ONE command buffer.
    /// gate → up → GEGLU → down → Q8 quantize → next layer.
    /// All on GPU, no CPU return between layers.
    pub fn multi_layer_q4_ffn(
        &self,
        layers_q4: &[(&[u8], &[u8], &[u8])], // [(gate, up, down_t)]
        x: &[f32],
        inter: usize,
        hidden: usize,
    ) -> Vec<f32> {
        ops::q4_batched::multi_layer_ffn(
            &self.queue,
            &self.bufs,
            &self.q4,
            &self.ffn.geglu_pipeline,
            &self.quant.q8_quant_pipeline,
            layers_q4,
            x,
            inter,
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

    /// `multi_layer_q4_ffn` dispatches gate → up → GEGLU → down → Q8
    /// per layer in a single command buffer.
    #[test]
    fn multi_layer_q4_ffn_dispatches_two_layers() {
        let m = backend();
        let block_bytes = 18usize;
        let hidden = 32usize;
        let inter = 64usize;
        let blocks_per_row = hidden / 32;
        let gate = vec![0u8; inter * blocks_per_row * block_bytes];
        let up = vec![0u8; inter * blocks_per_row * block_bytes];
        let down = vec![0u8; hidden * (inter / 32) * block_bytes];
        let layers = vec![
            (gate.as_slice(), up.as_slice(), down.as_slice()),
            (gate.as_slice(), up.as_slice(), down.as_slice()),
        ];
        let x = vec![0.0f32; hidden];
        let out = m.multi_layer_q4_ffn(&layers, &x, inter, hidden);
        assert_eq!(out.len(), hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
