//! The traced, gates-only view of a decoder-layer chain.
//!
//! Split out of `mod.rs` to keep the production path — which reads
//! exactly one vector back — apart from the surface that reads every
//! boundary. Reading the traced planes in the real trajectory cost 64 ms
//! a token against 20 ms of actual GPU work, so the two must not be one
//! code path with a flag.

use super::super::grouped_experts::GroupedError;
use super::{read_u32, KimiLayerCall, KimiLayerWeights, LayerScratch};
use crate::MetalBackend;
use metal::Buffer;

/// Every boundary the layer crosses, read back.
///
/// Production reads only `output`. This exists so a disagreement names
/// its own stage, and so the router's decisions — which no production
/// path ever observes — stay checkable against the CPU's.
#[derive(Debug, Clone)]
pub struct KimiLayerPlanes {
    pub input_normed: Vec<f32>,
    pub attention: Vec<f32>,
    pub after_attention: Vec<f32>,
    pub post_attention_normed: Vec<f32>,
    pub router_logits: Vec<f32>,
    pub router_scores: Vec<f32>,
    pub router_selection_scores: Vec<f32>,
    pub selected_ids: Vec<u32>,
    /// What the MoE actually multiplied by: `top_k` routed weights, then
    /// the shared branch's unscaled `1.0`.
    pub combine_weights: Vec<f32>,
    /// The offset table the router wrote and the expert kernel read.
    pub expert_offsets: Vec<u32>,
    /// `[top_k + 1, hidden]` unweighted per-slot expert outputs.
    pub expert_outputs: Vec<f32>,
    pub output: Vec<f32>,
    pub gpu_ms: f64,
}

impl MetalBackend {
    /// The same layer, reading back every boundary. Gates only — the
    /// extra device→host reads are precisely what production must not do.
    pub fn kimi_decoder_layer_traced(
        &self,
        w: KimiLayerWeights<'_>,
        x: &[f32],
    ) -> Result<KimiLayerPlanes, GroupedError> {
        let mut p = self.kimi_decoder_layers_traced(&[KimiLayerCall { weights: w }], x)?;
        Ok(p.remove(0))
    }

    /// The multi-layer path with every layer's boundaries read back.
    ///
    /// Gates only. The readbacks are what production must not do — see
    /// [`Self::kimi_decoder_layers`], which reads exactly one vector.
    pub fn kimi_decoder_layers_traced(
        &self,
        layers: &[KimiLayerCall<'_>],
        x: &[f32],
    ) -> Result<Vec<KimiLayerPlanes>, GroupedError> {
        let hidden = layers
            .first()
            .ok_or(GroupedError::NoExpertsSelected)?
            .weights
            .attention
            .hidden();
        let (scratch, kda_scratch, _, gpu_ms) = self.encode_layer_chain(layers, None, x)?;
        let mut planes = Vec::with_capacity(layers.len());
        for (i, (call, s)) in layers.iter().zip(&scratch).enumerate() {
            let experts = call.weights.ffn.experts();
            let mut p = self.read_layer_planes(
                s,
                hidden,
                experts,
                call.weights.ffn.slots(),
                call.weights.ffn.top_k(),
                gpu_ms,
            );
            p.attention = crate::buffers::read_buffer_f32(kda_scratch[i].out(), hidden);
            planes.push(p);
        }
        self.recycle_chain(scratch, kda_scratch);
        Ok(planes)
    }

    #[allow(clippy::too_many_arguments)]
    fn read_layer_planes(
        &self,
        s: &LayerScratch,
        hidden: usize,
        experts: usize,
        slots: usize,
        top_k: usize,
        gpu_ms: f64,
    ) -> KimiLayerPlanes {
        let f = |b: &Buffer, n: usize| crate::buffers::read_buffer_f32(b, n);
        KimiLayerPlanes {
            input_normed: f(&s.input_normed, hidden),
            attention: Vec::new(),
            after_attention: f(&s.after_attention, hidden),
            post_attention_normed: f(&s.post_normed, hidden),
            router_logits: f(&s.logits, experts),
            router_scores: f(&s.scores, experts),
            router_selection_scores: f(&s.sel_scores, experts),
            selected_ids: read_u32(&s.chosen, top_k),
            combine_weights: f(&s.weights, slots),
            expert_offsets: read_u32(&s.offsets, slots),
            expert_outputs: f(&s.expert_out, slots * hidden),
            output: f(&s.out, hidden),
            gpu_ms,
        }
    }
}
