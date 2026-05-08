use ndarray::Array2;

use super::{LayerGraph, LayerOutput};
use crate::ffn::FfnBackend;
use crate::model::ModelWeights;
use larql_compute::prelude::*;

// ── Walk: dense attention + vindex walk FFN ──

/// Walk layer graph: dense attention + vindex walk FFN.
/// This is the working walk path, wrapped in the LayerGraph trait.
pub struct WalkLayerGraph<'a> {
    pub ffn: &'a dyn FfnBackend,
    pub backend: Option<&'a dyn ComputeBackend>,
}

impl<'a> LayerGraph for WalkLayerGraph<'a> {
    fn forward_layer(
        &self,
        weights: &ModelWeights,
        h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput> {
        let (h_post_attn, _attn_proj, _) =
            crate::attention::run_attention_block_gpu(weights, h, layer, false, self.backend)?;
        let (h_out, _) = crate::forward::run_ffn(weights, &h_post_attn, layer, self.ffn, false);
        Some(LayerOutput {
            residual: h_out,
            activation: None,
            attention: None,
        })
    }

    fn name(&self) -> &str {
        "walk"
    }
}

// ── Pipelined: CPU attention + batched GPU Q4 FFN ──

/// Pipelined layer graph: runs attention on CPU, batches all FFN layers
/// in one Metal Q4 command buffer via `multi_layer_q4_ffn`.
///
/// The layer loop splits each layer into:
/// 1. Attention (CPU BLAS) → post-attention residual
/// 2. FFN norm → normalized input for FFN
/// 3. Q4 FFN (batched on GPU after all layers)
///
/// For layers where attention and FFN are interleaved (each FFN feeds next attention),
/// this runs a hybrid: per-layer attention + per-layer Q4 FFN via the compute backend,
/// but with the Q4 overhead amortized by reusing the GPU command queue.
pub struct PipelinedLayerGraph<'a> {
    pub index: &'a dyn larql_vindex::GateIndex,
    pub backend: &'a dyn ComputeBackend,
    pub layer_range: std::ops::Range<usize>,
}

impl<'a> LayerGraph for PipelinedLayerGraph<'a> {
    fn forward_layer(
        &self,
        weights: &ModelWeights,
        h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput> {
        if !self.layer_range.contains(&layer) {
            return None;
        }

        // Attention: CPU BLAS (fast, no GPU overhead)
        let (h_post_attn, _attn_proj, _) =
            crate::attention::run_attention_block_gpu(weights, h, layer, false, None)?;

        // FFN: use WalkFfn which handles Q4 dispatch internally.
        // WalkFfn checks for Q4 interleaved data and routes to Metal Q4
        // when backend.has_q4(), falling back to f32 BLAS otherwise.
        // This ensures the norm/residual logic matches exactly.
        let walk_ffn =
            crate::vindex::WalkFfn::new_unlimited_with_backend(weights, self.index, self.backend);
        let (h_out, _) = crate::forward::run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
        Some(LayerOutput {
            residual: h_out,
            activation: None,
            attention: None,
        })
    }

    fn name(&self) -> &str {
        "pipelined"
    }
}

/// Split-residency layer graph for Linux/NVIDIA-scale models.
///
/// Attention and dense transformer state run through `attention_backend` (CUDA
/// on NVIDIA when available). The MoE/FFN side uses the vindex walk path and may
/// intentionally receive no compute backend, keeping expert/vindex tensors
/// mmap-backed and paged from disk instead of staging all experts in VRAM.
pub struct GpuAttentionMmapExpertsGraph<'a> {
    pub index: &'a dyn larql_vindex::GateIndex,
    pub attention_backend: &'a dyn ComputeBackend,
    pub expert_backend: Option<&'a dyn ComputeBackend>,
    pub layer_range: std::ops::Range<usize>,
}

impl<'a> LayerGraph for GpuAttentionMmapExpertsGraph<'a> {
    fn forward_layer(
        &self,
        weights: &ModelWeights,
        h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput> {
        if !self.layer_range.contains(&layer) {
            return None;
        }

        let (h_post_attn, _attn_proj, _) = crate::attention::run_attention_block_gpu(
            weights,
            h,
            layer,
            false,
            Some(self.attention_backend),
        )?;

        let walk_ffn = match self.expert_backend {
            Some(backend) => {
                crate::vindex::WalkFfn::new_unlimited_with_backend(weights, self.index, backend)
            }
            None => crate::vindex::WalkFfn::new_unlimited(weights, self.index),
        };
        let (h_out, _) = crate::forward::run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
        Some(LayerOutput {
            residual: h_out,
            activation: None,
            attention: None,
        })
    }

    fn name(&self) -> &str {
        "gpu-attention-mmap-experts"
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
/// Split-residency graph with persistent CUDA-resident attention projections.
///
/// Unlike `GpuAttentionMmapExpertsGraph`, this path reuses per-layer Q/K/V/O
/// device buffers through `CudaAttentionResidency`. Experts still go through the
/// vindex walk FFN path; `expert_backend: None` keeps them mmap/CPU-backed.
pub struct CudaResidentAttentionMmapExpertsGraph<'a> {
    pub index: &'a dyn larql_vindex::GateIndex,
    pub cuda: &'a larql_compute::CudaBackend,
    pub resident_attention: &'a [crate::attention::gpu::CudaAttentionResidency],
    pub expert_backend: Option<&'a dyn ComputeBackend>,
    pub layer_range: std::ops::Range<usize>,
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
impl<'a> CudaResidentAttentionMmapExpertsGraph<'a> {
    fn resident_for_layer(
        &self,
        layer: usize,
    ) -> Option<&crate::attention::gpu::CudaAttentionResidency> {
        self.resident_attention
            .iter()
            .find(|resident| resident.layer() == layer)
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
impl<'a> LayerGraph for CudaResidentAttentionMmapExpertsGraph<'a> {
    fn forward_layer(
        &self,
        weights: &ModelWeights,
        h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput> {
        if !self.layer_range.contains(&layer) {
            return None;
        }
        let resident = self.resident_for_layer(layer)?;
        let (h_post_attn, _attn_proj, _) =
            crate::attention::gpu::run_attention_block_cuda_resident(
                weights, h, layer, false, self.cuda, resident,
            )?;
        let walk_ffn = match self.expert_backend {
            Some(backend) => {
                crate::vindex::WalkFfn::new_unlimited_with_backend(weights, self.index, backend)
            }
            None => crate::vindex::WalkFfn::new_unlimited(weights, self.index),
        };
        let (h_out, _) = crate::forward::run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
        Some(LayerOutput {
            residual: h_out,
            activation: None,
            attention: None,
        })
    }

    fn name(&self) -> &str {
        "cuda-resident-attention-mmap-experts"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::test_utils::make_test_weights;
    use crate::ffn::WeightFfn;
    use larql_models::ModelWeights;
    use ndarray::Array2;
    use std::sync::OnceLock;

    fn weights() -> &'static ModelWeights {
        static W: OnceLock<ModelWeights> = OnceLock::new();
        W.get_or_init(make_test_weights)
    }

    fn input(seq: usize, hidden: usize) -> Array2<f32> {
        let data: Vec<f32> = (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.01).collect();
        Array2::from_shape_vec((seq, hidden), data).unwrap()
    }

    // ── WalkLayerGraph ────────────────────────────────────────────────────────

    #[test]
    fn walk_name() {
        let w = weights();
        let ffn = WeightFfn { weights: w };
        let g = WalkLayerGraph {
            ffn: &ffn,
            backend: None,
        };
        assert_eq!(g.name(), "walk");
    }

    #[test]
    fn walk_forward_shape_single_token() {
        let w = weights();
        let ffn = WeightFfn { weights: w };
        let g = WalkLayerGraph {
            ffn: &ffn,
            backend: None,
        };
        let h = input(1, w.hidden_size);
        let out = g.forward_layer(w, &h, 0).expect("layer 0");
        assert_eq!(out.residual.shape(), &[1, w.hidden_size]);
        assert!(out.residual.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn walk_forward_all_layers() {
        let w = weights();
        let ffn = WeightFfn { weights: w };
        let g = WalkLayerGraph {
            ffn: &ffn,
            backend: None,
        };
        let h = input(1, w.hidden_size);
        for layer in 0..w.num_layers {
            let out = g.forward_layer(w, &h, layer).expect("layer {layer}");
            assert_eq!(out.residual.shape(), &[1, w.hidden_size], "layer {layer}");
        }
    }

    #[test]
    fn walk_never_captures_activation_or_attention() {
        let w = weights();
        let ffn = WeightFfn { weights: w };
        let g = WalkLayerGraph {
            ffn: &ffn,
            backend: None,
        };
        let out = g.forward_layer(w, &input(2, w.hidden_size), 0).unwrap();
        assert!(out.activation.is_none());
        assert!(out.attention.is_none());
    }

    // ── PipelinedLayerGraph ───────────────────────────────────────────────────

    #[test]
    fn pipelined_name() {
        let w = weights();
        let idx = crate::engines::test_utils::make_test_vindex(w);
        let g = PipelinedLayerGraph {
            index: &idx,
            backend: &larql_compute::CpuBackend,
            layer_range: 0..w.num_layers,
        };
        assert_eq!(g.name(), "pipelined");
    }

    #[test]
    fn pipelined_out_of_range_returns_none() {
        let w = weights();
        let idx = crate::engines::test_utils::make_test_vindex(w);
        let g = PipelinedLayerGraph {
            index: &idx,
            backend: &larql_compute::CpuBackend,
            layer_range: 5..10, // range that excludes layer 0
        };
        let h = input(1, w.hidden_size);
        // Layer 0 is outside range 5..10 → None
        let out = g.forward_layer(w, &h, 0);
        assert!(out.is_none(), "layer outside range should return None");
    }

    #[test]
    fn pipelined_in_range_produces_output() {
        let w = weights();
        let idx = crate::engines::test_utils::make_test_vindex(w);
        let g = PipelinedLayerGraph {
            index: &idx,
            backend: &larql_compute::CpuBackend,
            layer_range: 0..w.num_layers,
        };
        let h = input(1, w.hidden_size);
        let out = g.forward_layer(w, &h, 0);
        assert!(out.is_some(), "layer in range should produce output");
        assert_eq!(out.unwrap().residual.shape(), &[1, w.hidden_size]);
    }

    #[test]
    fn gpu_attention_mmap_experts_graph_names_split_residency() {
        let w = weights();
        let idx = crate::engines::test_utils::make_test_vindex(w);
        let g = GpuAttentionMmapExpertsGraph {
            index: &idx,
            attention_backend: &larql_compute::CpuBackend,
            expert_backend: None,
            layer_range: 0..w.num_layers,
        };

        assert_eq!(g.name(), "gpu-attention-mmap-experts");
        assert!(g.expert_backend.is_none());
    }

    #[test]
    fn gpu_attention_mmap_experts_graph_runs_attention_backend_with_mmap_experts() {
        let w = weights();
        let idx = crate::engines::test_utils::make_test_vindex(w);
        let g = GpuAttentionMmapExpertsGraph {
            index: &idx,
            attention_backend: &larql_compute::CpuBackend,
            expert_backend: None,
            layer_range: 0..w.num_layers,
        };

        let out = g.forward_layer(w, &input(1, w.hidden_size), 0);

        assert!(out.is_some(), "split-residency layer should run");
        assert_eq!(out.unwrap().residual.shape(), &[1, w.hidden_size]);
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn gpu_attention_mmap_experts_graph_accepts_cuda_attention_without_expert_backend() {
        let Some(cuda) = larql_compute::CudaBackend::new() else {
            eprintln!("CUDA unavailable; skipping split-residency CUDA smoke");
            return;
        };
        let w = weights();
        let idx = crate::engines::test_utils::make_test_vindex(w);
        let g = GpuAttentionMmapExpertsGraph {
            index: &idx,
            attention_backend: &cuda,
            expert_backend: None,
            layer_range: 0..w.num_layers,
        };

        let out = g.forward_layer(w, &input(1, w.hidden_size), 0);

        assert!(out.is_some(), "CUDA attention + mmap experts should run");
        assert!(g.attention_backend.name().contains("cuda"));
        assert!(g.expert_backend.is_none(), "experts stay mmap/CPU-backed");
    }
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn cuda_resident_attention_graph_runs_with_mmap_experts() {
        let Some(cuda) = larql_compute::CudaBackend::new() else {
            eprintln!("CUDA unavailable; skipping resident attention graph smoke");
            return;
        };
        let w = weights();
        let idx = crate::engines::test_utils::make_test_vindex(w);
        let residents: Vec<_> = (0..w.num_layers)
            .map(|layer| {
                crate::attention::gpu::CudaAttentionResidency::from_layer(w, &cuda, layer)
                    .expect("resident attention layer")
            })
            .collect();
        let g = CudaResidentAttentionMmapExpertsGraph {
            index: &idx,
            cuda: &cuda,
            resident_attention: &residents,
            expert_backend: None,
            layer_range: 0..w.num_layers,
        };

        let out = g.forward_layer(w, &input(1, w.hidden_size), 0);

        assert!(
            out.is_some(),
            "resident CUDA attention + mmap experts should run"
        );
        assert_eq!(out.unwrap().residual.shape(), &[1, w.hidden_size]);
        assert_eq!(g.name(), "cuda-resident-attention-mmap-experts");
        assert!(g.expert_backend.is_none(), "experts stay mmap/CPU-backed");
    }
}
