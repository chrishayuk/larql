//! Stage 1 — gate vectors (streaming, one layer at a time).

use std::io::{BufWriter, Write};

use crate::config::dtype::write_floats;
use crate::config::VindexLayerInfo;
use crate::error::VindexError;
use crate::extract::stage_labels::*;
use crate::extract::streaming::context::StreamingContext;
use crate::extract::streaming::tensor_io::{get_tensor_f32, normalize_key, GateSink};
use crate::format::filenames::*;

impl<'a> StreamingContext<'a> {
    /// Stage 1 — gate vectors (streaming, one layer at a time).
    ///
    /// If `drop_gate_vectors` is set we still walk every layer to build
    /// `layer_infos` (num_features per layer is part of `index.json`)
    /// but redirect writes to `/dev/null` (`io::sink`). The gate bytes
    /// are recoverable from `interleaved_kquant.bin` at load time.
    pub(in crate::extract::streaming) fn write_gate_vectors(&mut self) -> Result<(), VindexError> {
        self.callbacks.on_stage(STAGE_GATE_VECTORS);
        let gate_path = self.output_dir.join(GATE_VECTORS_BIN);

        // Auto-resume: if a prior run finished the gate phase and saved
        // `gate_layer_infos`, reuse it and skip the gate loop entirely.
        let resumed_gate = self
            .checkpoint
            .is_complete(crate::extract::checkpoint::ExtractPhase::Gate)
            && self.checkpoint.gate_layer_infos.is_some();
        self.layer_infos = if resumed_gate {
            eprintln!(
                "  Skipping gate phase ({} layer infos restored from checkpoint; \
                 reusing existing {})",
                self.checkpoint
                    .gate_layer_infos
                    .as_ref()
                    .map(|v| v.len())
                    .unwrap_or(0),
                GATE_VECTORS_BIN,
            );
            self.callbacks.on_stage_done(STAGE_GATE_VECTORS, 0.0);
            self.checkpoint.gate_layer_infos.clone().unwrap_or_default()
        } else {
            Vec::new()
        };

        // Only allocate the writer + run the loop when the phase isn't
        // already done.
        let mut gate_file: GateSink = if resumed_gate || self.drop_gate_vectors {
            GateSink::Discard(std::io::sink())
        } else {
            GateSink::File(BufWriter::new(std::fs::File::create(&gate_path)?))
        };
        let mut offset: u64 = 0;
        let prefixes: Vec<&str> = self.prefixes.iter().map(|s| s.as_str()).collect();

        // Skip the per-layer gate loop entirely on resume.
        let layer_count_for_loop = if resumed_gate { 0 } else { self.num_layers };
        for layer in 0..layer_count_for_loop {
            self.callbacks
                .on_layer_start(COMP_GATE, layer, self.num_layers);
            let start = std::time::Instant::now();

            if self.expert_format == larql_models::ExpertFormat::PackedMxfp4 {
                // MXFP4 packed experts: dequantize gate_up_proj_blocks per layer
                // The fused tensor is [num_experts, 2*intermediate, groups, 16]
                // First half of output features = gate, second half = up
                let blocks_key = self
                    .arch
                    .packed_gate_up_blocks_key(layer)
                    .unwrap_or_default();
                let scales_key = self
                    .arch
                    .packed_gate_up_scales_key(layer)
                    .unwrap_or_default();

                if let (Some(blocks_info), Some(scales_info)) = (
                    self.tensor_index.get(&blocks_key),
                    self.tensor_index.get(&scales_key),
                ) {
                    let blocks_st = safetensors::SafeTensors::deserialize(
                        &self.shard_mmaps[blocks_info.0].mmap,
                    )
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
                    let scales_st = safetensors::SafeTensors::deserialize(
                        &self.shard_mmaps[scales_info.0].mmap,
                    )
                    .map_err(|e| VindexError::Parse(e.to_string()))?;

                    let blocks_view = blocks_st
                        .tensor(&blocks_info.1)
                        .map_err(|e| VindexError::Parse(e.to_string()))?;
                    let scales_view = scales_st
                        .tensor(&scales_info.1)
                        .map_err(|e| VindexError::Parse(e.to_string()))?;

                    let shape = blocks_view.shape();
                    let n_exp = shape[0];
                    let out_features = shape[1]; // 2 * intermediate (fused gate+up)
                    let groups = shape[2];
                    let in_features = groups * 32;
                    let half = out_features / 2; // gate portion

                    let experts = crate::format::quant::mxfp4::dequantize_all_experts(
                        blocks_view.data(),
                        scales_view.data(),
                        n_exp,
                        out_features,
                        groups,
                    )?;

                    let mut total_features = 0usize;
                    let mut layer_bytes = 0u64;

                    for expert_data in &experts {
                        // Extract gate portion (first half rows)
                        let gate_data = &expert_data[..half * in_features];
                        layer_bytes += write_floats(&mut gate_file, gate_data, self.dtype)?;
                        total_features += half;
                    }

                    if total_features > 0 {
                        self.layer_infos.push(VindexLayerInfo {
                            layer,
                            num_features: total_features,
                            offset,
                            length: layer_bytes,
                            num_experts: Some(n_exp),
                            num_features_per_expert: Some(half),
                        });
                        offset += layer_bytes;
                    }
                }
            } else if self.expert_format == larql_models::ExpertFormat::PackedBF16 && self.is_moe {
                // Hybrid MoE (Gemma 4 26B A4B): packed experts stored separately.
                // gate_vectors.bin uses the dense FFN gate for KNN walk routing.
                let gate_key = normalize_key(&self.arch.ffn_gate_key(layer), &prefixes);
                if let Some(tensor) =
                    get_tensor_f32(&self.shard_mmaps, &self.tensor_index, &gate_key)?
                {
                    let num_features = tensor.shape()[0];
                    let data = tensor.as_slice().unwrap();
                    let length = write_floats(&mut gate_file, data, self.dtype)?;
                    self.layer_infos.push(VindexLayerInfo {
                        layer,
                        num_features,
                        offset,
                        length,
                        num_experts: None,
                        num_features_per_expert: None,
                    });
                    offset += length;
                }
            } else if self.is_moe && self.n_experts > 0 {
                // Standard MoE (Mixtral): per-expert gate tensors. Two modes:
                //
                //  • Default: write the full per-expert gate matrix
                //    (shape [num_features, hidden]) — fine for low-expert-count
                //    MoE (Mixtral's 8 per layer = ~1.8 GB/layer at hidden=4096).
                //
                //  • Summary: when LARQL_SUMMARY_FEATURES_PER_EXPERT is set
                //    to a positive integer K, do a top-K randomized SVD of
                //    each expert's gate_proj and write only the top-K right
                //    singular vectors (K × hidden floats per expert). Required
                //    for many-experts MoE (DeepSeek-V4 family at 256-384
                //    experts/layer would otherwise produce 100s of GB).
                let summary_k = std::env::var("LARQL_SUMMARY_FEATURES_PER_EXPERT")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);

                let mut total_features = 0usize;
                let mut layer_bytes = 0u64;
                let mut features_per_expert = 0usize;

                for expert in 0..self.n_experts {
                    let gate_key = match self.arch.expert_ffn_gate_key(layer, expert) {
                        Some(k) => normalize_key(&k, &prefixes),
                        None => continue,
                    };

                    if let Some(tensor) =
                        get_tensor_f32(&self.shard_mmaps, &self.tensor_index, &gate_key)?
                    {
                        let data: Vec<f32>;
                        let n_feat: usize;
                        if summary_k > 0 && tensor.shape()[0] > summary_k {
                            // SVD-summary path: top-K right singular vectors.
                            // Seed with (layer, expert) so re-runs are
                            // bit-identical but per-expert uncorrelated.
                            let seed = ((layer as u64) << 32) | (expert as u64);
                            let vt = crate::extract::moe_svd::top_k_right_singular_vectors(
                                tensor.view(),
                                summary_k,
                                /*p_iters=*/ 4,
                                seed,
                            );
                            n_feat = summary_k;
                            data = vt.as_slice().unwrap().to_vec();
                        } else {
                            // Full-matrix path: original behaviour.
                            n_feat = tensor.shape()[0];
                            data = tensor.as_slice().unwrap().to_vec();
                        }
                        features_per_expert = n_feat;
                        total_features += n_feat;
                        layer_bytes += write_floats(&mut gate_file, &data, self.dtype)?;
                    }
                }

                if total_features > 0 {
                    self.layer_infos.push(VindexLayerInfo {
                        layer,
                        num_features: total_features,
                        offset,
                        length: layer_bytes,
                        num_experts: Some(self.n_experts),
                        num_features_per_expert: Some(features_per_expert),
                    });
                    offset += layer_bytes;
                }
            } else {
                // Dense: single gate matrix per layer
                let gate_key = normalize_key(&self.arch.ffn_gate_key(layer), &prefixes);
                if let Some(tensor) =
                    get_tensor_f32(&self.shard_mmaps, &self.tensor_index, &gate_key)?
                {
                    let num_features = tensor.shape()[0];
                    let data = tensor.as_slice().unwrap();
                    let length = write_floats(&mut gate_file, data, self.dtype)?;
                    self.layer_infos.push(VindexLayerInfo {
                        layer,
                        num_features,
                        offset,
                        length,
                        num_experts: None,
                        num_features_per_expert: None,
                    });
                    offset += length;
                }
            }

            self.callbacks
                .on_layer_done(COMP_GATE, layer, start.elapsed().as_secs_f64() * 1000.0);
        }
        gate_file.flush()?;
        // If we were only sinking bytes, don't leave a zero-byte
        // gate_vectors.bin behind for the loader to trip over.
        drop(gate_file);
        if self.drop_gate_vectors && gate_path.exists() && !resumed_gate {
            let _ = std::fs::remove_file(&gate_path);
        }
        if !resumed_gate {
            self.callbacks.on_stage_done(STAGE_GATE_VECTORS, 0.0);
            self.checkpoint
                .mark_gate_complete(self.layer_infos.clone(), self.output_dir)?;
        }
        Ok(())
    }
}
