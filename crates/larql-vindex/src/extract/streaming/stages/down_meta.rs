//! Stage 3 — down meta (streaming).

use ndarray::Array2;

use crate::error::VindexError;
use crate::extract::constants::FEATURE_PROJECTION_BATCH;
use crate::extract::stage_labels::*;
use crate::extract::streaming::context::StreamingContext;
use crate::extract::streaming::tensor_io::normalize_key;
use crate::format::filenames::*;

impl<'a> StreamingContext<'a> {
    /// Stage 3 — down meta (streaming).
    ///
    /// Auto-resume: skip the entire down-meta phase if the prior run
    /// already wrote `down_meta.bin`. The file is opaque to us here
    /// (we don't reload it), but the loader at the end uses it
    /// directly off disk via `mmap`, and the config-write doesn't
    /// need any per-layer state from this phase — so a clean skip is
    /// safe.
    pub(in crate::extract::streaming) fn write_down_meta(&mut self) -> Result<(), VindexError> {
        let resumed_down = self
            .checkpoint
            .is_complete(crate::extract::checkpoint::ExtractPhase::DownMeta);
        self.callbacks.on_stage(STAGE_DOWN_META);
        if resumed_down {
            eprintln!(
                "  Skipping down_meta phase (reusing existing {})",
                DOWN_META_BIN,
            );
        }
        let mut all_down_meta: Vec<Option<Vec<Option<crate::FeatureMeta>>>> =
            vec![None; self.num_layers];

        let embed = self
            .embed
            .as_ref()
            .expect("embeddings stage must run before down_meta stage");

        // Build whole-word vocab once
        let (_ww_ids, _ww_embed) = crate::extract::build_helpers::build_whole_word_vocab(
            self.tokenizer,
            embed,
            self.vocab_size,
            self.hidden_size,
        );

        let prefixes: Vec<&str> = self.prefixes.iter().map(|s| s.as_str()).collect();
        let down_layer_count = if resumed_down { 0 } else { self.num_layers };
        // Index-based loop (rather than `iter_mut().enumerate()`) so the
        // mutable borrow on `all_down_meta[layer]` is released between
        // iterations — letting the per-layer incremental flush below take
        // an immutable borrow of the whole accumulator.
        for layer in 0..down_layer_count {
            let layer_down_meta = &mut all_down_meta[layer];
            self.callbacks
                .on_layer_start(COMP_DOWN, layer, self.num_layers);
            let start = std::time::Instant::now();

            // Get down matrices for this layer
            let down_matrices: Vec<Array2<f32>> = if self.expert_format
                == larql_models::ExpertFormat::PackedMxfp4
            {
                // MXFP4: dequantize down_proj_blocks. Safetensors-only —
                // GGUF has no equivalent packed-MXFP4 format.
                let (shard_mmaps, tensor_index) = match self.tensor_source.safetensors_view() {
                    Some(v) => v,
                    None => {
                        self.callbacks.on_layer_done(COMP_DOWN, layer, 0.0);
                        continue;
                    }
                };
                let blocks_key = self.arch.packed_down_blocks_key(layer).unwrap_or_default();
                let scales_key = self.arch.packed_down_scales_key(layer).unwrap_or_default();
                if let (Some(bi), Some(si)) = (
                    tensor_index.get(&blocks_key),
                    tensor_index.get(&scales_key),
                ) {
                    let bst = safetensors::SafeTensors::deserialize(&shard_mmaps[bi.0].mmap)
                        .map_err(|e| VindexError::Parse(e.to_string()))?;
                    let sst = safetensors::SafeTensors::deserialize(&shard_mmaps[si.0].mmap)
                        .map_err(|e| VindexError::Parse(e.to_string()))?;
                    let bv = bst
                        .tensor(&bi.1)
                        .map_err(|e| VindexError::Parse(e.to_string()))?;
                    let sv = sst
                        .tensor(&si.1)
                        .map_err(|e| VindexError::Parse(e.to_string()))?;
                    let shape = bv.shape();
                    let n_exp = shape[0];
                    let out_features = shape[1];
                    let groups = shape[2];
                    let in_features = groups * 32;
                    let experts = crate::format::quant::mxfp4::dequantize_all_experts(
                        bv.data(),
                        sv.data(),
                        n_exp,
                        out_features,
                        groups,
                    )?;
                    experts
                        .into_iter()
                        .map(|data| {
                            Array2::from_shape_vec((out_features, in_features), data).unwrap()
                        })
                        .collect()
                } else {
                    self.callbacks.on_layer_done(COMP_DOWN, layer, 0.0);
                    continue;
                }
            } else if self.expert_format == larql_models::ExpertFormat::PackedBF16 && self.is_moe {
                // Hybrid MoE (Gemma 4 26B A4B): use dense FFN down for down_meta.
                // Expert down matrices live per-layer at `layers/layer_{L:02}.weights`
                // (Q4_K), written by the q4k weight writer.
                let down_key = normalize_key(&self.arch.ffn_down_key(layer), &prefixes);
                match self.tensor_source.get_tensor_f32(&down_key)? {
                    Some(t) => vec![t],
                    None => {
                        self.callbacks.on_layer_done(COMP_DOWN, layer, 0.0);
                        continue;
                    }
                }
            } else if self.is_moe && self.n_experts > 0 {
                let mut mats = Vec::new();
                for expert in 0..self.n_experts {
                    if let Some(key) = self.arch.expert_ffn_down_key(layer, expert) {
                        let nk = normalize_key(&key, &prefixes);
                        if let Some(t) = self.tensor_source.get_tensor_f32(&nk)? {
                            mats.push(t);
                        }
                    }
                }
                mats
            } else {
                let down_key = normalize_key(&self.arch.ffn_down_key(layer), &prefixes);
                match self.tensor_source.get_tensor_f32(&down_key)? {
                    Some(t) => vec![t],
                    None => {
                        self.callbacks.on_layer_done(COMP_DOWN, layer, 0.0);
                        continue;
                    }
                }
            };

            if down_matrices.is_empty() {
                self.callbacks.on_layer_done(COMP_DOWN, layer, 0.0);
                continue;
            }

            // Same `LARQL_SUMMARY_FEATURES_PER_EXPERT` env that gates the
            // gate-vectors SVD path also caps how many down_proj feature
            // columns we compute meta for. Without this cap, many-experts
            // MoE explodes:
            //   43 layers × 256 experts × 2048 features × (vocab × hidden)
            //   ≈ 12 PFLOPs ≈ 67 hours of CPU.
            // With K=64: ~32× speedup → ~2 hrs total. Limitation: records
            // meta for the first K columns rather than the SVD-selected
            // most-important features. Acceptable for the summary tier —
            // full meta is still available via the default
            // `--summary-features-per-expert 0`.
            let summary_k = std::env::var("LARQL_SUMMARY_FEATURES_PER_EXPERT")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);

            let mut feature_offset = 0usize;
            for w_down in &down_matrices {
                let full_features = w_down.shape()[1];
                let num_features = if summary_k > 0 && full_features > summary_k {
                    summary_k
                } else {
                    full_features
                };
                let batch_size = FEATURE_PROJECTION_BATCH;

                for batch_start in (0..num_features).step_by(batch_size) {
                    let batch_end = (batch_start + batch_size).min(num_features);
                    self.callbacks.on_feature_progress(
                        "down",
                        layer,
                        feature_offset + batch_start,
                        down_matrices.iter().map(|m| m.shape()[1]).sum(),
                    );

                    let w_chunk = w_down
                        .slice(ndarray::s![.., batch_start..batch_end])
                        .to_owned();
                    let cpu = larql_compute::CpuBackend;
                    use larql_compute::MatMul;
                    let chunk_logits = cpu.matmul(embed.view(), w_chunk.view());

                    for feat in batch_start..batch_end {
                        let col = chunk_logits.column(feat - batch_start);
                        let mut scores: Vec<(usize, f32)> =
                            col.iter().copied().enumerate().collect();
                        let k = self.down_top_k.min(scores.len());
                        if k > 0 && k < scores.len() {
                            scores.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
                        }
                        scores.truncate(k);
                        scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                        let top_k_entries: Vec<larql_models::TopKEntry> = scores
                            .into_iter()
                            .filter_map(|(idx, logit)| {
                                self.tokenizer
                                    .decode(&[idx as u32], true)
                                    .ok()
                                    .map(|s| s.trim().to_string())
                                    .filter(|s| !s.is_empty())
                                    .map(|token| larql_models::TopKEntry {
                                        token,
                                        token_id: idx as u32,
                                        logit,
                                    })
                            })
                            .collect();

                        let (top_token, top_token_id, c_score) =
                            if let Some(first) = top_k_entries.first() {
                                (first.token.clone(), first.token_id, first.logit)
                            } else {
                                (String::new(), 0, 0.0)
                            };

                        let feat_idx = feature_offset + feat;
                        if layer_down_meta.is_none() {
                            *layer_down_meta = Some(Vec::new());
                        }
                        if let Some(ref mut metas) = layer_down_meta {
                            while metas.len() <= feat_idx {
                                metas.push(None);
                            }
                            metas[feat_idx] = Some(crate::FeatureMeta {
                                top_token,
                                top_token_id,
                                c_score,
                                top_k: top_k_entries,
                            });
                        }
                    }
                }
                feature_offset += num_features;
            }

            self.callbacks
                .on_layer_done(COMP_DOWN, layer, start.elapsed().as_secs_f64() * 1000.0);

            // Incremental flush: after each layer's projection finishes,
            // snapshot the accumulator to `down_meta.bin` so an interrupted
            // run preserves completed layers. `write_binary` already uses
            // a tempfile + atomic rename so the on-disk file is never in
            // a partial state. Cost is one ~1.5 MB write per layer — well
            // under the per-layer matmul time even on dense models.
            crate::format::down_meta::write_binary(
                self.output_dir,
                &all_down_meta,
                self.down_top_k,
            )?;
        }

        if !resumed_down {
            // Final write (idempotent — same content as the last
            // per-layer snapshot above when the loop ran to completion).
            crate::format::down_meta::write_binary(
                self.output_dir,
                &all_down_meta,
                self.down_top_k,
            )?;
            self.callbacks.on_stage_done(STAGE_DOWN_META, 0.0);
            self.checkpoint.mark(
                crate::extract::checkpoint::ExtractPhase::DownMeta,
                self.output_dir,
            )?;
        }
        Ok(())
    }
}
