//! Q4_K / Q6_K streaming writer — separate from `write_f32` because
//! the Q4_K pipeline owns its own QuantBlockFormat manifest, padding
//! helpers, and per-tensor quantisation policy.
//!
//! Carved out of the monolithic `write.rs` in the 2026-04-25 reorg.

use crate::extract::stage_labels::*;
use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::{VindexConfig, VindexModelConfig};
use crate::error::VindexError;
use crate::extract::callbacks::IndexBuildCallbacks;
use crate::format::filenames::*;

use super::capabilities::{ensure_standard_attention_supported, SURFACE_Q4K_WEIGHT_WRITER};
use super::write_f32::{kind, WeightEntry, WeightSource};

// ── Q4_K / Q6_K streaming writer ──────────────────────────────────────────

/// Per-block quantisation format for a single tensor in the Q4_K pipeline.
/// Serde writes / reads the literal strings `"Q4_K"` and `"Q6_K"` to match
/// llama.cpp / Ollama on-disk conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantBlockFormat {
    #[serde(rename = "Q4_K")]
    Q4K,
    #[serde(rename = "Q6_K")]
    Q6K,
}

// Manifest entry shape moved to `super::manifest::Q4kManifestEntry`
// so the loaders in `index/storage/ffn_store.rs` can deserialise into
// it directly instead of poking `serde_json::Value` with string keys.
use super::manifest::Q4kManifestEntry as Q4kAttnEntry;

pub mod feature_major_down;
use feature_major_down::FeatureMajorDownState;

/// Pad a row-major f32 buffer to the next multiple of 256 with zeros
/// (Q4_K/Q6_K super-blocks require length % 256 == 0).
///
/// Kept only for unit-test coverage of the flat-padding helper pattern;
/// production paths now use [`pad_rows_to_block`] since the shader reads
/// each row as a fixed number of super-blocks.
#[cfg(test)]
fn pad_to_block(data: &[f32]) -> Vec<f32> {
    let block = larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
    let padded_len = data.len().div_ceil(block) * block;
    if padded_len == data.len() {
        data.to_vec()
    } else {
        let mut v = Vec::with_capacity(padded_len);
        v.extend_from_slice(data);
        v.resize(padded_len, 0.0);
        v
    }
}

/// Pad each row of a 2-D row-major matrix to the next multiple of 256 with
/// zeros. Returns `(padded_flat, padded_cols)`.
///
/// Why this exists: Q4_K/Q6_K super-blocks hold exactly 256 values, so the
/// Metal matvec shader computes `bytes_per_row = (cols / 256) * block_size`.
/// When `cols % 256 != 0` (e.g. Gemma 4 26B A4B's `intermediate_size=2112`),
/// flat-padding the whole tensor leaves row boundaries misaligned with
/// super-block boundaries and every row past row 0 reads wrong bytes. Per-row
/// padding realigns each row onto a super-block boundary at the cost of a
/// small storage overhead (the padding columns are zero and contribute
/// nothing to the dot product at dispatch time, provided the caller also
/// zero-pads the input vector to `padded_cols`).
pub(super) fn pad_rows_to_block(data: &[f32], rows: usize, cols: usize) -> (Vec<f32>, usize) {
    debug_assert_eq!(data.len(), rows * cols);
    let block = larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
    let padded_cols = cols.div_ceil(block) * block;
    if padded_cols == cols {
        return (data.to_vec(), cols);
    }
    let mut out = Vec::with_capacity(rows * padded_cols);
    let pad = padded_cols - cols;
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        out.extend_from_slice(row);
        out.extend(std::iter::repeat_n(0.0f32, pad));
    }
    (out, padded_cols)
}

/// Options for [`write_model_weights_q4k_with_opts`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Q4kWriteOptions {
    /// Quantise FFN down-proj as Q4_K instead of Q6_K. Default `false`
    /// preserves the Ollama-compatible "Q4_K_M" mix (Q4_K for gate/up,
    /// Q6_K for down). Setting `true` uses Q4_K uniformly — saves ~30MB
    /// per layer on 31B (1.8GB total) and drops down matmul cost ~1.5-1.7×
    /// to match up-proj timings. Quantisation noise on the scatter-sum
    /// averages across the intermediate dimension; empirically close.
    pub down_q4k: bool,

    /// Emit `down_features_q4k.bin` alongside `interleaved_q4k.bin`.
    /// When set, the down weights are also stored in feature-major
    /// `[intermediate, hidden]` orientation (Q4_K/Q6_K matching
    /// `down_q4k`), so per-feature decode can skip the
    /// `q4k_ffn_layer` whole-layer dequant + transpose cache. Adds
    /// roughly the same disk footprint as the down portion of
    /// `interleaved_q4k.bin` (~14 MB / layer at Gemma 4B dims).
    /// Recommended for CPU sparse walk and grid/MoE workloads where
    /// the ~840 MB heap cache ceiling is the binding constraint.
    /// Default `false` so existing extracts don't grow on disk.
    pub feature_major_down: bool,
}

/// Write model weights in Q4_K/Q6_K format, zero f32 intermediate on disk.
///
/// Emits:
///   attn_weights_q4k.bin + attn_weights_q4k_manifest.json
///     — Q/K/O → Q4_K, V → Q6_K
///     — On layers where V reuses K (Gemma 4 31B global layers), the K
///       bytes are written into the V slot so 4-per-layer indexing stays
///       valid and downstream kernels reading V get K.
///   interleaved_q4k.bin
///     — [gate Q4_K | up Q4_K | down Q6_K] per layer, regular stride.
///     — With `down_q4k=true`: [gate | up | down] all Q4_K.
///   lm_head_q4.bin
///     — Q4_K of the output projection (falls back to embed_tokens when tied).
///   norms.bin (f32, unchanged from non-Q4 path).
///
/// The source's per-tensor f32 materialisation is transient — one tensor's
/// worth of heap (~350 MB peak on 31B global layer Q) quantised then dropped.
pub fn write_model_weights_q4k(
    source: &dyn WeightSource,
    dir: &Path,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    write_model_weights_q4k_with_opts(source, dir, callbacks, Q4kWriteOptions::default())
}

/// Like [`write_model_weights_q4k`] but accepts a [`Q4kWriteOptions`] knob
/// to toggle the FFN down-proj quantisation format.
pub fn write_model_weights_q4k_with_opts(
    source: &dyn WeightSource,
    dir: &Path,
    callbacks: &mut dyn IndexBuildCallbacks,
    opts: Q4kWriteOptions,
) -> Result<(), VindexError> {
    callbacks.on_stage(STAGE_MODEL_WEIGHTS_Q4K);
    let start = std::time::Instant::now();

    let arch = source.arch();
    ensure_standard_attention_supported(arch, SURFACE_Q4K_WEIGHT_WRITER)?;
    let num_layers = source.num_layers();

    // ── attn_weights_q4k.bin ──
    let attn_path = dir.join(ATTN_WEIGHTS_Q4K_BIN);
    let mut attn_file = BufWriter::new(std::fs::File::create(&attn_path)?);
    let mut attn_offset: u64 = 0;
    let mut attn_manifest: Vec<Q4kAttnEntry> = Vec::with_capacity(num_layers * 4);

    for layer in 0..num_layers {
        callbacks.on_layer_start(COMP_ATTN_Q4K, layer, num_layers);

        // Resolve each tensor. For V, fall back to K when v_shares_k=true or
        // v_proj simply isn't present (global layers on 31B).
        let q_key = arch.attn_q_key(layer);
        let k_key = arch.attn_k_key(layer);
        let v_key = arch.attn_v_key(layer);
        let o_key = arch.attn_o_key(layer);

        let q = source.get_tensor(&q_key);
        let k = source.get_tensor(&k_key);
        let v = resolve_v_tensor(source.get_tensor(&v_key), &k, arch.v_shares_k(layer));
        let o = source.get_tensor(&o_key);

        // Q, K, V, O in that order — use the same key string for V even when
        // the data is K's, so loaders that look up by position still work.
        #[allow(clippy::type_complexity)]
        let slots: [(&str, Option<(Vec<f32>, usize, usize)>); 4] = [
            (q_key.as_str(), q),
            (k_key.as_str(), k),
            (v_key.as_str(), v),
            (o_key.as_str(), o),
        ];

        for (i, (key, tensor)) in slots.iter().enumerate() {
            let (data, rows, cols) = match tensor {
                Some(t) => t.clone(),
                None => continue, // tensor genuinely absent — skip
            };

            // V (index 2) gets Q6_K, others get Q4_K.
            let is_v = i == 2;
            // Row-pad to 256 so each row aligns to a super-block boundary.
            // Critical for models with non-256 inner dims (e.g. Gemma 4 26B A4B
            // where the dense intermediate is 2112). `padded_cols` is what the
            // matvec shader must use as `K`; callers also need to zero-pad the
            // input vector to the same width.
            let (padded, padded_cols) = pad_rows_to_block(&data, rows, cols);
            let q_bytes = if is_v {
                quantize_q6_k(&padded)
            } else {
                quantize_q4_k(&padded)
            };
            let format = if is_v {
                QuantBlockFormat::Q6K
            } else {
                QuantBlockFormat::Q4K
            };

            attn_file.write_all(&q_bytes)?;
            let length = q_bytes.len() as u64;
            attn_manifest.push(Q4kAttnEntry {
                key: key.to_string(),
                shape: vec![rows, padded_cols],
                format,
                offset: attn_offset,
                length,
            });
            attn_offset += length;
        }

        callbacks.on_layer_done(COMP_ATTN_Q4K, layer, 0.0);
    }
    attn_file.flush()?;
    drop(attn_file);

    let manifest_json = serde_json::to_string_pretty(&attn_manifest)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(dir.join(ATTN_WEIGHTS_Q4K_MANIFEST_JSON), manifest_json)?;

    // ── interleaved_q4k.bin (FFN gate/up/down) + manifest ──
    //
    // Layer-major: for each layer, `gate Q4_K + up Q4_K + down Q6_K`
    // concatenated. Stride is regular across layers but block sizes
    // depend on the architecture's hidden / intermediate, so we emit a
    // sidecar manifest symmetric with `attn_weights_q4k_manifest.json`.
    // Downstream readers resolve by key + layer instead of recomputing
    // byte offsets; a shape/stride mismatch now fails at load rather
    // than silently corrupting.
    let ff_path = dir.join(INTERLEAVED_Q4K_BIN);
    let mut ff_file = BufWriter::new(std::fs::File::create(&ff_path)?);
    let mut ff_offset: u64 = 0;
    let mut ff_manifest: Vec<Q4kAttnEntry> = Vec::with_capacity(num_layers * 3);

    // ── down_features_q4k.bin (W2 feature-major down, opt-in) ──
    //
    // Captures the same down-proj data as interleaved_q4k.bin's down
    // slot, but transposed to [intermediate, hidden] orientation and
    // re-quantised at the same precision. Lets per-feature decode at
    // load time skip the cache. Allocated lazily so non-opt-in
    // extracts pay nothing.
    let mut fm_state: Option<FeatureMajorDownState> = if opts.feature_major_down {
        Some(FeatureMajorDownState::new(
            &dir.join(DOWN_FEATURES_Q4K_BIN),
            num_layers,
        )?)
    } else {
        None
    };

    for layer in 0..num_layers {
        callbacks.on_layer_start(COMP_FFN_Q4K, layer, num_layers);
        for (i, key) in [
            arch.ffn_gate_key(layer),
            arch.ffn_up_key(layer),
            arch.ffn_down_key(layer),
        ]
        .iter()
        .enumerate()
        {
            if let Some((data, rows, cols)) = source.get_tensor(key) {
                // Row-pad to 256 so each row aligns to a super-block boundary.
                // Without this, matrices with `cols % 256 != 0` (e.g. Gemma 4
                // 26B A4B's down_proj with inner dim 2112) store contiguous
                // quantisation that every row past row 0 reads wrong. See
                // `pad_rows_to_block` docs.
                let (padded, padded_cols) = pad_rows_to_block(&data, rows, cols);
                // Gate (i=0) and up (i=1) always Q4_K. Down (i=2) defaults
                // to Q6_K for llama.cpp compatibility, Q4_K when opts.down_q4k.
                let is_down = i == 2;
                let use_q6 = is_down && !opts.down_q4k;
                let q_bytes = if use_q6 {
                    quantize_q6_k(&padded)
                } else {
                    quantize_q4_k(&padded)
                };
                let format = if use_q6 {
                    QuantBlockFormat::Q6K
                } else {
                    QuantBlockFormat::Q4K
                };
                ff_file.write_all(&q_bytes)?;
                let length = q_bytes.len() as u64;
                ff_manifest.push(Q4kAttnEntry {
                    key: key.clone(),
                    shape: vec![rows, padded_cols],
                    format,
                    offset: ff_offset,
                    length,
                });
                ff_offset += length;

                if is_down {
                    if let Some(state) = fm_state.as_mut() {
                        state.append_layer(key.clone(), &padded, rows, padded_cols, format)?;
                    }
                }
            }
        }
        callbacks.on_layer_done(COMP_FFN_Q4K, layer, 0.0);
    }
    ff_file.flush()?;
    drop(ff_file);

    let ff_manifest_json = serde_json::to_string_pretty(&ff_manifest)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(dir.join(INTERLEAVED_Q4K_MANIFEST_JSON), ff_manifest_json)?;

    if let Some(state) = fm_state.take() {
        state.finalize(&dir.join(DOWN_FEATURES_Q4K_MANIFEST_JSON))?;
    }

    // ── layers/ — per-layer FFN weights (§5.12) ──────────────────────────
    //
    // For MoE models (hybrid MoE PackedBF16, e.g. Gemma 4 26B A4B):
    //   Source BF16 tensors are quantized to Q4_K per expert, written to
    //   layers/layer_{L:02}.weights with num_entries=num_experts.
    //
    // For dense models: interleaved_q4k.bin remains the primary FFN store.
    // Per-layer format for dense is a future migration (--ffn-layout flag).
    //
    // Replaces the old BF16 experts_packed.bin monolithic blob.
    if arch.is_hybrid_moe() && arch.expert_format() == larql_models::ExpertFormat::PackedBF16 {
        use super::write_layers::{quantize_moe_entries, write_layer_weights, LayerWeightFormat};

        let num_experts = arch.num_experts();
        let moe_inter = arch.moe_intermediate_size();
        let hidden = arch.config().hidden_size;

        for layer in 0..num_layers {
            let gu_key = arch.packed_experts_gate_up_key(layer);
            let dn_key = arch.packed_experts_down_key(layer);
            let gu_bytes = gu_key.as_ref().and_then(|k| source.get_packed_bf16(k));
            let dn_bytes = dn_key.as_ref().and_then(|k| source.get_packed_bf16(k));

            if let (Some(gu), Some(dn)) = (gu_bytes, dn_bytes) {
                // Default: Q4_K for the whole file. Format is uniform — no mixing.
                let fmt = LayerWeightFormat::Q4_K;
                let entries = quantize_moe_entries(&gu, &dn, num_experts, moe_inter, hidden, fmt);
                write_layer_weights(dir, layer, fmt, &entries, moe_inter, hidden)?;
            }
        }
    }

    // ── norms.bin (f32, small) ──
    let norms_path = dir.join(NORMS_BIN);
    let mut norms_file = BufWriter::new(std::fs::File::create(&norms_path)?);
    let norms_dtype = crate::config::dtype::StorageDtype::F32;
    let mut norms_offset: u64 = 0;
    let mut norm_entries: Vec<WeightEntry> = Vec::new();

    for layer in 0..num_layers {
        let keys: Vec<String> = [
            Some(arch.input_layernorm_key(layer)),
            Some(arch.post_attention_layernorm_key(layer)),
            arch.pre_feedforward_layernorm_key(layer),
            arch.post_feedforward_layernorm_key(layer),
            arch.attn_q_norm_key(layer),
            arch.attn_k_norm_key(layer),
            // Gemma 4 per-layer scalar multiplier. Stored as a 0-D scalar
            // in safetensors, surfaced through WeightSource as a 1-element
            // vector. The forward path multiplies h by this value after
            // FFN; omitting it silently produced garbage on 31B.
            arch.layer_scalar_key(layer),
            // Gemma 4 E2B per-layer embedding post-norm.
            if arch.has_per_layer_embeddings() {
                arch.post_per_layer_input_norm_key(layer)
            } else {
                None
            },
        ]
        .into_iter()
        .flatten()
        .collect();

        for key in keys {
            if let Some(data) = source.get_vector(&key) {
                let bytes = crate::config::dtype::encode_floats(&data, norms_dtype);
                norms_file.write_all(&bytes)?;
                norm_entries.push(WeightEntry {
                    key: key.clone(),
                    kind: kind::VECTOR.into(),
                    shape: vec![data.len()],
                    offset: norms_offset,
                    length: bytes.len() as u64,
                    file: NORMS_BIN.into(),
                });
                norms_offset += bytes.len() as u64;
            }
        }

        // MoE router + norms (hybrid MoE, e.g. Gemma 4 26B A4B).
        // router.proj.weight is 2D [num_experts, hidden] — flatten and store as "vector".
        // All other MoE keys are 1D vectors.
        if arch.is_hybrid_moe() {
            // 2D router projection — flatten
            if let Some(key) = arch.moe_router_key(layer) {
                if let Some((data, _, _)) = source.get_tensor(&key) {
                    let bytes = crate::config::dtype::encode_floats(&data, norms_dtype);
                    norms_file.write_all(&bytes)?;
                    norm_entries.push(WeightEntry {
                        key: key.clone(),
                        kind: kind::VECTOR.into(),
                        shape: vec![data.len()],
                        offset: norms_offset,
                        length: bytes.len() as u64,
                        file: NORMS_BIN.into(),
                    });
                    norms_offset += bytes.len() as u64;
                }
            }
            // 1D MoE vectors
            let moe_vec_keys: Vec<String> = [
                arch.moe_router_scale_key(layer),
                arch.moe_router_per_expert_scale_key(layer),
                arch.moe_router_norm_key(layer),
                arch.moe_pre_experts_norm_key(layer),
                arch.moe_post_ffn1_norm_key(layer),
                arch.moe_post_experts_norm_key(layer),
                // Outer post-FFN norm used to re-normalise (h1 + h2) before
                // the residual add in hybrid MoE (HF Gemma 4). Distinct from
                // post_ffn1_norm, which is the dense-branch norm.
                arch.moe_post_outer_norm_key(layer),
            ]
            .into_iter()
            .flatten()
            .collect();
            for key in moe_vec_keys {
                if let Some(data) = source.get_vector(&key) {
                    let bytes = crate::config::dtype::encode_floats(&data, norms_dtype);
                    norms_file.write_all(&bytes)?;
                    norm_entries.push(WeightEntry {
                        key: key.clone(),
                        kind: kind::VECTOR.into(),
                        shape: vec![data.len()],
                        offset: norms_offset,
                        length: bytes.len() as u64,
                        file: NORMS_BIN.into(),
                    });
                    norms_offset += bytes.len() as u64;
                }
            }
        }
    }

    // Final model norm (after last layer)
    if let Some(data) = source.get_vector("norm.weight") {
        let bytes = crate::config::dtype::encode_floats(&data, norms_dtype);
        norms_file.write_all(&bytes)?;
        norm_entries.push(WeightEntry {
            key: "norm.weight".into(),
            kind: kind::VECTOR.into(),
            shape: vec![data.len()],
            offset: norms_offset,
            length: bytes.len() as u64,
            file: NORMS_BIN.into(),
        });
        norms_offset += bytes.len() as u64;
    }

    // Gemma 4 E2B PLE global projection norm (small vector).
    if arch.has_per_layer_embeddings() {
        if let Some(data) = source.get_vector("per_layer_projection_norm.weight") {
            let bytes = crate::config::dtype::encode_floats(&data, norms_dtype);
            norms_file.write_all(&bytes)?;
            norm_entries.push(WeightEntry {
                key: "per_layer_projection_norm.weight".into(),
                kind: kind::VECTOR.into(),
                shape: vec![data.len()],
                offset: norms_offset,
                length: bytes.len() as u64,
                file: NORMS_BIN.into(),
            });
        }
    }
    norms_file.flush()?;
    drop(norms_file);

    // ── ple_weights.bin — Per-Layer Embedding tensors (Gemma 4 E2B only) ──
    //
    // Stored as f16 — NOT Q4_K. The two globals (`per_layer_model_projection`,
    // `embed_tokens_per_layer`) and the per-layer input_gate/projection
    // matrices behave like embedding tables: each super-block of 256 values
    // spans a wide dynamic range with a handful of outliers, and Q4_K's
    // per-super-block (d, dmin) calibration zeros out the majority of cells
    // to accommodate those outliers. PLE contributions are additive into
    // every layer's residual, so the cell-level noise compounds across 35
    // layers — the observable result was "arrays" / "amphibians" instead
    // of "Paris" on Gemma 4 E2B. f16 halves the BF16 footprint (~4.7 GB for
    // the big lookup on E2B) and preserves enough precision for accurate
    // per-token PLE retrieval.
    if arch.has_per_layer_embeddings() {
        let ple_path = dir.join("ple_weights.bin");
        let mut ple_file = BufWriter::new(std::fs::File::create(&ple_path)?);
        let mut ple_offset: u64 = 0;
        let ple_dtype = crate::config::dtype::StorageDtype::F16;

        let write_tensor = |file: &mut BufWriter<std::fs::File>,
                            manifest: &mut Vec<WeightEntry>,
                            offset: &mut u64,
                            key: String,
                            data: Option<(Vec<f32>, usize, usize)>|
         -> Result<(), VindexError> {
            if let Some((floats, rows, cols)) = data {
                let bytes = crate::config::dtype::encode_floats(&floats, ple_dtype);
                file.write_all(&bytes)?;
                manifest.push(WeightEntry {
                    key,
                    kind: kind::TENSOR_F16.into(),
                    shape: vec![rows, cols],
                    offset: *offset,
                    length: bytes.len() as u64,
                    file: "ple_weights.bin".into(),
                });
                *offset += bytes.len() as u64;
            }
            Ok(())
        };

        // Global: model projection [ple_dim·num_layers, hidden]
        write_tensor(
            &mut ple_file,
            &mut norm_entries,
            &mut ple_offset,
            "per_layer_model_projection.weight".into(),
            source.get_tensor("per_layer_model_projection.weight"),
        )?;

        // Global: big embedding table [vocab, ple_dim·num_layers]
        if let Some(key) = arch.per_layer_embed_key() {
            write_tensor(
                &mut ple_file,
                &mut norm_entries,
                &mut ple_offset,
                key.clone(),
                source.get_tensor(&key),
            )?;
        }

        // Per-layer: input_gate + projection
        for layer in 0..num_layers {
            if let Some(k) = arch.per_layer_input_gate_key(layer) {
                write_tensor(
                    &mut ple_file,
                    &mut norm_entries,
                    &mut ple_offset,
                    k.clone(),
                    source.get_tensor(&k),
                )?;
            }
            if let Some(k) = arch.per_layer_projection_key(layer) {
                write_tensor(
                    &mut ple_file,
                    &mut norm_entries,
                    &mut ple_offset,
                    k.clone(),
                    source.get_tensor(&k),
                )?;
            }
        }

        ple_file.flush()?;
    }

    // ── lm_head_q4.bin ──
    if let Some((data, rows, cols)) = source.lm_head() {
        let (padded, padded_cols) = pad_rows_to_block(&data, rows, cols);
        let q_bytes = quantize_q4_k(&padded);
        std::fs::write(dir.join(LM_HEAD_Q4_BIN), &q_bytes)?;
        // Record in norms manifest so a single weight_manifest.json references
        // everything non-quantised-via-layout. Shape records the stored
        // `padded_cols` — callers route through the matvec dispatch which
        // uses shape[1] as `K`, so the padding stays invisible provided the
        // input activation buffer is zero-padded to match.
        norm_entries.push(WeightEntry {
            key: "lm_head.weight".into(),
            kind: kind::TENSOR_Q4K.into(),
            shape: vec![rows, padded_cols],
            offset: 0,
            length: q_bytes.len() as u64,
            file: LM_HEAD_Q4_BIN.into(),
        });
    }

    // norms + lm_head manifest (expert weights now in layers/ files, not manifest)
    let all_entries = norm_entries;
    let manifest_json = serde_json::to_string_pretty(&all_entries)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(dir.join(WEIGHT_MANIFEST_JSON), manifest_json)?;

    // ── Update index.json: has_model_weights=true, quant=q4k ──
    let config_path = dir.join(INDEX_JSON);
    let config_text = std::fs::read_to_string(&config_path)?;
    let mut config: VindexConfig =
        serde_json::from_str(&config_text).map_err(|e| VindexError::Parse(e.to_string()))?;

    config.has_model_weights = true;
    config.quant = crate::QuantFormat::Q4K;
    if arch.is_hybrid_moe() {
        config.ffn_layout = Some("per_layer".into());
    }

    let cfg = arch.config();
    config.model_config = Some(VindexModelConfig {
        model_type: cfg.model_type.clone(),
        head_dim: cfg.head_dim,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        rope_base: cfg.rope_base,
        sliding_window: cfg.sliding_window,
        moe: if arch.is_moe() {
            Some(crate::MoeConfig {
                num_experts: arch.num_experts(),
                top_k: arch.num_experts_per_token(),
                shared_expert: arch.num_shared_experts() > 0,
                router_type: arch.moe_router_type().into(),
                moe_intermediate_size: if arch.moe_intermediate_size() > 0 {
                    Some(arch.moe_intermediate_size())
                } else {
                    None
                },
                hybrid: arch.is_hybrid_moe(),
            })
        } else {
            None
        },
        global_head_dim: cfg.global_head_dim,
        num_global_kv_heads: cfg.num_global_kv_heads,
        partial_rotary_factor: cfg.partial_rotary_factor,
        sliding_window_pattern: cfg.sliding_window_pattern,
        layer_types: cfg.layer_types.clone(),
        attention_k_eq_v: cfg.attention_k_eq_v,
        num_kv_shared_layers: cfg.num_kv_shared_layers,
        per_layer_embed_dim: cfg.per_layer_embed_dim,
        rope_local_base: cfg.rope_local_base,
        query_pre_attn_scalar: cfg.query_pre_attn_scalar,
        final_logit_softcapping: cfg.final_logit_softcapping,
        ..Default::default()
    });

    let config_json =
        serde_json::to_string_pretty(&config).map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(&config_path, config_json)?;

    callbacks.on_stage_done(
        STAGE_MODEL_WEIGHTS_Q4K,
        start.elapsed().as_secs_f64() * 1000.0,
    );
    Ok(())
}

/// Resolve the V tensor for a layer in the Q4_K writer.
///
/// When `v_proj` is absent from the source (e.g. Gemma 4 31B global
/// layers ship without one), fall back to K's tensor if the
/// architecture advertises `v_shares_k(layer) == true`. This keeps
/// the 4-per-layer attn manifest contiguous: each layer emits exactly
/// Q / K / V / O even when V physically reuses K's bytes.
fn resolve_v_tensor<T: Clone>(v: Option<T>, k: &Option<T>, v_shares_k: bool) -> Option<T> {
    v.or_else(|| if v_shares_k { k.clone() } else { None })
}

#[cfg(test)]
mod helper_tests {
    use super::*;

    // ── resolve_v_tensor ──

    #[test]
    fn resolve_v_returns_v_when_present() {
        let k = Some(2);
        assert_eq!(resolve_v_tensor(Some(1), &k, false), Some(1));
        assert_eq!(
            resolve_v_tensor(Some(1), &k, true),
            Some(1),
            "v_shares_k must not override a present v"
        );
    }

    #[test]
    fn resolve_v_falls_back_to_k_when_v_shared() {
        let k = Some(42);
        assert_eq!(
            resolve_v_tensor(None::<i32>, &k, true),
            Some(42),
            "Gemma 4 31B global-layer fallback"
        );
    }

    #[test]
    fn resolve_v_none_when_missing_and_not_shared() {
        let k = Some(7);
        assert_eq!(
            resolve_v_tensor(None::<i32>, &k, false),
            None,
            "no v_proj + v_shares_k=false → tensor is genuinely absent"
        );
    }

    #[test]
    fn resolve_v_none_when_v_missing_and_k_missing() {
        let k: Option<i32> = None;
        assert_eq!(resolve_v_tensor(None, &k, true), None);
        assert_eq!(resolve_v_tensor(None, &k, false), None);
    }

    // ── pad_to_block ──

    #[test]
    fn pad_to_block_noop_when_exact_multiple() {
        let v = vec![1.0_f32; 256];
        let padded = pad_to_block(&v);
        assert_eq!(padded.len(), 256, "exact multiple must not grow");
        assert_eq!(padded, v);

        let v = vec![1.0_f32; 512];
        let padded = pad_to_block(&v);
        assert_eq!(padded.len(), 512);
    }

    #[test]
    fn pad_to_block_zero_fills_to_next_block() {
        let v = vec![1.0_f32; 200];
        let padded = pad_to_block(&v);
        assert_eq!(padded.len(), 256, "padded to next super-block");
        // First 200 preserved, last 56 zeroed.
        assert!(padded[..200].iter().all(|&x| x == 1.0));
        assert!(padded[200..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn pad_to_block_handles_one_below_multiple() {
        let v = vec![1.0_f32; 255];
        let padded = pad_to_block(&v);
        assert_eq!(padded.len(), 256);
        assert_eq!(padded[255], 0.0);
    }

    #[test]
    fn pad_to_block_handles_one_above_multiple() {
        let v = vec![1.0_f32; 257];
        let padded = pad_to_block(&v);
        assert_eq!(
            padded.len(),
            512,
            "one above block boundary → next full block"
        );
        assert!(padded[..257].iter().all(|&x| x == 1.0));
        assert!(padded[257..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn pad_to_block_empty_input_stays_empty() {
        let v: Vec<f32> = Vec::new();
        let padded = pad_to_block(&v);
        assert_eq!(padded.len(), 0);
    }
}
