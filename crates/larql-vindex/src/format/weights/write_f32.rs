//! Model weights serialization to/from .vindex directories.
//!
//! Split format (v2): separate files per component, no duplication.
//!   attn_weights.bin  — Q, K, V, O per layer
//!   up_weights.bin    — FFN up projections (gate is in gate_vectors.bin)
//!   down_weights.bin  — FFN down projections
//!   norms.bin         — all LayerNorm/RMSNorm vectors
//!   lm_head.bin       — output projection
//!
//! Both the build path (full ModelWeights in RAM) and the streaming path
//! (mmap'd safetensors) write through the same `write_model_weights` function
//! via the `WeightSource` trait.

use crate::extract::stage_labels::*;
use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::{VindexConfig, VindexModelConfig};
use crate::error::VindexError;
use crate::extract::callbacks::IndexBuildCallbacks;
use crate::format::filenames::*;
use crate::format::load::load_vindex_config;

use super::capabilities::{ensure_standard_attention_supported, SURFACE_F32_WEIGHT_WRITER};
use larql_models::ModelWeights;

/// Manifest `kind` discriminators — wire-format strings written into
/// `weights.json`. Constants exist so writers and the loader's match
/// arm dispatch on the same source-of-truth. A typo on a constant
/// fails to compile; a typo in a string literal would silently route
/// the wrong format and reproduce the Q4_K-vs-Q4_0 lm_head bug.
pub mod kind {
    /// 1D float vector (norms, biases, scalars), stored as f32 or f16
    /// raw bytes. Decoded via `crate::config::dtype::decode_floats`.
    pub const VECTOR: &str = "vector";
    /// 2D f32/f16 dense tensor (raw row-major bytes). Used by the legacy
    /// `write_f32` writer for attn/FFN weights.
    pub const TENSOR: &str = "tensor";
    /// 2D Q4_K-quantised tensor (256-element super-blocks, 144 B/block).
    pub const TENSOR_Q4K: &str = "tensor_q4k";
    /// 2D f16 tensor (e.g. Gemma 4 PLE weights).
    pub const TENSOR_F16: &str = "tensor_f16";
    /// 3D BF16-packed expert tensor (Gemma 4 26B-A4B `experts.gate_up_proj`,
    /// `experts.down_proj`). Range-tracked, not cloned (can be 43 GB).
    pub const PACKED_BF16: &str = "packed_bf16";
}

#[derive(Serialize, Deserialize)]
pub struct WeightEntry {
    pub key: String,
    pub kind: String,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub length: u64,
    #[serde(default)]
    pub file: String,
}

// ── WeightSource trait ──

/// Abstraction over where model weights come from.
///
/// Implemented by `ModelWeights` (build path — everything in RAM)
/// and `StreamingWeights` (streaming path — mmap'd safetensors on demand).
pub trait WeightSource {
    /// Get a 2D weight tensor by normalized key. Returns (data, rows, cols).
    fn get_tensor(&self, key: &str) -> Option<(Vec<f32>, usize, usize)>;

    /// Get a 1D vector (norm weights, biases) by normalized key.
    fn get_vector(&self, key: &str) -> Option<Vec<f32>>;

    /// Architecture handle for key generation.
    fn arch(&self) -> &dyn larql_models::ModelArchitecture;

    /// Number of layers.
    fn num_layers(&self) -> usize;

    /// LM head matrix. Returns (data, rows, cols).
    fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)>;

    /// All 1D vector names (for norms).
    fn vector_names(&self) -> Vec<String>;

    /// Raw BF16 bytes for a packed expert tensor (e.g. Gemma 4 experts.gate_up_proj).
    /// Returns None if the key is absent or the tensor is not BF16.
    fn get_packed_bf16(&self, key: &str) -> Option<Vec<u8>>;
}

// ── ModelWeights implementation ──

impl WeightSource for ModelWeights {
    fn get_tensor(&self, key: &str) -> Option<(Vec<f32>, usize, usize)> {
        let t = self.tensors.get(key)?;
        Some((t.as_slice()?.to_vec(), t.shape()[0], t.shape()[1]))
    }

    fn get_vector(&self, key: &str) -> Option<Vec<f32>> {
        self.vectors.get(key).cloned()
    }

    fn arch(&self) -> &dyn larql_models::ModelArchitecture {
        &*self.arch
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)> {
        let h = &self.lm_head;
        Some((h.as_slice()?.to_vec(), h.shape()[0], h.shape()[1]))
    }

    fn vector_names(&self) -> Vec<String> {
        self.vectors.keys().cloned().collect()
    }

    fn get_packed_bf16(&self, key: &str) -> Option<Vec<u8>> {
        self.raw_bytes.get(key).cloned()
    }
}

// ── Streaming implementation ──

/// Weight source backed by mmap'd safetensors files.
/// Tensors are deserialized on demand — peak memory is one tensor at a time.
pub struct StreamingWeights<'a> {
    pub shard_mmaps: &'a [&'a [u8]],
    pub tensor_index: &'a HashMap<String, (usize, String)>,
    pub arch: &'a dyn larql_models::ModelArchitecture,
    pub num_layers: usize,
}

impl<'a> StreamingWeights<'a> {
    fn read_tensor_raw(&self, key: &str) -> Option<(Vec<f32>, Vec<usize>)> {
        let (shard_idx, tensor_name) = self.tensor_index.get(key)?;
        let st = safetensors::SafeTensors::deserialize(self.shard_mmaps[*shard_idx]).ok()?;
        let view = st.tensor(tensor_name).ok()?;
        let shape = view.shape().to_vec();

        let data = match view.dtype() {
            safetensors::Dtype::F32 => view
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            safetensors::Dtype::F16 => crate::format::quant::half::decode_f16(view.data()),
            safetensors::Dtype::BF16 => crate::format::quant::half::decode_bf16(view.data()),
            _ => return None,
        };
        Some((data, shape))
    }
}

impl<'a> WeightSource for StreamingWeights<'a> {
    fn get_tensor(&self, key: &str) -> Option<(Vec<f32>, usize, usize)> {
        let (data, shape) = self.read_tensor_raw(key)?;
        if shape.len() != 2 {
            return None;
        }
        Some((data, shape[0], shape[1]))
    }

    fn get_vector(&self, key: &str) -> Option<Vec<f32>> {
        let (data, shape) = self.read_tensor_raw(key)?;
        if shape.len() != 1 {
            return None;
        }
        Some(data)
    }

    fn arch(&self) -> &dyn larql_models::ModelArchitecture {
        self.arch
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)> {
        // Try common lm_head key names
        for key in &["lm_head.weight", "output.weight"] {
            if let Some(t) = self.get_tensor(key) {
                return Some(t);
            }
        }
        None
    }

    fn vector_names(&self) -> Vec<String> {
        // Return all 1D tensor keys (norms, biases)
        let mut names = Vec::new();
        for key in self.tensor_index.keys() {
            if key.contains("layernorm") || key.contains("norm") || key.contains("bias") {
                names.push(key.clone());
            }
        }
        names.sort();
        names
    }

    fn get_packed_bf16(&self, key: &str) -> Option<Vec<u8>> {
        let (shard_idx, tensor_name) = self.tensor_index.get(key)?;
        let st = safetensors::SafeTensors::deserialize(self.shard_mmaps[*shard_idx]).ok()?;
        let view = st.tensor(tensor_name).ok()?;
        if view.dtype() != safetensors::Dtype::BF16 {
            return None;
        }
        Some(view.data().to_vec())
    }
}

// ── Write model weights (generic over source) ──

/// Options for [`write_model_weights_with_opts`]. Use
/// `WriteWeightsOptions::default()` to get the legacy behavior (writes
/// every component file — equivalent to `ExtractLevel::All`).
#[derive(Clone, Copy, Debug)]
pub struct WriteWeightsOptions {
    /// Extract tier — controls which component files are written.
    /// Attention tier writes attn + norms only; Inference adds FFN;
    /// All adds lm_head. See [`crate::ExtractLevel`] for full semantics.
    ///
    /// **Default is `All`, not `Browse`.** Callers of `write_model_weights`
    /// have already decided weights should be written; the CLI-facing
    /// `ExtractLevel::default() == Browse` is the "I want a KNN-only
    /// vindex" intent and is gated out earlier in the extract pipeline.
    pub level: crate::ExtractLevel,

    /// Skip writing `up_weights.bin` + `down_weights.bin`. The up/down
    /// weights are expected to be available via feature-major
    /// `up_features.bin` + `down_features.bin` — the loader
    /// reconstructs the hidden-major tensors from those when the
    /// manifest-referenced files are missing.
    ///
    /// On a 4B f16 vindex this saves ~3.4 GB (1.7 GB per tensor). On a
    /// 31B vindex, proportionally ~14 GB. The cost is non-zero load
    /// time (one mmap + transpose per layer for down, direct view for
    /// up).
    ///
    /// Only take this option if `up_features.bin` and `down_features.bin`
    /// are already in the output directory or will be produced
    /// afterwards; otherwise downstream dense paths
    /// (`WeightFfn::forward`, MEMIT) will panic on missing tensors.
    pub ffn_compact: bool,
}

impl Default for WriteWeightsOptions {
    fn default() -> Self {
        Self {
            level: crate::ExtractLevel::All,
            ffn_compact: false,
        }
    }
}

/// Write model weights to split component files.
///
/// Works with any `WeightSource`: ModelWeights (build path) or
/// StreamingWeights (streaming path from mmap'd safetensors).
pub fn write_model_weights(
    source: &dyn WeightSource,
    dir: &Path,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    write_model_weights_with_opts(source, dir, callbacks, WriteWeightsOptions::default())
}

/// Explicit-options variant of [`write_model_weights`].
pub fn write_model_weights_with_opts(
    source: &dyn WeightSource,
    dir: &Path,
    callbacks: &mut dyn IndexBuildCallbacks,
    opts: WriteWeightsOptions,
) -> Result<(), VindexError> {
    callbacks.on_stage(STAGE_MODEL_WEIGHTS);
    let start = std::time::Instant::now();

    let dtype = load_vindex_config(dir)
        .map(|c| c.dtype)
        .unwrap_or(crate::config::dtype::StorageDtype::F32);

    let arch = source.arch();
    ensure_standard_attention_supported(arch, SURFACE_F32_WEIGHT_WRITER)?;
    let num_layers = source.num_layers();
    let mut entries: Vec<WeightEntry> = Vec::new();

    // ── Attention weights ── (skipped when level < Attention)
    let write_attn = opts.level.writes_attn();
    let write_ffn = opts.level.writes_ffn() && !opts.ffn_compact;
    let write_lm_head = opts.level.writes_lm_head();

    if write_attn {
        let attn_path = dir.join(ATTN_WEIGHTS_BIN);
        let mut attn_file = BufWriter::new(std::fs::File::create(&attn_path)?);
        let mut attn_offset: u64 = 0;

        for layer in 0..num_layers {
            callbacks.on_layer_start(COMP_ATTN_WEIGHTS, layer, num_layers);
            for key in &[
                arch.attn_q_key(layer),
                arch.attn_k_key(layer),
                arch.attn_v_key(layer),
                arch.attn_o_key(layer),
            ] {
                if let Some((data, rows, cols)) = source.get_tensor(key) {
                    let len = write_floats(&mut attn_file, &data, dtype)?;
                    entries.push(WeightEntry {
                        key: key.clone(),
                        kind: kind::TENSOR.into(),
                        shape: vec![rows, cols],
                        offset: attn_offset,
                        length: len,
                        file: ATTN_WEIGHTS_BIN.into(),
                    });
                    attn_offset += len;
                }
            }

            // QK norms (1D vectors, stored alongside attention)
            for key in [arch.attn_q_norm_key(layer), arch.attn_k_norm_key(layer)]
                .iter()
                .flatten()
            {
                if let Some(data) = source.get_vector(key) {
                    let bytes = crate::config::dtype::encode_floats(&data, dtype);
                    attn_file.write_all(&bytes)?;
                    entries.push(WeightEntry {
                        key: key.clone(),
                        kind: kind::VECTOR.into(),
                        shape: vec![data.len()],
                        offset: attn_offset,
                        length: bytes.len() as u64,
                        file: ATTN_WEIGHTS_BIN.into(),
                    });
                    attn_offset += bytes.len() as u64;
                }
            }

            callbacks.on_layer_done(COMP_ATTN_WEIGHTS, layer, 0.0);
        }
        attn_file.flush()?;
    } // end if write_attn

    // ── FFN up + down weights (gate is in gate_vectors.bin) ──
    //
    // Skipped entirely when `opts.level < Inference` OR
    // `opts.ffn_compact && !is_moe` (see `ffn_compact` doc for the
    // compact-mode caveats).
    //
    // MoE compact mode is not yet supported: the MoE branch below packs
    // the per-expert up/down weights *and* the router matrix into
    // `up_weights.bin`, and the loader would need expert-aware feature
    // files that don't exist yet. Refuse instead of silently corrupting.
    if opts.ffn_compact && arch.is_moe() && opts.level.writes_ffn() {
        return Err(VindexError::Parse(
            "ffn_compact not yet supported for MoE architectures — \
             per-expert feature-major files don't exist yet"
                .into(),
        ));
    }

    if write_ffn {
        let up_path = dir.join(UP_WEIGHTS_BIN);
        let mut up_file = BufWriter::new(std::fs::File::create(&up_path)?);
        let mut up_offset: u64 = 0;

        let down_path = dir.join(DOWN_WEIGHTS_BIN);
        let mut down_file = BufWriter::new(std::fs::File::create(&down_path)?);
        let mut down_offset: u64 = 0;

        for layer in 0..num_layers {
            callbacks.on_layer_start(COMP_UP_DOWN_WEIGHTS, layer, num_layers);

            if arch.is_moe() {
                for expert in 0..arch.num_experts() {
                    if let Some(key) = arch.expert_ffn_up_key(layer, expert) {
                        if let Some((data, rows, cols)) = source.get_tensor(&key) {
                            let len = write_floats(&mut up_file, &data, dtype)?;
                            entries.push(WeightEntry {
                                key,
                                kind: kind::TENSOR.into(),
                                shape: vec![rows, cols],
                                offset: up_offset,
                                length: len,
                                file: UP_WEIGHTS_BIN.into(),
                            });
                            up_offset += len;
                        }
                    }
                    if let Some(key) = arch.expert_ffn_down_key(layer, expert) {
                        if let Some((data, rows, cols)) = source.get_tensor(&key) {
                            let len = write_floats(&mut down_file, &data, dtype)?;
                            entries.push(WeightEntry {
                                key,
                                kind: kind::TENSOR.into(),
                                shape: vec![rows, cols],
                                offset: down_offset,
                                length: len,
                                file: DOWN_WEIGHTS_BIN.into(),
                            });
                            down_offset += len;
                        }
                    }
                }
                if let Some(key) = arch.moe_router_key(layer) {
                    if let Some((data, rows, cols)) = source.get_tensor(&key) {
                        let len = write_floats(&mut up_file, &data, dtype)?;
                        entries.push(WeightEntry {
                            key,
                            kind: kind::TENSOR.into(),
                            shape: vec![rows, cols],
                            offset: up_offset,
                            length: len,
                            file: UP_WEIGHTS_BIN.into(),
                        });
                        up_offset += len;
                    }
                }
            } else {
                let up_key = arch.ffn_up_key(layer);
                if let Some((data, rows, cols)) = source.get_tensor(&up_key) {
                    let len = write_floats(&mut up_file, &data, dtype)?;
                    entries.push(WeightEntry {
                        key: up_key,
                        kind: kind::TENSOR.into(),
                        shape: vec![rows, cols],
                        offset: up_offset,
                        length: len,
                        file: UP_WEIGHTS_BIN.into(),
                    });
                    up_offset += len;
                }

                let down_key = arch.ffn_down_key(layer);
                if let Some((data, rows, cols)) = source.get_tensor(&down_key) {
                    let len = write_floats(&mut down_file, &data, dtype)?;
                    entries.push(WeightEntry {
                        key: down_key,
                        kind: kind::TENSOR.into(),
                        shape: vec![rows, cols],
                        offset: down_offset,
                        length: len,
                        file: DOWN_WEIGHTS_BIN.into(),
                    });
                    down_offset += len;
                }
            }

            callbacks.on_layer_done(COMP_UP_DOWN_WEIGHTS, layer, 0.0);
        }
        up_file.flush()?;
        down_file.flush()?;
    } // end if write_ffn

    // ── Norms ── (paired with attention; skipped when level < Attention)
    if write_attn {
        let norms_path = dir.join(NORMS_BIN);
        let mut norms_file = BufWriter::new(std::fs::File::create(&norms_path)?);
        let mut norms_offset: u64 = 0;

        // Per-layer norms
        for layer in 0..num_layers {
            let mut norm_keys: Vec<String> = [
                Some(arch.input_layernorm_key(layer)),
                Some(arch.post_attention_layernorm_key(layer)),
                arch.pre_feedforward_layernorm_key(layer),
                arch.post_feedforward_layernorm_key(layer),
            ]
            .into_iter()
            .flatten()
            .collect();

            // Hybrid MoE additions: the pre_2/post_1/post_2 weights plus
            // the outer post_feedforward_layernorm that wraps (h1+h2).
            if arch.is_hybrid_moe() {
                for k in [
                    arch.moe_pre_experts_norm_key(layer),
                    arch.moe_post_ffn1_norm_key(layer),
                    arch.moe_post_experts_norm_key(layer),
                    arch.moe_post_outer_norm_key(layer),
                ]
                .into_iter()
                .flatten()
                {
                    if !norm_keys.contains(&k) {
                        norm_keys.push(k);
                    }
                }
            }

            for key in norm_keys {
                if let Some(data) = source.get_vector(&key) {
                    let bytes = crate::config::dtype::encode_floats(&data, dtype);
                    norms_file.write_all(&bytes)?;
                    entries.push(WeightEntry {
                        key,
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

        // Final norm (model.norm.weight)
        if let Some(data) = source.get_vector("norm.weight") {
            let bytes = crate::config::dtype::encode_floats(&data, dtype);
            norms_file.write_all(&bytes)?;
            entries.push(WeightEntry {
                key: "norm.weight".into(),
                kind: kind::VECTOR.into(),
                shape: vec![data.len()],
                offset: norms_offset,
                length: bytes.len() as u64,
                file: NORMS_BIN.into(),
            });
        }
        norms_file.flush()?;
    }

    // ── LM Head ── (skipped when level < Inference)
    if write_lm_head {
        if let Some((data, rows, cols)) = source.lm_head() {
            let lm_bytes = crate::config::dtype::encode_floats(&data, dtype);
            std::fs::write(dir.join(LM_HEAD_BIN), &lm_bytes)?;
            entries.push(WeightEntry {
                key: "lm_head.weight".into(),
                kind: kind::TENSOR.into(),
                shape: vec![rows, cols],
                offset: 0,
                length: lm_bytes.len() as u64,
                file: LM_HEAD_BIN.into(),
            });
        }
    }

    // ── Manifest ──
    let manifest_json =
        serde_json::to_string_pretty(&entries).map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(dir.join(WEIGHT_MANIFEST_JSON), manifest_json)?;

    // ── Update index.json ──
    let config_path = dir.join(INDEX_JSON);
    let config_text = std::fs::read_to_string(&config_path)?;
    let mut config: VindexConfig =
        serde_json::from_str(&config_text).map_err(|e| VindexError::Parse(e.to_string()))?;

    config.has_model_weights = true;

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
        // Per-layer geometry (Gemma 4)
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

    callbacks.on_stage_done(STAGE_MODEL_WEIGHTS, start.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}

use crate::config::dtype::write_floats;
