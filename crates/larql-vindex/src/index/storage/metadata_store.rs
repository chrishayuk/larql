//! `MetadataStore` — owns down-meta heap/mmap state and per-feature
//! overrides (INSERT/DELETE-side mutations).
//!
//! Carved out of `VectorIndex` in the 2026-04-25 reorg.

use std::collections::HashMap;
use std::sync::Arc;

use larql_models::TopKEntry;
use serde::Deserialize;

use crate::index::types::{DownMetaMmap, FeatureMeta};

#[derive(Clone, Debug, Deserialize)]
pub struct GgufDownMetaManifest {
    pub layers: Vec<GgufDownMetaLayerManifest>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct GgufGateManifest {
    pub layers: Vec<GgufGateLayerManifest>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct GgufGateLayerManifest {
    pub layer: usize,
    pub tensor: String,
    pub source_file: String,
    pub tensor_type: u32,
    pub rows: usize,
    pub cols: usize,
    pub experts: usize,
    pub features: Option<usize>,
    pub tensor_offset: u64,
    pub data_offset: u64,
}

impl GgufGateLayerManifest {
    pub fn feature_count(&self) -> usize {
        self.features
            .unwrap_or_else(|| self.rows.saturating_mul(self.experts))
    }
}

impl GgufGateManifest {
    pub fn layer(&self, layer: usize) -> Option<&GgufGateLayerManifest> {
        self.layers.iter().find(|entry| entry.layer == layer)
    }

    pub fn loaded_layers(&self) -> Vec<usize> {
        self.layers.iter().map(|entry| entry.layer).collect()
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct GgufDownMetaLayerManifest {
    pub layer: usize,
    pub tensor: String,
    pub source_file: String,
    pub tensor_type: u32,
    pub rows: usize,
    pub cols: usize,
    pub experts: usize,
    pub features: usize,
    pub tensor_offset: u64,
    pub data_offset: u64,
}

impl GgufDownMetaManifest {
    pub fn layer(&self, layer: usize) -> Option<&GgufDownMetaLayerManifest> {
        self.layers.iter().find(|entry| entry.layer == layer)
    }

    pub fn loaded_layers(&self) -> Vec<usize> {
        self.layers.iter().map(|entry| entry.layer).collect()
    }

    pub fn total_features(&self) -> usize {
        self.layers.iter().map(|entry| entry.features).sum()
    }

    pub fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        let entry = self.layer(layer)?;
        if feature >= entry.features || entry.cols == 0 {
            return None;
        }
        let expert = feature / entry.cols;
        if expert >= entry.experts {
            return None;
        }
        let local_feature = feature % entry.cols;
        let token = format!("gguf:{}:E{}:F{}", entry.tensor, expert, local_feature);
        Some(FeatureMeta {
            top_token: token.clone(),
            top_token_id: feature as u32,
            c_score: 0.0,
            top_k: vec![TopKEntry {
                token,
                token_id: feature as u32,
                logit: 0.0,
            }],
        })
    }
}

#[derive(Clone)]
pub struct MetadataStore {
    /// Per-layer, per-feature output token metadata (heap mode).
    pub down_meta: Vec<Option<Vec<Option<FeatureMeta>>>>,
    /// Mmap'd down_meta.bin (zero-copy mode).
    pub down_meta_mmap: Option<Arc<DownMetaMmap>>,
    /// Down vector overrides — `(layer, feature) → hidden_size f32`.
    pub down_overrides: HashMap<(usize, usize), Vec<f32>>,
    /// Up vector overrides — same shape; written by INSERT.
    pub up_overrides: HashMap<(usize, usize), Vec<f32>>,
    /// Compact GGUF down-meta manifest for over-budget Kimi/DeepSeek2 browse vindexes.
    pub gguf_down_meta_manifest: Option<Arc<GgufDownMetaManifest>>,
    /// Compact GGUF gate manifest for bounded query-time scans.
    pub gguf_gate_manifest: Option<Arc<GgufGateManifest>>,
}

impl MetadataStore {
    pub fn empty(num_layers: usize) -> Self {
        Self {
            down_meta: vec![None; num_layers],
            down_meta_mmap: None,
            down_overrides: HashMap::new(),
            up_overrides: HashMap::new(),
            gguf_down_meta_manifest: None,
            gguf_gate_manifest: None,
        }
    }
}
